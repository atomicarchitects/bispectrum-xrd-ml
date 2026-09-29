"""
Train-time background-noise augmentation (Crystalyze paper recipe, extended
with an optional CNRS-domain noise profile).

For each pattern in a batch:
  1. Pick a noise profile: RRUFF-fit (default, matches the published results)
     or CNRS-fit, per-example, according to `cnrs_mix_prob`.
  2. Sample (log_a, log_scale) from that profile's fitted multivariate
     log-normal, rejecting and resampling above that profile's ceiling (see
     below).
  3. Draw a Gamma(a, scale) noise vector of length n_points.
  4. Add it to the raw [0,1]-normalised XRD pattern.

The RRUFF profile's parameters come from data_gen_full_aug.BG_NOISE_LOGN_MEAN
/COV (fit to RRUFF observed-vs-simulated background residuals). The CNRS
profile comes from mp_full/fit_bg_noise_cnrs.py, run against opXRD CNRS
patterns with a known atomic structure -- CNRS backgrounds measured ~12x
noisier on average than RRUFF's (mean noise level 0.15 vs 0.013), which is
the actual motivation for adding this second profile: RRUFF-only augmentation
under-represents how noisy real CNRS-domain scans are.

`cnrs_mix_prob` defaults to 0.0 (pure RRUFF, unchanged from the original
behavior) -- pass a nonzero value (e.g. 0.5) explicitly to mix in CNRS-domain
noise for a CNRS-aware retrain, without silently changing behavior for
existing/other training runs that don't ask for it.

Usage in training loop (add to each training batch, before model forward):
    from mp_full.bg_noise_aug import add_bg_noise
    data = add_bg_noise(data, xrd_std=xrd_std)                       # RRUFF-only (default)
    data = add_bg_noise(data, xrd_std=xrd_std, cnrs_mix_prob=0.5)    # 50/50 RRUFF/CNRS mix
    # data: [B, 1, N], float32, on GPU
"""
import numpy as np
import torch

# RRUFF profile -- hyper-parameters copied from generate_augmented_data.py
_LOGN_MEAN_RRUFF = np.array([1.06709083, -5.39670499])   # (log_a, log_scale)
_LOGN_COV_RRUFF  = np.array([[0.19984825, 0.14809546],
                              [0.14809546, 1.31019437]])
_LOG_A_MAX_RRUFF     = np.log(10.0)    # rejection ceiling for log(a)     (~2.76 sigma above the RRUFF mean)
_LOG_SCALE_MAX_RRUFF = np.log(0.05)    # rejection ceiling for log(scale) (~2.10 sigma above the RRUFF mean)

# CNRS profile -- from mp_full/fit_bg_noise_cnrs.py (n=808 opXRD CNRS patterns)
_LOGN_MEAN_CNRS = np.array([1.58177438, -4.21359437])
_LOGN_COV_CNRS  = np.array([[1.39591577, -0.64502092],
                             [-0.64502092, 1.56407708]])
# Same ~2.5 sigma-above-mean rule as the RRUFF ceilings (not the same absolute
# values -- CNRS's fit has a much higher mean/variance, so reusing RRUFF's
# absolute ceiling would reject most CNRS draws and silently distort the fit).
_LOG_A_MAX_CNRS     = np.log(93.27)
_LOG_SCALE_MAX_CNRS = np.log(0.337)

_PROFILES = {
    "rruff": (_LOGN_MEAN_RRUFF, _LOGN_COV_RRUFF, _LOG_A_MAX_RRUFF, _LOG_SCALE_MAX_RRUFF),
    "cnrs":  (_LOGN_MEAN_CNRS, _LOGN_COV_CNRS, _LOG_A_MAX_CNRS, _LOG_SCALE_MAX_CNRS),
}


def _sample_gamma_params_one_profile(n: int, rng: np.random.Generator, profile: str):
    """Return arrays (a, scale) of shape (n,) via rejection sampling from one profile."""
    mean, cov, log_a_max, log_scale_max = _PROFILES[profile]
    a_out     = np.empty(n)
    scale_out = np.empty(n)
    remaining = np.arange(n)

    while remaining.size > 0:
        draws = rng.multivariate_normal(mean, cov, size=remaining.size)
        log_a, log_scale = draws[:, 0], draws[:, 1]
        accepted = (log_scale <= log_scale_max) & (log_a <= log_a_max)
        idx = remaining[accepted]
        a_out[idx]     = np.exp(log_a[accepted])
        scale_out[idx] = np.exp(log_scale[accepted])
        remaining = remaining[~accepted]

    return a_out, scale_out


def _sample_gamma_params(batch_size: int, rng: np.random.Generator, cnrs_mix_prob: float = 0.0):
    """Return arrays (a, scale) of shape (batch_size,), each element drawn from
    the CNRS profile with probability cnrs_mix_prob and the RRUFF profile
    otherwise."""
    if cnrs_mix_prob <= 0.0:
        return _sample_gamma_params_one_profile(batch_size, rng, "rruff")
    if cnrs_mix_prob >= 1.0:
        return _sample_gamma_params_one_profile(batch_size, rng, "cnrs")

    is_cnrs = rng.random(batch_size) < cnrs_mix_prob
    a_out, scale_out = np.empty(batch_size), np.empty(batch_size)

    n_cnrs = int(is_cnrs.sum())
    if n_cnrs > 0:
        a_out[is_cnrs], scale_out[is_cnrs] = _sample_gamma_params_one_profile(n_cnrs, rng, "cnrs")
    n_rruff = batch_size - n_cnrs
    if n_rruff > 0:
        a_out[~is_cnrs], scale_out[~is_cnrs] = _sample_gamma_params_one_profile(n_rruff, rng, "rruff")

    return a_out, scale_out


def add_bg_noise(
    batch: torch.Tensor,
    xrd_std: float,
    rng: np.random.Generator | None = None,
    cnrs_mix_prob: float = 0.0,
) -> torch.Tensor:
    """
    Add Gamma background noise to a batch of normalised XRD patterns.

    Args:
        batch:   FloatTensor of shape [B, 1, N] — normalised XRD patterns on GPU.
        xrd_std: The scalar std used when normalising the XRD dataset (from
                 xrd_std = xrd_train_tensor.std()).  Noise is drawn in the
                 original [0,1] space and then divided by xrd_std before adding.
        rng:     Optional numpy Generator for reproducibility.
        cnrs_mix_prob: Per-example probability of drawing from the CNRS-fit
                 noise profile instead of the RRUFF-fit one. 0.0 (default)
                 reproduces the original RRUFF-only behavior exactly.

    Returns:
        Noisy batch tensor, same shape/device/dtype as input.
    """
    if rng is None:
        rng = np.random.default_rng()

    B, _, N = batch.shape
    a_arr, scale_arr = _sample_gamma_params(B, rng, cnrs_mix_prob=cnrs_mix_prob)

    # Draw Gamma noise per pattern in [0,1] space.
    noise_np = np.stack([
        rng.gamma(a_arr[i], scale_arr[i], size=N).astype(np.float32)
        for i in range(B)
    ])  # shape [B, N]

    noise = torch.from_numpy(noise_np).unsqueeze(1).to(batch.device)  # [B, 1, N]

    # Convert noise to normalised space and add; no clipping needed in normed space.
    return batch + noise / (xrd_std + 1e-8)
