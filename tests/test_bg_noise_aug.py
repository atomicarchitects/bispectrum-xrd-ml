import numpy as np
import torch

from bg_noise_aug import (
    _sample_gamma_params,
    add_bg_noise,
    _LOG_SCALE_MAX_RRUFF,
    _LOG_A_MAX_RRUFF,
    _LOG_SCALE_MAX_CNRS,
    _LOG_A_MAX_CNRS,
)


def test_sample_gamma_params_respects_rejection_bounds():
    # cnrs_mix_prob defaults to 0.0 -- pure RRUFF, unchanged from the original
    # single-profile behavior.
    rng = np.random.default_rng(0)
    a, scale = _sample_gamma_params(200, rng)

    assert a.shape == (200,)
    assert scale.shape == (200,)
    assert np.all(np.log(scale) <= _LOG_SCALE_MAX_RRUFF)
    assert np.all(np.log(a) <= _LOG_A_MAX_RRUFF)
    assert np.all(a > 0) and np.all(scale > 0)


def test_sample_gamma_params_cnrs_profile_respects_its_own_bounds():
    rng = np.random.default_rng(0)
    a, scale = _sample_gamma_params(200, rng, cnrs_mix_prob=1.0)

    assert np.all(np.log(scale) <= _LOG_SCALE_MAX_CNRS)
    assert np.all(np.log(a) <= _LOG_A_MAX_CNRS)
    assert np.all(a > 0) and np.all(scale > 0)
    # CNRS is fit to noisier data -- its mean draw should exceed RRUFF's.
    a_rruff, scale_rruff = _sample_gamma_params(200, np.random.default_rng(0))
    assert (a * scale).mean() > (a_rruff * scale_rruff).mean()


def test_sample_gamma_params_mix_prob_produces_both_profiles():
    # With a mid-range mix probability and enough samples, draws should span
    # both the RRUFF and CNRS ceilings (i.e. mixing is actually happening,
    # not silently collapsing to one profile).
    rng = np.random.default_rng(0)
    a, scale = _sample_gamma_params(2000, rng, cnrs_mix_prob=0.5)

    assert np.any(np.log(a) > _LOG_A_MAX_RRUFF)  # some draws only possible from the CNRS profile
    assert np.all(np.log(a) <= _LOG_A_MAX_CNRS)  # but never past the CNRS ceiling


def test_add_bg_noise_only_increases_values():
    rng = np.random.default_rng(0)
    batch = torch.zeros((4, 1, 100), dtype=torch.float32)
    noisy = add_bg_noise(batch, xrd_std=0.1, rng=rng)

    assert noisy.shape == batch.shape
    assert torch.all(noisy >= batch)
    assert not torch.equal(noisy, batch)


def test_add_bg_noise_scales_inversely_with_xrd_std():
    batch = torch.zeros((4, 1, 100), dtype=torch.float32)
    small_std = add_bg_noise(batch, xrd_std=0.01, rng=np.random.default_rng(1))
    large_std = add_bg_noise(batch, xrd_std=10.0, rng=np.random.default_rng(1))

    # Same seed => same raw Gamma draws; dividing by a larger xrd_std must
    # shrink the added noise.
    assert small_std.mean().item() > large_std.mean().item()
