"""
Fit the intensity-perturbation Gamma distribution by MLE on RRUFF observed-vs-
simulated peak-intensity differences (paper recipe).

Workflow:
  1. Load `rruff_data_dict.pkl` — DIF entries (mineral → XRDStructure with the
     refinement-derived calculated peak list (2theta, intensity)).
  2. For each `XY_Processed/*.txt` raw observed scan, look up the matching DIF
     entry by RRUFF ID.
  3. Generate the *simulated* continuous PXRD by convolving the DIF peaks with
     a Gaussian pseudo-Voigt at the paper's nominal Caglioti params
     (U=0.05, V=-0.06, W=0.07). Both observed and simulated patterns are then
     baseline-subtracted and max-normalised.
  4. At each DIF peak position, sample the local-max intensity from the
     simulated *and* observed patterns and record |I_obs − I_sim| as the
     empirical "percentage difference" (both on the [0, 1] scale).
  5. MLE-fit Gamma(a, scale) to those differences with `loc=0` fixed.
  6. Pickle the result so `generate_augmented_data.py --augmentation caglioti4x` can read it.

Usage:
    python fit_intensity_gamma.py \
        --dif    $POWDERXRD_DATA_ROOT/rruff/rruff_data_dict.pkl \
        --xy-dir $POWDERXRD_DATA_ROOT/rruff/XY_Processed \
        --out    $POWDERXRD_DATA_ROOT/rruff/intens_gamma.pkl
"""
import argparse
import glob
import os
import pickle
import re
import sys
import types
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import gamma as gamma_dist
from tqdm import tqdm

from utilities import caglioti_fwhm


# Caglioti params used to (re)simulate the reference pattern.
SIM_U, SIM_V, SIM_W = 0.05, -0.06, 0.07


# ---------------------------------------------------------------------------
# Stub XRDStructure so we can unpickle without the heavy data_prep import chain.
# ---------------------------------------------------------------------------
@dataclass
class XRDStructure:
    metadata: pd.DataFrame
    atoms: pd.DataFrame
    pattern: pd.DataFrame


def _install_xrdstructure_stub():
    sys.modules.setdefault("RRUFF", types.ModuleType("RRUFF"))
    mod = types.ModuleType("RRUFF.data_prep")
    mod.XRDStructure = XRDStructure
    sys.modules["RRUFF.data_prep"] = mod
    main_mod = sys.modules.get("__main__")
    if main_mod is not None and not hasattr(main_mod, "XRDStructure"):
        main_mod.XRDStructure = XRDStructure


class _RruffUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if name == "XRDStructure":
            return XRDStructure
        return super().find_class(module, name)


def load_dif(path):
    _install_xrdstructure_stub()
    with open(path, "rb") as f:
        return _RruffUnpickler(f).load()


# ---------------------------------------------------------------------------
# XY parsing & pattern simulation
# ---------------------------------------------------------------------------
RID_RE = re.compile(r"(R\d+)(?:-\d+)?")
_NUM_RE = re.compile(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?")


def load_xy(path):
    """Parse a RRUFF XY_Processed file. Handles whitespace- *and* comma-
    separated value lines, with metadata lines starting with `##`."""
    rows = []
    with open(path, "r", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("##"):
                continue
            nums = _NUM_RE.findall(line)
            if len(nums) < 2:
                continue
            try:
                rows.append((float(nums[0]), float(nums[1])))
            except ValueError:
                continue
    return np.asarray(rows, dtype=np.float64)


def simulate_continuous(centers, intensities, two_theta_grid,
                        U=SIM_U, V=SIM_V, W=SIM_W):
    """Vectorised superimposed Gaussian pseudo-Voigt simulation on the given grid."""
    centers = np.asarray(centers, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)
    if centers.size == 0:
        return np.zeros_like(two_theta_grid)
    fwhm = caglioti_fwhm(centers, U, V, W)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))
    dx = two_theta_grid[None, :] - centers[:, None]
    gauss = np.exp(-(dx ** 2) / (2.0 * sigma[:, None] ** 2))
    pattern = (intensities[:, None] * gauss).sum(axis=0)
    return pattern


def baseline_median(intensity, k=51):
    if k % 2 == 0:
        k += 1
    half = k // 2
    pad = np.concatenate([np.full(half, intensity[0]), intensity,
                          np.full(half, intensity[-1])])
    out = np.empty_like(intensity)
    for i in range(len(intensity)):
        out[i] = np.median(pad[i:i + k])
    return out


# ---------------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------------
def collect_diffs(dif_dict, xy_dir, peak_window=0.30, baseline_k=51,
                  min_peak_intensity=0.02, two_theta_min=5.0, two_theta_max=90.0,
                  verbose_first=3):
    """
    Returns a 1-D array of |I_obs − I_sim| values (each on the [0, 1] scale,
    one per DIF peak above `min_peak_intensity` after normalisation).
    """
    dif_by_rid = {}
    for k in dif_dict.keys():
        m = RID_RE.search(k)
        if m:
            dif_by_rid.setdefault(m.group(1), []).append(k)

    files = sorted(glob.glob(os.path.join(xy_dir, "*.txt")))
    print(f"  XY files: {len(files):,} | DIF entries: {len(dif_dict):,}")

    all_diffs = []
    rejected = {"no_match": 0, "xy_empty": 0, "pat_empty": 0,
                "obs_zero": 0, "calc_zero": 0, "ok": 0}
    seen = 0

    for f in tqdm(files, desc="scans"):
        base = os.path.basename(f)
        m = RID_RE.search(base)
        if not m or m.group(1) not in dif_by_rid:
            rejected["no_match"] += 1
            continue
        mineral = base.split("__", 1)[0]
        candidates = dif_by_rid[m.group(1)]
        dif_key = next((k for k in candidates if k.startswith(mineral + "__")),
                       candidates[0])
        entry = dif_dict[dif_key]

        xy = load_xy(f)
        if xy.size == 0 or len(xy) < max(102, 2 * baseline_k):
            rejected["xy_empty"] += 1
            if seen < verbose_first:
                print(f"   [skip xy_empty] {base[:60]} → shape={xy.shape}")
            seen += 1
            continue

        if entry.pattern is None or len(entry.pattern) == 0:
            rejected["pat_empty"] += 1
            continue

        two_theta = xy[:, 0]
        intensity = xy[:, 1]
        # Restrict to the standard 5–90° window.
        mask = (two_theta >= two_theta_min) & (two_theta <= two_theta_max)
        two_theta = two_theta[mask]
        intensity = intensity[mask]
        if len(two_theta) < max(102, 2 * baseline_k):
            rejected["xy_empty"] += 1
            continue

        # Observed: baseline-subtract & max-normalise.
        bl = baseline_median(intensity, k=baseline_k)
        obs = np.clip(intensity - bl, 0.0, None)
        if obs.max() <= 0:
            rejected["obs_zero"] += 1
            continue
        obs = obs / obs.max()

        # Simulated: build continuous PXRD via Caglioti(0.05, -0.06, 0.07).
        calc_2t = entry.pattern["2theta"].to_numpy(dtype=np.float64)
        calc_I = entry.pattern["intensity"].to_numpy(dtype=np.float64)
        keep = (calc_2t >= two_theta_min) & (calc_2t <= two_theta_max) & (calc_I > 0)
        calc_2t = calc_2t[keep]; calc_I = calc_I[keep]
        if calc_2t.size == 0:
            rejected["calc_zero"] += 1
            continue
        sim = simulate_continuous(calc_2t, calc_I, two_theta)
        if sim.max() <= 0:
            rejected["calc_zero"] += 1
            continue
        sim = sim / sim.max()

        # Local-max sampling at each calc peak position.
        d2t = float(np.median(np.diff(two_theta)))
        win = max(1, int(round(peak_window / d2t)))

        n_peaks = 0
        for theta_c in calc_2t:
            idx = int(np.clip(np.searchsorted(two_theta, theta_c), 0,
                              len(two_theta) - 1))
            lo, hi = max(0, idx - win), min(len(obs), idx + win + 1)
            I_obs = obs[lo:hi].max()
            I_sim = sim[lo:hi].max()
            # Skip peaks too weak in the simulated pattern (rounded down by
            # broadening / overlap with neighbours).
            if I_sim < min_peak_intensity:
                continue
            all_diffs.append(abs(I_obs - I_sim))
            n_peaks += 1

        if n_peaks > 0:
            rejected["ok"] += 1

    return np.asarray(all_diffs, dtype=np.float64), rejected


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def main():
    _root = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
    p = argparse.ArgumentParser()
    p.add_argument("--dif",
                   default=f"{_root}/rruff/rruff_data_dict.pkl")
    p.add_argument("--xy-dir",
                   default=f"{_root}/rruff/XY_Processed")
    p.add_argument("--out",
                   default=f"{_root}/rruff/intens_gamma.pkl")
    p.add_argument("--peak-window", type=float, default=0.30)
    p.add_argument("--baseline-k", type=int, default=51)
    p.add_argument("--min-peak-intensity", type=float, default=0.02,
                   help="Skip simulated peaks weaker than this (max-normalised scale).")
    p.add_argument("--max-diff", type=float, default=None,
                   help="Optional clip on |I_obs−I_sim| before fitting.")
    args = p.parse_args()

    print(f"Loading DIF dict from {args.dif} ...")
    dif = load_dif(args.dif)
    print(f"  {len(dif)} entries")

    diffs, rej = collect_diffs(
        dif, args.xy_dir,
        peak_window=args.peak_window,
        baseline_k=args.baseline_k,
        min_peak_intensity=args.min_peak_intensity,
    )

    print(f"\nScans accepted: {rej['ok']:,} | rejected:")
    for k, v in rej.items():
        if k != "ok":
            print(f"  {k}: {v:,}")
    print(f"Collected {len(diffs):,} peak-difference samples")
    if len(diffs) == 0:
        sys.exit("No diffs collected — aborting.")
    print(f"  mean   = {diffs.mean():.4f}")
    print(f"  median = {np.median(diffs):.4f}")
    print(f"  p95    = {np.quantile(diffs, 0.95):.4f}")
    print(f"  max    = {diffs.max():.4f}")

    if args.max_diff is not None:
        kept = diffs <= args.max_diff
        print(f"  clipping at {args.max_diff}: keep {kept.sum():,} / {len(diffs):,}")
        diffs = diffs[kept]

    # Drop zeros (Gamma support is (0, ∞))
    nz = diffs[diffs > 0]
    print(f"Dropping {len(diffs) - len(nz):,} zero-diff samples for fit "
          f"(Gamma needs strictly positive support)")
    a, _, scale = gamma_dist.fit(nz, floc=0)
    print(f"\nMLE Gamma fit: a = {a:.6f}, scale = {scale:.6f}")
    print(f"  mean (a*scale)        = {a * scale:.4f}")
    print(f"  std  (sqrt(a)*scale)  = {np.sqrt(a) * scale:.4f}")

    out = {
        "a": float(a),
        "scale": float(scale),
        "loc": 0.0,
        "n_samples": int(len(nz)),
        "n_scans": int(rej["ok"]),
        "fit_args": {
            "peak_window": args.peak_window,
            "baseline_k": args.baseline_k,
            "min_peak_intensity": args.min_peak_intensity,
            "max_diff": args.max_diff,
            "sim_UVW": (SIM_U, SIM_V, SIM_W),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"\n💾 Saved to {args.out}")
    print(f"\nUse with:")
    print(f"  python generate_augmented_data.py --augmentation caglioti4x "
          f"--intens-gamma-a {a:.6f} --intens-gamma-scale {scale:.6f}")


if __name__ == "__main__":
    main()
