"""
Generates the two figures for the cross-split leakage audit:
  1. xrd_similarity_hist_linear.png -- distribution of best-match XRD
     cosine similarity (linear y-axis, per earlier discussion: log-y
     histograms tend to be misread; linear works here because the count
     axis itself doesn't span many orders of magnitude the way the error
     values do in figure 2).
  2. similarity_vs_error_density.png -- hexbin density of per-material
     length MAPE vs. best-match similarity, log-y (necessary: length MAPE
     spans ~0.01%-200%), color scale capped at the 99th percentile hexbin
     count to avoid saturation from a few extreme-density bins.

Reads directly from the audit pkls (mpfull_xrd_leakage_audit.pkl,
mp20_xrd_leakage_audit.pkl) plus their corresponding already-computed
inversion/prediction result files. See also {mpfull,mp20}_similarity_pairs.csv
(same underlying per-material data as the audit pkls, exported flat for
use outside this pipeline, e.g. cross-referencing against external tools).

Usage:
    python plot_similarity_vs_error.py
"""
import os
import io
import contextlib
import sys
import pickle

import h5py
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    from analyze_completed_inversion_res import niggli_params

DATA_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
OUT_DIR = os.path.dirname(os.path.abspath(__file__))

DATASETS = {
    "Full MP": dict(
        audit=f"{DATA_ROOT}/eval_results/mpfull_xrd_leakage_audit.pkl",
        results=f"{DATA_ROOT}/final_test_results/mpfull_aug/bispec_test_results_inverted.h5",
    ),
    "MP-20": dict(
        audit=f"{DATA_ROOT}/eval_results/mp20_xrd_leakage_audit.pkl",
        results=f"{DATA_ROOT}/final_test_results/mp20_correct_cag/bispec_test_results_inverted.h5",
    ),
}
SIM_THRESHOLD = 0.999


def gather(audit_path, results_path, split="test"):
    """Per-material (similarity, length_mape) for every material with both
    an audit entry (similarity to nearest train match) and a computed
    prediction. length_mape uses the same Niggli-reduced positional
    comparison used everywhere else in this project."""
    audit = pickle.load(open(audit_path, "rb"))
    r = audit[split]
    mp_id_to_sim = dict(zip(r["mp_ids"], r["best_sim"]))

    f = h5py.File(results_path, "r")
    sims, lens = [], []
    for mid in f.keys():
        if mid not in mp_id_to_sim:
            continue
        g = f[mid]
        try:
            pred_lat = g["pred"][()].reshape(3, 3)
            true_lat = g["true_lattice"][()].reshape(3, 3)
            pa, pb, pc, pal, pbe, pga = niggli_params(pred_lat)
            ta, tb, tc, tal, tbe, tga = niggli_params(true_lat)
            l = [abs(p - t) / t for p, t in [(pa, ta), (pb, tb), (pc, tc)] if t > 1e-6]
        except Exception:
            continue
        sims.append(mp_id_to_sim[mid])
        lens.append(100 * np.mean(l))
    f.close()
    return np.array(sims), np.array(lens)


def plot_histogram(all_sims):
    fig, axes = plt.subplots(1, len(all_sims), figsize=(6.5 * len(all_sims), 5))
    if len(all_sims) == 1:
        axes = [axes]
    for ax, (name, sims) in zip(axes, all_sims.items()):
        ax.hist(sims, bins=100, log=False, color="steelblue")
        ax.axvline(SIM_THRESHOLD, color="red", linestyle="--", label=f"{SIM_THRESHOLD}")
        ax.set_xlabel("XRD-pattern cosine similarity to best-matching train material")
        ax.set_ylabel("count")
        ax.set_title(f"{name} (n={len(sims)})")
        ax.legend()
    plt.tight_layout()
    out = f"{OUT_DIR}/xrd_similarity_hist_linear.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def plot_density(all_data):
    fig, axes = plt.subplots(1, len(all_data), figsize=(6.5 * len(all_data), 5.5))
    if len(all_data) == 1:
        axes = [axes]
    for ax, (name, (sims, lens)) in zip(axes, all_data.items()):
        logl = np.log10(np.clip(lens, 1e-4, None))
        # probe counts first to find a sane color cap (99th percentile),
        # avoiding saturation from a handful of extreme-density bins
        probe = ax.hexbin(sims, logl, gridsize=60, mincnt=1).get_array()
        vmax = np.percentile(probe, 99)
        ax.clear()
        hb = ax.hexbin(sims, logl, gridsize=60, cmap="viridis", mincnt=1, vmin=1, vmax=vmax)

        bins = np.linspace(0.5, 1.0, 26)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        medians = [
            np.median(lens[(sims >= bins[i]) & (sims < bins[i + 1])])
            if np.sum((sims >= bins[i]) & (sims < bins[i + 1])) > 5 else np.nan
            for i in range(len(bins) - 1)
        ]
        ax.plot(bin_centers, np.log10(np.clip(medians, 1e-4, None)), color="red", linewidth=2, label="binned median")
        ax.axvline(SIM_THRESHOLD, color="white", linestyle="--", linewidth=1.2, label=f"threshold={SIM_THRESHOLD}")

        yticks = [-2, -1, 0, 1, 2]
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{10**y:.2g}" for y in yticks])
        ax.set_xlabel("XRD-pattern similarity to nearest training structure")
        ax.set_ylabel("length MAPE (%)")
        ax.set_title(name)
        ax.legend(loc="lower left")
        fig.colorbar(hb, ax=ax, label=f"count (capped at p99={vmax:.0f})")
    plt.tight_layout()
    out = f"{OUT_DIR}/similarity_vs_error_density.png"
    plt.savefig(out, dpi=150)
    plt.close(fig)
    print(f"saved {out}")


def main():
    all_sims = {}
    all_data = {}
    for name, cfg in DATASETS.items():
        sims, lens = gather(cfg["audit"], cfg["results"])
        all_sims[name] = sims
        all_data[name] = (sims, lens)
        rho = np.corrcoef(np.argsort(np.argsort(sims)), np.argsort(np.argsort(lens)))[0, 1]
        print(f"{name}: n={len(sims)}  fraction > {SIM_THRESHOLD}: {np.mean(sims > SIM_THRESHOLD):.4f}")

    plot_histogram(all_sims)
    plot_density(all_data)


if __name__ == "__main__":
    main()
