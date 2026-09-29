"""Histogram of per-structure inversion wall-clock times saved by time_inversion.py.

Usage:
    python plot_time_inversion.py --tag mp20 --n-censored 41
    python plot_time_inversion.py --tag mpfull

--n-censored: number of sampled materials that never finished (e.g. the allocation ended first).
They are the slowest ones by construction, so the plotted distribution is right-censored and its
upper percentiles are lower bounds; the count is stated in the legend and printed.
"""
import argparse
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["font.size"] = 12

p = argparse.ArgumentParser()
p.add_argument("--tag", default="mp20")
p.add_argument("--settings", default="default")
p.add_argument("--n-censored", type=int, default=0)
p.add_argument("--out", default=None)
p.add_argument("--linear", action="store_true", help="linear x-axis (use when the times span less than one decade, e.g. with a cap)")
p.add_argument("--plain", action="store_true", help="no reference lines or legend; title states the sampled n (unfinished count goes in the caption)")
p.add_argument("--title", default=None, help="plot title prefix (default: the tag)")
p.add_argument("--first-k", type=int, default=0,
               help="use only the first K materials in submission order (needs --n, --seed, --predictions-file "
                    "as used for the timing run): unbiased while the run is still going, since the rest are "
                    "counted as unfinished instead of silently dropped")
p.add_argument("--n", type=int, default=5000)
p.add_argument("--seed", type=int, default=0)
p.add_argument("--predictions-file", default=None)
args = p.parse_args()

d = pickle.load(open(f"time_inversion_{args.tag}_{args.settings}.pkl", "rb"))
t = np.array(d["times"])
n_total = len(t) + args.n_censored
if args.first_k:
    pk = pickle.load(open(args.predictions_file, "rb"))
    ids = list(pk["material_ids"])
    sel = np.random.default_rng(args.seed).choice(len(ids), size=min(args.n, len(ids)), replace=False)
    first = [ids[i] for i in sel[:args.first_k]]
    time_of = dict(zip(d["material_ids"], d["times"]))
    t = np.array([time_of[m] for m in first if m in time_of])
    args.n_censored = len(first) - len(t)
    n_total = len(first)

print(f"n finished = {len(t)}, censored = {args.n_censored} ({100 * args.n_censored / n_total:.1f}% of sample)")
print(f"mean {t.mean():.1f}s  median {np.median(t):.1f}s  p90 {np.percentile(t, 90):.1f}s  "
      f"p95 {np.percentile(t, 95):.1f}s  p99 {np.percentile(t, 99):.1f}s  max {t.max():.1f}s (finished only)")
for th in (30, 60, 150, 200, 600):
    print(f"  fraction of finished under {th}s: {100 * np.mean(t < th):.1f}%   "
          f"(of full sample incl. censored: {100 * np.sum(t < th) / n_total:.1f}%)")

fig, ax = plt.subplots(figsize=(6.5, 4))
bins = np.linspace(0, t.max() * 1.02, 45) if args.linear else np.logspace(np.log10(t.min()), np.log10(t.max()), 50)
counts, _, _ = ax.hist(t, bins=bins, color="steelblue", alpha=0.85)
ax.set_ylim(0, counts.max() * (1.2 if args.plain else 1.75))  # headroom so the legend does not cover the bars
if not args.linear:
    ax.set_xscale("log")
ax.axvline(np.median(t), color="black", ls="-", lw=1.5, label=f"median {np.median(t):.0f} s")
if not args.plain:
    ax.axvline(30, color="tab:green", ls="--", lw=1.5, label="30 s")
    ax.axvline(150, color="tab:orange", ls="--", lw=1.5, label="150 s (15 candidates $\\times$ 10 s)")
    ax.axvline(200, color="tab:red", ls="--", lw=1.5, label="200 s (per-structure cap in batch runs)")
ax.set_xlabel("Inversion wall-clock time per structure (s)")
ax.set_ylabel("Number of structures")
title = f"{args.title or args.tag}: n={len(t)} finished"
if args.n_censored:
    title += f" (+{args.n_censored} unfinished, slower than all shown)"
if args.plain:
    title = f"{args.title or args.tag} (n={n_total})"
ax.set_title(title, fontsize=11)
ax.legend(fontsize=8.5, loc="upper left", ncol=1 if args.plain else 2)
plt.tight_layout()
out = args.out or f"time_inversion_{args.tag}_{args.settings}_hist"
plt.savefig(out + ".pdf", bbox_inches="tight")
plt.savefig(out + ".png", dpi=150, bbox_inches="tight")
print(f"saved {out}.pdf / .png")
