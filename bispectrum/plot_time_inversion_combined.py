"""End-to-end inversion time per structure under the batch-run procedure (cap, then retry).

First pass (time_inversion_<tag>_default.pkl, run with --cap): a structure that finished has its own time;
a structure that hit the cap counts the cap time plus the time of its retry (time_inversion_<retry_tag>_default.pkl,
which was run on exactly the structures that hit the cap). Structures that hit the cap again on the retry are
unresolved (total >= 2 caps) and are left out of the histogram; they are counted and reported.

Usage:
    python plot_time_inversion_combined.py --first mp20_cap --retry mp20_cap_retry [--retry2 mp20_cap_retry2]
"""
import argparse
import pickle

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

mpl.rcParams["font.size"] = 12
p = argparse.ArgumentParser()
p.add_argument("--first", default="mp20_cap")
p.add_argument("--retry", default="mp20_cap_retry")
p.add_argument("--retry2", default=None, help="optional third attempt for structures that hit the cap twice")
p.add_argument("--no-cap-cost", action="store_true",
               help="for a structure that was stopped and retried, use only the time of the attempt that finished "
                    "(do not add the ~200 s spent on the stopped attempt)")
p.add_argument("--title", default="MP-20 test structures")
p.add_argument("--plain-title", action="store_true", help="title only, without the n / unresolved counts (put them in the caption)")
p.add_argument("--xlabel", default=None, help="override the x-axis label")
p.add_argument("--no-cap-line", action="store_true", help="omit the dotted line marking the 200 s cap")
p.add_argument("--out", default="time_inversion_mp20_combined_hist")
args = p.parse_args()


def load(tag):
    d = pickle.load(open(f"time_inversion_{tag}_default.pkl", "rb"))
    return {m: (t, bool(f)) for m, t, f in zip(d["material_ids"], d["times"], d["timed_out"])}


first, retry = load(args.first), load(args.retry)
retry2 = load(args.retry2) if args.retry2 else {}

total, unresolved = [], []
for m, (t1, hit1) in first.items():
    if not hit1:
        total.append(t1)
        continue
    t2, hit2 = retry[m]
    if not hit2:
        total.append(t2 if args.no_cap_cost else t1 + t2)
    elif m in retry2 and not retry2[m][1]:
        total.append(retry2[m][0] if args.no_cap_cost else t1 + t2 + retry2[m][0])
    else:
        unresolved.append(m)
total = np.array(total)
n = len(first)
print(f"n={n}: resolved {len(total)}, unresolved after retry {len(unresolved)} ({100 * len(unresolved) / n:.1f}%)")
print(f"resolved only: median {np.median(total):.0f}s  mean {total.mean():.0f}s  p90 {np.percentile(total, 90):.0f}s  "
      f"p95 {np.percentile(total, 95):.0f}s  max {total.max():.0f}s")
for th in (30, 120, 200, 300, 400):
    print(f"  finished within {th}s (incl. retries): {100 * np.sum(total < th) / n:.1f}% of all {n}")

fig, ax = plt.subplots(figsize=(6.5, 4))
ax.hist(total, bins=np.arange(0, total.max() + 15, 10), color="steelblue", alpha=0.85)
ax.axvline(np.median(total), color="black", lw=1.5, label=f"median {np.median(total):.0f} s")
if not args.no_cap_line:
    ax.axvline(200, color="gray", ls=":", lw=1.2)
default_label = "Inversion time of the attempt that finished (s)" if args.no_cap_cost else "Total inversion time per structure, including retries (s)"
ax.set_xlabel(args.xlabel or default_label)
ax.set_ylabel("Number of structures")
ax.set_title(args.title if args.plain_title else f"{args.title} (n={len(total)} of {n}; {len(unresolved)} unresolved)", fontsize=11)
ax.legend(fontsize=9, loc="upper right")
plt.tight_layout()
plt.savefig(args.out + ".pdf", bbox_inches="tight")
plt.savefig(args.out + ".png", dpi=150, bbox_inches="tight")
print(f"saved {args.out}.pdf / .png")
