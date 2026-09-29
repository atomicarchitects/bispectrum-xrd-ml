"""How close is the winning database start to the true lattice?

For each inverted material, look up the winning start's UNPERTURBED database lattice (start_id in
materials_project_lat_bispec.h5), Niggli-reduce it, and compare to the Niggli-reduced true lattice.
This is the retrieval-only error (what you would get by returning the database entry's cell as-is),
and the fraction of materials with a near-duplicate winning start (leakage / retrieval baseline).

start_id is only trustworthy for results produced after the winner-tracking fix in
run_alg_inversion.py (older files store the LAST candidate tried). Old files are still valid for
materials that converged (min_loss < 1e-3): the loop breaks on the winning candidate there, so the
last one tried is the winner. Use --converged-only for such files.

Usage:
    python analyze_start_leakage.py FILE.h5 [FILE2.h5 ...] [--converged-only]
"""
import argparse
import os

import h5py
import numpy as np
from pymatgen.core import Lattice

ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
DB = f"{ROOT}/materials_project_lat_bispec.h5"
CUTOFF = 1e-3  # convergence cutoff inside process_row


def niggli(mat33):
    n = Lattice(mat33).get_niggli_reduced_lattice()
    return np.array([n.a, n.b, n.c]), np.array([n.alpha, n.beta, n.gamma])


def errs(L, Lt):
    (a, al), (at, alt) = niggli(L), niggli(Lt)
    return 100 * np.mean(np.abs(a - at) / at), np.mean(np.abs(al - alt))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--converged-only", action="store_true")
    args = p.parse_args()

    with h5py.File(DB, "r") as f:
        db_ids = f["mp_ids"][:].astype(str)
        db_recip = f["recip_lattice"][:]
    index = {m: i for i, m in enumerate(db_ids)}

    for path in args.files:
        rows, n_none, n_missing, n_skipped = [], 0, 0, 0
        with h5py.File(path, "r") as f:
            for k in f.keys():
                g = f[k]
                sid = g.attrs.get("start_id")
                sid = sid.decode() if isinstance(sid, bytes) else str(sid)
                if args.converged_only and not g.attrs["min_loss"] < CUTOFF:
                    n_skipped += 1
                    continue
                if sid == "None":
                    n_none += 1
                    continue
                if sid not in index or "true_lattice" not in g:
                    n_missing += 1
                    continue
                try:
                    db_real = np.linalg.inv(db_recip[index[sid]].reshape(3, 3)).T
                    true = g["true_lattice"][()].reshape(3, 3)
                    pred = g["pred"][()].reshape(3, 3)
                    rl, ra = errs(db_real, true)
                    pl, pa = errs(pred, true)
                except Exception:
                    n_missing += 1
                    continue
                rows.append((rl, ra, pl, pa))
        r = np.array(rows)
        print(f"\n{os.path.basename(path)}")
        print(f"  scored {len(r)}  (random-start winners {n_none}, unusable {n_missing}, "
              f"skipped non-converged {n_skipped})")
        if not len(r):
            continue
        print(f"  retrieval-only (winning DB lattice, unperturbed) : length MAPE mean {r[:,0].mean():.1f}% "
              f"median {np.median(r[:,0]):.1f}%   angle MAE mean {r[:,1].mean():.1f} deg")
        print(f"  final prediction                                  : length MAPE mean {r[:,2].mean():.1f}% "
              f"median {np.median(r[:,2]):.1f}%   angle MAE mean {r[:,3].mean():.1f} deg")
        for th in (1, 2, 5, 10):
            print(f"  winning start within {th:2d}% length MAPE of truth: "
                  f"{100 * np.mean(r[:,0] < th):5.1f}%   (retrieval-only angle < 3 deg too: "
                  f"{100 * np.mean((r[:,0] < th) & (r[:,1] < 3)):5.1f}%)")
        print(f"  prediction better than the retrieved lattice on {100 * np.mean(r[:,2] < r[:,0]):.0f}% of materials")
        print("  stratified by how close the winning start is to the truth (retrieval length MAPE):")
        for lo, hi in ((0, 2), (2, 10), (10, np.inf)):
            m = (r[:, 0] >= lo) & (r[:, 0] < hi)
            if m.any():
                print(f"    {lo:>2}-{'inf' if hi == np.inf else hi:>3}%: n={m.sum():3d}  retrieval {r[m,0].mean():5.1f}%  "
                      f"final prediction {r[m,2].mean():5.1f}% (median {np.median(r[m,2]):.1f}%)  angle MAE {r[m,3].mean():.1f} deg")


if __name__ == "__main__":
    main()
