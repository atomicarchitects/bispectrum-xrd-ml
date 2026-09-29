"""
Full-MP version of mp20_xrd_leakage.py: cross-split leakage audit based on
the actual simulated XRD PATTERN (the literal model input), not bispec or
structure-matching -- both of which turned out to be purely geometric
(bispec here is computed from recip_lattice alone, no composition; the
same is true of StructureMatcher with FrameworkComparator), and were shown
(on MP-20) to badly over-count by flagging chemically-unrelated compounds
that merely share a common simple lattice prototype (e.g. Sc2CoCu vs
Be2Ag) as "duplicates," when their actual XRD intensities (which depend
on atomic form factors, not just lattice geometry) differ substantially.

Uses mpfull_crystalyze_cag/{train,val,test}_sim_xrd.pkl -- single canonical
curve per material (bare mp-id keys, no augmentation multiplier), much
smaller than train_sim_xrd_combined.pkl (3.16GB vs 31.5GB) and avoids the
suffix-filtering workaround entirely.

Usage:
    python mpfull_xrd_leakage.py
"""
import os
import io
import contextlib
import sys
import pickle
import time

import h5py
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
_buf = io.StringIO()
with contextlib.redirect_stdout(_buf):
    from analyze_completed_inversion_res import niggli_params

DATA_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
SIM_THRESHOLD = 0.999
LEN_APE_THRESHOLD = 3.0
ANG_THRESHOLD = 3.0
CHUNK = 500  # smaller than mp20's -- train pool is ~3.4x bigger here


def sorted_niggli_params(recip_mat33):
    real = np.linalg.pinv(recip_mat33.T)
    a, b, c, al, be, ga = niggli_params(real)
    return np.array(sorted([a, b, c])), np.array(sorted([al, be, ga]))


def lattice_diff(recip_a, recip_b):
    la, aa = sorted_niggli_params(recip_a)
    lb, ab = sorted_niggli_params(recip_b)
    len_ape = 100 * np.abs(la - lb) / np.maximum(lb, 1e-6)
    ang_diff = np.abs(aa - ab)
    return float(len_ape.max()), float(ang_diff.max())


def main():
    base = f"{DATA_ROOT}/mpfull_crystalyze_cag"

    print("Loading train/val/test curves (clean, single curve per material)...")
    t0 = time.time()
    train_curves = pickle.load(open(f"{base}/train_sim_xrd.pkl", "rb"))
    val_curves = pickle.load(open(f"{base}/val_sim_xrd.pkl", "rb"))
    test_curves = pickle.load(open(f"{base}/test_sim_xrd.pkl", "rb"))
    print(f"  train={len(train_curves)}  val={len(val_curves)}  test={len(test_curves)}  ({time.time()-t0:.1f}s)")

    print("\nLoading materials_project_lat_bispec.h5 for recip_lattice (confirmation step only)...")
    f = h5py.File(f"{DATA_ROOT}/materials_project_lat_bispec.h5", "r")
    mp_ids_raw = f["mp_ids"][:]
    mp_ids = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in mp_ids_raw])
    recip = f["recip_lattice"][:]
    f.close()
    id_to_ridx = {mid: i for i, mid in enumerate(mp_ids)}

    print("\nBuilding normalized curve matrices...")
    train_ids = list(train_curves.keys())
    train_mat = np.stack([np.asarray(train_curves[m], dtype=np.float32) for m in train_ids])
    train_norms = np.linalg.norm(train_mat, axis=1, keepdims=True)
    train_norms[train_norms == 0] = 1.0
    train_mat_n = train_mat / train_norms
    del train_curves

    results = {}
    for split_name, curves in [("val", val_curves), ("test", test_curves)]:
        ids = list(curves.keys())
        n = len(ids)
        mat = np.stack([np.asarray(curves[m], dtype=np.float32) for m in ids])
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat_n = mat / norms

        best_sim = np.full(n, -2.0, dtype=np.float32)
        best_train_id = np.empty(n, dtype=object)

        print(f"\nSearching {split_name} ({n}) against train ({len(train_ids)}) by XRD curve cosine similarity...")
        t0 = time.time()
        for start in range(0, n, CHUNK):
            end = min(start + CHUNK, n)
            sims = mat_n[start:end] @ train_mat_n.T
            top = np.argmax(sims, axis=1)
            best_sim[start:end] = sims[np.arange(end - start), top]
            best_train_id[start:end] = [train_ids[t] for t in top]
            if (start // CHUNK) % 10 == 0 and start > 0:
                print(f"  {end}/{n}  ({time.time()-t0:.1f}s elapsed)")
        print(f"  done in {time.time()-t0:.1f}s")

        flagged_mask = best_sim > SIM_THRESHOLD
        flagged = np.sum(flagged_mask)
        print(f"{split_name}: {flagged}/{n} ({100*flagged/n:.2f}%) XRD-curve-flagged (>{SIM_THRESHOLD})")

        flagged_idx = np.where(flagged_mask)[0]
        max_len_ape = np.full(n, np.nan, dtype=np.float32)
        max_ang_diff = np.full(n, np.nan, dtype=np.float32)
        t0 = time.time()
        n_no_lattice = 0
        for j, i in enumerate(flagged_idx):
            a_id, b_id = ids[i], best_train_id[i]
            if a_id not in id_to_ridx or b_id not in id_to_ridx:
                n_no_lattice += 1
                continue
            try:
                la, ad = lattice_diff(recip[id_to_ridx[a_id]].reshape(3, 3), recip[id_to_ridx[b_id]].reshape(3, 3))
                max_len_ape[i] = la
                max_ang_diff[i] = ad
            except Exception:
                pass
        print(f"  confirmed {len(flagged_idx)} candidates in {time.time()-t0:.1f}s ({n_no_lattice} missing lattice data)")

        confirmed_mask = flagged_mask & (max_len_ape < LEN_APE_THRESHOLD) & (max_ang_diff < ANG_THRESHOLD)
        confirmed = np.sum(confirmed_mask)
        if flagged:
            print(f"{split_name}: {confirmed}/{n} ({100*confirmed/n:.2f}%) CONFIRMED near-duplicate XRD pattern "
                  f"(false-positive rate of curve-similarity-only: {100*(flagged-confirmed)/flagged:.1f}%)")

        results[split_name] = dict(mp_ids=np.array(ids), best_sim=best_sim, best_train_id=best_train_id,
                                    max_len_ape=max_len_ape, max_ang_diff=max_ang_diff,
                                    confirmed_mask=confirmed_mask)

    out_path = f"{DATA_ROOT}/eval_results/mpfull_xrd_leakage_audit.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump(results, fh)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
