"""
Cross-split leakage audit for MP-20 based on the actual simulated XRD
PATTERN (the literal model input), not bispectrum or structure-matching.

This supersedes the bispec-based (mp20_bispec_leakage.py) and structure-
matching (mp20_structural_resplit_clusters.py) approaches: both of those
turned out to be purely GEOMETRIC criteria -- bispectrum here is computed
import os
from recip_lattice alone (get_true_sig(recip_lattice, ...), no species/
composition input at all), and StructureMatcher with FrameworkComparator
also ignores species by design. Neither reflects that XRD peak
INTENSITIES depend on atomic form factors / composition, not just lattice
geometry -- so both under/over-count relative to what the model actually
sees. Comparing the simulated XRD curves directly sidesteps this: two
materials only count as near-duplicate if their actual model-input curves
are close, which naturally incorporates both geometry (peak positions)
and composition (peak intensities via form factors).

Uses cosine similarity on the raw 8500-point dense curves (train_sim_xrd.pkl
/ val_sim_xrd.pkl / test_sim_xrd.pkl, single canonical curve per material,
mp20_crystalyze_cag) -- top-1 nearest-neighbor per val/test material
against train, same two-stage design as check_bispec_leakage.py (curve
similarity as the fast filter, Niggli-reduced lattice comparison as
confirmation, using mp20_lat_bispec.h5's recip_lattice for that step).

Usage:
    python mp20_xrd_leakage.py
"""
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
CHUNK = 1000


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
    base = f"{DATA_ROOT}/mp20_crystalyze_cag"
    print("Loading dense XRD curves...")
    curves = {}
    split_ids = {}
    for split in ["train", "val", "test"]:
        t0 = time.time()
        d = pickle.load(open(f"{base}/{split}_sim_xrd.pkl", "rb"))
        split_ids[split] = list(d.keys())
        curves.update(d)
        del d
        print(f"  {split}: {len(split_ids[split])} curves loaded in {time.time()-t0:.1f}s")

    print("\nLoading mp20_lat_bispec.h5 for recip_lattice (confirmation step only)...")
    f = h5py.File(f"{DATA_ROOT}/mp20_lat_bispec.h5", "r")
    mp_ids_raw = f["mp_ids"][:]
    mp_ids = np.array([x.decode() if isinstance(x, bytes) else str(x) for x in mp_ids_raw])
    recip = f["recip_lattice"][:]
    f.close()
    id_to_ridx = {mid: i for i, mid in enumerate(mp_ids)}

    print("\nBuilding normalized curve matrices...")
    train_ids = split_ids["train"]
    train_mat = np.stack([np.asarray(curves[m], dtype=np.float32) for m in train_ids])
    train_norms = np.linalg.norm(train_mat, axis=1, keepdims=True)
    train_norms[train_norms == 0] = 1.0
    train_mat_n = train_mat / train_norms

    results = {}
    for split_name in ["val", "test"]:
        ids = split_ids[split_name]
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
        print(f"{split_name}: {confirmed}/{n} ({100*confirmed/n:.2f}%) CONFIRMED near-duplicate XRD pattern "
              f"(false-positive rate of curve-similarity-only: "
              f"{100*(flagged-confirmed)/flagged:.1f}%)" if flagged else "")

        results[split_name] = dict(mp_ids=np.array(ids), best_sim=best_sim, best_train_id=best_train_id,
                                    max_len_ape=max_len_ape, max_ang_diff=max_ang_diff,
                                    confirmed_mask=confirmed_mask)

    out_path = f"{DATA_ROOT}/eval_results/mp20_xrd_leakage_audit.pkl"
    with open(out_path, "wb") as fh:
        pickle.dump(results, fh)
    print(f"\nSaved -> {out_path}")


if __name__ == "__main__":
    main()
