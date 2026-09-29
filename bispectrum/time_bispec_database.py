"""
Cost of the bispectrum lookup database (materials_project_lat_bispec.h5): its size,
construction time and lookup cost.

  1. Size: entries, per-array bytes, on-disk size, in-RAM footprint.
  2. Construction: recompute the bispectrum of a random sample of database lattices
     with the training settings (generate_augmented_data.BISPEC_*), single thread,
     check they reproduce the stored values, and time it per material. The database
     entry is a function of the primitive reciprocal lattice only (no XRD simulation,
     no species), so this is the whole per-entry cost apart from reading the structure.
  3. Lookup: time one nearest-neighbour query exactly as run_alg_inversion.process_row
     does it (float64 copy, cube-root scaling, MSE against every entry, top-k).

Usage (small, single-thread; fine on a login/interactive node):
    python time_bispec_database.py [--n 100] [--queries 5]
"""
import argparse
import os
import resource
import sys
import time

import h5py
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))                # utilities.py
sys.path.insert(0, os.path.join(HERE, "..", "training", "mp_full"))
from utilities import get_bispectrum, get_neighbors, training_data_gen
from generate_augmented_data import BISPEC_WAVELENGTH, BISPEC_LMAX, BISPEC_RADIAL, BISPEC_N_BASIS

_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
DB = f"{_ROOT}/materials_project_lat_bispec.h5"
COB = f"{_ROOT}/cob_lmax6.pkl"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=100)
    p.add_argument("--queries", type=int, default=5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--end-to-end", type=int, default=0, metavar="N",
                   help="also time the production call (utilities.training_data_gen, which includes "
                        "the primitive-cell reduction) on N random structures from "
                        "$POWDERXRD_DATA_ROOT/materials_project_structures.pkl")
    args = p.parse_args()
    torch.set_num_threads(1)

    print(f"file: {DB}  {os.path.getsize(DB) / 1e6:.1f} MB on disk;  cob file {os.path.getsize(COB) / 1e6:.1f} MB")
    with h5py.File(DB, "r") as f:
        ids = f["mp_ids"][:]
        bispec = f["bispec"][:]
        recip = f["recip_lattice"][:].reshape(-1, 3, 3)
    n_db = len(ids)
    print(f"entries: {n_db}   bispec {bispec.shape} {bispec.dtype} = {bispec.nbytes / 1e6:.1f} MB;  "
          f"recip {recip.shape} = {recip.nbytes / 1e6:.1f} MB;  ids {ids.nbytes / 1e6:.1f} MB")
    nz = np.count_nonzero(np.any(np.abs(bispec[:2000]) > 1e-10, axis=0))
    print(f"non-zero (10,35) channels in first 2000 entries: {nz}/350")

    cob = torch.load(COB, map_location="cpu", weights_only=False)
    k_max = 2.0 / BISPEC_WAVELENGTH
    print(f"settings: wavelength={BISPEC_WAVELENGTH} lmax={BISPEC_LMAX} n_basis={BISPEC_N_BASIS} k_max={k_max:.3f}")

    rng = np.random.default_rng(args.seed)
    sel = rng.choice(n_db, size=args.n, replace=False)
    times, diffs, nrefl = [], [], []
    for i in sel:
        r = recip[i]
        t0 = time.perf_counter()
        miller = get_neighbors(r, k_max)
        neighs = torch.tensor(miller.reshape(-1, 3) @ r, dtype=torch.float64)
        b = get_bispectrum(neighs, k_max, BISPEC_LMAX, BISPEC_RADIAL, BISPEC_N_BASIS, cob).numpy()
        times.append(time.perf_counter() - t0)
        diffs.append(np.max(np.abs(b - bispec[i])))
        nrefl.append(len(miller))
    times = np.array(times)
    print(f"\nrecomputed {args.n} random entries (single thread)")
    print(f"  max |recomputed - stored|: median {np.median(diffs):.2e}, worst {np.max(diffs):.2e}  (stored scale ~{np.abs(bispec[sel]).max():.2e})")
    print(f"  time per entry: mean {times.mean() * 1e3:.1f} ms, median {np.median(times) * 1e3:.1f} ms, p95 {np.percentile(times, 95) * 1e3:.1f} ms;"
          f"  reflections in shell: median {int(np.median(nrefl))}")
    total = times.mean() * n_db
    print(f"  => full database, single core: {total / 3600:.1f} h;  on 128 workers: {total / 128 / 60:.1f} min  (excludes reading/reducing the structures)")

    test = torch.tensor(bispec[sel[0]], dtype=torch.float64)
    test = torch.sign(test) * torch.pow(torch.abs(test), 1 / 3)
    qt = []
    for q in range(args.queries):
        t0 = time.perf_counter()
        mask = ids != ids[sel[q]]
        cand = torch.tensor(bispec[mask], dtype=torch.float64)
        cand = torch.sign(cand) * torch.pow(torch.abs(cand), 1 / 3)
        losses = torch.nn.MSELoss(reduction="none")(test.unsqueeze(0).expand_as(cand), cand).mean(dim=(1, 2))
        torch.topk(-losses, 30)
        qt.append(time.perf_counter() - t0)
    print(f"\nnearest-neighbour query over {n_db} entries (top-30, single thread): {np.mean(qt):.2f} s each")
    print(f"  transient float64 copy per query: {cand.numel() * 8 / 1e6:.0f} MB (plus the cube-root and expanded tensors)")
    print(f"peak RSS of this process: {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6:.2f} GB")

    if args.end_to_end:
        end_to_end(args, cob)


def end_to_end(args, cob):
    """Time the call data_gen_full.py makes per structure: primitive-cell reduction, reciprocal
    lattice, neighbours and bispectrum (single thread, structures already in memory)."""
    import pickle
    torch.set_default_dtype(torch.float64)
    with open(f"{_ROOT}/materials_project_structures.pkl", "rb") as f:
        structs = pickle.load(f)
    keys = list(structs)
    rng = np.random.default_rng(args.seed)
    sel = rng.choice(len(keys), size=min(args.end_to_end, len(keys)), replace=False)
    times, natoms, nprim = [], [], []
    for i in sel:
        s = structs[keys[i]]
        t0 = time.perf_counter()
        training_data_gen([{"crystal": s}], cob_bi=cob)
        times.append(time.perf_counter() - t0)
        natoms.append(len(s))
    times = np.array(times)
    n_all = len(keys)
    print(f"\nend-to-end (primitive reduction + bispectrum) on {len(sel)} random structures, single thread")
    print(f"  atoms per stored structure: median {int(np.median(natoms))}, max {max(natoms)}")
    print(f"  time per structure: mean {times.mean() * 1e3:.1f} ms, median {np.median(times) * 1e3:.1f} ms, "
          f"p95 {np.percentile(times, 95) * 1e3:.1f} ms, max {times.max() * 1e3:.0f} ms")
    total = times.mean() * n_all
    print(f"  => all {n_all} structures, single core: {total / 3600:.1f} h;  on 128 workers: {total / 128 / 60:.1f} min "
          f"(excludes loading the structures)")


if __name__ == "__main__":
    main()
