"""
Check how the CNRS "atoms per structure" count used to define the <=20-atom
subset relates to the PRIMITIVE-cell count (what MP-20's atom limit refers to).

The earlier count (opxrd_cnrs_atom_counts.pkl) is len(basis) of the first phase
in each raw JSON, i.e. the atoms of the cell as deposited -- which may be a
centered/conventional cell. Here each phase is rebuilt as a pymatgen Structure
and reduced to its primitive cell; both counts are saved and compared.

Writes opxrd_cnrs_prim_atom_counts.pkl ({pid: dict(n_basis, n_prim, partial_occ)})
next to the old file (the old file is not modified).

Usage:
    python count_cnrs_primitive_atoms.py [--workers 32]
"""
import argparse
import json
import os
import pickle
import re
from multiprocessing import Pool

import numpy as np
from pymatgen.core import Element, Lattice, Structure

_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
CNRS_DIR = f"{_ROOT}/opxrd_cnrs_eval"
RAW_DIR = f"{CNRS_DIR}/CNRS"
VERIFIED_IDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits", "cnrs_verified_ids.txt")
TRUE_BISPEC = f"{_ROOT}/eval_results/opxrd_cnrs/opxrd_cnrs_true_bispec.pkl"


def count_one(fname):
    pid = "CNRS_" + fname.replace(".json", "")
    try:
        d = json.load(open(os.path.join(RAW_DIR, fname)))
        phase = json.loads(json.loads(d["label"])["phases"][0])
        basis = phase["basis"]
        basis = json.loads(basis) if isinstance(basis, str) else basis
        atoms = [json.loads(a) if isinstance(a, str) else a for a in basis]
        lat = phase["lattice"]
        if isinstance(lat, str):  # e.g. "(6.09, 6.09, 12.9, 90, 90, 90)" -- not valid JSON
            lat = [float(x) for x in re.findall(r"[-+]?\d+\.?\d*(?:[eE][-+]?\d+)?", lat)]
        assert len(lat) == 6, f"lattice has {len(lat)} values"
        species = [Element(re.match(r"[A-Z][a-z]?", a["symbol"]).group(0)) for a in atoms]
        coords = [[a["x"], a["y"], a["z"]] for a in atoms]
        partial = any(float(a.get("occupancy", 1)) < 1 for a in atoms)
        s = Structure(Lattice.from_parameters(*lat), species, coords)
        n_prim = len(s.get_primitive_structure(tolerance=0.25))
        return pid, dict(n_basis=len(atoms), n_prim=n_prim, partial_occ=partial)
    except Exception as e:
        return pid, dict(error=repr(e)[:100])


def load_verified_ids():
    with open(VERIFIED_IDS) as f:
        return {l.strip() for l in f if l.strip() and not l.startswith("#")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--workers", type=int, default=32)
    args = p.parse_args()

    files = sorted(f for f in os.listdir(RAW_DIR) if f.endswith(".json"))
    with Pool(args.workers) as pool:
        res = dict(pool.map(count_one, files, chunksize=4))
    ok = {k: v for k, v in res.items() if "n_prim" in v}
    print(f"{len(ok)}/{len(files)} converted; {len(res) - len(ok)} failed")
    if len(ok) < len(res):
        errs = [v["error"] for v in res.values() if "error" in v]
        print("  example errors:", errs[:3])
    pickle.dump(ok, open(f"{CNRS_DIR}/opxrd_cnrs_prim_atom_counts.pkl", "wb"))

    nb = np.array([v["n_basis"] for v in ok.values()])
    npm = np.array([v["n_prim"] for v in ok.values()])
    print(f"n_basis: median {np.median(nb):.0f}   n_prim: median {np.median(npm):.0f}")
    print(f"ratio n_basis/n_prim: {dict(zip(*np.unique(np.round(nb / npm).astype(int), return_counts=True)))}")
    print(f"partial occupancy: {sum(v['partial_occ'] for v in ok.values())} patterns")

    vids = load_verified_ids()
    has_bispec = set(pickle.load(open(TRUE_BISPEC, "rb")))
    print("\n<=20 atoms, by definition (all / verified / verified with true bispec = evaluable):")
    for lab, key in [("len(basis)   ", "n_basis"), ("primitive cell", "n_prim")]:
        s_all = {k for k, v in ok.items() if v[key] <= 20}
        print(f"  {lab}: {len(s_all)} / {len(s_all & vids)} / {len(s_all & vids & has_bispec)}")
    a = {k for k, v in ok.items() if v["n_basis"] <= 20} & vids & has_bispec
    b = {k for k, v in ok.items() if v["n_prim"] <= 20} & vids & has_bispec
    print(f"  evaluable sets: both={len(a & b)}  only-by-basis={len(a - b)}  only-by-primitive={len(b - a)}")


if __name__ == "__main__":
    main()
