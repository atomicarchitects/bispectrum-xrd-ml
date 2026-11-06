"""
Build the target file for the Crystalyze RRUFF evaluation (eval_rruff_crystalyze.py,
compute_rruff_true_bispec.py): the 148 entries of splits/rruff_crystalyze_ids.txt, in that order,
with the primitive-cell lattice parameters (a, b, c, alpha, beta, gamma) of each structure.

Input is Crystalyze's released RRUFF test set (test.csv, one row per mineral with its CIF in the
`cif` column; Riesel et al., 2024), placed in $POWDERXRD_DATA_ROOT/rruff_crystalyze/. The primitive
cell is pymatgen's get_primitive_structure(), the convention used for every other target in this
repository.

Output: a pickle {'mineral_names': ['Ferberite__R040044', ...], 'targets': float array [148, 6]},
by default $POWDERXRD_DATA_ROOT/rruff_crystalyze/rruff_crystalyze_targets.pkl.

Usage:
    python build_rruff_crystalyze_targets.py
    python build_rruff_crystalyze_targets.py --test-csv /path/to/test.csv --out /path/to/targets.pkl
"""
import argparse
import os
import pickle

import numpy as np
import pandas as pd
from pymatgen.core import Structure

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")


def read_ids(path):
    """(mineral, rruff_id) pairs from a tab-separated split file, skipping '#' comments."""
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                mineral, rruff_id = line.split("\t")
                pairs.append((mineral, rruff_id))
    return pairs


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--test-csv", default=f"{_ROOT}/rruff_crystalyze/test.csv")
    p.add_argument("--ids", default=os.path.join(HERE, "splits", "rruff_crystalyze_ids.txt"))
    p.add_argument("--out", default=f"{_ROOT}/rruff_crystalyze/rruff_crystalyze_targets.pkl")
    args = p.parse_args()

    pairs = read_ids(args.ids)
    cifs = dict(zip(*[pd.read_csv(args.test_csv)[c] for c in ("material_id", "cif")]))
    missing = [m for m, _ in pairs if m not in cifs]
    if missing:
        raise SystemExit(f"{len(missing)} minerals of {args.ids} are not in {args.test_csv}: {missing[:5]}")

    targets = []
    for mineral, _ in pairs:
        lat = Structure.from_str(cifs[mineral], fmt="cif").get_primitive_structure().lattice
        targets.append(list(lat.abc) + list(lat.angles))

    out = {"mineral_names": [f"{m}__{r}" for m, r in pairs], "targets": np.array(targets)}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "wb") as f:
        pickle.dump(out, f)
    print(f"wrote {args.out}: {len(pairs)} entries")


if __name__ == "__main__":
    main()
