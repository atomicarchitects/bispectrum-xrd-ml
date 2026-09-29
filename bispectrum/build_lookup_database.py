"""
Build the bispectrum lookup database used by run_alg_inversion.py
(materials_project_lat_bispec.h5).

The database has one entry per Materials Project structure in the full-MP set, i.e.
the train, validation and test splits are all included (run_alg_inversion.py excludes
only the query material itself from its nearest-neighbour search).

Note: the database used for the paper's results has 156,801 entries. This script
reproduces its first 154,879 entries exactly (checked with --compare). The remaining
1,922 entries are the MP-20 structures that are not in the full-MP splits, so that the
database covers every MP-20 structure as well.

    mp_ids         (N,)        fixed-length strings
    bispec         (N, 10, 35) bispectrum of the primitive reciprocal lattice
    recip_lattice  (N, 3, 3)   reciprocal_lattice_crystallographic matrix of the structure

Inputs (under $POWDERXRD_DATA_ROOT):
    materials_project_structures.pkl          {mp_id: pymatgen Structure}
    mpfull/{train,val,test}_data_bispec_list.pkl   {mp_id: bispectrum}, produced by
                                              training/mp_full/data_gen_full.py --mode bispec
                                              (which writes them directly into
                                              $POWDERXRD_DATA_ROOT; in our runs they were then
                                              kept under mpfull/, hence the --bispec-dir default)

Usage:
    python build_lookup_database.py [--output PATH] [--compare EXISTING.h5]
"""
import argparse
import os
import pickle

import h5py
import numpy as np

ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")


def load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--structures", default=f"{ROOT}/materials_project_structures.pkl")
    p.add_argument("--bispec-dir", default=f"{ROOT}/mpfull")
    p.add_argument("--output", default=f"{ROOT}/materials_project_lat_bispec_rebuilt.h5")
    p.add_argument("--compare", default=None,
                   help="existing database h5 to compare the rebuilt one against")
    p.add_argument("--check-primitive", type=int, default=0, metavar="N",
                   help="before building, check on N random structures that the stored lattice "
                        "equals the lattice of get_primitive_structure() (the cell the bispectra "
                        "were computed from, see utilities.training_data_gen)")
    p.add_argument("--compare-only", action="store_true",
                   help="do not build anything: compare the existing --output file with --compare")
    args = p.parse_args()

    if args.compare_only:
        compare(args.output, args.compare)
        return

    structures = load(args.structures)
    bispec = {}
    for split in ("train", "val", "test"):
        bispec.update(load(f"{args.bispec_dir}/{split}_data_bispec_list.pkl"))
    print(f"{len(structures)} structures, {len(bispec)} bispectra")

    mp_ids = list(bispec.keys())
    missing = [m for m in mp_ids if m not in structures]
    if missing:
        raise KeyError(f"{len(missing)} bispectra have no structure, e.g. {missing[:3]}")

    if args.check_primitive:
        rng = np.random.default_rng(0)
        sample = rng.choice(len(mp_ids), size=min(args.check_primitive, len(mp_ids)), replace=False)
        same_lattice = not_primitive = 0
        for i in sample:
            s = structures[mp_ids[i]]
            prim = s.get_primitive_structure()
            not_primitive += len(prim) != len(s)
            same_lattice += np.allclose(prim.lattice.matrix, s.lattice.matrix, atol=1e-6)
        print(f"primitive check on {len(sample)} structures: stored lattice == get_primitive_structure() "
              f"lattice for {same_lattice}; stored cell not primitive (atom count differs) for {not_primitive}")

    recip = np.stack([structures[m].lattice.reciprocal_lattice_crystallographic.matrix for m in mp_ids])
    spec = np.stack([np.asarray(bispec[m]) for m in mp_ids])

    if os.path.exists(args.output):
        raise FileExistsError(f"{args.output} exists; refusing to overwrite")
    with h5py.File(args.output, "w") as f:
        f.create_dataset("bispec", data=spec)
        f.create_dataset("recip_lattice", data=recip)
        f.create_dataset("mp_ids", data=np.array(mp_ids, dtype="S"))
    print(f"wrote {len(mp_ids)} entries -> {args.output} "
          f"(bispec {spec.shape} {spec.dtype}, recip {recip.shape})")

    if args.compare:
        compare(args.output, args.compare)


def compare(new_path, old_path):
    """Compare a rebuilt database with an existing one. Values are compared on the ids the two
    files share, and the ids present in only one of them are listed."""
    with h5py.File(new_path, "r") as fn, h5py.File(old_path, "r") as fo:
        ids_n = fn["mp_ids"][:].astype(str)
        ids_o = fo["mp_ids"][:].astype(str)
        print(f"compare: rebuilt {len(ids_n)} entries, existing {len(ids_o)} entries")
        print(f"compare: same ids and order: {len(ids_n) == len(ids_o) and bool(np.array_equal(ids_n, ids_o))}")
        only_old = sorted(set(ids_o) - set(ids_n))
        only_new = sorted(set(ids_n) - set(ids_o))
        print(f"compare: only in existing: {len(only_old)} (e.g. {only_old[:5]});  "
              f"only in rebuilt: {len(only_new)} (e.g. {only_new[:5]})")
        pos_n = {m: i for i, m in enumerate(ids_n)}
        pos_o = {m: i for i, m in enumerate(ids_o)}
        common = [m for m in ids_n if m in pos_o]
        sel_n = np.array([pos_n[m] for m in common])
        sel_o = np.array([pos_o[m] for m in common])
        for key in ("bispec", "recip_lattice"):
            a, b = fn[key][:][sel_n], fo[key][:][sel_o]
            diff = np.abs(a - b).reshape(len(common), -1).max(axis=1)
            print(f"compare: {key} on {len(common)} shared ids: max abs diff {diff.max():.3e}, "
                  f"entries differing by >1e-9: {int((diff > 1e-9).sum())}")


if __name__ == "__main__":
    main()
