"""
Computes true Bravais lattice type (+ true lattice) per structure, independent of
any model prediction, from one of three source formats. Output is a pickle
consumed via inversion_results_cctbx.py's --true-bravais flag.

Source formats (pick one subcommand):
  csv      A CSV with an embedded CIF string per row (e.g. $POWDERXRD_DATA_ROOT/mp20/test.csv,
           $POWDERXRD_DATA_ROOT/mpfull/test.csv). Bravais type is computed from the CIF via
           SpacegroupAnalyzer. Output columns: material_id, true_bravais, true_lattice.

  cif-dir  A directory of standalone .cif files (material_id = filename without
           .cif). Same SpacegroupAnalyzer computation as csv mode, just a
           different input format. Output columns: material_id, true_bravais,
           true_lattice.

  rruff    $POWDERXRD_DATA_ROOT/rruff_data_dict.pkl. RRUFF entries already carry a
           space_group symbol in their metadata, so no CIF parsing/symmetry
           detection is needed -- just a symbol -> Bravais lookup (with a
           fallback table for non-standard RRUFF symbols pymatgen can't parse
           directly). Output columns: pattern_id, mineral_name, space_group,
           true_bravais, true_lattice.

Usage:
    python compute_true_bravais.py csv --csv $POWDERXRD_DATA_ROOT/mp20/test.csv --output <out.pkl>
    python compute_true_bravais.py cif-dir --cif-dir <dir> --output <out.pkl>
    python compute_true_bravais.py rruff --output <out.pkl>
"""

import argparse
import os
import pickle
import re
import sys

import pandas as pd
from pymatgen.core.structure import Structure
from pymatgen.io.cif import CifParser
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.symmetry.groups import SpaceGroup

sys.path.append("..")
from utilities import get_bravais_from_spg_number


def parse_cif_str(cif_str, symprec):
    struct = Structure.from_str(cif_str, fmt="cif")
    sga = SpacegroupAnalyzer(struct, symprec=symprec)
    bravais = get_bravais_from_spg_number(sga.get_space_group_number())
    return struct.lattice.matrix, bravais


def parse_cif_file(path, symprec):
    parser = CifParser(path, occupancy_tolerance=10.0)
    struct = parser.parse_structures(primitive=False)[0]
    sga = SpacegroupAnalyzer(struct, symprec=symprec)
    bravais = get_bravais_from_spg_number(sga.get_space_group_number())
    return struct.lattice.matrix, bravais


# Direct Bravais mapping for non-standard RRUFF symbols that pymatgen cannot parse.
# Inferred from centering letter + crystal system of each symbol.
_NONSTANDARD_BRAVAIS = {
    # Monoclinic: A/B/F/I-centered -> mC (equivalent to C-centered); P -> mP
    "A2": "mC", "A2/a": "mC", "A2/m": "mC", "Aa": "mC",
    "B2/b": "mC", "B2_1": "mC",
    "F2/d": "mC", "F2/m": "mC",
    "I2": "mC", "I2/a": "mC", "I2/c": "mC", "I2/m": "mC", "Ia": "mC",
    "P2/a": "mP", "P2/n": "mP",
    "P2_1/a": "mP", "P2_1/b": "mP", "P2_1/n": "mP",
    "Pa": "mP", "Pn": "mP",
    # Orthorhombic: A/B/C-centered -> oC; P -> oP
    "Abma": "oC", "Acam": "oC",
    "Bb2_1m": "oC", "Bba2": "oC", "Bbcm": "oC",
    "Ccm2_1": "oC", "Ccmb": "oC", "Cmca": "oC",
    "P2_12_12_": "oP",
    "P2_1ca": "oP", "P2_1cn": "oP", "P2_1nb": "oP", "P2_1nm": "oP",
    "Pbc2_1": "oP", "Pbn2_1": "oP",
    "Pn2_1a": "oP", "Pn2_1m": "oP", "Pnm2_1": "oP",
}


class FlexibleUnpickler(pickle.Unpickler):
    """Unpickler that handles missing XRDStructure class gracefully."""
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            class GenericObj:
                def __init__(self, *args, **kwargs):
                    pass
            return GenericObj


def get_bravais_from_symbol(symbol: str) -> str:
    # 1. Try pymatgen directly
    try:
        return get_bravais_from_spg_number(SpaceGroup(symbol).int_number)
    except Exception:
        pass

    # 2. Normalize cubic symbols: insert '-' before bare '3' (e.g. Fd3m -> Fd-3m)
    normalized = re.sub(r'(?<![- ])3', '-3', symbol)
    if normalized != symbol:
        try:
            return get_bravais_from_spg_number(SpaceGroup(normalized).int_number)
        except Exception:
            pass

    # 3. Manual mapping for known non-standard RRUFF symbols
    if symbol in _NONSTANDARD_BRAVAIS:
        return _NONSTANDARD_BRAVAIS[symbol]

    raise ValueError(f"Could not resolve space group '{symbol}'")


def report(out, id_col):
    n_failed = out['true_lattice'].isna().sum()
    if n_failed:
        print(f"Warning: {n_failed} entries failed (true_lattice will be None)")
    print("\nBravais type distribution:")
    print(out['true_bravais'].value_counts().to_string())
    print(f"\n{len(out)} entries, keyed by '{id_col}'")


def run_csv(args):
    csv = pd.read_csv(args.csv)
    print(f"Loaded {len(csv)} entries from {args.csv}")
    print(f"Using symprec={args.symprec}")

    true_lattices, true_bravais = [], []
    for _, row in csv.iterrows():
        try:
            lattice, bravais = parse_cif_str(row[args.cif_col], args.symprec)
            true_lattices.append(lattice)
            true_bravais.append(bravais)
        except Exception as e:
            print(f"  Failed for {row[args.id_col]}: {e}")
            true_lattices.append(None)
            true_bravais.append('aP')

    out = pd.DataFrame({
        'material_id': csv[args.id_col],
        'true_bravais': true_bravais,
        'true_lattice': true_lattices,
    })
    report(out, 'material_id')
    out.to_pickle(args.output)
    print(f"Saved to {args.output}")


def run_cif_dir(args):
    cif_files = sorted(f for f in os.listdir(args.cif_dir) if f.endswith('.cif'))
    print(f"Found {len(cif_files)} CIF files in {args.cif_dir}")
    print(f"Using symprec={args.symprec}")

    rows = []
    for fname in cif_files:
        mat_id = fname[:-4]  # strip .cif
        path = os.path.join(args.cif_dir, fname)
        try:
            lattice, bravais = parse_cif_file(path, args.symprec)
            rows.append({'material_id': mat_id, 'true_bravais': bravais, 'true_lattice': lattice})
        except Exception as e:
            print(f"  Failed for {fname}: {e}")
            rows.append({'material_id': mat_id, 'true_bravais': 'aP', 'true_lattice': None})

    out = pd.DataFrame(rows)
    report(out, 'material_id')
    out.to_pickle(args.output)
    print(f"Saved to {args.output}")


def run_rruff(args):
    with open(args.input, "rb") as f:
        data = FlexibleUnpickler(f).load()
    print(f"Loaded {len(data)} entries from {args.input}")

    from pymatgen.core import Lattice

    pattern_ids, mineral_names, spg_symbols, true_bravais, true_lattices = [], [], [], [], []
    n_failed = 0
    for key, entry in data.items():
        meta = entry.metadata.iloc[0]
        spg_symbol = meta["space_group"]
        mineral = meta["mineral_name"]
        try:
            bravais = get_bravais_from_symbol(spg_symbol)
        except Exception as e:
            print(f"  Failed for {key} (spg='{spg_symbol}'): {e}")
            bravais = "aP"
            n_failed += 1

        lattice = Lattice.from_parameters(
            meta["a"], meta["b"], meta["c"],
            meta["alpha"], meta["beta"], meta["gamma"]
        ).get_niggli_reduced_lattice().matrix

        pattern_ids.append(key)
        mineral_names.append(mineral)
        spg_symbols.append(spg_symbol)
        true_bravais.append(bravais)
        true_lattices.append(lattice)

    out = pd.DataFrame({
        "pattern_id": pattern_ids,
        "mineral_name": mineral_names,
        "space_group": spg_symbols,
        "true_bravais": true_bravais,
        "true_lattice": true_lattices,
    })
    if n_failed:
        print(f"\nWarning: {n_failed} entries failed (defaulted to 'aP')")
    report(out, 'pattern_id')
    out.to_pickle(args.output)
    print(f"Saved to {args.output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='source', required=True)

    p_csv = sub.add_parser('csv', help='CSV with an embedded CIF string per row (MP-20/MP-Full).')
    p_csv.add_argument('--csv', required=True, help='CSV with id and CIF columns.')
    p_csv.add_argument('--id-col', default='material_id')
    p_csv.add_argument('--cif-col', default='cif')
    p_csv.add_argument('--output', required=True)
    p_csv.add_argument('--symprec', type=float, default=1e-2)
    p_csv.set_defaults(func=run_csv)

    p_dir = sub.add_parser('cif-dir', help='Directory of standalone .cif files.')
    p_dir.add_argument('--cif-dir', required=True)
    p_dir.add_argument('--output', required=True)
    p_dir.add_argument('--symprec', type=float, default=1e-2)
    p_dir.set_defaults(func=run_cif_dir)

    p_rruff = sub.add_parser('rruff', help='rruff_data_dict.pkl (space group already in metadata).')
    root = os.environ.get('POWDERXRD_DATA_ROOT', './data')
    p_rruff.add_argument('--input', default=os.path.join(root, 'rruff_data_dict.pkl'))
    p_rruff.add_argument('--output', default=os.path.join(root, 'rruff_true_bravais.pkl'))
    p_rruff.set_defaults(func=run_rruff)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
