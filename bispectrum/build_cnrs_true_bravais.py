"""
Reshapes $POWDERXRD_DATA_ROOT/opxrd_cnrs_metadata.pkl (nested dict, one entry per
pattern_id: {'spacegroup', 'bravais_lattice', 'lattice_system'}) into the
flat pattern_id/true_bravais table inversion_results_cctbx.py's
--true-bravais expects (same schema as rruff_alpha_true_bravais.pkl).

Usage:
    python build_cnrs_true_bravais.py
"""
import os
import pickle

import pandas as pd

ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
INPUT = os.path.join(ROOT, 'opxrd_cnrs_metadata.pkl')
OUTPUT = os.path.join(ROOT, 'opxrd_cnrs_true_bravais.pkl')


def main():
    meta = pickle.load(open(INPUT, 'rb'))
    df = pd.DataFrame({
        'pattern_id': list(meta.keys()),
        'true_bravais': [v['bravais_lattice'] for v in meta.values()],
    })
    df.to_pickle(OUTPUT)
    print(f'Wrote {len(df)} entries to {OUTPUT}')
    print(df['true_bravais'].value_counts())


if __name__ == '__main__':
    main()
