"""
Filter processed_xy inversion result pkl files to only the entries
whose RRUFF ID appears in $POWDERXRD_DATA_ROOT/rruff_filtered.csv.

Filtered copies are written alongside each original as
inversion_results_cctbx_filtered.pkl.
"""

import glob
import os
import re

import pandas as pd

ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
FILTERED_CSV = os.path.join(ROOT, 'rruff_filtered.csv')
PROCESSED_XY = os.path.join(ROOT, 'final_test_results_cctbx', 'rruff', 'processed_xy')

# Load the 913 valid RRUFF IDs
valid_ids = set(pd.read_csv(FILTERED_CSV)['rruff_id'])
print(f'Filtering to {len(valid_ids)} RRUFF IDs from {FILTERED_CSV}')

def extract_rruff_id(pattern_id):
    m = re.search(r'__(R\d+(?:-\d+)?(?:-\w+)?)__', str(pattern_id))
    return m.group(1) if m else None

pkl_files = sorted(glob.glob(os.path.join(PROCESSED_XY, '**', 'inversion_results_cctbx.pkl'),
                             recursive=True))

for pkl_path in pkl_files:
    df = pd.read_pickle(pkl_path)
    mask = df['pattern_id'].apply(extract_rruff_id).isin(valid_ids)
    df_filtered = df[mask].reset_index(drop=True)

    out_path = pkl_path.replace('inversion_results_cctbx.pkl',
                                'inversion_results_cctbx_filtered.pkl')
    df_filtered.to_pickle(out_path)

    label = os.path.relpath(pkl_path, PROCESSED_XY)
    print(f'  {label}: {len(df_filtered)} / {len(df)} rows  →  {os.path.basename(out_path)}')

print('\nDone.')
