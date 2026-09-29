"""
Filter materials_project_lat_bispec.h5 to only the mp20 material IDs,
producing a smaller HDF5 suitable for mp20-trained model inversion.

Usage:
    python make_mp20_bispec_h5.py
"""
import os

import h5py
import numpy as np
import pandas as pd

_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
IN_H5   = f'{_ROOT}/materials_project_lat_bispec.h5'
MP20_CSV = f'{_ROOT}/mp20/data/all_mp20.csv'
OUT_H5  = f'{_ROOT}/mp20_lat_bispec.h5'

# Load mp20 material IDs
df = pd.read_csv(MP20_CSV)
mp20_ids = set(df['material_id'].astype(str).tolist())
print(f'mp20 total IDs: {len(mp20_ids)}')

# Load mpfull h5
print('Loading mpfull h5...')
with h5py.File(IN_H5, 'r') as f:
    all_ids    = f['mp_ids'][:].astype(str)
    all_bispec = f['bispec'][:]         # (N, 10, 35)
    all_recip  = f['recip_lattice'][:]  # (N, 9)

print(f'mpfull DB size: {len(all_ids)}')

# Find indices of mp20 IDs in mpfull
mask = np.array([mid in mp20_ids for mid in all_ids])
n_found = mask.sum()
print(f'mp20 IDs found in mpfull DB: {n_found} / {len(mp20_ids)}')

mp20_ids_out    = all_ids[mask]
mp20_bispec_out = all_bispec[mask]
mp20_recip_out  = all_recip[mask]

# Write filtered h5
print(f'Writing to {OUT_H5} ...')
with h5py.File(OUT_H5, 'w') as f:
    f.create_dataset('mp_ids',        data=mp20_ids_out.astype('S'),
                     compression='gzip')
    f.create_dataset('bispec',        data=mp20_bispec_out,
                     compression='gzip', chunks=(1000, *mp20_bispec_out.shape[1:]))
    f.create_dataset('recip_lattice', data=mp20_recip_out,
                     compression='gzip', chunks=(1000, *mp20_recip_out.shape[1:]))

print(f'Done. Saved {n_found} structures to {OUT_H5}')
