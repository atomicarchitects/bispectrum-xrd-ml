"""
Build eval pkl files from RRUFF XY_Processed patterns + rruff_alpha_diffract CIF files.

Outputs (saved to OUT_DIR):
    rruff_alpha_xrd.pkl      : dict {rruff_id -> np.array shape (8500,), intensity normalized 0-1}
    rruff_alpha_prim_lat.pkl : dict {rruff_id -> np.array shape (6,) [a,b,c,alpha,beta,gamma]}

Key format: 'Mineral__RID-n' (e.g. 'Actinolite__R040063-1')
"""

import os
import pickle
import numpy as np
from pymatgen.io.cif import CifParser

_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
XY_DIR  = f'{_ROOT}/rruff/XY_Processed'
CIF_DIR = f'{_ROOT}/rruff_alpha_diffract/rruff_cifs'
OUT_DIR = f'{_ROOT}/rruff_alpha_eval'

os.makedirs(OUT_DIR, exist_ok=True)


def prefix(fname):
    return '__'.join(os.path.basename(fname).split('__')[:2])


# Build lookup dicts: prefix -> full filename
xy_map  = {prefix(f): os.path.join(XY_DIR,  f) for f in os.listdir(XY_DIR)}
cif_map = {prefix(f): os.path.join(CIF_DIR, f) for f in os.listdir(CIF_DIR)}

common = sorted(set(xy_map) & set(cif_map))
print(f'XY files: {len(xy_map)}  CIF files: {len(cif_map)}  Matched: {len(common)}')

TRAIN_GRID = np.arange(5, 90, 0.010)  # 8500 points

xrd_dict = {}
lat_dict = {}
skipped  = []

for key in common:
    # --- XRD pattern ---
    try:
        lines = open(xy_map[key]).readlines()
        data_lines = [l for l in lines if not l.startswith('#') and ',' in l and l.strip()]
        xy = np.array([list(map(float, l.split(','))) for l in data_lines])
        two_theta, intensities = xy[:, 0], xy[:, 1]
        # Interpolate to training grid; zeros outside measured range
        intensities = np.interp(TRAIN_GRID, two_theta, intensities, left=0.0, right=0.0)
        mx = intensities.max()
        if mx > 0:
            intensities = intensities / mx
    except Exception as e:
        skipped.append((key, f'XY parse error: {e}'))
        continue

    # --- Lattice from CIF (primitive cell) ---
    try:
        parser = CifParser(cif_map[key], occupancy_tolerance=100)
        struct = parser.parse_structures(primitive=True)[0]
        lat    = struct.lattice
        params = np.array([lat.a, lat.b, lat.c, lat.alpha, lat.beta, lat.gamma])
    except Exception as e:
        skipped.append((key, f'CIF parse error: {e}'))
        continue

    xrd_dict[key] = intensities.astype(np.float32)
    lat_dict[key] = params.astype(np.float32)

print(f'Saved: {len(xrd_dict)}  Skipped: {len(skipped)}')
if skipped:
    for k, reason in skipped:
        print(f'  SKIP {k}: {reason}')

xrd_path = os.path.join(OUT_DIR, 'rruff_alpha_xrd.pkl')
lat_path  = os.path.join(OUT_DIR, 'rruff_alpha_prim_lat.pkl')
pickle.dump(xrd_dict, open(xrd_path, 'wb'))
pickle.dump(lat_dict, open(lat_path, 'wb'))
print(f'Written:\n  {xrd_path}\n  {lat_path}')
