"""
Compute true bispectra for the 148 RRUFF minerals from their CIF structures
(primitive cell), and compare cosine similarity against model predictions.

Output: true_bispecs_148.pkl  ->  {mineral_name__RRRUFF_id: np.array (10, 35)}

Usage:
    python compute_rruff_true_bispec.py
"""
import os, sys, pickle
import numpy as np
import torch
import pandas as pd
from io import StringIO

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pymatgen.io.cif import CifParser
import e3nn_jax
from e3nn import o3, io
from utilities import get_bispectrum, get_neighbors

# ── paths ──────────────────────────────────────────────────────────────────
_ROOT = os.environ.get('POWDERXRD_DATA_ROOT', "./data")
RRUFF_DIR   = f'{_ROOT}/rruff_crystalyze'
TEST_CSV    = f'{RRUFF_DIR}/test.csv'
REF_PKL     = f'{_ROOT}/eval_results/rruff/rruff_crystalyze_results/rruff_mpfull_bispec.pkl'
OUT_PATH    = f'{RRUFF_DIR}/true_bispecs_148.pkl'

PRED_PKLS = {
    'mp20_cag_aug':       'rruff_mp20_crystalyze_aug_bispec.pkl',
    'mpfull_bispec':      'rruff_mpfull_bispec.pkl',
    'mpfull_orig':        'rruff_mpfull_orig_bispec.pkl',
    'mpfull_aug':         'rruff_mpfull_aug_bispec.pkl',
    'mpfull_aug_penalty': 'rruff_mpfull_caglioti_nonoise_bispec.pkl',
    'mpfull_cag_noise':   'rruff_mpfull_caglioti_noise_bispec.pkl',
    'mpfull_cag_nonoise': 'rruff_mpfull_caglioti_nonoise_bispec.pkl',
    'mpfull_crystalyze':  'rruff_mpfull_crystalyze_bispec.pkl',
}
PRED_BASE = f'{_ROOT}/eval_results/rruff/rruff_crystalyze_results/'

L_MAX  = 6
K_MAX  = 2 / 3
N_BASIS = 10

# ── build cob_bi once ──────────────────────────────────────────────────────
print('Building cob_bi ...')
rtp_bi = e3nn_jax.reduced_symmetric_tensor_product_basis(
    io.SphericalTensor(L_MAX, 1, -1), 3, keep_ir=['0o', '0e']
)
cob_bi = torch.tensor(rtp_bi.array, dtype=torch.float64)
print(f'  cob_bi shape: {cob_bi.shape}')

# ── load mineral ordering from reference pkl ───────────────────────────────
with open(REF_PKL, 'rb') as f:
    ref = pickle.load(f)
mineral_names = list(ref['mineral_names'])   # e.g. 'Ferberite__R040044'
print(f'Mineral names: {len(mineral_names)}')

# ── load CIFs from test.csv ────────────────────────────────────────────────
df = pd.read_csv(TEST_CSV)
df_lookup = {row['material_id']: row['cif'] for _, row in df.iterrows()}

# ── compute true bispectra ─────────────────────────────────────────────────
true_bispecs = {}
failed = []

for mn in mineral_names:
    mineral = mn.split('__')[0]
    if mineral not in df_lookup:
        print(f'  [MISS] {mn} not in test.csv')
        failed.append(mn)
        continue
    try:
        parser = CifParser(StringIO(df_lookup[mineral]))
        struct = parser.get_structures()[0]
        prim   = struct.get_primitive_structure()
        recip  = prim.lattice.reciprocal_lattice_crystallographic.matrix
        miller = get_neighbors(recip, float(K_MAX),
                               filter_orig=True, filter_new=False,
                               miller_ind_threshold=3)
        neighs = torch.tensor(miller.reshape(-1, 3) @ recip, dtype=torch.float64)
        bispec = get_bispectrum(neighs, float(K_MAX), lmax=L_MAX,
                                radial_function='bessel', n_basis=N_BASIS,
                                cob_bi=cob_bi)
        true_bispecs[mn] = bispec.numpy()
    except Exception as e:
        print(f'  [FAIL] {mn}: {e}')
        failed.append(mn)

print(f'\nComputed {len(true_bispecs)}/148 true bispectra ({len(failed)} failed)')
if failed:
    print('Failed:', failed)

with open(OUT_PATH, 'wb') as f:
    pickle.dump(true_bispecs, f)
print(f'Saved to {OUT_PATH}')
print(f'Bispec shape: {next(iter(true_bispecs.values())).shape}')

# ── cosine similarity vs model predictions ─────────────────────────────────
def cos_sim_flat(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))

print('\n── Cosine similarity (pred vs true bispectrum) ──')
print(f'{"Model":<22} {"mean":>8} {"median":>8} {"min":>8}  n')
print('-' * 55)

for name, fname in PRED_PKLS.items():
    path = PRED_BASE + fname
    if not os.path.exists(path):
        print(f'{name:<22}  file not found')
        continue
    with open(path, 'rb') as f:
        d = pickle.load(f)
    preds        = np.array(d['predictions'])        # (148, 10, 35)
    pred_names   = list(d['mineral_names'])

    sims = []
    for i, mn in enumerate(pred_names):
        if mn not in true_bispecs:
            continue
        sim = cos_sim_flat(preds[i], true_bispecs[mn])
        sims.append(sim)

    if sims:
        print(f'{name:<22} {np.mean(sims):>8.4f} {np.median(sims):>8.4f} {np.min(sims):>8.4f}  {len(sims)}')
    else:
        print(f'{name:<22}  no overlap')
