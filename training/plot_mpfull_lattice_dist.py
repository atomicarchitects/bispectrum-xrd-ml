"""
Plot the c-axis (and a, b) distribution of the mpfull training set
vs the RRUFF true lattice parameters.

Usage:
    python plot_mpfull_lattice_dist.py
"""
import os
import pickle, numpy as np
import matplotlib.pyplot as plt
import h5py
from pymatgen.core import Lattice

# ── mpfull: load primitive structures and extract lattice params ────────────
_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
MP_STRUCT_FILE = f'{_ROOT}/materials_project_structures.pkl'
RRUFF_H5       = f'{_ROOT}/rruff_crystalyze/rruff_mpfull_noaug.pkl'
OUT_PNG        = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mpfull_lattice_dist.png')

print('Loading mpfull structures...')
with open(MP_STRUCT_FILE, 'rb') as f:
    data = pickle.load(f)
print(f'  {len(data)} structures')

mp_a, mp_b, mp_c = [], [], []
for mpid, struct in data.items():
    try:
        prim = struct.get_primitive_structure()
        abc  = sorted(prim.lattice.abc)   # sort so a<=b<=c
        mp_a.append(abc[0]); mp_b.append(abc[1]); mp_c.append(abc[2])
    except:
        pass
mp_a = np.array(mp_a); mp_b = np.array(mp_b); mp_c = np.array(mp_c)
print(f'  Extracted {len(mp_c)} primitive lattices')

# ── RRUFF true lattice params from h5 ─────────────────────────────────────
rr_a, rr_b, rr_c = [], [], []
with h5py.File(RRUFF_H5, 'r') as f:
    for mineral in f.keys():
        try:
            mat = np.array(f[mineral]['true_lattice'])
            n   = Lattice(mat).get_niggli_reduced_lattice()
            abc = sorted(n.abc)
            rr_a.append(abc[0]); rr_b.append(abc[1]); rr_c.append(abc[2])
        except:
            pass
rr_a = np.array(rr_a); rr_b = np.array(rr_b); rr_c = np.array(rr_c)
print(f'  {len(rr_c)} RRUFF true lattices')

# ── print summary stats ────────────────────────────────────────────────────
print(f'\n{"":10} {"mean":>7} {"median":>8} {"p75":>7} {"p90":>7} {"p99":>7} {">10Å":>7}')
print('mpfull:')
for name, arr in [('a', mp_a), ('b', mp_b), ('c', mp_c)]:
    print(f'  {name:<8} {np.mean(arr):>7.2f} {np.median(arr):>8.2f} '
          f'{np.percentile(arr,75):>7.2f} {np.percentile(arr,90):>7.2f} '
          f'{np.percentile(arr,99):>7.2f} {100*np.mean(arr>10):>6.1f}%')
print('RRUFF:')
for name, arr in [('a', rr_a), ('b', rr_b), ('c', rr_c)]:
    print(f'  {name:<8} {np.mean(arr):>7.2f} {np.median(arr):>8.2f} '
          f'{np.percentile(arr,75):>7.2f} {np.percentile(arr,90):>7.2f} '
          f'{np.percentile(arr,99):>7.2f} {100*np.mean(arr>10):>6.1f}%')

# ── plot ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(14, 4))
bins = np.linspace(0, 25, 80)

for ax, name, mp_arr, rr_arr in zip(axes,
        ['a (shortest)', 'b (middle)', 'c (longest)'],
        [mp_a, mp_b, mp_c],
        [rr_a, rr_b, rr_c]):
    ax.hist(mp_arr, bins=bins, density=True, alpha=0.6, label='mpfull train', color='steelblue')
    ax.hist(rr_arr, bins=bins, density=True, alpha=0.7, label='RRUFF true',   color='tomato')
    ax.axvline(np.median(mp_arr), color='steelblue', lw=1.5, ls='--')
    ax.axvline(np.median(rr_arr), color='tomato',    lw=1.5, ls='--')
    ax.set_xlabel(f'{name} (Å)')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)
    ax.set_xlim(0, 25)

fig.suptitle('Niggli-sorted lattice parameter distributions: mpfull training vs RRUFF', y=1.01)
plt.tight_layout()
plt.savefig(OUT_PNG, dpi=150, bbox_inches='tight')
print(f'\nSaved to {OUT_PNG}')
