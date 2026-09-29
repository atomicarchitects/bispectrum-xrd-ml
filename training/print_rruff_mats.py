import os

import h5py, numpy as np
from pymatgen.core import Lattice

def niggli_abcang(mat):
    try:
        n = Lattice(mat).get_niggli_reduced_lattice()
        abc = sorted(n.abc)
        return abc, list(n.angles)
    except:
        return None, None

_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
files = {
    'mpfull_noaug':        f'{_ROOT}/rruff_crystalyze/rruff_mpfull_noaug.pkl',
    'mpfull_orig_noise':   f'{_ROOT}/rruff_crystalyze/rruff_mpfull_orig_noise.h5',
    'mp20_aug':            f'{_ROOT}/rruff_crystalyze/rruff_mp20aug.h5',
    'mpfull_cag_nonoise':  f'{_ROOT}/rruff_crystalyze/rruff_mpfull_caglioti_nonoise.h5',
    'mpfull_aug':          f'{_ROOT}/rruff_crystalyze/rruff_mpfull_aug.pkl',
    'mp20_crystalyze_aug': f'{_ROOT}/rruff_crystalyze/rruff_mp20_crystalyze_aug.h5',
}

print(f"{'Model':<24} {'n':>4}  {'MAE_a':>6} {'MAE_b':>6} {'MAE_c':>6}  {'MAE_al':>7} {'MAE_be':>7} {'MAE_ga':>7}  {'<2Å+10°':>8}")
print('-'*95)

for label, f_path in files.items():
    print(label)
    import os
    if not os.path.exists(f_path):
        print(f"{label:<24}  NOT FOUND"); continue

    rows = []
    try:
        with h5py.File(f_path, 'r') as f:
            for mineral in f.keys():
                g = f[mineral]
                try:
                    t_abc, t_ang = niggli_abcang(np.array(g['true_lattice']))
                    p_abc, p_ang = niggli_abcang(np.array(g['pred']))
                    if t_abc and p_abc:
                        rows.append((t_abc, p_abc, t_ang, p_ang))
                        print(mineral)
                        print((t_abc,p_abc,t_ang,p_ang))
                except: pass
    except:
        pass
