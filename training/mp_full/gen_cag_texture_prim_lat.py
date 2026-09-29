"""
Generate {split}_prim_lat.pkl for the {dataset}_cag and {dataset}_texture_cag
variants of a dataset (mp20 or mpfull) by duplicating each material's
primitive lattice once per key variant, matching the key format already
used in those directories.

{dataset}_cag keys:     {mpid}_0  ..  {mpid}_3
{dataset}_texture keys: {mpid}_texture_1 .. {mpid}_texture_3

Usage:
    python gen_cag_texture_prim_lat.py --dataset mp20
    python gen_cag_texture_prim_lat.py --dataset mpfull
    python gen_cag_texture_prim_lat.py --dataset mpfull --splits val test
"""

import argparse
import os
import pickle

BASE = os.environ.get("POWDERXRD_DATA_ROOT", "./data")

# mp20's base lattice pickles live under mp20/data/, mpfull's live directly
# under mpfull/ -- the one structural difference between the two datasets.
LAT_SUBDIR = {
    'mp20': 'mp20/data',
    'mpfull': 'mpfull',
}


def load(path):
    with open(path, 'rb') as f:
        return pickle.load(f)


def save(obj, path):
    with open(path, 'wb') as f:
        pickle.dump(obj, f)
    print(f'  saved {len(obj)} entries -> {path}')


def expand(xrd_dict, lat, suffix_sep, label):
    out, missing = {}, []
    for k in xrd_dict:
        base = k.rsplit(suffix_sep, 1)[0]
        if base in lat:
            out[k] = lat[base]
        else:
            missing.append(k)
    if missing:
        print(f'  WARNING: {len(missing)} {label} keys had no lattice match, e.g. {missing[:3]}')
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--dataset', required=True, choices=['mp20', 'mpfull'])
    p.add_argument('--splits', nargs='+', default=['train', 'val', 'test'])
    args = p.parse_args()

    lat_dir = f'{BASE}/{LAT_SUBDIR[args.dataset]}'
    cag_dir = f'{BASE}/{args.dataset}_cag'
    tex_dir = f'{BASE}/{args.dataset}_texture_cag'

    for split in args.splits:
        print(f'\n=== {args.dataset} / {split} ===')
        lat = load(f'{lat_dir}/{split}_data_prim_lat.pkl')

        cag_xrd = load(f'{cag_dir}/{split}_sim_xrd_caglioti.pkl')
        cag_lat = expand(cag_xrd, lat, '_', 'cag')
        save(cag_lat, f'{cag_dir}/{split}_prim_lat.pkl')

        tex_xrd = load(f'{tex_dir}/{split}_sim_xrd_texture.pkl')
        tex_lat = expand(tex_xrd, lat, '_texture_', 'texture')
        save(tex_lat, f'{tex_dir}/{split}_prim_lat.pkl')

    print('\nDone.')


if __name__ == '__main__':
    main()
