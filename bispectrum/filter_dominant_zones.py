"""
Identify dominant-zone powder patterns using simulated XRD.

A dominant zone occurs when most diffracted intensity can be indexed using
only two Miller indices (one index is consistently zero), e.g. h0l patterns.
Detected by computing simulated Cu-Kα powder patterns and measuring what
fraction of total intensity falls in each zero-index zone (h=0, k=0, l=0).

Usage:
    python filter_dominant_zones.py [--dataset mpfull] [--threshold 0.5]
"""

import argparse
import os
import pickle
import numpy as np
import pandas as pd
from tqdm import tqdm
from pymatgen.core import Structure
from pymatgen.analysis.diffraction.xrd import XRDCalculator

TWO_THETA_RANGE = (5.0, 70.0)
WAVELENGTH = 'CuKa'


def zone_intensity_fractions(struct, calc, two_theta_range=TWO_THETA_RANGE):
    """
    Compute the fraction of total XRD intensity falling in each zero-index zone.

    A peak at a given 2θ is counted in the h=0 zone only if ALL contributing
    hkl reflections have h=0 (similarly for k=0 and l=0). Peaks with mixed
    contributors (accidental overlap) are excluded from all zone counts.

    Returns a dict with frac_h0, frac_k0, frac_l0, dominant_zone_frac,
    dominant_zero_index, and n_peaks, or None on failure.
    """
    try:
        pattern = calc.get_pattern(struct, two_theta_range=two_theta_range)
    except Exception:
        return None

    if len(pattern.y) == 0 or sum(pattern.y) == 0:
        return None

    total = sum(pattern.y)
    zone_intensity = [0.0, 0.0, 0.0]  # h=0, k=0, l=0

    for y, hkls in zip(pattern.y, pattern.hkls):
        for idx in range(3):
            if all(hkl['hkl'][idx] == 0 for hkl in hkls):
                zone_intensity[idx] += y

    fracs = [z / total for z in zone_intensity]
    best = int(np.argmax(fracs))
    labels = ['h', 'k', 'l']

    return {
        'frac_h0':             fracs[0],
        'frac_k0':             fracs[1],
        'frac_l0':             fracs[2],
        'dominant_zone_frac':  fracs[best],
        'dominant_zero_index': labels[best],
        'n_peaks':             len(pattern.y),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='mpfull',
                        help='Dataset name under $POWDERXRD_DATA_ROOT (default: mpfull)')
    parser.add_argument('--threshold', type=float, default=0.9,
                        help='Min fraction of intensity in one zone to flag as dominant (default: 0.9)')
    parser.add_argument('--two-theta-max', type=float, default=70.0,
                        help='Maximum 2θ for XRD pattern in degrees (default: 70)')
    args = parser.parse_args()

    data_dir = os.path.join(os.environ.get('POWDERXRD_DATA_ROOT', './data'), args.dataset)
    two_theta_range = (5.0, args.two_theta_max)

    df = pd.read_csv(os.path.join(data_dir, 'test.csv'))
    print(f'Loaded {len(df)} structures from {data_dir}/test.csv')
    print(f'Using wavelength={WAVELENGTH}, 2θ range={two_theta_range}, threshold={args.threshold}')

    calc = XRDCalculator(wavelength=WAVELENGTH)

    rows = []
    n_failed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc='Computing XRD patterns'):
        entry = {'material_id': row['material_id']}
        try:
            struct = Structure.from_str(row['cif'], fmt='cif')
            result = zone_intensity_fractions(struct, calc, two_theta_range)
            if result is None:
                raise ValueError('empty pattern')
            entry.update(result)
        except Exception as e:
            n_failed += 1
            entry.update({
                'frac_h0': np.nan, 'frac_k0': np.nan, 'frac_l0': np.nan,
                'dominant_zone_frac': np.nan, 'dominant_zero_index': None,
                'n_peaks': 0,
            })
        rows.append(entry)

    out = pd.DataFrame(rows)
    if n_failed:
        print(f'Warning: {n_failed} structures failed and are excluded from filtering')

    # Merge in Bravais type
    bravais_path = os.path.join(data_dir, 'test_true_bravais.pkl')
    if os.path.exists(bravais_path):
        with open(bravais_path, 'rb') as f:
            df_bravais = pickle.load(f)
        out = out.merge(df_bravais[['material_id', 'true_bravais']], on='material_id', how='left')

    # Merge in theoretical flag
    if 'theoretical' in df.columns:
        out = out.merge(df[['material_id', 'theoretical']], on='material_id', how='left')

    # Filter dominant zones
    dominant = out[out['dominant_zone_frac'] >= args.threshold].copy()
    dominant = dominant.sort_values('dominant_zone_frac', ascending=False)

    print(f'\nTotal structures: {len(out)}')
    print(f'Dominant zone (≥{args.threshold:.0%} intensity in one zone): {len(dominant)}')
    print(f'\nBreakdown by dominant zero index:')
    print(dominant['dominant_zero_index'].value_counts().to_string())
    if 'true_bravais' in dominant.columns:
        print(f'\nBravais type breakdown:')
        print(dominant['true_bravais'].value_counts().to_string())
    if 'theoretical' in dominant.columns:
        n_exp = (dominant['theoretical'] == False).sum()
        print(f'\nExperimentally observed: {n_exp} / {len(dominant)}')

    cols = ['material_id', 'frac_h0', 'frac_k0', 'frac_l0',
            'dominant_zone_frac', 'dominant_zero_index', 'n_peaks']
    if 'true_bravais' in out.columns:
        cols.append('true_bravais')
    if 'theoretical' in out.columns:
        cols.append('theoretical')

    all_path = os.path.join(data_dir, 'dominant_zones.csv')
    out[cols].to_csv(all_path, index=False)
    print(f'\nSaved all {len(out)} entries to {all_path}')

    dom_path = os.path.join(data_dir, 'dominant_zones_filtered.csv')
    dominant[cols].to_csv(dom_path, index=False)
    print(f'Saved {len(dominant)} dominant-zone entries to {dom_path}')


if __name__ == '__main__':
    main()
