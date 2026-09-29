"""
Fit the background-noise Gamma distribution by per-pattern MLE on opXRD CNRS
residuals, mirroring fit_intensity_gamma.py's RRUFF-based approach but for
BG_NOISE_LOGN_MEAN/COV (generate_augmented_data.py / bg_noise_aug.py) instead
of the caglioti4x intensity-perturbation Gamma.

For each CNRS pattern with a usable atomic basis (same filters as
prep_opxrd_cnrs.py --source raw: wavelength known, basis present, <=MAX_ATOMS
atoms):
  1. Rebuild the reported Structure and simulate a clean Caglioti pattern
     (Crystalyze U=0.05, V=-0.06, W=0.07), normalized the same way training
     data is.
  2. Build the observed pattern the same way prep_opxrd_cnrs.py does (CuKa
     conversion, interpolation onto the model grid, max-normalization).
  3. residual = observed - simulated, restricted to points where the
     simulated pattern is near zero (background-only region, away from any
     predicted peak).
  4. MLE-fit Gamma(a, scale) to the positive part of that residual for this
     one pattern (Gamma support is (0, inf), same convention as
     fit_intensity_gamma.py).
  5. Collect (log_a, log_scale) across all fitted patterns and fit a 2D
     multivariate normal -- the CNRS analog of BG_NOISE_LOGN_MEAN/COV.

Usage:
    python fit_bg_noise_cnrs.py \\
        --raw-dir $POWDERXRD_DATA_ROOT/opxrd_cnrs_eval/CNRS \\
        --out $POWDERXRD_DATA_ROOT/opxrd_cnrs_eval/bg_noise_cnrs.pkl
"""
import argparse
import json
import os
import pickle
import re
import sys

import numpy as np
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure
from scipy.stats import gamma as gamma_dist

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _THIS_DIR)
sys.path.insert(0, os.path.dirname(_THIS_DIR))
from prep_opxrd_cnrs import MAX_ATOMS, RAW_DIR as DEFAULT_RAW_DIR, build_xrd_array
from generate_augmented_data import CRYSTALYZE_U, CRYSTALYZE_V, CRYSTALYZE_W, simulate_pv_xrd

_ELEMENT_RE = re.compile(r'^[A-Za-z]+')
_CALC = XRDCalculator(wavelength='CuKa')


def _build_structure(lattice_str, basis_raw):
    a, b, c, alpha, beta, gamma = (float(x) for x in lattice_str.strip('()').split(','))
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    basis_list = json.loads(basis_raw) if isinstance(basis_raw, str) else basis_raw
    sites = [json.loads(s) if isinstance(s, str) else s for s in basis_list]
    species = [_ELEMENT_RE.match(s['symbol']).group(0) for s in sites]
    coords = [[s['x'], s['y'], s['z']] for s in sites]
    return Structure(lattice, species, coords)


def fit_one_pattern(fname, raw_dir, sim_threshold, min_bg_points):
    """Return (a, scale, residual_mean) for this pattern, or None if unusable."""
    with open(os.path.join(raw_dir, fname)) as f:
        d = json.load(f)

    label = json.loads(d['label'])
    phases = label.get('phases') or []
    phase0 = json.loads(phases[0]) if phases else {}
    lattice_str = phase0.get('lattice')
    basis_raw = phase0.get('basis')

    xray_info = label.get('xray_info')
    xray_info = json.loads(xray_info) if isinstance(xray_info, str) else (xray_info or {})
    wl = xray_info.get('primary_wavelength')

    if not lattice_str or not basis_raw or wl is None:
        return None

    basis_list = json.loads(basis_raw) if isinstance(basis_raw, str) else basis_raw
    if len(basis_list) > MAX_ATOMS:
        return None

    observed, _ = build_xrd_array(d['two_theta_values'], d['intensities'], float(wl))
    if observed is None:
        return None

    try:
        structure = _build_structure(lattice_str, basis_raw)
        pattern = _CALC.get_pattern(structure)
        simulated = simulate_pv_xrd(np.asarray(pattern.x), np.asarray(pattern.y),
                                     CRYSTALYZE_U, CRYSTALYZE_V, CRYSTALYZE_W)
    except Exception:
        return None

    bg_mask = simulated < sim_threshold
    if bg_mask.sum() < min_bg_points:
        return None
    residual = observed[bg_mask] - simulated[bg_mask]
    residual = residual[residual > 0]
    if len(residual) < min_bg_points // 2:
        return None

    try:
        a, _, scale = gamma_dist.fit(residual, floc=0)
    except Exception:
        return None
    if a <= 0 or scale <= 0:
        return None
    return a, scale, float(residual.mean())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--raw-dir', default=DEFAULT_RAW_DIR)
    p.add_argument('--out', default=os.environ.get("POWDERXRD_DATA_ROOT", "./data") + '/opxrd_cnrs_eval/bg_noise_cnrs.pkl')
    p.add_argument('--sim-threshold', type=float, default=0.01,
                    help='Points where the simulated pattern is below this are treated as background.')
    p.add_argument('--min-bg-points', type=int, default=200,
                    help='Minimum background-region points required to fit a pattern.')
    p.add_argument('--max-patterns', type=int, default=None,
                    help='Cap the number of patterns processed (for a quick test run).')
    args = p.parse_args()

    files = sorted(f for f in os.listdir(args.raw_dir) if f.endswith('.json'))
    if args.max_patterns is not None:
        files = files[:args.max_patterns]
    print(f'Total entries: {len(files)}')

    fitted = []
    for fname in files:
        result = fit_one_pattern(fname, args.raw_dir, args.sim_threshold, args.min_bg_points)
        if result is not None:
            fitted.append((fname, *result))

    print(f'Fitted {len(fitted)}/{len(files)} patterns')
    if len(fitted) < 10:
        sys.exit('Too few fitted patterns -- aborting.')

    a_arr = np.array([r[1] for r in fitted])
    scale_arr = np.array([r[2] for r in fitted])
    resid_means = np.array([r[3] for r in fitted])

    log_a = np.log(a_arr)
    log_scale = np.log(scale_arr)
    logn_mean = np.array([log_a.mean(), log_scale.mean()])
    logn_cov = np.cov(np.stack([log_a, log_scale]))

    print(f'\nCNRS background-noise Gamma fit (n={len(fitted)}):')
    print(f'  log_a:     mean={logn_mean[0]:.4f}  std={log_a.std():.4f}')
    print(f'  log_scale: mean={logn_mean[1]:.4f}  std={log_scale.std():.4f}')
    print(f"  mean 'a'    : {a_arr.mean():.4f}")
    print(f"  mean 'scale': {scale_arr.mean():.6f}")
    print(f'  mean noise level (a*scale): {(a_arr * scale_arr).mean():.4f}')
    print(f'  residual-mean percentiles: p5={np.percentile(resid_means, 5):.4f} '
          f'p50={np.percentile(resid_means, 50):.4f} p95={np.percentile(resid_means, 95):.4f}')

    out = {
        'logn_mean': logn_mean,
        'logn_cov': logn_cov,
        'n_patterns': len(fitted),
        'fit_args': {
            'sim_threshold': args.sim_threshold,
            'min_bg_points': args.min_bg_points,
            'caglioti_UVW': (CRYSTALYZE_U, CRYSTALYZE_V, CRYSTALYZE_W),
        },
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, 'wb') as f:
        pickle.dump(out, f)
    print(f'\nSaved to {args.out}')
    print(f'\nUse with bg_noise_aug.py by replacing:')
    print(f'  _LOGN_MEAN = np.array([{logn_mean[0]:.8f}, {logn_mean[1]:.8f}])')
    print(f'  _LOGN_COV  = np.array([[{logn_cov[0, 0]:.8f}, {logn_cov[0, 1]:.8f}],')
    print(f'                          [{logn_cov[1, 0]:.8f}, {logn_cov[1, 1]:.8f}]])')


if __name__ == '__main__':
    main()