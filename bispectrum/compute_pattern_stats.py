"""
Compute quantitative XRD pattern statistics for dominant zone structures.

For each structure computes:
- Rwp: weighted profile R-factor on Gaussian-broadened profile
- Peak match rate: fraction of true peaks with a predicted peak within ±0.2° 2θ
- Mean 2θ shift: average positional error of matched peaks

Inputs:
  $POWDERXRD_DATA_ROOT/mpfull/dominant_zones_prediction_quality.csv

Output:
  $POWDERXRD_DATA_ROOT/mpfull/dominant_zones_pattern_stats.csv
"""

import os
import sys
sys.path.append("..")

import numpy as np
import pandas as pd
from pymatgen.core import Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from tqdm import tqdm
from utilities import struct_from_lat

# --- Settings ---
WAVELENGTH     = 1.5406
TWO_THETA_RANGE = (5, 60)
SIGMA          = 0.15   # peak broadening (degrees)
N_PTS          = 2000   # profile grid points
MATCH_TOL      = 0.2    # 2theta tolerance for peak matching (degrees)

DATA_DIR = os.path.join(os.environ.get("POWDERXRD_DATA_ROOT", "./data"), "mpfull")

# --- Load ---
df = pd.read_csv(f"{DATA_DIR}/dominant_zones_prediction_quality.csv")
print(f"Loaded {len(df)} structures")

quadrant_order = ['both_good', 'bispec_good_direct_bad', 'bispec_bad_direct_good', 'both_bad']

xrd_calc = XRDCalculator(wavelength=WAVELENGTH)
x_grid = np.linspace(TWO_THETA_RANGE[0], TWO_THETA_RANGE[1], N_PTS)


def get_pattern(a, b, c, alpha, beta, gamma):
    struct = struct_from_lat(Lattice.from_parameters(a, b, c, alpha, beta, gamma).matrix)
    try:
        return xrd_calc.get_pattern(struct, two_theta_range=TWO_THETA_RANGE)
    except (ValueError, Exception):
        return None


def to_profile(pattern, x_grid):
    y = np.zeros_like(x_grid)
    for tt, I in zip(pattern.x, pattern.y):
        y += (I / 100.0) * np.exp(-0.5 * ((x_grid - tt) / SIGMA) ** 2)
    return y


def rwp(y_true, y_pred):
    num = np.sum((y_true - y_pred) ** 2)
    den = np.sum(y_true ** 2)
    return np.sqrt(num / den)


def m20(true_tth, pred_tth, wavelength=WAVELENGTH):
    """
    de Wolff M20 figure of merit.
    Treats true_tth as 'observed' and pred_tth as 'calculated'.
    M20 = Q20 / (2 * eps_mean * N20)
      Q20      = Q of the 20th observed line
      eps_mean = mean |Q_obs - Q_calc| for first 20 observed lines (matched to nearest calc)
      N20      = number of calculated lines up to the 2theta of the 20th observed line
    Q = (2 sin theta / lambda)^2  [Angstrom^-2]
    """
    def to_Q(tth_deg):
        return (2 * np.sin(np.deg2rad(tth_deg / 2)) / wavelength) ** 2

    obs = np.sort(true_tth)
    calc = np.sort(pred_tth)
    n_obs = min(20, len(obs))
    if n_obs == 0 or len(calc) == 0:
        return np.nan

    obs20 = obs[:n_obs]
    Q_obs = to_Q(obs20)
    Q20 = Q_obs[-1]

    # match each observed line to nearest calculated Q
    Q_calc = to_Q(calc)
    Q_calc_unique = np.unique(np.round(Q_calc, decimals=8))
    N20 = np.sum(Q_calc_unique <= Q20)
    eps = []
    for qo in Q_obs:
        nearest = Q_calc[np.argmin(np.abs(Q_calc - qo))]
        eps.append(abs(qo - nearest))
    eps_mean = np.mean(eps)
    if eps_mean == 0:
        return np.inf

    # N20: number of calc lines with 2theta <= 20th observed 2theta
    N20 = np.sum(calc <= obs20[-1])
    if N20 == 0:
        return np.nan

    return Q20 / (2 * eps_mean * N20)


def peak_match_stats(true_peaks, pred_peaks, tol=MATCH_TOL):
    matched_shifts = []
    for tt_true in true_peaks:
        dists = np.abs(np.array(pred_peaks) - tt_true)
        if len(dists) > 0 and dists.min() <= tol:
            matched_shifts.append(dists.min())
    match_rate = len(matched_shifts) / len(true_peaks) if len(true_peaks) > 0 else np.nan
    mean_shift = np.mean(matched_shifts) if matched_shifts else np.nan
    return match_rate, mean_shift


# --- Compute ---
stats_rows = []
n_skipped = 0
for _, row in tqdm(df.iterrows(), total=len(df), desc="Computing pattern stats"):
    true_pat   = get_pattern(row['true_a'],        row['true_b'],        row['true_c'],
                             row['true_alpha'],     row['true_beta'],     row['true_gamma'])
    bispec_pat = get_pattern(row['bispec_pred_a'],  row['bispec_pred_b'], row['bispec_pred_c'],
                             row['bispec_pred_alpha'], row['bispec_pred_beta'], row['bispec_pred_gamma'])
    direct_pat = get_pattern(row['direct_pred_a'],  row['direct_pred_b'], row['direct_pred_c'],
                             row['direct_pred_alpha'], row['direct_pred_beta'], row['direct_pred_gamma'])

    if true_pat is None or bispec_pat is None or direct_pat is None:
        n_skipped += 1
        continue

    y_true   = to_profile(true_pat,   x_grid)
    y_bispec = to_profile(bispec_pat, x_grid)
    y_direct = to_profile(direct_pat, x_grid)

    mr_b, ms_b = peak_match_stats(true_pat.x, bispec_pat.x)
    mr_d, ms_d = peak_match_stats(true_pat.x, direct_pat.x)

    stats_rows.append({
        'mat_id':            row['mat_id'],
        'true_bravais':      row['true_bravais'],
        'quadrant':          row.get('quadrant', None),
        'anisotropy':        row['anisotropy'],
        'bispec_rwp':        rwp(y_true, y_bispec),
        'direct_rwp':        rwp(y_true, y_direct),
        'bispec_m20':        m20(true_pat.x, bispec_pat.x),
        'direct_m20':        m20(true_pat.x, direct_pat.x),
        'bispec_match_rate': mr_b,
        'direct_match_rate': mr_d,
        'bispec_mean_shift': ms_b,
        'direct_mean_shift': ms_d,
        'bispec_len_mape':   row['bispec_len_mape'],
        'direct_len_mape':   row['direct_len_mape'],
    })

df_stats = pd.DataFrame(stats_rows)

out_path = f"{DATA_DIR}/dominant_zones_pattern_stats.csv"
df_stats.to_csv(out_path, index=False)
print(f"\nSaved {len(df_stats)} rows to {out_path}  (skipped {n_skipped} with empty patterns)")

# --- Overall summary ---
print(f"\n{'Metric':<25} {'Bispec':>10} {'Direct':>10}")
print("-" * 47)
for metric, b_col, d_col in [
    ('Rwp (median)',       'bispec_rwp',        'direct_rwp'),
    ('M20 (median)',       'bispec_m20',        'direct_m20'),
    ('Peak match rate',    'bispec_match_rate', 'direct_match_rate'),
    ('Mean 2θ shift (°)',  'bispec_mean_shift', 'direct_mean_shift'),
]:
    print(f"{metric:<25} {df_stats[b_col].median():>10.3f} {df_stats[d_col].median():>10.3f}")

# --- Per quadrant ---
if 'quadrant' in df_stats.columns and df_stats['quadrant'].notna().any():
    print(f"\nPer quadrant — median Rwp:")
    print(f"{'Quadrant':<30} {'N':>5} {'Bispec Rwp':>12} {'Direct Rwp':>12}")
    print("-" * 62)
    for q in quadrant_order:
        sub = df_stats[df_stats['quadrant'] == q]
        if len(sub) == 0:
            continue
        print(f"{q:<30} {len(sub):>5} {sub['bispec_rwp'].median():>12.3f} {sub['direct_rwp'].median():>12.3f}")
