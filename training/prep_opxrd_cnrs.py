"""
Preprocess the opxrd CNRS dataset for model evaluation.

Two source formats are supported:

  aggregated (default) - a collaborator-built single JSON with parallel
      dicts keyed by pattern id (data['theta2'][k], data['I'][k],
      data['primary_wavelength'][k], data['unit_cell'][k] with angles in
      radians, data['file_name'][k]).

  raw - a directory of per-pattern opXRD JSON files (e.g. straight from an
      opXRD dataset download), each shaped like:
      {"two_theta_values": [...], "intensities": [...],
       "label": '{"phases": ["{\\"lattice\\": \\"(a, b, c, alpha, beta, gamma)\\",
                                \\"basis\\": \\"[...atom sites...]\\", ...}"],
                   "xray_info": "{\\"primary_wavelength\\": \\"1.5406\\", ...}", ...}',
       "metadata": "{...}"}
      Lattice angles here are already in degrees. The reported cell is not
      guaranteed to be primitive or in a consistent setting (rhombohedral vs.
      hexagonal axes, unreduced vs. reduced, etc), so the raw path rebuilds a
      full Structure from the reported lattice + atomic basis and calls
      get_primitive_structure() on it -- the same primitive-cell convention
      used for MP training targets (mp_full/generate_augmented_data.py) and
      RRUFF alpha eval targets (mp_full/gen_rruff_alpha_eval.py). Patterns
      with no atomic basis in the source data are skipped rather than stored
      with an unreduced/inconsistent cell.

Both paths convert all patterns to CuKa-equivalent 2theta (using Bragg's law
for non-CuKa wavelengths), interpolate onto the model's fixed grid (5-90
deg, 0.01 deg step), normalize, and save:

  opxrd_cnrs_xrd.pkl : dict {pattern_id -> np.float32 array (8500,)}
  opxrd_cnrs_lat.pkl : dict {pattern_id -> np.float32 array [a,b,c,alpha,beta,gamma]}

pattern ids are 'CNRS_' + the source pattern's base filename (e.g.
'CNRS_pattern_0'), so ids line up across both source formats when they
derive from the same underlying opXRD dump.
"""

import argparse
import json
import os
import pickle
import re

import numpy as np
from pymatgen.core import Lattice, Structure
from scipy.interpolate import interp1d

_ELEMENT_RE = re.compile(r'^[A-Za-z]+')

# aggregated source: a single JSON of pre-curated CNRS patterns. It is an external input with no
# default: set CNRS_JSON_PATH or pass --json-path, or use --source raw instead (see module docstring).
JSON_PATH = os.environ.get('CNRS_JSON_PATH')
OUT_DIR   = os.environ.get("POWDERXRD_DATA_ROOT", "./data") + "/opxrd_cnrs_eval"
RAW_DIR   = os.path.join(OUT_DIR, 'CNRS')

TWO_THETA_GRID = np.arange(5, 90, 0.010)  # 8500 points, matches model input grid
LAMBDA_CUKA    = 1.5406                    # Angstrom
MAX_ATOMS      = 500                       # matches mp_full/filter_alpha_diffract.py's MP training cutoff;
                                            # get_primitive_structure()'s translation search is impractically
                                            # slow well beyond this (a few CNRS patterns report 10k+ atom bases)


def to_cuka_twotheta(theta2_deg, wavelength):
    """Convert 2theta values from a given wavelength to CuKa-equivalent."""
    theta_rad = np.radians(theta2_deg / 2.0)
    sin_val   = np.sin(theta_rad) * wavelength / LAMBDA_CUKA
    # Peaks beyond 90 deg in CuKa space are outside the model range — drop them
    valid = sin_val <= 1.0
    theta2_cuka = np.full_like(theta2_deg, np.nan)
    theta2_cuka[valid] = np.degrees(2.0 * np.arcsin(sin_val[valid]))
    return theta2_cuka


def build_xrd_array(theta2, intensity, wavelength):
    """Sort, CuKa-convert if needed, interpolate onto TWO_THETA_GRID, normalize.

    Returns (xrd float32 array or None if unusable, was_converted bool).
    """
    theta2    = np.asarray(theta2, dtype=np.float64)
    intensity = np.asarray(intensity, dtype=np.float64)
    order = np.argsort(theta2)
    theta2, intensity = theta2[order], intensity[order]

    converted = abs(wavelength - LAMBDA_CUKA) > 0.01
    if converted:
        theta2_conv = to_cuka_twotheta(theta2, wavelength)
        valid = ~np.isnan(theta2_conv)
        theta2, intensity = theta2_conv[valid], intensity[valid]

    if len(theta2) < 2:
        return None, converted

    interp = interp1d(theta2, intensity, kind='linear', bounds_error=False, fill_value=0.0)
    xrd = interp(TWO_THETA_GRID)

    max_val = xrd.max()
    if max_val <= 0:
        return None, converted
    return (xrd / max_val).astype(np.float32), converted


def load_aggregated(json_path):
    """Return (entries, n_skipped) from the collaborator-built aggregated JSON,
    where entries is a list of (pattern_id, theta2, intensity, wavelength, lattice_degrees)
    and n_skipped counts entries dropped here for missing wavelength."""
    print(f'Loading {json_path}...')
    with open(json_path) as f:
        data = json.load(f)

    keys = list(data['theta2'].keys())
    print(f'Total entries: {len(keys)}')

    entries, skipped = [], 0
    for k in keys:
        wl = data['primary_wavelength'][k]
        if wl is None:
            skipped += 1
            continue
        pid = 'CNRS_' + os.path.basename(data['file_name'][k]).replace('.json', '')
        uc = data['unit_cell'][k]
        lattice = np.array(
            [float(uc[0]), float(uc[1]), float(uc[2]),
             float(np.degrees(uc[3])), float(np.degrees(uc[4])), float(np.degrees(uc[5]))],
            dtype=np.float32,
        )
        entries.append((pid, data['theta2'][k], data['I'][k], float(wl), lattice))
    return entries, skipped


def primitive_lattice_from_raw_phase(lattice_str, basis_raw):
    """Build the reported unit cell + atomic basis into a pymatgen Structure and
    reduce it with get_primitive_structure(), matching the primitive-cell
    convention used for MP training targets (generate_augmented_data.py) and
    RRUFF alpha eval targets (gen_rruff_alpha_eval.py), so opXRD CNRS eval
    targets are in the same setting the model was trained to predict."""
    a, b, c, alpha, beta, gamma = (float(x) for x in lattice_str.strip('()').split(','))
    lattice = Lattice.from_parameters(a, b, c, alpha, beta, gamma)

    basis_list = json.loads(basis_raw) if isinstance(basis_raw, str) else basis_raw
    sites = [json.loads(s) if isinstance(s, str) else s for s in basis_list]
    species = [_ELEMENT_RE.match(s['symbol']).group(0) for s in sites]
    coords = [[s['x'], s['y'], s['z']] for s in sites]

    structure = Structure(lattice, species, coords)
    prim = structure.get_primitive_structure()
    pa, pb, pc = prim.lattice.abc
    palpha, pbeta, pgamma = prim.lattice.angles
    return np.array([pa, pb, pc, palpha, pbeta, pgamma], dtype=np.float32)


def load_raw_dir(raw_dir):
    """Return (entries, n_skipped) from a directory of per-pattern raw opXRD JSON
    files, where entries is a list of (pattern_id, theta2, intensity, wavelength,
    primitive lattice_degrees) and n_skipped counts entries dropped here for
    missing phase/lattice/basis/wavelength or a failed primitive reduction."""
    files = sorted(f for f in os.listdir(raw_dir) if f.endswith('.json'))
    print(f'Total entries: {len(files)}')

    entries, skipped = [], 0
    for fname in files:
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
            skipped += 1
            continue

        basis_list = json.loads(basis_raw) if isinstance(basis_raw, str) else basis_raw
        if len(basis_list) > MAX_ATOMS:
            print(f'  {fname}: skipping, {len(basis_list)} atoms > MAX_ATOMS={MAX_ATOMS}', flush=True)
            skipped += 1
            continue

        try:
            lattice = primitive_lattice_from_raw_phase(lattice_str, basis_raw)
        except Exception as e:
            print(f'  {fname}: primitive reduction failed - {e}', flush=True)
            skipped += 1
            continue

        pid = 'CNRS_' + fname.replace('.json', '')
        entries.append((pid, d['two_theta_values'], d['intensities'], float(wl), lattice))
    return entries, skipped


def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--source', choices=['aggregated', 'raw'], default='aggregated',
                    help="'aggregated' (default): collaborator single JSON. "
                         "'raw': directory of per-pattern opXRD JSON files.")
    p.add_argument('--json-path', default=JSON_PATH, help='aggregated source: path to the single JSON file')
    p.add_argument('--raw-dir', default=RAW_DIR, help='raw source: directory of pattern_*.json files')
    p.add_argument('--out-dir', default=OUT_DIR)
    return p


def main():
    args = build_parser().parse_args()

    if args.source == 'aggregated':
        if not args.json_path:
            raise SystemExit('--source aggregated needs --json-path (or CNRS_JSON_PATH); use --source raw for a directory of pattern files')
        entries, skipped = load_aggregated(args.json_path)
    else:
        entries, skipped = load_raw_dir(args.raw_dir)

    xrd_dict, lat_dict = {}, {}
    converted = 0

    for pid, theta2, intensity, wavelength, lattice in entries:
        xrd, was_converted = build_xrd_array(theta2, intensity, wavelength)
        if xrd is None:
            skipped += 1
            continue
        if was_converted:
            converted += 1
        xrd_dict[pid] = xrd
        lat_dict[pid] = lattice

    n_cuka = len(xrd_dict) - converted
    print(f'Total kept: {len(xrd_dict)}  |  CuKa as-is: {n_cuka}  |  wavelength-converted: {converted}  |  skipped: {skipped}')

    os.makedirs(args.out_dir, exist_ok=True)
    xrd_path = os.path.join(args.out_dir, 'opxrd_cnrs_xrd.pkl')
    lat_path = os.path.join(args.out_dir, 'opxrd_cnrs_lat.pkl')
    with open(xrd_path, 'wb') as f:
        pickle.dump(xrd_dict, f)
    with open(lat_path, 'wb') as f:
        pickle.dump(lat_dict, f)

    print(f'Saved XRD patterns   -> {xrd_path}')
    print(f'Saved lattice params -> {lat_path}')

    pid0 = next(iter(xrd_dict))
    print(f'\nSample entry: {pid0}')
    print(f'  XRD shape: {xrd_dict[pid0].shape}, max={xrd_dict[pid0].max():.3f}')
    print(f'  Lattice:   {lat_dict[pid0]}')


if __name__ == '__main__':
    main()
