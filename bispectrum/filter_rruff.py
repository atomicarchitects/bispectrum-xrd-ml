"""
RRUFF dataset filtering pipeline.

Steps:
  1. Start: all DIF files + all XY_Processed files
  2. Filter DIF: must have cell parameters, space group, and wavelength
  3. Parse DIF into pymatgen Lattice objects (drop failures)
  4. Check lattice-symmetry consistency (rtol=1e-3) and volume < 100,000 Å³
  5. Match valid DIF entries against XY_Processed by RRUFF ID
  6. Cross-validate: DIF and XY_Processed must agree on crystal system and
     lattice parameters (lengths within 5% rtol, angles within 1°)
  7. (Optional) Filter to entries whose RRUFF ID appears in a set of h5 files
     (e.g. processed_xy inversion results). Pass --h5-dir to enable.

Outputs:
  $POWDERXRD_DATA_ROOT/rruff_filtered.csv   — final matched entries
"""

import argparse
import glob
import os
import re

import h5py
import pandas as pd
from pymatgen.core import Lattice
from pymatgen.symmetry.groups import SpaceGroup

ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
DIF_DIR = os.path.join(ROOT, 'DIF')
XY_DIR  = os.path.join(ROOT, 'XY_Processed')
OUT_CSV = os.path.join(ROOT, 'rruff_filtered.csv')


# ── helpers ───────────────────────────────────────────────────────────────────

def extract_rruff_id(fname):
    m = re.search(r'__(R\d+(?:-\d+)?(?:-\w+)?)__', fname)
    return m.group(1) if m else None


def parse_dif(path):
    """
    Parse a DIF file. Returns dict with keys:
        a, b, c, alpha, beta, gamma, space_group, wavelength
    or None if any required field is missing / unparseable.
    """
    cell = spg = wl = None
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if cell is None and re.search(r'CELL PARAMETERS', line):
                nums = re.findall(r'[\d.]+', line)
                if len(nums) >= 6:
                    cell = [float(x) for x in nums[:6]]
            if spg is None and re.match(r'\s+SPACE GROUP:', line):
                spg = line.split('SPACE GROUP:')[1].strip()
            if wl is None and re.search(r'X-RAY WAVELENGTH', line):
                nums = re.findall(r'[\d.]+', line)
                if nums:
                    wl = float(nums[0])   # take first (or only) value
    if cell is None or spg is None or wl is None:
        return None
    return dict(a=cell[0], b=cell[1], c=cell[2],
                alpha=cell[3], beta=cell[4], gamma=cell[5],
                space_group=spg, wavelength=wl)


def crystal_system_from_spg(spg_symbol):
    """Return crystal system string for a space group symbol, or None."""
    try:
        return SpaceGroup(spg_symbol).crystal_system
    except Exception:
        # try inserting '-' before '3' (e.g. Fd3m → Fd-3m)
        normalized = re.sub(r'(?<![- ])3', '-3', spg_symbol)
        try:
            return SpaceGroup(normalized).crystal_system
        except Exception:
            return None


# Crystal-system constraints for lattice-symmetry check.
# Each value is (equal_lengths, required_angles_90, required_gamma_120, equal_all_lengths)
_CS_CONSTRAINTS = {
    'cubic':        dict(ab=True,  ac=True,  ang90=['alpha','beta','gamma'], ang120=[]),
    'tetragonal':   dict(ab=True,  ac=False, ang90=['alpha','beta','gamma'], ang120=[]),
    'orthorhombic': dict(ab=False, ac=False, ang90=['alpha','beta','gamma'], ang120=[]),
    'hexagonal':    dict(ab=True,  ac=False, ang90=['alpha','beta'],         ang120=['gamma']),
    'trigonal':     dict(ab=True,  ac=False, ang90=['alpha','beta'],         ang120=['gamma']),
    'monoclinic':   dict(ab=False, ac=False, ang90=['alpha','gamma'],        ang120=[]),
    'triclinic':    dict(ab=False, ac=False, ang90=[],                       ang120=[]),
}


def check_lattice_symmetry(params, crystal_system, rtol=1e-3):
    """
    Return True if lattice parameters are consistent with crystal_system
    within the given relative tolerance.
    """
    cs = crystal_system.lower()
    if cs not in _CS_CONSTRAINTS:
        return True   # unknown system — don't discard
    con = _CS_CONSTRAINTS[cs]
    a, b, c = params['a'], params['b'], params['c']
    alpha, beta, gamma = params['alpha'], params['beta'], params['gamma']

    if con['ab'] and abs(a - b) / max(a, b) > rtol:
        return False
    if con['ac'] and abs(a - c) / max(a, c) > rtol:
        return False
    for ang_name in con['ang90']:
        val = {'alpha': alpha, 'beta': beta, 'gamma': gamma}[ang_name]
        if abs(val - 90.0) / 90.0 > rtol:
            return False
    for ang_name in con['ang120']:
        val = {'alpha': alpha, 'beta': beta, 'gamma': gamma}[ang_name]
        if abs(val - 120.0) / 120.0 > rtol:
            return False
    return True


def parse_xy_header(path):
    """
    Parse ##CELL PARAMETERS line from an XY_Processed file.
    Returns dict with a,b,c,alpha,beta,gamma,crystal_system or None.
    """
    with open(path, 'r', errors='replace') as f:
        for line in f:
            if not line.startswith('#'):
                break
            if line.startswith('##CELL PARAMETERS='):
                body = line.split('=', 1)[1]
                try:
                    a     = float(re.search(r'a:\s*([\d.]+)',     body).group(1))
                    b     = float(re.search(r'b:\s*([\d.]+)',     body).group(1))
                    c     = float(re.search(r'c:\s*([\d.]+)',     body).group(1))
                    alpha = float(re.search(r'alpha:\s*([\d.]+)', body).group(1))
                    beta  = float(re.search(r'beta:\s*([\d.]+)',  body).group(1))
                    gamma = float(re.search(r'gamma:\s*([\d.]+)', body).group(1))
                    cs_m  = re.search(r'crystal system:\s*(\w+)', body)
                    cs    = cs_m.group(1).lower() if cs_m else None
                    return dict(a=a, b=b, c=c, alpha=alpha, beta=beta,
                                gamma=gamma, crystal_system=cs)
                except Exception:
                    return None
    return None


def params_match(dif, xy, length_rtol=0.05, angle_atol=1.0):
    """
    Check that DIF and XY_Processed lattice parameters agree.
    """
    for key in ('a', 'b', 'c'):
        d, x = dif[key], xy[key]
        if abs(d - x) / max(d, x) > length_rtol:
            return False
    for key in ('alpha', 'beta', 'gamma'):
        if abs(dif[key] - xy[key]) > angle_atol:
            return False
    return True


def rruff_ids_from_h5_dir(h5_dir):
    """Return set of RRUFF IDs found in all h5 files under h5_dir."""
    ids = set()
    for path in glob.glob(os.path.join(h5_dir, '**', '*.h5'), recursive=True):
        with h5py.File(path, 'r') as f:
            for key in f.keys():
                rid = extract_rruff_id(key)
                if rid:
                    ids.add(rid)
    return ids


# ── main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Filter RRUFF dataset.')
    parser.add_argument('--h5-dir', default=None,
                        help='Directory of h5 result files to filter by. '
                             'Only entries whose RRUFF ID appears in these h5 keys are kept.')
    args = parser.parse_args()

    dif_files = sorted(glob.glob(os.path.join(DIF_DIR, '*.txt')))
    xy_files  = sorted(glob.glob(os.path.join(XY_DIR,  '*.txt')))

    print(f'Step 1 — Starting counts')
    print(f'  DIF files:         {len(dif_files)}')
    print(f'  XY_Processed:      {len(xy_files)}')

    # ── Step 2: parse DIF metadata ────────────────────────────────────────────
    dif_records = []
    for path in dif_files:
        rid = extract_rruff_id(os.path.basename(path))
        params = parse_dif(path)
        if params and rid:
            dif_records.append(dict(rruff_id=rid, path=path, **params))

    print(f'\nStep 2 — DIF with valid cell params + space group + wavelength')
    print(f'  Kept: {len(dif_records)} / {len(dif_files)}')
    print(f'  Dropped (missing fields): {len(dif_files) - len(dif_records)}')

    # ── Step 3: parse into pymatgen Lattice ───────────────────────────────────
    valid = []
    failed = 0
    for rec in dif_records:
        try:
            lat = Lattice.from_parameters(
                rec['a'], rec['b'], rec['c'],
                rec['alpha'], rec['beta'], rec['gamma']
            )
            cs = crystal_system_from_spg(rec['space_group'])
            if cs is None:
                raise ValueError(f"Unknown space group: {rec['space_group']}")
            rec['crystal_system'] = cs
            rec['volume'] = lat.volume
            valid.append(rec)
        except Exception:
            failed += 1

    print(f'\nStep 3 — Parse into pymatgen Lattice + resolve space group')
    print(f'  Kept: {len(valid)} / {len(dif_records)}')
    print(f'  Dropped (parse errors): {failed}')

    # ── Step 4: lattice-symmetry consistency + volume filter ──────────────────
    consistent = []
    for rec in valid:
        vol_ok  = rec['volume'] < 100_000
        sym_ok  = check_lattice_symmetry(rec, rec['crystal_system'], rtol=1e-3)
        if vol_ok and sym_ok:
            consistent.append(rec)

    dropped_sym = len(valid) - len(consistent)
    print(f'\nStep 4 — Lattice-symmetry consistency (rtol=1e-3) + volume < 100,000 Å³')
    print(f'  Kept: {len(consistent)} / {len(valid)}')
    print(f'  Dropped: {dropped_sym}')

    # ── Step 5: match against XY_Processed ───────────────────────────────────
    xy_by_id = {}
    for path in xy_files:
        rid = extract_rruff_id(os.path.basename(path))
        if rid:
            xy_by_id[rid] = path

    matched = [rec for rec in consistent if rec['rruff_id'] in xy_by_id]
    for rec in matched:
        rec['xy_path'] = xy_by_id[rec['rruff_id']]

    print(f'\nStep 5 — Match DIF against XY_Processed')
    print(f'  Kept: {len(matched)} / {len(consistent)}')
    print(f'  Dropped (no PXRD match): {len(consistent) - len(matched)}')

    # ── Step 6: cross-validate crystal system + lattice params ───────────────
    final = []
    dropped_cs = 0
    dropped_params = 0
    for rec in matched:
        xy_meta = parse_xy_header(rec['xy_path'])
        if xy_meta is None:
            dropped_cs += 1
            continue
        # crystal system must agree
        if xy_meta['crystal_system'] and xy_meta['crystal_system'] != rec['crystal_system'].lower():
            dropped_cs += 1
            continue
        # lattice parameters must agree
        if not params_match(rec, xy_meta):
            dropped_params += 1
            continue
        rec['xy_crystal_system'] = xy_meta['crystal_system']
        final.append(rec)

    print(f'\nStep 6 — Cross-validate crystal system + lattice parameters')
    print(f'  Kept: {len(final)} / {len(matched)}')
    print(f'  Dropped (crystal system mismatch): {dropped_cs}')
    print(f'  Dropped (lattice param mismatch):  {dropped_params}')

    # ── Step 7 (optional): filter to h5 keys ─────────────────────────────────
    h5_ids = None
    if args.h5_dir:
        h5_ids = rruff_ids_from_h5_dir(args.h5_dir)
        before = len(final)
        final = [rec for rec in final if rec['rruff_id'] in h5_ids]
        print(f'\nStep 7 — Filter to RRUFF IDs present in h5 files under {args.h5_dir}')
        print(f'  Kept: {len(final)} / {before}')
        print(f'  Dropped: {before - len(final)}')

    # ── Save ──────────────────────────────────────────────────────────────────
    cols = ['rruff_id', 'crystal_system', 'space_group', 'wavelength',
            'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'volume', 'path', 'xy_path']
    df = pd.DataFrame(final)[cols]
    df.to_csv(OUT_CSV, index=False)
    print(f'\nSaved {len(df)} entries to {OUT_CSV}')

    print(f'\n── Summary ──────────────────────────────────────')
    print(f'  DIF files:                    {len(dif_files)}')
    print(f'  After field filter:           {len(dif_records)}')
    print(f'  After lattice parse:          {len(valid)}')
    print(f'  After symmetry+volume filter: {len(consistent)}')
    print(f'  After PXRD match:             {len(matched)}')
    print(f'  After cross-validation:       {len(final)}')
    if h5_ids is not None:
        print(f'  After h5 key filter:          {len(final)}')


if __name__ == '__main__':
    main()
