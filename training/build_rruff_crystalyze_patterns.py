"""
Build the measured patterns for the Crystalyze RRUFF evaluation (test_pv_xrd.pt, read by
eval_rruff_crystalyze.py) from RRUFF's processed powder files.

For each of the 148 entries of splits/rruff_crystalyze_ids.txt, the RRUFF "XY_Processed" powder
file (download from https://www.rruff.net/; file names start with '<mineral>__<rruff_id>') is
interpolated onto the model's grid, 2theta = 5 to 90 deg in 0.01 deg steps (8500 points, zero outside
the measured range), and divided by its maximum -- the same processing as
mp_full/gen_rruff_alpha_eval.py.

This reproduces the patterns used for the paper exactly (to float32 rounding) for 144 of the 148
entries. For Stephanite R050143, Pyrargyrite R050318, Senarmontite R060002 and Proustite R060356 the
processed files RRUFF currently serves differ from the ones the paper's patterns were made from, so
those four are built from the current files and results on them can differ from the reported ones.

Output: {mineral: float32 tensor [1, 8500]}, by default $POWDERXRD_DATA_ROOT/rruff_crystalyze/test_pv_xrd.pt.

Usage:
    python build_rruff_crystalyze_patterns.py --xy-dir /path/to/XY_Processed
"""
import argparse
import glob
import os

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
GRID = np.arange(5, 90, 0.010)  # 8500 points, as in training
CHANGED_UPSTREAM = {"R050143", "R050318", "R060002", "R060356"}


def read_ids(path):
    pairs = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#"):
                mineral, rruff_id = line.split("\t")
                pairs.append((mineral, rruff_id))
    return pairs


def read_xy(path):
    """(2theta, intensity) rows of a RRUFF XY file; header and '##' lines are skipped."""
    rows = []
    with open(path, errors="ignore") as f:
        for line in f:
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                rows.append((float(parts[0]), float(parts[1])))
            except ValueError:
                continue
    return np.array(rows)


def to_grid(xy):
    y = np.interp(GRID, xy[:, 0], xy[:, 1], left=0.0, right=0.0)
    return y / y.max() if y.max() > 0 else y


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--xy-dir", required=True, help="directory of RRUFF XY_Processed powder files")
    p.add_argument("--ids", default=os.path.join(HERE, "splits", "rruff_crystalyze_ids.txt"))
    p.add_argument("--out", default=f"{_ROOT}/rruff_crystalyze/test_pv_xrd.pt")
    args = p.parse_args()

    out, missing = {}, []
    for mineral, rruff_id in read_ids(args.ids):
        files = sorted(glob.glob(os.path.join(args.xy_dir, f"{mineral}__{rruff_id}*XY_Processed*")))
        if not files:
            missing.append(f"{mineral}__{rruff_id}")
            continue
        if len(files) > 1:
            print(f"  {mineral}__{rruff_id}: {len(files)} processed files, using {os.path.basename(files[0])}")
        out[mineral] = torch.tensor(to_grid(read_xy(files[0])), dtype=torch.float32).unsqueeze(0)
    if missing:
        raise SystemExit(f"no XY_Processed file in {args.xy_dir} for {len(missing)} entries: {missing[:5]}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    torch.save(out, args.out)
    print(f"wrote {args.out}: {len(out)} patterns "
          f"({len(CHANGED_UPSTREAM)} built from RRUFF files that changed since the paper; see the docstring)")


if __name__ == "__main__":
    main()
