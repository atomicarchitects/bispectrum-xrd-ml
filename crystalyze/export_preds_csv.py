#!/usr/bin/env python3
"""Export Crystalyze predicted (and ground-truth) lattice parameters to CSV.

Reads an eval_recon_<label>.pt written by Crystalyze's scripts/evaluate.py and
writes the pred_a/pred_b/.../pred_gamma + gt_a/.../gt_gamma columns that
`bispectrum/inversion_results_cctbx.py --crystalyze-csv` expects.

The ids come from the test CSV, in row order, which is the order
`input_data_batch` is in (evaluate.py runs the test loader without shuffling).

`--reduce mean` averages over the num_evals Langevin samples (used for the
148-entry RRUFF run, which was evaluated with --num_evals 5); `--reduce index
--eval-idx 0` takes a single sample.

Usage:
    python export_preds_csv.py \
        --recon-pt /path/to/Crystalyze/their_model/eval_recon_rruff_cag_force_na.pt \
        --test-csv $POWDERXRD_DATA_ROOT/rruff_crystalyze/test.csv \
        --output   /path/to/eval_recon_rruff_cag.csv
"""

import argparse

import pandas as pd
import torch


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--recon-pt', required=True, help='eval_recon_<label>.pt from evaluate.py.')
    p.add_argument('--test-csv', required=True,
                   help="test.csv of the evaluated set ('material_id' column gives the ids).")
    p.add_argument('--output', required=True, help='Output CSV path.')
    p.add_argument('--reduce', choices=['mean', 'index'], default='mean',
                   help='How to collapse the num_evals axis (default: mean).')
    p.add_argument('--eval-idx', type=int, default=0,
                   help='Which eval to take when --reduce index (default: 0).')
    return p.parse_args()


def main():
    args = parse_args()

    d = torch.load(args.recon_pt, map_location='cpu', weights_only=False)

    # lengths/angles: (num_evals, n_crystals, 3)
    if args.reduce == 'mean':
        pred_lengths = d['lengths'].mean(0)
        pred_angles = d['angles'].mean(0)
    else:
        pred_lengths = d['lengths'][args.eval_idx]
        pred_angles = d['angles'][args.eval_idx]

    gt = d['input_data_batch']
    gt_lengths, gt_angles = gt.lengths, gt.angles

    material_ids = pd.read_csv(args.test_csv)['material_id'].tolist()
    if len(material_ids) != pred_lengths.shape[0]:
        raise SystemExit(f'Mismatch: {len(material_ids)} ids in {args.test_csv} vs '
                         f'{pred_lengths.shape[0]} predictions in {args.recon_pt}')

    rows = []
    for i, mid in enumerate(material_ids):
        rows.append({
            'material_id': mid,
            'pred_a': pred_lengths[i, 0].item(),
            'pred_b': pred_lengths[i, 1].item(),
            'pred_c': pred_lengths[i, 2].item(),
            'pred_alpha': pred_angles[i, 0].item(),
            'pred_beta': pred_angles[i, 1].item(),
            'pred_gamma': pred_angles[i, 2].item(),
            'gt_a': gt_lengths[i, 0].item(),
            'gt_b': gt_lengths[i, 1].item(),
            'gt_c': gt_lengths[i, 2].item(),
            'gt_alpha': gt_angles[i, 0].item(),
            'gt_beta': gt_angles[i, 1].item(),
            'gt_gamma': gt_angles[i, 2].item(),
        })

    df = pd.DataFrame(rows)
    df.to_csv(args.output, index=False)
    print(f'Saved {len(df)} rows to {args.output}')


if __name__ == '__main__':
    main()
