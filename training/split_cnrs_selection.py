"""
Split the corrected opXRD CNRS eval set into two disjoint slices:

  selection : used to pick a checkpoint (e.g. from the periodic snapshots
              train.py now saves via train.snapshot_every_epochs) by real-
              domain performance.
  reporting : held out, untouched by that selection -- this is the set
              whose performance actually gets reported (papers, etc).

Selecting a checkpoint and reporting its performance on the SAME data is a
form of test-set leakage (picking the best of several noisy measurements on
a fixed eval set inflates the reported number). Keeping these two slices
disjoint avoids that, the same way a train/val/test split does.

Split is a fixed-seed random shuffle (seed=42, matching conf/config.yaml's
training seed) of pattern IDs -- not stratified by anything, since there's
no known quality/structure-type label to stratify on yet.

Usage:
    python split_cnrs_selection.py \
        --xrd-pkl  $POWDERXRD_DATA_ROOT/opxrd_cnrs_eval/opxrd_cnrs_xrd.pkl \
        --lat-pkl  $POWDERXRD_DATA_ROOT/opxrd_cnrs_eval/opxrd_cnrs_lat.pkl \
        --out-dir  $POWDERXRD_DATA_ROOT/opxrd_cnrs_eval \
        --selection-frac 0.2 --seed 42
"""
import argparse
import os
import pickle
import random


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--xrd-pkl', required=True)
    p.add_argument('--lat-pkl', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--selection-frac', type=float, default=0.2)
    p.add_argument('--seed', type=int, default=42)
    args = p.parse_args()

    xrd = pickle.load(open(args.xrd_pkl, 'rb'))
    lat = pickle.load(open(args.lat_pkl, 'rb'))
    ids = sorted(set(xrd) & set(lat))
    print(f'Total patterns: {len(ids)}')

    rng = random.Random(args.seed)
    shuffled = ids[:]
    rng.shuffle(shuffled)
    n_selection = round(len(shuffled) * args.selection_frac)
    selection_ids = sorted(shuffled[:n_selection])
    reporting_ids = sorted(shuffled[n_selection:])
    print(f'selection: {len(selection_ids)}  reporting: {len(reporting_ids)}')

    os.makedirs(args.out_dir, exist_ok=True)
    for name, id_list in [('selection', selection_ids), ('reporting', reporting_ids)]:
        xrd_out = {pid: xrd[pid] for pid in id_list}
        lat_out = {pid: lat[pid] for pid in id_list}
        with open(os.path.join(args.out_dir, f'opxrd_cnrs_xrd_{name}.pkl'), 'wb') as f:
            pickle.dump(xrd_out, f)
        with open(os.path.join(args.out_dir, f'opxrd_cnrs_lat_{name}.pkl'), 'wb') as f:
            pickle.dump(lat_out, f)
        with open(os.path.join(args.out_dir, f'opxrd_cnrs_{name}_ids.txt'), 'w') as f:
            f.write('\n'.join(id_list) + '\n')
        print(f'Saved {name}: '
              f'{os.path.join(args.out_dir, f"opxrd_cnrs_xrd_{name}.pkl")}, '
              f'{os.path.join(args.out_dir, f"opxrd_cnrs_lat_{name}.pkl")}, '
              f'{os.path.join(args.out_dir, f"opxrd_cnrs_{name}_ids.txt")}')


if __name__ == '__main__':
    main()
