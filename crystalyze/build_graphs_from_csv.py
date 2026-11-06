#!/usr/bin/env python3
"""Build test.pt (CrystalNN crystal graphs) from a CSV of CIF strings.

prepare_test_data.py already writes test.pt with its own vendored copies of
build_crystal/build_crystal_graph. This script does the same thing but imports
those functions from the Crystalyze checkout itself, so the graphs are built by
exactly the code that evaluate.py's dataloader expects. Use it when a test.pt
has to be rebuilt without regenerating the CSV and the pseudo-Voigt patterns
(this is how rruff_crystalyze/test.pt was produced).

Usage:
    python build_graphs_from_csv.py \
        --crystalyze-dir /path/to/Crystalyze \
        --csv /path/to/test_set_dir/test.csv \
        --output /path/to/test_set_dir/test.pt
"""

import argparse
import sys

import pandas as pd
import torch
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--crystalyze-dir', required=True,
                   help='Root of the Crystalyze checkout (added to sys.path).')
    p.add_argument('--csv', required=True,
                   help="CSV with 'material_id' and 'cif' columns.")
    p.add_argument('--output', required=True, help='Output test.pt path.')
    p.add_argument('--niggli', action='store_true', default=True)
    p.add_argument('--primitive', action='store_true', default=False)
    p.add_argument('--graph-method', default='crystalnn')
    return p.parse_args()


def main():
    args = parse_args()
    sys.path.insert(0, args.crystalyze_dir)
    from cdvae.common.data_utils import build_crystal, build_crystal_graph

    df = pd.read_csv(args.csv)
    print(f'Building graphs for {len(df)} entries...')

    graph_dict = {}
    skipped = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        mid = row['material_id']
        try:
            crystal = build_crystal(row['cif'], niggli=args.niggli, primitive=args.primitive)
            graph_dict[mid] = build_crystal_graph(crystal, graph_method=args.graph_method)
        except Exception as e:
            print(f'  skipped {mid}: {e}')
            skipped.append(mid)

    print(f'Built {len(graph_dict)} graphs, skipped {len(skipped)}')
    torch.save(graph_dict, args.output)
    print(f'Saved to {args.output}')


if __name__ == '__main__':
    main()
