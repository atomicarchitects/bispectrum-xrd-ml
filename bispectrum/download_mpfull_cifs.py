"""
Downloads CIF strings and space group numbers from the Materials Project API
for all mat_ids in the mpfull inversion results H5 file.

Usage:
    python download_mpfull_cifs.py \
        --h5 $POWDERXRD_DATA_ROOT/final_test_results/mpfull/inversion_results_merged.h5 \
        --output $POWDERXRD_DATA_ROOT/mpfull/test.csv \
        --api-key YOUR_API_KEY
"""

import argparse
import h5py
import pandas as pd
from tqdm import tqdm
from mp_api.client import MPRester


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5", required=True, help="Path to mpfull H5 file.")
    parser.add_argument("--output", required=True, help="Path to save output CSV.")
    parser.add_argument("--api-key", required=True, help="Materials Project API key.")
    parser.add_argument("--batch-size", type=int, default=100, help="Number of IDs per API request (default: 100).")
    args = parser.parse_args()

    with h5py.File(args.h5, 'r') as f:
        mat_ids = list(f.keys())
    print(f"Found {len(mat_ids)} mat_ids in {args.h5}")

    import os
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    rows = []
    batches = [mat_ids[i:i+args.batch_size] for i in range(0, len(mat_ids), args.batch_size)]

    with MPRester(args.api_key) as mpr:
        for batch in tqdm(batches, desc="Downloading from MP"):
            try:
                docs = mpr.materials.search(
                    material_ids=batch,
                    fields=["material_id", "structure", "symmetry", "theoretical"]
                )
                for doc in docs:
                    rows.append({
                        'material_id': doc.material_id,
                        'cif': doc.structure.to(fmt="cif"),
                        'spacegroup.number': doc.symmetry.number,
                        'theoretical': doc.theoretical,
                    })
            except Exception as e:
                print(f"Batch failed: {e}, retrying individually...")
                for mat_id in batch:
                    try:
                        docs = mpr.materials.search(
                            material_ids=[mat_id],
                            fields=["material_id", "structure", "symmetry", "theoretical"]
                        )
                        for doc in docs:
                            rows.append({
                                'material_id': doc.material_id,
                                'cif': doc.structure.to(fmt="cif"),
                                'spacegroup.number': doc.symmetry.number,
                                'theoretical': doc.theoretical,
                            })
                    except Exception as e2:
                        print(f"  Failed for {mat_id}: {e2}")

    df = pd.DataFrame(rows)
    print(f"Downloaded {len(df)}/{len(mat_ids)} entries")
    n_missing = len(mat_ids) - len(df)
    if n_missing > 0:
        print(f"Warning: {n_missing} mat_ids failed to download")

    df.to_csv(args.output, index=False)
    print(f"Saved to {args.output}")


if __name__ == "__main__":
    main()
