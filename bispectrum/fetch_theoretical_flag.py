"""
Patches the 'theoretical' flag from the Materials Project into test.csv.
Structures with theoretical=False have ICSD entries (experimentally observed).

Uses mpr.summary.search() which returns SummaryDoc objects that carry the
'theoretical' field (mpr.materials.search returns MPDataDoc which does not).

Usage:
    python fetch_theoretical_flag.py --api-key YOUR_API_KEY
    python fetch_theoretical_flag.py --api-key YOUR_API_KEY --batch-size 100
"""

import argparse
import os
import pandas as pd
from tqdm import tqdm
from mp_api.client import MPRester

CSV_PATH = os.path.join(os.environ.get("POWDERXRD_DATA_ROOT", "./data"), "mpfull", "test.csv")

parser = argparse.ArgumentParser()
parser.add_argument("--api-key", required=True)
parser.add_argument("--batch-size", type=int, default=100)
args = parser.parse_args()

df = pd.read_csv(CSV_PATH)
mat_ids = df["material_id"].tolist()
print(f"Fetching 'theoretical' flag for {len(mat_ids)} structures...")

results = {}
batches = [mat_ids[i:i+args.batch_size] for i in range(0, len(mat_ids), args.batch_size)]

with MPRester(args.api_key) as mpr:
    for batch in tqdm(batches):
        try:
            docs = mpr.summary.search(
                material_ids=batch,
                fields=["material_id", "theoretical"]
            )
            for doc in docs:
                results[doc.material_id] = doc.theoretical
        except Exception as e:
            print(f"Batch failed: {e}, retrying individually...")
            for mat_id in batch:
                try:
                    docs = mpr.summary.search(
                        material_ids=[mat_id],
                        fields=["material_id", "theoretical"]
                    )
                    for doc in docs:
                        results[doc.material_id] = doc.theoretical
                except Exception as e2:
                    print(f"  Failed for {mat_id}: {e2}")

df["theoretical"] = df["material_id"].map(results)
n_missing = df["theoretical"].isna().sum()
if n_missing:
    print(f"Warning: {n_missing} entries missing (left as NaN)")

df.to_csv(CSV_PATH, index=False)
n_exp = (df["theoretical"] == False).sum()
n_theo = (df["theoretical"] == True).sum()
print(f"Done. Experimentally observed: {n_exp}, Theoretical: {n_theo}, Missing: {n_missing}")
print(f"Saved to {CSV_PATH}")
