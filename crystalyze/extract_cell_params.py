import argparse
import pandas as pd
import torch
from pathlib import Path

parser = argparse.ArgumentParser(description="Extract cell params from an eval_recon .pt file")
parser.add_argument("--recon_pt", required=True, help="Path to eval_recon_*.pt file")
parser.add_argument("--test_csv", default=None,
                    help="Path to test.csv (default: test.csv in same dir as recon_pt)")
parser.add_argument("--output", default=None,
                    help="Output .pt path (default: cell_params_by_mpid_<label>.pt in same dir)")
parser.add_argument("--eval_idx", type=int, default=0,
                    help="Which eval index to use if num_evals > 1 (default: 0)")
args = parser.parse_args()

recon_pt = Path(args.recon_pt)
test_csv = Path(args.test_csv) if args.test_csv else recon_pt.parent / "test.csv"
output = Path(args.output) if args.output else recon_pt.parent / f"cell_params_by_mpid_{recon_pt.stem}.pt"

df = pd.read_csv(test_csv)
data = torch.load(recon_pt, map_location="cpu")

mp_ids = df["material_id"].tolist()
lengths = data["lengths"][args.eval_idx]
angles  = data["angles"][args.eval_idx]

assert len(mp_ids) == lengths.shape[0] == angles.shape[0], \
    f"Mismatch: {len(mp_ids)} mp-ids vs {lengths.shape[0]} predictions"

result = {}
for i, mp_id in enumerate(mp_ids):
    result[mp_id] = {
        "lengths": lengths[i].numpy(),
        "angles":  angles[i].numpy(),
    }

torch.save(result, output)
print(f"Saved {len(result)} entries to {output}")
