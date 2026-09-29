"""
Build an inversion-ready input pkl (bispectrum/run_alg_inversion.py's
expected format: {material_ids, predictions, targets}) for one or more
checkpoints scored against the opXRD CNRS *selection* slice.

`targets` here is (N,6) true lattice params (a,b,c,alpha,beta,gamma), NOT
bispectra -- run_alg_inversion.py's main() only uses `targets` as the RRUFF-
style true_lat_params fallback (used when material_ids aren't found in the
MP database, exactly our case), for the final predicted-vs-true Niggli
comparison, not for seeding candidates (those come from nearest-neighbor
bispectra in the MP database, mp_id-independent).

Usage:
    python build_cnrs_selection_inversion_input.py \\
        --name peaklist_bgmix_epoch9 \\
        --checkpoint-path /path/to/epoch9_....pt --input-type sparse
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_opxrd_cnrs_selection_cosine import load_context, eval_checkpoint

_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
CNRS_DIR = f"{_ROOT}/opxrd_cnrs_eval"
OUT_DIR = f"{_ROOT}/eval_results/opxrd_cnrs/inversion_inputs"
VERIFIED_IDS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "splits", "cnrs_verified_ids.txt")


def load_verified_ids():
    with open(VERIFIED_IDS) as f:
        return {l.strip() for l in f if l.strip() and not l.startswith("#")}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--name", required=True, help="Output file stem, e.g. peaklist_bgmix_epoch9")
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--input-type", required=True, choices=["dense", "sparse"])
    p.add_argument("--slice", default="selection", choices=["selection", "reporting", "all"],
                   help="'selection' (162, checkpoint-picking) or 'reporting' (650, "
                        "held-out for the actual reported number) or 'all' (both).")
    p.add_argument("--norm-stats", default=None,
                   help="Dense-model train-set norm stats json (default: full-MP model's). "
                        "Required for a checkpoint trained on a different dataset, e.g. mp20aug.")
    p.add_argument("--allowed-from-checkpoint", action="store_true",
                   help="Take the structured-head mask from the checkpoint's own head.row_idx/"
                        "col_idx instead of recomputing it from the full-MP training targets.")
    p.add_argument("--verified-max-atoms", type=int, default=None,
                   help="Restrict to patterns in the verified set with <= this many atoms.")
    p.add_argument("--primitive", action="store_true",
                   help="With --verified-max-atoms: count atoms in the primitive cell "
                        "(opxrd_cnrs_prim_atom_counts.pkl) instead of len(basis) of the deposited cell.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    kw = {}
    if args.allowed_from_checkpoint:
        sd = ckpt["model_state_dict"]
        kw["allowed_indices"] = torch.stack([sd["head.row_idx"], sd["head.col_idx"]], dim=1).tolist()
    if args.norm_stats:
        kw["norm_stats_path"] = args.norm_stats
    if args.verified_max_atoms is not None:
        vids = load_verified_ids()
        if args.primitive:
            # atoms in the PRIMITIVE cell (count_cnrs_primitive_atoms.py), what MP-20's limit refers to
            prim = pickle.load(open(f"{CNRS_DIR}/opxrd_cnrs_prim_atom_counts.pkl", "rb"))
            atoms = {k: v["n_prim"] for k, v in prim.items()}
        else:
            atoms = pickle.load(open(f"{CNRS_DIR}/opxrd_cnrs_atom_counts.pkl", "rb"))  # len(basis), deposited cell
        kw["ids"] = {k for k in vids if k in atoms and atoms[k] <= args.verified_max_atoms}
        print(f"{len(vids)} verified; {len(kw['ids'])} with <= {args.verified_max_atoms} "
              f"{'primitive-cell' if args.primitive else 'deposited-cell'} atoms")

    ctx = load_context(device=args.device, slice_name=args.slice, **kw)
    print(f"{len(ctx.keys)} patterns evaluated")
    epoch, val_loss, sims = eval_checkpoint(args.checkpoint_path, args.input_type, ctx)
    print(f"epoch={epoch} val_loss={val_loss} cos_mean={np.mean(sims):.4f} (sanity check, matches tracker)")

    # Re-run to get the raw predicted bispec array (eval_checkpoint only returns cosine sims).
    # Cheap to redo -- at most a few hundred entries.
    from model import XRDTransformerEncoder
    from eval_opxrd_cnrs_selection_cosine import MODEL_KW, MAX_PEAKS, pad_peaks
    from generate_augmented_data import _extract_peaks_via_find_peaks

    model = XRDTransformerEncoder(
        **MODEL_KW, input_type=args.input_type, max_peaks=MAX_PEAKS, use_intensity=True,
        allowed_indices=ctx.allowed_indices,
    ).to(ctx.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    with torch.no_grad():
        if args.input_type == "dense":
            if ctx.xrd_mean is not None:
                x = (ctx.dense_curves - ctx.xrd_mean) / (ctx.xrd_std + 1e-8)
            else:
                x = (ctx.dense_curves - ctx.dense_curves.mean()) / (ctx.dense_curves.std() + 1e-8)
            data = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(ctx.device)
        else:
            peak_lists = [pad_peaks(_extract_peaks_via_find_peaks(c, MAX_PEAKS)) for c in ctx.dense_curves]
            data = torch.tensor(peak_lists, dtype=torch.float32).to(ctx.device)
        pred = model(data).cpu().numpy()  # (N, 10, 35)

    lat_file = "opxrd_cnrs_lat.pkl" if args.slice == "all" else f"opxrd_cnrs_lat_{args.slice}.pkl"
    lat_sel = pickle.load(open(f"{CNRS_DIR}/{lat_file}", "rb"))
    targets = np.stack([np.asarray(lat_sel[k], dtype=np.float64) for k in ctx.keys])  # (N, 6)

    out = dict(material_ids=list(ctx.keys), predictions=pred.astype(np.float64), targets=targets)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{args.name}.pkl")
    with open(out_path, "wb") as f:
        pickle.dump(out, f)
    print(f"Saved {len(ctx.keys)} entries -> {out_path}")
    print(f"  predictions shape: {pred.shape}  targets shape: {targets.shape}")


if __name__ == "__main__":
    main()
