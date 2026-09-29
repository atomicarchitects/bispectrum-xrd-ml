"""
Cosine-similarity screen of trained checkpoints against the opXRD CNRS
*selection* slice (162 patterns, disjoint from the 650-pattern `reporting`
slice -- see split_cnrs_selection.py) for real-data domain-gap comparison.

True bispectra come from compute_cnrs_true_bispec.py's output (computed from
the raw CNRS structure JSONs, reduced to primitive cell -- bispectrum depends
only on the reciprocal lattice, not atomic positions/species). Predicted
bispectra come from running each checkpoint's model on the *same* selection
XRD curves:
  - dense checkpoints (input_type=dense, e.g. mpfull_aug / mpfull_aug_cnrs_mix)
    get the raw dense curve normalized by the training set's xrd_mean/std
    (see compute_xrd_norm_stats.py -- cached, not recomputed here).
  - sparse checkpoints (input_type=sparse, e.g. mpfull_aug_final_peaklist) get
    peaks extracted from the *same* dense curve via
    generate_augmented_data._extract_peaks_via_find_peaks, matching the exact
    extraction pipeline used to build their training data (train/eval
    consistency -- the whole point of the peak-list approach).

This is a *screen*, not the final reported number: it exists to rank
checkpoints/architectures cheaply before spending L-BFGS inversion compute
only on the finalists, and only ever touches the selection slice -- the
reporting slice stays untouched for the actual number that goes in the paper.

The per-checkpoint loading/inference logic here is also imported by
track_snapshot_cosine.py to watch a run's cosine similarity across all its
saved snapshots over time (e.g. to catch real-domain overfitting even while
clean val_loss keeps improving).

Usage:
    python eval_opxrd_cnrs_selection_cosine.py
    python eval_opxrd_cnrs_selection_cosine.py --checkpoints original cnrs_mix peaklist
    python eval_opxrd_cnrs_selection_cosine.py --checkpoints \\
        --extra-checkpoint epoch18 /path/to/epoch18_....pt sparse
"""
import argparse
import json
import os
import pickle
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp_full"))
from model import XRDTransformerEncoder
from generate_augmented_data import _extract_peaks_via_find_peaks

_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
CNRS_DIR = f"{_ROOT}/opxrd_cnrs_eval"
TRUE_BISPEC_PATH = f"{_ROOT}/eval_results/opxrd_cnrs/opxrd_cnrs_true_bispec.pkl"
DENSE_TARGET_PATH = f"{_ROOT}/mpfull_aug_cag_strain_texture/train_bispec_combined.pkl"
NORM_STATS_PATH = f"{_ROOT}/mpfull_aug_cag_strain_texture/mpfull_aug_final_xrd_norm_stats.json"
MASK_LOSS_THRESHOLD = 1e-10
MAX_PEAKS = 30

CHECKPOINTS = {
    "original": dict(
        path=f"{_ROOT}/final_saved_models/mpfull_aug/mpfull_aug_bispec_best_model.pt",
        input_type="dense",
    ),
    "cnrs_mix": dict(
        path=f"{_ROOT}/mpfull_aug_final_cnrs_mix/transformerbispec/models/"
             "nlayers12_nhead8_dm256_hd512_lossL1_smallvalsFalse_masklossTrue.pt",
        input_type="dense",
    ),
    "peaklist": dict(
        path=f"{_ROOT}/mpfull_aug_final_peaklist/transformerbispec/models/"
             "nlayers12_nhead8_dm256_hd512_lossL1_smallvalsFalse_masklossTrue.pt",
        input_type="sparse",
    ),
    "peaklist_bgmix": dict(
        path=f"{_ROOT}/mpfull_aug_final_peaklist_bgmix/transformerbispec/models/"
             "nlayers12_nhead8_dm256_hd512_lossL1_smallvalsFalse_masklossTrue.pt",
        input_type="sparse",
    ),
}

MODEL_KW = dict(out_dim=(10, 35), d_model=256, h_dim=512, transformer_proc="tokenization",
                 n_self_layer=12, n_head=8, attn_pdrop=0.2, resid_pdrop=0.2)


def cos_sim_flat(a, b):
    a, b = a.flatten(), b.flatten()
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def pad_peaks(peak_list, max_peaks=MAX_PEAKS, pad=(-1.0, -1.0)):
    peak_list = sorted(peak_list, key=lambda p: p[0])[:max_peaks]
    while len(peak_list) < max_peaks:
        peak_list.append(pad)
    return peak_list


class EvalContext:
    """Shared, expensive-to-load state (selection slice, true bispecs,
    structured-head mask, dense-model norm stats) for scoring one or many
    checkpoints against the CNRS selection slice without reloading it each
    time -- see load_context()."""
    def __init__(self, keys, dense_curves, true_mat, allowed_indices, xrd_mean, xrd_std, device):
        self.keys = keys
        self.dense_curves = dense_curves
        self.true_mat = true_mat
        self.allowed_indices = allowed_indices
        self.xrd_mean = xrd_mean
        self.xrd_std = xrd_std
        self.device = device


def load_context(device=None, verbose=True, slice_name="selection",
                 allowed_indices=None, norm_stats_path=NORM_STATS_PATH, ids=None):
    """slice_name: 'selection' (162 patterns, checkpoint-picking) or
    'reporting' (650 patterns, held untouched for the actual reported
    number -- see split_cnrs_selection.py), or 'all' (selection + reporting).
    TRUE_BISPEC_PATH covers the full 812-pattern CNRS set already, so it
    needs no per-slice variant.

    Defaults are the full-MP model's structured-head mask and norm stats. For
    a different checkpoint (e.g. mp20_aug) pass its own `allowed_indices`
    (skips loading the full-MP target file) and `norm_stats_path`. `ids`
    optionally restricts the patterns evaluated."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    xrd_file = "opxrd_cnrs_xrd.pkl" if slice_name == "all" else f"opxrd_cnrs_xrd_{slice_name}.pkl"
    xrd_sel = pickle.load(open(f"{CNRS_DIR}/{xrd_file}", "rb"))
    true_bispecs = pickle.load(open(TRUE_BISPEC_PATH, "rb"))
    keys = sorted(set(xrd_sel) & set(true_bispecs))
    if ids is not None:
        keys = [k for k in keys if k in ids]
    missing = set(xrd_sel) - set(true_bispecs)
    if verbose:
        print(f"{slice_name} slice: {len(xrd_sel)} patterns, {len(keys)} with a true bispec"
              f" ({len(missing)} missing: {sorted(missing)[:5]}{'...' if len(missing) > 5 else ''})")

    dense_curves = np.stack([np.asarray(xrd_sel[k], dtype=np.float64) for k in keys])
    true_mat = np.stack([true_bispecs[k] for k in keys])  # (N, 10, 35)

    if allowed_indices is None:
        if verbose:
            print(f"loading {DENSE_TARGET_PATH} for the structured-head mask ...")
        target_train = pickle.load(open(DENSE_TARGET_PATH, "rb"))
        all_targets = torch.stack(list(target_train.values()))
        loss_mask = torch.any(torch.abs(all_targets) > MASK_LOSS_THRESHOLD, dim=0)
        allowed_indices = torch.nonzero(loss_mask, as_tuple=False).tolist()
        del target_train, all_targets
    if verbose:
        print(f"  {len(allowed_indices)} allowed (10,35) entries")

    xrd_mean = xrd_std = None
    if os.path.exists(norm_stats_path):
        stats = json.load(open(norm_stats_path))
        xrd_mean, xrd_std = stats["xrd_mean"], stats["xrd_std"]
        if verbose:
            print(f"loaded cached norm stats: mean={xrd_mean:.6f} std={xrd_std:.6f}")
    elif norm_stats_path != NORM_STATS_PATH:
        raise FileNotFoundError(f"{norm_stats_path} not found -- run compute_xrd_norm_stats.py first")
    elif verbose:
        print(f"[WARN] {NORM_STATS_PATH} not found -- dense checkpoints will be "
              f"normalized using the SELECTION SET's own mean/std, which does not "
              f"match what the model saw at train time. Run compute_xrd_norm_stats.py "
              f"--data mpfull_aug_final first for a correct comparison.")

    return EvalContext(keys, dense_curves, true_mat, allowed_indices, xrd_mean, xrd_std, device)


def eval_checkpoint(path, input_type, ctx, verbose=True):
    """Load one checkpoint and score it against ctx's selection slice.
    Returns (epoch, val_loss, sims) where sims is a list of per-entry cosine
    similarities (len == len(ctx.keys)); raises if path doesn't exist."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    epoch = ckpt.get("epoch")
    val_loss = ckpt.get("val_loss")
    val_loss = float(val_loss) if val_loss is not None else None
    if verbose:
        print(f"  {path}  (epoch={epoch}, val_loss={val_loss}, input_type={input_type})")

    model = XRDTransformerEncoder(
        **MODEL_KW,
        input_type=input_type,
        max_peaks=MAX_PEAKS,
        use_intensity=True,
        allowed_indices=ctx.allowed_indices,
    ).to(ctx.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    with torch.no_grad():
        if input_type == "dense":
            if ctx.xrd_mean is not None:
                x = (ctx.dense_curves - ctx.xrd_mean) / (ctx.xrd_std + 1e-8)
            else:
                x = (ctx.dense_curves - ctx.dense_curves.mean()) / (ctx.dense_curves.std() + 1e-8)
            data = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(ctx.device)
        else:
            peak_lists = [pad_peaks(_extract_peaks_via_find_peaks(c, MAX_PEAKS)) for c in ctx.dense_curves]
            data = torch.tensor(peak_lists, dtype=torch.float32).to(ctx.device)
        pred = model(data).cpu().numpy()  # (N, 10, 35)

    sims = [cos_sim_flat(pred[i], ctx.true_mat[i]) for i in range(len(ctx.keys))]

    del model
    if ctx.device == "cuda":
        torch.cuda.empty_cache()

    return epoch, val_loss, sims


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoints", nargs="*", default=list(CHECKPOINTS.keys()),
                   choices=list(CHECKPOINTS.keys()))
    p.add_argument("--extra-checkpoint", nargs=3, action="append", default=[],
                   metavar=("NAME", "PATH", "INPUT_TYPE"),
                   help="Check an arbitrary checkpoint (e.g. a snapshot) not in the fixed "
                        "CHECKPOINTS table -- INPUT_TYPE is 'dense' or 'sparse'. Repeatable.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    for name, path, input_type in args.extra_checkpoint:
        CHECKPOINTS[name] = dict(path=path, input_type=input_type)
        if name not in args.checkpoints:
            args.checkpoints.append(name)

    print(f"device: {args.device}")
    ctx = load_context(device=args.device)

    results = {}
    for name in args.checkpoints:
        spec = CHECKPOINTS[name]
        if not os.path.exists(spec["path"]):
            print(f"\n[{name}] checkpoint not found: {spec['path']} -- skipping")
            continue
        print(f"\n[{name}]")
        epoch, val_loss, sims = eval_checkpoint(spec["path"], spec["input_type"], ctx)
        results[name] = sims
        print(f"  n={len(sims)}  mean={np.mean(sims):.4f}  median={np.median(sims):.4f}"
              f"  min={np.min(sims):.4f}  max={np.max(sims):.4f}")

    print("\n── summary (cosine similarity, pred vs true bispec, selection slice) ──")
    print(f"{'checkpoint':<16} {'mean':>8} {'median':>8} {'min':>8}  n")
    print("-" * 50)
    for name, sims in results.items():
        print(f"{name:<16} {np.mean(sims):>8.4f} {np.median(sims):>8.4f} {np.min(sims):>8.4f}  {len(sims)}")

    out_path = f"{_ROOT}/eval_results/opxrd_cnrs/selection_cosine_sims.pkl"
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump({"keys": ctx.keys, "results": results}, f)
    print(f"\nSaved per-entry cosine similarities -> {out_path}")


if __name__ == "__main__":
    main()
