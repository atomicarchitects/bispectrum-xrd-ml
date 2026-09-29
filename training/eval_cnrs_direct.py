"""
Evaluate a transformerdirect_sparse checkpoint (direct lattice-parameter
regression, no bispectrum/inversion) on the opXRD CNRS selection slice.

Unlike the bispec models, this needs no L-BFGS inversion step -- the model
outputs (a,b,c,alpha,beta,gamma) directly. The only wrinkle: is_normalized:
true means the model was trained on (raw - lat_mean)/lat_std targets, and
lat_mean/lat_std (computed from the training set at train.py runtime) are
NOT saved in the checkpoint -- this script recomputes them exactly as
train.py does (same align_dicts intersection of the xrd_train_file and
target_train_file keys) before un-normalizing predictions.

Usage (full-MP model, selection slice):
    python eval_cnrs_direct.py --checkpoint-path /path/to/epoch18_....pt

Usage (MP-20 model, on exactly the patterns of an inversion-input pkl, writing results in the
same h5 format as run_alg_inversion.py so the same scoring applies):
    python eval_cnrs_direct.py --checkpoint-path /path/to/model.pt \\
        --data-dir $POWDERXRD_DATA_ROOT/mp20_aug/all_aug \\
        --xrd-train-file train_peaklist_bgmix.pkl \\
        --target-train-file all_aug_train_data_prim_lat.pkl \\
        --ids-from-pkl $POWDERXRD_DATA_ROOT/eval_results/opxrd_cnrs/inversion_inputs/<name>.pkl \\
        --out-h5 $POWDERXRD_DATA_ROOT/eval_results/opxrd_cnrs/inversion_inputs/<name>_direct_inverted.h5
"""
import argparse
import os
import pickle
import sys

import numpy as np
import torch
from pymatgen.core.lattice import Lattice

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp_full"))
from model import XRDTransformerEncoder
from generate_augmented_data import _extract_peaks_via_find_peaks
from eval_opxrd_cnrs_selection_cosine import pad_peaks, MAX_PEAKS

_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
CNRS_DIR = f"{_ROOT}/opxrd_cnrs_eval"
DATA_DIR = f"{_ROOT}/mpfull_aug_cag_strain_texture"

MODEL_KW = dict(out_dim=(1, 6), d_model=256, h_dim=512, transformer_proc="tokenization",
                 n_self_layer=12, n_head=8, attn_pdrop=0.2, resid_pdrop=0.2)


def niggli_params(mat_or_vec):
    if len(np.asarray(mat_or_vec).shape) == 1:
        a, b, c, al, be, ga = mat_or_vec
        lat = Lattice.from_parameters(a, b, c, al, be, ga)
    else:
        lat = Lattice(np.asarray(mat_or_vec).reshape(3, 3))
    nig = lat.get_niggli_reduced_lattice()
    return nig.a, nig.b, nig.c, nig.alpha, nig.beta, nig.gamma


def recompute_lat_norm_stats(xrd_train_file, data_dir=DATA_DIR, target_train_file="train_prim_lat_combined.pkl"):
    """Mirror train.py exactly: align xrd_train_file's keys against
    the target file, then mean/std over the aligned subset."""
    with open(os.path.join(data_dir, xrd_train_file), "rb") as f:
        xrd_train = pickle.load(f)
    with open(os.path.join(data_dir, target_train_file), "rb") as f:
        target_train = pickle.load(f)
    common_keys = sorted(set(xrd_train.keys()) & set(target_train.keys()))
    target_list = [target_train[k] for k in common_keys]
    lattice_tensor_raw = torch.tensor(np.array(target_list), dtype=torch.float32)
    lat_mean = lattice_tensor_raw.mean(dim=0)
    lat_std = lattice_tensor_raw.std(dim=0)
    print(f"Recomputed from {len(common_keys)} training examples ({xrd_train_file}):")
    print(f"  lat_mean = {lat_mean.numpy()}")
    print(f"  lat_std  = {lat_std.numpy()}")
    return lat_mean, lat_std


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint-path", required=True)
    p.add_argument("--xrd-train-file", default="train_peaklist_bgmix_combined.pkl",
                   help="Must match the data config's xrd_train_file -- determines "
                        "which training-set keys the lat_mean/lat_std were computed over.")
    p.add_argument("--data-dir", default=DATA_DIR,
                   help="Directory of the training data the checkpoint was trained on (default: the full-MP one).")
    p.add_argument("--target-train-file", default="train_prim_lat_combined.pkl",
                   help="Must match the data config's target_train_file.")
    p.add_argument("--input-type", default="sparse", choices=["sparse", "dense"],
                   help="'sparse' = peak-list model (default); 'dense' = full-pattern model.")
    p.add_argument("--norm-stats", default=None,
                   help="Dense models only: json with the training set's xrd_mean/xrd_std "
                        "(from compute_xrd_norm_stats.py).")
    p.add_argument("--slice", default="selection", choices=["selection", "reporting", "all"])
    p.add_argument("--ids-from-pkl", default=None,
                   help="Evaluate exactly the material_ids of this inversion-input pkl (implies --slice all).")
    p.add_argument("--out-h5", default=None,
                   help="Write per-pattern results in run_alg_inversion.py's h5 format (pred and true_lattice "
                        "as 3x3 real-space matrices), so the same scoring applies to direct and inverted models.")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    lat_mean, lat_std = recompute_lat_norm_stats(args.xrd_train_file, args.data_dir, args.target_train_file)

    slice_name = "all" if args.ids_from_pkl else args.slice
    suffix = "" if slice_name == "all" else f"_{slice_name}"
    dense = pickle.load(open(f"{CNRS_DIR}/opxrd_cnrs_xrd{suffix}.pkl", "rb"))
    lat_sel = pickle.load(open(f"{CNRS_DIR}/opxrd_cnrs_lat{suffix}.pkl", "rb"))
    keys = sorted(dense.keys())
    if args.ids_from_pkl:
        wanted = set(pickle.load(open(args.ids_from_pkl, "rb"))["material_ids"])
        keys = [k for k in keys if k in wanted]
        print(f"{len(keys)} of {len(wanted)} requested ids found")

    if args.input_type == "sparse":
        peak_lists = [pad_peaks(_extract_peaks_via_find_peaks(np.asarray(dense[k], dtype=np.float64), MAX_PEAKS))
                      for k in keys]
        data = torch.tensor(peak_lists, dtype=torch.float32).to(args.device)
    else:
        if not args.norm_stats:
            raise SystemExit("--input-type dense needs --norm-stats (the training set's xrd mean/std)")
        import json
        stats = json.load(open(args.norm_stats))
        curves = np.stack([np.asarray(dense[k], dtype=np.float64) for k in keys])
        x = (curves - stats["xrd_mean"]) / (stats["xrd_std"] + 1e-8)
        data = torch.tensor(x, dtype=torch.float32).unsqueeze(1).to(args.device)

    model = XRDTransformerEncoder(**MODEL_KW, input_type=args.input_type, max_peaks=MAX_PEAKS,
                                   use_intensity=True, allowed_indices=None).to(args.device)
    ckpt = torch.load(args.checkpoint_path, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    print(f"Loaded checkpoint: epoch={ckpt.get('epoch')} val_loss={ckpt.get('val_loss')}")

    with torch.no_grad():
        # chunked forward pass: a dense model on all patterns at once is very slow / memory-heavy on CPU
        chunks = []
        for i in range(0, len(data), 16):
            chunks.append(model(data[i:i + 16]).cpu())
            print(f"  forward pass {min(i + 16, len(data))}/{len(data)}", flush=True)
        pred_norm = torch.cat(chunks)  # (N, 1, 6) or (N, 6)
    pred_norm = pred_norm.reshape(len(keys), 6)
    pred_real = (pred_norm * lat_std + lat_mean).numpy()
    if args.out_h5:
        np.save(args.out_h5 + ".pred_real.npy", pred_real)  # safety copy: keys order = sorted ids
        with open(args.out_h5 + ".keys.txt", "w") as fk:
            fk.write("\n".join(keys) + "\n")

    h5 = None
    if args.out_h5:
        import h5py
        if os.path.exists(args.out_h5):
            raise FileExistsError(f"{args.out_h5} exists; refusing to overwrite")
        h5 = h5py.File(args.out_h5, "w")

    import signal

    def _alarm(signum, frame):
        raise TimeoutError("niggli reduction timed out")
    signal.signal(signal.SIGALRM, _alarm)

    len_apes, ang_apes = [], []
    for i, k in enumerate(keys):
        if i % 50 == 0:
            print(f"  scoring {i}/{len(keys)}", flush=True)
        true_raw = np.asarray(lat_sel[k], dtype=np.float64)
        if h5 is not None:
            try:
                g = h5.create_group(k)
                g.create_dataset("pred", data=Lattice.from_parameters(*[float(x) for x in pred_real[i]]).matrix)
                g.create_dataset("true_lattice", data=Lattice.from_parameters(*[float(x) for x in true_raw]).matrix)
            except Exception:
                if k in h5:
                    del h5[k]
        try:
            signal.alarm(10)
            pa, pb, pc, pal, pbe, pga = niggli_params(pred_real[i])
            ta, tb, tc, tal, tbe, tga = niggli_params(true_raw)
            signal.alarm(0)
        except Exception:
            signal.alarm(0)
            continue
        l = [abs(p - t) / t for p, t in [(pa, ta), (pb, tb), (pc, tc)] if t > 1e-6]
        a = [abs(p - t) / t for p, t in [(pal, tal), (pbe, tbe), (pga, tga)] if t > 1e-6]
        len_apes.append(100 * np.mean(l))
        ang_apes.append(100 * np.mean(a))

    if h5 is not None:
        print(f"wrote {len(h5.keys())} results -> {args.out_h5}")
        h5.close()
    n = len(len_apes)
    print(f"\nn={n}/{len(keys)}")
    print(f"length MAPE: {np.mean(len_apes):.2f}%  (median {np.median(len_apes):.2f}%)")
    print(f"angle MAPE:  {np.mean(ang_apes):.2f}%  (median {np.median(ang_apes):.2f}%)")


if __name__ == "__main__":
    main()
