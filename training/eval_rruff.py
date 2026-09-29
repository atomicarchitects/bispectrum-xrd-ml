#!/usr/bin/env python3
"""Evaluate a trained XRD direct-prediction model on the RRUFF dataset.

The RRUFF data (rruff_data_dict.pkl) contains XRDStructure objects with:
  - .pattern   : DataFrame with columns [2theta, intensity, ...]
  - .metadata  : DataFrame with columns [a, b, c, alpha, beta, gamma, mineral_name, ...]

XRD profiles are simulated from the RRUFF peak lists using the same pseudo-Voigt
pipeline used during training. Normalization stats (mean/std) are computed from
the mp20 training set so the model sees the same input distribution.

Usage:
    python eval_rruff.py \
        --model_path /path/to/best_model_mp20_direct.pt \
        --train_data_dir /path/to/mp20/data \
        --rruff_pkl /path/to/rruff_data_dict.pkl \
        --model_config conf/model/transformerdirect.yaml \
        --output_path /path/to/rruff_mp20_direct.pkl
"""

import argparse
import os
import pickle
import sys
import types

import numpy as np
import torch
import torch.nn as nn
from omegaconf import OmegaConf
from tqdm import tqdm

# ---------------------------------------------------------------------------
# Register XRDStructure so the RRUFF pkl can be unpickled
# ---------------------------------------------------------------------------
import pandas as pd
from dataclasses import dataclass

mod = types.ModuleType('data_prep')

@dataclass
class XRDStructure:
    metadata: pd.DataFrame
    atoms: pd.DataFrame
    pattern: pd.DataFrame

mod.XRDStructure = XRDStructure
sys.modules['data_prep'] = mod

# ---------------------------------------------------------------------------
# Pseudo-Voigt XRD simulation (same as training pipeline)
# ---------------------------------------------------------------------------

def caglioti_fwhm(theta, U, V, W):
    rad_theta = np.radians(theta / 2)
    return (U * np.tan(rad_theta)**2 + V * np.tan(rad_theta) + W) ** 0.5


def pseudo_voigt(x, center, amplitude, U, V, W, eta):
    fwhm = caglioti_fwhm(center, U, V, W)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    lorentzian = amplitude * (fwhm**2 / ((x - center)**2 + fwhm**2))
    gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    return eta * lorentzian + (1 - eta) * gaussian


def simulate_pv_xrd(pattern_df, U=0.1, V=0.1, W=0.1, eta=0.0):
    """Simulate pseudo-Voigt XRD profile from a RRUFF pattern DataFrame."""
    x = np.arange(5, 90, 0.010)  # 8500 points, same as training
    total = np.zeros_like(x)
    for _, row in pattern_df.iterrows():
        total += pseudo_voigt(x, row['2theta'], row['intensity'], U, V, W, eta)
    mx = total.max()
    if mx > 0:
        total = total / mx
    return total


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XRD model on RRUFF dataset")
    parser.add_argument("--model_path", required=True,
                        help="Path to model checkpoint (.pt)")
    parser.add_argument("--train_data_dir", required=True,
                        help="Directory containing mp20 training data "
                             "(needs train_data_sim_xrd.pkl and train_data_prim_lat.pkl)")
    parser.add_argument("--rruff_pkl", required=True,
                        help="Path to rruff_data_dict.pkl")
    parser.add_argument("--model_config", default="conf/model/transformerdirect.yaml",
                        help="Path to model config yaml")
    parser.add_argument("--train_config", default="conf/train/default_direct.yaml",
                        help="Path to train config yaml (used to reconstruct allowed_indices for bispectrum models)")
    parser.add_argument("--output_path", required=True,
                        help="Path to save results pkl")
    parser.add_argument("--prediction_type", choices=["direct", "bispectrum"], default="direct",
                        help="'direct' predicts lattice params (unnormalized on output); "
                             "'bispectrum' predicts bispectrum coefficients (raw output)")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--train_xrd_file", default="train_data_sim_xrd.pkl")
    parser.add_argument("--train_lat_file", default="train_data_prim_lat.pkl")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # ------------------------------------------------------------------ model
    model_cfg = OmegaConf.load(args.model_config)
    train_cfg = OmegaConf.load(args.train_config)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import XRDTransformerEncoder

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    # Remap old key naming (linear_out -> head) for plain linear checkpoints
    if "linear_out.weight" in state_dict:
        state_dict["head.weight"] = state_dict.pop("linear_out.weight")

    # Infer whether the checkpoint used StructuredBispectrumHead or plain linear
    # by inspecting the state_dict keys — match eval.py which always uses allowed_indices=None
    # unless the checkpoint actually has the structured head keys.
    allowed_indices = None
    if "head.linear.weight" in state_dict:
        # checkpoint was saved with StructuredBispectrumHead — reconstruct allowed_indices
        lat_train = pickle.load(
            open(os.path.join(args.train_data_dir, args.train_lat_file), "rb"))
        if isinstance(lat_train, dict):
            lat_train = list(lat_train.values())
        all_targets = torch.tensor(np.array(lat_train, dtype=np.float32))
        loss_mask = torch.any(torch.abs(all_targets) > train_cfg.mask_loss_threshold, dim=0)
        allowed_indices = torch.nonzero(loss_mask, as_tuple=False).tolist()
        print(f"Using StructuredBispectrumHead with {len(allowed_indices)} allowed indices")

    model = XRDTransformerEncoder(
        out_dim=model_cfg.out_dim,
        d_model=model_cfg.d_model,
        h_dim=model_cfg.h_dim,
        transformer_proc=model_cfg.transformer_proc,
        n_self_layer=model_cfg.n_self_layer,
        n_head=model_cfg.n_head,
        attn_pdrop=model_cfg.attn_pdrop,
        resid_pdrop=model_cfg.resid_pdrop,
        allowed_indices=allowed_indices,
    ).to(device)

    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # ------------------------------------------------ training normalization stats
    print("Computing normalization stats from training data...")
    sim_xrd_train = pickle.load(
        open(os.path.join(args.train_data_dir, args.train_xrd_file), "rb"))

    # XRD mean/std: needed for both direct and bispectrum
    if isinstance(sim_xrd_train, dict):
        sim_xrd_train = list(sim_xrd_train.values())
    xrd_train_tensor = torch.FloatTensor(np.array(sim_xrd_train)).unsqueeze(1)
    xrd_mean = xrd_train_tensor.mean()
    xrd_std  = xrd_train_tensor.std()
    del xrd_train_tensor
    print(f"  XRD  mean={xrd_mean:.4f}  std={xrd_std:.4f}")

    # Lattice mean/std: only needed for direct prediction (bispectrum targets are not normalized)
    lat_mean = lat_std = None
    if args.prediction_type == "direct":
        lat_train = pickle.load(
            open(os.path.join(args.train_data_dir, args.train_lat_file), "rb"))
        if isinstance(lat_train, dict):
            lat_train = list(lat_train.values())
        lat_train_tensor = torch.tensor(np.array(lat_train, dtype=np.float32))
        lat_mean = lat_train_tensor.mean(dim=0).to(device)
        lat_std  = lat_train_tensor.std(dim=0).to(device)
        print(f"  Lattice means: {lat_mean.cpu().numpy()}")
        print(f"  Lattice stds:  {lat_std.cpu().numpy()}")

    # ---------------------------------------------------------------- RRUFF data
    print(f"Loading RRUFF data from {args.rruff_pkl} ...")
    rruff_dict = pickle.load(open(args.rruff_pkl, "rb"))
    print(f"  {len(rruff_dict)} entries")

    mineral_names, xrd_list, lat_list = [], [], []
    skipped = 0
    for key, entry in tqdm(rruff_dict.items(), desc="Simulating XRD"):
        try:
            meta = entry.metadata.iloc[0]
            a, b, c = float(meta['a']), float(meta['b']), float(meta['c'])
            al, be, ga = float(meta['alpha']), float(meta['beta']), float(meta['gamma'])
            if any(np.isnan(v) for v in [a, b, c, al, be, ga]):
                skipped += 1
                continue
            xrd = simulate_pv_xrd(entry.pattern)
            mineral_names.append(key)
            xrd_list.append(xrd)
            lat_list.append([a, b, c, al, be, ga])
        except Exception as e:
            skipped += 1
            continue

    print(f"  {len(xrd_list)} usable entries ({skipped} skipped)")

    # ------------------------------------------------------------ inference
    xrd_tensor = torch.FloatTensor(np.array(xrd_list)).unsqueeze(1)
    xrd_tensor = (xrd_tensor - xrd_mean) / (xrd_std + 1e-8)
    lat_tensor = torch.FloatTensor(np.array(lat_list))

    all_preds = []
    with torch.no_grad():
        for i in tqdm(range(0, len(xrd_tensor), args.batch_size), desc="Inference"):
            batch = xrd_tensor[i:i+args.batch_size].to(device)
            out = model(batch)
            if args.prediction_type == "direct":
                out = out * (lat_std + 1e-8) + lat_mean  # un-normalize
            all_preds.append(out.cpu())

    predictions = torch.cat(all_preds, dim=0)  # (N, 6) or (N, bispectrum_dim)

    # ------------------------------------------------------------ metrics / save
    results = {
        "predictions": predictions,
        "mineral_names": mineral_names,
        "prediction_type": args.prediction_type,
    }

    if args.prediction_type == "direct":
        targets = lat_tensor  # (N, 6)
        mae = torch.mean(torch.abs(predictions - targets), dim=0)
        param_names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        print("\nMAE per lattice parameter:")
        for name, val in zip(param_names, mae.tolist()):
            print(f"  {name:>6}: {val:.4f}")
        print(f"  {'mean':>6}: {mae.mean().item():.4f}")
        results["targets"] = targets
        results["mae_per_param"] = dict(zip(param_names, mae.tolist()))
        results["mae_mean"] = mae.mean().item()
    else:
        print(f"\nBispectrum predictions shape: {predictions.shape}")
        results["targets"] = lat_tensor  # (N, 6) true lattice params [a,b,c,alpha,beta,gamma]
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {args.output_path}")
    print(f"  predictions shape: {predictions.shape}")
    print(f"  num samples: {len(mineral_names)}")


if __name__ == "__main__":
    main()
