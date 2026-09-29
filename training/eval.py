#!/usr/bin/env python3
"""Evaluation script for XRD Transformer model.

Usage (direct prediction):
    python eval.py \
        --model_path /path/to/model.pt \
        --data_config ./conf/data/mp20_direct.yaml \
        --model_config ./conf/model/transformerdirect.yaml \
        --train_config ./conf/train/default_direct.yaml \
        --prediction_type direct \
        --output_path /path/to/results/test_results.pkl

Usage (bispectrum prediction):
    python eval.py \
        --model_path /path/to/model.pt \
        --data_config ./conf/data/mp20.yaml \
        --model_config ./conf/model/transformerbispec.yaml \
        --train_config ./conf/train/default.yaml \
        --prediction_type bispectrum \
        --output_path /path/to/results/test_results.pkl
"""

import argparse
import os
import pickle

import numpy as np
import torch
from omegaconf import OmegaConf
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate XRD Transformer model on test set")
    parser.add_argument("--model_path", required=True,
                        help="Path to model checkpoint (.pt file)")
    parser.add_argument("--data_config", required=True,
                        help="Path to data config yaml (e.g. conf/data/mp20_direct.yaml)")
    parser.add_argument("--model_config", required=True,
                        help="Path to model config yaml (e.g. conf/model/transformerdirect.yaml)")
    parser.add_argument("--train_config", required=True,
                        help="Path to train config yaml (e.g. conf/train/default_direct.yaml)")
    parser.add_argument("--prediction_type", choices=["direct", "bispectrum"], required=True,
                        help="'direct' predicts lattice params (out_dim=(1,6)); "
                             "'bispectrum' predicts bispectrum coefficients (out_dim=(10,35))")
    parser.add_argument("--output_path", required=True,
                        help="Path to save evaluation results as a .pkl file")
    parser.add_argument("--train_bispec_file", default="train_data_prim_lat.pkl",
                        help="Training bispectrum/lattice pickle filename in DATA_DIR "
                             "(default: train_data_prim_lat.pkl)")
    parser.add_argument("--test_bispec_file", default="test_data_prim_lat.pkl",
                        help="Test bispectrum/lattice pickle filename in DATA_DIR "
                             "(default: test_data_prim_lat.pkl)")
    parser.add_argument("--device", default="cuda",
                        help="Device to run inference on (default: cuda)")
    return parser.parse_args()


def main():
    args = parse_args()

    # ------------------------------------------------------------------ config
    data_cfg = OmegaConf.load(args.data_config)
    model_cfg = OmegaConf.load(args.model_config)
    train_cfg = OmegaConf.load(args.train_config)
    config = OmegaConf.create({"data": data_cfg, "model": model_cfg, "train": train_cfg})
    print("Config:")
    print(OmegaConf.to_yaml(config))

    # ---------------------------------------------------------- training data
    # (needed for normalization stats and allowed_indices before model build)
    from model import XRDTransformerEncoder
    from train import align_dicts, prepare_data, prepare_data_direct

    print("Loading training data for normalization stats...")
    sim_xrd_train = pickle.load(
        open(os.path.join(config.data.DATA_DIR, config.data.xrd_train_file), "rb")
    )
    bispec_train = pickle.load(
        open(os.path.join(config.data.DATA_DIR, config.data.target_train_file), "rb")
    )
    train_ids, sim_xrd_train, bispec_train = align_dicts(sim_xrd_train, bispec_train)

    xrd_array = np.stack(sim_xrd_train).astype(np.float32)
    xrd_train_tensor = torch.from_numpy(xrd_array).unsqueeze(1)
    xrd_mean = xrd_train_tensor.mean()
    xrd_std = xrd_train_tensor.std()

    if isinstance(bispec_train[0], torch.Tensor):
        target_train = torch.stack(bispec_train).float()
    else:
        target_train = torch.tensor(np.array(bispec_train))
    lattice_means = target_train.mean(dim=0)  # keep on CPU for prepare_data_direct
    lattice_stds = target_train.std(dim=0)

    # compute allowed_indices before building the model
    allowed_indices = None
    if args.prediction_type == "bispectrum" and getattr(config.train, "mask_zeros_in_loss", False):
        loss_mask = torch.any(
            torch.abs(target_train) > config.train.mask_loss_threshold, dim=0
        )
        allowed_indices = torch.nonzero(loss_mask, as_tuple=False).tolist()
        print(f"Using bispectrum loss mask with {len(allowed_indices)} allowed indices")

    # ------------------------------------------------------------------ model
    out_dim = (1, 6) if args.prediction_type == "direct" else (10, 35)

    model_dict = torch.load(args.model_path, map_location=args.device, weights_only=False)["model_state_dict"]

    model = XRDTransformerEncoder(
        out_dim=out_dim,
        d_model=config.model.d_model,
        h_dim=config.model.h_dim,
        transformer_proc=config.model.transformer_proc,
        n_self_layer=config.model.n_self_layer,
        n_head=config.model.n_head,
        attn_pdrop=config.model.attn_pdrop,
        resid_pdrop=config.model.resid_pdrop,
        allowed_indices=allowed_indices,
    ).to(args.device)

    model.load_state_dict(model_dict)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    # ------------------------------------------------------------ test data
    print("Loading test data...")
    sim_xrd_test = pickle.load(
        open(os.path.join(config.data.DATA_DIR, config.data.xrd_test_file), "rb")
    )
    bispec_test = pickle.load(
        open(os.path.join(config.data.DATA_DIR, args.test_bispec_file), "rb")
    )
    test_keys, sim_xrd_test, bispec_test = align_dicts(sim_xrd_test, bispec_test)

    # --------------------------------------------------------- data loader
    if args.prediction_type == "direct":
        test_loader = prepare_data_direct(
            sim_xrd_test, bispec_test, test_keys,
            batch_size=config.data.batch_size,
            indices=config.data.indices,
            d_model=config.model.d_model,
            is_normalized=config.data.is_normalized,
            shuffle=False,
            structural_mask=None,
            xrd_mean=xrd_mean,
            xrd_std=xrd_std,
            epsilon=False,
            lat_mean=lattice_means,
            lat_std=lattice_stds,
        )
    else:
        test_loader = prepare_data(
            sim_xrd_test, bispec_test, test_keys,
            batch_size=config.data.batch_size,
            indices=config.data.indices,
            d_model=config.model.d_model,
            is_normalized=config.data.is_normalized,
            shuffle=False,
            structural_mask=None,
            xrd_mean=xrd_mean,
            xrd_std=xrd_std,
        )

    # --------------------------------------------------------- evaluation
    print(f"Running evaluation ({args.prediction_type} mode)...")
    all_outputs, all_targets, all_material_ids = [], [], []

    with torch.no_grad():
        sample_index = 0
        for data, target, mat_id in tqdm(test_loader):
            batch_size = target.size(0)
            batch_keys = test_keys[sample_index: sample_index + batch_size]
            sample_index += batch_size

            data = data.to(args.device).float()
            target = target.to(args.device).float()

            output = model(data)

            if args.prediction_type == "direct":
                # un-normalize model output back to raw lattice-param space
                unnorm_output = output * (lattice_stds.to(args.device) + 1e-8) + lattice_means.to(args.device)
                unnorm_target = target * (lattice_stds.to(args.device) + 1e-8) + lattice_means.to(args.device)
            else:
                unnorm_output = output
                unnorm_target = target

            all_outputs.append(unnorm_output.cpu())
            all_targets.append(unnorm_target.cpu())
            all_material_ids.extend(batch_keys)

    test_dict = {
        "predictions": torch.cat(all_outputs),
        "targets": torch.cat(all_targets),
        "material_ids": all_material_ids,
    }

    # --------------------------------------------------------- save results
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, "wb") as f:
        pickle.dump(test_dict, f)
    print(f"Saved predictions to {args.output_path}")
    print(f"  predictions shape : {test_dict['predictions'].shape}")
    print(f"  targets shape     : {test_dict['targets'].shape}")
    print(f"  num samples       : {len(test_dict['material_ids'])}")


if __name__ == "__main__":
    main()
