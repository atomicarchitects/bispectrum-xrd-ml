"""
Evaluate trained models on the RRUFF alpha-diffract eval set built from
measured XY_Processed patterns + rruff_alpha_diffract CIF primitive lattices.

Usage (bispectrum):
    python eval_rruff_alpha.py \
        --model_path /path/to/best_model.pt \
        --train_data_dir /path/to/train_data \
        --train_xrd_file train_sim_xrd.pkl \
        --train_lat_file train_bispec.pkl \
        --prediction_type bispectrum \
        --model_config conf/model/transformerbispec.yaml \
        --train_config conf/train/default.yaml \
        --output_path /path/to/results.pkl

Usage (direct):
    python eval_rruff_alpha.py \
        --model_path /path/to/best_model.pt \
        --train_data_dir /path/to/train_data \
        --train_xrd_file train_sim_xrd.pkl \
        --train_lat_file train_data_prim_lat.pkl \
        --prediction_type direct \
        --model_config conf/model/transformerdirect.yaml \
        --train_config conf/train/default_direct.yaml \
        --output_path /path/to/results.pkl
"""

import argparse, os, pickle, sys
import numpy as np
import torch
from omegaconf import OmegaConf

RRUFF_ALPHA_DIR = os.environ.get("POWDERXRD_DATA_ROOT", "./data") + '/rruff_alpha_eval'
XRD_PKL = os.path.join(RRUFF_ALPHA_DIR, 'rruff_alpha_xrd.pkl')
LAT_PKL = os.path.join(RRUFF_ALPHA_DIR, 'rruff_alpha_prim_lat.pkl')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',      required=True)
    p.add_argument('--train_data_dir',  required=True)
    p.add_argument('--train_xrd_file',  default='train_sim_xrd.pkl')
    p.add_argument('--train_lat_file',  default='train_bispec.pkl')
    p.add_argument('--prediction_type', choices=['direct', 'bispectrum'], required=True)
    p.add_argument('--model_config',    default='conf/model/transformerbispec.yaml')
    p.add_argument('--train_config',    default='conf/train/default.yaml')
    p.add_argument('--output_path',     required=True)
    p.add_argument('--batch_size',      type=int, default=64)
    p.add_argument('--device',          default='cuda')
    p.add_argument('--eval_xrd_pkl',    default=None,
                   help='Override default RRUFF alpha XRD pkl with a custom eval set')
    p.add_argument('--eval_lat_pkl',    default=None,
                   help='Override default RRUFF alpha lattice pkl with a custom eval set')
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model_cfg = OmegaConf.load(args.model_config)
    train_cfg = OmegaConf.load(args.train_config)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import XRDTransformerEncoder

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    if 'linear_out.weight' in state_dict:
        state_dict['head.weight'] = state_dict.pop('linear_out.weight')

    allowed_indices = None
    if 'head.linear.weight' in state_dict:
        lat_train = pickle.load(open(os.path.join(args.train_data_dir, args.train_lat_file), 'rb'))
        if isinstance(lat_train, dict):
            lat_train = list(lat_train.values())
        if isinstance(lat_train[0], torch.Tensor):
            all_targets = torch.stack(lat_train).float()
        else:
            all_targets = torch.tensor(np.array(lat_train, dtype=np.float32))
        loss_mask = torch.any(torch.abs(all_targets) > train_cfg.mask_loss_threshold, dim=0)
        allowed_indices = torch.nonzero(loss_mask, as_tuple=False).tolist()
        print(f'StructuredBispectrumHead with {len(allowed_indices)} allowed indices')

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
    print(f'Loaded model from {args.model_path}')

    # Training normalization stats
    print('Computing XRD normalization stats from training data...')
    sim_xrd_train = pickle.load(open(os.path.join(args.train_data_dir, args.train_xrd_file), 'rb'))
    if isinstance(sim_xrd_train, dict):
        sim_xrd_train = list(sim_xrd_train.values())
    xrd_array = np.stack(sim_xrd_train).astype(np.float32)
    xrd_train_tensor = torch.from_numpy(xrd_array).unsqueeze(1)
    xrd_mean = xrd_train_tensor.mean()
    xrd_std  = xrd_train_tensor.std()
    del xrd_train_tensor
    print(f'  XRD mean={xrd_mean:.4f}  std={xrd_std:.4f}')

    lat_mean = lat_std = None
    if args.prediction_type == 'direct':
        lat_train = pickle.load(open(os.path.join(args.train_data_dir, args.train_lat_file), 'rb'))
        if isinstance(lat_train, dict):
            lat_train = list(lat_train.values())
        if isinstance(lat_train[0], torch.Tensor):
            lat_tensor = torch.stack(lat_train).float()
        else:
            lat_tensor = torch.tensor(np.array(lat_train, dtype=np.float32))
        lat_mean = lat_tensor.mean(dim=0).to(device)
        lat_std  = lat_tensor.std(dim=0).to(device)
        print(f'  Lattice means: {lat_mean.cpu().numpy()}')

    # Load eval data (custom paths override RRUFF alpha defaults)
    xrd_pkl = args.eval_xrd_pkl or XRD_PKL
    lat_pkl = args.eval_lat_pkl or LAT_PKL
    print(f'Loading eval XRD from  {xrd_pkl}...')
    print(f'Loading eval lats from {lat_pkl}...')
    xrd_dict = pickle.load(open(xrd_pkl, 'rb'))
    lat_dict = pickle.load(open(lat_pkl, 'rb'))

    # align keys
    keys = sorted(set(xrd_dict) & set(lat_dict))
    print(f'  {len(keys)} entries')

    xrd_array = np.stack([xrd_dict[k] for k in keys]).astype(np.float32)
    xrd_tensor = torch.from_numpy(xrd_array).unsqueeze(1)
    xrd_tensor = (xrd_tensor - xrd_mean) / (xrd_std + 1e-8)

    targets = torch.tensor(
        np.stack([lat_dict[k] for k in keys]).astype(np.float32)
    )  # (N, 6) primitive lattice params

    # Inference
    all_preds = []
    with torch.no_grad():
        for i in range(0, len(xrd_tensor), args.batch_size):
            batch = xrd_tensor[i:i+args.batch_size].to(device)
            out = model(batch)
            if args.prediction_type == 'direct':
                out = out * (lat_std + 1e-8) + lat_mean
            all_preds.append(out.cpu())
    predictions = torch.cat(all_preds, dim=0)
    print(f'Predictions shape: {predictions.shape}')

    if args.prediction_type == 'direct':
        mae  = torch.abs(predictions - targets).mean(dim=0)
        mape = (torch.abs(predictions - targets) / (torch.abs(targets) + 1e-8)).mean(dim=0) * 100
        names = ['a', 'b', 'c', 'alpha', 'beta', 'gamma']
        print('\nMAE / MAPE per lattice parameter:')
        for n, m, p in zip(names, mae.tolist(), mape.tolist()):
            print(f'  {n:>6}: MAE={m:.4f}  MAPE={p:.2f}%')
        print(f'  {"mean":>6}: MAE={mae.mean():.4f}  MAPE={mape.mean():.2f}%')

    results = {
        'predictions':     predictions,
        'targets':         targets,
        'material_ids':    keys,
        'prediction_type': args.prediction_type,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved to {args.output_path}  (N={len(keys)})')


if __name__ == '__main__':
    main()
