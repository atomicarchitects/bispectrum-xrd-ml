"""
Run a bispectrum or direct model on the pre-computed 148 RRUFF-crystalyze XRD patterns.

Usage (bispectrum):
    python eval_rruff_crystalyze.py \
        --model_path <path>.pt \
        --train_data_dir <dir> \
        --train_xrd_file train_sim_xrd.pkl \
        --train_lat_file train_bispec.pkl \
        --prediction_type bispectrum \
        --output_path <out>.pkl

Usage (direct):
    python eval_rruff_crystalyze.py \
        --model_path <path>.pt \
        --train_data_dir <dir> \
        --train_xrd_file train_sim_xrd.pkl \
        --train_lat_file train_data_prim_lat.pkl \
        --prediction_type direct \
        --model_config conf/model/transformerdirect.yaml \
        --train_config conf/train/default_direct.yaml \
        --output_path <out>.pkl
"""

import argparse, os, pickle, sys
import numpy as np
import torch
from omegaconf import OmegaConf
import pandas as pd

_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
CRYSTALYZE_DIR = f'{_ROOT}/rruff_crystalyze'
XRD_PT   = f'{CRYSTALYZE_DIR}/test_pv_xrd.pt'
CSV_PATH = f'{CRYSTALYZE_DIR}/rruff_pxrd_entries_used_148_with_rruff_ids.csv'
REF_PKL  = f'{_ROOT}/eval_results/rruff/rruff_crystalyze_results/rruff_mp20_crystalyze_aug_bispec.pkl'

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path',     required=True)
    p.add_argument('--train_data_dir', required=True)
    p.add_argument('--train_xrd_file', default='train_sim_xrd.pkl')
    p.add_argument('--train_lat_file',  default='train_bispec.pkl')
    p.add_argument('--prediction_type', choices=['bispectrum', 'direct'], default='bispectrum')
    p.add_argument('--input_type', choices=['dense', 'sparse'], default='dense',
                   help="'sparse' = peak-list model: peaks are extracted from each curve with the same "
                        "pipeline used to build its training data; no normalization statistics are needed. "
                        "Use a sparse --model_config (e.g. conf/model/transformerbispec_sparse.yaml).")
    p.add_argument('--model_config',    default='conf/model/transformerbispec.yaml')
    p.add_argument('--train_config',    default='conf/train/default.yaml')
    p.add_argument('--output_path',     required=True)
    p.add_argument('--batch_size',      type=int, default=64)
    p.add_argument('--device',          default='cuda')
    return p.parse_args()

def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    # Load model
    model_cfg = OmegaConf.load(args.model_config)
    train_cfg = OmegaConf.load(args.train_config)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import XRDTransformerEncoder

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    allowed_indices = None
    if 'head.linear.weight' in state_dict:
        lat_train = pickle.load(open(os.path.join(args.train_data_dir, args.train_lat_file), 'rb'))
        if isinstance(lat_train, dict):
            lat_train = list(lat_train.values())
        all_targets = torch.tensor(np.array(lat_train, dtype=np.float32))
        loss_mask = torch.any(torch.abs(all_targets) > train_cfg.mask_loss_threshold, dim=0)
        allowed_indices = torch.nonzero(loss_mask, as_tuple=False).tolist()
        print(f'StructuredBispectrumHead with {len(allowed_indices)} allowed indices')

    sparse_kw = {}
    if args.input_type == 'sparse':
        sparse_kw = dict(input_type='sparse', max_peaks=model_cfg.max_peaks,
                         use_intensity=model_cfg.get('use_intensity', True))
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
        **sparse_kw,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'Loaded model from {args.model_path}')

    # Training normalization stats
    xrd_mean = xrd_std = None
    if args.input_type == 'dense':
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

    # Load the 148 pre-computed XRD patterns and align to CSV ordering
    df = pd.read_csv(CSV_PATH)
    xrd_dict = torch.load(XRD_PT, weights_only=False)  # {mineral_name: [1, 8500]}

    # Load reference pkl for targets and mineral_names ordering
    with open(REF_PKL, 'rb') as f:
        ref = pickle.load(f)
    mineral_names = list(ref['mineral_names'])  # e.g. ['Ferberite__R040044', ...]
    targets = ref['targets']                    # [148, 6]

    # Build XRD tensor in the same order as mineral_names
    # mineral_names use 'Ferberite__R040044', xrd_dict uses 'Ferberite'
    xrd_list = []
    for mn in mineral_names:
        short = mn.split('__')[0]
        if short not in xrd_dict:
            print(f'  [WARN] {short} not in xrd_dict, using zeros')
            xrd_list.append(torch.zeros(1, 8500))
        else:
            xrd_list.append(xrd_dict[short])

    xrd_tensor = torch.cat(xrd_list, dim=0).unsqueeze(1)  # [148, 1, 8500]
    if args.input_type == 'dense':
        xrd_tensor = (xrd_tensor - xrd_mean) / (xrd_std + 1e-8)
    else:
        sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mp_full'))
        from generate_augmented_data import _extract_peaks_via_find_peaks
        from eval_opxrd_cnrs_selection_cosine import pad_peaks
        curves = xrd_tensor.squeeze(1).numpy().astype(np.float64)
        peak_lists = [pad_peaks(_extract_peaks_via_find_peaks(c, model_cfg.max_peaks)) for c in curves]
        xrd_tensor = torch.tensor(peak_lists, dtype=torch.float32)
    print(f'XRD tensor shape: {xrd_tensor.shape}')

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

    results = {
        'predictions':     predictions,
        'mineral_names':   mineral_names,
        'prediction_type': args.prediction_type,
        'targets':         targets,
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved to {args.output_path}')

if __name__ == '__main__':
    main()
