"""
Evaluate a sparse (peak-list) bispectrum model on the RRUFF alpha-diffract
eval set.  Dense patterns are converted to peak lists with the same
extraction pipeline used during training.

Usage:
    python eval_rruff_alpha_peaklist.py \
        --model_path $R/mpfull_aug_final_peaklist_bgmix_wlfix/transformerbispec/models/nlayers12_nhead8_dm256_hd512_lossL1_smallvalsFalse_masklossTrue.pt \
        --train_data_dir $R/mpfull_aug_cag_strain_texture_bgmix_wlfix \
        --train_lat_file train_bispec_combined.pkl \
        --model_config conf/model/transformerbispec_sparse.yaml \
        --train_config conf/train/default.yaml \
        --output_path $R/eval_results/rruff/rruff_alpha_mpfull_peaklist_bgmix_cnrswlfix_bispec.pkl
"""

import argparse, os, pickle, sys
import numpy as np
import torch
from omegaconf import OmegaConf

_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
RRUFF_ALPHA_DIR = f'{_ROOT}/rruff_alpha_eval'
XRD_PKL = os.path.join(RRUFF_ALPHA_DIR, 'rruff_alpha_xrd.pkl')
LAT_PKL = os.path.join(RRUFF_ALPHA_DIR, 'rruff_alpha_prim_lat.pkl')


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model_path', required=True)
    p.add_argument('--train_data_dir', default=None,
                   help='only needed for checkpoints that do not store the structured-head mask')
    p.add_argument('--train_lat_file', default='train_bispec_combined.pkl')
    p.add_argument('--model_config', default='conf/model/transformerbispec_sparse.yaml')
    p.add_argument('--train_config', default='conf/train/default.yaml')
    p.add_argument('--output_path', required=True)
    p.add_argument('--batch_size', type=int, default=64)
    p.add_argument('--device', default='cuda')
    p.add_argument('--eval_xrd_pkl', default=None)
    p.add_argument('--eval_lat_pkl', default=None)
    return p.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f'Device: {device}')

    model_cfg = OmegaConf.load(args.model_config)
    train_cfg = OmegaConf.load(args.train_config)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from model import XRDTransformerEncoder

    sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mp_full'))
    from generate_augmented_data import _extract_peaks_via_find_peaks
    from eval_opxrd_cnrs_selection_cosine import pad_peaks

    checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
    state_dict = checkpoint.get('model_state_dict', checkpoint)

    if 'linear_out.weight' in state_dict:
        state_dict['head.weight'] = state_dict.pop('linear_out.weight')

    allowed_indices = None
    if 'head.row_idx' in state_dict:
        # the structured-head mask is stored in the checkpoint (and load_state_dict overwrites it anyway)
        allowed_indices = torch.stack([state_dict['head.row_idx'], state_dict['head.col_idx']], dim=1).tolist()
        print(f'StructuredBispectrumHead with {len(allowed_indices)} allowed indices (from checkpoint)')
    elif 'head.linear.weight' in state_dict:
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
        input_type='sparse',
        max_peaks=model_cfg.max_peaks,
        use_intensity=model_cfg.get('use_intensity', True),
        allowed_indices=allowed_indices,
    ).to(device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f'Loaded model from {args.model_path}')

    xrd_pkl = args.eval_xrd_pkl or XRD_PKL
    lat_pkl = args.eval_lat_pkl or LAT_PKL
    print(f'Loading eval XRD from  {xrd_pkl}')
    print(f'Loading eval lats from {lat_pkl}')
    xrd_dict = pickle.load(open(xrd_pkl, 'rb'))
    lat_dict = pickle.load(open(lat_pkl, 'rb'))

    keys = sorted(set(xrd_dict) & set(lat_dict))
    print(f'  {len(keys)} entries')

    print('Extracting peak lists...')
    max_peaks = model_cfg.max_peaks
    curves = np.stack([xrd_dict[k] for k in keys]).astype(np.float64)
    peak_lists = [pad_peaks(_extract_peaks_via_find_peaks(c, max_peaks), max_peaks=max_peaks)
                  for c in curves]
    xrd_tensor = torch.tensor(peak_lists, dtype=torch.float32)
    print(f'  Peak tensor shape: {xrd_tensor.shape}')

    targets = torch.tensor(
        np.stack([lat_dict[k] for k in keys]).astype(np.float32)
    )

    all_preds = []
    with torch.no_grad():
        for i in range(0, len(xrd_tensor), args.batch_size):
            batch = xrd_tensor[i:i + args.batch_size].to(device)
            out = model(batch)
            all_preds.append(out.cpu())
    predictions = torch.cat(all_preds, dim=0).numpy()
    print(f'Predictions shape: {predictions.shape}')

    results = {
        'predictions': predictions,
        'targets': targets.numpy(),
        'material_ids': keys,
        'prediction_type': 'bispectrum',
    }
    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    with open(args.output_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'Saved to {args.output_path}  (N={len(keys)})')


if __name__ == '__main__':
    main()
