"""
Plot bispectrum prediction error vs lattice error (length and angle separately)
for a given model's test set or RRUFF alpha eval results.

Usage (mpfull test set):
    python plot_bispec_vs_lat_error.py \
        --eval_pkl /path/to/bispec_test_results.pkl \
        --h5 /path/to/bispec_test_results_inverted.h5 \
        --output_path /path/to/output.png \
        --sample 2000

Usage (RRUFF alpha, true bispectra computed on the fly):
    python plot_bispec_vs_lat_error.py \
        --eval_pkl /path/to/rruff_alpha_bispec_results.pkl \
        --h5 /path/to/inversion_results.h5 \
        --output_path /path/to/output.png \
        --compute_true_bispec
"""
import argparse, sys, os, pickle, h5py
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utilities import get_bispectrum, get_neighbors, convert_lats_to_niggli

TRAIN_LAT_PATH = os.environ.get('POWDERXRD_DATA_ROOT', './data') + '/mpfull_aug_cag_strain_texture/train_bispec_combined.pkl'
L_MAX, K_MAX, N_BASIS = 6, 2/3, 10


def get_allowed_indices(mask_threshold=1e-10):
    train_lat = pickle.load(open(TRAIN_LAT_PATH, 'rb'))
    if isinstance(train_lat, dict):
        train_lat = list(train_lat.values())
    all_targets = torch.stack(train_lat).float()
    loss_mask = torch.any(torch.abs(all_targets) > mask_threshold, dim=0)
    return torch.nonzero(loss_mask, as_tuple=False).tolist()


def cuberoot(x):
    return np.sign(x) * np.abs(x) ** (1/3)


def compute_true_bispec(true_lat_mat, cob_bi):
    from pymatgen.core import Lattice
    lat_pmg = Lattice(true_lat_mat)
    recip = lat_pmg.reciprocal_lattice_crystallographic.matrix
    miller = get_neighbors(recip, K_MAX, filter_orig=True, filter_new=False, miller_ind_threshold=3)
    neighs = torch.tensor(miller.reshape(-1, 3) @ recip, dtype=torch.float64)
    return cuberoot(get_bispectrum(neighs, K_MAX, lmax=L_MAX, radial_function='bessel',
                                   n_basis=N_BASIS, cob_bi=cob_bi).numpy())


def build_cob_bi():
    import e3nn_jax
    from e3nn import io
    rtp_bi = e3nn_jax.reduced_symmetric_tensor_product_basis(
        io.SphericalTensor(L_MAX, 1, -1), 3, keep_ir=['0o', '0e'])
    return torch.tensor(rtp_bi.array, dtype=torch.float64)


def collect_data(eval_pkl, h5_path, allowed_indices, sample=None, compute_true=False):
    eval_d = pickle.load(open(eval_pkl, 'rb'))
    pred_bispec = eval_d['predictions'].numpy()
    mat_ids = eval_d['material_ids']
    id_to_idx = {mid: i for i, mid in enumerate(mat_ids)}

    has_true_bispec = 'targets' in eval_d and eval_d['targets'].shape == eval_d['predictions'].shape
    if has_true_bispec:
        true_bispec = eval_d['targets'].numpy()

    cob_bi = build_cob_bi() if compute_true else None

    bispec_errors, length_maes, angle_maes = [], [], []

    with h5py.File(h5_path, 'r') as f:
        keys = list(f.keys())
        if sample:
            np.random.seed(42)
            keys = list(np.random.choice(keys, size=min(sample, len(keys)), replace=False))

        for k in keys:
            if k not in id_to_idx:
                continue
            idx = id_to_idx[k]
            pred_lat_mat = f[k]['pred'][()]
            true_lat_mat = f[k]['true_lattice'][()]

            # bispectrum error
            try:
                if has_true_bispec:
                    true_bs = true_bispec[idx]
                elif compute_true:
                    true_bs = compute_true_bispec(true_lat_mat, cob_bi)
                else:
                    continue
                pred_allowed = np.array([pred_bispec[idx, i, j] for i, j in allowed_indices])
                true_allowed = np.array([true_bs[i, j] for i, j in allowed_indices])
                l1_err = np.abs(pred_allowed - true_allowed).mean()
            except Exception:
                continue

            # lattice errors split into lengths and angles
            try:
                lat_pred, lat_true = convert_lats_to_niggli(pred_lat_mat, true_lat_mat, is_reciprocal=False)
                p = np.array([lat_pred.a, lat_pred.b, lat_pred.c, lat_pred.alpha, lat_pred.beta, lat_pred.gamma])
                t = np.array([lat_true.a, lat_true.b, lat_true.c, lat_true.alpha, lat_true.beta, lat_true.gamma])
                ae = np.abs(p - t)
                length_mae = ae[:3].mean()
                angle_mae = ae[3:].mean()
            except Exception:
                continue

            bispec_errors.append(l1_err)
            length_maes.append(length_mae)
            angle_maes.append(angle_mae)

    return np.array(bispec_errors), np.array(length_maes), np.array(angle_maes)


def make_plot(bispec_errors, length_maes, angle_maes, title, output_path, gridsize=40):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, maes, label in zip(axes,
                                [length_maes, angle_maes],
                                ['Length MAE (Angstrom)', 'Angle MAE (degrees)']):
        corr = np.corrcoef(bispec_errors, maes)[0, 1]
        hb = ax.hexbin(bispec_errors, maes, gridsize=gridsize, cmap='Blues', mincnt=1, bins='log')
        plt.colorbar(hb, ax=ax, label='log10(count)')
        ax.set_xlabel('Bispectrum L1 Error (cube-root space)')
        ax.set_ylabel(label)
        ax.text(0.05, 0.95, f'r = {corr:.3f}  (n={len(bispec_errors)})',
                transform=ax.transAxes, va='top', fontsize=10)

    fig.suptitle(title, fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    print(f'Saved to {output_path}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--eval_pkl', required=True)
    p.add_argument('--h5', required=True)
    p.add_argument('--output_path', required=True)
    p.add_argument('--title', default='Bispectrum Error vs Lattice Error')
    p.add_argument('--sample', type=int, default=None)
    p.add_argument('--compute_true_bispec', action='store_true',
                   help='Compute true bispectra from true_lattice in h5 (for RRUFF alpha eval)')
    args = p.parse_args()

    print('Loading allowed indices...')
    allowed_indices = get_allowed_indices()
    print(f'  {len(allowed_indices)} / {10*35} allowed indices')

    print('Collecting data...')
    bispec_errors, length_maes, angle_maes = collect_data(
        args.eval_pkl, args.h5, allowed_indices,
        sample=args.sample, compute_true=args.compute_true_bispec)
    print(f'  n={len(bispec_errors)}')
    print(f'  Bispec L1:   mean={bispec_errors.mean():.4f}, median={np.median(bispec_errors):.4f}')
    print(f'  Length MAE:  mean={length_maes.mean():.4f}, median={np.median(length_maes):.4f}')
    print(f'  Angle MAE:   mean={angle_maes.mean():.4f}, median={np.median(angle_maes):.4f}')

    make_plot(bispec_errors, length_maes, angle_maes, args.title, args.output_path)


if __name__ == '__main__':
    main()
