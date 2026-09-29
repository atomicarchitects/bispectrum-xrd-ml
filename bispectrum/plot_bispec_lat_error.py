"""
Plot bispectrum prediction error vs lattice error (length and angle).

Usage:
    python plot_bispec_lat_error.py \
        --bispec_h5  $POWDERXRD_DATA_ROOT/final_test_results/mpfull_aug/bispec_prediction_errors.h5 \
        --inv_h5     $POWDERXRD_DATA_ROOT/final_test_results/mpfull_aug/bispec_test_results_inverted.h5 \
        --output     bispec_lat_error.pdf \
        --sample     5000          # optional
"""
import argparse, os, h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.stats import pearsonr, spearmanr

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utilities import convert_lats_to_niggli

plt.rcParams.update({
    'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 18,
    'xtick.labelsize': 16, 'ytick.labelsize': 16,
})


def _cache_path(path):
    base = os.path.splitext(path)[0]
    return base + '_lat_error_cache_v2.npz'


def _mapes(p, t):
    ae = np.abs(p - t)
    len_mape = (ae[:3] / t[:3]).mean() * 100
    ang_mape = (ae[3:] / t[3:]).mean() * 100
    return ae[:3].mean(), ae[3:].mean(), len_mape, ang_mape


def collect_data_bispec_pct(bispec_pkl, cctbx_pkl, mag_thresh=1.0, no_cache=False):
    """Like collect_data_cctbx but uses bispectrum MAPE (%) over active channels as x-axis."""
    base = os.path.splitext(cctbx_pkl)[0]
    cache = base + '_lat_error_bispec_pct_cache.npz'
    if not no_cache and os.path.exists(cache):
        print(f'  Loading cached data from {cache}')
        d = np.load(cache)
        return (d['bispec_mapes'], d['length_maes'], d['angle_maes'],
                d['length_mapes'], d['angle_mapes'])

    import pickle
    with open(bispec_pkl, 'rb') as f:
        res = pickle.load(f)
    pred = res['predictions'].float().numpy()   # (N, n_basis, n_channels)
    targ = res['targets'].float().numpy()
    mat_ids_bispec = res['material_ids']

    chan_mag = np.sqrt((targ ** 2).mean(axis=(0, 1)))   # (n_channels,)
    active = chan_mag >= mag_thresh
    pred_a = pred[:, :, active]
    targ_a = targ[:, :, active]
    # percent error only over non-zero target entries
    abs_err = np.abs(pred_a - targ_a)                   # (N, n_basis, n_active)
    abs_targ = np.abs(targ_a)
    nonzero = abs_targ > 1e-3                            # mask near-zero entries
    safe_targ = np.where(nonzero, abs_targ, 1.0)        # avoid divide-by-zero in both branches
    pct_err = np.where(nonzero, abs_err / safe_targ, np.nan)
    per_struct_mape = np.nanmean(pct_err, axis=(1, 2)) * 100  # (N,)
    bispec_mape_dict = dict(zip(mat_ids_bispec, per_struct_mape))

    with open(cctbx_pkl, 'rb') as f:
        df = pickle.load(f)
    df = df.dropna(subset=['cctbx_true', 'cctbx_pred']).set_index('mat_id')

    bispec_mapes, length_maes, angle_maes, length_mapes, angle_mapes = [], [], [], [], []
    for mid in df.index:
        if mid not in bispec_mape_dict:
            continue
        t = df.loc[mid, 'cctbx_true']
        p = df.loc[mid, 'cctbx_pred']
        tp = np.array([t.a, t.b, t.c, t.alpha, t.beta, t.gamma])
        pp = np.array([p.a, p.b, p.c, p.alpha, p.beta, p.gamma])
        lmae, amae, lmape, amape = _mapes(pp, tp)
        bispec_mapes.append(bispec_mape_dict[mid])
        length_maes.append(lmae); angle_maes.append(amae)
        length_mapes.append(lmape); angle_mapes.append(amape)

    arrays = (np.array(bispec_mapes), np.array(length_maes), np.array(angle_maes),
              np.array(length_mapes), np.array(angle_mapes))
    np.savez(cache, bispec_mapes=arrays[0], length_maes=arrays[1], angle_maes=arrays[2],
             length_mapes=arrays[3], angle_mapes=arrays[4])
    print(f'  Cached to {cache}')
    return arrays


def collect_data_cctbx(bispec_h5, cctbx_pkl, no_cache=False):
    """Load lattice errors from cctbx pickle (uses cctbx_true/cctbx_pred Lattice objects)."""
    cache = _cache_path(cctbx_pkl)
    if not no_cache and os.path.exists(cache):
        print(f'  Loading cached data from {cache}')
        d = np.load(cache)
        return (d['bispec_errs'], d['length_maes'], d['angle_maes'],
                d['vol_apes'], d['length_mapes'], d['angle_mapes'])

    import pickle
    with open(cctbx_pkl, 'rb') as f:
        df = pickle.load(f)
    df = df.dropna(subset=['cctbx_true', 'cctbx_pred']).set_index('mat_id')

    bispec_errs, length_maes, angle_maes, vol_apes, length_mapes, angle_mapes = [], [], [], [], [], []
    with h5py.File(bispec_h5, 'r') as fb:
        for mid in df.index:
            if mid not in fb:
                continue
            be = float(fb[mid][()])
            t = df.loc[mid, 'cctbx_true']
            p = df.loc[mid, 'cctbx_pred']
            tp = np.array([t.a, t.b, t.c, t.alpha, t.beta, t.gamma])
            pp = np.array([p.a, p.b, p.c, p.alpha, p.beta, p.gamma])
            lmae, amae, lmape, amape = _mapes(pp, tp)
            bispec_errs.append(be); length_maes.append(lmae); angle_maes.append(amae)
            vol_apes.append(abs(p.volume - t.volume) / t.volume * 100)
            length_mapes.append(lmape); angle_mapes.append(amape)

    arrays = (np.array(bispec_errs), np.array(length_maes), np.array(angle_maes),
              np.array(vol_apes), np.array(length_mapes), np.array(angle_mapes))
    np.savez(cache, bispec_errs=arrays[0], length_maes=arrays[1], angle_maes=arrays[2],
             vol_apes=arrays[3], length_mapes=arrays[4], angle_mapes=arrays[5])
    print(f'  Cached to {cache}')
    return arrays


def collect_data(bispec_h5, inv_h5, sample=None, no_cache=False):
    cache = _cache_path(inv_h5)
    if not no_cache and sample is None and os.path.exists(cache):
        print(f'  Loading cached data from {cache}')
        d = np.load(cache)
        return (d['bispec_errs'], d['length_maes'], d['angle_maes'],
                d['vol_apes'], d['length_mapes'], d['angle_mapes'])

    with h5py.File(bispec_h5, 'r') as fb, h5py.File(inv_h5, 'r') as fi:
        common = sorted(set(fb.keys()) & set(fi.keys()))
        if sample:
            rng = np.random.default_rng(42)
            common = list(rng.choice(common, size=min(sample, len(common)), replace=False))

        bispec_errs, length_maes, angle_maes, vol_apes, length_mapes, angle_mapes = [], [], [], [], [], []
        for k in common:
            be = float(fb[k][()])
            pred_mat = fi[k]['pred'][()]
            true_mat = fi[k]['true_lattice'][()]
            try:
                lat_true, lat_pred = convert_lats_to_niggli(true_mat, pred_mat, is_reciprocal=False)
                p = np.array([lat_pred.a, lat_pred.b, lat_pred.c,
                              lat_pred.alpha, lat_pred.beta, lat_pred.gamma])
                t = np.array([lat_true.a, lat_true.b, lat_true.c,
                              lat_true.alpha, lat_true.beta, lat_true.gamma])
                lmae, amae, lmape, amape = _mapes(p, t)
                vol_ape = abs(lat_pred.volume - lat_true.volume) / lat_true.volume * 100
            except Exception:
                continue
            bispec_errs.append(be); length_maes.append(lmae); angle_maes.append(amae)
            vol_apes.append(vol_ape); length_mapes.append(lmape); angle_mapes.append(amape)

    arrays = (np.array(bispec_errs), np.array(length_maes), np.array(angle_maes),
              np.array(vol_apes), np.array(length_mapes), np.array(angle_mapes))
    if sample is None:
        np.savez(cache, bispec_errs=arrays[0], length_maes=arrays[1], angle_maes=arrays[2],
                 vol_apes=arrays[3], length_mapes=arrays[4], angle_mapes=arrays[5])
        print(f'  Cached to {cache}')
    return arrays


def hexbin_panel(ax, fig, x, y, xlabel, ylabel, title, add_colorbar=False,
                 mape_frac=None, log_scale=True):
    xlo, xhi = np.percentile(x, 1), np.percentile(x, 99)
    ylo, yhi = np.percentile(y, 1), np.percentile(y, 99)
    xpad, ypad = (xhi - xlo) * 0.02, (yhi - ylo) * 0.02

    hb = ax.hexbin(x, y, gridsize=35, cmap='Blues', mincnt=1,
                   bins='log' if log_scale else None,
                   linewidths=0.1,
                   extent=[xlo - xpad, xhi + xpad, ylo - ypad, yhi + ypad])
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    if add_colorbar:
        fig.colorbar(hb, cax=cax, label='log₁₀(count)' if log_scale else 'count')
    else:
        cax.set_visible(False)

    r2 = pearsonr(x, y)[0] ** 2
    txt = f'R²={r2:.3f}'
    if mape_frac is not None:
        txt += f'\n{mape_frac:.1f}% MAPE ≤5%'
    ax.text(0.05, 0.95, txt,
            transform=ax.transAxes, fontsize=18, va='top',
            bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75))
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_title(title, fontsize=18)
    ax.tick_params(labelsize=16)


def bispec_histogram_panel(ax, bispec_errs):
    """Histogram of bispectrum errors with quantile bin boundaries."""
    ax.hist(bispec_errs, bins=80, color='steelblue', alpha=0.75, edgecolor='none')
    ax.set_ylabel('Count', fontsize=18)
    ax.set_title('Bispectrum error distribution', fontsize=18)
    ax.tick_params(labelsize=16)
    ax.grid(True, alpha=0.25)


def conditional_fraction_panel(ax, bispec_errs, ang_mapes, len_mapes,
                                n_bins=7, thresh=5.0):
    """Fraction of structures with MAPE ≤ thresh vs binned bispectrum error."""
    import pandas as pd
    df = pd.DataFrame({'be': bispec_errs, 'ang': ang_mapes, 'len': len_mapes})
    df['bin'] = pd.qcut(df['be'], q=n_bins, labels=range(n_bins))
    bin_edges = pd.qcut(df['be'], q=n_bins, retbins=True)[1]
    bin_mids  = (bin_edges[:-1] + bin_edges[1:]) / 2

    grouped = df.groupby('bin', observed=True)
    frac_ang  = grouped.apply(lambda g: (g['ang'] <= thresh).mean(), include_groups=False).values
    frac_len  = grouped.apply(lambda g: (g['len'] <= thresh).mean(), include_groups=False).values
    frac_both = grouped.apply(
        lambda g: ((g['ang'] <= thresh) & (g['len'] <= thresh)).mean(), include_groups=False
    ).values

    xs = np.concatenate([[bin_edges[0]], bin_mids, [bin_edges[-1]]])
    def _ext(f): return np.concatenate([[f[0]], f, [f[-1]]]) * 100
    ax.plot(xs, _ext(frac_ang),  'o-', color='seagreen',   lw=2, label=f'Angle MAPE ≤{thresh}%')
    ax.plot(xs, _ext(frac_len),  's-', color='steelblue',  lw=2, label=f'Length MAPE ≤{thresh}%')
    ax.plot(xs, _ext(frac_both), '^-', color='darkorange', lw=2, label='Both')

    ax.set_ylim(0, 105)
    ax.set_ylabel('% of structures', fontsize=18)
    ax.set_title('Recovery rate vs bispectrum error', fontsize=18)
    ax.legend(fontsize=13, loc='upper right')
    ax.grid(True, alpha=0.25)
    ax.tick_params(labelsize=16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bispec_h5',  default=None, help='bispec_prediction_errors.h5 (L1 error mode)')
    ap.add_argument('--bispec_pkl', default=None, help='bispec_test_results.pkl (MAPE % mode)')
    ap.add_argument('--inv_h5',     default=None, help='bispec_test_results_inverted.h5')
    ap.add_argument('--cctbx-pkl',  default=None, help='cctbx result pickle (uses cctbx_true/pred)')
    ap.add_argument('--output',     default='bispec_lat_error.pdf')
    ap.add_argument('--sample',     type=int, default=None)
    ap.add_argument('--title',      default='Bispectrum error vs lattice error')
    ap.add_argument('--linear',     action='store_true')
    ap.add_argument('--no-cache',   action='store_true')
    args = ap.parse_args()

    print('Loading data...')
    if args.bispec_pkl and args.cctbx_pkl:
        bispec_errs, length_maes, angle_maes, length_mapes, angle_mapes = \
            collect_data_bispec_pct(args.bispec_pkl, args.cctbx_pkl, no_cache=args.no_cache)
        x_label = 'Bispectrum percent error (%, non-zero entries)'
    elif args.cctbx_pkl:
        bispec_errs, length_maes, angle_maes, _, length_mapes, angle_mapes = \
            collect_data_cctbx(args.bispec_h5, args.cctbx_pkl, no_cache=args.no_cache)
        x_label = 'Bispectrum L1 error'
    elif args.inv_h5:
        bispec_errs, length_maes, angle_maes, _, length_mapes, angle_mapes = \
            collect_data(args.bispec_h5, args.inv_h5, sample=args.sample, no_cache=args.no_cache)
        x_label = 'Bispectrum L1 error'
    else:
        raise ValueError('Provide --bispec_pkl + --cctbx-pkl, or --bispec_h5 + (--inv_h5 or --cctbx-pkl)')

    mask = bispec_errs <= 200
    bispec_errs  = bispec_errs[mask]
    length_maes  = length_maes[mask]
    angle_maes   = angle_maes[mask]
    length_mapes = length_mapes[mask]
    angle_mapes  = angle_mapes[mask]
    print(f'  n={len(bispec_errs)}')
    print(f'  Bispec err: mean={bispec_errs.mean():.3f}, median={np.median(bispec_errs):.3f}')
    print(f'  Length MAE: mean={length_maes.mean():.4f}, median={np.median(length_maes):.4f}')
    print(f'  Angle MAE:  mean={angle_maes.mean():.4f}, median={np.median(angle_maes):.4f}')
    r2_len  = pearsonr(bispec_errs, length_mapes)[0] ** 2
    r2_ang  = pearsonr(bispec_errs, angle_mapes)[0] ** 2
    rho_len = spearmanr(bispec_errs, length_mapes)[0]
    rho_ang = spearmanr(bispec_errs, angle_mapes)[0]
    print(f'  Overall R²:  length={r2_len:.3f}, angle={r2_ang:.3f}')
    print(f'  Overall rho: length={rho_len:.3f}, angle={rho_ang:.3f}')

    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharex=True)
    conditional_fraction_panel(axes[0], bispec_errs, angle_mapes, length_mapes)
    bispec_histogram_panel(axes[1], bispec_errs)

    fig.supxlabel(x_label, fontsize=18)
    fig.suptitle(args.title, fontsize=18)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, bbox_inches='tight')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
