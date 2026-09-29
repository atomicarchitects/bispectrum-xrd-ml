"""
Scatter (hexbin) of length/angle MAPE vs per-channel bispectrum error.

Usage:
    python plot_bispec_channel_error.py \
        --bispec_pkl  $POWDERXRD_DATA_ROOT/final_test_results/mpfull_aug/bispec_test_results.pkl \
        --cctbx_pkl   $POWDERXRD_DATA_ROOT/final_test_results_cctbx/mpfull_aug/inversion_results_cctbx_bispec.pkl \
        --output      $POWDERXRD_DATA_ROOT/final_test_results_cctbx/mpfull_aug/bispec_channel_error.pdf \
        --mag_thresh  1.0
"""
import argparse, os, pickle
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

plt.rcParams.update({
    'font.size': 11, 'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
})


def load_data(bispec_pkl, cctbx_pkl, mag_thresh=1.0):
    with open(bispec_pkl, 'rb') as f:
        res = pickle.load(f)
    pred = res['predictions'].float().numpy()   # (N, n_basis, n_channels)
    targ = res['targets'].float().numpy()
    mat_ids = res['material_ids']

    # per-channel error: mean over radial basis -> (N, n_channels)
    chan_err = np.abs(pred - targ).mean(axis=1)

    # per-channel RMS magnitude of true values
    chan_mag = np.sqrt((targ ** 2).mean(axis=(0, 1)))
    active = np.where(chan_mag >= mag_thresh)[0]
    print(f'Active channels (mag ≥ {mag_thresh}): {active.tolist()}')

    with open(cctbx_pkl, 'rb') as f:
        df = pickle.load(f)
    df = df.dropna(subset=['cctbx_true', 'cctbx_pred']).set_index('mat_id')

    len_mapes, ang_mapes, errs = [], [], []
    for mid, ce in zip(mat_ids, chan_err):
        if mid not in df.index:
            continue
        t = df.loc[mid, 'cctbx_true']
        p = df.loc[mid, 'cctbx_pred']
        tl = np.array([t.a, t.b, t.c])
        pl = np.array([p.a, p.b, p.c])
        ta = np.array([t.alpha, t.beta, t.gamma])
        pa = np.array([p.alpha, p.beta, p.gamma])
        len_mapes.append(np.mean(np.abs(pl - tl) / tl) * 100)
        ang_mapes.append(np.mean(np.abs(pa - ta) / ta) * 100)
        errs.append(ce)

    errs = np.array(errs)          # (M, n_channels)
    len_mapes = np.array(len_mapes)
    ang_mapes = np.array(ang_mapes)
    return errs, len_mapes, ang_mapes, active, chan_mag


def hexbin_panel(ax, x, y, xlabel, ylabel, title):
    mask = np.isfinite(x) & np.isfinite(y)
    x, y = x[mask], y[mask]
    xlo, xhi = np.percentile(x, 1), np.percentile(x, 99)
    ylo, yhi = np.percentile(y, 1), np.percentile(y, 99)
    xpad = (xhi - xlo) * 0.02
    ypad = (yhi - ylo) * 0.02
    ax.hexbin(x, y, gridsize=25, cmap='Blues', mincnt=1, bins='log',
              linewidths=0.1,
              extent=[xlo - xpad, xhi + xpad, ylo - ypad, yhi + ypad])
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--bispec_pkl', required=True)
    ap.add_argument('--cctbx_pkl',  required=True)
    ap.add_argument('--output',     default='bispec_channel_error.pdf')
    ap.add_argument('--mag_thresh', type=float, default=1.0)
    args = ap.parse_args()

    print('Loading data...')
    errs, len_mapes, ang_mapes, active, chan_mag = load_data(
        args.bispec_pkl, args.cctbx_pkl, args.mag_thresh)

    n = len(active)
    ncols = min(n, 5)
    nrows = 2 * ((n + ncols - 1) // ncols)   # 2 rows (len, ang) per grid row
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3.2, nrows * 2.8))
    axes = np.array(axes).reshape(nrows, ncols)

    for idx, ch in enumerate(active):
        grid_row = (idx // ncols) * 2
        col = idx % ncols
        x = errs[:, ch]
        mag = chan_mag[ch]
        hexbin_panel(axes[grid_row,     col], x, len_mapes,
                     f'Ch{ch} error (mag={mag:.0f})', 'Length MAPE (%)', f'Ch{ch} – length')
        hexbin_panel(axes[grid_row + 1, col], x, ang_mapes,
                     f'Ch{ch} error (mag={mag:.0f})', 'Angle MAPE (%)',  f'Ch{ch} – angle')

    # hide unused axes
    for idx in range(len(active), ((nrows // 2) * ncols)):
        grid_row = (idx // ncols) * 2
        col = idx % ncols
        axes[grid_row,     col].set_visible(False)
        axes[grid_row + 1, col].set_visible(False)

    fig.suptitle('Lattice error vs per-channel bispectrum error', fontsize=13)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, bbox_inches='tight')
    print(f'Saved {args.output}')


if __name__ == '__main__':
    main()
