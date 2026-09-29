"""
Plot length MAPE and angle MAPE vs. unit cell volume (quantile-binned, log x-axis).
Two panels side by side; four lines: {Dataset A, Dataset B} x {Bispec, Direct}.

Usage:
    python plot_volume_mape.py --datasets mpfull
    python plot_volume_mape.py --datasets mp20 mp20_aug mpfull
    python plot_volume_mape.py --n-bins 6 --output volume_mape.pdf
"""

import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 17,
    'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 14,
})

BASE = os.path.join(os.environ.get('POWDERXRD_DATA_ROOT', './data'), 'final_test_results_cctbx')

# Registry: key -> (label_a, dir_a, label_b, dir_b, output_suffix)
DATASET_REGISTRY = {
    'mpfull': (
        'MPFull Aug', 'mpfull_aug',
        'MPFull',     'mpfull_crystalyze_cag',
        'mpfull',
    ),
    'mp20': (
        'MP20 Aug', 'mp20_final_aug',
        'MP20',     'mp20_correct_cag',
        'mp20',
    ),
}

def build_models(label_a, dir_a, label_b, dir_b):
    return {
        f'{label_a} Bispec': (f'{BASE}/{dir_a}/inversion_results_cctbx_bispec.pkl',
                               'darkorange', '-'),
        f'{label_b} Bispec': (f'{BASE}/{dir_b}/inversion_results_cctbx_bispec.pkl',
                               'darkorange', '--'),
        f'{label_a} Direct': (f'{BASE}/{dir_a}/inversion_results_cctbx_direct.pkl',
                               'steelblue',  '-'),
        f'{label_b} Direct': (f'{BASE}/{dir_b}/inversion_results_cctbx_direct.pkl',
                               'steelblue',  '--'),
    }


def _angle_diff(a, b):
    d = abs(a - b)
    return min(d, 360 - d)


def load_and_compute(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df = df.dropna(subset=['cctbx_true', 'cctbx_pred']).reset_index(drop=True)

    vols, len_mapes, ang_mapes = [], [], []
    for t, p in zip(df['cctbx_true'], df['cctbx_pred']):
        vols.append(t.volume)

        t_l = np.array([t.a, t.b, t.c])
        p_l = np.array([p.a, p.b, p.c])
        len_mapes.append(np.mean(np.abs(p_l - t_l) / t_l) * 100)

        t_a = np.array([t.alpha, t.beta, t.gamma])
        p_a = np.array([p.alpha, p.beta, p.gamma])
        diffs = np.array([_angle_diff(pa, ta) for pa, ta in zip(p_a, t_a)])
        ang_mapes.append(np.mean(diffs / t_a) * 100)

    return pd.DataFrame({'volume': vols, 'len_mape': len_mapes, 'ang_mape': ang_mapes})


def bin_by_volume(df: pd.DataFrame, n_bins: int):
    """Quantile-bin on log(volume); return bin midpoints and per-bin stats."""
    log_vol = np.log10(df['volume'])
    labels  = range(n_bins)
    df = df.copy()
    df['bin'] = pd.qcut(log_vol, q=n_bins, labels=labels)
    bin_edges = pd.qcut(log_vol, q=n_bins, retbins=True)[1]
    bin_mids  = 10 ** ((bin_edges[:-1] + bin_edges[1:]) / 2)

    grouped = df.groupby('bin', observed=True)
    stats = {}
    for metric in ('len_mape', 'ang_mape'):
        stats[metric] = {
            'mean':   grouped[metric].mean().values,
            'median': grouped[metric].median().values,
            'q25':    grouped[metric].quantile(0.25).values,
            'q75':    grouped[metric].quantile(0.75).values,
            'n':      grouped[metric].count().values,
        }
    return bin_mids, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-bins', type=int, default=7)
    ap.add_argument('--stat', choices=['mean', 'median'], default='median')
    ap.add_argument('--datasets', nargs='+', default=['mpfull', 'mp20'],
                    choices=list(DATASET_REGISTRY.keys()))
    args = ap.parse_args()

    for ds_key in args.datasets:
        label_a, dir_a, label_b, dir_b, slug = DATASET_REGISTRY[ds_key]
        models = build_models(label_a, dir_a, label_b, dir_b)
        output = os.path.join(BASE, f'volume_mape_{slug}.pdf')
        run_dataset(models, args.n_bins, args.stat, output, slug)


def run_dataset(models, n_bins, stat, output, title_slug):
    data = {}
    for name, (path, color, ls) in models.items():
        if not os.path.exists(path):
            print(f'[skip] {name}: {path} not found')
            continue
        # Deduplicate: if same path already loaded, reuse
        print(f'Loading {name}...')
        data[name] = (load_and_compute(path), color, ls)

    fig, axes = plt.subplots(1, 2, figsize=(16, 5), sharey=False)
    panels = [
        (axes[0], 'len_mape', 'Length MAPE (%)'),
        (axes[1], 'ang_mape', 'Angle MAPE (%)'),
    ]

    for name, (df, color, ls) in data.items():
        bin_mids, bstats = bin_by_volume(df, n_bins)
        for ax, metric, ylabel in panels:
            s = bstats[metric]
            y = s[stat]
            ax.plot(bin_mids, y, color=color, linestyle=ls, linewidth=2,
                    marker='o', markersize=5, label=name)
            ax.fill_between(bin_mids, s['q25'], s['q75'],
                            color=color, alpha=0.10)

    for ax, metric, ylabel in panels:
        ax.set_xscale('log')
        ax.set_xlabel('Unit cell volume (Å³)')
        ax.set_ylabel(ylabel)
        ax.grid(True, which='both', alpha=0.25)
        ax.set_title(ylabel.replace(' (%)', '') + ' vs. volume')

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='center left',
               bbox_to_anchor=(1.0, 0.5), frameon=True)

    fig.suptitle(f'MAPE vs. unit cell volume — {n_bins} quantile bins', fontsize=18)
    plt.tight_layout()
    plt.subplots_adjust(right=0.78)
    os.makedirs(os.path.dirname(os.path.abspath(output)), exist_ok=True)
    plt.savefig(output, bbox_inches='tight')
    print(f'Saved {output}')

    # Print bin summary
    for name, (df, _, _) in data.items():
        bin_mids, bstats = bin_by_volume(df, n_bins)
        print(f'\n{name}')
        print(f'  {"Vol mid (Å³)":>12}  {"n":>6}  {"Len MAPE":>10}  {"Ang MAPE":>10}')
        for mid, n, lm, am in zip(bin_mids,
                                   bstats['len_mape']['n'],
                                   bstats['len_mape'][stat],
                                   bstats['ang_mape'][stat]):
            print(f'  {mid:>12.1f}  {n:>6}  {lm:>10.2f}  {am:>10.2f}')


if __name__ == '__main__':
    main()
