"""
Failure mode analysis for lattice parameter prediction.

For each dataset produces:
  - Bravais type confusion heatmap (normalized by true type)
  - Crystal system confusion heatmap
  - Threshold fraction bars per Bravais / crystal system
    (len MAPE ≤5%, ang MAPE ≤5%, both ≤5%, Bravais match rate)
  - Stacked failure-category bars (4 quadrants: geo correct/wrong × label correct/wrong)
  - Misprediction distribution: for each true Bravais, what does it get predicted as?

Usage:
    python failure_mode_analysis.py --datasets MPFull
    python failure_mode_analysis.py --datasets MPFull --model Bispec
    python failure_mode_analysis.py --cctbx-dir $POWDERXRD_DATA_ROOT/final_test_results_cctbx
"""

import argparse, os, pickle
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.patches import Patch

# ── constants (mirror make_comparison_plots.py) ───────────────────────────────
BRAVAIS_ORDER = ["aP", "mP", "mC", "oP", "oC", "oI", "oF",
                 "tP", "tI", "hR", "hP", "cP", "cI", "cF"]

BRAVAIS_TO_CRYSTAL_SYSTEM = {
    "aP": "Triclinic",
    "mP": "Monoclinic",  "mC": "Monoclinic",
    "oP": "Orthorhombic","oC": "Orthorhombic","oI": "Orthorhombic","oF": "Orthorhombic",
    "tP": "Tetragonal",  "tI": "Tetragonal",
    "hR": "Trigonal",
    "hP": "Hexagonal",
    "cP": "Cubic",       "cI": "Cubic",       "cF": "Cubic",
}

CRYSTAL_SYSTEM_ORDER = ["Triclinic", "Monoclinic", "Orthorhombic",
                        "Tetragonal", "Trigonal", "Hexagonal", "Cubic"]

plt.rcParams.update({
    'font.size': 18, 'axes.titlesize': 18, 'axes.labelsize': 17,
    'xtick.labelsize': 16, 'ytick.labelsize': 16, 'legend.fontsize': 14,
})


# ── per-pair metrics ──────────────────────────────────────────────────────────
def _angle_diff(a, b):
    d = abs(a - b)
    return min(d, 360 - d)


def _row_mapes(true_lat, pred_lat):
    t_l = np.array([true_lat.a, true_lat.b, true_lat.c])
    p_l = np.array([pred_lat.a, pred_lat.b, pred_lat.c])
    len_mape = np.mean(np.abs(p_l - t_l) / t_l) * 100

    t_a = np.array([true_lat.alpha, true_lat.beta, true_lat.gamma])
    p_a = np.array([pred_lat.alpha, pred_lat.beta, pred_lat.gamma])
    diffs = np.array([_angle_diff(pa, ta) for pa, ta in zip(p_a, t_a)])
    ang_mape = np.mean(diffs / t_a) * 100
    return len_mape, ang_mape


def add_metric_cols(df: pd.DataFrame) -> pd.DataFrame:
    """Add len_mape, ang_mape, geo_ok, label_ok, crystal_system columns."""
    df = df.copy()
    len_mapes, ang_mapes = [], []
    for t, p in zip(df['cctbx_true'], df['cctbx_pred']):
        lm, am = _row_mapes(t, p)
        len_mapes.append(lm)
        ang_mapes.append(am)
    df['len_mape'] = len_mapes
    df['ang_mape'] = ang_mapes
    df['geo_ok']   = (df['len_mape'] <= 5) & (df['ang_mape'] <= 5)
    df['label_ok'] = df['pred_bravais'] == df['true_bravais']
    df['true_crystal_system'] = df['true_bravais'].map(BRAVAIS_TO_CRYSTAL_SYSTEM)
    df['pred_crystal_system'] = df['pred_bravais'].map(BRAVAIS_TO_CRYSTAL_SYSTEM)
    return df


# ── confusion matrix ──────────────────────────────────────────────────────────
def confusion_matrix_norm(df: pd.DataFrame, true_col: str, pred_col: str,
                           ordered: list) -> pd.DataFrame:
    """Row-normalised confusion matrix (fraction of each true type)."""
    counts = pd.crosstab(df[true_col], df[pred_col])
    counts = counts.reindex(index=ordered, columns=ordered, fill_value=0)
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    return counts.div(row_sums, axis=0).fillna(0)


def plot_confusion(cm: pd.DataFrame, title: str, path: str, annot_fmt: str = '.0%'):
    n = len(cm)
    fig, ax = plt.subplots(figsize=(max(8, n * 0.65), max(6, n * 0.55)))
    im = ax.imshow(cm.values, cmap='Blues', vmin=0, vmax=1, aspect='auto')
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02, label='Fraction of true type')

    ax.set_xticks(range(n)); ax.set_xticklabels(cm.columns, rotation=45, ha='right')
    ax.set_yticks(range(n)); ax.set_yticklabels(cm.index)
    ax.set_xlabel('Predicted'); ax.set_ylabel('True')
    ax.set_title(title)

    for i in range(n):
        for j in range(n):
            val = cm.values[i, j]
            if val > 0.01:
                color = 'white' if val > 0.55 else 'black'
                ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                        fontsize=8, color=color)
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight'); plt.close()


# ── threshold fraction bars ───────────────────────────────────────────────────
def threshold_rates(df: pd.DataFrame, group_col: str, ordered: list) -> pd.DataFrame:
    """
    Returns DataFrame indexed by group with columns:
      len≤5%, ang≤5%, both≤5%, brav_match, n
    """
    rows = {}
    for grp, sub in df.groupby(group_col):
        rows[grp] = {
            'len≤5%':     (sub['len_mape'] <= 5).mean() * 100,
            'ang≤5%':     (sub['ang_mape'] <= 5).mean() * 100,
            'both≤5%':    sub['geo_ok'].mean() * 100,
            'brav_match': sub['label_ok'].mean() * 100,
            'n':          len(sub),
        }
    return pd.DataFrame(rows).T.reindex([g for g in ordered if g in rows])


def plot_threshold_bars(rates: pd.DataFrame, title: str, path: str):
    metrics = ['len≤5%', 'ang≤5%', 'both≤5%']
    colors  = ['steelblue', 'seagreen', 'darkorange']
    labels  = ['Len MAPE ≤5%', 'Ang MAPE ≤5%', 'Both ≤5%']

    n_groups = len(rates)
    x = np.arange(n_groups)
    width = 0.22
    offsets = np.linspace(-1, 1, 3) * width

    fig, ax = plt.subplots(figsize=(max(10, n_groups * 0.9), 5))
    for offset, metric, color, label in zip(offsets, metrics, colors, labels):
        vals = rates[metric].fillna(0).values
        ax.bar(x + offset, vals, width, label=label, color=color, alpha=0.85)

    ax.set_xticks(x); ax.set_xticklabels(rates.index, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.set_ylabel('% of structures')
    ax.set_title(title)
    ax.legend(loc='upper left')
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight'); plt.close()


# ── failure category stacked bars ─────────────────────────────────────────────
# 4 quadrants: geo_ok × label_ok
CAT_COLORS = {
    'correct label + correct geo':  '#2ecc71',
    'correct label + wrong geo':    '#3498db',
    'wrong label + correct geo':    '#f39c12',
    'wrong label + wrong geo':      '#e74c3c',
}

def failure_category_rates(df: pd.DataFrame, group_col: str,
                            ordered: list) -> pd.DataFrame:
    cats = list(CAT_COLORS.keys())
    rows = {}
    for grp, sub in df.groupby(group_col):
        n = len(sub)
        rows[grp] = {
            'correct label + correct geo':  (sub['label_ok'] & sub['geo_ok']).sum() / n * 100,
            'correct label + wrong geo':    (sub['label_ok'] & ~sub['geo_ok']).sum() / n * 100,
            'wrong label + correct geo':    (~sub['label_ok'] & sub['geo_ok']).sum() / n * 100,
            'wrong label + wrong geo':      (~sub['label_ok'] & ~sub['geo_ok']).sum() / n * 100,
        }
    return pd.DataFrame(rows).T.reindex([g for g in ordered if g in rows])


def plot_failure_categories(cats: pd.DataFrame, title: str, path: str):
    n = len(cats)
    x = np.arange(n)
    fig, ax = plt.subplots(figsize=(max(10, n * 0.9), 5))

    bottoms = np.zeros(n)
    for cat, color in CAT_COLORS.items():
        vals = cats[cat].fillna(0).values
        ax.bar(x, vals, bottom=bottoms, label=cat, color=color, alpha=0.9)
        bottoms += vals

    ax.set_xticks(x); ax.set_xticklabels(cats.index, rotation=45, ha='right')
    ax.set_ylim(0, 105)
    ax.set_ylabel('% of structures')
    ax.set_title(title)
    ax.legend(loc='upper left', fontsize=10,
              handles=[Patch(color=c, label=l) for l, c in CAT_COLORS.items()])
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0f}%'))
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight'); plt.close()


# ── misprediction distribution ────────────────────────────────────────────────
def plot_mispred_distribution(df: pd.DataFrame, title: str, path: str,
                               ordered: list = BRAVAIS_ORDER):
    """
    For each true Bravais type, show a stacked bar of what predicted types
    the WRONG predictions fall into.
    """
    wrong = df[~df['label_ok']]
    if len(wrong) == 0:
        return

    # Count predicted Bravais for each (true, pred) pair among wrong rows
    counts = pd.crosstab(wrong['true_bravais'], wrong['pred_bravais'])
    counts = counts.reindex(index=[b for b in ordered if b in counts.index],
                            columns=[b for b in ordered if b in counts.columns],
                            fill_value=0)
    # Normalize by row (fraction of wrong predictions for that true type)
    row_sums = counts.sum(axis=1).replace(0, np.nan)
    fracs = counts.div(row_sums, axis=0).fillna(0)

    n_true = len(fracs)
    n_pred = len(fracs.columns)
    cmap = plt.get_cmap('tab20', n_pred)
    colors = [cmap(i) for i in range(n_pred)]

    x = np.arange(n_true)
    fig, ax = plt.subplots(figsize=(max(10, n_true * 0.9), 5))
    bottoms = np.zeros(n_true)
    for j, pred_type in enumerate(fracs.columns):
        vals = fracs[pred_type].values
        ax.bar(x, vals, bottom=bottoms, label=pred_type, color=colors[j], alpha=0.9)
        bottoms += vals

    ax.set_xticks(x); ax.set_xticklabels(fracs.index, rotation=45, ha='right')
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Fraction of wrong predictions')
    ax.set_title(title)
    ax.legend(title='Predicted as', fontsize=9, ncol=3,
              loc='upper left', bbox_to_anchor=(0.0, 1.0))
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f'{x:.0%}'))
    plt.tight_layout()
    plt.savefig(path, bbox_inches='tight'); plt.close()


# ── summary table ─────────────────────────────────────────────────────────────
def print_summary_table(df: pd.DataFrame, group_col: str, ordered: list, label: str):
    rates = threshold_rates(df, group_col, ordered)
    print(f"\n{'':=<72}")
    print(f"  {label}")
    print(f"{'':=<72}")
    hdr = f"{'Group':<18} {'N':>6}  {'len≤5%':>7}  {'ang≤5%':>7}  {'both≤5%':>8}  {'brav%':>7}"
    print(hdr)
    print("-" * 60)
    for grp in rates.index:
        r = rates.loc[grp]
        print(f"{grp:<18} {int(r['n']):>6}  {r['len≤5%']:>6.1f}%  "
              f"{r['ang≤5%']:>6.1f}%  {r['both≤5%']:>7.1f}%  {r['brav_match']:>6.1f}%")


# ── dataset loader ────────────────────────────────────────────────────────────
def load_df(path: str) -> pd.DataFrame:
    df = pd.read_pickle(path)
    df = df.dropna(subset=['cctbx_true', 'cctbx_pred']).reset_index(drop=True)
    return add_metric_cols(df)


# ── dataset registry (same structure as make_comparison_plots.py) ─────────────
def get_datasets(base: str, suffix: str = '') -> list:
    def p(*parts):
        path = os.path.join(base, *parts)
        if suffix:
            path = path.replace('.pkl', f'{suffix}.pkl')
        return path

    return [
        ('mp20_correct_cag', 'MP20', {
            'Bispec': p('mp20_correct_cag', 'inversion_results_cctbx_bispec.pkl'),
            'Direct': p('mp20_correct_cag', 'inversion_results_cctbx_direct.pkl'),
        }),
        ('mp20_final_aug', 'MP20 Aug', {
            'Bispec': p('mp20_final_aug', 'inversion_results_cctbx_bispec.pkl'),
            'Direct': p('mp20_final_aug', 'inversion_results_cctbx_direct.pkl'),
        }),
        ('mpfull_aug', 'MPFull Aug', {
            'Bispec': p('mpfull_aug', 'inversion_results_cctbx_bispec.pkl'),
            'Direct': p('mpfull_aug', 'inversion_results_cctbx_direct.pkl'),
        }),
        ('MPFull', 'MPFull', {
            'Bispec': p('mpfull_crystalyze_cag', 'inversion_results_cctbx_bispec.pkl'),
            'Direct': p('mpfull_crystalyze_cag', 'inversion_results_cctbx_direct.pkl'),
        }),
    ]


DATASET_KEYS = {
    'mp20_correct_cag', 'mp20_final_aug', 'mpfull_aug', 'MPFull',
}


# ── main per-model runner ─────────────────────────────────────────────────────
def run_model(df: pd.DataFrame, model_name: str, plot_dir: str, label: str):
    os.makedirs(plot_dir, exist_ok=True)
    slug = model_name.lower().replace(' ', '_')

    present_bravais  = [b for b in BRAVAIS_ORDER if b in df['true_bravais'].values]
    present_systems  = [s for s in CRYSTAL_SYSTEM_ORDER if s in df['true_crystal_system'].values]

    # ── Bravais confusion ────────────────────────────────────────────────────
    all_bravais = [b for b in BRAVAIS_ORDER
                   if b in df['true_bravais'].values or b in df['pred_bravais'].values]
    cm_brav = confusion_matrix_norm(df, 'true_bravais', 'pred_bravais', all_bravais)
    plot_confusion(cm_brav,
                   f'Bravais confusion — {label} {model_name}',
                   os.path.join(plot_dir, f'confusion_bravais_{slug}.pdf'))

    # ── Crystal system confusion ─────────────────────────────────────────────
    cm_sys = confusion_matrix_norm(df, 'true_crystal_system', 'pred_crystal_system',
                                   CRYSTAL_SYSTEM_ORDER)
    plot_confusion(cm_sys,
                   f'Crystal system confusion — {label} {model_name}',
                   os.path.join(plot_dir, f'confusion_system_{slug}.pdf'))

    # ── Threshold fraction bars — Bravais ────────────────────────────────────
    rates_brav = threshold_rates(df, 'true_bravais', present_bravais)
    plot_threshold_bars(rates_brav,
                        f'Recovery rates per Bravais type — {label} {model_name}',
                        os.path.join(plot_dir, f'threshold_bravais_{slug}.pdf'))

    # ── Threshold fraction bars — crystal system ─────────────────────────────
    rates_sys = threshold_rates(df, 'true_crystal_system', present_systems)
    plot_threshold_bars(rates_sys,
                        f'Recovery rates per crystal system — {label} {model_name}',
                        os.path.join(plot_dir, f'threshold_system_{slug}.pdf'))

    # ── Failure categories — Bravais ─────────────────────────────────────────
    cats_brav = failure_category_rates(df, 'true_bravais', present_bravais)
    plot_failure_categories(cats_brav,
                            f'Failure categories per Bravais — {label} {model_name}',
                            os.path.join(plot_dir, f'failure_cat_bravais_{slug}.pdf'))

    # ── Failure categories — crystal system ──────────────────────────────────
    cats_sys = failure_category_rates(df, 'true_crystal_system', present_systems)
    plot_failure_categories(cats_sys,
                            f'Failure categories per crystal system — {label} {model_name}',
                            os.path.join(plot_dir, f'failure_cat_system_{slug}.pdf'))

    # ── Misprediction distribution ────────────────────────────────────────────
    plot_mispred_distribution(df,
                              f'Mispredictions: predicted-as distribution — {label} {model_name}',
                              os.path.join(plot_dir, f'mispred_dist_{slug}.pdf'))

    # ── Console summary ───────────────────────────────────────────────────────
    print_summary_table(df, 'true_bravais',        present_bravais, f'{label} {model_name} — Bravais')
    print_summary_table(df, 'true_crystal_system', present_systems, f'{label} {model_name} — Crystal System')

    print(f'  Saved plots to {plot_dir}/')


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cctbx-dir',
                    default=os.path.join(os.environ.get('POWDERXRD_DATA_ROOT', './data'),
                                         'final_test_results_cctbx'),
                    help='Base directory containing cctbx result pickles')
    ap.add_argument('--datasets', nargs='+', default=['mpfull_aug'],
                    choices=sorted(DATASET_KEYS),
                    help='Which datasets to process')
    ap.add_argument('--model', default=None,
                    help='Only process this model (e.g. Bispec or Direct). Default: all.')
    ap.add_argument('--suffix', default='',
                    help='Pkl suffix e.g. _delta001')
    ap.add_argument('--plot-dir', default=None,
                    help='Override output directory')
    args = ap.parse_args()

    all_datasets = {key: entry for key, entry in
                    [(e[0], e) for e in get_datasets(args.cctbx_dir, args.suffix)]}

    for ds_key in args.datasets:
        if ds_key not in all_datasets:
            print(f'Unknown dataset: {ds_key}')
            continue
        _, label, model_paths = all_datasets[ds_key]
        print(f'\n=== {label} ===')

        for model_name, pkl_path in model_paths.items():
            if args.model and model_name != args.model:
                continue
            if not os.path.exists(pkl_path):
                print(f'  [skip] {model_name}: {pkl_path} not found')
                continue
            print(f'  Loading {model_name}: {pkl_path}')
            df = load_df(pkl_path)
            print(f'  n={len(df)}')

            out_dir = args.plot_dir or os.path.join(
                args.cctbx_dir, ds_key.lower(), 'failure_analysis')
            run_model(df, model_name, out_dir, label)


if __name__ == '__main__':
    main()
