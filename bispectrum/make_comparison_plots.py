"""
Comparison plots for every dataset in final_test_results_cctbx/.

Metrics reported:
  - Mean and Median APE for lengths (Å) and angles (°), per Bravais type
  - MAE in physical units (Å for lengths, ° for angles)
  - Volume MAPE (median)
  - Bravais type match rate (pred_bravais == true_bravais)
  - Symmetry-aware angle APE: only free angles per Bravais type

Usage:
    python make_comparison_plots.py [--cctbx-dir PATH]
"""

import argparse
import os
import pickle
import re
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pymatgen.core import Lattice


# ── RRUFF filtering helpers ───────────────────────────────────────────────────
class _FlexibleUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        try:
            return super().find_class(module, name)
        except (AttributeError, ModuleNotFoundError):
            class _Generic:
                def __init__(self, *_): pass
            return _Generic


_DISORDERED_IDS = None  # set or None
_ATOM_COUNTS = None    # dict (pattern_id -> n_atoms) or None

def _load_rruff_metadata(rruff_pkl: str):
    """Load disordered IDs and atom counts from rruff_data_dict.pkl (cached)."""
    global _DISORDERED_IDS, _ATOM_COUNTS
    if _DISORDERED_IDS is not None:
        return
    if not os.path.exists(rruff_pkl):
        print(f'  [warn] rruff_data_dict not found at {rruff_pkl}, skipping RRUFF filters')
        _DISORDERED_IDS = set()
        _ATOM_COUNTS = {}
        return
    with open(rruff_pkl, 'rb') as f:
        data = _FlexibleUnpickler(f).load()
    disordered = set()
    atom_counts = {}
    for key, entry in data.items():
        if entry.atoms is not None and hasattr(entry.atoms, 'occupancy'):
            if (entry.atoms['occupancy'] < 1.0).any():
                disordered.add(key)
            atom_counts[key] = len(entry.atoms)
    print(f'  Loaded RRUFF metadata: {len(disordered)} disordered / {len(data)} total entries')
    _DISORDERED_IDS = disordered
    _ATOM_COUNTS = atom_counts

def _load_disordered_ids(rruff_pkl: str) -> set:
    _load_rruff_metadata(rruff_pkl)
    return _DISORDERED_IDS


def _is_multiphase(pattern_id: str) -> bool:
    return bool(re.search(r'R\d+-9__', pattern_id))

# ── constants ────────────────────────────────────────────────────────────────
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

CRYSTAL_SYSTEM_ORDER = ["Triclinic", "Monoclinic", "Orthorhombic", "Tetragonal",
                        "Trigonal", "Hexagonal", "Cubic"]

FREE_ANGLE_IDX_SYSTEM = {
    "Triclinic":    [0, 1, 2],
    "Monoclinic":   [1],
    "Orthorhombic": [],
    "Tetragonal":   [],
    "Trigonal":     [],
    "Hexagonal":    [],
    "Cubic":        [],
}

# Indices (into [alpha, beta, gamma]) of the free angles per Bravais type.
# Fixed angles (constrained to 90° or 120° by symmetry) are excluded.
FREE_ANGLE_IDX = {
    "aP": [0, 1, 2],   # all free
    "mP": [1],          # only beta
    "mC": [1],
    "oP": [],           # all 90°
    "oC": [],
    "oI": [],
    "oF": [],
    "tP": [],
    "tI": [],
    "hP": [],           # gamma=120° fixed; alpha=beta=90°
    "hR": [],           # in hexagonal setting all fixed; rhombohedral setting has one free
    "cP": [],
    "cI": [],
    "cF": [],
}

plt.rcParams.update({
    'font.size':        18,
    'axes.titlesize':   20,
    'axes.labelsize':   18,
    'xtick.labelsize':  16,
    'ytick.labelsize':  16,
    'legend.fontsize':  16,
    'figure.titlesize': 22,
})

def angle_diff(pred, true):
    diff = np.abs(pred - true)
    return np.minimum(diff, 360 - diff)

# ── per-pair error vectors ────────────────────────────────────────────────────
def param_errors(true_lat: Lattice, pred_lat: Lattice):
    true_p = np.array([true_lat.a, true_lat.b, true_lat.c,
                       true_lat.alpha, true_lat.beta, true_lat.gamma])
    pred_p = np.array([pred_lat.a, pred_lat.b, pred_lat.c,
                       pred_lat.alpha, pred_lat.beta, pred_lat.gamma])

    abs_err = np.zeros_like(true_p)

    # lengths (first 3)
    abs_err[:3] = np.abs(pred_p[:3] - true_p[:3])

    # angles (last 3) — use periodic distance
    diff = np.abs(pred_p[3:] - true_p[3:])
    abs_err[3:] = np.minimum(diff, 360 - diff)

    ape = abs_err / np.abs(true_p) * 100
    return abs_err, ape


def volume_ape(true_lat: Lattice, pred_lat: Lattice) -> float:
    return abs(pred_lat.volume - true_lat.volume) / true_lat.volume * 100


# ── aggregate over a DataFrame ────────────────────────────────────────────────
def compute_errors(df, group_col='true_bravais'):
    """
    Returns {group_key: {'ape': (n,6), 'mae': (n,6), 'vol_ape': (n,)}}
    """
    out = {}
    for bravais, group in df.groupby(group_col):
        ape_list, mae_list, vol_list = [], [], []
        for t, p in zip(group['cctbx_true'], group['cctbx_pred']):
            ae, ape = param_errors(t, p)
            ape_list.append(ape)
            mae_list.append(ae)
            vol_list.append(volume_ape(t, p))
        out[bravais] = {
            'ape':     np.array(ape_list),   # (n, 6)
            'mae':     np.array(mae_list),   # (n, 6)  units: Å / deg
            'vol_ape': np.array(vol_list),   # (n,)
        }
    return out


def build_summary(errs: dict, free_angle_idx: dict = None, order: list = None) -> pd.DataFrame:
    """
    One row per group key (Bravais type or crystal system) with error metric columns.
    Pass free_angle_idx=FREE_ANGLE_IDX_SYSTEM and order=CRYSTAL_SYSTEM_ORDER for
    crystal-system-level summaries.
    """
    if free_angle_idx is None:
        free_angle_idx = FREE_ANGLE_IDX
    if order is None:
        order = BRAVAIS_ORDER
    rows = {}
    for bravais, d in errs.items():
        ape = d['ape']   # (n,6)
        mae = d['mae']   # (n,6)
        free = free_angle_idx.get(bravais, [0, 1, 2])

        row = {
            'Mean Length APE (%)':   ape[:, :3].mean(),
            'Median Length APE (%)': np.median(ape[:, :3]),
            'MAE Length (Å)':        mae[:, :3].mean(),
            'Mean Angle APE (%)':    ape[:, 3:].mean(),
            'Median Angle APE (%)':  np.median(ape[:, 3:]),
            'MAE Angle (°)':         mae[:, 3:].mean(),
            'Mean Volume APE (%)': np.mean(d['vol_ape']),
            'N': len(ape),
        }
        # Symmetry-aware: only free angles
        if free:
            free_cols = [3 + i for i in free]
            row['Sym Median Angle APE (%)'] = np.median(ape[:, free_cols])
            row['Sym MAE Angle (°)']        = mae[:, free_cols].mean()
        else:
            row['Sym Median Angle APE (%)'] = np.nan
            row['Sym MAE Angle (°)']        = np.nan

        rows[bravais] = row

    return pd.DataFrame(rows).T.reindex(
        [b for b in order if b in rows]
    )


def compute_match_rate(df: pd.DataFrame, group_col: str = 'true_bravais') -> pd.Series:
    """Fraction of rows where predicted group == true group, per group key."""
    if 'pred_bravais' not in df.columns:
        return pd.Series(dtype=float)
    rates = {}
    if group_col == 'true_crystal_system':
        for system, group in df.groupby('true_crystal_system'):
            pred_sys = group['pred_bravais'].map(BRAVAIS_TO_CRYSTAL_SYSTEM)
            rates[system] = (pred_sys == system).mean()
    else:
        for bravais, group in df.groupby('true_bravais'):
            rates[bravais] = (group['pred_bravais'] == bravais).mean()
    return pd.Series(rates)


# ── plotting ─────────────────────────────────────────────────────────────────
# Fixed color per model name so colors are consistent across all datasets.
# Bispec = orange, Direct = blue (matching original notebook convention).
MODEL_COLORS = {
    'Bispec':           'darkorange',
    'Direct':           'steelblue',
    'Crystalyze':       'seagreen',
    'mp20 Bispec':      'darkorange',
    'mp20aug Bispec':   'seagreen',
    'mpfull Bispec':    'crimson',
    'mp20 Direct':      'darkorange',
    'mp20aug Direct':   'seagreen',
    'mpfull Direct':    'crimson',
    'MP20 CAG Direct':  'steelblue',
    'MP20 CAG Bispec':  'darkorange',
    'Direct No Aug':    'steelblue',
    'Bispec No Aug':    'darkorange',
    'MP20 Aug Direct':  'steelblue',
    'MP20 Aug Bispec':  'darkorange',
    'MPFull Aug Direct': 'saddlebrown',
    'MPFull Aug Bispec': 'seagreen',
}
_FALLBACK_COLORS = ['steelblue', 'darkorange', 'seagreen', 'crimson', 'mediumpurple', 'saddlebrown']

def _model_color(name: str, index: int) -> str:
    return MODEL_COLORS.get(name, _FALLBACK_COLORS[index % len(_FALLBACK_COLORS)])


def _bar_panel(ax, x, ordered, summaries, metric, width, offsets, title):
    """Draw grouped bars with a small intra-group gap and return mean values."""
    means = {}
    for i, (name, summary) in enumerate(summaries.items()):
        color = _model_color(name, i)
        vals = summary.reindex(ordered)[metric].fillna(0)
        ax.bar(x + offsets[i], vals, width, label=name, color=color)
        means[name] = vals[vals > 0].mean()
    ax.set_xticks(x); ax.set_xticklabels(ordered, rotation=45, fontsize=22)
    ax.set_title(title, fontsize=22)
    ax.legend(fontsize=18, loc='upper left', bbox_to_anchor=(1.01, 1.0),
              bbox_transform=ax.transAxes, borderaxespad=0)
    return means


def _means_table(ax_table, means: dict, ylabel: str):
    """Render a small table of overall-mean values below the bar chart axis."""
    ax_table.axis('off')
    col_labels = list(means.keys())
    cell_text  = [[f'{means[m]:.2f}' for m in col_labels]]
    tbl = ax_table.table(
        cellText=cell_text,
        colLabels=col_labels,
        rowLabels=[ylabel],
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(13)
    tbl.scale(1, 1.6)


def _group_offsets(n: int, width: float, gap: float):
    """Offsets that keep pairs visually grouped with a small gap between models."""
    raw = np.linspace(-(n-1)/2, (n-1)/2, n) * width
    if n <= 1:
        return raw
    centre = (raw[0] + raw[-1]) / 2
    shifts = np.where(raw < centre, -gap / 2, np.where(raw > centre, gap / 2, 0.0))
    return raw + shifts


def make_bar_plots(models: dict, plot_dir: str, label: str,
                   all_summaries: dict, ordered: list,
                   group_label: str = 'Bravais Lattice', slug: str = 'bravais'):
    x = np.arange(len(ordered))
    n = len(models)
    width = 0.8 / n
    gap   = 0.03          # small visual gap between bars in each group
    offsets = _group_offsets(n, width, gap)

    metrics = [
        ('Mean Length APE (%)',         'MAPE Length (%)',               'mean_length_ape'),
        ('Median Length APE (%)',        'Median Length APE (%)',         'median_length_ape'),
        ('MAE Length (Å)',               'MAE Length (Å)',                'mae_length'),
        ('Mean Angle APE (%)',           'MAPE Angle (%)',                'mean_angle_ape'),
        ('Median Angle APE (%)',         'Median Angle APE (%)',          'median_angle_ape'),
        ('Sym Median Angle APE (%)',     'Sym-aware Median Angle APE (%)', 'sym_median_angle_ape'),
        ('MAE Angle (°)',                'MAE Angle (°)',                 'mae_angle'),
        ('Mean Volume APE (%)',          'MAPE Volume (%)',           'mean_vol_ape'),
    ]

    for col, ylabel, metric_slug in metrics:
        fig, (ax, ax_tbl) = plt.subplots(
            2, 1,
            figsize=(11,4.5),
            gridspec_kw={'height_ratios': [5, 1]},
        )
        means = _bar_panel(ax, x, ordered, all_summaries, col, width, offsets,
                           f'{ylabel} per {group_label} — {label}')
        ax.set_ylabel(ylabel, fontsize=20)
        ax.tick_params(axis='y', labelsize=18)
        _means_table(ax_tbl, means, 'Overall mean')
        plt.tight_layout()
        path = os.path.join(plot_dir, f'{metric_slug}_by_{slug}.pdf')
        plt.savefig(path, bbox_inches='tight'); plt.close()
        print(f'  Saved {path}')


def _flat_metrics(df: pd.DataFrame):
    """Compute flat (micro-average) error arrays over all samples in df."""
    rows = [(t, p, b) for t, p, b in
            zip(df['cctbx_true'], df['cctbx_pred'], df['true_bravais'])
            if t is not None and p is not None]
    if not rows:
        return None
    ae_all  = np.array([param_errors(t, p)[0] for t, p, _ in rows])
    ape_all = np.array([param_errors(t, p)[1] for t, p, _ in rows])
    vol_all = np.array([volume_ape(t, p)       for t, p, _ in rows])
    sym_ae_per_row = []
    for i, (_, _, b) in enumerate(rows):
        free_cols = [3 + j for j in FREE_ANGLE_IDX.get(b, [])]
        sym_ae_per_row.append(ae_all[i, free_cols].mean() if free_cols else np.nan)
    sym_ae = np.array([v for v in sym_ae_per_row if not np.isnan(v)])
    return ae_all, ape_all, vol_all, sym_ae


def make_overall_summary_table(all_dfs: dict, plot_dir: str, label: str):
    """Single PDF table of flat (micro-average) MAE/APE metrics across all models."""
    model_names = list(all_dfs.keys())
    row_labels = [
        'N',
        'MAE Length (Å)',
        'Median AE Length (Å)',
        'MAE Angle (°)',
        'Median AE Angle (°)',
        'Sym MAE Angle (°)',
        'Mean Length APE (%)',
        'Median Length APE (%)',
        'Mean Angle APE (%)',
        'Median Angle APE (%)',
        'Mean Volume APE (%)',
    ]
    cell_text = []
    for row_label in row_labels:
        row = []
        for df in all_dfs.values():
            result = _flat_metrics(df)
            if result is None:
                row.append('—')
                continue
            ae_all, ape_all, vol_all, sym_ae = result
            if row_label == 'N':
                row.append(str(len(ae_all)))
            elif row_label == 'MAE Length (Å)':
                row.append(f'{ae_all[:, :3].mean():.2f}')
            elif row_label == 'Median AE Length (Å)':
                row.append(f'{np.median(ae_all[:, :3]):.2f}')
            elif row_label == 'MAE Angle (°)':
                row.append(f'{ae_all[:, 3:].mean():.2f}')
            elif row_label == 'Median AE Angle (°)':
                row.append(f'{np.median(ae_all[:, 3:]):.2f}')
            elif row_label == 'Sym MAE Angle (°)':
                row.append(f'{sym_ae.mean():.2f}' if len(sym_ae) else '—')
            elif row_label == 'Mean Length APE (%)':
                row.append(f'{ape_all[:, :3].mean():.2f}')
            elif row_label == 'Median Length APE (%)':
                row.append(f'{np.median(ape_all[:, :3]):.2f}')
            elif row_label == 'Mean Angle APE (%)':
                row.append(f'{ape_all[:, 3:].mean():.2f}')
            elif row_label == 'Median Angle APE (%)':
                row.append(f'{np.median(ape_all[:, 3:]):.2f}')
            elif row_label == 'Mean Volume APE (%)':
                row.append(f'{vol_all.mean():.2f}')
            else:
                row.append('—')
        cell_text.append(row)

    fig, ax = plt.subplots(figsize=(max(6, 2 * len(model_names) + 2), 4))
    ax.axis('off')
    tbl = ax.table(
        cellText=cell_text,
        rowLabels=row_labels,
        colLabels=model_names,
        loc='center',
        cellLoc='center',
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(11)
    tbl.scale(1, 1.5)
    ax.set_title(f'Overall metrics — {label}', fontsize=13, pad=12)
    plt.tight_layout()
    path = os.path.join(plot_dir, 'overall_summary_table.pdf')
    plt.savefig(path, bbox_inches='tight'); plt.close()
    print(f'  Saved {path}')


def make_match_rate_plot(models: dict, plot_dir: str, label: str, ordered: list,
                         group_label: str = 'Bravais Lattice', slug: str = 'bravais',
                         group_col: str = 'true_bravais'):
    rates = {name: compute_match_rate(df, group_col=group_col) for name, df in models.items()}
    if all(r.empty for r in rates.values()):
        return

    x = np.arange(len(ordered))
    n = len(models)
    width = 0.6 / n
    gap   = 0.04
    offsets = _group_offsets(n, width, gap)

    fig, ax = plt.subplots(figsize=(18, 5))
    for i, (name, rate) in enumerate(rates.items()):
        vals = [rate.get(b, np.nan) for b in ordered]
        ax.bar(x + offsets[i], vals, width, label=name, color=_model_color(name, i))
    ax.set_xticks(x); ax.set_xticklabels(ordered, rotation=45)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel('Match rate')
    ax.set_title(f'{group_label} match rate (pred == true) — {label}')
    ax.legend(fontsize=25)
    plt.tight_layout()
    path = os.path.join(plot_dir, f'{slug}_match_rate.pdf')
    plt.savefig(path, bbox_inches='tight'); plt.close()
    print(f'  Saved {path}')


def make_cdf_plots(models: dict, plot_dir: str, label: str):
    """Side-by-side CDFs of length MAE and angle MAE across all models."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

    panels = [
        (axes[0], slice(0, 3), 'Mean length absolute error (Å)', 5.0,
         [(0.5, 'refinement init.'), (1.0, 'indexing limit')]),
        (axes[1], slice(3, 6), 'Mean angle absolute error (°)',  20.0,
         [(2.0, 'good indexing'), (5.0, 'indexing tol.')]),
    ]

    for ax, cols, xlabel, xlim, cutoffs in panels:
        for i, (name, df) in enumerate(models.items()):
            pairs = [(t, p) for t, p in zip(df['cctbx_true'], df['cctbx_pred'])
                     if t is not None and p is not None]
            vals = np.array([
                np.abs(np.array([t.a,t.b,t.c,t.alpha,t.beta,t.gamma])
                       - np.array([p.a,p.b,p.c,p.alpha,p.beta,p.gamma]))[cols].mean()
                for t, p in pairs
            ])
            sorted_vals = np.sort(vals)
            cdf = np.arange(1, len(sorted_vals) + 1) / len(sorted_vals) * 100
            ls = ':' if 'No Aug' in name else '-'
            ax.plot(sorted_vals, cdf, label=name, color=_model_color(name, i), lw=2, linestyle=ls)
        for x, _ in cutoffs:
            ax.axvline(x, color='gray', lw=1, linestyle='--', alpha=0.7)
        ax.set_xlabel(xlabel, fontsize=16)
        ax.set_xlim(0, xlim)
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=14)

    axes[0].set_ylabel('Cumulative % of structures', fontsize=16)
    axes[1].legend(fontsize=14, loc='upper left', bbox_to_anchor=(1.02, 1.0),
                   bbox_transform=axes[1].transAxes, borderaxespad=0)
    fig.suptitle(f'Error CDF — {label}', fontsize=16)
    plt.tight_layout()
    path = os.path.join(plot_dir, 'error_cdf.pdf')
    plt.savefig(path, bbox_inches='tight'); plt.close()
    print(f'  Saved {path}')


def make_boxplots(models: dict, plot_dir: str, label: str,
                  all_errs: dict, ordered: list,
                  group_label: str = 'Bravais Lattice', slug: str = 'bravais'):
    for metric_name, key, cols, ylabel in [
        ('length_ape',  'ape', slice(0, 3), 'Length APE (%)'),
        ('angle_ape',   'ape', slice(3, 6), 'Angle APE (%)'),
        ('length_mae',  'mae', slice(0, 3), 'Length MAE (Å)'),
        ('angle_mae',   'mae', slice(3, 6), 'Angle MAE (°)'),
    ]:
        n = len(models)
        _, axes = plt.subplots(n, 1, figsize=(16, 6 * n), sharex=True)
        if n == 1:
            axes = [axes]
        for ax, (name, errs) in zip(axes, all_errs.items()):
            data = [errs[b][key][:, cols].flatten() if b in errs else np.array([0])
                    for b in ordered]
            ax.boxplot(data, labels=ordered)
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel} ({name}) — {label}')
        plt.tight_layout()
        path = os.path.join(plot_dir, f'{metric_name}_boxplot_by_{slug}.pdf')
        plt.savefig(path, bbox_inches='tight'); plt.close()
        print(f'  Saved {path}')


def _hexbin_panel(ax, fig, true_vals, pred_vals, xlabel, ylabel, title,
                  add_colorbar=False, scatter_below=300, vmax=None):
    """Draw one parity panel; returns (mae, r2).

    Uses a hexbin density plot when there are enough points, otherwise
    falls back to a plain scatter (hexbin looks empty/sparse at low N).
    """
    all_vals = np.concatenate([true_vals, pred_vals])
    lo = np.percentile(all_vals, 1)
    hi = np.percentile(all_vals, 99)
    pad = (hi - lo) * 0.02
    lo -= pad; hi += pad

    use_scatter = len(true_vals) <= scatter_below
    if use_scatter:
        ax.scatter(true_vals, pred_vals, s=25, color='steelblue',
                   alpha=0.7, edgecolors='none', zorder=4)
    else:
        hb = ax.hexbin(true_vals, pred_vals, gridsize=35, cmap='Blues',
                       mincnt=1, linewidths=0.1, bins='log',
                       extent=[lo, hi, lo, hi],
                       vmin=1 if vmax else None, vmax=vmax)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        if add_colorbar:
            fig.colorbar(hb, cax=cax, label='log₁₀(count)')
        else:
            cax.set_visible(False)
    ax.plot([lo, hi], [lo, hi], 'r--', lw=1.2, zorder=5)
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_aspect('equal')

    mae = np.mean(np.abs(true_vals - pred_vals))
    r2  = pearsonr(true_vals, pred_vals)[0] ** 2
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18)
    if use_scatter:
        ax.set_title(f'{title}\nMAE={mae:.3f}  R²={r2:.3f}', fontsize=16)
    else:
        ax.set_title(title, fontsize=18)
        ax.text(0.05, 0.95, f'MAE={mae:.3f}\nR²={r2:.3f}',
                transform=ax.transAxes, fontsize=18, va='top',
                bbox=dict(boxstyle='round,pad=0.3', fc='white', alpha=0.75))
    ax.tick_params(labelsize=16)
    return mae, r2


def make_scatter_plots(models: dict, plot_dir: str, label: str):
    """Length (a, b, c) + volume parity plots — one figure per model."""
    params = [('a', 'Å'), ('b', 'Å'), ('c', 'Å')]
    for name, df in models.items():
        fig, axes = plt.subplots(1, 4, figsize=(18, 4.5))

        for i, (param, unit) in enumerate(params):
            true_vals = np.array([getattr(lat, param) for lat in df['cctbx_true']])
            pred_vals = np.array([getattr(lat, param) for lat in df['cctbx_pred']])
            _hexbin_panel(axes[i], fig, true_vals, pred_vals,
                          f'True {param} ({unit})', f'Pred {param} ({unit})', param)

        true_vols = np.array([lat.volume for lat in df['cctbx_true']])
        pred_vols = np.array([lat.volume for lat in df['cctbx_pred']])
        _hexbin_panel(axes[3], fig, true_vols, pred_vols,
                      'True volume (Å³)', 'Pred volume (Å³)', 'Volume',
                      add_colorbar=True)

        fig.suptitle(f'Length & volume parity — {label} ({name})', fontsize=18)
        plt.tight_layout()
        path = os.path.join(plot_dir, f'scatter_{name.lower().replace(" ", "_")}.pdf')
        plt.savefig(path, bbox_inches='tight'); plt.close()
        print(f'  Saved {path}')


# Crystal systems that have at least one non-trivially-fixed angle.
# Show all 3 angle panels per system: free angles reveal prediction quality,
# constrained ones (expected ~90°) show whether symmetry is respected.
_ANGLE_SYSTEMS = ['Triclinic', 'Monoclinic']
_ANGLE_PARAMS  = [('alpha', 'α'), ('beta', 'β'), ('gamma', 'γ')]


def make_angle_parity_plots(models: dict, plot_dir: str, label: str):
    """One figure per crystal system; rows = models, cols = α β γ."""
    # find which systems appear in any model
    all_systems = set()
    for df in models.values():
        all_systems |= set(df['true_crystal_system'].unique())
    systems_present = [s for s in _ANGLE_SYSTEMS if s in all_systems]

    for sys_name in systems_present:
        model_names = [n for n, df in models.items()
                       if sys_name in df['true_crystal_system'].values]
        n_rows = len(model_names)
        if n_rows == 0:
            continue

        fig, axes = plt.subplots(n_rows, 3, figsize=(18, 4.5 * n_rows), squeeze=False)

        for row, name in enumerate(model_names):
            sub = models[name][models[name]['true_crystal_system'] == sys_name]
            n = len(sub)
            for col, (param, sym) in enumerate(_ANGLE_PARAMS):
                is_last_col = (col == 2)
                true_vals = np.array([getattr(lat, param) for lat in sub['cctbx_true']])
                pred_vals = np.array([getattr(lat, param) for lat in sub['cctbx_pred']])
                _hexbin_panel(axes[row, col], fig, true_vals, pred_vals,
                              f'True {sym} (°)', f'Pred {sym} (°)',
                              f'{name} — {sym}  (N={n})',
                              add_colorbar=is_last_col)

        fig.suptitle(f'{sys_name} angle parity — {label}', fontsize=18)
        plt.tight_layout()
        slug = sys_name.lower()
        path = os.path.join(plot_dir, f'angle_parity_{slug}.pdf')
        plt.savefig(path, bbox_inches='tight'); plt.close()
        print(f'  Saved {path}')


def save_summary_csv(all_summaries: dict, plot_dir: str):
    """Write one CSV per model with all metrics."""
    for name, summary in all_summaries.items():
        path = os.path.join(plot_dir, f'summary_{name.lower().replace(" ","_")}.csv')
        summary.to_csv(path, float_format='%.4f')
        print(f'  Saved {path}')


# ── dataset runner ────────────────────────────────────────────────────────────
def _id_col(df: pd.DataFrame) -> str:
    """Return the ID column name used in this results dataframe."""
    for col in ('pattern_id', 'material_id', 'mat_id'):
        if col in df.columns:
            return col
    return 'mat_id'


def _normalize_id(id_str: str) -> str:
    """Strip __RXXXXXX suffix so plain mineral names and keyed IDs compare equal."""
    return id_str.split('__')[0]


_DISPLAY_NAMES = {
    'mp20_correct_cag':     'MP20',
    'mp20_final_aug':       'MP20 Aug',
    'rruff_crystalyze':     'RRUFF',
    'mpfull_aug':           'MPFull Aug',
    'rruff_alpha':          'RRUFF Alpha MPFull Aug',
    'rruff_alpha_crystalyze': 'RRUFF Alpha MPFull No Aug',
}

def run_dataset(label: str, models: dict, plot_dir: str, rruff_pkl: str = None):
    display = _DISPLAY_NAMES.get(label, label)
    is_rruff = label.startswith('rruff') and not label.startswith('rruff_alpha')
    disordered_ids = _load_disordered_ids(rruff_pkl) if (is_rruff and rruff_pkl) else set()
    loaded = {}
    for name, path in models.items():
        if path is None or not os.path.exists(path):
            print(f'  [skip] {name}: {path}')
            continue
        df = pd.read_pickle(path)
        df = df.dropna(subset=['cctbx_true', 'cctbx_pred']).reset_index(drop=True)
        if is_rruff:
            col = _id_col(df)
            before = len(df)
            df = df[~df[col].apply(_is_multiphase)].reset_index(drop=True)
            n_multi = before - len(df)
            before2 = len(df)
            df = df[~df[col].isin(disordered_ids)].reset_index(drop=True)
            n_dis = before2 - len(df)
            # < 20 atoms filter
            before3 = len(df)
            if _ATOM_COUNTS:
                df = df[df[col].map(lambda x: _ATOM_COUNTS.get(x, 0)) < 20].reset_index(drop=True)
            n_atoms = before3 - len(df)
            # one pattern per mineral (keep first alphabetically by pattern_id)
            before4 = len(df)
            df = df.assign(_mineral=df[col].str.split('__').str[0])
            df = df.sort_values(col).drop_duplicates(subset='_mineral').drop(columns='_mineral').reset_index(drop=True)
            n_mineral = before4 - len(df)
            print(f'  Loaded {name}: {len(df)} rows '
                  f'({n_multi} multiphase, {n_dis} disordered, {n_atoms} >20 atoms, {n_mineral} dup mineral removed)')
        else:
            print(f'  Loaded {name}: {len(df)} rows')
        loaded[name] = df

    if not loaded:
        print(f'  No files found for {label}, skipping.')
        return

    # Restrict to the intersection of IDs across all models so comparisons are fair.
    # Normalize IDs by stripping __RXXXXXX suffix so plain mineral names and
    # keyed IDs (e.g. "Ferberite" vs "Ferberite__R040044") compare equal.
    # For is_rruff=False datasets (e.g. rruff_alpha), use exact IDs since all
    # models share the same ID format.
    if len(loaded) > 1:
        id_fn = (lambda x: x) if not is_rruff else _normalize_id
        norm_sets = []
        for name, df in loaded.items():
            col = _id_col(df)
            norm_sets.append(set(df[col].map(id_fn)))
        common_norm = norm_sets[0].intersection(*norm_sets[1:])
        for name, df in loaded.items():
            col = _id_col(df)
            before = len(df)
            loaded[name] = df[df[col].map(id_fn).isin(common_norm)].reset_index(drop=True)
            after = len(loaded[name])
            if before != after:
                print(f'  [{name}] restricted to common IDs: {before} -> {after} rows')
        print(f'  Common IDs across models: {len(common_norm)}')

    os.makedirs(plot_dir, exist_ok=True)

    # Add crystal system column to all loaded dataframes
    for name, df in loaded.items():
        df['true_crystal_system'] = df['true_bravais'].map(BRAVAIS_TO_CRYSTAL_SYSTEM).fillna('Unknown')
        loaded[name] = df

    # ── per Bravais lattice ──────────────────────────────────────────────────
    all_errs      = {name: compute_errors(df)       for name, df in loaded.items()}
    all_summaries = {name: build_summary(errs)       for name, errs in all_errs.items()}

    all_bravais = set()
    for s in all_summaries.values():
        all_bravais |= set(s.index)
    ordered_bravais = [b for b in BRAVAIS_ORDER if b in all_bravais]

    make_bar_plots(loaded, plot_dir, display, all_summaries, ordered_bravais,
                   group_label='Bravais Lattice', slug='bravais')
    make_overall_summary_table(loaded, plot_dir, display)
    make_match_rate_plot(loaded, plot_dir, display, ordered_bravais,
                         group_label='Bravais Lattice', slug='bravais', group_col='true_bravais')
    make_boxplots(loaded, plot_dir, display, all_errs, ordered_bravais,
                  group_label='Bravais Lattice', slug='bravais')
    save_summary_csv(all_summaries, plot_dir)

    # ── per crystal system ───────────────────────────────────────────────────
    all_errs_sys      = {name: compute_errors(df, group_col='true_crystal_system')
                         for name, df in loaded.items()}
    all_summaries_sys = {name: build_summary(errs, free_angle_idx=FREE_ANGLE_IDX_SYSTEM,
                                             order=CRYSTAL_SYSTEM_ORDER)
                         for name, errs in all_errs_sys.items()}

    all_systems = set()
    for s in all_summaries_sys.values():
        all_systems |= set(s.index)
    ordered_sys = [s for s in CRYSTAL_SYSTEM_ORDER if s in all_systems]

    make_bar_plots(loaded, plot_dir, display, all_summaries_sys, ordered_sys,
                   group_label='Crystal System', slug='system')
    make_match_rate_plot(loaded, plot_dir, display, ordered_sys,
                         group_label='Crystal System', slug='system', group_col='true_crystal_system')
    make_boxplots(loaded, plot_dir, display, all_errs_sys, ordered_sys,
                  group_label='Crystal System', slug='system')

    make_scatter_plots(loaded, plot_dir, display)
    make_angle_parity_plots(loaded, plot_dir, display)
    make_cdf_plots(loaded, plot_dir, display)
    print(f'  Done: {label}\n')


# ── dataset registry ─────────────────────────────────────────────────────────
def get_datasets(base: str, no_crystalyze: bool = False, suffix: str = '') -> list:
    def p(*parts):
        path = os.path.join(base, *parts)
        if suffix:
            path = path.replace('.pkl', f'{suffix}.pkl')
        return path

    mp20_models = {
        'Direct':     p('mp20',       'inversion_results_cctbx_direct.pkl'),
        'Bispec':     p('mp20',       'inversion_results_cctbx_bispec.pkl'),
    }
    if not no_crystalyze:
        mp20_models['Crystalyze'] = p('crystalyze_uvw', 'inversion_results_cctbx_crystalyze_uvw.pkl')

    return [
        ('mp20', mp20_models, p('mp20', 'comparison_plots')),

        ('mp20_aug', {
            'Direct': p('mp20_aug', 'inversion_results_cctbx_direct_new.pkl'),
            'Bispec': p('mp20_aug', 'inversion_results_cctbx_bispec.pkl'),
        }, p('mp20_aug', 'comparison_plots')),

        ('mpfull', {
            'Direct': p('mpfull', 'inversion_results_cctbx_direct.pkl'),
            'Bispec': p('mpfull', 'inversion_results_cctbx_bispec.pkl'),
        }, p('mpfull', 'comparison_plots')),

        ('rruff_mp20', {
            'Bispec': p('rruff', 'mp20_bispec', 'inversion_results_cctbx.pkl'),
            'Direct': p('rruff', 'mp20_direct', 'inversion_results_cctbx.pkl'),
        }, p('rruff', 'mp20_bispec', 'comparison_plots')),

        ('rruff_mp20aug', {
            'Bispec': p('rruff', 'mp20aug_bispec', 'inversion_results_cctbx.pkl'),
            'Direct': p('rruff', 'mp20aug_direct', 'inversion_results_cctbx.pkl'),
        }, p('rruff', 'mp20aug_bispec', 'comparison_plots')),

        ('rruff_mpfull', {
            'Bispec': p('rruff', 'mpfull_bispec', 'inversion_results_cctbx.pkl'),
            'Direct': p('rruff', 'mpfull_direct', 'inversion_results_cctbx.pkl'),
        }, p('rruff', 'mpfull_bispec', 'comparison_plots')),

        ('rruff_processed_xy_mp20', {
            'Bispec': p('rruff', 'processed_xy', 'mp20_bispec',  'inversion_results_cctbx.pkl'),
            'Direct': p('rruff', 'processed_xy', 'mp20_direct',  'inversion_results_cctbx.pkl'),
        }, p('rruff', 'processed_xy', 'mp20_bispec', 'comparison_plots')),

        ('rruff_processed_xy_mp20aug', {
            'Bispec': p('rruff', 'processed_xy', 'mp20aug_bispec', 'inversion_results_cctbx.pkl'),
            'Direct': p('rruff', 'processed_xy', 'mp20aug_direct', 'inversion_results_cctbx.pkl'),
        }, p('rruff', 'processed_xy', 'mp20aug_bispec', 'comparison_plots')),

        ('rruff_processed_xy_mp20_valaug', {
            'Bispec': p('rruff', 'processed_xy', 'mp20_valaug_bispec', 'inversion_results_cctbx.pkl'),
            'Direct': p('rruff', 'processed_xy', 'mp20_valaug_direct', 'inversion_results_cctbx.pkl'),
        }, p('rruff', 'processed_xy', 'mp20_valaug_bispec', 'comparison_plots')),

        ('rruff_processed_xy_mpfull', {
            'Bispec': p('rruff', 'processed_xy', 'mpfull_bispec', 'inversion_results_cctbx.pkl'),
            'Direct': p('rruff', 'processed_xy', 'mpfull_direct', 'inversion_results_cctbx.pkl'),
        }, p('rruff', 'processed_xy', 'mpfull_bispec', 'comparison_plots')),

        ('mp20_final_aug', {
            'Direct': p('mp20_final_aug', 'inversion_results_cctbx_direct.pkl'),
            'Bispec': p('mp20_final_aug', 'inversion_results_cctbx_bispec.pkl'),
        }, p('mp20_final_aug', 'comparison_plots')),

        ('mpfull_aug', {
            'Direct': p('mpfull_aug', 'inversion_results_cctbx_direct.pkl'),
            'Bispec': p('mpfull_aug', 'inversion_results_cctbx_bispec.pkl'),
        }, p('mpfull_aug', 'comparison_plots')),

        ('MPFull', {
            'Direct': p('mpfull_crystalyze_cag', 'inversion_results_cctbx_direct.pkl'),
            'Bispec': p('mpfull_crystalyze_cag', 'inversion_results_cctbx_bispec.pkl'),
        }, p('mpfull_crystalyze_cag', 'comparison_plots')),

        ('rruff_crystalyze', {
            'Crystalyze':    p('rruff_crystalyze', 'crystalyze',       'inversion_results_cctbx.pkl'),
            'Direct No Aug': p('rruff_crystalyze', 'mp20_correct_cag', 'inversion_results_cctbx_direct.pkl'),
            'Bispec No Aug': p('rruff_crystalyze', 'mp20_correct_cag', 'inversion_results_cctbx_bispec.pkl'),
            'Direct':        p('rruff_crystalyze', 'mp20_final_aug',   'inversion_results_cctbx_direct.pkl'),
            'Bispec':        p('rruff_crystalyze', 'mp20_final_aug',   'inversion_results_cctbx_bispec.pkl'),
        }, p('rruff_crystalyze', 'comparison_plots')),

        ('rruff_crystalyze_delta02', {
            'Crystalyze': p('rruff_crystalyze', 'crystalyze',    'inversion_results_cctbx_delta02.pkl'),
            'Direct':     p('rruff_crystalyze', 'mp20_final_aug', 'inversion_results_cctbx_direct_delta02.pkl'),
            'Bispec':     p('rruff_crystalyze', 'mp20_final_aug', 'inversion_results_cctbx_bispec_delta02.pkl'),
        }, p('rruff_crystalyze', 'comparison_plots_delta02')),

        ('mp20_correct_cag', {
            'Direct': p('mp20_correct_cag', 'inversion_results_cctbx_direct.pkl'),
            'Bispec': p('mp20_correct_cag', 'inversion_results_cctbx_bispec.pkl'),
        }, p('mp20_correct_cag', 'comparison_plots')),

        ('rruff_alpha', {
            'Direct': p('rruff_alpha', 'mpfull_aug', 'inversion_results_cctbx_direct.pkl'),
            'Bispec': p('rruff_alpha', 'mpfull_aug', 'inversion_results_cctbx_bispec.pkl'),
        }, p('rruff_alpha', 'comparison_plots')),

        ('rruff_alpha_medium', {
            'Direct': p('rruff_alpha', 'mpfull_aug_medium', 'inversion_results_cctbx_direct.pkl'),
            'Bispec': p('rruff_alpha', 'mpfull_aug_medium', 'inversion_results_cctbx_bispec.pkl'),
        }, p('rruff_alpha', 'mpfull_aug_medium', 'comparison_plots')),

        ('rruff_alpha_crystalyze', {
            'Direct': p('rruff_alpha', 'mpfull_crystalyze_cag', 'inversion_results_cctbx_direct.pkl'),
            'Bispec': p('rruff_alpha', 'mpfull_crystalyze_cag', 'inversion_results_cctbx_bispec.pkl'),
        }, p('rruff_alpha', 'mpfull_crystalyze_cag', 'comparison_plots')),

        ('rruff_alpha_large', {
            'MPFull Aug':       p('rruff_alpha', 'mpfull_aug',       'inversion_results_cctbx_bispec.pkl'),
            'MPFull Aug Large': p('rruff_alpha', 'mpfull_aug_large', 'inversion_results_cctbx_bispec.pkl'),
        }, p('rruff_alpha', 'comparison_plots_large')),

        ('rruff_raw_xy', {
            'mp20 Direct':    p('rruff', 'raw_xy', 'mp20_direct',    'inversion_results_cctbx.pkl'),
            'mp20aug Direct': p('rruff', 'raw_xy', 'mp20aug_direct', 'inversion_results_cctbx.pkl'),
            'mpfull Direct':  p('rruff', 'raw_xy', 'mpfull_direct',  'inversion_results_cctbx.pkl'),
        }, p('rruff', 'raw_xy', 'comparison_plots')),
    ]


# ── main ─────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--cctbx-dir',
                        default=os.path.join(os.environ.get('POWDERXRD_DATA_ROOT', './data'),
                                             'final_test_results_cctbx'),
                        help='Root of final_test_results_cctbx/ tree.')
    parser.add_argument('--no-crystalyze', action='store_true',
                        help='Exclude Crystalyze from mp20 comparison.')
    parser.add_argument('--datasets', nargs='+', metavar='LABEL',
                        help='Only run these dataset labels (e.g. rruff_processed_xy_mp20aug).')
    parser.add_argument('--suffix', default='',
                        help='Suffix to append to pkl filenames, e.g. "_filtered".')
    parser.add_argument('--rruff-pkl',
                        default=os.path.join(os.environ.get('POWDERXRD_DATA_ROOT', './data'),
                                             'rruff_data_dict.pkl'),
                        help='Path to rruff_data_dict.pkl for disorder filtering.')
    args = parser.parse_args()

    all_datasets = get_datasets(args.cctbx_dir, no_crystalyze=args.no_crystalyze, suffix=args.suffix)
    if args.datasets:
        all_datasets = [(l, m, d) for l, m, d in all_datasets if l in args.datasets]

    for label, models, plot_dir in all_datasets:
        print(f'=== {label} ===')
        run_dataset(label, models, plot_dir, rruff_pkl=args.rruff_pkl)

    print('All done.')


if __name__ == '__main__':
    main()
