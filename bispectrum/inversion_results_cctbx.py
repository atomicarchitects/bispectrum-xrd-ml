#from mp_api.client import MPRester
from pymatgen.core import Structure
from pymatgen.core import Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import numpy as np
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import pylab as pl
from pymatgen.symmetry.groups import SpaceGroup
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
import os
import sys
import argparse
sys.path.append("..")
import pickle
from utilities import check_prim_lattices, get_neighbors, struct_from_lat, get_bispectrum, convert_lats_to_selling, get_cctbx_candidates, get_cctbx_true_aligned, get_bravais_from_spg_number, match_cell_setting, niggli_reduce_with_bravais
import collections
from matplotlib.colors import Normalize
from collections import Counter
from matplotlib.colors import LogNorm
from tqdm import tqdm
#import seaborn as sns
#from sklearn.metrics import mean_squared_error
#from scipy.stats import pearsonr

bravais_order = ["aP", "mP", "mC", "oP", "oC", "oI", "oF", "tP", "tI", "hR", "hP", "cP", "cI", "cF"]

def load_lattice_params_pt(pt_path):
    """Load lattice_params.pt and return a dict of {mp_id: Lattice}."""
    import torch
    data = torch.load(pt_path, weights_only=False,map_location='cpu')
    return {
        mp_id: Lattice.from_parameters(
            *entry['lengths'].tolist(), *entry['angles'].tolist()
        )
        for mp_id, entry in data.items()
    }


def load_crystalyze_csv(csv_path):
    """Load eval_recon_with_mpids.csv and return a dict of {mp_id: Lattice}."""
    df = pd.read_csv(csv_path)
    return {
        row['material_id']: Lattice.from_parameters(
            row['pred_a'], row['pred_b'], row['pred_c'],
            row['pred_alpha'], row['pred_beta'], row['pred_gamma']
        )
        for _, row in df.iterrows()
    }


def lattice_parameter_mape(true_lat, pred_lat):
    """Returns MAPE for each lattice parameter [a,b,c,alpha,beta,gamma]."""
    true_params = np.array([true_lat.a, true_lat.b, true_lat.c,
                            true_lat.alpha, true_lat.beta, true_lat.gamma])
    pred_params = np.array([pred_lat.a, pred_lat.b, pred_lat.c,
                            pred_lat.alpha, pred_lat.beta, pred_lat.gamma])
    return np.abs((pred_params - true_params) / true_params) * 100


def make_mape_plots_by_bravais(df, plot_dir):
    """Bar charts and boxplots of MAPE per Bravais lattice, ordered by symmetry."""
    errs_per_bravais = {}
    summary = {}
    for bravais, group in df.groupby('true_bravais'):
        all_mape = np.array([lattice_parameter_mape(t, p)
                             for t, p in zip(group['cctbx_true'], group['cctbx_pred'])])
        errs_per_bravais[bravais] = all_mape
        summary[bravais] = {
            'Length MAPE (%)': all_mape[:, :3].mean(),
            'Angle MAPE (%)':  all_mape[:, 3:].mean(),
        }

    ordered = [b for b in bravais_order if b in summary]
    x = np.arange(len(ordered))
    width = 0.5
    length_mapes = [summary[b]['Length MAPE (%)'] for b in ordered]
    angle_mapes  = [summary[b]['Angle MAPE (%)']  for b in ordered]

    # Bar chart
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].bar(x, length_mapes, width, color='steelblue')
    axes[0].axhline(np.mean(length_mapes), color='steelblue', linestyle='--', label='Mean')
    axes[0].set_xticks(x); axes[0].set_xticklabels(ordered, rotation=45)
    axes[0].set_ylabel('MAPE (%)'); axes[0].set_title('Length MAPE per Bravais Lattice (Inversion)')
    axes[0].legend()
    axes[1].bar(x, angle_mapes, width, color='darkorange')
    axes[1].axhline(np.mean(angle_mapes), color='darkorange', linestyle='--', label='Mean')
    axes[1].set_xticks(x); axes[1].set_xticklabels(ordered, rotation=45)
    axes[1].set_ylabel('MAPE (%)'); axes[1].set_title('Angle MAPE per Bravais Lattice (Inversion)')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/inversion_mape_by_bravais.pdf', bbox_inches='tight')
    plt.close()

    # Boxplots: lengths
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot([errs_per_bravais[b][:, :3].flatten() for b in ordered], tick_labels=ordered)
    ax.set_ylabel('Length Error (%)'); ax.set_title('Distribution of Length Errors (Inversion) per Bravais Lattice')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/inversion_length_errors_boxplot.pdf', bbox_inches='tight')
    plt.close()

    # Boxplots: angles
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.boxplot([errs_per_bravais[b][:, 3:].flatten() for b in ordered], tick_labels=ordered)
    ax.set_ylabel('Angle Error (%)'); ax.set_title('Distribution of Angle Errors (Inversion) per Bravais Lattice')
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/inversion_angle_errors_boxplot.pdf', bbox_inches='tight')
    plt.close()


bravais_lattice_names = {
    'aP': 'Triclinic Primitive',
    'mP': 'Monoclinic Primitive',
    'mC': 'Monoclinic Base-Centered',
    'oP': 'Orthorhombic Primitive',
    'oC': 'Orthorhombic Base-Centered',
    'oI': 'Orthorhombic Body-Centered',
    'oF': 'Orthorhombic Face-Centered',
    'tP': 'Tetragonal Primitive',
    'tI': 'Tetragonal Body-Centered',
    'hP': 'Hexagonal Primitive',
    'hR': 'Hexagonal (Rhombohedral)',
    'cP': 'Cubic Primitive',
    'cI': 'Cubic Body-Centered',
    'cF': 'Cubic Face-Centered',
}

import concurrent.futures
from concurrent.futures.process import BrokenProcessPool


def restart_executor(executor):
    """Terminate the worker and start a fresh one. shutdown(wait=False) alone leaves a worker
    stuck inside a C extension running (and using a full core) for the rest of the job."""
    for proc in list(getattr(executor, '_processes', {}).values()):
        proc.terminate()
    executor.shutdown(wait=False, cancel_futures=True)
    return concurrent.futures.ProcessPoolExecutor(max_workers=1)


def niggli_fallback(executor, true_input, pred_input, symprec, is_recip, label, timeout=60):
    """Niggli reduction + Bravais assignment in the worker process, with a timeout, so a
    pathological cell cannot stall the run. Falls back to the unreduced lattices ('aP') on
    timeout or error. Returns (executor, true_lat, pred_lat, true_bravais, pred_bravais)."""
    try:
        result = executor.submit(niggli_reduce_with_bravais, true_input, pred_input, symprec, is_recip).result(timeout=timeout)
        return (executor, *result)
    except concurrent.futures.TimeoutError:
        print(f"Niggli fallback timed out for {label}; using unreduced lattices")
        executor = restart_executor(executor)
    except BrokenProcessPool as e:
        print(f"Niggli fallback worker died for {label}: {e}; using unreduced lattices")
        executor = restart_executor(executor)
    except Exception as e:
        print(f"Niggli fallback failed for {label}: {e}; using unreduced lattices")
    return executor, struct_from_lat(true_input).lattice, struct_from_lat(pred_input).lattice, 'aP', 'aP'


def make_plots_per_lattice(df, plot_dir, to_filter=None, filter_val=None, bravais_lattice_names=bravais_lattice_names, id_col='mat_id'):
    import numpy as np
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.stats import pearsonr
    from sklearn.metrics import mean_squared_error

    plt.rcParams.update({
    'font.size': 16,          # base font size
    'axes.titlesize': 16,     # title size for axes
    'axes.labelsize': 16,     # x/y label size
    'xtick.labelsize': 16,    # x tick size
    'ytick.labelsize': 16,    # y tick size
    'legend.fontsize': 16,    # legend font size
    'figure.titlesize': 18    # suptitle size
    })

    if to_filter == 'true_bravais':
        filtered = df[df['true_bravais'] == filter_val]
        filter_val = bravais_lattice_names[filter_val] + f' ({filter_val})'
    elif to_filter == 'crystal_sys':
        filtered = df[df['crystal_sys'] == filter_val]
    else:
        filtered = df
        filter_val = "All"
    
    lattice_dict = {}
    for index, row in filtered.iterrows():
        #pred_lattice, true_lattice = convert_lats_to_selling(row['pred'], row['true_lattice'])
        #crystal_sys = row['crystal_sys']
        bravais = row['true_bravais']#['cctbx_lat_type']  # Use cctbx_lat_type for bravais lattice type
        lattice_dict[row[id_col]] = {
            'pred_lattice': row['cctbx_pred'],
            'true_lattice': row['cctbx_true'],
            #'crystal_sys': crystal_sys,
            'bravais': bravais
        }
    #print(lattice_dict)
    pred_a, pred_b, pred_c, pred_alpha, pred_beta, pred_gamma, pred_vol = [], [], [], [], [], [], []
    true_a, true_b, true_c, true_alpha, true_beta, true_gamma, true_vol = [], [], [], [], [], [], []

    # Extract lattice parameters from the lattice_dict

    ### NOTE, depending on the lattice system, these should potentially be sorted or not?
    for mat_id, lattices in lattice_dict.items():
        pred_lattice = lattices['pred_lattice']
        true_lattice = lattices['true_lattice']

        pred_a.append(pred_lattice.a)
        pred_b.append(pred_lattice.b)
        pred_c.append(pred_lattice.c)
        pred_alpha.append(pred_lattice.alpha)
        pred_beta.append(pred_lattice.beta)
        pred_gamma.append(pred_lattice.gamma)
        pred_vol.append(pred_lattice.volume)

        true_a.append(true_lattice.a)
        true_b.append(true_lattice.b)
        true_c.append(true_lattice.c)
        true_alpha.append(true_lattice.alpha)
        true_beta.append(true_lattice.beta)
        true_gamma.append(true_lattice.gamma)
        true_vol.append(true_lattice.volume)

    # Create figure and axes
    fig, axes = plt.subplots(4, 2, figsize=(14, 13))
    axes = axes.flatten()

    param_data = [
        (pred_a, true_a, 'Lattice parameter a', 'a'),
        (pred_b, true_b, 'Lattice parameter b', 'b'),
        (pred_c, true_c, 'Lattice parameter c', 'c'),
        (pred_alpha, true_alpha, 'Lattice parameter alpha', 'alpha'),
        (pred_beta, true_beta, 'Lattice parameter beta', 'beta'),
        (pred_gamma, true_gamma, 'Lattice parameter gamma', 'gamma'),
        (pred_vol, true_vol, 'Lattice volume', 'volume')
    ]

    for i, (pred, true, title, xlabel) in enumerate(param_data):
        ax = axes[i]
        # KDE checks
        if np.std(pred) > 1e-8:
            sns.kdeplot(pred, label='Predicted', shade=True, ax=ax, color='blue', alpha=0.2)
        else:
            ax.axvline(pred[0], color='blue', linestyle='--', label='Predicted')

        if np.std(true) > 1e-8:
            sns.kdeplot(true, label='True', shade=True, ax=ax, color='red', alpha=0.2)
        else:
            ax.axvline(true[0], color='red', linestyle='--', label='True')

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel('Density')
        ax.legend()

    for j in range(len(param_data), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('True vs. Predicted Distributions ' + filter_val)
    plt.tight_layout()
    plt.savefig(f'{plot_dir}/lattice_distributions_{filter_val}.png')

    # Scatter plots
    fig, axes = plt.subplots(4, 2, figsize=(14, 13))
    axes = axes.flatten()

    # Collect all true-pred pairs from all parameters
    all_true_pred_pairs = []
    for pred, true, *_ in param_data:
        all_true_pred_pairs.extend(zip(true, pred))

    # Round to avoid float precision issues
    rounded_pairs = [(round(t, 4), round(p, 4)) for t, p in all_true_pred_pairs]
    global_counts = Counter(rounded_pairs)

    # Determine global frequency range
    all_freqs = np.array(list(global_counts.values()))
    global_min = all_freqs.min()
    global_max = all_freqs.max()

    norm = LogNorm(vmin=1, vmax=all_freqs.max())  # vmin=1 to avoid log(0)

    for i, (pred, true, title, label) in enumerate(param_data):
        ax = axes[i]
        # Count frequency of each (true, pred) pair
        # Round for frequency lookup
        rounded_this_plot = [(round(t, 4), round(p, 4)) for t, p in zip(true, pred)]
        freqs = np.array([global_counts[pair] for pair in rounded_this_plot])

        # Plot with shared norm
        sc = ax.scatter(true, pred, c=freqs, cmap='Blues', edgecolors='k', s=10, alpha=0.5, norm=norm)
        # Add colorbar
        #cbar = plt.colorbar(sc, ax=ax)
        #cbar.set_label('Frequency')
        #ax.scatter(true, pred, alpha=0.5, label='Data', color='blue', edgecolors='k', s=20)

        try:
            if np.std(true) < 1e-8:
                mean_val = true[0]
                ax.axvline(mean_val, color='red', linestyle='--', label='True Value')
                ax.text(0.05, 0.95, f"True values constant\n(mean = {mean_val:.3f})", 
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))
            else:
                # Diagonal line y = x
                min_val = min(min(true), min(pred))
                max_val = max(max(true), max(pred))
                ax.plot([min_val, max_val], [min_val, max_val], color='red', linestyle='--', lw=1, label='y=x')

                # Stats
                corr, _ = pearsonr(true, pred)
                rmse = np.sqrt(mean_squared_error(true, pred))#, squared=False)
                ax.text(0.05, 0.95, f"r = {corr:.3f}\nRMSE = {rmse:.3f}", 
                        transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', bbox=dict(boxstyle="round", facecolor="white", alpha=0.6))
        except Exception as e:
            ax.text(0.05, 0.95, f"Error in stats:\n{e}", 
                    transform=ax.transAxes, fontsize=14,
                    verticalalignment='top', bbox=dict(boxstyle="round", facecolor="pink", alpha=0.6))

        ax.set_title(f'{title}: Predicted vs True')
        ax.set_xlabel(f'True {label}')
        ax.set_ylabel(f'Predicted {label}')
        #ax.legend()

    for j in range(len(param_data), len(axes)):
        fig.delaxes(axes[j])

    plt.suptitle('Predicted vs True Scatter Plots ' + filter_val)
    # Add a shared colorbar for all subplots
    # Adjust layout to make room at the bottom
    
    fig.subplots_adjust(bottom=0.1)

    # Add horizontal colorbar at bottom center
    plt.tight_layout(rect=[0, 0.1, 1, 1])
    fig.subplots_adjust(bottom=0.15)
    cbar_ax = fig.add_axes([0.3, 0.05, 0.4, 0.02])
    cbar = fig.colorbar(sc, cax=cbar_ax, orientation='horizontal')
    cbar.set_label('Point Frequency (log scale)', fontsize=14)
    cbar.ax.tick_params(labelsize=12)
    plt.savefig(f'{plot_dir}/lattice_scatter_{filter_val}.png')

def canonicalize_lattice_params(lattice, bravais):
    if bravais is None:
        return lattice

    a, b, c = lattice.a, lattice.b, lattice.c
    alpha, beta, gamma = lattice.alpha, lattice.beta, lattice.gamma
    lengths = [a, b, c]
    angles = [alpha, beta, gamma]  # angles[i] is opposite lengths[i]

    def permuted(perm):
        """Return Lattice with axes reordered by perm (e.g. [2,0,1])."""
        return Lattice.from_parameters(
            lengths[perm[0]], lengths[perm[1]], lengths[perm[2]],
            angles[perm[0]], angles[perm[1]], angles[perm[2]]
        )

    if bravais in ['cP', 'cI', 'cF']:
        # Sort a<=b<=c, permuting angles with lengths
        perm = sorted(range(3), key=lambda i: lengths[i])
        return permuted(perm)

    elif bravais in ['tP', 'tI']:
        # Identify the equal pair → those become a,b; the odd one becomes c
        cross_diffs = [abs(b - c), abs(a - c), abs(a - b)]
        equal_pair_idx = int(np.argmin(cross_diffs))
        # equal_pair_idx=0: b≈c equal, a unique → perm (1,2,0)
        # equal_pair_idx=1: a≈c equal, b unique → perm (0,2,1)
        # equal_pair_idx=2: a≈b equal, c unique → perm (0,1,2)
        perm_map = {0: [1, 2, 0], 1: [0, 2, 1], 2: [0, 1, 2]}
        return permuted(perm_map[equal_pair_idx])

    elif bravais in ['hP', 'hR']:
        # Same logic as tetragonal — equal pair becomes a,b
        cross_diffs = [abs(b - c), abs(a - c), abs(a - b)]
        equal_pair_idx = int(np.argmin(cross_diffs))
        perm_map = {0: [1, 2, 0], 1: [0, 2, 1], 2: [0, 1, 2]}
        return permuted(perm_map[equal_pair_idx])

    elif bravais in ['oP', 'oI', 'oF', 'oC']:
        # a<=b<=c, all angles ~90 so permutation is safe either way
        perm = sorted(range(3), key=lambda i: lengths[i])
        return permuted(perm)

    elif bravais in ['mP', 'mC']:
        # Find the unique axis: the one whose opposite angle deviates most from 90°
        angle_devs = [abs(alpha - 90), abs(beta - 90), abs(gamma - 90)]
        unique_idx = int(np.argmax(angle_devs))
        # Put unique axis in b position; sort the other two so a<=c
        other = [i for i in range(3) if i != unique_idx]
        if lengths[other[0]] > lengths[other[1]]:
            other = [other[1], other[0]]
        perm = [other[0], unique_idx, other[1]]
        return permuted(perm)

    else:  # aP
        perm = sorted(range(3), key=lambda i: lengths[i])
        return permuted(perm)

def get_canonical_perm(lattice, bravais):
    """Return the permutation indices for canonical axis ordering, determined from lattice."""
    lengths = [lattice.a, lattice.b, lattice.c]
    angles = [lattice.alpha, lattice.beta, lattice.gamma]

    if bravais in ['tP', 'tI', 'hP', 'hR']:
        cross_diffs = [abs(lengths[1]-lengths[2]),
                       abs(lengths[0]-lengths[2]),
                       abs(lengths[0]-lengths[1])]
        eq = int(np.argmin(cross_diffs))
        return {0: [1,2,0], 1: [0,2,1], 2: [0,1,2]}[eq]

    elif bravais in ['mP', 'mC']:
        angle_devs = [abs(a - 90) for a in angles]
        unique_idx = int(np.argmax(angle_devs))
        other = [i for i in range(3) if i != unique_idx]
        if lengths[other[0]] > lengths[other[1]]:
            other = [other[1], other[0]]
        return [other[0], unique_idx, other[1]]

    else:  # cP/cI/cF, oP/oI/oF/oC, aP
        return sorted(range(3), key=lambda i: lengths[i])


def apply_perm(lattice, perm):
    """Apply axis permutation to a lattice."""
    lengths = [lattice.a, lattice.b, lattice.c]
    angles = [lattice.alpha, lattice.beta, lattice.gamma]
    return Lattice.from_parameters(
        lengths[perm[0]], lengths[perm[1]], lengths[perm[2]],
        angles[perm[0]], angles[perm[1]], angles[perm[2]]
    )
    
import signal

class TimeoutException(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutException()

# Set timeout signal handler
signal.signal(signal.SIGALRM, timeout_handler)

def main():
    parser = argparse.ArgumentParser(description="Run cctbx lattice reduction and generate plots.")
    parser.add_argument("--input", required=True, help="Path to input inversion results pickle file.")
    parser.add_argument("--output", required=True, help="Path to save/load cctbx results pickle file.")
    parser.add_argument("--plot-dir", required=True, help="Directory to save plots.")
    parser.add_argument("--max-delta", type=float, default=0.1, help="Max delta for cctbx candidates on true lattice (default: 0.1).")
    parser.add_argument("--max-delta-pred", type=float, default=None, help="Max delta for cctbx candidates on predicted lattice (default: same as --max-delta).")
    parser.add_argument("--symprec", type=float, default=1e-2, help="Symmetry precision for Niggli reduction fallback (default: 1e-2).")
    parser.add_argument("--reciprocal", action="store_true", help="Use reciprocal space lattices.")
    parser.add_argument("--make-df", action="store_true", help="Recompute cctbx results instead of loading from --output.")
    parser.add_argument("--skip-mat-id", default=None, nargs='+', help="One or more material IDs to skip (e.g. --skip-mat-id mp-1222534 mp-123).")
    parser.add_argument("--hdf-key", default=None, help="HDF5 key to read from input .h5 file (default: first key in file).")
    parser.add_argument("--crystalyze-params", default=None, help="Path to lattice_params.pt file with Crystalyze predicted lattice params keyed by mp-id.")
    parser.add_argument("--crystalyze-csv", default=None, help="Path to eval_recon_with_mpids.csv with Crystalyze predictions (pred_a/b/c/alpha/beta/gamma columns).")
    parser.add_argument("--no-reduce-pred", action="store_true", help="Skip cctbx/Niggli reduction on Crystalyze predictions; use raw predicted lattice directly.")
    parser.add_argument("--canonicalize", action="store_true", help="Canonicalize lattice parameters after reduction.")
    parser.add_argument("--niggli-only", action="store_true", help="Use only Niggli reduction (skip cctbx entirely).")
    parser.add_argument("--align-to-true", action="store_true", help="Run cctbx on the true lattice only and apply the same cb_op to the prediction.")
    parser.add_argument("--true-bravais", default=None, help="Path to pickle with material_id/pattern_id and true_bravais columns. If provided, skips all on-the-fly true_bravais computation.")
    parser.add_argument("--true-bravais-id-col", default=None, help="Column name to use as key in the true-bravais pickle (overrides default: 'pattern_id' in rruff mode, 'material_id' otherwise).")
    parser.add_argument("--input-id-col", default=None, help="Column name to use as ID in the input pickle (overrides default: 'pattern_id' in rruff mode, 'mat_id' otherwise).")
    parser.add_argument("--rruff", action="store_true", help="Use RRUFF data mode: ID column is 'pattern_id' instead of 'mat_id'.")
    parser.add_argument("--cell-match-length-tol", type=float, default=0.05,
                        help="Relative length tolerance for re-expressing a prediction in the setting of the "
                             "same lattice closest to the true cell (default: 0.05).")
    parser.add_argument("--cell-match-angle-tol", type=float, default=5.0,
                        help="Angle tolerance in degrees for the same setting match (default: 5).")
    parser.add_argument("--no-cell-matching", action="store_true",
                        help="Skip the setting match (reproduces results computed before it was added).")
    args = parser.parse_args()

    id_col = args.input_id_col or ('pattern_id' if args.rruff else 'mat_id')
    bravais_id_col = args.true_bravais_id_col or ('pattern_id' if args.rruff else 'material_id')

    max_delta = args.max_delta
    max_delta_pred = args.max_delta_pred
    symprec = args.symprec
    plot_dir = args.plot_dir
    os.makedirs(plot_dir, exist_ok=True)
    recip = args.reciprocal
    print("using reciprocal lattices:", recip)
    cctbx_file = args.output

    # Load pre-computed true Bravais and true lattice lookups if provided
    true_bravais_lookup = {}
    true_lattice_lookup = {}
    if args.true_bravais:
        bravais_df = pd.read_pickle(args.true_bravais)
        true_bravais_lookup = dict(zip(bravais_df[bravais_id_col], bravais_df['true_bravais']))
        if 'true_lattice' in bravais_df.columns:
            true_lattice_lookup = dict(zip(bravais_df[bravais_id_col], bravais_df['true_lattice']))
        print(f"Loaded {len(true_bravais_lookup)} true Bravais entries from {args.true_bravais}")
        if true_lattice_lookup:
            print(f"Loaded {len(true_lattice_lookup)} true lattice entries from {args.true_bravais}")

    if args.input.endswith('.h5') or args.input.endswith('.hdf5'):
        import h5py as _h5py_check
        with _h5py_check.File(args.input, 'r') as _f:
            input_ids = set(_f.keys())
    else:
        _raw = pd.read_pickle(args.input)
        if isinstance(_raw, dict) and 'material_ids' in _raw:
            input_ids = set(_raw['material_ids'])
        elif isinstance(_raw, dict) and 'mineral_names' in _raw:
            input_ids = set(_raw['mineral_names'])
        else:
            input_ids = set(_raw[id_col])

    if true_bravais_lookup:
        overlap = input_ids & set(true_bravais_lookup.keys())
        missing = input_ids - set(true_bravais_lookup.keys())
        print(f"{id_col} overlap with true_bravais lookup: {len(overlap)}/{len(input_ids)}")
        if missing:
            print(f"Warning: {len(missing)} {id_col}s not in true_bravais lookup, will fall back to 'aP'")

    if args.input.endswith('.h5') or args.input.endswith('.hdf5'):
        import h5py
        mat_ids, preds, true_lats = [], [], []
        with h5py.File(args.input, 'r') as f:
            for mat_id, grp in tqdm(f.items(), desc="Loading HDF5", total=len(f)):
                mat_ids.append(mat_id)
                preds.append(grp['pred'][()])
                true_lats.append(grp['true_lattice'][()])
        inversion_df = pd.DataFrame({
            id_col: mat_ids,
            'pred': preds,
            'mp20_cif_lattice': true_lats,
        })
        n_dupes = inversion_df[id_col].duplicated().sum()
        if n_dupes > 0:
            print(f"Warning: {n_dupes} duplicate {id_col}s found, keeping last occurrence.")
            inversion_df = inversion_df.drop_duplicates(subset=id_col, keep='last').reset_index(drop=True)
        if true_lattice_lookup:
            before = len(inversion_df)
            inversion_df = inversion_df[inversion_df[id_col].isin(true_lattice_lookup)].reset_index(drop=True)
            dropped = before - len(inversion_df)
            if dropped:
                print(f"Dropped {dropped} h5 entries not in true_lattice lookup")
        if true_bravais_lookup:
            inversion_df['true_bravais'] = inversion_df[id_col].map(true_bravais_lookup).fillna('aP')
        else:
            print("Computing true_bravais from true lattices...")
            true_bravais_list = []
            for mat in tqdm(inversion_df['mp20_cif_lattice'], desc="Computing Bravais types"):
                try:
                    sga = SpacegroupAnalyzer(struct_from_lat(mat), symprec=symprec)
                    spg = sga.get_space_group_number()
                    true_bravais_list.append(get_bravais_from_spg_number(spg))
                except Exception:
                    true_bravais_list.append('aP')
            inversion_df['true_bravais'] = true_bravais_list
    else:
        raw = pd.read_pickle(args.input)
        _id_key = 'material_ids' if isinstance(raw, dict) and 'material_ids' in raw \
                  else 'mineral_names' if isinstance(raw, dict) and 'mineral_names' in raw \
                  else None
        if _id_key is not None:
            # Raw model output dict: {predictions, targets, material_ids}
            # true_lattice_lookup (from --true-bravais) takes priority over
            # mp20_cif_lattice, so targets are not converted here.
            preds = [
                Lattice.from_parameters(*p.tolist()).matrix
                for p in raw['predictions']
            ]
            trues = [
                Lattice.from_parameters(*t.tolist())
                for t in raw['targets']
            ]
            inversion_df = pd.DataFrame({
                id_col: raw[_id_key],
                'pred': preds,
                'cctbx_true': trues,
            })
        else:
            inversion_df = raw
        if true_lattice_lookup:
            before = len(inversion_df)
            inversion_df = inversion_df[inversion_df[id_col].isin(true_lattice_lookup)].reset_index(drop=True)
            dropped = before - len(inversion_df)
            if dropped:
                print(f"Dropped {dropped} entries with no true lattice in lookup")
        if true_bravais_lookup:
            inversion_df['true_bravais'] = inversion_df[id_col].map(true_bravais_lookup).fillna('aP')
        elif 'true_bravais' not in inversion_df.columns:
            print("Computing true_bravais from true lattices...")
            true_bravais_list = []
            for mat in tqdm(inversion_df['cctbx_true'], desc="Computing Bravais types"):
                try:
                    sga = SpacegroupAnalyzer(struct_from_lat(mat), symprec=symprec)
                    true_bravais_list.append(get_bravais_from_spg_number(sga.get_space_group_number()))
                except Exception:
                    true_bravais_list.append('aP')
            inversion_df['true_bravais'] = true_bravais_list

    # Load Crystalyze predictions if provided
    crystalyze_map = {}
    if args.crystalyze_csv:
        print(f"Loading Crystalyze predictions from {args.crystalyze_csv}...")
        crystalyze_map = load_crystalyze_csv(args.crystalyze_csv)
        print(f"Loaded {len(crystalyze_map)} Crystalyze entries")
    elif args.crystalyze_params:
        print(f"Loading Crystalyze predictions from {args.crystalyze_params}...")
        crystalyze_map = load_lattice_params_pt(args.crystalyze_params)
        print(f"Loaded {len(crystalyze_map)} Crystalyze entries")

    ### TODO: fix this timeout error and bravais lattice calculations
    ### Should I be using the cif lattice parameters... probably
    if args.make_df:
        cctbx_preds = []
        cctbx_trues = []
        cctbx_true_bravais = []
        cctbx_pred_bravais = []
        executor = concurrent.futures.ProcessPoolExecutor(max_workers=1)
        for index, row in tqdm(inversion_df.iterrows(), total=len(inversion_df), desc="Processing lattices"):
            print(f'{id_col}:', row[id_col])
            if crystalyze_map and row[id_col] in crystalyze_map and args.no_reduce_pred:
                # Skip cctbx/Niggli — use raw Crystalyze Lattice directly as cctbx_pred
                true_lat = true_lattice_lookup[row[id_col]] if true_lattice_lookup else row['cctbx_true']
                cctbx_trues.append(true_lat)
                cctbx_preds.append(crystalyze_map[row[id_col]])
                cctbx_true_bravais.append(row['true_bravais'])
                cctbx_pred_bravais.append(None)
                continue
            elif crystalyze_map and row[id_col] in crystalyze_map:
                # Crystalyze pred is real-space; use true lattice from lookup if available
                true_input = true_lattice_lookup[row[id_col]] if true_lattice_lookup else row['cctbx_true'].matrix
                pred_input = crystalyze_map[row[id_col]].matrix
                is_recip = False
            else:
                if true_lattice_lookup and row[id_col] in true_lattice_lookup:
                    true_input = true_lattice_lookup[row[id_col]]
                elif 'mp20_cif_lattice' in row:
                    true_input = row['mp20_cif_lattice']
                elif 'cctbx_true' in row:
                    true_input = row['cctbx_true'].matrix
                else:
                    true_input = row['true_lattice']
                pred_input = row['pred']
                is_recip = recip
            if args.niggli_only:
                executor, t_lat, p_lat, t_brav, p_brav = niggli_fallback(
                    executor, true_input, pred_input, symprec, is_recip, row[id_col])
                cctbx_trues.append(t_lat); cctbx_preds.append(p_lat)
                cctbx_true_bravais.append(t_brav); cctbx_pred_bravais.append(p_brav)
                continue
            if args.align_to_true:
                try:
                    future = executor.submit(get_cctbx_true_aligned, true_input, pred_input, max_delta, is_recip)
                    true_cell, _, true_bravais, pred_cell, _, pred_bravais = future.result(timeout=60)
                    cctbx_trues.append(Lattice.from_parameters(*true_cell))
                    cctbx_preds.append(Lattice.from_parameters(*pred_cell))
                    cctbx_true_bravais.append(true_bravais)
                    cctbx_pred_bravais.append(pred_bravais)
                except Exception as e:
                    print(f"align-to-true failed for {row[id_col]} ({type(e).__name__}: {e}), falling back to Niggli")
                    if isinstance(e, (concurrent.futures.TimeoutError, BrokenProcessPool)):
                        executor = restart_executor(executor)
                    executor, t_lat, p_lat, t_brav, p_brav = niggli_fallback(
                        executor, true_input, pred_input, symprec, is_recip, row[id_col])
                    cctbx_trues.append(t_lat); cctbx_preds.append(p_lat)
                    cctbx_true_bravais.append(t_brav); cctbx_pred_bravais.append(p_brav)
                continue
            if args.skip_mat_id and (row[id_col] in args.skip_mat_id):
                print(f"Skipping {row[id_col]} due to known issues with cctbx")
                cctbx_trues.append(struct_from_lat(true_input).lattice)
                cctbx_preds.append(struct_from_lat(pred_input).lattice)
                cctbx_true_bravais.append(row.get('true_bravais', None))
                cctbx_pred_bravais.append(None)
                continue
                #print(f"Pred SG: {pred_sg}, True SG: {true_sg}")
            try:
                future = executor.submit(get_cctbx_candidates, true_input, pred_input, max_delta, max_delta_pred, is_recip)
                true_cell, true_vol, true_bravais, pred_cell, pred_vol, pred_bravais = future.result(timeout=60)
                lat_true_cctbx = Lattice.from_parameters(*true_cell)
                lat_pred_cctbx = Lattice.from_parameters(*pred_cell)
                cctbx_trues.append(lat_true_cctbx)
                cctbx_true_bravais.append(true_bravais)
                cctbx_pred_bravais.append(pred_bravais)
                cctbx_preds.append(lat_pred_cctbx)
            except Exception as e:
                print(f"cctbx failed for {row[id_col]} ({type(e).__name__}: {e}), falling back to Niggli")
                if isinstance(e, (concurrent.futures.TimeoutError, BrokenProcessPool)):
                    executor = restart_executor(executor)
                executor, t_lat, p_lat, t_brav, p_brav = niggli_fallback(
                    executor, true_input, pred_input, symprec, is_recip, row[id_col])
                cctbx_trues.append(t_lat); cctbx_preds.append(p_lat)
                cctbx_true_bravais.append(t_brav); cctbx_pred_bravais.append(p_brav)

        restart_executor(executor).shutdown(wait=False)

        inversion_df['cctbx_pred'] = cctbx_preds
        inversion_df['pred_bravais'] = cctbx_pred_bravais
        if not crystalyze_map:
            inversion_df['cctbx_true'] = cctbx_trues
            if not true_bravais_lookup:
                inversion_df['true_bravais'] = cctbx_true_bravais
        # Canonicalize both sides using authoritative true_bravais
        if args.canonicalize:
            canon_preds, canon_trues = [], []
            for true_lat, pred_lat, bravais in zip(cctbx_trues, cctbx_preds, inversion_df['true_bravais']):
                perm = get_canonical_perm(true_lat, bravais)
                canon_trues.append(apply_perm(true_lat, perm))
                canon_preds.append(apply_perm(pred_lat, perm))
            inversion_df['cctbx_true'] = canon_trues
            inversion_df['cctbx_pred'] = canon_preds
        # After canonicalization, so re-running rescore_cell_matching.py on saved results
        # gives identical output (ties between equal axes are broken the same way).
        cell_matched = [False] * len(inversion_df)
        if not args.no_cell_matching:
            matched_preds = []
            for i, (true_lat, pred_lat) in enumerate(zip(inversion_df['cctbx_true'], inversion_df['cctbx_pred'])):
                lat, cell_matched[i] = match_cell_setting(
                    true_lat, pred_lat, args.cell_match_length_tol, args.cell_match_angle_tol)
                matched_preds.append(lat)
            inversion_df['cctbx_pred'] = matched_preds
            print(f"Setting match re-expressed {sum(cell_matched)}/{len(cell_matched)} predictions")
        inversion_df['cell_matched'] = cell_matched
        inversion_df.to_pickle(cctbx_file)
    else:
        inversion_df = pd.read_pickle(args.output)

    bravais = np.unique(inversion_df['true_bravais'])
    make_plots_per_lattice(inversion_df, plot_dir, to_filter=None, filter_val=None, id_col=id_col)
    for b in bravais:
        print(f"Bravais lattice: {b}")
        make_plots_per_lattice(inversion_df, plot_dir, to_filter='true_bravais', filter_val=b, id_col=id_col)
    make_mape_plots_by_bravais(inversion_df, plot_dir)
        
if __name__ == "__main__":
    main()
