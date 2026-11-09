"""
Generate ALL data files required by Crystalyze's evaluate.py from a bare CSV
that only has: material_id, cif, formation_energy_per_atom, band_gap,
pretty_formula, e_above_hull, elements, spacegroup.number.

This script produces:
  1. An augmented CSV with the missing columns:
     - xrd_peak_locations      (list of 2theta peak positions)
     - xrd_peak_intensities    (list of peak intensities)
     - atomic_numbers          (list of atomic numbers of all atoms)
     - disc_sim_xrd            (256-bin discretized simulated XRD pattern)

  2. test.pt  — dict mapping material_id -> graph_arrays tuple
    (frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms)

  3. test_pv_xrd.pt — dict mapping material_id -> FloatTensor(1, 8500)
     Pseudo-Voigt XRD profile evaluated at np.arange(5, 90, 0.01).

Usage:
    python prepare_test_data.py --csv_path /path/to/test.csv [--output_dir /path/to/output] [--n_workers 8]

Requirements (install in your Crystalyze conda environment):
    pymatgen, torch, numpy, pandas, tqdm
"""

import argparse
import os
import multiprocessing as mp
from functools import partial

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from pymatgen.core.structure import Structure
from pymatgen.core.lattice import Lattice
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis import local_env
from pymatgen.analysis.diffraction.xrd import XRDCalculator


# ---------------------------------------------------------------------------
#  Per-worker global objects (initialised once per pool worker)
# ---------------------------------------------------------------------------

_worker_xrd_calculator = None
_worker_CrystalNN = None


def _init_worker():
    """Initialise heavy objects once per worker process."""
    global _worker_xrd_calculator, _worker_CrystalNN
    _worker_xrd_calculator = XRDCalculator(wavelength='CuKa', symprec=0.1)
    _worker_CrystalNN = local_env.CrystalNN(
        distance_cutoffs=None, x_diff_weight=-1, porous_adjustment=False)


# ---------------------------------------------------------------------------
#  Lattice helpers (from cdvae/common/data_utils.py)
# ---------------------------------------------------------------------------

def abs_cap(val, max_abs_val=1):
    return max(min(val, max_abs_val), -max_abs_val)


def lattice_params_to_matrix(a, b, c, alpha, beta, gamma):
    angles_r = np.radians([alpha, beta, gamma])
    cos_alpha, cos_beta, cos_gamma = np.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = np.sin(angles_r)
    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    val = abs_cap(val)
    gamma_star = np.arccos(val)
    vector_a = [a * sin_beta, 0.0, a * cos_beta]
    vector_b = [
        -b * sin_alpha * np.cos(gamma_star),
        b * sin_alpha * np.sin(gamma_star),
        b * cos_alpha,
    ]
    vector_c = [0.0, 0.0, float(c)]
    return np.array([vector_a, vector_b, vector_c])


# ---------------------------------------------------------------------------
#  Crystal building (from cdvae/common/data_utils.py)
# ---------------------------------------------------------------------------

def build_crystal(crystal_str, niggli=True, primitive=False):
    """Build crystal from cif string."""
    crystal = Structure.from_str(crystal_str, fmt='cif')
    if primitive:
        crystal = crystal.get_primitive_structure()
    if niggli:
        crystal = crystal.get_reduced_structure()
    canonical_crystal = Structure(
        lattice=Lattice.from_parameters(*crystal.lattice.parameters),
        species=crystal.species,
        coords=crystal.frac_coords,
        coords_are_cartesian=False,
    )
    return canonical_crystal


def build_crystal_graph(crystal, graph_method='crystalnn'):
    """Build crystal graph and return graph_arrays tuple."""
    if graph_method == 'crystalnn':
        crystal_graph = StructureGraph.with_local_env_strategy(
            crystal, _worker_CrystalNN)
    elif graph_method == 'none':
        pass
    else:
        raise NotImplementedError

    frac_coords = crystal.frac_coords
    atom_types = crystal.atomic_numbers
    lattice_parameters = crystal.lattice.parameters
    lengths = lattice_parameters[:3]
    angles = lattice_parameters[3:]

    assert np.allclose(crystal.lattice.matrix,
                       lattice_params_to_matrix(*lengths, *angles))

    edge_indices, to_jimages = [], []
    if graph_method != 'none':
        for i, j, to_jimage in crystal_graph.graph.edges(data='to_jimage'):
            edge_indices.append([j, i])
            to_jimages.append(to_jimage)
            edge_indices.append([i, j])
            to_jimages.append(tuple(-tj for tj in to_jimage))

    atom_types = np.array(atom_types)
    lengths, angles = np.array(lengths), np.array(angles)
    edge_indices = np.array(edge_indices)
    to_jimages = np.array(to_jimages)
    num_atoms = atom_types.shape[0]

    return frac_coords, atom_types, lengths, angles, edge_indices, to_jimages, num_atoms


# ---------------------------------------------------------------------------
#  XRD simulation helpers (from cdvae/xrd_utils/pv_utils.py)
# ---------------------------------------------------------------------------

def caglioti_fwhm(theta, U, V, W):
    rad_theta = np.radians(theta / 2)
    return (U * np.tan(rad_theta)**2 + V * np.tan(rad_theta) + W)**0.5


def pseudo_voigt(x, center, amplitude, U, V, W, eta, noise_sd=0.0):
    fwhm = caglioti_fwhm(center, U, V, W)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    noisy_percentage = (100 + noise_sd) / 100
    amplitude = amplitude * noisy_percentage
    lorentzian = amplitude * (fwhm**2 / ((x - center)**2 + fwhm**2))
    gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    return eta * lorentzian + (1 - eta) * gaussian


def superimposed_pseudo_voigt(x, xy_merge, U, V, W, eta, noise_sd=0.0):
    total = np.zeros_like(x)
    for row in xy_merge:
        center, amplitude = row
        total += pseudo_voigt(x, center, amplitude, U, V, W, eta, noise_sd)
    mx = max(total)
    if mx > 0:
        total = total / mx
    return total


def simulate_pv_xrd_for_row(xy_merge, U, V, W, noise=0.0):
    x = np.arange(5, 90, 0.010)  # 8500 points
    eta = 0
    sim_xrd = superimposed_pseudo_voigt(x, xy_merge, U, V, W, eta, noise_sd=noise)
    return sim_xrd


def get_sim_xrd_from_pattern(pattern, U=0.05, V=-0.06, W=0.07):
    """Full pseudo-Voigt XRD from a pymatgen DiffractionPattern."""
    xy_merge = np.column_stack((pattern.x, pattern.y))
    sim_xrd = simulate_pv_xrd_for_row(xy_merge, U, V, W, noise=0.0)
    return sim_xrd


# ---------------------------------------------------------------------------
#  Discretized sim XRD (256-bin histogram of peak intensities)
# ---------------------------------------------------------------------------

def discretize_xrd_pattern(pattern, n_bins=256, two_theta_range=(0, 90)):
    bins = np.zeros(n_bins)
    bin_width = (two_theta_range[1] - two_theta_range[0]) / n_bins

    for two_theta, intensity in zip(pattern.x, pattern.y):
        if two_theta_range[0] <= two_theta < two_theta_range[1]:
            bin_idx = int((two_theta - two_theta_range[0]) / bin_width)
            bin_idx = min(bin_idx, n_bins - 1)
            bins[bin_idx] += intensity

    mx = bins.max()
    if mx > 0:
        bins = bins / mx
    return bins


# ---------------------------------------------------------------------------
#  Single-structure worker function (runs in pool workers)
# ---------------------------------------------------------------------------

def process_one_structure(args_tuple):
    """
    Process a single CIF string. Designed to be called via Pool.imap.

    Accepts a tuple: (idx, material_id, cif_str, niggli, primitive, graph_method)

    Returns a dict with all derived data (numpy arrays only — no torch tensors,
    because tensors with shared CUDA/CPU memory don't pickle cleanly across
    process boundaries). Tensor conversion happens in the main process.
    """
    idx, material_id, cif_str, niggli, primitive, graph_method, U, V, W = args_tuple

    result = {
        'idx': idx,
        'material_id': material_id,
        'graph_arrays': None,
        'xrd_peak_locations': None,
        'xrd_peak_intensities': None,
        'atomic_numbers': None,
        'disc_sim_xrd': None,
        'pv_xrd': None,       # numpy array (8500,), converted to tensor later
        'error': None,
    }

    try:
        crystal = build_crystal(cif_str, niggli=niggli, primitive=primitive)

        result['graph_arrays'] = build_crystal_graph(crystal, graph_method=graph_method)
        result['atomic_numbers'] = list(crystal.atomic_numbers)

        pattern = _worker_xrd_calculator.get_pattern(crystal)
        result['xrd_peak_locations'] = pattern.x.tolist()
        result['xrd_peak_intensities'] = pattern.y.tolist()
        result['disc_sim_xrd'] = discretize_xrd_pattern(pattern)
        result['pv_xrd'] = get_sim_xrd_from_pattern(pattern, U=U, V=V, W=W)

    except Exception as e:
        result['error'] = str(e)

    return result


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Generate augmented CSV + test.pt from a bare CSV with CIF strings.')
    parser.add_argument('--csv_path', type=str, required=True,
                        help='Path to input CSV (must have material_id and cif columns)')
    parser.add_argument('--output_dir', type=str, default=None,
                        help='Directory for outputs. Defaults to same directory as input CSV.')
    parser.add_argument('--niggli', type=bool, default=True)
    parser.add_argument('--primitive', type=bool, default=False)
    parser.add_argument('--graph_method', type=str, default='crystalnn')
    parser.add_argument('--n_workers', type=int, default=None,
                        help='Number of parallel workers (default: number of CPU cores)')
    parser.add_argument('--U', type=float, default=0.05, help='Caglioti U parameter')
    parser.add_argument('--V', type=float, default=-0.06, help='Caglioti V parameter')
    parser.add_argument('--W', type=float, default=0.07, help='Caglioti W parameter')
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.csv_path))
    os.makedirs(args.output_dir, exist_ok=True)

    n_workers = args.n_workers or mp.cpu_count()

    # --- Read input CSV ---
    df = pd.read_csv(args.csv_path)
    assert 'material_id' in df.columns, "CSV must have 'material_id' column"
    assert 'cif' in df.columns, "CSV must have 'cif' column"

    print(f"Processing {len(df)} structures from {args.csv_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Using {n_workers} parallel workers")

    # --- Build task list ---
    tasks = [
        (idx, row['material_id'], row['cif'],
         args.niggli, args.primitive, args.graph_method, args.U, args.V, args.W)
        for idx, row in df.iterrows()
    ]

    # --- Parallel processing ---
    # Pre-allocate lists aligned to df index order
    n = len(df)
    xrd_peak_locs = [None] * n
    xrd_peak_ints = [None] * n
    atomic_nums = [None] * n
    disc_sim_xrds = [None] * n
    graph_dict = {}
    pv_xrd_dict = {}
    failed = []

    with mp.Pool(processes=n_workers, initializer=_init_worker) as pool:
        for result in tqdm(
            pool.imap_unordered(process_one_structure, tasks),
            total=n,
            desc="Processing structures",
        ):
            i = result['idx']
            mid = result['material_id']

            if result['error'] is not None:
                failed.append((mid, result['error']))
                xrd_peak_locs[i] = [0] * 256
                xrd_peak_ints[i] = [0] * 256
                atomic_nums[i] = [0]
                disc_sim_xrds[i] = np.zeros(256)
                pv_xrd_dict[mid] = torch.zeros(1, 8500, dtype=torch.float32)
            else:
                graph_dict[mid] = result['graph_arrays']
                pv_xrd_dict[mid] = torch.tensor(
                    result['pv_xrd'], dtype=torch.float32).unsqueeze(0)
                xrd_peak_locs[i] = result['xrd_peak_locations']
                xrd_peak_ints[i] = result['xrd_peak_intensities']
                atomic_nums[i] = result['atomic_numbers']
                disc_sim_xrds[i] = result['disc_sim_xrd']

    if failed:
        print(f"\n{len(failed)} structures failed:")
        for mid, err in failed[:10]:
            print(f"  {mid}: {err}")
        if len(failed) > 10:
            print(f"  ... and {len(failed) - 10} more")
    else:
        print("\nAll structures processed successfully.")

    # --- Save test.pt ---
    pt_path = os.path.join(args.output_dir, 'test.pt')
    torch.save(graph_dict, pt_path)
    print(f"Saved {len(graph_dict)} graphs -> {pt_path}")

    # --- Save test_pv_xrd.pt ---
    pv_xrd_path = os.path.join(args.output_dir, 'test_pv_xrd.pt')
    torch.save(pv_xrd_dict, pv_xrd_path)
    print(f"Saved {len(pv_xrd_dict)} PV XRD profiles -> {pv_xrd_path}")

    # --- Augment and save CSV ---
    # Format columns to match what Crystalyze's preprocess() expects:
    #   xrd_peak_locations / xrd_peak_intensities / atomic_numbers
    #       -> Python list literals, parsed with ast.literal_eval
    #   disc_sim_xrd
    #       -> "[0. 0. 0.5 ...]" format (square brackets, space-separated,
    #          no commas), parsed by: lambda x: [float(v) for v in x[1:-1].split() if v]
    df['xrd_peak_locations'] = [str(locs) for locs in xrd_peak_locs]
    df['xrd_peak_intensities'] = [str(ints) for ints in xrd_peak_ints]
    df['atomic_numbers'] = [str(nums) for nums in atomic_nums]
    df['disc_sim_xrd'] = [
        np.array2string(arr, max_line_width=np.inf)
        for arr in disc_sim_xrds
    ]

    csv_out_path = os.path.join(args.output_dir, 'test.csv')
    df.to_csv(csv_out_path, index=False)
    print(f"Saved augmented CSV -> {csv_out_path}")

    print(f"""
Output files:
  {csv_out_path}      (CSV with added columns)
  {pt_path}           (graph dict: material_id -> graph_arrays)
  {pv_xrd_path}       (PV XRD dict: material_id -> FloatTensor(1, 8500))
""")


if __name__ == '__main__':
    main()
