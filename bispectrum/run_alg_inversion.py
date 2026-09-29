import os
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
import traceback
import numpy as np
import pandas as pd
import scipy.linalg
import plotly
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import torch
import e3nn
from e3nn import o3, io
from tqdm import tqdm
import matplotlib.pyplot as plt
from itertools import compress
from itertools import product
from pymatgen.core import Structure
from pymatgen.core import Lattice
from matscipy.neighbours import neighbour_list
import ase
import argparse
import sys
sys.path.append("..")
from utilities import check_prim_lattices_niggli, get_neighbors, struct_from_lat, get_bispectrum
from scipy.optimize import minimize
# change this based on gpu usage
#os.environ["CUDA_VISIBLE_DEVICES"] = "2,3"
import concurrent.futures
os.environ["CUDA_VISIBLE_DEVICES"] = ""
torch.set_default_dtype(torch.float64)
import random
from pymatgen.io.cif import CifParser
from io import StringIO
import functools
from threading import Lock
import h5py
from concurrent.futures import ProcessPoolExecutor, as_completed
from tqdm import tqdm
import time
import multiprocessing

MAX_RETRIES = 3  # how many times to retry a failed task
RETRY_DELAY = 5  # seconds between retries


# Globals in each process
# -------------------- GLOBALS --------------------
COB_BASES = None
ALL_BISPECS = None
ALL_MP_IDS = None
ALL_RECLAT = None
### Script for inverting bispectrum to get to lattice parameters and trying different random seeds

###############################################################################                                   
# Command line arguments
######################## 
def getArgumentParser():
    """ Get arguments from command line"""
    parser = argparse.ArgumentParser(description="Script to run bfgs to obtain lattice parameters from bispectrum.")
    parser.add_argument('-i',
                        '--input',
                        dest='input',
                        help='input dataframe to invert',
                        default = f'{_ROOT}/mp20aug/transformerbispec/eval_results/nlayers12_nhead8_dm256_hd512_maskloss/test_results.pkl')
                        #'$POWDERXRD_DATA_ROOT/mpfull/transformerbispec/eval_results/nlayers12_nhead8_dm256_hd512_maskloss/test_results.pkl') \
                        #'eval_results/nlayers12_nhead8_dm256_hd512_lossL1_maskloss_head/test_results.pkl')
                        #'$POWDERXRD_DATA_ROOT/mp20/transformerbispec/eval_results/nlayers8_nhead8_dm256_hd512_lossL1_maskloss_head/test_results.pkl')
                        #'$POWDERXRD_DATA_ROOT/mp20/transformerbispec/eval_results/nlayers8_nhead8_dm256_hd512_lossL1_maskloss/test_results.pkl')
                        #'$POWDERXRD_DATA_ROOT/mpfull/transformerbispec/eval_results/nlayers16_nhead12/test_results.pkl')
    parser.add_argument('-mat_proj_df',
                        '--mat_proj_df',
                        dest = 'mat_proj_df',
                        help = 'materials dataframe for dynamic programming',
                        default = f'{_ROOT}/materials_project_lat_bispec.h5')
    parser.add_argument('-r',
                        '--radial',
                        dest = 'radial',
                        help='radial function to use',
                        default = 'bessel')
    parser.add_argument('-n',
                        '--nbasis',
                        dest='nbasis',
                        help = 'number of basis functions',
                        default = 10)
    parser.add_argument('-w',
                        '--wavelength',
                        dest = 'wavelength',
                        help = 'wavelength for XRD',
                        default = 3)
    parser.add_argument('-lmax',
                        '--lmax',
                        dest = 'lmax',
                        help = 'max l to use',
                        default = 6)
    parser.add_argument('-cob',
                        '--cob',
                        dest = 'cob',
                        help = 'file for change of basis matrices',
                        default = f'{_ROOT}/cob_lmax6.pkl')
    parser.add_argument('-begin_ind',
                        '--begin_ind',
                        dest = 'begin_ind',
                        help = 'index in dataframe of where to start the inversion',
                        default = 0)
    parser.add_argument('-num_entries',
                        '--num_entries',
                        dest = 'num_entries',
                        help = 'number of entries to invert',
                        default = 500)
    parser.add_argument('-max_workers',
                        '--max_workers',
                        dest = 'max_workers',
                        help = 'maximum CPU workers',
                        default = 32)
    parser.add_argument('-o',
                        '--output',
                        dest = 'output',
                        help = 'output dataframe to build',
                        default = f'{_ROOT}/mp20aug/transformerbispec/eval_results/nlayers12_nhead8_dm256_hd512_maskloss/test_results.pkl')
    parser.add_argument('-mat_file',
                        '--material_file',
                        dest = 'material_file',
                        help = 'Optional list of material ids to invert',
                        default=None)
    parser.add_argument('-timeout',
                        '--timeout',
                        dest = 'timeout',
                        help = 'wall-clock timeout in seconds for the WHOLE material '
                               '(all num_to_opt + random_restarts candidates combined) -- '
                               'does NOT control the per-candidate L-BFGS-B budget, see -candidate_timeout',
                        default = 200)
    parser.add_argument('-candidate_timeout',
                        '--candidate_timeout',
                        dest = 'candidate_timeout',
                        type = int,
                        help = 'per-candidate L-BFGS-B timeout in seconds (each of the '
                               'num_to_opt + random_restarts starts gets up to this long). '
                               'Make sure -timeout is >= (num_to_opt + random_restarts) * '
                               'candidate_timeout in the worst case, or the whole material '
                               'can get killed by -timeout before every candidate is tried.',
                        default = 10)
                        #'$POWDERXRD_DATA_ROOT/mp20/transformerbispec/eval_results/nlayers12_nhead8_dm256_hd512_lossL1_maskloss/test_results.pkl')
                        #'$POWDERXRD_DATA_ROOT/mp20/transformerbispec/eval_results/nlayers8_nhead8_dm256_hd512_lossL1_maskloss/test_results.pkl')#'$POWDERXRD_DATA_ROOT/mp20/transformerbispec/eval_results/nlayers8_nhead8_dm256_hd512_lossL1_smallvals1e-12/test_results.pkl')#'$POWDERXRD_DATA_ROOT/mpfull/transformerbispec/eval_results/nlayers16_nhead12/inversion_results_test.pkl')
    parser.add_argument('-random_restarts',
                        '--random_restarts',
                        dest='random_restarts',
                        type=int,
                        help='number of random triclinic starting lattices to add to the candidate pool (0 = disabled)',
                        default=0)
    parser.add_argument('-random_restart_min_abc',
                        '--random_restart_min_abc',
                        dest='random_restart_min_abc',
                        type=float,
                        help='lower bound (Angstrom) for random-restart real-space a/b/c sampling',
                        default=3.0)
    parser.add_argument('-random_restart_max_abc',
                        '--random_restart_max_abc',
                        dest='random_restart_max_abc',
                        type=float,
                        help='upper bound (Angstrom) for random-restart real-space a/b/c sampling '
                             '-- widen this to cover a dataset with larger true cells than the '
                             'default 3-15 A range (e.g. opXRD CNRS goes up to ~50.6 A)',
                        default=15.0)
    parser.add_argument('-num_to_opt',
                        '--num_to_opt',
                        dest='num_to_opt',
                        type=int,
                        help='number of MP nearest-neighbour starts to use per material',
                        default=15)
    parser.add_argument('--init-from-true',
                        dest='init_from_true',
                        action='store_true',
                        default=False,
                        help='optimise from the true lattice only (diagnostic mode)')
    parser.add_argument('--true-start-timeout',
                        dest='true_start_timeout',
                        type=int,
                        default=150,
                        help='per-candidate timeout (seconds) when --init-from-true is set '
                             '(default 150, independent of --num_to_opt)')
    parser.add_argument('--basin-hopping',
                        dest='basin_hopping',
                        action='store_true',
                        default=False,
                        help='use basin hopping instead of a single L-BFGS-B run per candidate; '
                             'adds random perturbations between local minimizations to escape local minima')
    parser.add_argument('--bh-niter',
                        dest='bh_niter',
                        type=int,
                        default=20,
                        help='number of basin hopping steps per candidate (default 20)')
    parser.add_argument('--bh-stepsize',
                        dest='bh_stepsize',
                        type=float,
                        default=0.1,
                        help='step size for basin hopping random perturbations (default 0.1)')

    return parser

def sample_random_recip_lattices(n, min_abc=3.0, max_abc=15.0):
    """Sample n random reciprocal lattice matrices from a uniform triclinic prior
    in REAL space (a,b,c ~ Uniform(min_abc, max_abc) Angstrom, angles ~
    Uniform(60,120) degrees), converted to reciprocal afterward. The default
    bounds (3-15 A) undercover real datasets with larger cells -- e.g. the
    opXRD CNRS selection slice has true a/b/c up to ~50.6 A (long c-axis
    layered structures), ~12% of its a/b/c values outside [3,15] -- for which
    this sampler can never produce a nearby candidate at the default bounds.
    Widen max_abc (via -random_restart_max_abc) to cover a specific dataset's
    actual range when using --random-restarts as an MP-database-independent
    fallback."""
    from pymatgen.core import Lattice as PmgLattice
    candidates = []
    while len(candidates) < n:
        a = np.random.uniform(min_abc, max_abc)
        b = np.random.uniform(min_abc, max_abc)
        c = np.random.uniform(min_abc, max_abc)
        al = np.random.uniform(60, 120)
        be = np.random.uniform(60, 120)
        ga = np.random.uniform(60, 120)
        try:
            lat = PmgLattice.from_parameters(a, b, c, al, be, ga)
            candidates.append(lat.reciprocal_lattice_crystallographic.matrix)
        except Exception:
            continue
    return candidates


# --- Global HDF5 lock to prevent race conditions ---
hdf_lock = Lock()

# Initialize each worker process
# ----------------- Top-level worker -----------------
def worker(args):
    """
    Wrapper for process_row to be used with ProcessPoolExecutor.
    Expects a tuple: (mp_id, test_bispectrum, radial_function, n_basis, wavelength, lmax[, true_lat_params,
    random_restarts, num_to_opt, init_from_true, true_start_timeout, basin_hopping, bh_niter, bh_stepsize,
    random_restart_min_abc, random_restart_max_abc, candidate_timeout])
    true_lat_params: optional array [a, b, c, alpha, beta, gamma] used when mp_id is not in the MP database (e.g. RRUFF)
    """
    true_lat_params = args[6] if len(args) > 6 else None
    random_restarts = args[7] if len(args) > 7 else 0
    num_to_opt = args[8] if len(args) > 8 else 15
    init_from_true = args[9] if len(args) > 9 else False
    true_start_timeout = args[10] if len(args) > 10 else 150
    basin_hopping = args[11] if len(args) > 11 else False
    bh_niter = args[12] if len(args) > 12 else 20
    bh_stepsize = args[13] if len(args) > 13 else 0.1
    random_restart_min_abc = args[14] if len(args) > 14 else 3.0
    random_restart_max_abc = args[15] if len(args) > 15 else 15.0
    candidate_timeout = args[16] if len(args) > 16 else 10
    mp_id, test_bispectrum, radial_function, n_basis, wavelength, lmax = args[:6]
    try:
        result = process_row(
            mp_id,
            test_bispectrum,
            radial_function,
            n_basis,
            wavelength,
            lmax,
            true_lat_params=true_lat_params,
            random_restarts=random_restarts,
            num_to_opt=num_to_opt,
            timeout_seconds=candidate_timeout,
            init_from_true=init_from_true,
            true_start_timeout=true_start_timeout,
            basin_hopping=basin_hopping,
            bh_niter=bh_niter,
            bh_stepsize=bh_stepsize,
            random_restart_min_abc=random_restart_min_abc,
            random_restart_max_abc=random_restart_max_abc,
        )
        print(result)
        if result is None:
            print(f"[ERROR] Material {mp_id} failed during processing.", flush=True)
        return result
    except Exception as e:
        print(f"[ERROR] Material {mp_id} failed: {e}", flush=True)
        traceback.print_exc()
        return None

# ----------------- Worker initializer -----------------
def init_globals(cob_path, bispec_file):
    """
    Load globals for each worker process.
    """
    global COB_BASES, ALL_BISPECS, ALL_MP_IDS, ALL_RECLAT
    np.random.seed(os.getpid())
    COB_BASES = torch.load(cob_path,weights_only=True)
    with h5py.File(bispec_file, 'r') as f:
        ALL_MP_IDS = f["mp_ids"][:].astype(str)
        ALL_BISPECS = f["bispec"][:]
        ALL_RECLAT = f["recip_lattice"][:]
    #print("[INFO] Worker globals initialized.", flush=True)

def append_to_hdf5_safe(res, file_path):
    """
    Incrementally save one material to HDF5.
    Expects res dict with keys:
      'mat_id', 'pred', 'start_matrix', 'true_lattice', 'min_loss', 'start_id'
    """
    mat_id = res["mat_id"]
    pred = res["pred"].tolist()
    start_matrix = res["start_matrix"].tolist()
    true_lattice = res["true_lattice"].tolist() if res["true_lattice"] is not None else None
    min_loss = res["min_loss"]
    start_id = res["start_id"]

    with h5py.File(file_path, "a") as f:
        if mat_id in f:
            #print(f"[INFO] Material {mat_id} already in file, skipping.")
            return
        grp = f.create_group(mat_id)
        grp.create_dataset("pred", data=pred)
        grp.create_dataset("start_matrix", data=start_matrix)
        if true_lattice is not None:
            grp.create_dataset("true_lattice", data=true_lattice)
        grp.attrs["min_loss"] = min_loss
        grp.attrs["start_id"] = start_id
        print(f"[INFO] Material {mat_id} saved successfully.")

def append_batch_to_hdf5_safe(batch, f):
    """
    Save a batch of materials at once to an open HDF5 file.
    Expects batch = [res1, res2, ...]
    """
    for res in batch:
        mat_id = res["mat_id"]
        if mat_id in f:
            #print(f"[INFO] Material {mat_id} already in file, skipping.")
            continue
        grp = f.create_group(mat_id)
        grp.create_dataset("pred", data=res["pred"].tolist())
        grp.create_dataset("start_matrix", data=res["start_matrix"].tolist())
        if res["true_lattice"] is not None:
            grp.create_dataset("true_lattice", data=res["true_lattice"].tolist())
        grp.attrs["min_loss"] = res["min_loss"]
        grp.attrs["start_id"] = res["start_id"]
        #print(f"[INFO] Material {mat_id} saved successfully.")


def append_to_pickle(df_row, output_file):
    new_df = pd.DataFrame([df_row])

    if os.path.exists(output_file):
        existing_df = pd.read_pickle(output_file)
        # Remove any existing entry for this material
        existing_df = existing_df[existing_df["mat_id"] != df_row["mat_id"]]
        out_df = pd.concat([existing_df, new_df], ignore_index=True)
    else:
        out_df = new_df

    out_df.to_pickle(output_file)

# make these arguments
def get_true_sig(recip_lattice, wavelength,radial_function,n_basis,lmax,cob_bases):
    k_max = 2/wavelength
    miller_ind = get_neighbors(recip_lattice,k_max)
    neighs = torch.tensor(miller_ind.reshape(-1,3)@recip_lattice)
    true_bispec = get_bispectrum(neighs,k_max,lmax,radial_function,n_basis,cob_bases)#,cob_bases[str(lmax)])
    return true_bispec

def signal_from_params(mat,k_max,lmax,radial_function,n_basis):
    global COB_BASES
    # might need to detach this
    if torch.is_tensor(mat):
        mat_copy = mat.cpu().detach().clone()
        miller_ind = get_neighbors(mat_copy,k_max)
    else:
        miller_ind = get_neighbors(mat,k_max)
    if len(miller_ind) == 0:
        ### TODO if this is None, retry with increased cutoff??
        #breakpoint()
        #print("No neighbors found, restarting with a new lattice.")
        #return None  # Indicate failure or trigger retry logic
        # increasing k_max
        if torch.is_tensor(mat):
            mat_copy = mat.cpu().detach().clone().numpy()
            miller_ind = get_neighbors(mat_copy,2.0*np.max(mat_copy))
        else:
            miller_ind = get_neighbors(mat,2.0*np.max(mat))
    #neighs = torch.tensor(miller_ind.reshape(-1,3)).to(torch.double).to(device)@mat
    neighs = torch.tensor(miller_ind.reshape(-1,3)).to(torch.double)@mat
    curr_bispec = get_bispectrum(neighs,k_max,lmax,radial_function,n_basis,COB_BASES)#,cob_bases[str(lmax)])#.to(device))
    curr_bispec = torch.sign(curr_bispec)*torch.pow(torch.abs(curr_bispec),(1/3))
    return curr_bispec

# Define objective function for use with L-BFGS
def objective_function(params, obs_spectra, k_max, lmax, radial_function, n_basis):
    params = params.reshape((3,3))
    est_spectra = signal_from_params(params, k_max, lmax, radial_function, n_basis)
    loss_fn = torch.nn.MSELoss()
    if est_spectra is None:
        return float('inf')
    else:
        return loss_fn(est_spectra, obs_spectra).item()

def optimize_with_lbfgs(params, obs_spectra, k_max, lmax, radial_function, n_basis, cob_bases):
    """Changing to use Pytorch L-BFGS to run on GPU."""
    mat = torch.tensor(params.reshape((3,3)), dtype=torch.float64)#.to(device)
    mat.requires_grad = True
    optimizer = torch.optim.LBFGS([mat], lr=1e-2, max_iter=15000,line_search_fn="strong_wolfe")

    best_loss = float('inf')
    best_param = None
    for _ in range(50):
        def closure():
            optimizer.zero_grad()
            loss = objective_function(mat, obs_spectra, k_max, lmax, radial_function, n_basis, cob_bases)
            
            if loss is None or loss == float('inf'):
                #print("Restarting optimization with a new lattice.")
                return torch.tensor(float('inf'), dtype=torch.float64, device=device)  # Return high loss
            
            loss.backward()
            # Check if gradients are non-zero
            #if mat.grad is not None:
            #    print("Gradients: ", mat.grad.norm())
            #else:
            #    print("No gradients calculated")
            nonlocal best_loss, best_param
            if loss.item() < best_loss:
                best_loss = loss.item()
                best_param = mat.detach().cpu().numpy().copy()
            
            return loss
        
        optimizer.step(closure)
        #print(f"Current Best Loss: {best_loss}")  # Debugging output

    # Return only if optimization succeeded
    if best_loss == float('inf'):
        return None, None

    return best_loss, best_param

    

# Define function for gradient to be used in the Hessian matrix
def gradient_function(params,obs_spectra,k_max,lmax,radial_function,n_basis):
    mat = torch.tensor(params,dtype=torch.float64).reshape((3,3))
    #penalty = 1e5
    #det_penalty = penalty*(torch.linalg.det(mat) - 1.0)**2
    mat.requires_grad = True
    loss_fn = torch.nn.MSELoss()
    est_spectra = signal_from_params(mat,k_max,lmax,radial_function,n_basis)
    loss = loss_fn(est_spectra,obs_spectra)
    loss.backward()
    gradient = mat.grad.numpy().flatten()
    # Check for NaN or Inf in gradient
    if np.any(np.isnan(gradient)) or np.any(np.isinf(gradient)):
        #print("NaN or Inf found in gradient!")
        gradient = np.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)  # Replace NaNs or Infs with 0.
    return gradient

def get_best_elems(test_bispectrum, row_to_save_id, num):
    """
    Return top-N matching reciprocal lattices (excluding the current material).
    """
    global ALL_BISPECS, ALL_MP_IDS, ALL_RECLAT

    loss_fn = torch.nn.MSELoss(reduction="none")

    # Mask out the material itself
    mask = ALL_MP_IDS != row_to_save_id
    candidate_bispecs = torch.tensor(ALL_BISPECS[mask], dtype=torch.float64)
    candidate_recips = ALL_RECLAT[mask]
    candidate_ids = ALL_MP_IDS[mask]

    # Cube-root scaling
    candidate_bispecs = torch.sign(candidate_bispecs) * torch.pow(torch.abs(candidate_bispecs), 1/3)
    test_tensor = test_bispectrum.unsqueeze(0).expand_as(candidate_bispecs)

    # Element-wise MSE
    losses = loss_fn(test_tensor, candidate_bispecs).mean(dim=(1, 2))

    # Top-N smallest losses
    top_vals, top_indices = torch.topk(-losses, num)
    top_vals = -top_vals
    top_ids = [candidate_ids[i] for i in top_indices]
    top_recips = [candidate_recips[i] for i in top_indices]

    return top_vals, top_ids, top_recips
    
def track_best_params(xk, obs_spectra, k_max, lmax, radial_function, n_basis, cob_bases, best_loss, best_param):
    # Evaluate the current loss using the provided parameters (xk)
    current_loss = objective_function(xk, obs_spectra, k_max, lmax, radial_function, n_basis, cob_bases)
    
    # Check if the current loss is better (lower)
    if current_loss < best_loss:
        best_loss = current_loss
        best_param = xk.reshape((3, 3))  # Save the current parameters corresponding to the best loss
        #print(f"New best loss found: {best_loss} with parameters: {best_param}")
    
    return best_loss, best_param  # Return the updated values
    
def run_minimize_with_timeout_callback(start_mat, obs_spectra, k_max, lmax, radial_function, n_basis, timeout_seconds=15):
    import time
    start_time = time.time()
    best_loss = np.inf
    best_param = None
    global COB_BASES
    def callback(xk):
        nonlocal best_loss, best_param
        current_loss = objective_function(xk, obs_spectra, k_max, lmax, radial_function, n_basis)
        if current_loss < best_loss:
            best_loss = current_loss
            best_param = xk.reshape((3,3))
        if time.time() - start_time > timeout_seconds:
            raise StopIteration  # stops minimize gracefully

    try:
        from scipy.optimize import minimize
        res = minimize(
            objective_function,
            start_mat,
            args=(obs_spectra, k_max, lmax, radial_function, n_basis),
            jac=gradient_function,
            method='L-BFGS-B',
            callback=callback,
            options={'maxiter': 2000, 'maxfun': 2000, 'gtol':1e-6}
        )
    except StopIteration:
        print(f"[INFO] Optimization timed out after {timeout_seconds}s", flush=True)

    return best_loss, best_param

def run_basinhopping_with_timeout(start_mat, obs_spectra, k_max, lmax, radial_function, n_basis,
                                   timeout_seconds=15, niter=20, stepsize=0.1):
    """Basin hopping with L-BFGS-B as the local optimizer and a wall-clock timeout."""
    import time
    from scipy.optimize import basinhopping

    start_time = time.time()
    best_loss = np.inf
    best_param = None

    minimizer_kwargs = {
        'method': 'L-BFGS-B',
        'args': (obs_spectra, k_max, lmax, radial_function, n_basis),
        'jac': gradient_function,
        'options': {'maxiter': 2000, 'maxfun': 2000, 'gtol': 1e-6},
    }

    def bh_callback(x, f, accepted):
        nonlocal best_loss, best_param
        if f < best_loss:
            best_loss = f
            best_param = x.reshape(3, 3).copy()
        return time.time() - start_time > timeout_seconds  # returning True stops basin hopping

    try:
        res = basinhopping(
            objective_function,
            start_mat,
            minimizer_kwargs=minimizer_kwargs,
            niter=niter,
            stepsize=stepsize,
            callback=bh_callback,
        )
        if res.fun < best_loss:
            best_loss = res.fun
            best_param = res.x.reshape(3, 3).copy()
    except Exception:
        pass

    return best_loss, best_param


def worker_wrapper(args, timeout):
    """Run worker with timeout in its own process"""
    q = multiprocessing.Queue()

    def target():
        try:
            q.put(worker(args))
        except Exception as e:
            q.put(e)

    p = multiprocessing.Process(target=target)
    p.start()
    p.join(timeout)

    if p.is_alive():
        p.terminate()
        p.join()
        raise TimeoutError(f"Material {args[0]} timed out after {timeout}s")

    result = q.get()
    if isinstance(result, Exception):
        raise result
    return result


def process_row(
    mp_id, test_bispectrum, radial_function, n_basis, wavelength, lmax,
    num_to_opt=15, threshold=1e-2, timeout_seconds=10, true_lat_params=None,
    random_restarts=0, init_from_true=False, true_start_timeout=150,
    basin_hopping=False, bh_niter=20, bh_stepsize=0.1,
    random_restart_min_abc=3.0, random_restart_max_abc=15.0,
):
    """
    Processes one row and inverts using HDF5 file per material.
    Returns a dictionary of the best lattice parameters.
    """
    global COB_BASES, ALL_BISPECS, ALL_MP_IDS, ALL_RECLAT

    # should follow this in the training procedure!!!
    test_bispectrum[np.abs(test_bispectrum) < threshold] = 1e-10
    obs_spectra = torch.tensor(test_bispectrum, dtype=torch.float64)
    #obs_spectra = torch.sign(obs_spectra) * torch.pow(torch.abs(obs_spectra), 1/3)
    idx_arr = np.where(ALL_MP_IDS == mp_id)[0]
    has_true_lattice = len(idx_arr) > 0
    if has_true_lattice:
        true_recip_lattice = ALL_RECLAT[idx_arr[0]].reshape((3,3))
    elif true_lat_params is not None:
        # RRUFF path: convert [a,b,c,alpha,beta,gamma] to real-space matrix
        from pymatgen.core import Lattice as PmgLattice
        a, b, c, al, be, ga = [float(x) for x in true_lat_params]
        true_real_lattice = PmgLattice.from_parameters(a, b, c, al, be, ga).matrix
        print(f"[INFO] {mp_id} true lattice (a,b,c,α,β,γ): {a:.4f} {b:.4f} {c:.4f} {al:.4f} {be:.4f} {ga:.4f}", flush=True)
        has_true_lattice = True
        true_recip_lattice = None  # signal to use real-space path below
    else:
        true_recip_lattice = None

    mask = ALL_MP_IDS != mp_id  # no-op if mp_id not in DB, but safe either way
    bispec_tensor = torch.tensor(ALL_BISPECS[mask], dtype=torch.float64)
    recip_lattices_list = ALL_RECLAT[mask]
    mp_ids = ALL_MP_IDS[mask]

    bispec_tensor = torch.sign(bispec_tensor) * torch.pow(torch.abs(bispec_tensor), 1/3)
    test_tensor = obs_spectra.unsqueeze(0).expand_as(bispec_tensor)

    loss_fn = torch.nn.MSELoss(reduction='none')
    losses = loss_fn(test_tensor, bispec_tensor).mean(dim=(1,2))
    top_vals, top_indices = torch.topk(-losses, num_to_opt)
    top_vals = -top_vals
    best_mp_ids = [mp_ids[i] for i in top_indices]
    best_recips = [recip_lattices_list[i] for i in top_indices]

    cutoff = 1e-3
    best_loss = np.inf
    best_param = None
    noise_rel = 0.05
    k_max = 2 / wavelength

    #print(f"[INFO] Analyzing material id {mp_id} with top {num_to_opt} candidates", flush=True)

    candidates = list(zip(best_mp_ids, best_recips))
    if random_restarts > 0:
        for recip in sample_random_recip_lattices(
            random_restarts, min_abc=random_restart_min_abc, max_abc=random_restart_max_abc
        ):
            candidates.append((None, recip))

    if init_from_true:
        if not has_true_lattice:
            print(f"[WARN] {mp_id}: --init-from-true requested but no true lattice available; "
                  f"falling back to NN starts.", flush=True)
        else:
            if true_recip_lattice is not None:
                true_recip_start = true_recip_lattice.reshape(3, 3)
            else:
                true_recip_start = np.linalg.inv(true_real_lattice.T)
            timeout_seconds = true_start_timeout
            candidates = [('__true__', true_recip_start)]
            print(f"[TRUE-START] {mp_id}: optimising from true lattice, timeout={timeout_seconds}s", flush=True)

    winning_id, winning_start = None, None
    for best_id, recip_lattice in candidates:
        start_mat = recip_lattice.reshape(9) + np.random.randn(9) * noise_rel * np.linalg.norm(recip_lattice)

        if basin_hopping:
            candidate_loss, candidate_param = run_basinhopping_with_timeout(
                start_mat, obs_spectra, k_max, lmax, radial_function, n_basis,
                timeout_seconds=timeout_seconds, niter=bh_niter, stepsize=bh_stepsize,
            )
        else:
            candidate_loss, candidate_param = run_minimize_with_timeout_callback(
                start_mat, obs_spectra, k_max, lmax, radial_function, n_basis, timeout_seconds
            )

        if best_id == '__true__':
            print(f"[TRUE-START] {mp_id}: loss={candidate_loss:.6f} "
                  f"(converged={'yes' if candidate_loss is not None and candidate_loss < cutoff else 'no'})",
                  flush=True)

        if candidate_loss < best_loss:
            best_loss = candidate_loss
            best_param = candidate_param
            winning_id, winning_start = best_id, start_mat  # the start that produced best_param

        if best_loss < cutoff:
            #print(f"[INFO] Early stopping with candidate {best_id}, best_loss = {best_loss}", flush=True)
            break

    # Check Niggli reduced lattice (only if ground truth is available)
    if has_true_lattice and true_recip_lattice is not None:
        # MP path: both pred and true are reciprocal lattice matrices
        best_param_reduced, true_lattice_reduced, diff = check_prim_lattices_niggli(
            best_param, true_recip_lattice, symprec=1e-1, tol=1e-2, is_reciprocal=True
        )
    elif has_true_lattice and true_recip_lattice is None:
        # RRUFF path: convert pred from reciprocal to real space, true is already real
        best_param_real = np.linalg.pinv(np.array(best_param).T)
        best_param_reduced, true_lattice_reduced, diff = check_prim_lattices_niggli(
            best_param_real, true_real_lattice, symprec=1e-1, tol=1e-2, is_reciprocal=False
        )
    else:
        best_param_reduced = best_param
        true_lattice_reduced = None

    return {
        'mat_id': mp_id,
        'pred': best_param_reduced,
        'min_loss': best_loss,
        # id / matrix of the start that produced the returned lattice (None id = random start).
        # Earlier versions returned the LAST candidate tried here, not the winner.
        'start_matrix': (winning_start if winning_start is not None else start_mat).reshape(3,3),
        'start_id': str(winning_id),
        'true_lattice': true_lattice_reduced
    }

def main():
    options = getArgumentParser().parse_args()

    radial_function = options.radial
    n_basis = options.nbasis
    wavelength = options.wavelength
    input_cob = options.cob
    bispec_check_file = options.mat_proj_df
    lmax = int(options.lmax)
    begin_ind = int(options.begin_ind)
    num_entries = int(options.num_entries)
    max_workers = int(options.max_workers)
    timeout = int(options.timeout)
    random_restarts = int(options.random_restarts)
    num_to_opt = int(options.num_to_opt)
    init_from_true = bool(options.init_from_true)
    true_start_timeout = int(options.true_start_timeout)
    basin_hopping = bool(options.basin_hopping)
    bh_niter = int(options.bh_niter)
    bh_stepsize = float(options.bh_stepsize)
    random_restart_min_abc = float(options.random_restart_min_abc)
    random_restart_max_abc = float(options.random_restart_max_abc)
    candidate_timeout = int(options.candidate_timeout)
    end_ind = begin_ind + num_entries

    output_dir = options.output
    if getattr(options, "material_file", None) is not None:
        # Append retried results directly into the specified output file
        output_file = output_dir
        output_dir = os.path.dirname(output_file)
    else:
        output_file = options.output
        output_dir = os.path.dirname(output_file)
    os.makedirs(output_dir, exist_ok=True)

    # Load input dictionary {mat_id: prediction}
    input_dict = pd.read_pickle(options.input)
    #losses = np.array(input_dict['per_sample_loss'])

    id_key = 'material_ids' if 'material_ids' in input_dict else 'mineral_names'
    # True lattice params [a,b,c,alpha,beta,gamma] per entry, if present (e.g. RRUFF eval output)
    true_lats = input_dict.get('targets', None)

    # check that these are lattice parameters
    if true_lats is not None:
        if true_lats.shape[1] != 6:
            true_lats = None

    mat_ids = [input_dict[id_key][i] for i in range(begin_ind,end_ind)]
    predictions = [input_dict['predictions'][i] for i in range(begin_ind,end_ind)]
    true_lat_list = [true_lats[i].tolist() if true_lats is not None else None for i in range(begin_ind, end_ind)]

    if getattr(options, "material_file", None) is not None:
        # Load material IDs from a file
        with open(options.material_file) as f:
            mat_ids = [line.strip() for line in f if line.strip()]

        # Map to predictions and true lattice params
        id_to_pred = dict(zip(input_dict[id_key], input_dict['predictions']))
        id_to_lat = dict(zip(input_dict[id_key], true_lats.tolist())) if true_lats is not None else {}
        predictions = [id_to_pred.get(mid, None) for mid in mat_ids]
        true_lat_list = [id_to_lat.get(mid, None) for mid in mat_ids]

        # Warn if some IDs are not in the input_dict
        missing_in_input = [mid for mid, pred in zip(mat_ids, predictions) if pred is None]
        if missing_in_input:
            print(f"[WARN] The following IDs not found in input: {missing_in_input}")

    else:
        # Default slicing behavior
        mat_ids = [str(input_dict[id_key][i]).strip() for i in range(begin_ind, end_ind)]
        predictions = [input_dict['predictions'][i] for i in range(begin_ind, end_ind)]
        true_lat_list = [true_lats[i].tolist() if true_lats is not None else None for i in range(begin_ind, end_ind)]

    # Prepare arguments for each worker
    # Keep track of already completed materials
    # Determine completed materials only if NOT retrying
    completed = set()
    if getattr(options, "material_file", None) is None and os.path.exists(output_file):
        try:
            with h5py.File(output_file, "r") as f:
                completed = set(str(k) for k in f.keys())
                print(f"[INFO] Found {len(completed)} completed materials, skipping them.")
        except Exception:
            print("[WARN] Could not read existing output file, starting fresh.")
    # Prepare worker arguments
    worker_args = [
        (mp_id, pred, radial_function, n_basis, wavelength, lmax, true_lat, random_restarts, num_to_opt, init_from_true, true_start_timeout, basin_hopping, bh_niter, bh_stepsize, random_restart_min_abc, random_restart_max_abc, candidate_timeout)
        for mp_id, pred, true_lat in zip(mat_ids, predictions, true_lat_list)
        # Only skip completed if NOT retrying
        if getattr(options, "material_file", None) is not None or mp_id not in completed
    ]
    print(f"[INFO] Number of materials to process: {len(worker_args)}")
    BATCH_SIZE = 20
    batch = []
    completed = []
    failed = []

    with ProcessPoolExecutor(
        max_workers=max_workers,
        initializer=init_globals,
        initargs=(input_cob, bispec_check_file)
    ) as executor:

        future_to_args = {
            executor.submit(worker_wrapper, args, timeout): args
            for args in worker_args
        }

        future_to_args[executor.submit(lambda: None)] = None  # Dummy future to avoid hanging?

        for future in tqdm(
            as_completed(future_to_args),
            total=len(future_to_args),
            desc="Processing materials"
        ):
            args = future_to_args[future]
            if args is None:
                continue  # Skip dummy future
            mp_id = args[0]

            try:
                res = future.result()  # No timeout here, worker_wrapper handles it

                if res is not None:
                    batch.append(res)
                    completed.append(mp_id)


                else:
                    failed.append(args)
                # Flush batch
                if len(batch) >= BATCH_SIZE:
                    with h5py.File(output_file, "a") as f:
                        append_batch_to_hdf5_safe(batch, f)
                    batch = []

            except Exception as e:
                print(f"[ERROR] Material {mp_id} failed: {e}", flush=True)
                failed.append(args)

    # Save leftover batch
    if len(batch) > 0:
        with h5py.File(output_file, "a") as f:
            append_batch_to_hdf5_safe(batch, f)

    print(f"[INFO] Finished batch {begin_ind}-{end_ind}. Results saved to {output_file}")

    from concurrent.futures import ThreadPoolExecutor, TimeoutError

    # Retry failed serially
    print(f"[INFO] Retrying {len(failed)} failures serially")
    init_globals(input_cob, bispec_check_file)
    timeout_seconds = 300  # short timeout for serial retry
    all_failed_ids = set()
    for args in tqdm(failed, desc="Serial retry"):
        #print(args)
        mp_id = args[0]
        try:
            res = worker(args)
            #print(res)
            if res is not None:
                with h5py.File(output_file, "a") as f:
                    append_batch_to_hdf5_safe([res], f)
            else:
                all_failed_ids.add(mp_id)
        except TimeoutError:
            print(f"[WARN] Serial retry timed out for {mp_id} after {timeout_seconds}s")
            all_failed_ids.add(mp_id)
        except Exception as e:
            print(f"[ERROR] Serial retry failed for {mp_id}: {e}")
            traceback.print_exc()
            all_failed_ids.add(mp_id)
    # Save all remaining failed IDs to a file
    if all_failed_ids:
        failed_file = os.path.join(output_dir, "failed_materials.txt")
        with open(failed_file, "w") as f:
            for mid in sorted(all_failed_ids):
                f.write(f"{mid}\n")
        print(f"[INFO] {len(all_failed_ids)} materials still failed. Saved to {failed_file}")


if __name__ == "__main__":
    main()
