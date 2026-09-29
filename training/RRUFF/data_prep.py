import sys
import os
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utilities import get_sim_xrd_from_pattern, save_results_pickle

from pymatgen.core import Structure
from pymatgen.core import Lattice
from pymatgen.analysis.diffraction.core import DiffractionPattern
from pymatgen.analysis.diffraction.xrd import XRDCalculator
import numpy as np
from pymatgen.analysis.structure_analyzer import SpacegroupAnalyzer
import pandas as pd
import scipy.stats as stats
import matplotlib.pyplot as plt
import ase
from matscipy.neighbours import neighbour_list
import plotly.graph_objects as go
import torch
import torch.nn as nn
import e3nn
from e3nn import o3, io
import e3nn_jax as e3nn_jax
import pickle
from dataclasses import dataclass

@dataclass
class XRDStructure:
    """Class to hold linked XRD data for a single pattern"""
    metadata: pd.DataFrame
    atoms: pd.DataFrame
    pattern: pd.DataFrame


def get_sim_xrd_from_rruff_df(rruff_df, index):
    """
    Get the simulated XRD pattern using the structure from the processed RRUFF dataframe.

    Returns:
    numpy array: simulated XRD pattern
    """
    pattern_df = rruff_df[index].pattern
    pattern = DiffractionPattern(pattern_df['2theta'], pattern_df['intensity'], None, None) #converts dataframe into DiffractionPattern
    sim_xrd = get_sim_xrd_from_pattern(pattern)

    return sim_xrd

def plot_xrd(result, key_list, index=None, save=False):
    """
    plot simulated XRD pattern
    """
    sim_xrd = get_sim_xrd_from_rruff_df(result, key_list[index])
    two_theta = np.arange(5, 90, 0.010)
    plt.figure(figsize=(10, 6))
    plt.plot(two_theta, sim_xrd)
    plt.xlabel('2θ (degrees)') #it is two theta
    plt.ylabel('Intensity (a.u.)')

    if index is not None:
        plt.title(f'XRD Pattern - Material {index}')

    if save:
        plt.savefig(f'plots/rruff_xrd_pattern_{index}.png', dpi=300, bbox_inches='tight')

    plt.show()

def get_neighbors(recip_lat,k_max,filter_orig=True,filter_new=False,miller_ind_threshold=3,norm = False, norm_threshold = 2):
    # atom type shouldn't matter
    recip = ase.Atoms(symbols = ['C'],positions = np.array([[0.0,0.0,0.0]]),cell=recip_lat,pbc=True)
    neighs_dist,neighs_miller = neighbour_list('dS',recip,cutoff=k_max)
    neighs_dist = np.asarray(neighs_dist)
    neighs_miller = np.asarray(neighs_miller)
    # have a cutoff for miller indices in neighbors
    # to yield more information
    # or have a radial cutoff
    # or penalize Gaussian basis functions more
    if filter_orig:
        try:
            mask = np.any(np.abs(neighs_miller) > miller_ind_threshold, axis=1)
            return neighs_miller[~mask]
        except:
            return neighs_miller
    elif filter_new:
        try:
            mask = np.any(np.abs(neighs_miller) > miller_ind_threshold, axis=1)
            filtered_dists = neighs_dist[mask]
            min_dist = min(filtered_dists)
            dist_mask = np.where(neighs_dist <= min_dist)
            filtered_neighs = neighs_miller[dist_mask]
            return filtered_neighs
        except:
            return neighs_miller
    elif norm:
        neighs_miller = neighs_miller/(np.linalg.norm(neighs_miller,axis=1)**norm_threshold)[:,None]
        return neighs_miller
    else:
        return neighs_miller

def get_bispectrum(vec,k_max,lmax,radial_function,n_basis,cob_bi):
    bases = ["bessel","gaussian", "cosine", "smooth_finite", "fourier","none"]
    assert radial_function in bases, "Invalid basis function"
    #res = torch.zeros(vec.shape[0],n_basis,(sph.lmax+1)**2)
    # get spherical harmonics
    sph_harm = o3.spherical_harmonics(range(0,lmax+1),vec,normalize="False")
    radii = torch.linalg.norm(vec,axis=-1)
    basis = e3nn.math.soft_one_hot_linspace(radii,0,k_max,number=n_basis,basis=radial_function,cutoff=False)
    # Add a new dimension to the tensors
    sph_harm = sph_harm.unsqueeze(1)  # shape: (136, 10, 1)
    basis = basis.unsqueeze(2)  # shape: (136, 1, 49)
    combined_tensor = torch.matmul(basis,sph_harm)
    radial_proj = torch.sum(combined_tensor,axis=0)
    bispec = torch.einsum("...i,...j,...k,ijkz->...z",radial_proj,radial_proj,radial_proj,cob_bi)
    return bispec

def training_data_gen(rruff_df, key_list, k_max=2/3, l_max=6):
    bispec_list = []
    for ind in tqdm(range(len(rruff_df))):
        raw_cr = rruff_df[key_list[ind]].metadata
        raw_lat = Lattice.from_parameters(raw_cr['a'].iloc[0], raw_cr['b'].iloc[0], raw_cr['c'].iloc[0],
                raw_cr['alpha'].iloc[0], raw_cr['beta'].iloc[0], raw_cr['gamma'].iloc[0]) #create Lattice
        raw_str = Structure(raw_lat, ['C'], np.zeros([1, 3]))
        si_prim = raw_str.get_primitive_structure()
        recip_lattice = si_prim.lattice.reciprocal_lattice_crystallographic
        miller_ind = get_neighbors(recip_lattice.matrix, float(k_max), filter_orig=True, filter_new=False, miller_ind_threshold=3) #cutoff hyperparam
        neighs = torch.tensor(miller_ind.reshape(-1, 3) @ recip_lattice.matrix)

        rtp_bi = e3nn_jax.reduced_symmetric_tensor_product_basis(io.SphericalTensor(l_max, 1, -1), 3, keep_ir=['0o', '0e'])
        cob_bi = torch.tensor(rtp_bi.array,dtype=torch.float64)
        bispec = get_bispectrum(neighs, float(k_max), lmax=l_max, radial_function='bessel', n_basis=10, cob_bi=cob_bi) #l_max hyperparam
        if id == 0:
            print(bispec.shape)
            print(type(bispec))
        bispec_list.append(bispec)
    return bispec_list


if __name__ == "__main__":
    with open('RRUFF/xrd_data.pkl', 'rb') as f:
        ordered_result = pickle.load(f)

    key_list = list(ordered_result.keys())
  #  sim_xrd = []
  #  for i in tqdm(range(len(ordered_result))):
  #      sim_xrd.append(get_sim_xrd_from_rruff_df(ordered_result,key_list[i]))

  #  save_results_pickle(sim_xrd, 'pickles/rruff_sim_xrd.pkl')
    
  #  bispec_list = training_data_gen(ordered_result, key_list)
  #  save_results_pickle(bispec_list, 'pickles/rruff_bispec_lat.pkl')
    plot_xrd(ordered_result, key_list, 7, True) 

