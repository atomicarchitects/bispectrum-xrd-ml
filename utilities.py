from enum import Enum, Flag
import hashlib
import numpy as np
import pymatgen
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.core.structure import Structure, Lattice
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from matscipy.neighbours import neighbour_list
import matplotlib.pyplot as plt
import ase
import e3nn
from e3nn import o3, io
import torch
import pickle

_xrd_calculator = XRDCalculator(wavelength='CuKa', symprec=0.1)

def struct_from_lat(lat_mat):
    return Structure(lat_mat, ['C'], np.zeros([1, 3]))

def gen_lattice(lattice,low_ucell,high_ucell):
    # ignore if we generate a bad structure
    vol = 1
    while vol <= 1:
        a = np.random.uniform(low=low_ucell,high=high_ucell)
        b = np.random.uniform(low=low_ucell,high=high_ucell)
        c = np.random.uniform(low=low_ucell,high=high_ucell)
        alpha = np.random.uniform(low=10,high=180)
        beta = np.random.uniform(low=10,high=180)
        gamma = np.random.uniform(low=10,high=180)
        #params = []
        if lattice == "cubic":
            lat = Lattice.cubic(a)
            params = [a]
        if lattice == "hexagonal":
            lat = Lattice.hexagonal(a,c)
            params = [a,c]
        if lattice == "rhombohedral":
            lat = Lattice.rhombohedral(a,alpha)
            params = [a,alpha]
        if lattice == "tetragonal":
            lat = Lattice.tetragonal(a,c)
            params = [a,c]
        if lattice == "orthorhombic":
            lat = Lattice.orthorhombic(a,b,c)
            params = [a,b,c]
        if lattice == "monoclinic":
            lat = Lattice.monoclinic(a,b,c,beta)
            params = [a,b,c,beta]
        if lattice == "triclinic":
            lat = Lattice.from_parameters(a,b,c,alpha,beta,gamma)
            params = [a,b,c,alpha,beta,gamma]
        vol = lat.volume
    return lat, params

def lattice_param_to_vertices(a,b,c,alpha, beta, gamma):
    # Convert degrees to radians
    alpha = np.radians(alpha)
    beta = np.radians(beta)
    gamma = np.radians(gamma)

    # Define lattice vectors
    a_vector = np.array([a, 0, 0])
    b_vector = np.array([b * np.cos(gamma), b * np.sin(gamma), 0])
    cx = c * np.cos(beta)
    cy = (c * np.cos(alpha) - c * np.cos(beta) * np.cos(gamma)) / np.sin(gamma)
    cz = np.sqrt(c**2 - cx**2 - cy**2)
    c_vector = np.array([cx, cy, cz])

    # Define vertices
    vertices = np.array([
        [0, 0, 0],
        a_vector,
        b_vector,
        a_vector + b_vector,
        c_vector,
        c_vector + a_vector,
        c_vector + b_vector,
        c_vector + a_vector + b_vector
    ])

    return vertices

def connect_vertices(lattice):
    vertices = lattice_param_to_vertices(lattice.a,lattice.b,lattice.c,
                                         lattice.alpha,lattice.beta,lattice.gamma)
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),  # Base
        (0, 4), (1, 5), (2, 6), (3, 7),  # Connect base to top
        (4, 5), (5, 7), (7, 6), (6, 4)   # Top
    ]
    x_edges = []
    y_edges = []
    z_edges = []
    for edge in edges:
        for vi in edge:
            x_edges.append(vertices[vi][0])
            y_edges.append(vertices[vi][1])
            z_edges.append(vertices[vi][2])
        x_edges.append(None)
        y_edges.append(None)
        z_edges.append(None)
    return x_edges, y_edges, z_edges

def gen_system_lattice(
    lattice,
    a_value,
    low_c=0.8,
    high_c=1.5,
    seed=None,
):
    """
    Generate a representative lattice for bispectrum visualization.
    Emphasizes symmetry differences over chemical realism.
    Reproducible if `seed` is provided.
    """

    rng = np.random.default_rng(seed)

    # helper: log-uniform scaling
    def log_uniform(low, high):
        return np.exp(rng.uniform(np.log(low), np.log(high)))

    angle_triclinic = (75, 105)
    beta_monoclinic = rng.normal(90, 5)

    if lattice == "cubic":
        lat = Lattice.cubic(a_value)
        params = [a_value]

    elif lattice == "tetragonal":
        c = a_value * log_uniform(low_c, high_c)
        lat = Lattice.tetragonal(a_value, c)
        params = [a_value, c]

    elif lattice == "orthorhombic":
        b = a_value * log_uniform(low_c, high_c)
        c = a_value * log_uniform(low_c, high_c)
        lat = Lattice.orthorhombic(a_value, b, c)
        params = [a_value, b, c]

    elif lattice == "hexagonal":
        c = a_value * log_uniform(low_c, high_c)
        lat = Lattice.hexagonal(a_value, c)
        params = [a_value, c]

    elif lattice == "rhombohedral":
        alpha = rng.normal(60, 2)
        lat = Lattice.rhombohedral(a_value, alpha)
        params = [a_value, alpha]

    elif lattice == "monoclinic":
        b = a_value * log_uniform(low_c, high_c)
        c = a_value * log_uniform(low_c, high_c)
        lat = Lattice.monoclinic(a_value, b, c, beta_monoclinic)
        params = [a_value, b, c, beta_monoclinic]

    elif lattice == "triclinic":
        b = a_value * log_uniform(low_c, high_c)
        c = a_value * log_uniform(low_c, high_c)
        alpha = rng.uniform(*angle_triclinic)
        beta = rng.uniform(*angle_triclinic)
        gamma = rng.uniform(*angle_triclinic)
        lat = Lattice.from_parameters(a_value, b, c, alpha, beta, gamma)
        params = [a_value, b, c, alpha, beta, gamma]

    else:
        raise ValueError(f"Unknown lattice type: {lattice}")

    # Volume sanity check
    vol_ref = a_value ** 3
    if not (0.3 * vol_ref < lat.volume < 3.0 * vol_ref):
        # shift seed deterministically to avoid infinite recursion
        return gen_system_lattice(
            lattice, a_value, low_c, high_c,
            seed=None if seed is None else seed + 1
        )

    return lat, params

def plot_bispectrum(bispec, title="Bispectrum", num_basis=10, vmin=None, vmax=None, cmap="viridis"):
    fig, ax = plt.subplots(figsize=(20, 10))

    im = ax.imshow(bispec, cmap=cmap, vmin=vmin, vmax=vmax)
    ax.set_xticks([])
    ax.set_yticks(np.linspace(0, bispec.shape[0] - 1, num=num_basis))
    ax.set_yticklabels([f"{i:.0f}" for i in np.linspace(0, bispec.shape[0] - 1, num=num_basis)], fontsize=18)
    ax.set_ylabel("Radial Basis Function", fontsize=18)
    ax.set_title(title, fontsize=22)

    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.tick_params(labelsize=16)

    return fig

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

# calculate the bispectrum for each interpolated matrix
def get_true_sig(recip_lattice, wavelength,radial_function,n_basis,lmax,cob_bi):
    k_max = 2/wavelength
    miller_ind = get_neighbors(recip_lattice,k_max,False, False, 3, False,-1)
    neighs = torch.tensor(miller_ind.reshape(-1,3)@recip_lattice)
    true_bispec = get_bispectrum(neighs,k_max,lmax,radial_function,n_basis,cob_bi)
    return true_bispec

## pymatgen lattice function so it is differentiable for sensitivity analysis
def lat_from_parameters(
    a: float,
    b: float,
    c: float,
    alpha: float,
    beta: float,
    gamma: float,
    threshold = 1e-6
):
    """Create a Lattice using unit cell lengths (in Angstrom) and angles (in degrees).

    Args:
        a (float): *a* lattice parameter.
        b (float): *b* lattice parameter.
        c (float): *c* lattice parameter.
        alpha (float): *alpha* angle in degrees.
        beta (float): *beta* angle in degrees.
        gamma (float): *gamma* angle in degrees.
        vesta (bool): True if you import Cartesian coordinates from VESTA.
        pbc (tuple): a tuple defining the periodic boundary conditions along the three
            axis of the lattice. If None periodic in all directions.

    Returns:
        Lattice with the specified lattice parameters.
    """
    angles_r = torch.tensor([alpha, beta, gamma])*(np.pi/180)
    print(angles_r)
    cos_alpha, cos_beta, cos_gamma = torch.cos(angles_r)
    sin_alpha, sin_beta, sin_gamma = torch.sin(angles_r)

    val = (cos_alpha * cos_beta - cos_gamma) / (sin_alpha * sin_beta)
    val = torch.clamp(val, -1, 1)  # rounding errors may cause values slightly > 1
    print(val)
    gamma_star = torch.arccos(val)
    print(cos_beta)
    vector_a = torch.tensor([a * sin_beta, 0.0, a * cos_beta])
    vector_b = torch.tensor([
        -b * sin_alpha * torch.cos(gamma_star),
        b * sin_alpha * torch.sin(gamma_star),
        b * cos_alpha,
    ])
    vector_c = torch.tensor([0.0, 0.0, float(c)])
    lat_mat = torch.stack([vector_a, vector_b, vector_c])
    #lat_mat_round = 
    # apply soft threshold?
    return lat_mat
    #return lat_mat*torch.tanh(lat_mat/threshold)
    

def check_prim_lattices(lat1,lat2,symprec=0.1,tol=1e-5,is_reciprocal=True):
    '''
    Checks if 2 lattices are the same by converting 
    to the primitive setting. Note we use the crystallographic setting
    and it is assumed that the inputs are reciprocal lattice.
    Default symmetry precision is 0.1.
    '''
    if is_reciprocal:
        # convert to real space lattice
        lat1_real = np.linalg.pinv(np.array(lat1).T)
        lat2_real = np.linalg.pinv(np.array(lat2).T)
    else:
        lat1_real = np.array(lat1)
        lat2_real = np.array(lat2)

    s1 = struct_from_lat(lat1_real)
    s2 = struct_from_lat(lat2_real)

    try:
        sga1 = SpacegroupAnalyzer(s1,symprec=symprec).get_primitive_standard_structure()
    except Exception:
        sga1 = s1
    try:
        sga2 = SpacegroupAnalyzer(s2,symprec=symprec).get_primitive_standard_structure()
    except Exception:
        sga2 = s2

    res = np.allclose(sga1.lattice.matrix, sga2.lattice.matrix,atol=tol)
    return res

def convert_lats_to_prim(lat1,lat2,symprec=0.1,tol=1e-5,is_reciprocal=True):
    '''
    Converts two lattices to the primitive standard setting. Note we use the crystallographic setting
    and it is assumed that the inputs are reciprocal lattice.
    Default symmetry precision is 0.1.
    '''
    if is_reciprocal:
        # convert to real space lattice
        lat1_real = np.linalg.pinv(np.array(lat1).T)
        lat2_real = np.linalg.pinv(np.array(lat2).T)
    else:
        lat1_real = np.array(lat1)
        lat2_real = np.array(lat2)

    s1 = struct_from_lat(lat1_real)
    s2 = struct_from_lat(lat2_real)

    try:
        sga1 = SpacegroupAnalyzer(s1,symprec=symprec).get_primitive_standard_structure()
    except Exception:
        sga1 = s1
    try:
        sga2 = SpacegroupAnalyzer(s2,symprec=symprec).get_primitive_standard_structure()
    except Exception:
        sga2 = s2
    return sga1.lattice, sga2.lattice

def convert_lats_to_selling(lat1,lat2,symprec=0.1,is_reciprocal=True):
    if is_reciprocal:
        # convert to real space lattice
        lat1_real = np.linalg.pinv(np.array(lat1).T)
        lat2_real = np.linalg.pinv(np.array(lat2).T)
    else:
        lat1_real = np.array(lat1)
        lat2_real = np.array(lat2)
    s1 = struct_from_lat(lat1_real)
    s2 = struct_from_lat(lat2_real)
    '''
    assumes ordering a,b,c,alpha,beta,gamma and that angle parameters are in degrees
    '''
    # do i need this? 
    #sga1 = SpacegroupAnalyzer(s1,symprec=symprec).get_primitive_standard_structure()
    prim1_lattice = s1.lattice
    #sga2 = SpacegroupAnalyzer(s2,symprec=symprec).get_primitive_standard_structure()
    prim2_lattice = s2.lattice
    l1_params = np.array([prim1_lattice.a, prim1_lattice.b, prim1_lattice.c,
                         np.radians(prim1_lattice.alpha),
                         np.radians(prim1_lattice.beta),
                         np.radians(prim1_lattice.gamma)])[None,:]
    l2_params = np.array([prim2_lattice.a, prim2_lattice.b, prim2_lattice.c,
                         np.radians(prim2_lattice.alpha),
                         np.radians(prim2_lattice.beta),
                         np.radians(prim2_lattice.gamma)])[None,:]
    l1_unit_cell_reduced, l1_hkl_transformation, l1_s6_reduced = selling_reduction(l1_params)
    l2_unit_cell_reduced, l2_hkl_transformation, l2_s6_reduced = selling_reduction(l2_params)
    # assume we are just reducing one unit cell at a time
    l1_unit_cell_reduced = l1_unit_cell_reduced[0]
    l2_unit_cell_reduced = l2_unit_cell_reduced[0]
    # hmm is the best way to define the distances... seems redundant
    # could directly measure distance in s6 space/that might make more sense
    l1_reduced_lat = Lattice.from_parameters(l1_unit_cell_reduced[0],
                                             l1_unit_cell_reduced[1],
                                             l1_unit_cell_reduced[2],
                                             np.degrees(l1_unit_cell_reduced[3]),
                                             np.degrees(l1_unit_cell_reduced[4]),
                                             np.degrees(l1_unit_cell_reduced[5]))
    l2_reduced_lat = Lattice.from_parameters(l2_unit_cell_reduced[0],
                                             l2_unit_cell_reduced[1],
                                             l2_unit_cell_reduced[2],
                                             np.degrees(l2_unit_cell_reduced[3]),
                                             np.degrees(l2_unit_cell_reduced[4]),
                                             np.degrees(l2_unit_cell_reduced[5]))
    
    return l1_reduced_lat, l2_reduced_lat

def convert_lats_to_niggli(lat1,lat2,symprec=0.1,is_reciprocal=True):
    if is_reciprocal:
        # convert to real space lattice
        lat1_real = np.linalg.pinv(np.array(lat1).T)
        lat2_real = np.linalg.pinv(np.array(lat2).T)
    else:
        lat1_real = np.array(lat1)
        lat2_real = np.array(lat2)
    s1 = struct_from_lat(lat1_real)
    s2 = struct_from_lat(lat2_real)

    try:
        prim1_lattice = SpacegroupAnalyzer(s1,symprec=symprec).get_primitive_standard_structure().lattice
    except Exception:
        prim1_lattice = s1.lattice
    try:
        prim2_lattice = SpacegroupAnalyzer(s2,symprec=symprec).get_primitive_standard_structure().lattice
    except Exception:
        prim2_lattice = s2.lattice

    prim1_lattice_niggli = prim1_lattice.get_niggli_reduced_lattice()
    prim2_lattice_niggli = prim2_lattice.get_niggli_reduced_lattice()
    return prim1_lattice_niggli, prim2_lattice_niggli

def check_prim_lattices_niggli(lat1,lat2,symprec=1e-2,tol=1e-5,is_reciprocal=True):
    if is_reciprocal:
        # convert to real space lattice
        lat1_real = np.linalg.pinv(np.array(lat1).T)
        lat2_real = np.linalg.pinv(np.array(lat2).T)
    else:
        lat1_real = np.array(lat1)
        lat2_real = np.array(lat2)
    s1 = struct_from_lat(lat1_real)
    s2 = struct_from_lat(lat2_real)

    try:
        prim1_lattice = SpacegroupAnalyzer(s1,symprec=symprec).get_primitive_standard_structure().lattice
    except Exception:
        prim1_lattice = s1.lattice
    try:
        prim2_lattice = SpacegroupAnalyzer(s2,symprec=symprec).get_primitive_standard_structure().lattice
    except Exception:
        prim2_lattice = s2.lattice

    prim1_lattice_niggli = prim1_lattice.get_niggli_reduced_lattice()
    prim2_lattice_niggli = prim2_lattice.get_niggli_reduced_lattice()
    res = np.allclose(prim1_lattice_niggli.matrix, prim2_lattice_niggli.matrix,atol=tol)
    return prim1_lattice_niggli.matrix, prim2_lattice_niggli.matrix, res
    

def get_lattice(lat,symprec=0.1):
    lat_real = np.linalg.pinv(np.array(lat).T)
    s = struct_from_lat(lat_real)
    try:
        return SpacegroupAnalyzer(s,symprec=symprec).get_primitive_standard_structure().lattice
    except Exception:
        return s.lattice
    return prim1_lattice_niggli, prim2_lattice_niggli

def cell_distance(true_metric,pred_metric):
    '''
    Computes the distance between two metric tensors.
    '''
    # true_metric and pred_metric are 3x3 matrices
    # we can use the Frobenius norm to compute the distance
    # between the two metric tensors
    return np.linalg.norm(true_metric - pred_metric)

def get_cctbx_candidates(true_lat, pred_lat,
                         max_delta=0.1,
                         max_delta_pred=None,
                         is_reciprocal=True):

    from cctbx import crystal
    from cctbx.sgtbx import lattice_symmetry

    if max_delta_pred is None:
        max_delta_pred = max_delta

    if is_reciprocal:
        true_lat_real = np.linalg.pinv(np.array(true_lat).T)
        pred_lat_real = np.linalg.pinv(np.array(pred_lat).T)
    else:
        true_lat_real = np.array(true_lat)
        pred_lat_real = np.array(pred_lat)

    s_true = struct_from_lat(true_lat_real)
    s_pred = struct_from_lat(pred_lat_real)

    # Niggli reduction
    lat_true = s_true.lattice.get_niggli_reduced_lattice()
    lat_pred = s_pred.lattice.get_niggli_reduced_lattice()

    # Supercell check: if pred volume ≈ n × true volume, check if pred is a genuine
    # integer supercell of true and reduce to the primitive sublattice if so.
    vol_ratio = lat_pred.volume / lat_true.volume
    n = round(vol_ratio)
    if 2 <= n <= 8 and abs(vol_ratio - n) < 0.05:
        M = lat_pred.matrix @ np.linalg.inv(lat_true.matrix)
        M_round = np.round(M).astype(int)
        if (np.abs(M - M_round).max() < 0.15 and
                abs(int(round(np.linalg.det(M_round))) - n) == 0):
            # pred IS a supercell of true — reduce to primitive sublattice
            # by applying M^{-1}: prim = M^{-1} @ pred = true (up to rotation)
            prim_matrix = np.linalg.inv(M_round) @ lat_pred.matrix
            from pymatgen.core import Lattice as PmgLattice
            lat_pred = PmgLattice(prim_matrix).get_niggli_reduced_lattice()

    symm_true = crystal.symmetry(
        unit_cell=lat_true.parameters,
        space_group_symbol="P1"
    )

    symm_pred = crystal.symmetry(
        unit_cell=lat_pred.parameters,
        space_group_symbol="P1"
    )

    true_ms = lattice_symmetry.metric_subgroups(
        symm_true,
        max_delta,
        bravais_types_only=True
    )
    true_ms_res = true_ms.result_groups

    pred_ms = lattice_symmetry.metric_subgroups(
        symm_pred,
        max_delta_pred,
        bravais_types_only=True
    )
    pred_ms_res = pred_ms.result_groups

    min_distance = float('inf')
    best_pair = (None, None)

    for i, true_entry in enumerate(true_ms_res):
        true_cell = true_entry['best_subsym'].unit_cell()
        true_G = np.array(true_cell.metrical_matrix())

        for j, pred_entry in enumerate(pred_ms_res):
            pred_cell = pred_entry['best_subsym'].unit_cell()
            pred_G = np.array(pred_cell.metrical_matrix())

            dist = cell_distance(true_G, pred_G)

            if dist < min_distance:
                min_distance = dist
                best_pair = (i, j)

    i, j = best_pair
    true_cell = true_ms_res[i]['best_subsym'].unit_cell()
    pred_cell = pred_ms_res[j]['best_subsym'].unit_cell()
    true_sg = true_ms_res[i]['best_subsym'].space_group_info().type().number()
    pred_sg = pred_ms_res[j]['best_subsym'].space_group_info().type().number()
    true_bravais = get_bravais_from_spg_number(true_sg)
    pred_bravais = get_bravais_from_spg_number(pred_sg)

    return (
        np.array(true_cell.parameters()),
        true_cell.volume(),
        true_bravais,
        np.array(pred_cell.parameters()),
        pred_cell.volume(),
        pred_bravais
    )

def run_gradient_descent_rot(max_iter, learned_sig, true_sig):
    '''
    Given an learned spatial signal and a true spatial signal, 
    runs gradient descent to find the rotation matrix to align them.
    '''
    curr_rot = o3.rand_quaternion(1)
    curr_rot.requires_grad = True
    opt = torch.optim.Adam([curr_rot],lr=1e-2)
    loss_fn = torch.nn.MSELoss()
    best_loss = 100
    for i in range(max_iter):
        rot_mat_irrep = sph.D_from_quaternion(curr_rot)[0,:,:]
        pred_sig_rot = rot_mat_irrep@learned_sig
        loss = loss_fn(true_sig,pred_sig_rot)
        if i % 100 == 0:
            print(loss)
        if loss.detach().numpy() < best_loss:
            best_loss = loss.detach().numpy()
            best_param = curr_rot.detach().clone()
        opt.zero_grad()
        loss.backward()
        opt.step()
    return best_loss,best_param,pred_sig_rot

def get_cctbx_true_aligned(true_lat, pred_lat, max_delta=0.1, is_reciprocal=True):
    """
    Run cctbx metric subgroup analysis on the true lattice only, then apply
    the same change-of-basis to the predicted lattice so both cells are expressed
    in the same Bravais-appropriate reference frame.
    """
    from cctbx import crystal
    from cctbx.sgtbx import lattice_symmetry

    if is_reciprocal:
        true_lat_real = np.linalg.pinv(np.array(true_lat).T)
        pred_lat_real = np.linalg.pinv(np.array(pred_lat).T)
    else:
        true_lat_real = np.array(true_lat)
        pred_lat_real = np.array(pred_lat)

    s_true = struct_from_lat(true_lat_real)
    s_pred = struct_from_lat(pred_lat_real)

    lat_true = s_true.lattice.get_niggli_reduced_lattice()
    lat_pred = s_pred.lattice.get_niggli_reduced_lattice()

    symm_true = crystal.symmetry(unit_cell=lat_true.parameters, space_group_symbol="P1")
    symm_pred = crystal.symmetry(unit_cell=lat_pred.parameters, space_group_symbol="P1")

    true_ms = lattice_symmetry.metric_subgroups(symm_true, max_delta, bravais_types_only=True)
    true_best = true_ms.result_groups[0]

    true_cell = true_best['best_subsym'].unit_cell()
    true_sg = true_best['best_subsym'].space_group_info().type().number()
    true_bravais = get_bravais_from_spg_number(true_sg)

    cb_op = true_best['cb_op_inp_best']
    pred_cell = symm_pred.change_basis(cb_op).unit_cell()

    return (
        np.array(true_cell.parameters()),
        true_cell.volume(),
        true_bravais,
        np.array(pred_cell.parameters()),
        pred_cell.volume(),
        true_bravais,
    )


def match_cell_setting(true_lat, pred_lat, length_tol=0.05, angle_tol=5.0):
    """
    Re-express pred_lat in the setting of the same lattice closest to true_lat, when
    some setting matches true_lat within tolerance but the given one does not.

    metric_subgroups offers no alternative settings for a triclinic cell, and two
    nearby lattices can reduce to different settings of the same cell (e.g. angles
    theta vs 180 - theta across the all-acute/all-obtuse Niggli boundary), so a
    correct prediction can otherwise be scored as a large angle error. Only the
    basis changes (unimodular matrix); the predicted lattice itself is unchanged.

    Returns (lattice, matched), where matched is True if the setting was changed.
    """
    from cctbx import uctbx
    try:
        t_cell = uctbx.unit_cell(true_lat.parameters)
        p_cell = uctbx.unit_cell(pred_lat.parameters)
        if t_cell.is_similar_to(p_cell, length_tol, angle_tol):
            return pred_lat, False
        mats = t_cell.similarity_transformations(p_cell, length_tol, angle_tol)
    except Exception:
        return pred_lat, False
    if len(mats) == 0:
        return pred_lat, False
    target = np.array(true_lat.parameters)
    candidates = [Lattice(np.array(m).reshape(3, 3).T @ pred_lat.matrix) for m in mats]
    best = min(candidates, key=lambda lat: np.abs(np.array(lat.parameters) - target).sum())
    return best, True


def niggli_reduce_with_bravais(true_lat, pred_lat, symprec=0.1, is_reciprocal=True):
    """Niggli-reduce both lattices and assign each a Bravais type with spglib. Used as the
    fallback when the cctbx search fails; returns (niggli_true, niggli_pred, true_bravais, pred_bravais)."""
    niggli_true, niggli_pred = convert_lats_to_niggli(true_lat, pred_lat, symprec=symprec, is_reciprocal=is_reciprocal)
    true_sg = SpacegroupAnalyzer(struct_from_lat(niggli_true.matrix), symprec=symprec).get_space_group_number()
    pred_sg = SpacegroupAnalyzer(struct_from_lat(niggli_pred.matrix), symprec=symprec).get_space_group_number()
    return niggli_true, niggli_pred, get_bravais_from_spg_number(true_sg), get_bravais_from_spg_number(pred_sg)


class space_group_to_bravais_lattice(Enum):
    sp230 = "cI"
    sp229 = "cI"
    sp228 = "cF"
    sp227 = "cF"
    sp226 = "cF"
    sp225 = "cF"
    sp224 = "cP"
    sp223 = "cP"
    sp222 = "cP"
    sp221 = "cP"
    sp220 = "cI"
    sp219 = "cF"
    sp218 = "cP"
    sp217 = "cI"
    sp216 = "cF"
    sp215 = "cP"
    sp214 = "cI"
    sp213 = "cP"
    sp212 = "cP"
    sp211 = "cI"
    sp210 = "cF"
    sp209 = "cF"
    sp208 = "cP"
    sp207 = "cP"
    sp206 = "cI"
    sp205 = "cP"
    sp204 = "cI"
    sp203 = "cF"
    sp202 = "cF"
    sp201 = "cP"
    sp200 = "cP"
    sp199 = "cI"
    sp198 = "cP"
    sp197 = "cI"
    sp196 = "cF"
    sp195 = "cP"
    sp194 = "hP"
    sp193 = "hP"
    sp192 = "hP"
    sp191 = "hP"
    sp190 = "hP"
    sp189 = "hP"
    sp188 = "hP"
    sp187 = "hP"
    sp186 = "hP"
    sp185 = "hP"
    sp184 = "hP"
    sp183 = "hP"
    sp182 = "hP"
    sp181 = "hP"
    sp180 = "hP"
    sp179 = "hP"
    sp178 = "hP"
    sp177 = "hP"
    sp176 = "hP"
    sp175 = "hP"
    sp174 = "hP"
    sp173 = "hP"
    sp172 = "hP"
    sp171 = "hP"
    sp170 = "hP"
    sp169 = "hP"
    sp168 = "hP"
    sp167 = "hR"
    sp166 = "hR"
    sp165 = "hP"
    sp164 = "hP"
    sp163 = "hP"
    sp162 = "hP"
    sp161 = "hR"
    sp160 = "hR"
    sp159 = "hP"
    sp158 = "hP"
    sp157 = "hP"
    sp156 = "hP"
    sp155 = "hR"
    sp154 = "hP"
    sp153 = "hP"
    sp152 = "hP"
    sp151 = "hP"
    sp150 = "hP"
    sp149 = "hP"
    sp148 = "hR"
    sp147 = "hP"
    sp146 = "hR"
    sp145 = "hP"
    sp144 = "hP"
    sp143 = "hP"
    sp142 = "tI"
    sp141 = "tI"
    sp140 = "tI"
    sp139 = "tI"
    sp138 = "tP"
    sp137 = "tP"
    sp136 = "tP"
    sp135 = "tP"
    sp134 = "tP"
    sp133 = "tP"
    sp132 = "tP"
    sp131 = "tP"
    sp130 = "tP"
    sp129 = "tP"
    sp128 = "tP"
    sp127 = "tP"
    sp126 = "tP"
    sp125 = "tP"
    sp124 = "tP"
    sp123 = "tP"
    sp122 = "tI"
    sp121 = "tI"
    sp120 = "tI"
    sp119 = "tI"
    sp118 = "tP"
    sp117 = "tP"
    sp116 = "tP"
    sp115 = "tP"
    sp114 = "tP"
    sp113 = "tP"
    sp112 = "tP"
    sp111 = "tP"
    sp110 = "tI"
    sp109 = "tI"
    sp108 = "tI"
    sp107 = "tI"
    sp106 = "tP"
    sp105 = "tP"
    sp104 = "tP"
    sp103 = "tP"
    sp102 = "tP"
    sp101 = "tP"
    sp100 = "tP"
    sp99 = "tP"
    sp98 = "tI"
    sp97 = "tI"
    sp96 = "tP"
    sp95 = "tP"
    sp94 = "tP"
    sp93 = "tP"
    sp92 = "tP"
    sp91 = "tP"
    sp90 = "tP"
    sp89 = "tP"
    sp88 = "tI"
    sp87 = "tI"
    sp86 = "tP"
    sp85 = "tP"
    sp84 = "tP"
    sp83 = "tP"
    sp82 = "tI"
    sp81 = "tP"
    sp80 = "tI"
    sp79 = "tI"
    sp78 = "tP"
    sp77 = "tP"
    sp76 = "tP"
    sp75 = "tP"
    sp74 = "oI"
    sp73 = "oI"
    sp72 = "oI"
    sp71 = "oI"
    sp70 = "oF"
    sp69 = "oF"
    sp68 = "oC"
    sp67 = "oC"
    sp66 = "oC"
    sp65 = "oC"
    sp64 = "oC"
    sp63 = "oC"
    sp62 = "oP"
    sp61 = "oP"
    sp60 = "oP"
    sp59 = "oP"
    sp58 = "oP"
    sp57 = "oP"
    sp56 = "oP"
    sp55 = "oP"
    sp54 = "oP"
    sp53 = "oP"
    sp52 = "oP"
    sp51 = "oP"
    sp50 = "oP"
    sp49 = "oP"
    sp48 = "oP"
    sp47 = "oP"
    sp46 = "oI"
    sp45 = "oI"
    sp44 = "oI"
    sp43 = "oF"
    sp42 = "oF"
    sp41 = "oC"
    sp40 = "oC"
    sp39 = "oC"
    sp38 = "oC"
    sp37 = "oC"
    sp36 = "oC"
    sp35 = "oC"
    sp34 = "oP"
    sp33 = "oP"
    sp32 = "oP"
    sp31 = "oP"
    sp30 = "oP"
    sp29 = "oP"
    sp28 = "oP"
    sp27 = "oP"
    sp26 = "oP"
    sp25 = "oP"
    sp24 = "oI"
    sp23 = "oI"
    sp22 = "oF"
    sp21 = "oC"
    sp20 = "oC"
    sp19 = "oP"
    sp18 = "oP"
    sp17 = "oP"
    sp16 = "oP"
    sp15 = "mC" 
    sp14 = "mP" 
    sp13 = "mP" 
    sp12 = "mC" 
    sp11 = "mP" 
    sp10 = "mP" 
    sp9 = "mC" 
    sp8 = "mC" 
    sp7 = "mP" 
    sp6 = "mP" 
    sp5 = "mC" 
    sp4 = "mP" 
    sp3 = "mP" 
    sp2 = "aP" 
    sp1 = "aP"

def get_bravais_from_spg_number(spg_number: int) -> str:
    try:
        return getattr(space_group_to_bravais_lattice, f"sp{spg_number}").value
    except AttributeError:
        raise ValueError(f"Space group number {spg_number} not found in mapping.")

### Note the functions below are from David's repository https://github.com/dwmoreau/MLI/blob/main/Reindexing.py ###

def get_s6_from_unit_cell(unit_cell):
    a = unit_cell[:, 0]
    b = unit_cell[:, 1]
    c = unit_cell[:, 2]
    alpha = unit_cell[:, 3]
    beta = unit_cell[:, 4]
    gamma = unit_cell[:, 5]

    ax = a
    bx = b*np.cos(gamma)
    by = b*np.sin(gamma)
    cx = c*np.cos(beta)
    arg = (np.cos(alpha) - np.cos(beta)*np.cos(gamma)) / np.sin(gamma)
    cy = c * arg
    cz = c * np.sqrt(np.sin(beta)**2 - arg**2)
    z = np.zeros(unit_cell.shape[0])
    om = np.array([
        [ax, bx, cx],
        [z,  by, cy],
        [z,  z,  cz]
        ])
    om = np.moveaxis(om, [0, 1, 2], [1, 2, 0])
    d = -np.sum(om, axis=2)

    s6 = np.column_stack((
        np.sum(om[:, :, 1] * om[:, :, 2], axis=1),
        np.sum(om[:, :, 0] * om[:, :, 2], axis=1),
        np.sum(om[:, :, 0] * om[:, :, 1], axis=1),
        np.sum(om[:, :, 0] * d, axis=1),
        np.sum(om[:, :, 1] * d, axis=1),
        np.sum(om[:, :, 2] * d, axis=1),
        ))
    return s6


def get_unit_cell_from_s6_with_warnings(s6):
    a = np.sqrt(-(s6[:, 3] + s6[:, 1] + s6[:, 2]))
    b = np.sqrt(-(s6[:, 4] + s6[:, 0] + s6[:, 2]))
    c = np.sqrt(-(s6[:, 5] + s6[:, 0] + s6[:, 1]))
    alpha = np.arccos(s6[:, 0] / (b*c))
    beta = np.arccos(s6[:, 1] / (a*c))
    gamma = np.arccos(s6[:, 2] / (a*b))

    unit_cell = np.column_stack((a, b, c, alpha, beta, gamma))
    return unit_cell


def get_unit_cell_from_s6(s6):
    a2 = -(s6[:, 3] + s6[:, 1] + s6[:, 2])
    b2 = -(s6[:, 4] + s6[:, 0] + s6[:, 2])
    c2 = -(s6[:, 5] + s6[:, 0] + s6[:, 1])
    nonphysical_lengths = np.any(np.stack((a2, b2, c2), axis=1) <= 0, axis=1)
    if np.count_nonzero(nonphysical_lengths) == 0:
        a = np.sqrt(a2)
        b = np.sqrt(b2)
        c = np.sqrt(c2)
        alpha_arg = s6[:, 0] / (b*c)
        beta_arg = s6[:, 1] / (a*c)
        gamma_arg = s6[:, 2] / (a*b)
        nonphysical_angles = np.any(
            np.abs(np.stack((alpha_arg, beta_arg, gamma_arg), axis=1)) > 1,
            axis=1
        )
    else:
        physical_lengths = np.invert(nonphysical_lengths)
        a = np.zeros(s6.shape[0])
        b = np.zeros(s6.shape[0])
        c = np.zeros(s6.shape[0])

        a[nonphysical_lengths] = np.nan
        b[nonphysical_lengths] = np.nan
        c[nonphysical_lengths] = np.nan

        a[physical_lengths] = np.sqrt(a2[physical_lengths])
        b[physical_lengths] = np.sqrt(b2[physical_lengths])
        c[physical_lengths] = np.sqrt(c2[physical_lengths])

        alpha_arg = np.zeros(s6.shape[0])
        beta_arg = np.zeros(s6.shape[0])
        gamma_arg = np.zeros(s6.shape[0])

        alpha_arg[nonphysical_lengths] = np.nan
        beta_arg[nonphysical_lengths] = np.nan
        gamma_arg[nonphysical_lengths] = np.nan

        alpha_arg[physical_lengths] = s6[physical_lengths, 0] / (b*c)[physical_lengths]
        beta_arg[physical_lengths] = s6[physical_lengths, 1] / (a*c)[physical_lengths]
        gamma_arg[physical_lengths] = s6[physical_lengths, 2] / (a*b)[physical_lengths]

        nonphysical_angles = nonphysical_lengths

        nonphysical_angles[physical_lengths] = np.any(np.abs(np.stack((
            alpha_arg[physical_lengths], 
            beta_arg[physical_lengths], 
            gamma_arg[physical_lengths]
            ), axis=1)) > 1, axis=1)

    if np.count_nonzero(nonphysical_angles) == 0:
        alpha = np.arccos(alpha_arg)
        beta = np.arccos(beta_arg)
        gamma = np.arccos(gamma_arg)
    else:
        physical_angles = np.invert(nonphysical_angles)
        alpha = np.zeros(s6.shape[0])
        beta = np.zeros(s6.shape[0])
        gamma = np.zeros(s6.shape[0])

        alpha[nonphysical_angles] = np.nan
        beta[nonphysical_angles] = np.nan
        gamma[nonphysical_angles] = np.nan

        alpha[physical_angles] = np.arccos(alpha_arg[physical_angles])
        beta[physical_angles] = np.arccos(beta_arg[physical_angles])
        gamma[physical_angles] = np.arccos(gamma_arg[physical_angles])

    unit_cell = np.column_stack((a, b, c, alpha, beta, gamma))
    return unit_cell


def selling_reduction(unit_cell, space='direct'):
    assert space == 'direct'

    reduction_op_bc = np.array([
        [-1, 0, 0, 0, 0, 0], 
        [1, 1, 0, 0, 0, 0],
        [1, 0, 0, 0, 1, 0],
        [-1, 0, 0, 1, 0, 0],
        [1, 0, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 1],
        ])
    reduction_op_bc_hkl = np.array([
        [1, 0, 0],
        [1, -1, 0],
        [0, 0, 1],
        ])

    reduction_op_ac = np.array([
        [1, 1, 0, 0, 0, 0],
        [0, -1, 0, 0, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, -1, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 1],
        ])
    reduction_op_ac_hkl = np.array([
        [1, -1, 0],
        [0, -1, 0],
        [0, 0, -1],
        ])

    reduction_op_ab = np.array([
        [1, 0, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 0, -1, 0, 0, 0],
        [0, 1, 1, 0, 0, 0],
        [0, 0, 1, 0, 1, 0],
        [0, 0, -1, 0, 0, 1],
        ])
    reduction_op_ab_hkl = np.array([
        [1, 0, -1],
        [0, -1, 0],
        [0, 0, -1],
        ])

    reduction_op_ad = np.array([
        [1, 0, 0, -1, 0, 0],
        [0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 0],
        [0, 0, 0, -1, 0, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 1, 0, 1],
        ])
    reduction_op_ad_hkl = np.array([
        [1, -1, -1],
        [0, -1, 0],
        [0, 0, -1],
        ])

    reduction_op_bd = np.array([
        [0, 0, 1, 0, 1, 0],
        [0, 1, 0, 0, -1, 0],
        [1, 0, 0, 0, 1, 0],
        [0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, -1, 0],
        [0, 0, 0, 0, 1, 1],
        ])
    reduction_op_bd_hkl = np.array([
        [-1, 0, 0],
        [-1, 1, -1],
        [0, 0, -1],
        ])

    reduction_op_cd = np.array([
        [0, 1, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 1],
        [0, 0, 1, 0, 0, -1],
        [0, 0, 0, 1, 0, 1],
        [0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, -1],
        ])
    reduction_op_cd_hkl = np.array([
        [-1, 0, 0],
        [0, -1, 0],
        [-1, -1, 1],
        ])

    reduction_ops = [
        reduction_op_bc,
        reduction_op_ac,
        reduction_op_ab,
        reduction_op_ad,
        reduction_op_bd,
        reduction_op_cd
        ]
    reduction_ops_hkl = [
        reduction_op_bc_hkl,
        reduction_op_ac_hkl,
        reduction_op_ab_hkl,
        reduction_op_ad_hkl,
        reduction_op_bd_hkl,
        reduction_op_cd_hkl
        ]

    reflections = np.stack([
        np.eye(6),
        np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            ]),
        np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            ]),
        np.array([
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            ]),
        np.array([
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            ]),
        np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            ]),
        np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            ]),
        np.array([
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            ]),
        np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            ]),
        np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            ]),
        np.array([
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            ]),
        np.array([
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            ]),
        np.array([
            [0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            ]),
        np.array([
            [0, 0, 0, 0, 0, 1],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            ]),
        np.array([
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            ]),
        np.array([
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            ]),
        ],
        axis=0
        )

    reflection_op_abc = reflections[0]
    reflection_op_acb = reflections[1]
    reflection_op_bac = reflections[4]
    reflection_op_bca = reflections[5]
    reflection_op_cab = reflections[8]
    reflection_op_cba = reflections[9]

    reflection_op_abc_hkl = np.eye(3) #012
    reflection_op_acb_hkl = np.array([ #021
        [1, 0, 0],
        [0, 0, 1],
        [0, 1, 0],
        ])
    reflection_op_bac_hkl = np.array([ #102
        [0, 1, 0],
        [1, 0, 0],
        [0, 0, 1],
        ])
    reflection_op_bca_hkl = np.array([ #120
        [0, 0, 1],
        [1, 0, 0],
        [0, 1, 0],
        ])
    reflection_op_cab_hkl = np.array([ #201
        [0, 1, 0],
        [0, 0, 1],
        [1, 0, 0],
        ])
    reflection_op_cba_hkl = np.array([ #210
        [0, 0, 1],
        [0, 1, 0],
        [1, 0, 0],
        ])

    s6 = get_s6_from_unit_cell(unit_cell)
    s6_reduced = s6.copy()
    hkl_transformation = np.repeat(np.eye(3)[np.newaxis], unit_cell.shape[0], axis=0)
    for iteration in range(20):
        s6_reduced_next = s6_reduced.copy()
        hkl_transformation_next = hkl_transformation.copy()
        s6_max_index = np.argmax(s6_reduced, axis=1)
        s6_max = np.take_along_axis(s6_reduced, s6_max_index[:, np.newaxis], axis=1)[:, 0]
        for axis_index in range(6):
            indices = np.logical_and(
                s6_max_index == axis_index,
                s6_max > 0
                )
            if indices.sum() > 0:
                s6_reduced_next[indices] = np.matmul(
                    reduction_ops[axis_index],
                    s6_reduced[indices][:, :, np.newaxis]
                    )[:, :, 0]
                hkl_transformation_next[indices] = np.matmul(
                    hkl_transformation[indices],
                    reduction_ops_hkl[axis_index][np.newaxis]
                    )
        indices = s6_reduced_next.sum(axis=1) >= s6_reduced.sum(axis=1)
        s6_reduced[indices] = s6_reduced_next[indices]
        hkl_transformation[indices] =  hkl_transformation_next[indices]

    # There is a numerical warning for invalid value encountered in sqrt.
    # Removing the square root should work and not produce the warning.
    order = np.argsort(np.column_stack((
        np.sqrt(-(s6_reduced[:, 3] + s6_reduced[:, 1] + s6_reduced[:, 2])),
        np.sqrt(-(s6_reduced[:, 4] + s6_reduced[:, 0] + s6_reduced[:, 2])),
        np.sqrt(-(s6_reduced[:, 5] + s6_reduced[:, 0] + s6_reduced[:, 1]))
        )), axis=1)
    #order = np.argsort(np.column_stack((
    #    -(s6_reduced[:, 3] + s6_reduced[:, 1] + s6_reduced[:, 2]),
    #    -(s6_reduced[:, 4] + s6_reduced[:, 0] + s6_reduced[:, 2]),
    #    -(s6_reduced[:, 5] + s6_reduced[:, 0] + s6_reduced[:, 1])
    #    )), axis=1)

    abc = np.all(order == np.array([0, 1, 2]), axis=1)
    acb = np.all(order == np.array([0, 2, 1]), axis=1)
    bac = np.all(order == np.array([1, 0, 2]), axis=1)
    bca = np.all(order == np.array([1, 2, 0]), axis=1)
    cab = np.all(order == np.array([2, 0, 1]), axis=1)
    cba = np.all(order == np.array([2, 1, 0]), axis=1)

    s6_reduced[abc] = (reflection_op_abc @ s6_reduced[abc][:, :, np.newaxis])[:, :, 0]
    s6_reduced[acb] = (reflection_op_acb @ s6_reduced[acb][:, :, np.newaxis])[:, :, 0]
    s6_reduced[bac] = (reflection_op_bac @ s6_reduced[bac][:, :, np.newaxis])[:, :, 0]
    s6_reduced[bca] = (reflection_op_bca @ s6_reduced[bca][:, :, np.newaxis])[:, :, 0]
    s6_reduced[cab] = (reflection_op_cab @ s6_reduced[cab][:, :, np.newaxis])[:, :, 0]
    s6_reduced[cba] = (reflection_op_cba @ s6_reduced[cba][:, :, np.newaxis])[:, :, 0]

    hkl_transformation[abc] = hkl_transformation[abc] @ reflection_op_abc_hkl[np.newaxis]
    hkl_transformation[acb] = hkl_transformation[acb] @ reflection_op_acb_hkl[np.newaxis]
    hkl_transformation[bac] = hkl_transformation[bac] @ reflection_op_bac_hkl[np.newaxis]
    hkl_transformation[bca] = hkl_transformation[bca] @ reflection_op_bca_hkl[np.newaxis]
    hkl_transformation[cab] = hkl_transformation[cab] @ reflection_op_cab_hkl[np.newaxis]
    hkl_transformation[cba] = hkl_transformation[cba] @ reflection_op_cba_hkl[np.newaxis]

    unit_cell_reduced = get_unit_cell_from_s6(s6_reduced)
    return unit_cell_reduced, hkl_transformation, s6_reduced


# ---------------------------------------------------------------------------
# XRD simulation (pseudo-Voigt broadening)
# ---------------------------------------------------------------------------

def caglioti_fwhm(theta, U, V, W):
    rad_theta = np.radians(theta / 2)
    arg = U * np.tan(rad_theta)**2 + V * np.tan(rad_theta) + W
    return np.clip(arg, 1e-12, None)**0.5

def _pseudo_voigt(x, center, amplitude, U, V, W, eta=0.0):
    fwhm = caglioti_fwhm(center, U, V, W)
    sigma = fwhm / (2 * np.sqrt(2 * np.log(2)))
    lorentzian = amplitude * (fwhm**2 / ((x - center)**2 + fwhm**2))
    gaussian = amplitude * np.exp(-(x - center)**2 / (2 * sigma**2))
    return eta * lorentzian + (1 - eta) * gaussian

def get_sim_xrd_from_pattern(pattern, U=0.05, V=-0.06, W=0.07):
    """Simulate a pseudo-Voigt XRD pattern from a pymatgen XRDPattern object."""
    x = np.arange(5, 90, 0.010)
    total = np.zeros_like(x)
    for center, amplitude in zip(pattern.x, pattern.y):
        total += _pseudo_voigt(x, center, amplitude, U, V, W)
    total = total / max(total)
    return total

def get_sim_xrd_from_preprocessed_results(preprocessed_results, index, isIndex=True, returnPattern=False):
    """Simulate XRD from a list of dicts with 'crystal' key, or directly from a Structure."""
    structure = preprocessed_results[index]['crystal'] if isIndex else preprocessed_results
    pattern = _xrd_calculator.get_pattern(structure)
    if returnPattern:
        return pattern
    return get_sim_xrd_from_pattern(pattern)

def save_results_pickle(results, filename):
    with open(filename, 'wb') as f:
        pickle.dump(results, f)

def seed_from_mpid(mpid: str, base_seed: int = 0) -> int:
    '''Deterministic per-material RNG seed, stable across runs/workers.'''
    h = hashlib.blake2b(mpid.encode(), digest_size=8).digest()
    return base_seed + int.from_bytes(h, "little")


# ---------------------------------------------------------------------------
# Bispectrum data generation
# ---------------------------------------------------------------------------

def training_data_gen(preprocessed_results, isIndex=True, k_max=2/3, l_max=6, cob_bi=None):
    """Compute bispectrum for a list of structures (isIndex=True) or a single Structure."""
    if cob_bi is None:
        import e3nn_jax
        rtp_bi = e3nn_jax.reduced_symmetric_tensor_product_basis(
            io.SphericalTensor(l_max, 1, -1), 3, keep_ir=['0o', '0e']
        )
        cob_bi = torch.tensor(rtp_bi.array, dtype=torch.float64)

    if isIndex:
        bispec_list = []
        for item in preprocessed_results:
            si_prim = item['crystal'].get_primitive_structure()
            recip_lattice = si_prim.lattice.reciprocal_lattice_crystallographic
            miller_ind = get_neighbors(recip_lattice.matrix, float(k_max))
            neighs = torch.tensor(miller_ind.reshape(-1, 3) @ recip_lattice.matrix)
            bispec_list.append(get_bispectrum(neighs, float(k_max), l_max, 'bessel', 10, cob_bi))
        return bispec_list
    else:
        si_prim = preprocessed_results
        recip_lattice = si_prim.lattice.reciprocal_lattice_crystallographic
        miller_ind = get_neighbors(recip_lattice.matrix, float(k_max))
        neighs = torch.tensor(miller_ind.reshape(-1, 3) @ recip_lattice.matrix)
        return get_bispectrum(neighs, float(k_max), l_max, 'bessel', 10, cob_bi)
