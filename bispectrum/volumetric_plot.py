"""
Generates the paper's volumetric bispectrum-signal figure: a 3D isosurface
rendering (plotly go.Volume) of the reconstructed real-space signal at a
cubic lattice's reciprocal-space neighbor points.

Usage:
    python volumetric_plot.py [--a 3.0] [--output volumetric_plot.html]
"""
import argparse
import os
import sys

import numpy as np
import torch
import e3nn
import e3nn_jax
from e3nn import o3, io
import plotly.graph_objects as go
from pymatgen.core import Lattice

torch.set_default_dtype(torch.float64)

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utilities import get_neighbors

WAVELENGTH = 3
RADIAL_FUNCTION = "bessel"
N_BASIS = 10
LMAX = 6


def get_bispectrum(vec, k_max, lmax, radial_function, n_basis, cob_bi, ret_dirac=False):
    sph = io.SphericalTensor(lmax, 1, -1)
    sph_harm = o3.spherical_harmonics(range(0, lmax + 1), vec, normalize="False")
    radii = torch.linalg.norm(vec, axis=-1)
    basis = e3nn.math.soft_one_hot_linspace(radii, 0, k_max, number=n_basis,
                                            basis=radial_function, cutoff=False)
    sph_harm = sph_harm.unsqueeze(1)
    basis = basis.unsqueeze(2)
    radial_proj = torch.sum(torch.matmul(basis, sph_harm), axis=0)
    bispec = torch.einsum("...i,...j,...k,ijkz->...z", radial_proj, radial_proj, radial_proj, cob_bi)
    if ret_dirac:
        dirac_proj = sph.sum_of_diracs(vec, values=vec.norm(2, -1))
        return bispec, dirac_proj, radial_proj, basis
    return bispec


def get_true_sig(recip_lattice, wavelength, radial_function, n_basis, lmax, cob_bi):
    k_max = 2 / wavelength
    miller_ind = get_neighbors(recip_lattice, k_max, True, False, 3, False, 4)
    neighs = torch.tensor(miller_ind.reshape(-1, 3) @ recip_lattice)
    return get_bispectrum(neighs, k_max, lmax, radial_function, n_basis, cob_bi, ret_dirac=True)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--a', type=float, default=3.0, help='Cubic lattice constant (Angstrom).')
    p.add_argument('--num-vals', type=int, default=50, help='Grid resolution per axis.')
    p.add_argument('--isomax', type=float, default=1000, help='go.Volume isomax (upper isosurface value).')
    p.add_argument('--output', default='volumetric_plot.html')
    args = p.parse_args()

    k_max = 2 / WAVELENGTH
    recip_lattice = Lattice.cubic(args.a).reciprocal_lattice_crystallographic.matrix

    miller_inds = get_neighbors(recip_lattice, k_max, False, False, 3, False, 4)
    true_neighs = torch.tensor(miller_inds.reshape(-1, 3) @ recip_lattice)

    sph = io.SphericalTensor(LMAX, 1, -1)
    rtp_bi = e3nn_jax.reduced_symmetric_tensor_product_basis(sph, 3, keep_ir=['0o', '0e'])
    cob_bi = torch.tensor(rtp_bi.array, dtype=torch.float64)

    _, _, radial_proj, _ = get_true_sig(recip_lattice, WAVELENGTH, RADIAL_FUNCTION, N_BASIS, LMAX, cob_bi)

    # Evaluate the reconstructed signal on a fractional-coordinate grid, mapped
    # to Cartesian via the reciprocal lattice.
    grid_vals = np.linspace(-1, 1, args.num_vals)
    U, V, W = np.meshgrid(grid_vals, grid_vals, grid_vals, indexing="ij")
    uvw_flat = np.stack([U, V, W], axis=-1).reshape(-1, 3)
    coords_flat = uvw_flat @ recip_lattice
    coords_grid = coords_flat.reshape(args.num_vals, args.num_vals, args.num_vals, 3)
    X, Y, Z = coords_grid[..., 0], coords_grid[..., 1], coords_grid[..., 2]
    vec_grid = torch.from_numpy(coords_grid)

    sph_harm = o3.spherical_harmonics(range(0, LMAX + 1), vec_grid, normalize="False")
    radii = torch.linalg.norm(vec_grid, axis=-1)
    basis = e3nn.math.soft_one_hot_linspace(radii, 0, k_max, number=N_BASIS,
                                            basis=RADIAL_FUNCTION, cutoff=True)
    vals = torch.einsum('ij,xyzj,xyzi->xyz', radial_proj, sph_harm, basis)

    vol_plot = go.Volume(
        x=X.flatten(), y=Y.flatten(), z=Z.flatten(), value=vals.flatten().numpy(),
        isomin=0, isomax=args.isomax, colorscale='blues', opacity=0.2, surface_count=20,
        colorbar=dict(title="Intensity", thickness=20, len=0.5, x=0.5, y=-0.15,
                      orientation="h", tickfont=dict(size=14), titlefont=dict(size=16)),
    )
    neighs_plot = go.Scatter3d(
        x=true_neighs[:, 0], y=true_neighs[:, 1], z=true_neighs[:, 2],
        mode='markers', marker=dict(size=8, color='grey'), showlegend=False,
    )
    fig = go.Figure(data=[vol_plot, neighs_plot])
    fig.update_layout(
        scene=dict(
            xaxis=dict(title='k_x', title_font=dict(size=20), tickfont=dict(size=14),
                      showgrid=False, showline=False, dtick=0.25),
            yaxis=dict(title='k_y', title_font=dict(size=20), tickfont=dict(size=14),
                      showgrid=False, showline=False, dtick=0.25),
            zaxis=dict(title='k_z', title_font=dict(size=20), tickfont=dict(size=14),
                      showgrid=False, showline=False, dtick=0.25),
        ),
        title=dict(text="Signal at Lattice Points (Reciprocal Space)", font=dict(size=24),
                  x=0.5, y=0.9, xanchor='center', yanchor='top'),
    )
    fig.write_html(args.output)
    print(f'saved {args.output}')


if __name__ == '__main__':
    main()
