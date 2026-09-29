import numpy as np
import pytest
import torch
from pymatgen.core.structure import Lattice, Structure

import utilities as u

# NOTE: this suite intentionally covers only the utilities.py functions that
# training/ and bispectrum/ actually call (verified via repo-wide grep). A large
# block of utilities.py -- gen_lattice, lattice_param_to_vertices,
# lat_from_parameters, convert_lats_to_prim/_selling, get_lattice,
# get_s6_from_unit_cell, get_unit_cell_from_s6(_with_warnings),
# selling_reduction -- has no callers anywhere in the repo (not even notebooks)
# and is deliberately left untested. The cctbx comparison functions are
# covered in test_cctbx.py.


def _cob_bi(l_max):
    """Build the change-of-basis tensor get_bispectrum/get_true_sig need,
    the same way training_data_gen does internally when cob_bi=None."""
    import e3nn_jax
    from e3nn import io
    rtp_bi = e3nn_jax.reduced_symmetric_tensor_product_basis(
        io.SphericalTensor(l_max, 1, -1), 3, keep_ir=["0o", "0e"]
    )
    return torch.tensor(rtp_bi.array, dtype=torch.float64)


# ---------------------------------------------------------------------------
# struct_from_lat
# ---------------------------------------------------------------------------

def test_struct_from_lat_places_single_atom_at_origin():
    lat_mat = np.eye(3) * 5.0
    struct = u.struct_from_lat(lat_mat)
    assert len(struct) == 1
    assert struct.species[0].symbol == "C"
    np.testing.assert_allclose(struct.lattice.matrix, lat_mat)
    np.testing.assert_allclose(struct[0].frac_coords, [0, 0, 0])


# ---------------------------------------------------------------------------
# Primitive / Niggli lattice comparisons (used by run_alg_inversion*, plot_bispec_vs_lat_error)
# ---------------------------------------------------------------------------

def test_check_prim_lattices_identical_lattice_is_true():
    lat = Lattice.cubic(4.0).reciprocal_lattice_crystallographic.matrix
    assert u.check_prim_lattices(lat, lat) is True


def test_check_prim_lattices_different_volumes_is_false():
    lat1 = Lattice.cubic(4.0).reciprocal_lattice_crystallographic.matrix
    lat2 = Lattice.cubic(8.0).reciprocal_lattice_crystallographic.matrix
    assert u.check_prim_lattices(lat1, lat2) is False


def test_check_prim_lattices_niggli_identical_lattice_is_true():
    lat = Lattice.cubic(4.0).reciprocal_lattice_crystallographic.matrix
    _, _, res = u.check_prim_lattices_niggli(lat, lat)
    assert res is True


def test_convert_lats_to_niggli_identical_lattice_matches():
    lat = Lattice.cubic(4.0).reciprocal_lattice_crystallographic.matrix
    niggli1, niggli2 = u.convert_lats_to_niggli(lat, lat)
    np.testing.assert_allclose(niggli1.matrix, niggli2.matrix, atol=1e-8)


# ---------------------------------------------------------------------------
# space_group_to_bravais_lattice enum
# ---------------------------------------------------------------------------

def test_space_group_enum_covers_all_230_groups():
    # Many space groups share a Bravais symbol, so Python's Enum treats later
    # same-valued members as aliases of the first; __members__ still exposes
    # every sp1..sp230 name via alias resolution (unlike iterating the enum).
    names = set(u.space_group_to_bravais_lattice.__members__.keys())
    expected = {f"sp{i}" for i in range(1, 231)}
    assert names == expected


@pytest.mark.parametrize(
    "sg,expected",
    [("sp1", "aP"), ("sp225", "cF"), ("sp221", "cP"), ("sp230", "cI"), ("sp194", "hP")],
)
def test_space_group_enum_known_values(sg, expected):
    assert u.space_group_to_bravais_lattice[sg].value == expected


# ---------------------------------------------------------------------------
# XRD pseudo-Voigt simulation
# ---------------------------------------------------------------------------

class _FakePattern:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_get_sim_xrd_from_pattern_is_normalized_and_peaked_at_center():
    pattern = _FakePattern(x=[30.0], y=[100.0])
    sim = u.get_sim_xrd_from_pattern(pattern)
    x_axis = np.arange(5, 90, 0.010)
    assert sim.max() == pytest.approx(1.0)
    assert x_axis[np.argmax(sim)] == pytest.approx(30.0, abs=0.01)


# ---------------------------------------------------------------------------
# get_neighbors (reciprocal-lattice neighbor search)
# ---------------------------------------------------------------------------

def test_get_neighbors_returns_miller_indices_within_threshold():
    recip = Lattice.cubic(1.0).reciprocal_lattice_crystallographic.matrix
    neighs = u.get_neighbors(recip, k_max=3.0, miller_ind_threshold=2)
    assert neighs.shape[1] == 3
    assert len(neighs) > 0
    assert np.all(np.abs(neighs) <= 2)


# ---------------------------------------------------------------------------
# get_bispectrum / get_true_sig (core of the bispectrum inversion pipeline
# used throughout bispectrum/run_alg_inversion*.py)
# ---------------------------------------------------------------------------

def test_get_bispectrum_is_rotation_invariant():
    # The bispectrum is built by contracting spherical harmonics with
    # rotation-invariant Clebsch-Gordan-like coefficients (cob_bi), so a
    # global rotation of the input vectors must leave it unchanged. This is
    # the defining correctness property the whole inversion pipeline (fitting
    # lattice params to match a target bispectrum) depends on.
    torch.manual_seed(0)
    vec = torch.randn(20, 3, dtype=torch.float64)
    cob_bi = _cob_bi(l_max=2)

    bispec = u.get_bispectrum(vec, k_max=3.0, lmax=2, radial_function="bessel", n_basis=4, cob_bi=cob_bi)

    from scipy.spatial.transform import Rotation
    rot = torch.tensor(Rotation.random(random_state=0).as_matrix(), dtype=torch.float64)
    rotated_bispec = u.get_bispectrum(vec @ rot.T, k_max=3.0, lmax=2, radial_function="bessel", n_basis=4, cob_bi=cob_bi)

    torch.testing.assert_close(bispec, rotated_bispec, atol=1e-6, rtol=1e-6)


def test_get_true_sig_returns_finite_tensor():
    recip_lattice = Lattice.cubic(4.0).reciprocal_lattice_crystallographic.matrix
    cob_bi = _cob_bi(l_max=2)
    true_bispec = u.get_true_sig(
        recip_lattice, wavelength=1.5, radial_function="bessel", n_basis=4, lmax=2, cob_bi=cob_bi
    )
    assert torch.isfinite(true_bispec).all()
    assert true_bispec.numel() > 0


def test_training_data_gen_single_structure():
    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
    bispec = u.training_data_gen(structure, isIndex=False, k_max=2 / 3, l_max=2)
    assert torch.isfinite(bispec).all()
    assert bispec.numel() > 0


# ---------------------------------------------------------------------------
# XRD simulation from a real Structure / pickle round trip
# ---------------------------------------------------------------------------

def test_get_sim_xrd_from_preprocessed_results_matches_pattern_path():
    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
    pattern = u.get_sim_xrd_from_preprocessed_results(structure, index=0, isIndex=False, returnPattern=True)
    direct = u.get_sim_xrd_from_pattern(pattern)
    via_structure = u.get_sim_xrd_from_preprocessed_results(structure, index=0, isIndex=False)
    np.testing.assert_allclose(via_structure, direct)


def test_save_results_pickle_round_trips(tmp_path):
    results = {"a": 1, "b": [1, 2, 3]}
    out_file = tmp_path / "results.pkl"
    u.save_results_pickle(results, str(out_file))

    import pickle
    with open(out_file, "rb") as f:
        loaded = pickle.load(f)
    assert loaded == results
