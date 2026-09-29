import numpy as np
import pytest
import torch
from pymatgen.core import Lattice

import run_alg_inversion as inv

K_MAX, LMAX, N_BASIS, RADIAL = 3.0, 2, 4, "bessel"


def _cob_bi(l_max):
    import e3nn_jax
    from e3nn import io
    rtp_bi = e3nn_jax.reduced_symmetric_tensor_product_basis(
        io.SphericalTensor(l_max, 1, -1), 3, keep_ir=["0o", "0e"]
    )
    return torch.tensor(rtp_bi.array, dtype=torch.float64)


@pytest.fixture(autouse=True)
def cob_bases():
    """signal_from_params reads the module-level COB_BASES global, which
    init_globals() normally populates from a pickled file; set it directly
    here so tests don't need real cob/HDF5 data on disk."""
    inv.COB_BASES = _cob_bi(LMAX)
    yield
    inv.COB_BASES = None


def _random_recip_matrix(seed):
    rng = np.random.default_rng(seed)
    a, b, c = rng.uniform(3, 8, size=3)
    alpha, beta, gamma = rng.uniform(70, 110, size=3)
    lat = Lattice.from_parameters(a, b, c, alpha, beta, gamma)
    return lat.reciprocal_lattice_crystallographic.matrix


class TestGetTrueSig:
    def test_returns_finite_bispectrum(self):
        recip = _random_recip_matrix(0)
        sig = inv.get_true_sig(recip, wavelength=1.5, radial_function=RADIAL,
                                n_basis=N_BASIS, lmax=LMAX, cob_bases=inv.COB_BASES)
        assert torch.isfinite(sig).all()
        assert sig.numel() > 0


class TestObjectiveAndGradient:
    def test_objective_is_zero_for_matching_lattice(self):
        recip = _random_recip_matrix(1)
        obs = inv.signal_from_params(recip, K_MAX, LMAX, RADIAL, N_BASIS)
        loss = inv.objective_function(recip.flatten(), obs, K_MAX, LMAX, RADIAL, N_BASIS)
        assert loss == pytest.approx(0.0, abs=1e-8)

    def test_objective_is_positive_for_different_lattice(self):
        obs = inv.signal_from_params(_random_recip_matrix(2), K_MAX, LMAX, RADIAL, N_BASIS)
        loss = inv.objective_function(_random_recip_matrix(3).flatten(), obs, K_MAX, LMAX, RADIAL, N_BASIS)
        assert loss > 0

    def test_gradient_matches_finite_differences(self):
        # This is the analytic gradient scipy.optimize uses as `jac` during
        # inversion -- if it disagreed with the true gradient of
        # objective_function, L-BFGS-B would silently fail to converge
        # without any obvious error.
        obs = inv.signal_from_params(_random_recip_matrix(4), K_MAX, LMAX, RADIAL, N_BASIS)
        params = _random_recip_matrix(5).flatten()

        analytic = inv.gradient_function(params, obs, K_MAX, LMAX, RADIAL, N_BASIS)

        eps = 1e-6
        numeric = np.zeros_like(params)
        for i in range(len(params)):
            p_plus, p_minus = params.copy(), params.copy()
            p_plus[i] += eps
            p_minus[i] -= eps
            f_plus = inv.objective_function(p_plus, obs, K_MAX, LMAX, RADIAL, N_BASIS)
            f_minus = inv.objective_function(p_minus, obs, K_MAX, LMAX, RADIAL, N_BASIS)
            numeric[i] = (f_plus - f_minus) / (2 * eps)

        np.testing.assert_allclose(analytic, numeric, atol=1e-4, rtol=1e-3)


def test_sample_random_recip_lattices_returns_valid_lattices():
    lats = inv.sample_random_recip_lattices(5)
    assert len(lats) == 5
    for mat in lats:
        assert np.asarray(mat).shape == (3, 3)
        # Reciprocal lattices from a valid triclinic cell must be invertible.
        assert abs(np.linalg.det(mat)) > 0
