import torch
from pymatgen.core import Lattice, Structure

import data_gen_full as dgf


def _cob_bi(l_max):
    import e3nn_jax
    from e3nn import io
    rtp_bi = e3nn_jax.reduced_symmetric_tensor_product_basis(
        io.SphericalTensor(l_max, 1, -1), 3, keep_ir=["0o", "0e"]
    )
    return torch.tensor(rtp_bi.array, dtype=torch.float64)


def test_bispec_worker_returns_mpid_and_finite_result():
    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
    # bispec_worker calls training_data_gen() without overriding l_max, so its
    # internal spherical harmonics are computed at the default l_max=6 --
    # cob_bi must match that basis size ((l_max+1)**2) or get_bispectrum's
    # einsum contraction will raise a shape-mismatch error.
    cob_bi = _cob_bi(l_max=6)

    out = dgf.bispec_worker(("mp-test", structure, cob_bi))

    assert out is not None
    mpid, bispec = out
    assert mpid == "mp-test"
    assert torch.isfinite(bispec).all()


def test_bispec_worker_returns_none_on_failure():
    out = dgf.bispec_worker(("mp-bad", "not a structure", None))
    assert out is None


def test_xrd_worker_returns_mpid_and_normalized_pattern():
    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])

    out = dgf.xrd_worker(("mp-test", structure))

    assert out is not None
    mpid, xrd = out
    assert mpid == "mp-test"
    assert xrd.max() == 1.0


def test_xrd_worker_returns_none_on_failure():
    out = dgf.xrd_worker(("mp-bad", "not a structure"))
    assert out is None
