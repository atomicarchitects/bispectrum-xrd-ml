import numpy as np
from pymatgen.core.lattice import Lattice

from analyze_completed_inversion_res import niggli_params


def test_niggli_params_matches_pymatgen_directly():
    mat = Lattice.from_parameters(3.0, 4.1, 5.3, 80.0, 95.0, 100.0).matrix
    a, b, c, alpha, beta, gamma = niggli_params(mat)

    expected = Lattice(mat).get_niggli_reduced_lattice()
    assert (a, b, c, alpha, beta, gamma) == (
        expected.a, expected.b, expected.c, expected.alpha, expected.beta, expected.gamma
    )


def test_niggli_params_cubic_is_already_reduced():
    mat = Lattice.cubic(5.0).matrix
    a, b, c, alpha, beta, gamma = niggli_params(mat)
    assert a == b == c == 5.0
    assert alpha == beta == gamma == 90.0
