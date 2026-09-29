import numpy as np
import pytest
from pymatgen.core import Lattice

import utilities as u

pytest.importorskip("cctbx")

CUBIC = 5.0 * np.eye(3)
# a different basis for the same lattice (unimodular, det = 1)
REBASE = np.array([[1, 1, 0], [0, 1, 0], [0, 0, 1]])
BCC = 4.0 / 2 * np.array([[-1, 1, 1], [1, -1, 1], [1, 1, -1]])


def _candidates(true_lat, pred_lat):
    return u.get_cctbx_candidates(true_lat, pred_lat, is_reciprocal=False)


def test_returns_cells_volumes_and_bravais_for_both():
    result = _candidates(CUBIC, CUBIC)
    assert len(result) == 6
    true_params, true_vol, true_bravais, pred_params, pred_vol, pred_bravais = result
    assert true_params.shape == (6,) and pred_params.shape == (6,)
    assert isinstance(true_vol, float) and isinstance(pred_vol, float)
    assert isinstance(true_bravais, str) and isinstance(pred_bravais, str)


@pytest.mark.parametrize("lat, bravais, params", [
    (CUBIC, "cP", [5, 5, 5, 90, 90, 90]),
    (np.diag([4.0, 4.0, 7.0]), "tP", [4, 4, 7, 90, 90, 90]),
    (Lattice.from_parameters(4, 6, 7, 90, 105, 90).matrix, "mP", [4, 6, 7, 90, 105, 90]),
])
def test_recovers_known_bravais_type(lat, bravais, params):
    true_params, _, true_bravais, _, _, _ = _candidates(lat, lat)
    assert true_bravais == bravais
    np.testing.assert_allclose(true_params, params, atol=1e-6)


@pytest.mark.parametrize("lat, bravais", [(CUBIC, "cP"), (BCC, "cI")])
def test_cell_choice_does_not_change_result(lat, bravais):
    """Two bases of the same lattice must give identical cells and zero error."""
    true_params, true_vol, true_bravais, pred_params, pred_vol, pred_bravais = _candidates(lat, REBASE @ lat)
    assert true_bravais == pred_bravais == bravais
    np.testing.assert_allclose(pred_params, true_params, atol=1e-6)
    assert pred_vol == pytest.approx(true_vol)


def test_centered_lattice_reported_in_conventional_cell():
    true_params, true_vol, true_bravais, _, _, _ = _candidates(BCC, BCC)
    assert true_bravais == "cI"
    np.testing.assert_allclose(true_params, [4, 4, 4, 90, 90, 90], atol=1e-6)
    assert true_vol == pytest.approx(64.0)


def test_permuted_axes_give_zero_error():
    mono = Lattice.from_parameters(4, 6, 7, 90, 105, 90).matrix
    true_params, _, _, pred_params, _, _ = _candidates(mono, mono[[1, 2, 0]])
    np.testing.assert_allclose(pred_params, true_params, atol=1e-6)


@pytest.mark.parametrize("alpha, bravais", [(90.05, "cP"), (90.3, "mP")])
def test_angle_tolerance(alpha, bravais):
    """max_delta = 0.1 degrees: smaller deviations snap to the higher symmetry."""
    distorted = Lattice.from_parameters(5, 5, 5, alpha, 90, 90).matrix
    _, _, _, _, _, pred_bravais = _candidates(CUBIC, distorted)
    assert pred_bravais == bravais


def test_true_aligned_cell_does_not_depend_on_prediction():
    a = u.get_cctbx_true_aligned(BCC, CUBIC, is_reciprocal=False)
    b = u.get_cctbx_true_aligned(BCC, np.diag([3.0, 8.0, 9.0]), is_reciprocal=False)
    assert a[2] == b[2] == "cI"
    np.testing.assert_allclose(a[0], b[0], atol=1e-6)


# A near-correct triclinic prediction (mp-759011) reduced to the other Niggli setting:
# angles near 62 deg where the true cell has 118 deg.
FLIP_TRUE = Lattice.from_parameters(5.53, 5.67, 5.91, 118.2, 117.9, 90.5)
FLIP_PRED = Lattice.from_parameters(5.50, 5.76, 5.87, 61.7, 62.1, 90.0)


def test_setting_match_recovers_flipped_triclinic_cell():
    matched_lat, matched = u.match_cell_setting(FLIP_TRUE, FLIP_PRED)
    assert matched
    np.testing.assert_allclose(matched_lat.angles, FLIP_TRUE.angles, atol=1.0)


def test_setting_match_only_changes_the_basis():
    matched_lat, _ = u.match_cell_setting(FLIP_TRUE, FLIP_PRED)
    assert matched_lat.volume == pytest.approx(FLIP_PRED.volume)
    np.testing.assert_allclose(matched_lat.get_niggli_reduced_lattice().parameters,
                               FLIP_PRED.get_niggli_reduced_lattice().parameters, atol=1e-4)
    assert FLIP_PRED.find_mapping(matched_lat, ltol=1e-5, atol=1e-3) is not None


def test_setting_match_leaves_similar_cells_unchanged():
    pred = Lattice.from_parameters(5.52, 5.68, 5.90, 118.0, 118.1, 90.4)
    matched_lat, matched = u.match_cell_setting(FLIP_TRUE, pred)
    assert not matched
    assert matched_lat is pred


def test_setting_match_ignores_unrelated_cells():
    _, matched = u.match_cell_setting(Lattice(CUBIC), Lattice(np.diag([3.0, 8.0, 9.0])))
    assert not matched
