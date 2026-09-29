import gc
import os
import pickle
import random

import numpy as np
import pytest
import torch
from pymatgen.core import Lattice, Structure

import generate_augmented_data as gad
from utilities import seed_from_mpid

# POWDERXRD_REAL_DATA_BASE is accepted for backward compatibility; prefer
# POWDERXRD_DATA_ROOT, which is what conf/data/*.yaml and the rest of the
# scripts in this repo use for the same purpose.
REAL_DATA_BASE = os.environ.get(
    "POWDERXRD_REAL_DATA_BASE",
    os.environ.get("POWDERXRD_DATA_ROOT", "./data"),
)


def _cob_bi(l_max):
    import e3nn_jax
    from e3nn import io
    rtp_bi = e3nn_jax.reduced_symmetric_tensor_product_basis(
        io.SphericalTensor(l_max, 1, -1), 3, keep_ir=["0o", "0e"]
    )
    return torch.tensor(rtp_bi.array, dtype=torch.float64)


@pytest.fixture(autouse=True)
def _reset_bispec_lmax():
    orig = gad.BISPEC_LMAX
    yield
    gad.BISPEC_LMAX = orig


@pytest.fixture(autouse=True)
def _gc_after_test():
    # The real-data tests each load multi-GB pkls; without an explicit
    # collect between them, running the whole suite in one process can
    # accumulate enough resident memory to get OOM-killed even though each
    # test is fine in isolation.
    yield
    gc.collect()


# ---------------------------------------------------------------------------
# Worker functions on a synthetic structure (fast, no real data needed)
# ---------------------------------------------------------------------------

def test_single_worker_normalized():
    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
    mpid, xrd = gad.single_worker(("mp-test", structure, 0))
    assert mpid == "mp-test"
    assert xrd.max() == pytest.approx(1.0)


def test_caglioti_worker_produces_4_sets_plus_metadata():
    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
    out = gad.caglioti_worker(("mp-test", structure, 0))
    xrd_keys = {k for k in out if not k.endswith("_caglioti")}
    assert xrd_keys == {f"mp-test_{i}" for i in range(4)}
    for k in xrd_keys:
        assert f"{k}_caglioti" in out
        assert out[k].max() == pytest.approx(1.0)


def test_caglioti4x_worker_produces_4_xrd_only():
    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
    out = gad.caglioti4x_worker(("mp-test", structure, 2.0, 0.15, 0))
    assert set(out) == {f"mp-test_{i}" for i in range(4)}
    for v in out.values():
        assert v.max() == pytest.approx(1.0)


def test_texture_worker_produces_n_variants():
    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])
    mpid, out = gad.texture_worker(("mp-test", structure, 3, 0.3, 0))
    assert mpid == "mp-test"
    assert set(out) == {"mp-test_texture_1", "mp-test_texture_2", "mp-test_texture_3"}


def test_strain_worker_produces_xrd_bispec_lattice():
    gad._cob_bi = _cob_bi(2)
    gad.BISPEC_LMAX = 2
    structure = Structure(Lattice.cubic(4.0), ["Si"], [[0, 0, 0]])

    result = gad.strain_worker(("mp-test", structure, 2, 0.05, 0))

    assert result is not None
    mpid, xrd_out, bisp_out, lattice_out = result
    assert mpid == "mp-test"
    expected_keys = {"mp-test_strain_1", "mp-test_strain_2"}
    assert set(xrd_out) == set(bisp_out) == set(lattice_out) == expected_keys
    for lat in lattice_out.values():
        assert len(lat) == 6


class _Args:
    def __init__(self, **kw):
        self.__dict__.update(kw)


def test_run_combine_merges_disjoint_key_namespaces(tmp_path):
    cag_dir, strain_dir, texture_dir, out_dir = (tmp_path / d for d in ("cag", "strain", "texture", "out"))
    for d in (cag_dir, strain_dir, texture_dir):
        d.mkdir()

    def dump(d, name, obj):
        with open(d / name, "wb") as f:
            pickle.dump(obj, f)

    dump(cag_dir, "train_sim_xrd_caglioti.pkl", {"mp-1_0": 1, "mp-1_1": 2})
    dump(cag_dir, "train_bispec_dict.pkl", {"mp-1_0": 10, "mp-1_1": 20})
    dump(cag_dir, "train_prim_lat.pkl", {"mp-1_0": [1] * 6, "mp-1_1": [1] * 6})

    dump(strain_dir, "train_sim_xrd_strain.pkl", {"mp-1_strain_1": 3})
    dump(strain_dir, "train_bispec_dict.pkl", {"mp-1_strain_1": 30})
    dump(strain_dir, "train_prim_lat.pkl", {"mp-1_strain_1": [2] * 6})

    dump(texture_dir, "train_sim_xrd_texture.pkl", {"mp-1_texture_1": 4})
    dump(texture_dir, "train_bispec_dict.pkl", {"mp-1_texture_1": 40})
    dump(texture_dir, "train_prim_lat.pkl", {"mp-1_texture_1": [3] * 6})

    args = _Args(splits=["train"], caglioti_dir=str(cag_dir), strain_dir=str(strain_dir),
                 texture_dir=str(texture_dir), output_dir=str(out_dir))
    gad.run_combine(args)

    xrd = gad._load(out_dir / "train_sim_xrd_combined.pkl")
    bispec = gad._load(out_dir / "train_bispec_combined.pkl")
    lattice = gad._load(out_dir / "train_prim_lat_combined.pkl")

    expected_keys = {"mp-1_0", "mp-1_1", "mp-1_strain_1", "mp-1_texture_1"}
    assert set(xrd) == set(bispec) == set(lattice) == expected_keys
    assert xrd["mp-1_texture_1"] == 4
    assert bispec["mp-1_strain_1"] == 30


@pytest.mark.parametrize(
    "sg_num,expected",
    [(1, "triclinic"), (16, "orthorhombic"), (76, "low_sym_hextrig"),
     (83, "high_sym_hextrig"), (195, "cubic")],
)
def test_sg_class(sg_num, expected):
    assert gad._sg_class(sg_num) == expected


# ---------------------------------------------------------------------------
# CLI validation
# ---------------------------------------------------------------------------

def test_cli_requires_bispec_dir_for_texture_and_caglioti():
    parser = gad.build_parser()
    args = parser.parse_args(["--augmentation", "texture", "--output-dir", "/tmp/x"])
    assert args.augmentation in gad.REQUIRES_BISPEC_DIR
    assert args.bispec_dir is None  # main() would parser.error() on this


def test_cli_defaults_to_mpfull_dataset():
    parser = gad.build_parser()
    args = parser.parse_args(["--augmentation", "strain"])
    assert args.dataset == "mpfull"
    assert args.struct_dir is None  # resolved from --dataset in main(), not parse_args


@pytest.mark.parametrize(
    "dataset,cli_symprec,expected",
    [
        ("mpfull", None, 0),
        ("mp20", None, 1e-3),
        ("mpfull", "1e-3", 1e-3),  # explicit override wins over the per-dataset default
    ],
)
def test_single_symprec_override_resolution(dataset, cli_symprec, expected):
    parser = gad.build_parser()
    argv = ["--augmentation", "single", "--dataset", dataset]
    if cli_symprec is not None:
        argv += ["--symprec", cli_symprec]
    args = parser.parse_args(argv)
    resolved = args.symprec if args.symprec is not None else gad.SINGLE_SYMPREC.get(args.dataset, 0)
    assert resolved == expected


@pytest.mark.parametrize(
    "dataset,augmentation,expected_suffix",
    [
        ("mpfull", "strain", "mpfull_strain"),
        ("mpfull", "texture", "mpfull_texture_cag"),
        ("mpfull", "caglioti4x", "mpfull_aug"),
        ("mpfull", "single", "mpfull_crystalyze_cag"),
        ("mp20", "strain", "mp20_strain"),
        ("mp20", "texture", "mp20_texture_cag"),
        ("mp20", "caglioti", "mp20_cag"),
        ("mp20", "single", "mp20_crystalyze_cag"),
    ],
)
def test_default_output_dir_per_dataset_and_augmentation(dataset, augmentation, expected_suffix):
    assert gad.DEFAULT_OUTPUT_DIR[dataset][augmentation].endswith(expected_suffix)


@pytest.mark.parametrize(
    "dataset,augmentation",
    [("mpfull", "caglioti"), ("mp20", "caglioti4x")],
)
def test_no_default_output_dir_where_unconfirmed(dataset, augmentation):
    # mpfull/caglioti: data_gen_full_cag.py never had a default, preserved
    # deliberately. mp20/caglioti4x: no real directory could be confirmed
    # against actual filenames (mp20_aug/, mp20aug/, mp20_crystalyze_aug/ all
    # exist but none matches the expected {split}_sim_xrd_aug.pkl pattern).
    assert augmentation not in gad.DEFAULT_OUTPUT_DIR[dataset]


# ---------------------------------------------------------------------------
# Regression check against real, already-generated training data.
#
# The unit tests above only prove the new code matches the *old scripts'
# logic* on a synthetic structure. These prove it reproduces what's actually
# sitting on disk and was actually used for training. Parametrized over both
# datasets so a machine with only the (much smaller, ~10.5GB vs mpfull's
# ~33GB) mp20 data mirrored under POWDERXRD_REAL_DATA_BASE still gets full
# coverage -- each dataset's cases skip independently if that dataset's
# directories aren't present, rather than skipping the whole test.
#
# All non-default parameters (--max-texture 0.5, etc.) were independently
# re-verified against mp20's real data, not assumed to carry over from
# mpfull -- see the max_texture/n_strains/max_strain brute-force checks in
# the commit history.
#
# Not covered: caglioti4x. Its seed is `--seed + enumeration index over the
# struct dict`, not seed_from_mpid(mpid), so reproducing one material's
# output requires replaying the exact mpid order + --seed of the original
# run, which isn't recoverable after the fact.
# ---------------------------------------------------------------------------

DATASETS = ["mpfull", "mp20"]


def _load(path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _sample_common(a, b, n, seed=0):
    common = sorted(set(a) & set(b))
    return random.Random(seed).sample(common, min(n, len(common)))


def _real_struct_dir(dataset):
    rel = os.path.relpath(gad.DEFAULT_STRUCT_DIR[dataset], gad._BASE)
    return os.path.normpath(os.path.join(REAL_DATA_BASE, rel))


def _real_output_dir(dataset, augmentation):
    if augmentation in gad.DEFAULT_OUTPUT_DIR[dataset]:
        rel = os.path.relpath(gad.DEFAULT_OUTPUT_DIR[dataset][augmentation], gad._BASE)
        return os.path.normpath(os.path.join(REAL_DATA_BASE, rel))
    # mpfull/caglioti has no script default (data_gen_full_cag.py never had
    # one, preserved deliberately) -- confirmed real directory is {dataset}_cag/.
    assert augmentation == "caglioti"
    return os.path.join(REAL_DATA_BASE, f"{dataset}_cag")


def _skip_unless_present(*dirs):
    missing = [d for d in dirs if not os.path.isdir(d)]
    if missing:
        pytest.skip(f"not present: {missing}")


@pytest.mark.parametrize("dataset", DATASETS)
def test_single_matches_real_training_data(dataset):
    struct_dir = _real_struct_dir(dataset)
    output_dir = _real_output_dir(dataset, "single")
    _skip_unless_present(struct_dir, output_dir)

    struct_data = _load(f"{struct_dir}/train_data.pkl")
    real_xrd = _load(f"{output_dir}/train_sim_xrd.pkl")
    mpids = _sample_common(struct_data, real_xrd, n=20)
    assert mpids, f"[{dataset}] no common mpids between struct_data and real single-cag output"

    symprec = gad.SINGLE_SYMPREC[dataset]
    for mpid in mpids:
        _, xrd = gad.single_worker((mpid, struct_data[mpid], symprec))
        expected = real_xrd[mpid]
        expected = expected.numpy() if hasattr(expected, "numpy") else expected
        np.testing.assert_allclose(xrd, expected, atol=1e-5,
                                    err_msg=f"[{dataset}] single-mode mismatch for {mpid}")


@pytest.mark.parametrize("dataset", DATASETS)
def test_texture_matches_real_training_data(dataset):
    struct_dir = _real_struct_dir(dataset)
    output_dir = _real_output_dir(dataset, "texture")
    _skip_unless_present(struct_dir, output_dir)

    struct_data = _load(f"{struct_dir}/train_data.pkl")
    real_xrd = _load(f"{output_dir}/train_sim_xrd_texture.pkl")
    base_mpids = {k.rsplit("_texture_", 1)[0] for k in real_xrd}
    mpids = _sample_common(struct_data, base_mpids, n=10)
    assert mpids, f"[{dataset}] no common mpids between struct_data and real texture output"

    for mpid in mpids:
        _, variants = gad.texture_worker((mpid, struct_data[mpid], 3, 0.5, seed_from_mpid(mpid)))
        for key, xrd in variants.items():
            assert key in real_xrd, f"[{dataset}] {key} missing from real texture output"
            np.testing.assert_allclose(xrd, real_xrd[key], atol=1e-5,
                                        err_msg=f"[{dataset}] texture mismatch for {key}")


@pytest.mark.parametrize("dataset", DATASETS)
def test_caglioti_matches_real_training_data(dataset):
    struct_dir = _real_struct_dir(dataset)
    output_dir = _real_output_dir(dataset, "caglioti")
    _skip_unless_present(struct_dir, output_dir)

    struct_data = _load(f"{struct_dir}/train_data.pkl")
    real_xrd = _load(f"{output_dir}/train_sim_xrd_caglioti.pkl")
    base_mpids = {k.rsplit("_", 1)[0] for k in real_xrd if not k.endswith("_caglioti")}
    mpids = _sample_common(struct_data, base_mpids, n=10)
    assert mpids, f"[{dataset}] no common mpids between struct_data and real caglioti output"

    for mpid in mpids:
        out = gad.caglioti_worker((mpid, struct_data[mpid], seed_from_mpid(mpid)))
        for key, val in out.items():
            if key.endswith("_caglioti"):
                continue
            assert key in real_xrd, f"[{dataset}] {key} missing from real caglioti output"
            np.testing.assert_allclose(val, real_xrd[key], atol=1e-5,
                                        err_msg=f"[{dataset}] caglioti mismatch for {key}")


@pytest.mark.parametrize("dataset", DATASETS)
def test_strain_matches_real_training_data(dataset):
    struct_dir = _real_struct_dir(dataset)
    output_dir = _real_output_dir(dataset, "strain")
    _skip_unless_present(struct_dir, output_dir)

    struct_data = _load(f"{struct_dir}/train_data.pkl")
    real_xrd = _load(f"{output_dir}/train_sim_xrd_strain.pkl")
    base_mpids = {k.rsplit("_strain_", 1)[0] for k in real_xrd}
    mpids = _sample_common(struct_data, base_mpids, n=5)
    assert mpids, f"[{dataset}] no common mpids between struct_data and real strain output"

    gad._strain_worker_init(gad.COB_DEFAULT)
    for mpid in mpids:
        result = gad.strain_worker((mpid, struct_data[mpid], 3, 0.04, seed_from_mpid(mpid)))
        assert result is not None
        _, xrd_out, _, _ = result
        for key, xrd in xrd_out.items():
            assert key in real_xrd, f"[{dataset}] {key} missing from real strain output"
            np.testing.assert_allclose(xrd, real_xrd[key], atol=1e-5,
                                        err_msg=f"[{dataset}] strain mismatch for {key}")


@pytest.mark.parametrize("dataset", DATASETS)
def test_combine_matches_real_combined_training_data(dataset):
    # Deliberately checks only the bispec files (tens of MB), not the ~9GB
    # combined XRD file -- run_combine's merge is the same plain dict.update()
    # for xrd/bispec/lattice alike (already verified: test_run_combine_
    # merges_disjoint_key_namespaces), so checking bispec here is enough to
    # catch a real regression without OOMing a shared login node.
    combined_dir = os.path.join(REAL_DATA_BASE, f"{dataset}_aug_cag_strain_texture")
    caglioti_dir = _real_output_dir(dataset, "caglioti")
    strain_dir = _real_output_dir(dataset, "strain")
    texture_dir = _real_output_dir(dataset, "texture")
    _skip_unless_present(combined_dir, caglioti_dir, strain_dir, texture_dir)

    merged_bispec = {}
    merged_bispec.update(_load(f"{caglioti_dir}/train_bispec_dict.pkl"))
    merged_bispec.update(_load(f"{strain_dir}/train_bispec_dict.pkl"))
    merged_bispec.update(_load(f"{texture_dir}/train_bispec_dict.pkl"))

    real_bispec = _load(f"{combined_dir}/train_bispec_combined.pkl")

    assert set(merged_bispec) == set(real_bispec)
    mpids = _sample_common(merged_bispec, real_bispec, n=20)
    for key in mpids:
        np.testing.assert_allclose(merged_bispec[key], real_bispec[key], atol=1e-6,
                                    err_msg=f"[{dataset}] combine mismatch for {key}")
