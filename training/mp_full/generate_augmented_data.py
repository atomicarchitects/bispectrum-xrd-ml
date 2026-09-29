"""
Generate augmented mpfull/mp20 XRD (+ bispectrum) datasets. Combines what used
to be five separate scripts (data_gen_mpfull_strain.py, data_gen_texture_cag.py,
data_gen_full_cag.py, data_gen_full_aug.py, data_gen_single_cag.py) into one
CLI, since they shared most of their Caglioti-simulation and Pool-worker
plumbing and had drifted out of sync with each other (see git history).

Augmentation types (--augmentation):
    strain      symmetry-preserving lattice strain per crystal class, plus
                recomputed bispectrum. Keys: {mpid}_strain_{1..n}.
    texture     preferred-orientation peak-intensity scaling (mirrors
                TextureGen from autoXRD/Crystalyze). Keys: {mpid}_texture_{1..n}.
    caglioti    4 fixed Caglioti (U,V,W) sets, no intensity perturbation.
                Keys: {mpid}_{0..3}.
    caglioti4x  same 4 Caglioti sets as `caglioti`, following the Crystalyze
                paper recipe: augs 0/2/3 also get a randomized per-peak
                intensity perturbation (Gamma-distributed, MLE-fit to RRUFF
                by fit_intensity_gamma.py). No bispectrum output -- lattice
                is unchanged so reuse whatever `caglioti`/base bispec you have.
                Keys: {mpid}_{0..3}.
    single      one fixed Caglioti set (Crystalyze reference), no augmentation
                diversity -- this is the current *training* base. Bare mpid
                keys, matching the existing bispec/prim_lat dicts directly.

All modes broaden peaks with the same Crystalyze reference params
(U=0.05, V=-0.06, W=0.07) unless noted otherwise.

Usage:
    python generate_augmented_data.py --augmentation strain \\
        --struct-dir /path/to/powderxrd --output-dir /path/to/mpfull_strain \\
        --n-strains 3 --max-strain 0.2

    python generate_augmented_data.py --augmentation texture \\
        --struct-dir /path/to/powderxrd --bispec-dir /path/to/mpfull \\
        --output-dir /path/to/mpfull_texture_cag --n-textures 3 --max-texture 0.3

    python generate_augmented_data.py --augmentation caglioti \\
        --struct-dir /path/to/powderxrd --bispec-dir /path/to/mpfull \\
        --output-dir /path/to/mpfull_cag

    python generate_augmented_data.py --augmentation caglioti4x \\
        --struct-dir /path/to/powderxrd --output-dir /path/to/mpfull_aug

    python generate_augmented_data.py --augmentation single \\
        --struct-dir /path/to/powderxrd --output-dir /path/to/mpfull_cag_0.05_-0.06_0.07
"""
import argparse
import os
import pickle
from multiprocessing import Pool, cpu_count

import numpy as np
import torch
from pymatgen.analysis.diffraction.xrd import XRDCalculator
from pymatgen.core import Lattice, Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from scipy.signal import find_peaks
from tqdm import tqdm

from utilities import caglioti_fwhm, get_bispectrum, get_neighbors, seed_from_mpid

# ---------------------------------------------------------------------------
# Shared simulation parameters
# ---------------------------------------------------------------------------
# Crystalyze reference Caglioti params -- used directly by strain/texture/single,
# and as CAGLIOTI_SETS[0] for caglioti/caglioti4x.
CRYSTALYZE_U, CRYSTALYZE_V, CRYSTALYZE_W = 0.05, -0.06, 0.07

CAGLIOTI_SETS = [
    {"U": 0.05, "V": -0.06, "W": 0.07,                 "perturb": True},   # 0
    {"U": 0.05, "V": -0.01, "W": 0.01,                 "perturb": False},  # 1
    {"U": 0.0,  "V":  0.0,  "W": 0.01,                 "perturb": True},   # 2
    {"U": 0.0,  "V":  0.0,  "W": ("unif", 0.001, 0.1), "perturb": True},   # 3
]

TWO_THETA_MIN, TWO_THETA_MAX, TWO_THETA_STEP = 5.0, 90.0, 0.01
TWO_THETA_GRID = np.arange(TWO_THETA_MIN, TWO_THETA_MAX, TWO_THETA_STEP)  # matches simulate_pv_xrd's implicit grid
PV_ETA = 0.0  # pure Gaussian peak shape

BISPEC_WAVELENGTH = 3.0
BISPEC_RADIAL = "bessel"
BISPEC_N_BASIS = 10
BISPEC_LMAX = 6
COB_DEFAULT = os.environ.get("POWDERXRD_DATA_ROOT", "./data") + "/cob_lmax6.pkl"

# caglioti4x intensity-perturbation Gamma, auto-loaded from
# fit_intensity_gamma.py's output if present.
INTENS_GAMMA_FIT_PATH = os.environ.get("POWDERXRD_DATA_ROOT", "./data") + "/rruff/intens_gamma.pkl"
INTENS_GAMMA_A_DEFAULT = 2.0
INTENS_GAMMA_SCALE_DEFAULT = 0.15
if os.path.exists(INTENS_GAMMA_FIT_PATH):
    try:
        with open(INTENS_GAMMA_FIT_PATH, "rb") as _f:
            _fit = pickle.load(_f)
        INTENS_GAMMA_A_DEFAULT = float(_fit["a"])
        INTENS_GAMMA_SCALE_DEFAULT = float(_fit["scale"])
        print(f"[generate_augmented_data] loaded RRUFF-MLE intensity gamma "
              f"a={INTENS_GAMMA_A_DEFAULT:.4f}, scale={INTENS_GAMMA_SCALE_DEFAULT:.4f}")
    except Exception as _e:
        print(f"[generate_augmented_data] could not load {INTENS_GAMMA_FIT_PATH}: {_e}")

# Train-time background-noise hyperparameters (bg_noise_aug.add_bg_noise samples
# from these); kept here purely as a documented reference, same values as there.
BG_NOISE_LOGN_MEAN = np.array([1.06709083, -5.39670499])
BG_NOISE_LOGN_COV = np.array([[0.19984825, 0.14809546],
                              [0.14809546, 1.31019437]])


# ---------------------------------------------------------------------------
# Shared XRD simulation
# ---------------------------------------------------------------------------
def simulate_pv_xrd(centers, intensities, U, V, W, eta=PV_ETA,
                     two_theta_min=TWO_THETA_MIN, two_theta_max=TWO_THETA_MAX,
                     step=TWO_THETA_STEP):
    """Vectorised superimposed pseudo-Voigt simulation (Gaussian when eta=0)."""
    x = np.arange(two_theta_min, two_theta_max, step)
    centers = np.asarray(centers, dtype=np.float64)
    intensities = np.asarray(intensities, dtype=np.float64)

    fwhm = caglioti_fwhm(centers, U, V, W)
    sigma = fwhm / (2.0 * np.sqrt(2.0 * np.log(2.0)))

    dx = x[None, :] - centers[:, None]
    gauss = np.exp(-(dx ** 2) / (2.0 * sigma[:, None] ** 2))
    if eta == 0.0:
        comp = gauss
    else:
        lor = (fwhm[:, None] ** 2) / (dx ** 2 + fwhm[:, None] ** 2)
        comp = eta * lor + (1.0 - eta) * gauss

    total = (intensities[:, None] * comp).sum(axis=0)
    m = total.max()
    if m > 0:
        total = total / m
    return total.astype(np.float32)


def _resolve_caglioti(spec, rng):
    U, V, W = spec["U"], spec["V"], spec["W"]
    if isinstance(W, tuple) and W[0] == "unif":
        W = float(rng.uniform(W[1], W[2]))
    return float(U), float(V), float(W)


def _load_structure(struct_dump):
    if isinstance(struct_dump, str):
        return Structure.from_str(struct_dump, fmt="cif")
    return struct_dump


# ---------------------------------------------------------------------------
# Per-worker lazy globals
# ---------------------------------------------------------------------------
_calculators = {}  # keyed by symprec -- different augmentation types use
                    # different symprec (strain/texture/caglioti: 1e-3,
                    # caglioti4x: 0.1, single: pymatgen's own default of 0),
                    # so a single unkeyed cache would silently return the
                    # wrong instance if two types ran in the same process.
_cob_bi = None


def _get_calculator(symprec=1e-3):
    if symprec not in _calculators:
        _calculators[symprec] = XRDCalculator(wavelength="CuKa", symprec=symprec)
    return _calculators[symprec]


def _strain_worker_init(cob_path):
    global _cob_bi
    _cob_bi = torch.load(cob_path, map_location="cpu", weights_only=False)


# ---------------------------------------------------------------------------
# strain
# ---------------------------------------------------------------------------
def _sg_class(sg_num):
    if sg_num in range(195, 231):
        return "cubic"
    if sg_num in range(16, 76):
        return "orthorhombic"
    if sg_num in range(3, 16):
        return "monoclinic"
    if sg_num in range(1, 3):
        return "triclinic"
    low_sym = set(range(75, 83)) | set(range(143, 149)) | set(range(168, 175))
    return "low_sym_hextrig" if sg_num in low_sym else "high_sym_hextrig"


def _build_strain_tensor(sg_cls, max_strain, rng):
    d = rng.uniform(1.0 - max_strain, 1.0 + max_strain, 3)  # diagonal
    o = rng.uniform(-max_strain, max_strain, 6)              # off-diagonal
    s11, s22, s33 = d
    s12, s13, s21, s23, s31, s32 = o

    if sg_cls == "cubic":
        return np.array([[s11, 0, 0], [0, s11, 0], [0, 0, s11]])
    if sg_cls == "high_sym_hextrig":
        return np.array([[s11, 0, 0], [0, s11, 0], [0, 0, s33]])
    if sg_cls == "orthorhombic":
        return np.array([[s11, 0, 0], [0, s22, 0], [0, 0, s33]])
    if sg_cls == "monoclinic":
        return np.array([[s11, 0, s13], [0, s22, 0], [s13, 0, s33]])
    if sg_cls == "low_sym_hextrig":
        return np.array([[s11, s12, 0], [-s12, s22, 0], [0, 0, s33]])
    # triclinic
    return np.array([[s11, s12, s13], [s21, s22, s23], [s31, s32, s33]])


def _apply_strain(structure, max_strain, rng):
    # Standardise to conventional cell so the strain tensor axes match the
    # crystallographic axes (e.g. monoclinic b-unique, hex a=b).
    try:
        structure = SpacegroupAnalyzer(structure).get_refined_structure()
    except Exception:
        pass
    try:
        sg = structure.get_space_group_info()[1]
    except Exception:
        sg = 1
    S = _build_strain_tensor(_sg_class(sg), max_strain, rng)
    new_matrix = S @ structure.lattice.matrix
    strained = structure.copy()
    strained.lattice = Lattice(new_matrix)
    return strained


def _compute_bispec(structure):
    structure = structure.get_primitive_structure()
    recip_lat = structure.lattice.reciprocal_lattice_crystallographic.matrix
    k_max = 2.0 / BISPEC_WAVELENGTH
    miller_ind = get_neighbors(recip_lat, k_max)
    neighs = torch.tensor(miller_ind.reshape(-1, 3) @ recip_lat, dtype=torch.float64)
    bispec = get_bispectrum(neighs, k_max, BISPEC_LMAX, BISPEC_RADIAL, BISPEC_N_BASIS, _cob_bi)
    return bispec.numpy()


def strain_worker(args):
    mpid, struct_dump, n_strains, max_strain, seed = args
    try:
        structure = _load_structure(struct_dump)
        calc = _get_calculator()
        rng = np.random.default_rng(seed)

        xrd_out, bisp_out, lattice_out = {}, {}, {}
        for i in range(1, n_strains + 1):
            strained = _apply_strain(structure, max_strain, rng)
            pattern = calc.get_pattern(strained)
            xrd = simulate_pv_xrd(
                np.asarray(pattern.x, dtype=np.float64),
                np.asarray(pattern.y, dtype=np.float64),
                CRYSTALYZE_U, CRYSTALYZE_V, CRYSTALYZE_W,
            )
            key = f"{mpid}_strain_{i}"
            xrd_out[key] = xrd
            bisp_out[key] = _compute_bispec(strained)
            lattice = strained.get_primitive_structure().lattice
            a, b, c = lattice.abc
            alpha, beta, gamma = lattice.angles
            lattice_out[key] = [a, b, c, alpha, beta, gamma]

        return mpid, xrd_out, bisp_out, lattice_out
    except Exception as e:
        print(f"  Error {mpid}: {type(e).__name__} - {e}", flush=True)
        return None


def run_strain(args):
    n_workers = args.workers or min(cpu_count(), 32)
    print(f"U=V=W={CRYSTALYZE_U},{CRYSTALYZE_V},{CRYSTALYZE_W}  |  n_strains={args.n_strains}  |  "
          f"max_strain={args.max_strain}  |  {n_workers} workers")

    for split in args.splits:
        struct_data = _load_struct_data(args, split)
        mpids = list(struct_data.keys())
        print(f"  {len(mpids)} structures")

        job_args = [(mpid, struct_data[mpid], args.n_strains, args.max_strain, seed_from_mpid(mpid))
                    for mpid in mpids]

        xrd_out, bisp_out, lattice_out = {}, {}, {}
        with Pool(n_workers, initializer=_strain_worker_init, initargs=(args.cob_path,)) as pool:
            for result in tqdm(pool.imap_unordered(strain_worker, job_args, chunksize=4),
                                total=len(job_args), desc="strain"):
                if result is None:
                    continue
                _, xrd_vars, bisp_vars, lattice_vars = result
                xrd_out.update(xrd_vars)
                bisp_out.update(bisp_vars)
                lattice_out.update(lattice_vars)

        n_expected = len(mpids) * args.n_strains
        print(f"  XRD entries: {len(xrd_out)}  (expected {n_expected})  |  "
              f"Bispec entries: {len(bisp_out)}  |  Lattice entries: {len(lattice_out)}")

        _save(xrd_out, args.output_dir, f"{split}_sim_xrd_strain.pkl")
        _save(bisp_out, args.output_dir, f"{split}_bispec_dict.pkl")
        _save(lattice_out, args.output_dir, f"{split}_prim_lat.pkl")


# ---------------------------------------------------------------------------
# texture
# ---------------------------------------------------------------------------
def _map_interval(v, max_texture):
    """Map v in [0,1] -> [1-max_texture, 1]."""
    bound = 1.0 - max_texture
    return bound + ((1.0 - bound) / 1.0) * v


def _textured_intensities(hkl_list, intensities, max_texture, rng):
    n_indices = len(hkl_list[0])
    pref = np.zeros(n_indices)
    while np.dot(pref, pref) == 0.0:
        pref = np.array([rng.choice([0, 1]) for _ in range(n_indices)], dtype=float)

    scaled = []
    for hkl, peak in zip(hkl_list, intensities):
        hkl = np.array(hkl, dtype=float)
        norm_h = np.linalg.norm(hkl)
        norm_p = np.linalg.norm(pref)
        tf = 0.0 if (norm_h == 0 or norm_p == 0) else abs(np.dot(hkl, pref) / (norm_h * norm_p))
        scaled.append(peak * _map_interval(tf, max_texture))
    return np.array(scaled, dtype=np.float64)


def texture_worker(args):
    """Return (mpid, {mpid_texture_1: xrd, ...})."""
    mpid, struct_dump, n_textures, max_texture, seed = args
    try:
        structure = _load_structure(struct_dump)
        calc = _get_calculator()
        pattern = calc.get_pattern(structure)
        peak_x = np.asarray(pattern.x, dtype=np.float64)
        peak_y = np.asarray(pattern.y, dtype=np.float64)
        hkl_list = [v[0]["hkl"] for v in pattern.hkls]

        rng = np.random.default_rng(seed)
        out = {}
        for i in range(1, n_textures + 1):
            y_tex = _textured_intensities(hkl_list, peak_y, max_texture, rng)
            xrd = simulate_pv_xrd(peak_x, y_tex, CRYSTALYZE_U, CRYSTALYZE_V, CRYSTALYZE_W)
            out[f"{mpid}_texture_{i}"] = xrd
        return mpid, out
    except Exception as e:
        print(f"  Error {mpid}: {type(e).__name__} - {e}", flush=True)
        return None


def run_texture(args):
    n_workers = args.workers or min(cpu_count(), 32)
    print(f"Caglioti U={CRYSTALYZE_U}, V={CRYSTALYZE_V}, W={CRYSTALYZE_W}  |  "
          f"n_textures={args.n_textures}  |  max_texture={args.max_texture}  |  {n_workers} workers")

    for split in args.splits:
        struct_data = _load_struct_data(args, split)
        mpids = list(struct_data.keys())
        print(f"  {len(mpids)} structures")
        bispec_src = _load_bispec_src(args, split)

        job_args = [(mpid, struct_data[mpid], args.n_textures, args.max_texture, seed_from_mpid(mpid))
                    for mpid in mpids]

        xrd_out, bsp_out = {}, {}
        with Pool(n_workers) as pool:
            for result in tqdm(pool.imap_unordered(texture_worker, job_args, chunksize=8),
                                total=len(job_args), desc="texture XRD"):
                if result is None:
                    continue
                mpid, variants = result
                xrd_out.update(variants)
                if mpid in bispec_src:
                    bispec_val = bispec_src[mpid]
                    for i in range(1, 1 + args.n_textures):
                        bsp_out[f"{mpid}_texture_{i}"] = bispec_val

        n_expected = len(mpids) * args.n_textures
        print(f"  XRD entries: {len(xrd_out)}  (expected {n_expected})  |  Bispec entries: {len(bsp_out)}")

        _save(xrd_out, args.output_dir, f"{split}_sim_xrd_texture.pkl")
        _save(bsp_out, args.output_dir, f"{split}_bispec_dict.pkl")


# ---------------------------------------------------------------------------
# caglioti (4 fixed sets, no perturbation) / caglioti4x (same sets, + perturbation)
# ---------------------------------------------------------------------------
def caglioti_worker(args):
    """No perturbation; also emits {mpid}_{i}_caglioti metadata."""
    mpid, struct_dump, seed = args
    try:
        structure = _load_structure(struct_dump)
        rng = np.random.default_rng(seed)
        pattern = _get_calculator().get_pattern(structure)
        x = np.asarray(pattern.x, dtype=np.float64)
        y = np.asarray(pattern.y, dtype=np.float64)

        out = {}
        for i, spec in enumerate(CAGLIOTI_SETS):
            U, V, W = _resolve_caglioti(spec, rng)
            out[f"{mpid}_{i}"] = simulate_pv_xrd(x, y, U, V, W)
            out[f"{mpid}_{i}_caglioti"] = (U, V, W)
        return out
    except Exception as e:
        print(f"  Error {mpid}: {type(e).__name__} - {e}", flush=True)
        return None


def caglioti4x_worker(args):
    """Crystalyze recipe: augs 0/2/3 also get randomized intensity perturbation."""
    mpid, struct_dump, intens_a, intens_scale, seed = args
    try:
        structure = _load_structure(struct_dump)
        rng = np.random.default_rng(seed)
        pattern = _get_calculator(symprec=0.1).get_pattern(structure)
        peak_x = np.asarray(pattern.x, dtype=np.float64)
        peak_y = np.asarray(pattern.y, dtype=np.float64)

        out = {}
        for i, spec in enumerate(CAGLIOTI_SETS):
            U, V, W = _resolve_caglioti(spec, rng)
            if spec["perturb"] and intens_scale > 0:
                delta = rng.gamma(intens_a, intens_scale, size=peak_y.shape)
                sign = rng.choice([-1.0, 1.0], size=peak_y.shape)
                factors = np.clip(1.0 + sign * delta, 0.0, None)
                y_aug = peak_y * factors
            else:
                y_aug = peak_y
            out[f"{mpid}_{i}"] = simulate_pv_xrd(peak_x, y_aug, U, V, W)
        return out
    except Exception as e:
        print(f"  Error {mpid}: {type(e).__name__} - {e}", flush=True)
        return None


def run_caglioti(args):
    n_workers = args.workers or min(cpu_count(), 32)
    for split in args.splits:
        struct_data = _load_struct_data(args, split)
        print(f"  {len(struct_data)} structures")
        bispec_src = _load_bispec_src(args, split, required=False)

        job_args = [(mpid, struct, seed_from_mpid(mpid)) for mpid, struct in struct_data.items()]
        results = {}
        with Pool(n_workers) as pool:
            for out in tqdm(pool.imap_unordered(caglioti_worker, job_args, chunksize=8),
                             total=len(job_args), desc="Caglioti augmentation"):
                if out:
                    results.update(out)

        xrd_out = {k: v for k, v in results.items() if not k.endswith("_caglioti")}
        bsp_out = {}
        missing = 0
        for mpid in struct_data:
            if mpid not in bispec_src:
                missing += 1
                continue
            for i in range(len(CAGLIOTI_SETS)):
                key = f"{mpid}_{i}"
                if key in xrd_out:
                    bsp_out[key] = bispec_src[mpid]
        if missing:
            print(f"  [WARN] {missing} mpids missing from bispec source")

        _save(xrd_out, args.output_dir, f"{split}_sim_xrd_caglioti.pkl")
        if bsp_out:
            _save(bsp_out, args.output_dir, f"{split}_bispec_dict.pkl")


def run_caglioti4x(args):
    n_workers = args.workers or min(cpu_count(), 32)
    print(f"caglioti4x  |  intens_a={args.intens_gamma_a}  intens_scale={args.intens_gamma_scale}  |  "
          f"{n_workers} workers")

    seed_offset = args.seed
    for split in args.splits:
        struct_data = _load_struct_data(args, split)
        print(f"  {len(struct_data)} structures  (-> {4 * len(struct_data)} XRDs)")

        job_args = [(mpid, struct, args.intens_gamma_a, args.intens_gamma_scale, seed_offset + i)
                    for i, (mpid, struct) in enumerate(struct_data.items())]
        seed_offset += len(struct_data)

        results = {}
        with Pool(n_workers) as pool:
            for out in tqdm(pool.imap_unordered(caglioti4x_worker, job_args, chunksize=8),
                             total=len(job_args), desc="caglioti4x XRD"):
                if out is not None:
                    results.update(out)

        _save(results, args.output_dir, f"{split}_sim_xrd_aug.pkl")


# ---------------------------------------------------------------------------
# single (fixed Caglioti, no augmentation diversity -- the training base)
# ---------------------------------------------------------------------------
SINGLE_SYMPREC = {"mpfull": 0, "mp20": 1e-3}


def single_worker(args):
    mpid, struct_dump, symprec = args
    try:
        structure = _load_structure(struct_dump)
        pattern = _get_calculator(symprec=symprec).get_pattern(structure)
        xrd = simulate_pv_xrd(
            np.asarray(pattern.x, dtype=np.float64), np.asarray(pattern.y, dtype=np.float64),
            CRYSTALYZE_U, CRYSTALYZE_V, CRYSTALYZE_W,
        )
        return mpid, xrd
    except Exception as e:
        print(f"  Error {mpid}: {e}", flush=True)
        return None


def peaklist_strain_worker(args):
    """Mirrors strain_worker's exact RNG-consuming sequence (same
    _get_calculator(), same rng = default_rng(seed), same per-i
    _apply_strain call) so that {mpid}_strain_{i} keys correspond to the
    IDENTICAL strained structures already used to build
    mpfull_strain/{split}_bispec_dict.pkl -- only what happens after
    _apply_strain differs (peak list instead of bispec + dense curve), so
    that existing bispec target file can be reused as-is instead of
    recomputing it."""
    mpid, struct_dump, n_strains, max_strain, seed, max_peaks = args
    try:
        structure = _load_structure(struct_dump)
        calc = _get_calculator()
        rng = np.random.default_rng(seed)

        peaks_out = {}
        for i in range(1, n_strains + 1):
            strained = _apply_strain(structure, max_strain, rng)
            pattern = calc.get_pattern(strained)
            dense_xrd = simulate_pv_xrd(
                np.asarray(pattern.x, dtype=np.float64), np.asarray(pattern.y, dtype=np.float64),
                CRYSTALYZE_U, CRYSTALYZE_V, CRYSTALYZE_W,
            )
            peaks_out[f"{mpid}_strain_{i}"] = _extract_peaks_via_find_peaks(dense_xrd, max_peaks)

        return mpid, peaks_out
    except Exception as e:
        print(f"  Error {mpid}: {type(e).__name__} - {e}", flush=True)
        return None


def run_peaklist_strain(args):
    n_workers = args.workers or min(cpu_count(), 32)
    print(f"n_strains={args.n_strains}  |  max_strain={args.max_strain}  |  max_peaks={args.peaklist_max_peaks}  |  "
          f"{n_workers} workers (peak lists only -- reusing existing bispec_dict/prim_lat targets from --strain-dir)")

    for split in args.splits:
        struct_data = _load_struct_data(args, split)
        mpids = list(struct_data.keys())
        print(f"  {len(mpids)} structures")

        job_args = [(mpid, struct_data[mpid], args.n_strains, args.max_strain, seed_from_mpid(mpid), args.peaklist_max_peaks)
                    for mpid in mpids]

        peaks_out = {}
        with Pool(n_workers) as pool:
            for result in tqdm(pool.imap_unordered(peaklist_strain_worker, job_args, chunksize=4),
                                total=len(job_args), desc="peaklist_strain"):
                if result is None:
                    continue
                _, peaks_vars = result
                peaks_out.update(peaks_vars)

        n_expected = len(mpids) * args.n_strains
        print(f"  peaklist entries: {len(peaks_out)}  (expected {n_expected})")

        _save(peaks_out, args.output_dir, f"{split}_peaklist_strain.pkl")


def run_single(args):
    symprec = args.symprec if args.symprec is not None else SINGLE_SYMPREC.get(args.dataset, 0)
    n_workers = args.workers or min(cpu_count(), 32)
    print(f"Caglioti U={CRYSTALYZE_U}, V={CRYSTALYZE_V}, W={CRYSTALYZE_W}  |  symprec={symprec}  |  {n_workers} workers")

    for split in args.splits:
        struct_data = _load_struct_data(args, split)
        print(f"  {len(struct_data)} structures")

        job_args = [(mpid, struct, symprec) for mpid, struct in struct_data.items()]
        results = {}
        with Pool(n_workers) as pool:
            for out in tqdm(pool.imap_unordered(single_worker, job_args, chunksize=8),
                             total=len(job_args), desc="XRD"):
                if out is not None:
                    mpid, xrd = out
                    results[mpid] = xrd

        _save(results, args.output_dir, f"{split}_sim_xrd.pkl")


# ---------------------------------------------------------------------------
# peaklist (peak positions/prominences for the sparse/peak-list model input
# path -- model.py's XRDTransformerEncoder(input_type='sparse')). Same
# structures/symprec/Caglioti calculator as `single`, and reuses the same
# simulate_pv_xrd broadened curve -- peaks are then extracted from that dense
# curve with scipy.signal.find_peaks, NOT read directly from pymatgen's exact
# theoretical stick pattern. That's deliberate: XRDCalculator's raw peaks are
# more accurate (exact positions, no overlap) but that's exactly why they're
# unrealistic -- no real peak-picker could reproduce them from an actual
# (already-broadened, noisy) experimental scan. Running the same find_peaks
# extraction on both the simulated broadened curve here and on real eval-time
# data keeps training and eval peak lists on the same footing. Peaks are
# ranked by prominence (robust to background level, unlike raw height) and
# the top --max-peaks kept. Width is deliberately not extracted as a feature:
# it reflects instrumental/crystallite broadening, not lattice geometry.
# Bispectrum targets are representation-agnostic (depend only on the
# structure's primitive reciprocal lattice, not which XRD simulation was
# used), so this deliberately writes into the SAME output directory as
# `single` and reuses its {split}_data_bispec_list.pkl target file as-is --
# see conf/data/*_peaklist.yaml.
# ---------------------------------------------------------------------------
def _extract_peaks_via_find_peaks(dense_xrd, max_peaks):
    """Run scipy.signal.find_peaks on a broadened, max-normalized dense XRD
    curve (same TWO_THETA_GRID as simulate_pv_xrd's output) and return the
    top max_peaks (position, prominence) pairs, sorted ascending by
    position. Same extraction function is meant to be used on real
    experimental data at eval time for a consistent train/eval pipeline."""
    peak_idx, props = find_peaks(dense_xrd, height=1e-6, prominence=1e-6)
    if len(peak_idx) == 0:
        return []
    prominences = props["prominences"]
    top = np.argsort(prominences)[::-1][:max_peaks]
    positions = TWO_THETA_GRID[peak_idx[top]]
    return sorted(zip(positions.tolist(), prominences[top].tolist()))


def peaklist_worker(args):
    mpid, struct_dump, symprec, max_peaks = args
    try:
        structure = _load_structure(struct_dump)
        pattern = _get_calculator(symprec=symprec).get_pattern(structure)
        dense_xrd = simulate_pv_xrd(
            np.asarray(pattern.x, dtype=np.float64), np.asarray(pattern.y, dtype=np.float64),
            CRYSTALYZE_U, CRYSTALYZE_V, CRYSTALYZE_W,
        )
        peaks = _extract_peaks_via_find_peaks(dense_xrd, max_peaks)
        return mpid, peaks
    except Exception as e:
        print(f"  Error {mpid}: {e}", flush=True)
        return None


def run_peaklist(args):
    symprec = args.symprec if args.symprec is not None else SINGLE_SYMPREC.get(args.dataset, 0)
    n_workers = args.workers or min(cpu_count(), 32)
    print(f"Caglioti U={CRYSTALYZE_U}, V={CRYSTALYZE_V}, W={CRYSTALYZE_W}  |  symprec={symprec}  |  "
          f"max_peaks={args.peaklist_max_peaks}  |  {n_workers} workers")

    for split in args.splits:
        struct_data = _load_struct_data(args, split)
        print(f"  {len(struct_data)} structures")

        job_args = [(mpid, struct, symprec, args.peaklist_max_peaks) for mpid, struct in struct_data.items()]
        results = {}
        with Pool(n_workers) as pool:
            for out in tqdm(pool.imap_unordered(peaklist_worker, job_args, chunksize=8),
                             total=len(job_args), desc="peaklist"):
                if out is not None:
                    mpid, peaks = out
                    results[mpid] = peaks

        _save(results, args.output_dir, f"{split}_peaklist.pkl")


# ---------------------------------------------------------------------------
# peaklist_from_xrd -- extract peaks from an EXISTING dense XRD pkl (e.g.
# mpfull_texture_cag/{split}_sim_xrd_texture.pkl or
# mpfull_aug/{split}_sim_xrd_aug.pkl) via the same find_peaks pipeline as
# `peaklist`/`peaklist_strain`, without recomputing anything from structures.
# find_peaks only needs the already-broadened array, so this is a cheap
# post-processing pass over data that's already been generated -- much
# faster than re-simulating from scratch, and guarantees the extracted
# peaks correspond to the exact same augmented patterns already used to
# train the dense-curve models (e.g. mpfull_aug), for a fair comparison.
# ---------------------------------------------------------------------------
def run_peaklist_from_xrd(args):
    noise_on = args.bg_noise_cnrs_mix_prob is not None
    rng = None
    if noise_on:
        from bg_noise_aug import _sample_gamma_params
        rng = np.random.default_rng(args.bg_noise_seed)
        print(f"bg-noise ON: cnrs_mix_prob={args.bg_noise_cnrs_mix_prob}  seed={args.bg_noise_seed}  "
              f"(one Gamma realization per entry, added before find_peaks; same key set/count as "
              f"the clean run, just written to '{{split}}_peaklist_bgmix.pkl' instead)")
    out_name = "peaklist_bgmix" if noise_on else "peaklist"

    print(f"max_peaks={args.peaklist_max_peaks}  |  reading dense XRD from "
          f"{args.xrd_pkl_dir}/{args.xrd_pkl_pattern}")

    for split in args.splits:
        path = os.path.join(args.xrd_pkl_dir, args.xrd_pkl_pattern.format(split=split))
        print(f"\n=== {split.upper()} === ({path})")
        dense_dict = _load(path)
        if args.limit:
            dense_dict = {k: dense_dict[k] for k in list(dense_dict.keys())[: args.limit]}
        print(f"  {len(dense_dict)} entries")

        results = {}
        for key, dense_xrd in tqdm(dense_dict.items(), desc=f"peaklist_from_xrd[{split}]"):
            curve = np.asarray(dense_xrd, dtype=np.float64)
            if noise_on:
                a, scale = _sample_gamma_params(1, rng, cnrs_mix_prob=args.bg_noise_cnrs_mix_prob)
                curve = curve + rng.gamma(a[0], scale[0], size=curve.shape[0])
            results[key] = _extract_peaks_via_find_peaks(curve, args.peaklist_max_peaks)

        _save(results, args.output_dir, f"{split}_{out_name}.pkl")


# ---------------------------------------------------------------------------
# combine (merge separate caglioti/strain/texture runs into one training set)
# ---------------------------------------------------------------------------
def run_combine(args):
    """Merge the outputs of separate caglioti/strain/texture runs (disjoint
    key namespaces: {mpid}_{0..3}, {mpid}_strain_{1..n}, {mpid}_texture_{1..n})
    into one combined XRD/bispec/prim_lat pkl per split. Verified against the
    real mp20_aug_cag_strain_texture/train_*_combined.pkl (exact match on a
    sampled subset)."""
    for split in args.splits:
        print(f"\n=== {split.upper()} ===")
        xrd, bispec, lattice = {}, {}, {}

        xrd.update(_load(os.path.join(args.caglioti_dir, f"{split}_sim_xrd_caglioti.pkl")))
        bispec.update(_load(os.path.join(args.caglioti_dir, f"{split}_bispec_dict.pkl")))
        lattice.update(_load(os.path.join(args.caglioti_dir, f"{split}_prim_lat.pkl")))

        xrd.update(_load(os.path.join(args.strain_dir, f"{split}_sim_xrd_strain.pkl")))
        bispec.update(_load(os.path.join(args.strain_dir, f"{split}_bispec_dict.pkl")))
        lattice.update(_load(os.path.join(args.strain_dir, f"{split}_prim_lat.pkl")))

        xrd.update(_load(os.path.join(args.texture_dir, f"{split}_sim_xrd_texture.pkl")))
        bispec.update(_load(os.path.join(args.texture_dir, f"{split}_bispec_dict.pkl")))
        lattice.update(_load(os.path.join(args.texture_dir, f"{split}_prim_lat.pkl")))

        print(f"  combined entries: {len(xrd)}")
        _save(xrd, args.output_dir, f"{split}_sim_xrd_combined.pkl")
        _save(bispec, args.output_dir, f"{split}_bispec_combined.pkl")
        _save(lattice, args.output_dir, f"{split}_prim_lat_combined.pkl")


def run_peaklist_combine(args):
    """Peak-list counterpart of run_combine: merges the {split}_peaklist.pkl
    outputs of separate caglioti4x/strain/texture peaklist_from_xrd runs
    (disjoint key namespaces: {mpid}_{0..3}, {mpid}_strain_{1..n},
    {mpid}_texture_{1..n}) into one combined peak-list pkl per split, keyed
    to align with the existing combined bispec/prim_lat targets (e.g.
    mpfull_aug_cag_strain_texture/train_bispec_combined.pkl) via align_dicts
    in train.py -- no bispec/prim_lat re-merge needed since those targets
    are representation-agnostic and already exist. Only 'train' has all
    three augmented sources in production; val/test use a separate
    unaugmented single-mode set generated directly via peaklist_from_xrd on
    the existing val_sim_xrd.pkl/test_sim_xrd.pkl (see peaklist_from_xrd),
    not this combine step."""
    in_name = "peaklist_bgmix" if args.bg_noise_cnrs_mix_prob is not None else "peaklist"
    out_name = f"{in_name}_combined"
    for split in args.splits:
        print(f"\n=== {split.upper()} === (reading {{split}}_{in_name}.pkl)")
        peaks = {}
        peaks.update(_load(os.path.join(args.caglioti_dir, f"{split}_{in_name}.pkl")))
        peaks.update(_load(os.path.join(args.strain_dir, f"{split}_{in_name}.pkl")))
        peaks.update(_load(os.path.join(args.texture_dir, f"{split}_{in_name}.pkl")))
        print(f"  combined entries: {len(peaks)}")
        _save(peaks, args.output_dir, f"{split}_{out_name}.pkl")


# ---------------------------------------------------------------------------
# Shared I/O helpers
# ---------------------------------------------------------------------------
def _load(path):
    print(f"  loading {path} ...")
    with open(path, "rb") as f:
        return pickle.load(f)



def _load_struct_data(args, split):
    path = os.path.join(args.struct_dir, f"{split}_data.pkl")
    print(f"\n=== {split.upper()} === ({path})")
    with open(path, "rb") as f:
        data = pickle.load(f)
    if args.limit:
        data = {k: data[k] for k in list(data.keys())[: args.limit]}
    return data


def _load_bispec_src(args, split, required=True):
    path = os.path.join(args.bispec_dir, args.bispec_file.format(split=split))
    if not os.path.exists(path):
        if required:
            raise FileNotFoundError(f"bispec file not found: {path}")
        print(f"  [WARN] bispec file not found: {path} -- skipping bispec output")
        return {}
    print(f"  Loading bispec from {path} ...")
    with open(path, "rb") as f:
        return pickle.load(f)


def _save(obj, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, "wb") as f:
        pickle.dump(obj, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"  Saved {len(obj)} entries -> {path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--augmentation", required=True,
                   choices=["strain", "texture", "caglioti", "caglioti4x", "single", "peaklist",
                            "peaklist_strain", "peaklist_from_xrd", "combine", "peaklist_combine"])
    p.add_argument("--dataset", choices=["mpfull", "mp20"], default="mpfull",
                   help="Selects defaults for --struct-dir/--bispec-dir/--output-dir; "
                        "explicit flags always override.")
    p.add_argument("--struct-dir", default=None,
                   help="Directory with {split}_data.pkl (mpid -> Structure/CIF). "
                        "Defaults per --dataset (see DEFAULT_STRUCT_DIR).")
    p.add_argument("--bispec-dir", default=None,
                   help="Directory with bispec pkl files (required for texture/caglioti). "
                        "Defaults per --dataset (see DEFAULT_BISPEC_DIR).")
    p.add_argument("--bispec-file", default="{split}_data_bispec_list.pkl",
                   help="Bispec filename pattern with {split} placeholder")
    p.add_argument("--output-dir", default=None,
                   help="Where to write output pkl files. Defaults to a fixed per-dataset, "
                        "per-augmentation directory (see DEFAULT_OUTPUT_DIR) where confirmed "
                        "against real data; otherwise required.")
    p.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--limit", type=int, default=None, help="(debug) only process first N structures per split")
    p.add_argument("--seed", type=int, default=42)

    # strain-only
    p.add_argument("--n-strains", type=int, default=3)
    p.add_argument("--max-strain", type=float, default=0.04)
    p.add_argument("--cob-path", default=COB_DEFAULT)

    # texture-only
    p.add_argument("--n-textures", type=int, default=3)
    p.add_argument("--max-texture", type=float, default=0.5)

    # single/peaklist-only
    p.add_argument("--symprec", type=float, default=None,
                   help="Override SINGLE_SYMPREC's per-dataset default (mpfull=0, mp20=1e-3). "
                        "Only matters if you want a fresh, self-consistent run rather than "
                        "reproducing the existing mpfull_crystalyze_cag/mp20_crystalyze_cag data.")

    # peaklist/peaklist_strain/peaklist_from_xrd-only
    p.add_argument("--peaklist-max-peaks", type=int, default=30,
                   help="Keep at most this many peaks per pattern, ranked by find_peaks prominence "
                        "(matches model.py's max_peaks/train.py's pad_peaks_to_fixed_length default).")

    # peaklist_from_xrd-only
    p.add_argument("--xrd-pkl-dir", default=None,
                   help="Directory containing an existing dense-XRD pkl to extract peaks from "
                        "(e.g. mpfull_texture_cag or mpfull_aug).")
    p.add_argument("--xrd-pkl-pattern", default="{split}_sim_xrd.pkl",
                   help="Filename pattern with a {split} placeholder, e.g. "
                        "'{split}_sim_xrd_texture.pkl' or '{split}_sim_xrd_aug.pkl'.")
    p.add_argument("--bg-noise-cnrs-mix-prob", type=float, default=None,
                   help="If set, add one Gamma background-noise realization (mirroring "
                        "bg_noise_aug.add_bg_noise's RRUFF/CNRS-mix sampling, but applied once "
                        "here to the raw dense curve before find_peaks, not per-batch at train "
                        "time) to each entry before peak extraction, and write to "
                        "'{split}_peaklist_bgmix.pkl' instead of '{split}_peaklist.pkl' -- same "
                        "entry count/keys either way, just noisier peak sets. 0.0 = pure RRUFF "
                        "profile, 1.0 = pure CNRS profile, default None = disabled (unchanged "
                        "behavior, no noise added).")
    p.add_argument("--bg-noise-seed", type=int, default=0)

    # caglioti4x-only
    p.add_argument("--intens-gamma-a", type=float, default=INTENS_GAMMA_A_DEFAULT)
    p.add_argument("--intens-gamma-scale", type=float, default=INTENS_GAMMA_SCALE_DEFAULT,
                   help="Set to 0 to disable intensity perturbation.")

    # combine-only: merges separate caglioti/strain/texture runs into one
    # training set (matches e.g. mp20_aug_cag_strain_texture). Each defaults
    # to DEFAULT_OUTPUT_DIR[--dataset][<type>] if not given explicitly.
    p.add_argument("--caglioti-dir", default=None)
    p.add_argument("--strain-dir", default=None)
    p.add_argument("--texture-dir", default=None)
    return p


RUNNERS = {
    "strain": run_strain,
    "texture": run_texture,
    "caglioti": run_caglioti,
    "caglioti4x": run_caglioti4x,
    "single": run_single,
    "peaklist": run_peaklist,
    "peaklist_strain": run_peaklist_strain,
    "peaklist_from_xrd": run_peaklist_from_xrd,
    "combine": run_combine,
    "peaklist_combine": run_peaklist_combine,
}
REQUIRES_BISPEC_DIR = {"texture", "caglioti"}

_BASE = os.environ.get("POWDERXRD_DATA_ROOT", "./data")

# All confirmed by exact match against real, already-generated data (see
# tests/test_generate_augmented_data.py) except where noted.
DEFAULT_STRUCT_DIR = {
    "mpfull": _BASE,
    "mp20": f"{_BASE}/mp20/data",
}
DEFAULT_BISPEC_DIR = DEFAULT_STRUCT_DIR  # {split}_data_bispec_list.pkl lives alongside {split}_data.pkl for both
DEFAULT_OUTPUT_DIR = {
    "mpfull": {
        "strain": f"{_BASE}/mpfull_strain",
        "texture": f"{_BASE}/mpfull_texture_cag",
        "caglioti4x": f"{_BASE}/mpfull_aug",
        "single": f"{_BASE}/mpfull_crystalyze_cag",
        "peaklist": f"{_BASE}/mpfull_crystalyze_cag",  # same dir as `single` -- shares its bispec targets
        "peaklist_strain": f"{_BASE}/mpfull_strain",  # same dir as `strain` -- reuses its bispec_dict/prim_lat
        "peaklist_combine": f"{_BASE}/mpfull_aug_cag_strain_texture",  # same dir as the dense `combine` output
        # caglioti (data_gen_full_cag.py) never had a default; stays required
        # even though mpfull_cag/ is confirmed correct, to preserve original
        # behavior (an explicit choice, not an oversight).
    },
    "mp20": {
        "strain": f"{_BASE}/mp20_strain",
        "texture": f"{_BASE}/mp20_texture_cag",
        "caglioti": f"{_BASE}/mp20_cag",
        "single": f"{_BASE}/mp20_crystalyze_cag",
        "peaklist": f"{_BASE}/mp20_crystalyze_cag",  # same dir as `single` -- shares its bispec targets
        "peaklist_strain": f"{_BASE}/mp20_strain",  # same dir as `strain` -- reuses its bispec_dict/prim_lat
        # caglioti4x has no confirmed mp20 directory -- mp20_aug/, mp20aug/,
        # and mp20_crystalyze_aug/ all exist but none matches the expected
        # {split}_sim_xrd_aug.pkl naming; stays required rather than guessing.
    },
}


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.struct_dir is None:
        args.struct_dir = DEFAULT_STRUCT_DIR[args.dataset]
    if args.bispec_dir is None:
        args.bispec_dir = DEFAULT_BISPEC_DIR[args.dataset]

    if args.augmentation in ("combine", "peaklist_combine"):
        for name, attr in [("caglioti", "caglioti_dir"), ("strain", "strain_dir"), ("texture", "texture_dir")]:
            if getattr(args, attr) is None:
                default = DEFAULT_OUTPUT_DIR[args.dataset].get(name)
                if default is None:
                    parser.error(f"--{attr.replace('_', '-')} is required (no default for "
                                 f"--dataset {args.dataset} --augmentation {name})")
                setattr(args, attr, default)
    elif args.augmentation in REQUIRES_BISPEC_DIR and not args.bispec_dir:
        parser.error(f"--bispec-dir is required for --augmentation {args.augmentation}")

    if args.output_dir is None:
        args.output_dir = DEFAULT_OUTPUT_DIR[args.dataset].get(args.augmentation)
        if args.output_dir is None:
            parser.error(f"--output-dir is required for --dataset {args.dataset} "
                         f"--augmentation {args.augmentation}")

    RUNNERS[args.augmentation](args)


if __name__ == "__main__":
    main()
