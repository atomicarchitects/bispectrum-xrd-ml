"""
Compute true bispectra for the opXRD CNRS patterns from their raw structure
JSONs (lattice + basis, reduced to primitive cell) -- bispectrum depends only
on the reciprocal lattice, not atomic positions/species, so this only needs
the lattice + a primitive-cell reduction (same principle as
compute_rruff_true_bispec.py's RRUFF/CIF pipeline).

Output: opxrd_cnrs_true_bispec.pkl -> {CNRS_pattern_id: np.array (10, 35)}

Usage:
    python compute_cnrs_true_bispec.py
"""
import json
import os
import pickle
import re
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "mp_full"))
from pymatgen.core import Lattice, Structure
from utilities import get_bispectrum, get_neighbors
from generate_augmented_data import BISPEC_WAVELENGTH, BISPEC_LMAX, BISPEC_RADIAL, BISPEC_N_BASIS

# ── paths ──────────────────────────────────────────────────────────────────
_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
CNRS_DIR   = f"{_ROOT}/opxrd_cnrs_eval"
RAW_DIR    = f"{CNRS_DIR}/CNRS"
XRD_PKL    = f"{CNRS_DIR}/opxrd_cnrs_xrd.pkl"
COB_PATH   = f"{_ROOT}/cob_lmax6.pkl"
OUT_PATH   = f"{_ROOT}/eval_results/opxrd_cnrs/opxrd_cnrs_true_bispec.pkl"

_ELEMENT_RE = re.compile(r"^[A-Za-z]+")

cob_bi = torch.load(COB_PATH, map_location="cpu", weights_only=False)
print("cob_bi shape:", cob_bi.shape)


def build_structure(lattice_str, basis_raw):
    a, b, c, al, be, ga = (float(x) for x in lattice_str.strip("()").split(","))
    lattice = Lattice.from_parameters(a, b, c, al, be, ga)
    basis_list = json.loads(basis_raw) if isinstance(basis_raw, str) else basis_raw
    sites = [json.loads(s) if isinstance(s, str) else s for s in basis_list]
    species = [_ELEMENT_RE.match(s["symbol"]).group(0) for s in sites]
    coords = [[s["x"], s["y"], s["z"]] for s in sites]
    return Structure(lattice, species, coords)


observed = pickle.load(open(XRD_PKL, "rb"))
k_max = 2.0 / BISPEC_WAVELENGTH

true_bispecs = {}
failed = 0
for pid in observed.keys():
    fname = pid.replace("CNRS_", "") + ".json"
    fpath = os.path.join(RAW_DIR, fname)
    if not os.path.exists(fpath):
        failed += 1
        continue
    with open(fpath) as f:
        d = json.load(f)
    label = json.loads(d["label"])
    phases = label.get("phases") or []
    if not phases:
        failed += 1
        continue
    phase0 = json.loads(phases[0])
    lattice_str = phase0.get("lattice")
    basis_raw = phase0.get("basis")
    if not lattice_str or not basis_raw:
        failed += 1
        continue
    try:
        structure = build_structure(lattice_str, basis_raw)
        prim = structure.get_primitive_structure()
        recip = prim.lattice.reciprocal_lattice_crystallographic.matrix
        miller = get_neighbors(recip, k_max)
        neighs = torch.tensor(miller.reshape(-1, 3) @ recip, dtype=torch.float64)
        bispec = get_bispectrum(neighs, k_max, BISPEC_LMAX, BISPEC_RADIAL, BISPEC_N_BASIS, cob_bi)
        true_bispecs[pid] = bispec.numpy()
    except Exception:
        failed += 1
        continue

print(f"Computed {len(true_bispecs)}/{len(observed)} true bispectra ({failed} failed)")
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
with open(OUT_PATH, "wb") as f:
    pickle.dump(true_bispecs, f)
print(f"Saved -> {OUT_PATH}")
