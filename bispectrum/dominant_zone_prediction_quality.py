"""
Compare bispec inversion vs direct prediction quality on experimentally
observed dominant zone structures.

Requires:
  $POWDERXRD_DATA_ROOT/mpfull/dominant_zones.csv  (with 'theoretical' column from fetch_theoretical_flag.py)
  $POWDERXRD_DATA_ROOT/final_test_results_cctbx/mpfull_aug/inversion_results_cctbx_{bispec,direct}.pkl
"""

import os

import numpy as np
import pandas as pd
import pickle
from pymatgen.core import Lattice

ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
DATA_DIR  = os.path.join(ROOT, "mpfull")
BISPEC_PKL = os.path.join(ROOT, "final_test_results_cctbx/mpfull_aug/inversion_results_cctbx_bispec.pkl")
DIRECT_PKL = os.path.join(ROOT, "final_test_results_cctbx/mpfull_aug/inversion_results_cctbx_direct.pkl")

DZ_THRESHOLD = 0.5  # must match filter_dominant_zones.py --threshold

# --- Load dominant zones (experimental + dominant-zone filtered) ---
dz = pd.read_csv(f"{DATA_DIR}/dominant_zones.csv")
dz = dz[dz["dominant_zone_frac"] >= DZ_THRESHOLD].copy()
print(f"Dominant zone structures (frac >= {DZ_THRESHOLD}): {len(dz)}")
if "theoretical" not in dz.columns:
    print("WARNING: 'theoretical' column missing — run fetch_theoretical_flag.py first.")
    print("Proceeding with all dominant zone structures.\n")
    exp_ids = set(dz["material_id"])
else:
    exp_ids = set(dz[dz["theoretical"] == False]["material_id"])
    print(f"Experimentally observed dominant zone structures: {len(exp_ids)}")

# --- Load prediction results ---
with open(BISPEC_PKL, "rb") as f:
    df_bispec = pickle.load(f)
with open(DIRECT_PKL, "rb") as f:
    df_direct = pickle.load(f)

# Filter to experimental dominant zone structures
df_bispec_dz = df_bispec[df_bispec["mat_id"].isin(exp_ids)].copy()
df_direct_dz = df_direct[df_direct["mat_id"].isin(exp_ids)].copy()

print(f"Bispec results available for {len(df_bispec_dz)} / {len(exp_ids)} structures")
print(f"Direct results available for {len(df_direct_dz)} / {len(exp_ids)} structures")

# Common set with both predictions
common_ids = set(df_bispec_dz["mat_id"]) & set(df_direct_dz["mat_id"])
print(f"Structures with both predictions: {len(common_ids)}\n")

df_bispec_dz = df_bispec_dz[df_bispec_dz["mat_id"].isin(common_ids)]
df_direct_dz = df_direct_dz[df_direct_dz["mat_id"].isin(common_ids)]

# --- MAPE function (matches final_compare_inversion_direct_pred.ipynb) ---
param_names = ["a", "b", "c", "alpha", "beta", "gamma"]

def lattice_parameter_mape(true_lat, pred_lat):
    true_params = np.array([true_lat.a, true_lat.b, true_lat.c,
                            true_lat.alpha, true_lat.beta, true_lat.gamma])
    pred_params = np.array([pred_lat.a, pred_lat.b, pred_lat.c,
                            pred_lat.alpha, pred_lat.beta, pred_lat.gamma])
    return np.abs((pred_params - true_params) / true_params) * 100

def compute_mapes(df):
    return np.array([
        lattice_parameter_mape(t, p)
        for t, p in zip(df["cctbx_true"], df["cctbx_pred"])
    ])

bispec_mapes = compute_mapes(df_bispec_dz)
direct_mapes = compute_mapes(df_direct_dz)

# --- Overall summary ---
print("=" * 60)
print("OVERALL (experimental dominant zones)")
print("=" * 60)
print(f"{'Param':<8} {'Bispec mean%':>13} {'Direct mean%':>13} {'Bispec median%':>15} {'Direct median%':>15}")
print("-" * 65)
for i, p in enumerate(param_names):
    print(f"{p:<8} {bispec_mapes[:,i].mean():>13.2f} {direct_mapes[:,i].mean():>13.2f}"
          f" {np.median(bispec_mapes[:,i]):>15.2f} {np.median(direct_mapes[:,i]):>15.2f}")
print()
print(f"Overall length MAPE (a,b,c mean):")
print(f"  Bispec: {bispec_mapes[:,:3].mean():.2f}%   Direct: {direct_mapes[:,:3].mean():.2f}%")
print(f"Overall angle MAPE (alpha,beta,gamma mean):")
print(f"  Bispec: {bispec_mapes[:,3:].mean():.2f}%   Direct: {direct_mapes[:,3:].mean():.2f}%")

# Bravais type accuracy
bispec_correct = (df_bispec_dz["true_bravais"].values == df_bispec_dz["pred_bravais"].values).mean()
direct_correct = (df_direct_dz["true_bravais"].values == df_direct_dz["pred_bravais"].values).mean()
print(f"\nBravais type accuracy:")
print(f"  Bispec: {100*bispec_correct:.1f}%   Direct: {100*direct_correct:.1f}%")

# --- Per-Bravais breakdown ---
# Merge true_bravais into bispec (already there), use for grouping
print("\n" + "=" * 60)
print("PER BRAVAIS TYPE — length MAPE (a,b,c) mean%")
print("=" * 60)

bravais_types = sorted(df_bispec_dz["true_bravais"].unique())
rows = []
for bv in bravais_types:
    b_sub = df_bispec_dz[df_bispec_dz["true_bravais"] == bv]
    d_sub = df_direct_dz[df_direct_dz["true_bravais"] == bv]
    if len(b_sub) == 0 or len(d_sub) == 0:
        continue
    b_mape = compute_mapes(b_sub)[:, :3].mean()
    d_mape = compute_mapes(d_sub)[:, :3].mean()
    rows.append({"bravais": bv, "n": len(b_sub), "bispec_len_mape": b_mape, "direct_len_mape": d_mape})

df_bv = pd.DataFrame(rows).sort_values("bispec_len_mape", ascending=False)
print(df_bv.to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# --- Per zone_type breakdown ---
if "zone_type" in dz.columns:
    dz_exp = dz[dz["material_id"].isin(common_ids)] if "theoretical" not in dz.columns \
        else dz[(dz["material_id"].isin(common_ids)) & (dz["theoretical"] == False)]
    bispec_dz_merged = df_bispec_dz.merge(dz_exp[["material_id", "zone_type"]], left_on="mat_id", right_on="material_id", how="left")
    direct_dz_merged = df_direct_dz.merge(dz_exp[["material_id", "zone_type"]], left_on="mat_id", right_on="material_id", how="left")

    print("\n" + "=" * 60)
    print("PER ZONE TYPE — length MAPE (a,b,c) mean%")
    print("=" * 60)
    zone_rows = []
    for zt in sorted(bispec_dz_merged["zone_type"].dropna().unique()):
        b_sub = df_bispec_dz[bispec_dz_merged["zone_type"].values == zt]
        d_sub = df_direct_dz[direct_dz_merged["zone_type"].values == zt]
        if len(b_sub) == 0 or len(d_sub) == 0:
            continue
        zone_rows.append({
            "zone_type": zt, "n": len(b_sub),
            "bispec_len_mape": compute_mapes(b_sub)[:, :3].mean(),
            "direct_len_mape": compute_mapes(d_sub)[:, :3].mean(),
        })
    print(pd.DataFrame(zone_rows).to_string(index=False, float_format=lambda x: f"{x:.2f}"))

# Save
out = df_bispec_dz[["mat_id", "true_bravais", "pred_bravais", "cctbx_true", "cctbx_pred"]].rename(
    columns={"pred_bravais": "bispec_pred_bravais", "cctbx_pred": "cctbx_pred_bispec"}
).merge(
    df_direct_dz[["mat_id", "pred_bravais", "cctbx_pred"]].rename(
        columns={"pred_bravais": "direct_pred_bravais", "cctbx_pred": "cctbx_pred_direct"}
    ),
    on="mat_id"
)
out["bispec_len_mape"] = bispec_mapes[:, :3].mean(axis=1)
out["direct_len_mape"] = direct_mapes[:, :3].mean(axis=1)
out["bispec_angle_mape"] = bispec_mapes[:, 3:].mean(axis=1)
out["direct_angle_mape"] = direct_mapes[:, 3:].mean(axis=1)

# Extract reduced lattice parameters from Lattice objects
for param in ["a", "b", "c", "alpha", "beta", "gamma"]:
    out[f"true_{param}"] = out["cctbx_true"].apply(lambda lat: getattr(lat, param))
    out[f"bispec_pred_{param}"] = out["cctbx_pred_bispec"].apply(lambda lat: getattr(lat, param))
    out[f"direct_pred_{param}"] = out["cctbx_pred_direct"].apply(lambda lat: getattr(lat, param))
out = out.drop(columns=["cctbx_true", "cctbx_pred_bispec", "cctbx_pred_direct"])

metric_cols = ["material_id", "dominant_zone_frac", "dominant_zero_index", "n_peaks",
               "frac_h0", "frac_k0", "frac_l0",
               "anisotropy", "zone_type", "short_axis", "long_axis",
               "cv", "z_short", "z_long", "short_dev", "long_dev"]
available = [c for c in metric_cols if c in dz.columns]
out = out.merge(dz[available], left_on="mat_id", right_on="material_id", how="left").drop(columns="material_id")
sort_col = "dominant_zone_frac" if "dominant_zone_frac" in out.columns else "bispec_len_mape"
out = out.sort_values(sort_col, ascending=False)

out_path = f"{DATA_DIR}/dominant_zones_prediction_quality.csv"
out.to_csv(out_path, index=False)
print(f"\nSaved per-structure results to {out_path}")
