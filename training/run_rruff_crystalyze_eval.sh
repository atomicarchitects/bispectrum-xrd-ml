#!/bin/bash
POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── mp20_correct_cag ────────────────────────────────────────────────────────
CAG_MODEL_DIR=${POWDERXRD_DATA_ROOT}/final_saved_models/mp20_correct_cag
CAG_OUT_DIR=${POWDERXRD_DATA_ROOT}/final_test_results/mp20_correct_cag
mkdir -p "$CAG_OUT_DIR"

python eval_rruff_crystalyze.py \
    --model_path "$CAG_MODEL_DIR/best_model_mp20_correct_cag_bispec.pt" \
    --train_data_dir ${POWDERXRD_DATA_ROOT}/mp20_crystalyze_cag \
    --train_xrd_file train_sim_xrd.pkl \
    --train_lat_file train_data_bispec_list.pkl \
    --prediction_type bispectrum \
    --model_config conf/model/transformerbispec.yaml \
    --train_config conf/train/default.yaml \
    --output_path "$CAG_OUT_DIR/rruff_crystalyze_bispec_results.pkl"

python eval_rruff_crystalyze.py \
    --model_path "$CAG_MODEL_DIR/best_model_mp20_correct_cag_direct.pt" \
    --train_data_dir ${POWDERXRD_DATA_ROOT}/mp20_crystalyze_cag_direct \
    --train_xrd_file train_sim_xrd.pkl \
    --train_lat_file train_data_prim_lat.pkl \
    --prediction_type direct \
    --model_config conf/model/transformerdirect.yaml \
    --train_config conf/train/default_direct.yaml \
    --output_path "$CAG_OUT_DIR/rruff_crystalyze_direct_results.pkl"

# ── mp20_final_aug ──────────────────────────────────────────────────────────
AUG_MODEL_DIR=${POWDERXRD_DATA_ROOT}/final_saved_models/mp20_final_aug
AUG_OUT_DIR=${POWDERXRD_DATA_ROOT}/final_test_results/mp20_final_aug
mkdir -p "$AUG_OUT_DIR"

python eval_rruff_crystalyze.py \
    --model_path "$AUG_MODEL_DIR/best_model_mp20_aug_final_bispec.pt" \
    --train_data_dir ${POWDERXRD_DATA_ROOT}/mp20_aug_cag_strain_texture \
    --train_xrd_file train_sim_xrd_combined.pkl \
    --train_lat_file train_bispec_combined.pkl \
    --prediction_type bispectrum \
    --model_config conf/model/transformerbispec.yaml \
    --train_config conf/train/default.yaml \
    --output_path "$AUG_OUT_DIR/rruff_crystalyze_bispec_results.pkl"

python eval_rruff_crystalyze.py \
    --model_path "$AUG_MODEL_DIR/best_model_mp20_aug_final_direct.pt" \
    --train_data_dir ${POWDERXRD_DATA_ROOT}/mp20_aug_cag_strain_texture \
    --train_xrd_file train_sim_xrd_combined.pkl \
    --train_lat_file train_prim_lat_combined.pkl \
    --prediction_type direct \
    --model_config conf/model/transformerdirect.yaml \
    --train_config conf/train/default_direct.yaml \
    --output_path "$AUG_OUT_DIR/rruff_crystalyze_direct_results.pkl"

# ── mpfull_crystalyze_cag ───────────────────────────────────────────────────
MPCAG_MODEL_DIR=${POWDERXRD_DATA_ROOT}/final_saved_models/mpfull_crystalyze_cag
MPCAG_OUT_DIR=${POWDERXRD_DATA_ROOT}/final_test_results/mpfull_crystalyze_cag
mkdir -p "$MPCAG_OUT_DIR"

python eval_rruff_crystalyze.py \
    --model_path "$MPCAG_MODEL_DIR/mpfull_crystalyze_cag_bispec_best_model.pt" \
    --train_data_dir ${POWDERXRD_DATA_ROOT}/mpfull_crystalyze_cag \
    --train_xrd_file train_sim_xrd.pkl \
    --train_lat_file train_data_bispec_list.pkl \
    --prediction_type bispectrum \
    --model_config conf/model/transformerbispec.yaml \
    --train_config conf/train/default.yaml \
    --output_path "$MPCAG_OUT_DIR/rruff_crystalyze_bispec_results.pkl"

# ── mpfull_aug_medium ───────────────────────────────────────────────────────
MPFULL_MED_OUT=${POWDERXRD_DATA_ROOT}/final_test_results/mpfull_aug_medium
mkdir -p "$MPFULL_MED_OUT"

python eval_rruff_crystalyze.py \
    --model_path ${POWDERXRD_DATA_ROOT}/final_saved_models/mpfull_aug/mpfull_aug_bispec_best_model_medium.pt \
    --train_data_dir ${POWDERXRD_DATA_ROOT}/mpfull_aug_cag_strain_texture \
    --train_xrd_file train_sim_xrd_combined.pkl \
    --train_lat_file train_bispec_combined.pkl \
    --prediction_type bispectrum \
    --model_config conf/model/transformerbispec_medium.yaml \
    --train_config conf/train/default.yaml \
    --output_path "$MPFULL_MED_OUT/rruff_crystalyze_bispec_results.pkl"
