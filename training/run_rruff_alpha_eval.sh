#!/bin/bash
POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

OPXRD_XRD=${POWDERXRD_DATA_ROOT}/opxrd_cnrs_eval/opxrd_cnrs_xrd.pkl
OPXRD_LAT=${POWDERXRD_DATA_ROOT}/opxrd_cnrs_eval/opxrd_cnrs_lat.pkl

# ── mp20_correct_cag ────────────────────────────────────────────────────────
#MODEL_DIR=${POWDERXRD_DATA_ROOT}/final_saved_models/mp20_correct_cag
#OUT_DIR=${POWDERXRD_DATA_ROOT}/final_test_results/mp20_correct_cag
#mkdir -p "$OUT_DIR"

# Bispectrum model
#python "$SCRIPT_DIR/eval_rruff_alpha.py" \
#    --model_path "$MODEL_DIR/best_model_mp20_correct_cag_bispec.pt" \
#    --train_data_dir ${POWDERXRD_DATA_ROOT}/mp20_crystalyze_cag \
#    --train_xrd_file train_sim_xrd.pkl \
#    --train_lat_file train_data_bispec_list.pkl \
#    --prediction_type bispectrum \
#    --model_config "$SCRIPT_DIR/conf/model/transformerbispec.yaml" \
#    --train_config "$SCRIPT_DIR/conf/train/default.yaml" \
#    --output_path "$OUT_DIR/rruff_alpha_bispec_results.pkl"

# Direct model
#python "$SCRIPT_DIR/eval_rruff_alpha.py" \
#    --model_path "$MODEL_DIR/best_model_mp20_correct_cag_direct.pt" \
#    --train_data_dir ${POWDERXRD_DATA_ROOT}/mp20_crystalyze_cag_direct \
#    --train_xrd_file train_sim_xrd.pkl \
#    --train_lat_file train_data_prim_lat.pkl \
#    --prediction_type direct \
#    --model_config "$SCRIPT_DIR/conf/model/transformerdirect.yaml" \
#    --train_config "$SCRIPT_DIR/conf/train/default_direct.yaml" \
#    --output_path "$OUT_DIR/rruff_alpha_direct_results.pkl"

# ── mpfull_crystalyze_cag ───────────────────────────────────────────────────
#MPCAG_MODEL_DIR=${POWDERXRD_DATA_ROOT}/final_saved_models/mpfull_crystalyze_cag
#MPCAG_OUT_DIR=${POWDERXRD_DATA_ROOT}/final_test_results/mpfull_crystalyze_cag
#mkdir -p "$MPCAG_OUT_DIR"

# RRUFF alpha
#python "$SCRIPT_DIR/eval_rruff_alpha.py" \
#    --model_path "$MPCAG_MODEL_DIR/mpfull_crystalyze_cag_bispec_best_model.pt" \
#    --train_data_dir ${POWDERXRD_DATA_ROOT}/mpfull_crystalyze_cag \
#    --train_xrd_file train_sim_xrd.pkl \
#    --train_lat_file train_data_bispec_list.pkl \
#    --prediction_type bispectrum \
#    --model_config "$SCRIPT_DIR/conf/model/transformerbispec.yaml" \
#    --train_config "$SCRIPT_DIR/conf/train/default.yaml" \
#    --output_path "$MPCAG_OUT_DIR/rruff_alpha_bispec_results.pkl"

# opXRD CNRS
#python "$SCRIPT_DIR/eval_rruff_alpha.py" \
#    --model_path "$MPCAG_MODEL_DIR/mpfull_crystalyze_cag_bispec_best_model.pt" \
#    --train_data_dir ${POWDERXRD_DATA_ROOT}/mpfull_crystalyze_cag \
#    --train_xrd_file train_sim_xrd.pkl \
#    --train_lat_file train_data_bispec_list.pkl \
#    --prediction_type bispectrum \
#    --model_config "$SCRIPT_DIR/conf/model/transformerbispec.yaml" \
#    --train_config "$SCRIPT_DIR/conf/train/default.yaml" \
#    --eval_xrd_pkl "$OPXRD_XRD" \
#    --eval_lat_pkl "$OPXRD_LAT" \
#    --output_path "$MPCAG_OUT_DIR/opxrd_cnrs_bispec_results.pkl"

# ── mpfull_aug ──────────────────────────────────────────────────────────────
#MPFULL_MODEL_DIR=${POWDERXRD_DATA_ROOT}/final_saved_models/mpfull_aug
#MPFULL_OUT_DIR=${POWDERXRD_DATA_ROOT}/final_test_results/mpfull_aug
#mkdir -p "$MPFULL_OUT_DIR"

# Bispectrum model
#python "$SCRIPT_DIR/eval_rruff_alpha.py" \
#    --model_path "$MPFULL_MODEL_DIR/mpfull_aug_bispec_best_model.pt" \
#    --train_data_dir ${POWDERXRD_DATA_ROOT}/mpfull_aug_cag_strain_texture \
#    --train_xrd_file train_sim_xrd_combined.pkl \
#    --train_lat_file train_bispec_combined.pkl \
#    --prediction_type bispectrum \
#    --model_config "$SCRIPT_DIR/conf/model/transformerbispec.yaml" \
#    --train_config "$SCRIPT_DIR/conf/train/default.yaml" \
#    --output_path "$MPFULL_OUT_DIR/rruff_alpha_bispec_results.pkl"

# Direct model
#python "$SCRIPT_DIR/eval_rruff_alpha.py" \
#    --model_path "$MPFULL_MODEL_DIR/mpfull_aug_direct_best_model.pt" \
#    --train_data_dir ${POWDERXRD_DATA_ROOT}/mpfull_aug_cag_strain_texture \
#    --train_xrd_file train_sim_xrd_combined.pkl \
#    --train_lat_file train_prim_lat_combined.pkl \
#    --prediction_type direct \
#    --model_config "$SCRIPT_DIR/conf/model/transformerdirect.yaml" \
#    --train_config "$SCRIPT_DIR/conf/train/default_direct.yaml" \
#    --output_path "$MPFULL_OUT_DIR/rruff_alpha_direct_results.pkl"

# ── mpfull_aug_medium ───────────────────────────────────────────────────────
MPFULL_MED_MODEL=${POWDERXRD_DATA_ROOT}/final_saved_models/mpfull_aug/mpfull_aug_bispec_best_model_medium.pt
MPFULL_MED_TRAIN=${POWDERXRD_DATA_ROOT}/mpfull_aug_cag_strain_texture
MPFULL_MED_OUT=${POWDERXRD_DATA_ROOT}/final_test_results/mpfull_aug_medium
mkdir -p "$MPFULL_MED_OUT"

# RRUFF alpha
python "$SCRIPT_DIR/eval_rruff_alpha.py" \
    --model_path "$MPFULL_MED_MODEL" \
    --train_data_dir "$MPFULL_MED_TRAIN" \
    --train_xrd_file train_sim_xrd_combined.pkl \
    --train_lat_file train_bispec_combined.pkl \
    --prediction_type bispectrum \
    --model_config "$SCRIPT_DIR/conf/model/transformerbispec_medium.yaml" \
    --train_config "$SCRIPT_DIR/conf/train/default.yaml" \
    --output_path "$MPFULL_MED_OUT/rruff_alpha_bispec_results.pkl"

# opXRD CNRS
python "$SCRIPT_DIR/eval_rruff_alpha.py" \
    --model_path "$MPFULL_MED_MODEL" \
    --train_data_dir "$MPFULL_MED_TRAIN" \
    --train_xrd_file train_sim_xrd_combined.pkl \
    --train_lat_file train_bispec_combined.pkl \
    --prediction_type bispectrum \
    --model_config "$SCRIPT_DIR/conf/model/transformerbispec_medium.yaml" \
    --train_config "$SCRIPT_DIR/conf/train/default.yaml" \
    --eval_xrd_pkl "$OPXRD_XRD" \
    --eval_lat_pkl "$OPXRD_LAT" \
    --output_path "$MPFULL_MED_OUT/opxrd_cnrs_bispec_results.pkl"

# ── mp20_final_aug ──────────────────────────────────────────────────────────
#MP20AUG_MODEL_DIR=${POWDERXRD_DATA_ROOT}/final_saved_models/mp20_final_aug
#MP20AUG_TRAIN=${POWDERXRD_DATA_ROOT}/mp20_aug_cag_strain_texture
#MP20AUG_OUT=${POWDERXRD_DATA_ROOT}/final_test_results/mp20_final_aug
#mkdir -p "$MP20AUG_OUT"

# RRUFF alpha
#python "$SCRIPT_DIR/eval_rruff_alpha.py" \
#    --model_path "$MP20AUG_MODEL_DIR/best_model_mp20_aug_final_bispec.pt" \
#    --train_data_dir "$MP20AUG_TRAIN" \
#    --train_xrd_file train_sim_xrd_combined.pkl \
#    --train_lat_file train_bispec_combined.pkl \
#    --prediction_type bispectrum \
#    --model_config "$SCRIPT_DIR/conf/model/transformerbispec.yaml" \
#    --train_config "$SCRIPT_DIR/conf/train/default.yaml" \
#    --output_path "$MP20AUG_OUT/rruff_alpha_bispec_results.pkl"

# opXRD CNRS
#python "$SCRIPT_DIR/eval_rruff_alpha.py" \
#    --model_path "$MP20AUG_MODEL_DIR/best_model_mp20_aug_final_bispec.pt" \
#    --train_data_dir "$MP20AUG_TRAIN" \
#    --train_xrd_file train_sim_xrd_combined.pkl \
#    --train_lat_file train_bispec_combined.pkl \
#    --prediction_type bispectrum \
#    --model_config "$SCRIPT_DIR/conf/model/transformerbispec.yaml" \
#    --train_config "$SCRIPT_DIR/conf/train/default.yaml" \
#    --eval_xrd_pkl "$OPXRD_XRD" \
#    --eval_lat_pkl "$OPXRD_LAT" \
#    --output_path "$MP20AUG_OUT/opxrd_cnrs_bispec_results.pkl"
