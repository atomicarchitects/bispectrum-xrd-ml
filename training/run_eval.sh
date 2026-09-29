#!/bin/bash

POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"

#MODEL_DIR=$POWDERXRD_DATA_ROOT/final_saved_models/mp20_correct_cag
#OUT_DIR=$POWDERXRD_DATA_ROOT/final_test_results/mp20_correct_cag
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

#mkdir -p "$OUT_DIR"

# Bispectrum model
#python "$SCRIPT_DIR/eval.py" \
#    --model_path "$MODEL_DIR/best_model_mp20_correct_cag_bispec.pt" \
#    --data_config "$SCRIPT_DIR/conf/data/mp20_crystalyze_cag.yaml" \
#    --model_config "$SCRIPT_DIR/conf/model/transformerbispec.yaml" \
#    --train_config "$SCRIPT_DIR/conf/train/default.yaml" \
#    --prediction_type bispectrum \
#    --test_bispec_file test_data_bispec_list.pkl \
#    --output_path "$OUT_DIR/bispec_test_results.pkl"

# Direct model
#python "$SCRIPT_DIR/eval.py" \
#    --model_path "$MODEL_DIR/best_model_mp20_correct_cag_direct.pt" \
#    --data_config "$SCRIPT_DIR/conf/data/mp20_crystalyze_cag_direct.yaml" \
#    --model_config "$SCRIPT_DIR/conf/model/transformerdirect.yaml" \
#    --train_config "$SCRIPT_DIR/conf/train/default_direct.yaml" \
#    --prediction_type direct \
#    --test_bispec_file test_data_prim_lat.pkl \
#    --output_path "$OUT_DIR/direct_test_results.pkl"

# ── mp20_final_aug ──────────────────────────────────────────────────────────
#AUG_MODEL_DIR=$POWDERXRD_DATA_ROOT/final_saved_models/mp20_final_aug
#AUG_OUT_DIR=$POWDERXRD_DATA_ROOT/final_test_results/mp20_final_aug
#mkdir -p "$AUG_OUT_DIR"

# Bispectrum model
#python "$SCRIPT_DIR/eval.py" \
#    --model_path "$AUG_MODEL_DIR/best_model_mp20_aug_final_bispec.pt" \
#    --data_config "$SCRIPT_DIR/conf/data/mp20_aug_final.yaml" \
#    --model_config "$SCRIPT_DIR/conf/model/transformerbispec.yaml" \
#    --train_config "$SCRIPT_DIR/conf/train/default.yaml" \
#    --prediction_type bispectrum \
#    --test_bispec_file test_data_bispec_list.pkl \
#    --output_path "$AUG_OUT_DIR/bispec_test_results.pkl"

# Direct model
#python "$SCRIPT_DIR/eval.py" \
#    --model_path "$AUG_MODEL_DIR/best_model_mp20_aug_final_direct.pt" \
#    --data_config "$SCRIPT_DIR/conf/data/mp20_aug_final_direct.yaml" \
#    --model_config "$SCRIPT_DIR/conf/model/transformerdirect.yaml" \
#    --train_config "$SCRIPT_DIR/conf/train/default_direct.yaml" \
#    --prediction_type direct \
#    --test_bispec_file test_data_prim_lat.pkl \
#    --output_path "$AUG_OUT_DIR/direct_test_results.pkl"

# ── mpfull_aug ──────────────────────────────────────────────────────────────
MPFULL_MODEL_DIR=${POWDERXRD_DATA_ROOT}/final_saved_models/mpfull_crystalyze_cag
MPFULL_OUT_DIR=${POWDERXRD_DATA_ROOT}/final_test_results/mpfull_crystalyze_cag
mkdir -p "$MPFULL_OUT_DIR"

# Bispectrum model
python "$SCRIPT_DIR/eval.py" \
    --model_path "$MPFULL_MODEL_DIR/mpfull_crystalyze_cag_bispec_best_model.pt" \
    --data_config "$SCRIPT_DIR/conf/data/mpfull_crystalyze_cag_nonoise.yaml" \
    --model_config "$SCRIPT_DIR/conf/model/transformerbispec.yaml" \
    --train_config "$SCRIPT_DIR/conf/train/default.yaml" \
    --prediction_type bispectrum \
    --test_bispec_file test_data_bispec_list.pkl \
    --output_path "$MPFULL_OUT_DIR/bispec_test_results.pkl"

# Direct model
#python "$SCRIPT_DIR/eval.py" \
#    --model_path "$MPFULL_MODEL_DIR/mpfull_crystalyze_cag_direct_best_model.pt" \
#    --data_config "$SCRIPT_DIR/conf/data/mpfull_crystalyze_cag_direct.yaml" \
#    --model_config "$SCRIPT_DIR/conf/model/transformerdirect.yaml" \
#    --train_config "$SCRIPT_DIR/conf/train/default_direct.yaml" \
#    --prediction_type direct \
#    --test_bispec_file test_data_prim_lat.pkl \
#    --output_path "$MPFULL_OUT_DIR/direct_test_results.pkl"
