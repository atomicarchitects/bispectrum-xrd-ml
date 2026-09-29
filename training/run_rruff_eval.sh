#!/bin/bash
POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"
# Run RRUFF evaluation for all available models in final_saved_models
# Usage: bash run_rruff_eval.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RRUFF_PKL=${POWDERXRD_DATA_ROOT}/rruff/rruff_data_dict.pkl
OUTPUT_DIR=${POWDERXRD_DATA_ROOT}/eval_results/rruff
MODELS_DIR=${POWDERXRD_DATA_ROOT}/final_saved_models

module load conda
set +u
source activate tda_env
set -u
cd "$SCRIPT_DIR"
mkdir -p "$OUTPUT_DIR"

# mp20 direct
CUDA_VISIBLE_DEVICES=0 python eval_rruff.py \
    --model_path "$MODELS_DIR/mp20/best_model_mp20_direct.pt" \
    --train_data_dir ${POWDERXRD_DATA_ROOT}/mp20/data \
    --train_xrd_file train_data_sim_xrd.pkl \
    --train_lat_file train_data_prim_lat.pkl \
    --rruff_pkl "$RRUFF_PKL" \
    --model_config conf/model/transformerdirect.yaml \
    --train_config conf/train/default_direct.yaml \
    --prediction_type direct \
    --output_path "$OUTPUT_DIR/rruff_mp20_direct.pkl" &

# mp20 bispec
CUDA_VISIBLE_DEVICES=1 python eval_rruff.py \
    --model_path ${POWDERXRD_DATA_ROOT}/mp20/transformerbispec/models/old_models/nlayers12_nhead8_dm256_hd512_lossL1_smallvalsFalse_masklossTrue_new.pt \
    --train_data_dir ${POWDERXRD_DATA_ROOT}/mp20/data \
    --train_xrd_file train_data_sim_xrd.pkl \
    --train_lat_file train_data_bispec_list.pkl \
    --rruff_pkl "$RRUFF_PKL" \
    --model_config conf/model/transformerbispec.yaml \
    --train_config conf/train/default.yaml \
    --prediction_type bispectrum \
    --output_path "$OUTPUT_DIR/rruff_mp20_bispec.pkl" &

# mp20aug bispec
CUDA_VISIBLE_DEVICES=2 python eval_rruff.py \
    --model_path "$MODELS_DIR/mp20_aug/best_model_mp20aug_bispec.pt" \
    --train_data_dir ${POWDERXRD_DATA_ROOT}/mp20_aug/all_aug \
    --train_xrd_file all_aug_train_data_sim_xrd.pkl \
    --train_lat_file all_aug_train_data_bispec.pkl \
    --rruff_pkl "$RRUFF_PKL" \
    --model_config conf/model/transformerbispec.yaml \
    --train_config conf/train/default.yaml \
    --prediction_type bispectrum \
    --output_path "$OUTPUT_DIR/rruff_mp20aug_bispec.pkl" &

# mp20aug direct
CUDA_VISIBLE_DEVICES=3 python eval_rruff.py \
    --model_path "$MODELS_DIR/mp20_aug/best_model_mp20aug_direct.pt" \
    --train_data_dir ${POWDERXRD_DATA_ROOT}/mp20_aug/all_aug \
    --train_xrd_file all_aug_train_data_sim_xrd.pkl \
    --train_lat_file all_aug_train_data_prim_lat.pkl \
    --rruff_pkl "$RRUFF_PKL" \
    --model_config conf/model/transformerdirect.yaml \
    --train_config conf/train/default_direct.yaml \
    --prediction_type direct \
    --output_path "$OUTPUT_DIR/rruff_mp20aug_direct.pkl" &

wait

# mpfull bispec
CUDA_VISIBLE_DEVICES=0 python eval_rruff.py \
    --model_path "$MODELS_DIR/mpfull/best_model_mpfull_bispec.pt" \
    --train_data_dir ${POWDERXRD_DATA_ROOT}/mpfull \
    --train_xrd_file train_data_sim_xrd.pkl \
    --train_lat_file train_data_bispec_list.pkl \
    --rruff_pkl "$RRUFF_PKL" \
    --model_config conf/model/transformerbispec.yaml \
    --train_config conf/train/default.yaml \
    --prediction_type bispectrum \
    --output_path "$OUTPUT_DIR/rruff_mpfull_bispec.pkl" &

# mpfull direct
CUDA_VISIBLE_DEVICES=1 python eval_rruff.py \
    --model_path "$MODELS_DIR/mpfull/best_model_mpfull_direct.pt" \
    --train_data_dir ${POWDERXRD_DATA_ROOT}/mpfull \
    --train_xrd_file train_data_sim_xrd.pkl \
    --train_lat_file train_data_prim_lat.pkl \
    --rruff_pkl "$RRUFF_PKL" \
    --model_config conf/model/transformerdirect.yaml \
    --train_config conf/train/default_direct.yaml \
    --prediction_type direct \
    --output_path "$OUTPUT_DIR/rruff_mpfull_direct.pkl" &

wait
echo "All RRUFF evaluations complete. Results in $OUTPUT_DIR"
