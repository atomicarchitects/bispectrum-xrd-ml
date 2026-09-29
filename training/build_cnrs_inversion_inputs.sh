#!/bin/bash
# Forward pass step for the CNRS-selection inversion pipeline: builds a
# predictions/targets/material_ids pkl (build_cnrs_selection_inversion_input.py)
# for each (name, checkpoint, input_type) triple below. Run this once per
# batch of checkpoints, then invert each resulting pkl separately with
# bispectrum/run_alg_inversion.py (that step is CPU-only and much slower,
# so keep it out of this script -- this one is just the cheap GPU/CPU-either
# forward pass that produces its input).
#
# Adjust CHECKPOINTS below to whatever you want to build inputs for next.
# Each entry writes to eval_results/opxrd_cnrs/inversion_inputs/<name>.pkl.
#
# Usage:
#   bash build_cnrs_inversion_inputs.sh          # auto device (cuda if available, else cpu)
#   bash build_cnrs_inversion_inputs.sh cpu      # force CPU (e.g. no GPU allocated)
set -euo pipefail

POWDERXRD_REPO_DIR="${POWDERXRD_REPO_DIR:?set POWDERXRD_REPO_DIR}"
POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"
DEVICE="${1:-}"

CNRS_MIX_DIR=${POWDERXRD_DATA_ROOT}/mpfull_aug_final_cnrs_mix/transformerbispec/models
PEAKLIST_BGMIX_DIR=${POWDERXRD_DATA_ROOT}/mpfull_aug_final_peaklist_bgmix/transformerbispec/models

# name                      checkpoint path                                                          input_type
CHECKPOINTS=(
  "cnrs_mix_epoch6          ${CNRS_MIX_DIR}/snapshots/epoch6_nlayers12_nhead8_dm256_hd512.pt          dense"
  "cnrs_mix_epoch15         ${CNRS_MIX_DIR}/snapshots/epoch15_nlayers12_nhead8_dm256_hd512.pt         dense"
  "cnrs_mix_epoch21         ${CNRS_MIX_DIR}/snapshots/epoch21_nlayers12_nhead8_dm256_hd512.pt         dense"
  "cnrs_mix_epoch34_live    ${CNRS_MIX_DIR}/nlayers12_nhead8_dm256_hd512_lossL1_smallvalsFalse_masklossTrue.pt  dense"
)

cd "${POWDERXRD_REPO_DIR}/training"

for entry in "${CHECKPOINTS[@]}"; do
    read -r name ckpt input_type <<< "$entry"
    if [ ! -f "$ckpt" ]; then
        echo "[SKIP] $name: checkpoint not found at $ckpt"
        continue
    fi
    echo "=== $name ($input_type) ==="
    if [ -n "$DEVICE" ]; then
        python3 build_cnrs_selection_inversion_input.py \
            --name "$name" --checkpoint-path "$ckpt" --input-type "$input_type" --device "$DEVICE"
    else
        python3 build_cnrs_selection_inversion_input.py \
            --name "$name" --checkpoint-path "$ckpt" --input-type "$input_type"
    fi
    echo
done

echo "Done. Inputs written to ${POWDERXRD_DATA_ROOT}/eval_results/opxrd_cnrs/inversion_inputs/"
