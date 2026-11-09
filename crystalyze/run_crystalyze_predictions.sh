#!/bin/bash
# Run the released Crystalyze model on the 148 RRUFF minerals and export its
# predicted lattice parameters. See README.md for how the test set and its
# conf/data/rruff_cag.yaml config are built first.
#
# Usage:
#   CRYSTALYZE_DIR=/path/to/Crystalyze bash run_crystalyze_predictions.sh

set -e

POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"
CRYSTALYZE_DIR="${CRYSTALYZE_DIR:?set CRYSTALYZE_DIR to the Crystalyze checkout}"
MODEL_DIR="${CRYSTALYZE_MODEL_DIR:-$CRYSTALYZE_DIR/their_model}"
GPU="${CUDA_VISIBLE_DEVICES:-0}"
LABEL=rruff_cag_force_na

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$CRYSTALYZE_DIR"

# evaluate.py refuses to overwrite and appends _1, _2, ... instead, so a stale
# output silently turns the next run into a differently-named file.
if [[ -e "$MODEL_DIR/eval_recon_$LABEL.pt" ]]; then
    echo "ERROR: $MODEL_DIR/eval_recon_$LABEL.pt already exists; move it aside first." >&2
    exit 1
fi

# A single batch (148 entries < batch_size 256), 5 Langevin samples.
CUDA_VISIBLE_DEVICES=$GPU python scripts/evaluate.py \
    --model_path "$MODEL_DIR" \
    --tasks recon \
    --num_batches 1 \
    --num_evals 5 \
    --force_num_atoms \
    --test_set_override data/rruff_cag \
    --label "$LABEL"

# Average the 5 samples per mineral.
python "$SCRIPT_DIR/export_preds_csv.py" \
    --recon-pt "$MODEL_DIR/eval_recon_$LABEL.pt" \
    --test-csv "$POWDERXRD_DATA_ROOT/rruff_crystalyze/test.csv" \
    --output   "$MODEL_DIR/eval_recon_rruff_cag.csv" \
    --reduce mean

echo "Done. Outputs in $MODEL_DIR"
