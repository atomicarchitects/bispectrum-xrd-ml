#!/bin/bash
#SBATCH -J train_bispec
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 07:00:00
#SBATCH -N 1
#SBATCH -C gpu&hbm80g
#SBATCH -o train_bispec_%j.out
#SBATCH -e train_bispec_%j.err

POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"

set -euo pipefail

echo "======================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================="

module load conda
set +u
source activate tda_env
set -u

SCRIPT="$(realpath "${BASH_SOURCE[0]}")"

# --- Arguments (override on command line or when submitting) ---
DATA=${1:-mpfull_crystalyze_cag}
MODEL=${2:-transformerbispec}
CKPT_PATH=${3:-${POWDERXRD_DATA_ROOT}/${DATA}/${MODEL}/models/${MODEL}_best.pt}

EARLY_STOP_FLAG="$(dirname "$CKPT_PATH")/early_stopped.flag"
MAX_EPOCHS=1000
EPOCHS_PER_JOB=30

echo "Data:       $DATA"
echo "Model:      $MODEL"
echo "Checkpoint: $CKPT_PATH"

# --- Check early stopping ---
if [ -f "$EARLY_STOP_FLAG" ]; then
    echo "Early stopping flag found. Training complete. Exiting."
    exit 0
fi

# --- Get current epoch from checkpoint ---
CURRENT_EPOCH=$(python - <<END || echo 0
import torch, os
try:
    if os.path.exists("$CKPT_PATH"):
        ckpt = torch.load("$CKPT_PATH", map_location="cpu", weights_only=False)
        print(int(ckpt.get("epoch", 0)))
    else:
        print(0)
except Exception:
    print(0)
END
)

echo "Current checkpoint epoch: $CURRENT_EPOCH"

REMAINING=$(( MAX_EPOCHS - CURRENT_EPOCH ))
if [ "$REMAINING" -le 0 ]; then
    echo "Training complete ($CURRENT_EPOCH >= $MAX_EPOCHS). Exiting."
    exit 0
fi

EPOCHS_TO_RUN=$(( REMAINING > EPOCHS_PER_JOB ? EPOCHS_PER_JOB : REMAINING ))
echo "Epochs to run this job: $EPOCHS_TO_RUN"

# --- Pretrain arg ---
if [ "$CURRENT_EPOCH" -gt 0 ]; then
    PRETRAIN_ARG="train.load_pretrained=True train.pretrained_path=$CKPT_PATH"
else
    PRETRAIN_ARG="train.load_pretrained=False"
fi

# --- Run training ---
cd "$(dirname "$SCRIPT")"
python train.py \
    --config-name=config \
    data=$DATA \
    model=$MODEL \
    train.use_wandb=True \
    train.epochs=$EPOCHS_TO_RUN \
    train.early_stopping=True \
    hydra.run.dir=${POWDERXRD_DATA_ROOT}/hydra_outputs \
    $PRETRAIN_ARG

# --- Resubmit if needed ---
if [ -f "$EARLY_STOP_FLAG" ]; then
    echo "Early stopping flag written. Training complete."
    exit 0
fi

EPOCH_AFTER=$(python - <<END || echo $CURRENT_EPOCH
import torch, os
try:
    if os.path.exists("$CKPT_PATH"):
        ckpt = torch.load("$CKPT_PATH", map_location="cpu", weights_only=False)
        print(int(ckpt.get("epoch", 0)))
    else:
        print($CURRENT_EPOCH)
except Exception:
    print($CURRENT_EPOCH)
END
)

if [ "$EPOCH_AFTER" -lt "$MAX_EPOCHS" ]; then
    echo "Resubmitting at $(date)..."
    sbatch "$SCRIPT" "$DATA" "$MODEL" "$CKPT_PATH"
else
    echo "All training epochs completed."
fi

echo "Job finished at $(date)"
