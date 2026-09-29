#!/bin/bash
#SBATCH -J train_mpfull_peaklist
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 19:00:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -o train_mpfull_peaklist_%j.out
#SBATCH -e train_mpfull_peaklist_%j.err

# Adjust -A above to your own NERSC allocation. POWDERXRD_REPO_DIR must be
# an absolute path to this checkout (see train_mpfull_aug_cnrs_mix.sh for
# the BASH_SOURCE/spool-copy note on why it can't be self-located).
POWDERXRD_REPO_DIR="${POWDERXRD_REPO_DIR:?set POWDERXRD_REPO_DIR}"
POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"

# Self-resubmitting peak-list counterpart of the mpfull_aug bispec run. Same
# structures / train-val-test split / bispec targets as mpfull_aug_final,
# but the XRD input is a (2theta, prominence) peak list (top 30 by
# prominence, scipy.signal.find_peaks on the broadened curve) instead of
# the 8500-point dense curve. See conf/data/mpfull_aug_final_peaklist.yaml
# and generate_augmented_data.py's peaklist_combine / peaklist_from_xrd.
#
# model=transformerbispec_sparse -> XRDTransformerEncoder(input_type='sparse'),
# use_intensity=true (position + prominence). For the position-only ablation
# swap MODEL=transformerbispec_sparse_posonly below (checkpoints land in a
# separate dir keyed by data.name, so the two can coexist).
#
# train=default is reused as-is: its bg_noise_aug=true is automatically
# skipped for sparse/peak-list input (train.py sets xrd_std=None when
# cfg.data.indices, which disables add_bg_noise()). +train.snapshot_every_epochs=3
# below adds periodic non-overwriting snapshots on top of it (default.yaml
# doesn't set this key itself -- only default_cnrs_mix.yaml/
# default_cnrs_only.yaml do -- hence the `+` to add rather than override).
#
# Like train_mpfull_aug_cnrs_mix.sh: no `set -e` (jobs expect to be killed
# by walltime mid-training and must still reach the resubmit logic).
set -uo pipefail

echo "======================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================="

module load conda
set +u
source activate tda_env
set -u

SCRIPT=${POWDERXRD_REPO_DIR}/training/train_mpfull_peaklist.sh

DATA=mpfull_aug_final_peaklist
MODEL=transformerbispec_sparse
TRAIN=default
CKPT_PATH=${POWDERXRD_DATA_ROOT}/${DATA}/${MODEL}/models/nlayers12_nhead8_dm256_hd512_lossL1_smallvalsFalse_masklossTrue.pt
EARLY_STOP_FLAG=${POWDERXRD_DATA_ROOT}/${DATA}/${MODEL}/models/early_stopped.flag

echo "Data:       $DATA"
echo "Model:      $MODEL"
echo "Train cfg:  $TRAIN"
echo "Checkpoint: $CKPT_PATH"

# --- Check early stopping (from a previous job in this chain) ---
if [ -f "$EARLY_STOP_FLAG" ]; then
    echo "Early stopping flag found. Training complete. Exiting."
    exit 0
fi

# --- Resume from the last checkpoint if one exists ---
if [ -f "$CKPT_PATH" ]; then
    PRETRAIN_ARG="train.load_pretrained=True train.pretrained_path=$CKPT_PATH"
else
    PRETRAIN_ARG="train.load_pretrained=False"
fi

cd ${POWDERXRD_REPO_DIR}/training
python train.py \
    --config-name=config \
    data=$DATA \
    model=$MODEL \
    train=$TRAIN \
    +train.snapshot_every_epochs=3 \
    train.use_wandb=True \
    train.wandb_run_name='mpfull_peaklist_posprom' \
    train.epochs=1000 \
    train.early_stopping=True \
    train.early_stopping_flag_path=$EARLY_STOP_FLAG \
    $PRETRAIN_ARG \
    hydra.run.dir=${POWDERXRD_DATA_ROOT}/hydra_outputs

# --- Resubmit unless training actually converged ---
if [ -f "$EARLY_STOP_FLAG" ]; then
    echo "Early stopping flag written. Training complete."
    exit 0
fi

echo "Resubmitting at $(date)..."
sbatch "$SCRIPT"

echo "Job finished at $(date)"
