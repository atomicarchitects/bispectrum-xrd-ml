#!/bin/bash
#SBATCH -J train_mp20_peaklist_bgmix
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -o train_mp20_peaklist_bgmix_%j.out
#SBATCH -e train_mp20_peaklist_bgmix_%j.err

# Adjust -A above to your own NERSC allocation. POWDERXRD_REPO_DIR must be
# an absolute path to this checkout (see train_mpfull_aug_cnrs_mix.sh for
# the BASH_SOURCE/spool-copy note on why it can't be self-located).
POWDERXRD_REPO_DIR="${POWDERXRD_REPO_DIR:?set POWDERXRD_REPO_DIR}"
POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"

# Noise-augmented ablation of train_mpfull_peaklist.sh, trained FROM SCRATCH
# (train.load_pretrained=False, separate data.name) rather than resumed from
# the clean run's epoch-15 checkpoint -- resuming would confound "trained on
# noisy data" with "trained on clean data first, then switched", making it
# impossible to cleanly attribute any difference to the noise augmentation
# itself. Same data.name-derived checkpoint dir as
# conf/data/mpfull_aug_final_peaklist_bgmix.yaml's `name` field
# (mpfull_aug_final_peaklist_bgmix), fully separate from the clean run's
# mpfull_aug_final_peaklist -- the two lineages never collide.
#
# Only the XRD input differs from the clean peaklist run: peaks extracted
# from clean curve + one Gamma noise realization (RRUFF/CNRS mixed 50/50,
# matching the dense cnrs_mix retrain's ratio) instead of the clean curve
# directly. Same 927,080 train keys/targets, same val/test files -- see the
# data config for details.
#
# +train.snapshot_every_epochs=3 is on from epoch 0 this time (the clean
# peaklist run only got it added partway through, after losing epoch
# 0-15's history to the overwrite-only "best" checkpoint).
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

SCRIPT=${POWDERXRD_REPO_DIR}/training/train_mp20_peaklist_bgmix.sh

DATA=mp20aug_peaklist_bgmix
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

# --- Resume from the last checkpoint if one exists (within THIS lineage
# only -- on the very first submission CKPT_PATH won't exist yet, so this
# correctly starts from scratch rather than pulling in the clean run's
# weights) ---
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
    train.wandb_run_name='mp20_peaklist_bgmix' \
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
