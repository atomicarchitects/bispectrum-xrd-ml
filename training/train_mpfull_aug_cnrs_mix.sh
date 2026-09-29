#!/bin/bash
#SBATCH -J train_mpfull_aug_cnrs_mix
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 19:00:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -o train_mpfull_aug_cnrs_mix_%j.out
#SBATCH -e train_mpfull_aug_cnrs_mix_%j.err

# Adjust -A above to your own NERSC allocation. POWDERXRD_REPO_DIR must be
# an absolute path to this checkout (see the BASH_SOURCE/spool-copy note
# below for why it can't be self-located).
POWDERXRD_REPO_DIR="${POWDERXRD_REPO_DIR:?set POWDERXRD_REPO_DIR}"
POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"

# Self-resubmitting retrain of mpfull_aug (data=mpfull_aug_final,
# model=transformerbispec) with train=default_cnrs_mix instead of the
# original train=default -- the only variable that changes vs. the existing
# final_saved_models/mpfull_aug/mpfull_aug_bispec_best_model.pt run. Same
# data config (same train/val/test split, since those are fixed files on
# disk), same seed (conf/config.yaml: seed=42).
#
# Each job runs up to the full 1000-epoch budget but SLURM kills it at the
# 6h walltime; on exit this script resubmits itself (unless the early-
# stopping flag file exists), and the next job resumes from whatever
# checkpoint was last saved -- no epoch-count bookkeeping needed since
# train.py itself tracks the current epoch in the checkpoint.
#
# IMPORTANT: this script must be run via `sbatch` (not pasted into an
# interactive shell) and referenced by its real path below -- SLURM copies
# sbatch scripts to a per-job spool directory before executing them, so
# ${BASH_SOURCE[0]}/realpath-based self-location resolves to that spool
# copy, not this file, and `cd`/resubmit using that would fail (this bit
# us once already: job 57838090 failed in 10s with "can't open file
# '/var/spool/slurmd/job.../train.py'"). Hardcoded absolute paths avoid it.
#
# train.py saves to output_dir/{data.name}/{model.type}/models/, derived
# ONLY from data.name + model.type -- it does NOT look at CKPT_PATH below,
# which is just this script's own bookkeeping. data.name is overridden to
# mpfull_aug_final_cnrs_mix (not the default "mpfull_aug_final") so this
# run's checkpoints can never collide with the original mpfull_aug run's
# live checkpoint (that collision happened once already: see git history
# around this file for what got overwritten and how it was recovered).
# data.DATA_DIR (where training data is actually read from) is unaffected
# by this override -- still mpfull_aug_cag_strain_texture, same as before.

# NOTE: deliberately no `set -e` here. Unlike train_mp20bispec_resubmit.sh
# (which sizes each job's epoch count to reliably finish on its own within
# its walltime), this script runs the full epochs=1000 budget each job and
# expects to normally be killed by SLURM's 6h walltime mid-training. That
# kill delivers SIGTERM, so `python train.py` exits nonzero -- with `set -e`
# active, the script would stop right there and never reach the resubmit
# logic below, silently breaking the chain on the very first (and most
# common) walltime hit. `set -u` is kept since it's safe here.
set -uo pipefail

echo "======================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================="

module load conda
set +u
source activate tda_env
set -u

SCRIPT=${POWDERXRD_REPO_DIR}/training/train_mpfull_aug_cnrs_mix.sh

DATA=mpfull_aug_final
DATA_NAME_OVERRIDE=mpfull_aug_final_cnrs_mix
MODEL=transformerbispec
TRAIN=default_cnrs_mix
CKPT_PATH=${POWDERXRD_DATA_ROOT}/${DATA_NAME_OVERRIDE}/${MODEL}/models/nlayers12_nhead8_dm256_hd512_lossL1_smallvalsFalse_masklossTrue.pt
EARLY_STOP_FLAG=${POWDERXRD_DATA_ROOT}/${DATA_NAME_OVERRIDE}/${MODEL}/models/early_stopped.flag

echo "Data:       $DATA (saved under data.name=$DATA_NAME_OVERRIDE)"
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
    data.name=$DATA_NAME_OVERRIDE \
    model=$MODEL \
    train=$TRAIN \
    train.use_wandb=True \
    train.wandb_run_name='mpfull_aug_cnrs_mix' \
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
