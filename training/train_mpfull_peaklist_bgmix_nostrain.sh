#!/bin/bash
#SBATCH -J train_peaklist_bgmix_nostrain
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 19:00:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -o train_peaklist_bgmix_nostrain_%j.out
#SBATCH -e train_peaklist_bgmix_nostrain_%j.err

# Strain ablation of train_mpfull_peaklist_bgmix.sh: identical setup (same
# caglioti4x + texture sources, same CNRS/RRUFF-mixed bg-noise realization)
# but the strain-augmented source is dropped from the training combine (see
# conf/data/mpfull_aug_final_peaklist_bgmix_nostrain.yaml and the
# train_peaklist_bgmix_nostrain_combined.pkl build in mpfull_aug_cag_strain_texture,
# 648,956 keys vs 927,080 with strain). Tests whether strain augmentation is
# itself contributing to the sim-to-real (CNRS) gap.
#
# Deliberately trained FROM SCRATCH (train.load_pretrained=False), same as
# the with-strain peaklist_bgmix run, so the comparison isn't confounded by
# pretraining on strain-augmented weights first.
#
# Capped at train.epochs=21 (not run to convergence) -- the with-strain
# ablations (cnrs_mix, peaklist_bgmix) all showed their real signal by
# epoch ~9-18, so this is enough for a directional comparison against
# peaklist_bgmix's epoch 9/18/21 checkpoints without burning days of GPU
# time chasing full convergence. snapshot_every_epochs=3 lines up snapshots
# at 3/6/9/12/15/18/21 for exactly that comparison.
#
# No `set -e` -- jobs expect to be killed by walltime mid-training and must
# still reach the resubmit logic.
set -uo pipefail

echo "======================================="
echo "Job started at $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "======================================="

POWDERXRD_REPO_DIR="${POWDERXRD_REPO_DIR:?set POWDERXRD_REPO_DIR}"
POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"

module load conda
set +u
source activate tda_env
set -u

SCRIPT=${POWDERXRD_REPO_DIR}/training/train_mpfull_peaklist_bgmix_nostrain.sh

DATA=mpfull_aug_final_peaklist_bgmix_nostrain
MODEL=transformerbispec_sparse
# NOTE: cfg.model.type inside transformerbispec_sparse.yaml is "transformerbispec"
# (no _sparse suffix) -- that's the literal directory train.py creates
# (model_dir = output_dir/data.name/model.type/models), NOT the config
# filename. Using $MODEL here would silently never find the checkpoint and
# restart from scratch on every resubmission.
MODEL_TYPE_DIR=transformerbispec
TRAIN=default
CKPT_PATH=${POWDERXRD_DATA_ROOT}/${DATA}/${MODEL_TYPE_DIR}/models/nlayers12_nhead8_dm256_hd512_lossL1_smallvalsFalse_masklossTrue.pt
EARLY_STOP_FLAG=${POWDERXRD_DATA_ROOT}/${DATA}/${MODEL_TYPE_DIR}/models/early_stopped.flag
EPOCH_CAP_FLAG=${POWDERXRD_DATA_ROOT}/${DATA}/${MODEL_TYPE_DIR}/models/epoch_cap_reached.flag

echo "Data:       $DATA"
echo "Model:      $MODEL"
echo "Train cfg:  $TRAIN"
echo "Checkpoint: $CKPT_PATH"

# --- Stop if early-stopped OR we've already reached the epoch cap ---
if [ -f "$EARLY_STOP_FLAG" ]; then
    echo "Early stopping flag found. Training complete. Exiting."
    exit 0
fi
if [ -f "$EPOCH_CAP_FLAG" ]; then
    echo "Epoch cap flag found (>=21 epochs already reached). Exiting."
    exit 0
fi

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
    train.wandb_run_name='mpfull_peaklist_bgmix_nostrain' \
    train.epochs=21 \
    train.early_stopping=True \
    train.early_stopping_flag_path=$EARLY_STOP_FLAG \
    $PRETRAIN_ARG \
    hydra.run.dir=${POWDERXRD_DATA_ROOT}/hydra_outputs

# --- Mark the epoch cap reached so we don't resubmit past epoch 21 even
# though train.epochs=21 naturally stops the python process -- belt and
# braces in case a walltime kill lands right at the boundary. ---
touch "$EPOCH_CAP_FLAG"

if [ -f "$EARLY_STOP_FLAG" ]; then
    echo "Early stopping flag written. Training complete."
    exit 0
fi

echo "Epoch cap (21) reached. Not resubmitting."
echo "Job finished at $(date)"
