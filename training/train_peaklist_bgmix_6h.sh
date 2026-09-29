#!/bin/bash
#SBATCH -J peaklist_bgmix_6h
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -o peaklist_bgmix_6h_%j.out
#SBATCH -e peaklist_bgmix_6h_%j.err

# One-off 6h resume of the peaklist_bgmix run (currently at epoch 19) --
# no self-resubmit, just a single job. Re-submit by hand (`sbatch
# train_peaklist_bgmix_6h.sh`) if you want to continue past 6h; for an
# unattended multi-day chain use train_mpfull_peaklist_bgmix.sh instead.

POWDERXRD_REPO_DIR="${POWDERXRD_REPO_DIR:?set POWDERXRD_REPO_DIR}"
POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"

module load conda
source activate tda_env

DATA=mpfull_aug_final_peaklist_bgmix
MODEL=transformerbispec_sparse
CKPT_PATH=${POWDERXRD_DATA_ROOT}/${DATA}/${MODEL}/models/nlayers12_nhead8_dm256_hd512_lossL1_smallvalsFalse_masklossTrue.pt
EARLY_STOP_FLAG=${POWDERXRD_DATA_ROOT}/${DATA}/${MODEL}/models/early_stopped.flag

echo "Data:       $DATA"
echo "Model:      $MODEL"
echo "Checkpoint: $CKPT_PATH"

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
    train=default \
    +train.snapshot_every_epochs=3 \
    train.use_wandb=True \
    train.wandb_run_name='mpfull_peaklist_bgmix' \
    train.epochs=1000 \
    train.early_stopping=True \
    train.early_stopping_flag_path=$EARLY_STOP_FLAG \
    $PRETRAIN_ARG \
    hydra.run.dir=${POWDERXRD_DATA_ROOT}/hydra_outputs

echo "Job finished at $(date)"
