#!/bin/bash

# Arrays of hyperparameters
D_MODELS=(128 256 384 384)
H_DIMS=(256 512 756 756)
N_LAYERS=(4 12 12 16)
N_HEADS=(4 8 12 12)

NUM_JOBS=${#D_MODELS[@]}

mkdir -p logs

for ((i=0; i<$NUM_JOBS; i++)); do
    sbatch --export=D_MODEL=${D_MODELS[i]},H_DIM=${H_DIMS[i]},N_LAYER=${N_LAYERS[i]},N_HEAD=${N_HEADS[i]} <<'EOF'
#!/bin/bash
#SBATCH -J train_mp20_${D_MODEL}_${H_DIM}
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH -C gpu&hbm80g
#SBATCH -o logs/job_%j.out
#SBATCH -e logs/job_%j.err

module load conda
conda activate tda_env

echo "Running: d_model=$D_MODEL, h_dim=$H_DIM, n_layers=$N_LAYER, n_heads=$N_HEAD, epsilon=false"

python train.py \
    data='mp20' \
    train.use_wandb=True \
    model.d_model=$D_MODEL \
    model.h_dim=$H_DIM \
    model.n_self_layer=$N_LAYER \
    model.n_head=$N_HEAD \
    train.mask_zeros_in_loss=True
EOF
done
