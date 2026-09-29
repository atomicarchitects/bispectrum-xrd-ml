#!/bin/bash

# Recommended smaller configs
declare -A d_model=( ["tiny"]=128 ["small"]=256 ["medium"]=256 )
declare -A h_dim=( ["tiny"]=512 ["small"]=512 ["medium"]=1024 )
declare -A n_self_layer=( ["tiny"]=8 ["small"]=8 ["medium"]=16 )
declare -A n_head=( ["tiny"]=4 ["small"]=8 ["medium"]=8 )

for size in tiny small medium; do
  job_name="train_mp20_${size}"
  out_file="${job_name}.out"
  err_file="${job_name}.err"

  job_script=$(mktemp)

  cat <<EOF > "$job_script"
#!/bin/bash
#SBATCH -J ${job_name}
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 6:00:00        
#SBATCH -N 1          
#SBATCH -C gpu&hbm80g
#SBATCH -o ${out_file}
#SBATCH -e ${err_file}

module load conda
conda activate tda_env

python train.py \
  data='mp20' \
  train.use_wandb=True \
  model.d_model=${d_model[$size]} \
  model.h_dim=${h_dim[$size]} \
  model.n_self_layer=${n_self_layer[$size]} \
  model.n_head=${n_head[$size]} \
  train.lr=5e-4 \
  train.epochs=500
EOF

  sbatch "$job_script"
  rm "$job_script"
done
