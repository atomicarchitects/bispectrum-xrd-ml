#!/bin/bash
#SBATCH -J train_full_mp
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 12:00:00        
#SBATCH -N 1          
#SBATCH -C gpu&hbm80g
#SBATCH -o train_full_mp.out
#SBATCH -e train_full_mp.err

POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"

module load conda
conda activate tda_env

python train.py --config-name=config_direct_pred data=mpfull_aug_final_direct model=transformerdirect_medium train.use_wandb=True

#model.d_model=512 model.h_dim=2048 model.n_self_layer=24 model.n_head=8 

#train.load_pretrained=True train.pretrained_path=${POWDERXRD_DATA_ROOT}/mpfull/transformerbispec/models/transformerbispec_best_nlayers24_nhead8.pt train.epochs=600

### submit to load pretrained

#python train.py train.use_wandb=True train.wandb_run_name='mp20_bispec_new_data_adamW_restart' data='mp20' train.load_pretrained=True train.pretrained_path=${POWDERXRD_DATA_ROOT}/mp20/transformerbispec/models/transformerbispec_best.pt train.epochs=600