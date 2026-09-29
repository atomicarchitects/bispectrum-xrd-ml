#!/bin/bash
#SBATCH -J train_mp20augval
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 07:00:00        
#SBATCH -N 1          
#SBATCH -C gpu&hbm80g
#SBATCH -o train_mp20augval.out
#SBATCH -e train_mp20augval.err

POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"

module load conda
conda activate tda_env
#python train.py data='mp20' train.use_wandb=True model.d_model=256 model.h_dim=512 model.n_self_layer=8 model.n_head=8 train.epsilon=False train.mask_zeros_in_loss=True train.epsilon=False
#python train.py data='mp20' train.use_wandb=True model.d_model=256 model.h_dim=512 model.n_self_layer=8 model.n_head=8 train.epsilon=false

python train.py data=mp20aug_valaug train.use_wandb=True train.load_pretrained=True train.pretrained_path=${POWDERXRD_DATA_ROOT}/mp20aug_valaug/transformerbispec/models/nlayers12_nhead8_dm256_hd512_lossL1_smallvalsFalse_masklossTrue.pt

#train.load_pretrained=True \
#train.pretrained_path=${POWDERXRD_DATA_ROOT}/mpfull/transformerbispec/models/nlayers12_nhead8_dm256_hd512_lossL1_smallvalsFalse_masklossTrue_new.pt

#python train.py data=mp20 \
#    train.use_wandb=True \
#    model.d_model=320 \
#    model.h_dim=640 \
#    model.n_self_layer=10 \
#    model.n_head=10 \
#    train.lr=5e-4 \
#    train.load_pretrained=True \
    #train.pretrained_path="${POWDERXRD_DATA_ROOT}/mp20/transformerbispec/models/transformerbispec_thr1e-08_nlayers10_nhead10_dm320_hd640_l1lossTrue.pt"

#model.d_model=512 model.h_dim=2048 model.n_self_layer=24 model.n_head=8 

#train.load_pretrained=True train.pretrained_path=${POWDERXRD_DATA_ROOT}/mpfull/transformerbispec/models/transformerbispec_best_nlayers24_nhead8.pt train.epochs=600

### submit to load pretrained

#python train.py train.use_wandb=True train.wandb_run_name='mp20_bispec_new_data_adamW_restart' data='mp20' train.load_pretrained=True train.pretrained_path=${POWDERXRD_DATA_ROOT}/mp20/transformerbispec/models/transformerbispec_best.pt train.epochs=600
