#!/bin/bash
#SBATCH -J mpfull_aug
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 08:00:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH -c 128
#SBATCH -o mpfull_aug_%j.out
#SBATCH -e mpfull_aug_%j.err

module load conda
conda activate tda_env

cd "$HOME/PowderXRD_Project/training/mp_full"

# Override --intens-gamma-a / --intens-gamma-scale once you have RRUFF-fit values.
python generate_augmented_data.py --augmentation strain
