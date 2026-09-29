#!/bin/bash
#SBATCH -J alg_run
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 10:00:00
#SBATCH --constraint=cpu
#SBATCH -o alg_run.out
#SBATCH -e alg_run.err

module load conda
conda activate tda_env

python run_alg_inversion.py
