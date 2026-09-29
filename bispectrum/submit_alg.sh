#!/bin/bash
#SBATCH -J alg_run
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 4:00:00        
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --constraint=cpu
#SBATCH -o alg_run.out
#SBATCH -e alg_run.err
​
module load conda
conda activate powderxrd
​
srun run_alg.sh