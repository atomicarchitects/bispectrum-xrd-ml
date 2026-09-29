#!/bin/bash
#SBATCH -J alg_array
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH --array=0-59    # (30449-500)/500 = ~60 chunks, adjust as needed
#SBATCH -o alg_%a_%j.out
#SBATCH -e alg_%a_%j.err

module load conda
conda activate tda_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

start=500
step=500
end=30449

i=$(( start + SLURM_ARRAY_TASK_ID * step ))
remaining=$(( end - i ))

if (( remaining <= 0 )); then
    exit 0
fi

if (( remaining >= step )); then
    num_entries=$step
else
    num_entries=$remaining
fi

python run_alg_inversion.py \
    -begin_ind ${i} \
    -num_entries ${num_entries}
