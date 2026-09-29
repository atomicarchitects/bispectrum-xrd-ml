#!/bin/bash

# Adjust -A below to your own NERSC allocation.
POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"

start=0
end=30449 #30449
step=500

for (( i=$start; i<$end; i+=step ))
do
    if (( i + step <= end )); then
        num_entries=$step
    else
        num_entries=$((end - i))
    fi

    (( num_entries > 0 )) || continue
    end_index=$((i + num_entries - 1))

    job_script=$(mktemp)

    cat <<EOF > "$job_script"
#!/bin/bash
#SBATCH -J alg_${i}_${end_index}
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH -o alg_${i}_${end_index}_%j.out
#SBATCH -e alg_${i}_${end_index}_%j.err

module load conda
conda activate tda_env

# prevent thread explosion
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

python run_alg_inversion.py \
    -i ${POWDERXRD_DATA_ROOT}/final_test_results/peaklists/mpfull/eval_results.pkl \
    -o ${POWDERXRD_DATA_ROOT}/final_test_results/peaklists/mpfull/inversion_results.pkl \
    -mat_proj_df ${POWDERXRD_DATA_ROOT}/materials_project_lat_bispec.h5 \
    -cob ${POWDERXRD_DATA_ROOT}/cob_lmax6.pkl \
    -begin_ind ${i} \
    -num_entries ${num_entries}
EOF

    sbatch "$job_script"
    rm "$job_script"
done
