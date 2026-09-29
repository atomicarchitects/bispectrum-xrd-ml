#!/bin/bash
# Chunked inversion for mpfull_aug bispec test results (30449 entries).
# Run run_eval.sh first to generate bispec_test_results.pkl.

# Adjust -A in the generated job_script below to your own NERSC allocation.
POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"
BISPEC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../bispectrum" && pwd)"
INPUT=${POWDERXRD_DATA_ROOT}/final_test_results/mpfull_crystalyze_cag/bispec_test_results.pkl
OUT_DIR=${POWDERXRD_DATA_ROOT}/final_test_results/mpfull_crystalyze_cag/inv_chunks
MAT_PROJ_DF=${POWDERXRD_DATA_ROOT}/materials_project_lat_bispec.h5
COB=${POWDERXRD_DATA_ROOT}/cob_lmax6.pkl

mkdir -p "$OUT_DIR"

start=0
end=30449
step=500

for (( i=$start; i<$end; i+=step )); do
    if (( i + step <= end )); then
        num_entries=$step
    else
        num_entries=$((end - i))
    fi

    (( num_entries > 0 )) || continue
    end_index=$((i + num_entries - 1))
    OUT_FILE="$OUT_DIR/inv_${i}_${end_index}.h5"

    job_script=$(mktemp)
    cat <<EOF > "$job_script"
#!/bin/bash
#SBATCH -J mpf_inv_${i}_${end_index}
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH -o ${OUT_DIR}/inv_${i}_${end_index}_%j.out
#SBATCH -e ${OUT_DIR}/inv_${i}_${end_index}_%j.err

module load conda
conda activate tda_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$BISPEC_DIR"

python run_alg_inversion.py \\
    -i $INPUT \\
    -o $OUT_FILE \\
    -mat_proj_df $MAT_PROJ_DF \\
    -cob $COB \\
    -begin_ind ${i} \\
    -num_entries ${num_entries}
EOF

    sbatch "$job_script"
    rm "$job_script"
done
