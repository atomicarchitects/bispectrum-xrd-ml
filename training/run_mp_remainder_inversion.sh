#!/bin/bash
# Submit inversion jobs to complete missing MP test set entries for each model.
# The inversion script automatically skips already-completed entries in the H5 file.
#
# Usage: bash run_mp_remainder_inversion.sh

BISPEC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../bispectrum" && pwd)"
POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"
MAT_PROJ_DF=$POWDERXRD_DATA_ROOT/materials_project_lat_bispec.h5
COB=$POWDERXRD_DATA_ROOT/cob_lmax6.pkl
NUM_ENTRIES=30449

declare -A MODELS
MODELS["mpfull_aug"]="$POWDERXRD_DATA_ROOT/final_test_results/mpfull_aug"
MODELS["mpfull_crystalyze_cag"]="$POWDERXRD_DATA_ROOT/final_test_results/mpfull_crystalyze_cag"

declare -A WALLTIME
WALLTIME["mpfull_aug"]="06:00:00"
WALLTIME["mpfull_crystalyze_cag"]="03:00:00"

for MODEL in "${!MODELS[@]}"; do
    OUT_DIR="${MODELS[$MODEL]}"
    INPUT="$OUT_DIR/bispec_test_results.pkl"
    H5FILE="$OUT_DIR/bispec_test_results_inverted.h5"

    sbatch <<EOF
#!/bin/bash
#SBATCH -J inv_remainder_${MODEL}
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t ${WALLTIME[$MODEL]}
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH -o ${OUT_DIR}/inv_remainder_%j.out
#SBATCH -e ${OUT_DIR}/inv_remainder_%j.err

module load conda
conda activate tda_env

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

cd "$BISPEC_DIR"

python run_alg_inversion.py \\
    -i $INPUT \\
    -o $H5FILE \\
    -mat_proj_df $MAT_PROJ_DF \\
    -cob $COB \\
    -begin_ind 0 \\
    -num_entries $NUM_ENTRIES \\
    -max_workers 32 \\
    --timeout 500
EOF

    echo "Submitted remainder job for $MODEL"
done
