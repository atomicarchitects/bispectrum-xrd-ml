#!/bin/bash
# Submit inversion jobs for all bispectrum RRUFF eval results.
# Usage:
#   bash run_rruff_inversion.sh          # full run
#   bash run_rruff_inversion.sh --retry  # retry only missing IDs

# Adjust -A in the generated job_script below to your own NERSC allocation.
POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"
# Interpreter used for the inline --retry diff below (outside the sbatch job,
# so it can't rely on `module load conda` having already run in this shell).
POWDERXRD_PYTHON="${POWDERXRD_PYTHON:-python3}"
RRUFF_EVAL_DIR=${POWDERXRD_DATA_ROOT}/eval_results/rruff/processed_xy_results
INV_OUT_DIR=$RRUFF_EVAL_DIR/inversion
BISPEC_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../bispectrum" && pwd)"

MAT_PROJ_DF=${POWDERXRD_DATA_ROOT}/materials_project_lat_bispec.h5
COB=${POWDERXRD_DATA_ROOT}/cob_lmax6.pkl

NUM_ENTRIES=1153
RETRY=false
[[ "$1" == "--retry" ]] && RETRY=true

for MODEL in mp20_valaug_bispec mpfull_bispec; do
    INPUT=$RRUFF_EVAL_DIR/rruff_${MODEL}_proc.pkl
    OUT_SUBDIR=$INV_OUT_DIR/${MODEL}
    H5FILE=${OUT_SUBDIR}/inversion_results_begin_0_end_${NUM_ENTRIES}.h5
    mkdir -p "$OUT_SUBDIR"

    if $RETRY; then
        # Find missing IDs by diffing pkl keys against HDF5 keys
        MISSING_FILE=${OUT_SUBDIR}/missing_ids.txt
        "$POWDERXRD_PYTHON" - <<PYEOF
import pickle, h5py
with open("$INPUT", "rb") as f:
    d = pickle.load(f)
id_key = "mineral_names" if "mineral_names" in d else "material_ids"
all_ids = set(str(x).strip() for x in d[id_key])
try:
    with h5py.File("$H5FILE", "r") as f:
        done = set(str(k) for k in f.keys())
except Exception:
    done = set()
missing = sorted(all_ids - done)
print(f"[INFO] $MODEL: {len(missing)} missing IDs")
with open("$MISSING_FILE", "w") as f:
    f.write("\n".join(missing))
PYEOF
        COUNT=$(wc -l < "$MISSING_FILE")
        if [[ "$COUNT" -eq 0 ]]; then
            echo "No missing IDs for $MODEL, skipping."
            continue
        fi
        echo "Submitting retry job for $MODEL ($COUNT missing IDs)"
        MAT_FILE_ARG="-mat_file $MISSING_FILE"
        JOB_NAME="rruff_retry_${MODEL}"
    else
        MAT_FILE_ARG=""
        JOB_NAME="rruff_inv_${MODEL}"
    fi

    job_script=$(mktemp)
    cat <<EOF > "$job_script"
#!/bin/bash
#SBATCH -J ${JOB_NAME}
#SBATCH -A YOUR_ACCOUNT
#SBATCH -q regular
#SBATCH -t 06:00:00
#SBATCH --nodes=1
#SBATCH --constraint=cpu
#SBATCH -o ${OUT_SUBDIR}/inv_%j.out
#SBATCH -e ${OUT_SUBDIR}/inv_%j.err

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
    $MAT_FILE_ARG
EOF

    sbatch "$job_script"
    rm "$job_script"
    echo "Submitted job for $MODEL"
done
