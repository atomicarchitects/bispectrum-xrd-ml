#!/bin/bash
# Runs inversion_results_cctbx.py for all datasets in final_test_results/
# Outputs to final_test_results_cctbx/ mirroring the input directory structure.

set -e

POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$POWDERXRD_DATA_ROOT/final_test_results"
OUT_DIR="$POWDERXRD_DATA_ROOT/final_test_results_cctbx"
DATASETS_DIR="$POWDERXRD_DATA_ROOT"

MP20_BRAVAIS="$DATASETS_DIR/mp20/test_true_bravais.pkl"
MPFULL_BRAVAIS="$DATASETS_DIR/mpfull/test_true_bravais.pkl"

mkdir -p "$OUT_DIR/mpfull_aug" "$OUT_DIR/mpfull_bispec" "$OUT_DIR/mpfull_crystalyze_cag" "$OUT_DIR/mpfull_aug_niggli"
#"$OUT_DIR/mp20_final_aug" "$OUT_DIR/mp20_aug" "$OUT_DIR/mpfull" "$OUT_DIR/crystalyze"

run() {
    local label=$1; local input=$2; local output=$3; local plot_dir=$4; local bravais=$5
    echo "=== $label ==="
    python "$SCRIPT_DIR/inversion_results_cctbx.py" \
        --input "$input" \
        --output "$output" \
        --plot-dir "$plot_dir" \
        --true-bravais "$bravais" \
        --make-df \
        --canonicalize
    echo "Done: $label"
    echo ""
}

#run "mp20 bispec" \
#     "$RESULTS_DIR/mp20_correct_cag/bispec_test_results_inverted.h5" \
#     "$OUT_DIR/mp20_correct_cag/inversion_results_cctbx_bispec.pkl" \
#     "$OUT_DIR/mp20_correct_cag/plots_bispec" \
#     "$MP20_BRAVAIS"

#run "mp20 direct" \
#     "$RESULTS_DIR/mp20_correct_cag/direct_test_results.pkl" \
#     "$OUT_DIR/mp20_correct_cag/inversion_results_cctbx_direct.pkl" \
#     "$OUT_DIR/mp20_correct_cag/plots_direct" \
#     "$MP20_BRAVAIS"
#

#run "mp20_aug direct" \
#    "$RESULTS_DIR/mp20_final_aug/direct_test_results.pkl" \
#    "$OUT_DIR/mp20_final_aug/inversion_results_cctbx_direct.pkl" \
#    "$OUT_DIR/mp20_final_aug/plots_direct" \
#    "$MP20_BRAVAIS"

#run "mp20_aug bispec" \
#    "$RESULTS_DIR/mp20_final_aug/bispec_test_results_inverted.h5" \
#    "$OUT_DIR/mp20_final_aug/inversion_results_cctbx_bispec.pkl" \
#    "$OUT_DIR/mp20_final_aug/plots_bispec" \
#    "$MP20_BRAVAIS"

# run "mp20_aug bispec" \
#     "$RESULTS_DIR/mp20_aug/test_mp20aug_bispec.h5" \
#     "$OUT_DIR/mp20_aug/inversion_results_cctbx_bispec.pkl" \
#     "$OUT_DIR/mp20_aug/plots_bispec" \
#     "$MP20_BRAVAIS"

#run "mp20_aug direct" \
#    "$RESULTS_DIR/mp20_aug/mp20aug_direct_test.pkl" \
#    "$OUT_DIR/mp20_aug/inversion_results_cctbx_direct_new.pkl" \
#    "$OUT_DIR/mp20_aug/plots_direct_new" \
#    "$MP20_BRAVAIS"

run "mpfull_aug bispec" \
    "$RESULTS_DIR/mpfull_aug/bispec_test_results_inverted.h5" \
    "$OUT_DIR/mpfull_aug_niggli/inversion_results_cctbx_bispec.pkl" \
    "$OUT_DIR/mpfull_aug_niggli/plots_bispec" \
    "$MPFULL_BRAVAIS" \
    

run "mpfull_aug direct" \
    "$RESULTS_DIR/mpfull_aug/direct_test_results.pkl" \
    "$OUT_DIR/mpfull_aug_niggli/inversion_results_cctbx_direct.pkl" \
    "$OUT_DIR/mpfull_aug_niggli/plots_direct" \
    "$MPFULL_BRAVAIS"

#run "mpfull_bispec bispec" \
#    "$RESULTS_DIR/mpfull_bispec/bispec_test_results_inverted.h5" \
#    "$OUT_DIR/mpfull_bispec/inversion_results_cctbx_bispec.pkl" \
#    "$OUT_DIR/mpfull_bispec/plots_bispec" \
#    "$MPFULL_BRAVAIS"

#run "mpfull_bispec direct" \
#    "$RESULTS_DIR/mpfull_bispec/direct_test_results.pkl" \
#    "$OUT_DIR/mpfull_bispec/inversion_results_cctbx_direct.pkl" \
#    "$OUT_DIR/mpfull_bispec/plots_direct" \
#    "$MPFULL_BRAVAIS"

#run "mpfull_crystalyze_cag bispec" \
#    "$RESULTS_DIR/mpfull_crystalyze_cag/bispec_test_results_inverted.h5" \
#    "$OUT_DIR/mpfull_crystalyze_cag/inversion_results_cctbx_bispec.pkl" \
#    "$OUT_DIR/mpfull_crystalyze_cag/plots_bispec" \
#    "$MPFULL_BRAVAIS"

#run "mpfull_crystalyze_cag direct" \
#    "$RESULTS_DIR/mpfull_crystalyze_cag/direct_test_results.pkl" \
#    "$OUT_DIR/mpfull_crystalyze_cag/inversion_results_cctbx_direct.pkl" \
#    "$OUT_DIR/mpfull_crystalyze_cag/plots_direct" \
#    "$MPFULL_BRAVAIS"

#echo "=== crystalyze ==="
#python "$SCRIPT_DIR/inversion_results_cctbx.py" \
#     --input "$RESULTS_DIR/mp20/test_mp20_bispec.h5" \
#     --output "$OUT_DIR/crystalyze/inversion_results_cctbx_crystalyze_uvw.pkl" \
#     --plot-dir "$OUT_DIR/crystalyze_uvw/plots" \
#     --true-bravais "$MP20_BRAVAIS" \
#     --crystalyze-params "$RESULTS_DIR/crystalyze/cell_params_by_mpid_eval_recon_full_data_uvw_force_atoms.pt" \
#     --make-df \
#     --canonicalize
# echo "Done: crystalyze"

echo "All done. Results in $OUT_DIR"
