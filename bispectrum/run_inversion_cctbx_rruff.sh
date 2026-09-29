#!/bin/bash
# Runs inversion_results_cctbx.py for RRUFF bispectrum inversion results.

set -e

POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
INVERSION_DIR="$POWDERXRD_DATA_ROOT/final_test_results/rruff/processed_xy_results/inversion"
DIRECT_DIR="$POWDERXRD_DATA_ROOT/final_test_results/rruff"
RAW_XY_DIR="$POWDERXRD_DATA_ROOT/final_test_results/rruff/processed_xy_results"
OUT_DIR="$POWDERXRD_DATA_ROOT/final_test_results_cctbx/rruff"
RRUFF_BRAVAIS="$POWDERXRD_DATA_ROOT/rruff_true_bravais.pkl"

mkdir -p "$OUT_DIR/mp20_bispec" "$OUT_DIR/mp20aug_bispec" "$OUT_DIR/mpfull_bispec" \
         "$OUT_DIR/mp20_direct" "$OUT_DIR/mp20aug_direct" "$OUT_DIR/mpfull_direct" \
         "$OUT_DIR/processed_xy/mp20_direct"    "$OUT_DIR/processed_xy/mp20aug_direct"    "$OUT_DIR/processed_xy/mpfull_direct" \
         "$OUT_DIR/processed_xy/mp20_bispec_symm_tol"  "$OUT_DIR/processed_xy/mp20aug_bispec_symm_tol"  "$OUT_DIR/processed_xy/mpfull_bispec_symm_tol" \
         "$OUT_DIR/processed_xy/mp20_direct_symm_tol"  "$OUT_DIR/processed_xy/mp20aug_direct_symm_tol"  "$OUT_DIR/processed_xy/mpfull_direct_symm_tol" \
         "$OUT_DIR/processed_xy/mp20_valaug_direct" \
         "$OUT_DIR/processed_xy/mp20_crystalyze_aug_bispec"

MAX_DELTA=0.2
MAX_DELTA_PRED=0.2
SYMPREC=0.1

run() {
    local label=$1; local input=$2; local output=$3; local plot_dir=$4
    echo "=== $label ==="
    python "$SCRIPT_DIR/inversion_results_cctbx.py" \
        --input "$input" \
        --output "$output" \
        --plot-dir "$plot_dir" \
        --true-bravais "$RRUFF_BRAVAIS" \
        --rruff \
        --make-df \
        --canonicalize 
        #--max-delta "$MAX_DELTA" \
        #--max-delta-pred "$MAX_DELTA_PRED" \
        #--symprec "$SYMPREC"
    echo "Done: $label"
    echo ""
}

# Bispectrum inversion results (already run — commented out)
#run "rruff mp20 bispec" \
#    "$INVERSION_DIR/mp20_bispec/inversion_results_begin_0_end_1153.h5" \
#    "$OUT_DIR/processed_xy/mp20_bispec_symm_tol/inversion_results_cctbx.pkl" \
#    "$OUT_DIR/processed_xy/mp20_bispec_symm_tol/plots"

#run "rruff mp20aug bispec" \
#    "$INVERSION_DIR/mp20aug_bispec/inversion_results_begin_0_end_1153.h5" \
#    "$OUT_DIR/processed_xy/mp20aug_bispec_symm_tol/inversion_results_cctbx.pkl" \
#    "$OUT_DIR/processed_xy/mp20aug_bispec_symm_tol/plots"

#run "rruff mpfull bispec" \
#    "$INVERSION_DIR/mpfull_bispec/inversion_results_begin_0_end_1153.h5" \
#    "$OUT_DIR/processed_xy/mpfull_bispec_symm_tol/inversion_results_cctbx.pkl" \
#    "$OUT_DIR/processed_xy/mpfull_bispec_symm_tol/plots"

#run "rruff mp20_valaug bispec" \
#    "$INVERSION_DIR/mp20_valaug_bispec/inversion_results_begin_0_end_1153.h5" \
#    "$OUT_DIR/processed_xy/mp20_valaug_bispec/inversion_results_cctbx.pkl" \
#    "$OUT_DIR/processed_xy/mp20_valaug_bispec/plots"

run "rruff mp20_crystalyze_aug bispec" \
    "$INVERSION_DIR/mp20_crystalyze_aug_bispec/inversion_results_begin_0_end_1153.h5" \
    "$OUT_DIR/processed_xy/mp20_crystalyze_aug_bispec/inversion_results_cctbx.pkl" \
    "$OUT_DIR/processed_xy/mp20_crystalyze_aug_bispec/plots"

# Direct model predictions (commented out — use raw_xy runs below)
#run "rruff mp20 direct" \
#    "$DIRECT_DIR/rruff_mp20_direct.pkl" \
#    "$OUT_DIR/mp20_direct/inversion_results_cctbx.pkl" \
#    "$OUT_DIR/mp20_direct/plots"

#run "rruff mp20aug direct" \
#    "$DIRECT_DIR/rruff_mp20aug_direct.pkl" \
#    "$OUT_DIR/mp20aug_direct/inversion_results_cctbx.pkl" \
#    "$OUT_DIR/mp20aug_direct/plots"

#run "rruff mpfull direct" \
#    "$DIRECT_DIR/rruff_mpfull_direct.pkl" \
#    "$OUT_DIR/mpfull_direct/inversion_results_cctbx.pkl" \
#    "$OUT_DIR/mpfull_direct/plots"

# Raw XY direct predictions
#run "rruff processed_xy mp20 direct" \
#    "$RAW_XY_DIR/rruff_mp20_direct_proc.pkl" \
#    "$OUT_DIR/processed_xy/mp20_direct_symm_tol/inversion_results_cctbx.pkl" \
#    "$OUT_DIR/processed_xy/mp20_direct_symm_tol/plots"

#run "rruff processed_xy mp20 valaug direct" \
#    "$RAW_XY_DIR/rruff_mp20_valaug_direct_proc.pkl" \
#    "$OUT_DIR/processed_xy/mp20_valaug_direct/inversion_results_cctbx.pkl" \
#    "$OUT_DIR/processed_xy/mp20_valaug_direct/plots"

#run "rruff processed_xy mpfull direct" \
#    "$RAW_XY_DIR/rruff_mpfull_direct_proc.pkl" \
#    "$OUT_DIR/processed_xy/mpfull_direct_symm_tol/inversion_results_cctbx.pkl" \
#    "$OUT_DIR/processed_xy/mpfull_direct_symm_tol/plots"

echo "All done. Results in $OUT_DIR"
