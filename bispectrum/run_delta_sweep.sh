#!/bin/bash
# Sweeps --max-delta-pred values for mpfull_crystalyze_cag bispec and direct.

set -e

POWDERXRD_DATA_ROOT="${POWDERXRD_DATA_ROOT:?set POWDERXRD_DATA_ROOT}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
RESULTS_DIR="$POWDERXRD_DATA_ROOT/final_test_results/mpfull_crystalyze_cag"
OUT_DIR="$POWDERXRD_DATA_ROOT/final_test_results_cctbx/mpfull_crystalyze_cag"
BRAVAIS="$POWDERXRD_DATA_ROOT/mpfull/test_true_bravais.pkl"

mkdir -p "$OUT_DIR"

for delta in 0.01; do
    tag="delta$(echo $delta | tr -d '.')"

    echo "=== bispec delta=$delta ==="
    python "$SCRIPT_DIR/inversion_results_cctbx.py" \
        --input    "$RESULTS_DIR/bispec_test_results_inverted.h5" \
        --output   "$OUT_DIR/inversion_results_cctbx_bispec_${tag}.pkl" \
        --plot-dir "$OUT_DIR/plots_bispec_${tag}" \
        --true-bravais "$BRAVAIS" \
        --make-df \
        --canonicalize \
        --max-delta-pred "$delta"
    echo "Done bispec delta=$delta"
    echo ""

    echo "=== direct delta=$delta ==="
    python "$SCRIPT_DIR/inversion_results_cctbx.py" \
        --input    "$RESULTS_DIR/direct_test_results.pkl" \
        --output   "$OUT_DIR/inversion_results_cctbx_direct_${tag}.pkl" \
        --plot-dir "$OUT_DIR/plots_direct_${tag}" \
        --true-bravais "$BRAVAIS" \
        --make-df \
        --canonicalize \
        --max-delta-pred "$delta"
    echo "Done direct delta=$delta"
    echo ""
done

echo "All done. Results in $OUT_DIR"
