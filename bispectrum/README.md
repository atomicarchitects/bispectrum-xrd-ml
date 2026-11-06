# bispectrum/

Inversion of predicted bispectra to lattice parameters, scoring of predicted lattices against the
true ones with cctbx, and the paper's analyses. Predictions come from `../training/`. Run the scripts
from this directory.

## Setup

`pip install -e .` from the repository root installs everything, including cctbx (`cctbx-base`).
The inversion needs two files under `$POWDERXRD_DATA_ROOT`, both included with the pretrained models:

```bash
python build_cob.py                  # change-of-basis tensor -> cob_lmax6.pkl
python build_lookup_database.py      # bispectrum lookup database -> materials_project_lat_bispec.h5
```

`make_mp20_bispec_h5.py` filters the database to the MP-20 materials.

## 1. Invert

```bash
python run_alg_inversion.py -i <predictions>.pkl -o <out>.h5 \
    -mat_proj_df $POWDERXRD_DATA_ROOT/materials_project_lat_bispec.h5 \
    -cob $POWDERXRD_DATA_ROOT/cob_lmax6.pkl \
    -begin_ind 0 -num_entries <N> -max_workers 32
```

Each prediction is optimized from the `-num_to_opt` (default 15) nearest database entries; the query
material itself is excluded. The MP test sets used the defaults. The RRUFF and CNRS runs used
`-num_to_opt 30 -random_restarts 10 -random_restart_max_abc 55 -candidate_timeout 30 -timeout 1500`.

Materials already in the output file are skipped, so a rerun resumes. To split a large set, run
chunks with `-begin_ind`/`-num_entries` into files named `inv_<begin>.h5` (keep the last chunk within
the number of entries), then

```bash
python merge_inv_chunks.py --chunks-dir <dir> --output <merged>.h5
```

Each output entry has `pred` (real-space lattice), `true_lattice`, `min_loss`, and the winning start
(`start_id`, `start_matrix`).

## 2. Score

True Bravais types, once per dataset:

```bash
python compute_true_bravais.py csv --csv <test.csv> --output <dataset>_true_bravais.pkl   # MP-20, MP-Full
python compute_true_bravais.py rruff
python build_cnrs_true_bravais.py                                                         # CNRS
```

(`download_mpfull_cifs.py` builds the MP-Full `test.csv`; it needs `mp-api`.) Then score:

```bash
python inversion_results_cctbx.py --input <out>.h5 --output <scored>.pkl --plot-dir <dir> \
    --true-bravais <dataset>_true_bravais.pkl --make-df --canonicalize      # add --rruff for RRUFF/CNRS
```

Direct-model results (`.pkl` from the eval scripts) are scored the same way.
`inversion_results_cctbx.py` Niggli-reduces both lattices and compares them in the
symmetry-consistent setting closest to the true cell, so the errors measure geometry, not cell
choice. `--crystalyze-params` / `--crystalyze-csv` score the Crystalyze baseline's lattices the same
way; see [../crystalyze/README.md](../crystalyze/README.md).

## 3. Analyses

| script | what |
|---|---|
| `analyze_completed_inversion_res.py` | quick summary of an inversion output |
| `make_comparison_plots.py` | per-dataset error distributions and parity plots |
| `failure_mode_analysis.py` | Bravais-type confusion and failure categories |
| `plot_volume_mape.py`, `plot_bispec_channel_error.py`, `plot_bispec_lat_error.py` | volume error; lattice vs. bispectrum error |
| `fetch_theoretical_flag.py`, `filter_dominant_zones.py`, `dominant_zone_prediction_quality.py`, `compute_pattern_stats.py` | dominant-zone analysis |
| `filter_rruff.py`, `filter_inversion_results.py` | RRUFF entry filtering |
| `time_inversion.py`, `plot_time_inversion*.py`, `time_bispec_database.py` | inversion and database cost |
| `mp20_xrd_leakage.py`, `mpfull_xrd_leakage.py`, `plot_similarity_vs_error.py`, `analyze_start_leakage.py` | train/test leakage |
| `plot_bispec_distribution.py`, `paper_bispec_figs.ipynb`, `volumetric_plot.py` | bispectrum figures |
