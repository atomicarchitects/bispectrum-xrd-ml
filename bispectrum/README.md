# bispectrum/

L-BFGS inversion of a predicted bispectrum back to lattice parameters, and the
analysis that compares predicted lattices against ground truth. The predicted
bispectra (and the direct-route lattice predictions) come from `training/`.

All scripts read and write under `$POWDERXRD_DATA_ROOT` (see the top-level
README). Run them from this directory, e.g. `cd bispectrum && python ...`.

## Pipeline

**1. Inversion.** Recover lattice parameters from a predicted bispectrum.
- `run_alg_inversion.py` is the L-BFGS inversion. It starts from the nearest
  entries of the lookup database (`-mat_proj_df`) and records which start won.
- `run_alg.sh`, `run_alg_single.sh`, `run_alg_single_chunks.sh`, `submit_alg.sh`
  are SLURM wrappers for bulk, single and chunked runs.
- `merge_inv_chunks.py` merges chunked `inv_*.h5` outputs into one file.
- `build_lookup_database.py` builds the lookup database
  (`materials_project_lat_bispec.h5`). `make_mp20_bispec_h5.py` filters it to
  the MP-20 materials.

Output: `$POWDERXRD_DATA_ROOT/final_test_results/<dataset>/`.

**2. Ground truth.** The true Bravais type and lattice of each test material,
independent of any prediction.
- `download_mpfull_cifs.py` builds `mpfull/test.csv` (CIF and `material_id`
  columns) from the Materials Project API. Needs `mp-api`.
- `compute_true_bravais.py` has one subcommand per source format:
  - `csv --csv <test.csv>`: a CSV with a CIF per row (MP-20 and MP-Full test splits).
  - `cif-dir --cif-dir <dir>`: a directory of `.cif` files.
  - `rruff`: `rruff_data_dict.pkl`, where the space group is already in the metadata.
- `build_cnrs_true_bravais.py` reshapes the verified CNRS space-group metadata
  (`opxrd_cnrs_metadata.pkl`, an input that is not produced by this repository)
  into the same format.

Output: a `*_true_bravais.pkl` per dataset, passed to stage 3 as `--true-bravais`.

**3. Comparison.** `inversion_results_cctbx.py` takes a stage 1 result and a
stage 2 pickle. It Niggli-reduces both lattices, then uses cctbx
(`lattice_symmetry.metric_subgroups`) to find the pair of symmetry-consistent
cells, one for the true lattice and one for the prediction, with the smallest
metric-tensor distance. The metric-subgroup search offers no alternative settings
for a triclinic cell, and two nearby lattices can reduce to different settings of
the same cell (e.g. angles θ vs. 180° − θ), so a prediction that matches the true
cell within 5% in lengths and 5° in angles in some other setting is re-expressed
in that setting (`utilities.match_cell_setting`, via cctbx
`similarity_transformations`). Only the basis changes, never the predicted
lattice; affected rows are flagged in the `cell_matched` column, and
`--no-cell-matching` turns the step off. Errors are then computed in that common
setting, so they reflect geometric differences rather than differences in cell
choice. Requires cctbx (see below). Wrappers:
- `run_inversion_cctbx_final.sh`: every dataset under `final_test_results/`.
- `run_inversion_cctbx_rruff.sh`: RRUFF experimental patterns.
- `run_delta_sweep.sh`: sweeps `--max-delta-pred`.

Useful flags: `--true-bravais`, `--rruff` (use `pattern_id` as the id column,
needed for RRUFF and CNRS), `--canonicalize`. See `--help` for the rest.

Output: `$POWDERXRD_DATA_ROOT/final_test_results_cctbx/<dataset>/`.

**4. Plots and analysis.** These read stage 3's pickles.
- `make_comparison_plots.py`: the per-dataset comparison figures (error
  distributions, CDFs, parity plots by Bravais type and crystal system).
- `failure_mode_analysis.py`: Bravais-type confusion and failure categories.
- `plot_volume_mape.py`, `plot_bispec_channel_error.py`, `plot_bispec_lat_error.py`:
  volume error, and lattice error against bispectrum prediction error.
- Dominant-zone analysis: `fetch_theoretical_flag.py` (adds the Materials
  Project `theoretical` flag to `mpfull/test.csv`, needs `mp-api`), then
  `filter_dominant_zones.py`, `dominant_zone_prediction_quality.py` and
  `compute_pattern_stats.py`.
- RRUFF filtering: `filter_rruff.py` cross-checks the RRUFF DIF and XY metadata
  and writes `rruff_filtered.csv`; `filter_inversion_results.py` restricts stage 3
  outputs to those entries.
- `analyze_completed_inversion_res.py`: quick summary of inversion outputs.

## Cost, leakage and paper figures

- Timing: `time_inversion.py` times the inversion per structure;
  `plot_time_inversion.py` and `plot_time_inversion_combined.py` plot the
  distribution (the combined version accounts for the cap-then-retry procedure).
  `time_bispec_database.py` measures the lookup database's size and cost.
- Train/test leakage: `mp20_xrd_leakage.py` and `mpfull_xrd_leakage.py` compute
  each test material's XRD-pattern similarity to its nearest training material;
  `plot_similarity_vs_error.py` plots error against that similarity.
  `analyze_start_leakage.py` measures how close the winning database start is to
  the true lattice (a retrieval-only baseline).
- `plot_bispec_distribution.py`: distribution of bispectrum coefficient values.
- `paper_bispec_figs.ipynb`: the bispectrum-per-crystal-system figure.
- `volumetric_plot.py`: the volumetric bispectrum-signal figure (writes an
  interactive plotly HTML file).

## Dependencies

`pip install -e .` from the repository root installs everything these scripts
need, including cctbx (the `cctbx-base` wheel on PyPI). The one exception is
`mp-api`, needed only by `download_mpfull_cifs.py` and `fetch_theoretical_flag.py`:
install it with `pip install -e ".[notebooks]"`.
