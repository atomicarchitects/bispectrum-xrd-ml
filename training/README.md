# Bispectrum Prediction and Lattice Inversion from Powder XRD

A transformer neural network that maps powder XRD patterns to crystallographic bispectra, followed by L-BFGS inversion to recover lattice parameters.

## Pipeline Overview

```
XRD pattern (8500-pt, 5–90°)
        │
        ▼
Transformer encoder
        │
        ▼
Predicted bispectrum [10 × 35]
        │
        ▼
L-BFGS inversion  ──or──  Direct prediction [a,b,c,α,β,γ]
        │
        ▼
Lattice parameters (Niggli-reduced)
```

---

## Data Generation

All data generation utilities live in `../utilities.py`:

- `get_sim_xrd_from_preprocessed_results` — simulates XRD from a pymatgen Structure using CuKα, Caglioti broadening (Crystalyze reference: U=0.05, V=-0.06, W=0.07), pseudo-Voigt peaks, normalized to max=1
- `training_data_gen` — computes bispectrum [10×35] from the reciprocal lattice using `e3nn` spherical harmonics + Bessel radial basis (l_max=6, n_basis=10, k_max=2/3)
- `get_bispectrum` / `get_neighbors` — lower-level bispectrum utilities
- `caglioti_fwhm` / `seed_from_mpid` — shared by the `mp_full/` scripts below

Data generation scripts (`mp_full/`, plus `prep_opxrd_cnrs.py` in `training/`):
- `data_gen_full.py --mode {xrd,bispec}` — splits `materials_project_structures.pkl` by reduced formula (60/20/20, seed 42) into `{split}_data.pkl`, then generates, for each split, simulated XRD (`--mode xrd`, default → `{split}_data_sim_xrd.pkl`) or bispectra (`--mode bispec` → `{split}_data_bispec_list.pkl`). Reads from and writes into `$POWDERXRD_DATA_ROOT`, which must be set.
- `generate_augmented_data.py --dataset {mpfull,mp20} --augmentation {strain,texture,caglioti,caglioti4x,single,combine}` — all augmentation types, one CLI (was 5 separate near-duplicate scripts):
  - `strain` — symmetry-preserving lattice strain, + recomputed bispectrum
  - `texture` — texture (preferred-orientation) augmentation (`--max-texture 0.5` for the real `mpfull_texture_cag`/`mp20_texture_cag` data, not the script's own former default of 0.3)
  - `caglioti` — Caglioti-only, 4 fixed (U,V,W) sets, no intensity perturbation
  - `caglioti4x` — same 4 sets + Crystalyze-recipe intensity perturbation. No confirmed default `--output-dir` for `--dataset mp20` (several candidate directories exist but none matches the expected filename pattern); required explicitly.
  - `single` — one fixed Caglioti set (Crystalyze reference); the **training** base (`mpfull_crystalyze_cag`/`mp20_crystalyze_cag`)
  - `combine` — merges separate `caglioti`/`strain`/`texture` runs (disjoint key namespaces) into one `{split}_*_combined.pkl` training set, e.g. `mp20_aug_cag_strain_texture`
  - `peaklist_from_xrd` — peak-list input variant: extracts up to `--peaklist-max-peaks` (30) peaks from an *existing* dense-XRD pickle (`--xrd-pkl-dir`, `--xrd-pkl-pattern` with a `{split}` placeholder), without re-simulating. With `--bg-noise-cnrs-mix-prob P` it first adds one Gamma background-noise realization per entry, drawn from the CNRS-fit profile with probability `P` and the RRUFF-fit profile otherwise (`P=0.5` is the 50/50 mix, `1.0` CNRS only, `0.0` RRUFF only), and writes `{split}_peaklist_bgmix.pkl`; without it, `{split}_peaklist.pkl`. The noise is a training-set device only: validation and test peak lists are extracted from the clean curves.
  `--dataset` only selects defaults for `--struct-dir`/`--bispec-dir`/`--output-dir`; explicit flags always override.
- `bg_noise_aug.py` — train-time background-noise augmentation (called from `train.py`)
- `fit_intensity_gamma.py` — fits the intensity-perturbation params from RRUFF
- `filter_alpha_diffract.py` — filters `{split}_data.pkl` by atom count/volume
- `gen_cag_texture_prim_lat.py` — lattice-param labels for `*_cag`/`*_texture_cag` keys
- `gen_rruff_alpha_eval.py` — builds the RRUFF-alpha eval pkl files
- `prep_opxrd_cnrs.py` — preprocesses the opxrd CNRS experimental dataset
- `fit_bg_noise_cnrs.py` — fits the CNRS background-noise profile used by `bg_noise_aug.py`
- `compute_xrd_norm_stats.py --data <conf/data name>` — streams the mean/std of a dataset's training XRD pickle to `<DATA_DIR>/<data>_xrd_norm_stats.json`; dense-input checkpoints must be evaluated with the statistics of the data they were trained on

---

## Model

**`model.py`** — `XRDTransformerEncoder`

- Input, `input_type: dense`: 8500-point XRD pattern (5–90°, 0.01° step), normalized by training set mean/std
- Input, `input_type: sparse` (peak list): up to `max_peaks` (30) peaks as (position, intensity) pairs (`use_intensity: false` for positions only)
- Architecture: tokenization MLP → transformer encoder → linear head
- Output modes:
  - **Bispectrum**: [10×35] tensor (350 values)
  - **Direct**: [6] tensor = [a, b, c, α, β, γ] in Å/degrees

Model size configs (`conf/model/`):

| Config | d_model | h_dim | n_head | n_layers |
|--------|---------|-------|--------|----------|
| `transformerbispec.yaml` | 256 | 512 | 8 | 12 |
| `transformerbispec_medium.yaml` | 320 | 640 | 8 | 12 |
| `transformerbispec_large.yaml` | 384 | 756 | 12 | 12 |
| `transformerdirect.yaml` | 256 | 512 | 8 | 12 |
| `transformerbispec_sparse.yaml` (peak list) | 256 | 512 | 8 | 12 |
| `transformerdirect_sparse.yaml` (peak list) | 256 | 512 | 8 | 12 |

`transformerbispec_sparse_posonly.yaml` is the position-only variant of the peak-list bispectrum model.

---

## Training

All `conf/data/*.yaml` configs resolve `DATA_DIR` as `${oc.env:POWDERXRD_DATA_ROOT}/<dataset subpath>`.
`POWDERXRD_DATA_ROOT` must be set (there is no default): point it at wherever you keep the
data, preserving the subdirectory names (e.g. `mp20_crystalyze_cag/`), and every config's
`DATA_DIR` follows automatically:

```bash
export POWDERXRD_DATA_ROOT=/path/to/your/data
export POWDERXRD_REPO_DIR=/path/to/this/checkout   # only for the SLURM scripts
```

The SLURM scripts also contain a placeholder account (`#SBATCH -A YOUR_ACCOUNT`).

```bash
cd training/
python train.py \
    --config-name=config \
    data=mpfull_aug_final \
    model=transformerbispec \
    train.use_wandb=True \
    hydra.run.dir=$POWDERXRD_DATA_ROOT/hydra_outputs
```

For self-resubmitting jobs (30 epochs per SLURM job, up to 1000 epochs):
```bash
sbatch train_mp20bispec_resubmit.sh [data] [model] [checkpoint_path]
sbatch train_resubmit.sh [data] [model] [checkpoint_path]
```

Checkpoints are saved to:
`$POWDERXRD_DATA_ROOT/{data}/{model}/models/{arch_string}.pt`

Key data configs (`conf/data/`):
- `mpfull_aug_final` — full MP with strain + texture augmentation (canonical)
- `mpfull_crystalyze_cag` — full MP with Caglioti-matched simulation
- `mpfull_aug_final_direct` — same data for direct prediction model
- `mpfull_aug_final_peaklist_bgmix`, `mp20aug_peaklist_bgmix` — peak-list input with CNRS/RRUFF-mixed background noise (use with `model=transformerbispec_sparse`)
- `mp20aug_peaklist_cnrsonly` — the same with the CNRS noise profile only
- `*_direct` variants of the above regress the lattice directly (use with `model=transformerdirect_sparse`, `--config-name=config_direct_pred`)

### Peak-list recipe

```bash
cd training/mp_full
# 1. train peak lists with the 50/50 CNRS/RRUFF background noise, from the existing dense curves
python generate_augmented_data.py --augmentation peaklist_from_xrd --dataset mp20 \
    --xrd-pkl-dir $POWDERXRD_DATA_ROOT/mp20_aug/all_aug \
    --xrd-pkl-pattern 'all_aug_{split}_data_sim_xrd.pkl' \
    --output-dir $POWDERXRD_DATA_ROOT/mp20_aug/all_aug \
    --splits train --bg-noise-cnrs-mix-prob 0.5 --bg-noise-seed 0
# 2. clean validation/test peak lists
python generate_augmented_data.py --augmentation peaklist_from_xrd --dataset mp20 \
    --xrd-pkl-dir $POWDERXRD_DATA_ROOT/mp20_aug/all_aug \
    --xrd-pkl-pattern '{split}_data_sim_xrd.pkl' \
    --output-dir $POWDERXRD_DATA_ROOT/mp20_aug/all_aug --splits val test
# 3. train
cd .. && python train.py --config-name=config data=mp20aug_peaklist_bgmix \
    model=transformerbispec_sparse +train.snapshot_every_epochs=3 \
    hydra.run.dir=$POWDERXRD_DATA_ROOT/hydra_outputs
```

Step 1 writes `train_peaklist_bgmix.pkl` and step 2 `val_peaklist.pkl`/`test_peaklist.pkl`, the files
the `*_peaklist_bgmix` configs expect. The generator always uses these names, so a variant
with a different noise profile (e.g. `mp20aug_peaklist_cnrsonly`) must go in its own directory.
`+train.snapshot_every_epochs=N` additionally saves a checkpoint every N epochs under
`.../models/snapshots/`. `train_mp20_peaklist_bgmix.sh` and the other `train_*peaklist*.sh` scripts
wrap this for SLURM.

---

## Evaluation

### MP test set

```bash
python eval.py --config-name=config data=mpfull_aug_final model=transformerbispec ...
```

### RRUFF alpha (real experimental, 228 minerals)

```bash
python eval_rruff_alpha.py \
    --model_path /path/to/model.pt \
    --train_data_dir /path/to/train_data \
    --train_xrd_file train_sim_xrd_combined.pkl \
    --train_lat_file train_bispec_combined.pkl \
    --prediction_type bispectrum \
    --model_config conf/model/transformerbispec.yaml \
    --train_config conf/train/default.yaml \
    --output_path /path/to/results.pkl
```

Use `--eval_xrd_pkl` / `--eval_lat_pkl` to evaluate on a custom dataset (e.g. opxrd CNRS).

Eval scripts for specific model sets: `run_rruff_alpha_eval.sh`, `run_rruff_crystalyze_eval.sh`

### Real data: opXRD CNRS

Data lives in `$POWDERXRD_DATA_ROOT/opxrd_cnrs_eval/`; the raw per-pattern JSON files go in its `CNRS/` subdirectory.

```bash
python prep_opxrd_cnrs.py --source raw          # -> opxrd_cnrs_xrd.pkl, opxrd_cnrs_lat.pkl (primitive-cell lattice labels)
python split_cnrs_selection.py --xrd-pkl ... --lat-pkl ... --out-dir ...   # disjoint selection / reporting slices
python compute_cnrs_true_bispec.py              # -> eval_results/opxrd_cnrs/opxrd_cnrs_true_bispec.pkl
python count_cnrs_primitive_atoms.py            # -> opxrd_cnrs_prim_atom_counts.pkl (atoms in the primitive cell)
python build_cnrs_selection_inversion_input.py \
    --name my_run --slice {selection,reporting,all} --input-type {dense,sparse} \
    --checkpoint-path /path/to/model.pt
```

The last step runs the checkpoint on the chosen patterns and writes
`eval_results/opxrd_cnrs/inversion_inputs/<name>.pkl`, which `bispectrum/run_alg_inversion.py`
takes as `-i`. Options: `--allowed-from-checkpoint` (use the structured-head mask stored in the
checkpoint, needed for checkpoints trained on a different dataset such as mp20aug),
`--norm-stats` (training-set normalization for dense models, from `compute_xrd_norm_stats.py`),
and `--verified-max-atoms N [--primitive]` (restrict to the quality-controlled patterns in
`splits/cnrs_verified_ids.txt` with at most N atoms, counted in the primitive cell with `--primitive`).
`eval_opxrd_cnrs_selection_cosine.py` and `track_snapshot_cosine.py` give a cheap cosine-similarity
screen of checkpoints against the true bispectra on the selection slice, before spending
inversion compute.

`prep_opxrd_cnrs.py --source aggregated` reads a single pre-curated JSON instead (path from
`CNRS_JSON_PATH` or `--json-path`; not included). The selection/reporting split exists so that
a checkpoint can be chosen on one slice and the reported numbers computed on the other.

---

## Bispectrum Inversion

L-BFGS optimization to recover a reciprocal lattice matrix from a predicted bispectrum. Located in `../bispectrum/run_alg_inversion.py`.

```bash
cd ../bispectrum/
python run_alg_inversion.py \
    -i /path/to/bispec_results.pkl \
    -o /path/to/output.h5 \
    -mat_proj_df $POWDERXRD_DATA_ROOT/materials_project_lat_bispec.h5 \
    -cob $POWDERXRD_DATA_ROOT/cob_lmax6.pkl \
    -begin_ind 0 -num_entries N \
    -max_workers 32 \
    --random_restarts 30 \
    --timeout 500
```

Key flags:
- `-mat_file` — text file of material IDs to process (for retrying missing entries)
- `--random_restarts` — extra random triclinic starting points (useful for RRUFF/opxrd where MP neighbors aren't available)
- `--num_to_opt` — number of MP-neighbor candidates to try (default 15)
- `--timeout` — wall time in seconds for the whole material (all candidates combined)
- `-candidate_timeout` — per-candidate L-BFGS budget in seconds; keep `-timeout` at least `(num_to_opt + random_restarts) * candidate_timeout` or a material can be cut off before every start is tried
- `-random_restart_max_abc` — upper bound (Å) on the edge lengths of the random starting cells; widen it for datasets with long cells
- `-random_restarts 0` — seed only from the database neighbors (no random starts)

The opXRD CNRS runs used `-num_to_opt 30 -random_restarts 10 -random_restart_max_abc 55 -candidate_timeout 30 -timeout 1500`.
The search skips materials already present in the output file, so an interrupted run can be resumed with the same command.

Output H5 format — each key is a material ID with datasets: `pred` (3×3 real-space lattice), `true_lattice` (3×3 Niggli-reduced), `start_matrix` (3×3 starting reciprocal lattice), plus `min_loss` (final bispectrum loss) and the attribute `start_id`. `start_id` and `start_matrix` describe the start that produced `pred` (`start_id` is `None` for a random start). Result files written by earlier versions of the script recorded the *last* candidate tried instead, so their `start_id` does not identify the winner.

For chunked SLURM inversion and merging chunks:
```bash
bash run_alg_single_chunks.sh          # submits 500-entry jobs
python merge_inv_chunks.py \
    --chunks-dir /path/to/inv_chunks \
    --output /path/to/merged.h5
```

### Analyzing results

```bash
python ../bispectrum/analyze_completed_inversion_res.py
```

Applies Niggli reduction to both pred and true before computing MAE per parameter.

### Lookup database

The database that `-mat_proj_df` points to is built with `../bispectrum/build_lookup_database.py`
(see the top-level README for its contents), and `../bispectrum/time_bispec_database.py` measures its
size, per-entry construction time and per-query search time. Each query is an exhaustive search
over the database that excludes only the query material itself, so related structures, including
validation and test entries, remain available as starting points.

### Leakage audit

`../bispectrum/mp20_xrd_leakage.py` and `mpfull_xrd_leakage.py` compare the simulated XRD pattern of
every validation/test material with every training material by cosine similarity and record, per
material, the best match and its similarity (threshold 0.999), in
`$POWDERXRD_DATA_ROOT/eval_results/{mp20,mpfull}_xrd_leakage_audit.pkl`.
`plot_similarity_vs_error.py` plots the similarity distribution and the relation between
similarity and prediction error.
