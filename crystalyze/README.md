# crystalyze/

Reproducing the **Crystalyze baseline predictions** that the `Crystalyze` series in
the paper's comparison plots is built from. Everything here runs *outside* this
repository, against a checkout of Crystalyze itself; the only thing this
directory contains is the glue: the scripts that build Crystalyze's input files
from our test sets, and the scripts that turn its output back into lattice
parameters keyed by material id. Those lattice parameters are then fed to
`bispectrum/inversion_results_cctbx.py` exactly like one of our own predictions
(`--crystalyze-params` / `--crystalyze-csv`), so that both models are reduced,
Bravais-classified and scored by the same code.

Crystalyze: <https://github.com/ML-PXRD/Crystalyze> (Ma et al.). It is a CDVAE
conditioned on an XRD pattern, so a "prediction" is a generated structure; we
use only its lattice.

## What gets produced

| run | input patterns | evaluate.py `--label` | consumed as |
|---|---|---|---|
| **MP-20 test set** | MP-20 test CIFs, patterns re-simulated with the Caglioti reference params (U=0.05, V=−0.06, W=0.07) | `full_data_uvw_force_atoms` | `cell_params_by_mpid_eval_recon_full_data_uvw_force_atoms.pt` → `--crystalyze-params`, the `Crystalyze` entry of the `mp20` comparison |
| **RRUFF (148)** | the 148 RRUFF minerals of the Crystalyze paper, patterns simulated from their RRUFF CIFs | `rruff_cag_force_na` | `eval_recon_rruff_cag.csv` → `--crystalyze-csv`, the `Crystalyze` entry of the `rruff_crystalyze` comparison |
| **MP-20 test set, our exact curves** | the `mp20_crystalyze_cag` test pickle itself, converted to Crystalyze's format | `mp20_cag_force_na` | cross-check that the MP-20 result is not an artifact of re-simulating the patterns |

The RRUFF comparison is reported on the 134 entries of
`training/splits/rruff_crystalyze_eval_ids.txt` (see the top-level README for
which 14 of the 148 are dropped and why); the filtering happens downstream, in
`bispectrum/make_comparison_plots.py`, not here.

All three are *reconstruction* runs (`--tasks recon`) with `--force_num_atoms`,
i.e. Crystalyze is told the true number of atoms. That is the most favorable
setting for the baseline, and the same information our models do not get.

## Prerequisites

1. **A Crystalyze checkout and its environment.** Follow their README
   (`conda env create -f env.yml`, `pip install -e .`). Copy `.env.template` to
   `.env` and set `PROJECT_ROOT` to the checkout; the hydra data configs resolve
   `${oc.env:PROJECT_ROOT}` through it.
2. **Their pretrained model folder**, from the Google Drive link in their README.
   `--model_path` must be a directory containing the checkpoint
   (`epoch=804-step=57154.ckpt` for the released model), `hparams.yaml`,
   `lattice_scaler.pt` and `prop_scaler.pt`. `evaluate.py` loads the
   highest-epoch `*.ckpt` in that directory, and writes its outputs there too.
   If loading fails on the GemNet `scale_file` path baked into the checkpoint's
   hparams, their `fix_checkpoint.py` rewrites it to the local
   `cdvae/pl_modules/gemnet/gemnet-dT.json`.

> On *potato* this was `/home/mink202/crystalyze/Crystalyze` with the venv
> `/data/NFS/potato/ehofgard/virtual_envs/crystalyze_env` and the model folder
> `their_model/`.

## 1. Build the input test sets

Crystalyze's dataloader wants three files per test set, in one directory:

- `test.csv` — the source CSV plus the columns their `preprocess()` expects
  (`xrd_peak_locations`, `xrd_peak_intensities`, `atomic_numbers`, `disc_sim_xrd`)
- `test.pt` — `{material_id: graph_arrays}`, CrystalNN graphs
- `test_pv_xrd.pt` — `{material_id: FloatTensor(1, 8500)}`, the pseudo-Voigt
  profile on `np.arange(5, 90, 0.01)`, which is the actual model input

`prepare_test_data.py` writes all three from a bare CSV that only has
`material_id` and `cif` (plus whatever property columns you already have):

```bash
cd <crystalyze-checkout>

# MP-20 test set, patterns re-simulated with the Caglioti reference params.
# --U/--V/--W default to 0.05 / -0.06 / 0.07, i.e. the values used for every
# simulated dataset in this paper; pass them explicitly to be sure.
python <this-repo>/crystalyze/prepare_test_data.py \
    --csv_path data/full_data/test.csv \
    --output_dir data/full_data_uvw \
    --U 0.05 --V -0.06 --W 0.07

# The 148 RRUFF minerals (CIFs from RRUFF; see the top-level README's Data section).
python <this-repo>/crystalyze/prepare_test_data.py \
    --csv_path $POWDERXRD_DATA_ROOT/rruff_crystalyze/test.csv \
    --output_dir $POWDERXRD_DATA_ROOT/rruff_crystalyze
```

Notes:

- `prepare_test_data.py` uses `XRDCalculator(wavelength='CuKa', symprec=0.1)`,
  Niggli-reduced conventional cells (`niggli=True, primitive=False`), the same
  convention as Crystalyze's own preprocessing.
- It carries its own copies of `build_crystal` / `build_crystal_graph`. If you
  need to rebuild only `test.pt` with the checkout's own versions (this is how
  `rruff_crystalyze/test.pt` was produced), use `build_graphs_from_csv.py`:

  ```bash
  python <this-repo>/crystalyze/build_graphs_from_csv.py \
      --crystalyze-dir <crystalyze-checkout> \
      --csv $POWDERXRD_DATA_ROOT/rruff_crystalyze/test.csv \
      --output $POWDERXRD_DATA_ROOT/rruff_crystalyze/test.pt
  ```

- For the third run, `test_pv_xrd.pt` is *not* re-simulated: it is a direct
  conversion of the test pickle our own MP-20 model is evaluated on
  (`mp20_crystalyze_cag/test_sim_xrd.pkl`, the output of
  `training/mp_full/generate_augmented_data.py --augmentation single --dataset mp20`),
  with `test.csv`/`test.pt` symlinked from the `full_data` set, so that
  Crystalyze sees bit-identical curves to ours:

  ```python
  import pickle, torch
  xrd = pickle.load(open(f'{DATA_ROOT}/mp20_crystalyze_cag/test_sim_xrd.pkl', 'rb'))
  torch.save({k: torch.tensor(v, dtype=torch.float32).unsqueeze(0) for k, v in xrd.items()},
             '<checkout>/data/mp20_cag/test_pv_xrd.pt')
  ```

Then register each directory as a hydra data config, because
`--test_set_override NAME` resolves to `<checkout>/conf/NAME.yaml`, not to a
path: copy `conf_data_template.yaml` to `<checkout>/conf/data/<name>.yaml` and
set `root_path` to the test-set directory (`${oc.env:PROJECT_ROOT}/data/<name>`
works for directories inside the checkout). The template is the config used for
the RRUFF run; `max_atoms: 20` and `batch_size: 256` matter.

## 2. Run Crystalyze

```bash
cd <crystalyze-checkout>

# MP-20 test set (9,046 entries -> 36 batches of 256), 1 Langevin sample each
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
    --model_path <checkout>/their_model \
    --tasks recon --num_batches 36 --num_evals 1 \
    --force_num_atoms \
    --test_set_override data/full_data_uvw \
    --label full_data_uvw_force_atoms

# 148 RRUFF minerals (a single batch), 5 Langevin samples
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
    --model_path <checkout>/their_model \
    --tasks recon --num_batches 1 --num_evals 5 \
    --force_num_atoms \
    --test_set_override data/rruff_cag \
    --label rruff_cag_force_na

# MP-20 test set on our own curves, 5 Langevin samples
CUDA_VISIBLE_DEVICES=0 python scripts/evaluate.py \
    --model_path <checkout>/their_model \
    --tasks recon --num_batches 999 --num_evals 5 \
    --force_num_atoms \
    --test_set_override data/mp20_cag \
    --label mp20_cag_force_na
```

`--num_batches` must be **at least** the number of batches in the loader
(`ceil(n_entries / 256)`), otherwise `reconstructon()` silently evaluates a
random `seed=42` subset of the batches instead of all of them — hence `999`
above for "all". Each run writes `their_model/eval_recon_<label>.pt`; the script
refuses to overwrite and appends `_1`, `_2`, … instead, so clear stale files
before re-running. Expect ~11 min per Langevin sample for the 148 RRUFF
entries on one A100, and correspondingly longer for the 9,046 MP-20 entries —
run them under `nohup`/`screen`.

`eval_recon_<label>.pt` holds `lengths` and `angles` of shape
`(num_evals, n_entries, 3)`, in the row order of the test CSV, plus the
`input_data_batch` with the ground-truth lattices.

## 3. Extract lattice parameters

Two output formats, both accepted by `inversion_results_cctbx.py`:

```bash
# .pt keyed by material_id, one Langevin sample (--eval_idx, default 0)
python <this-repo>/crystalyze/extract_cell_params.py \
    --recon_pt their_model/eval_recon_full_data_uvw_force_atoms.pt \
    --test_csv data/full_data_uvw/test.csv \
    --output   their_model/cell_params_by_mpid_eval_recon_full_data_uvw_force_atoms.pt

# CSV with pred_*/gt_* columns, averaged over the num_evals samples
python <this-repo>/crystalyze/export_preds_csv.py \
    --recon-pt their_model/eval_recon_rruff_cag_force_na.pt \
    --test-csv $POWDERXRD_DATA_ROOT/rruff_crystalyze/test.csv \
    --output   their_model/eval_recon_rruff_cag.csv \
    --reduce mean
```

The two differ in how they collapse the `num_evals` axis — `extract_cell_params.py`
takes one sample, `export_preds_csv.py --reduce mean` averages them. That matches
how each run was reported: the MP-20 run used `--num_evals 1` (so there is
nothing to average), the RRUFF run used `--num_evals 5` and was averaged. Use
`--reduce index --eval-idx N` if you want a single sample from a multi-eval run.

Both key the predictions by the `material_id` column of the test CSV, relying on
`evaluate.py` preserving test-loader order. If you ever evaluate a *subset* of
batches (`--num_batches` below the loader length), this alignment is wrong —
Crystalyze's own `combine_recon.py` exists to stitch such partial runs back into
CSV order first.

## 4. Score it alongside our models

From here the baseline goes through the same two stages as everything else
(see `bispectrum/README.md`):

```bash
cd ../bispectrum

# MP-20 test set (this is the invocation kept, commented out, at the bottom of
# run_inversion_cctbx_final.sh)
python inversion_results_cctbx.py \
    --input  $POWDERXRD_DATA_ROOT/final_test_results/mp20/test_mp20_bispec.h5 \
    --output $POWDERXRD_DATA_ROOT/final_test_results_cctbx/crystalyze_uvw/inversion_results_cctbx_crystalyze_uvw.pkl \
    --plot-dir $POWDERXRD_DATA_ROOT/final_test_results_cctbx/crystalyze_uvw/plots \
    --true-bravais $POWDERXRD_DATA_ROOT/mp20/test_true_bravais.pkl \
    --crystalyze-params <checkout>/their_model/cell_params_by_mpid_eval_recon_full_data_uvw_force_atoms.pt \
    --make-df --canonicalize

# RRUFF 148
python inversion_results_cctbx.py \
    --input  $POWDERXRD_DATA_ROOT/final_test_results/mp20_final_aug/rruff_crystalyze_direct_results.pkl \
    --output $POWDERXRD_DATA_ROOT/final_test_results_cctbx/rruff_crystalyze/crystalyze/inversion_results_cctbx.pkl \
    --plot-dir $POWDERXRD_DATA_ROOT/final_test_results_cctbx/rruff_crystalyze/crystalyze/plots \
    --true-bravais $POWDERXRD_DATA_ROOT/rruff_true_bravais.pkl \
    --crystalyze-csv <checkout>/their_model/eval_recon_rruff_cag.csv \
    --rruff --make-df --canonicalize
```

Three things to watch here:

- `--input` supplies only the *ids and the true lattices* to score against; the
  predictions come from `--crystalyze-params`/`--crystalyze-csv` for every id
  present in them. So pass whichever of our own result files covers the same
  entries — but it must be either an inverted `.h5` or a **direct**-prediction
  pickle, since a results dict is parsed as six lattice parameters per entry
  (a raw bispectrum pickle will not load).
- The output path matters: `make_comparison_plots.py` looks for
  `crystalyze_uvw/inversion_results_cctbx_crystalyze_uvw.pkl` and
  `rruff_crystalyze/crystalyze/inversion_results_cctbx.pkl`. The commented
  invocation in `run_inversion_cctbx_final.sh` writes to `crystalyze/` rather
  than `crystalyze_uvw/`; the command above uses the directory the plotting
  script actually reads.
- `--no-reduce-pred` skips the cctbx/Niggli reduction of the Crystalyze cells
  and scores the raw predicted lattice instead.

The output pickles are then picked up by name in
`bispectrum/make_comparison_plots.py` (`crystalyze_uvw/...` for the `mp20`
dataset, `rruff_crystalyze/crystalyze/...` for the `rruff_crystalyze` dataset):

```bash
python make_comparison_plots.py --datasets mp20 rruff_crystalyze
```

`--no-crystalyze` drops the baseline from the MP-20 figure.

## Files here

| file | purpose |
|---|---|
| `prepare_test_data.py` | bare CSV (`material_id`, `cif`) → `test.csv` + `test.pt` + `test_pv_xrd.pt` |
| `build_graphs_from_csv.py` | rebuild `test.pt` alone, using the checkout's own graph code |
| `conf_data_template.yaml` | hydra data config for `--test_set_override` |
| `extract_cell_params.py` | `eval_recon_*.pt` → `{material_id: {lengths, angles}}` `.pt` |
| `export_preds_csv.py` | `eval_recon_*.pt` → CSV with `pred_*`/`gt_*` columns |
