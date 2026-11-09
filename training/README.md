# training/

Data generation, training and evaluation of the transformer that maps an XRD pattern to a
bispectrum (`transformerbispec*`) or directly to the lattice parameters (`transformerdirect*`).
Bispectrum predictions are turned into lattices by the inversion in `../bispectrum/`.

Configs are Hydra (`conf/`). Every `conf/data/*.yaml` reads from `$POWDERXRD_DATA_ROOT/<dataset>/`.

## 1. Generate data

Scripts are in `mp_full/`. Starting from the Materials Project structures
(`$POWDERXRD_DATA_ROOT/materials_project_structures.pkl`):

```bash
cd mp_full
python data_gen_full.py --mode xrd       # full MP: split by reduced formula, simulate patterns
python data_gen_full.py --mode bispec    # full MP: bispectrum targets
python generate_augmented_data.py --dataset {mp20,mpfull} --augmentation <type>
```

`--augmentation` is one of:
- `single`: one Caglioti broadening (the Crystalyze reference); gives `mp20_cag`, `mpfull_cag`.
- `caglioti`, `strain`, `texture`: the augmentations. `combine` merges them into the training sets of
  `mp20_aug`, `mpfull_aug`.
- `peaklist_from_xrd`: peak-list inputs (up to 30 peaks) from existing dense patterns. With
  `--bg-noise-cnrs-mix-prob 0.5 --bg-noise-cnrs-profile cnrs_wlfix` it adds the mixed RRUFF/CNRS
  background noise before extracting the peaks (training split only).

## 2. Train

```bash
python train.py data=mp20_aug_final model=transformerbispec
python train.py --config-name=config_direct_pred data=mp20_aug_final_direct model=transformerdirect
```

Checkpoints go to `$POWDERXRD_DATA_ROOT/<data.name>/<model.type>/models/`. To resume, add
`train.load_pretrained=True train.pretrained_path=<checkpoint>`; `+train.snapshot_every_epochs=3` also
keeps periodic snapshots. The paper's runs used `train.epochs=1000 train.early_stopping=True`.

Model configs: `transformerbispec`, `transformerdirect` (dense input) and `*_sparse` (peak-list input).
All four use d_model 256, 12 layers and 8 heads.

### CNRS-noise models

These are the "+ CNRS noise" models of the opXRD CNRS ablation. Their background noise is drawn half
from the RRUFF-fit profile and half from the CNRS-fit profile `cnrs_wlfix`. (The original `cnrs`
profile was fit to patterns with an inverted wavelength conversion, since fixed in
`prep_opxrd_cnrs.py`. It is kept only to reproduce older models.) Dense models add the noise during
training; peak-list models need training peak lists built with it (step 1).

| model | `data=` | `model=` | other overrides |
|---|---|---|---|
| `mp20_dense_bispec` | `mp20aug_dense_cnrsmix_wlfix` | `transformerbispec` | `train=default_cnrs_mix +train.bg_noise_cnrs_profile=cnrs_wlfix` |
| `mp20_peaklist_bispec` | `mp20aug_peaklist_bgmix_wlfix` | `transformerbispec_sparse` | |
| `mp20_peaklist_direct` | `mp20aug_peaklist_bgmix_direct_wlfix` | `transformerdirect_sparse` | `--config-name=config_direct_pred` |
| `mpfull_peaklist_bispec` | `mpfull_aug_final_peaklist_bgmix_wlfix` | `transformerbispec_sparse` | |
| `mpfull_peaklist_direct` | `mpfull_aug_final_peaklist_bgmix_direct_wlfix` | `transformerdirect_sparse` | `--config-name=config_direct_pred` |

## 3. Evaluate

With a released checkpoint, pass its `xrd_norm_stats.json` and, for direct models,
`lat_norm_stats.json`. Without them, the statistics are recomputed from the training data.

**MP test set**

```bash
python eval.py --model_path <ckpt> --prediction_type {bispectrum,direct} \
    --data_config conf/data/mp20_aug_final.yaml --model_config conf/model/transformerbispec.yaml \
    --train_config conf/train/default.yaml --test_bispec_file test_data_bispec_list.pkl \
    --norm_stats <xrd_norm_stats.json> --output_path <out>.pkl
```

For direct models use `test_data_prim_lat.pkl`, the `*_direct` data config,
`transformerdirect.yaml`, `default_direct.yaml` and `--lat_norm_stats`.

**RRUFF, AlphaDiffract set** (228 entries): `mp_full/gen_rruff_alpha_eval.py` builds the inputs, then

```bash
python eval_rruff_alpha.py --model_path <ckpt> --prediction_type bispectrum \
    --model_config conf/model/transformerbispec.yaml --norm_stats <xrd_norm_stats.json> --output_path <out>.pkl
```

Peak-list models use `eval_rruff_alpha_peaklist.py`.

**RRUFF, Crystalyze set** (148 entries, 134 reported): build the inputs once, then evaluate.

```bash
python build_rruff_crystalyze_patterns.py --xy-dir <RRUFF XY_Processed dir>   # -> rruff_measured_xrd.pt
python build_rruff_crystalyze_targets.py     # from Crystalyze's test.csv -> rruff_crystalyze_targets.pkl
python eval_rruff_crystalyze.py --model_path <ckpt> --norm_stats <xrd_norm_stats.json> --output_path <out>.pkl
```

The patterns match the paper's for 144 entries. RRUFF has since changed the processed files for
Stephanite R050143, Pyrargyrite R050318, Senarmontite R060002 and Proustite R060356.

**opXRD CNRS** (raw JSON files in `$POWDERXRD_DATA_ROOT/opxrd_cnrs_eval/CNRS/`)

```bash
python prep_opxrd_cnrs.py --source raw      # patterns on the model's grid + primitive-cell lattices
python split_cnrs_selection.py ...          # selection slice (checkpoint choice) / reporting slice
python count_cnrs_primitive_atoms.py
python build_cnrs_selection_inversion_input.py --name <run> --slice all --input-type {dense,sparse} \
    --checkpoint-path <ckpt> --allowed-from-checkpoint --verified-max-atoms 20 --primitive \
    [--norm-stats <xrd_norm_stats.json>]
```

This writes a bispectrum prediction file for `../bispectrum/run_alg_inversion.py`.
`--verified-max-atoms 20 --primitive` gives the small-cell subset. For all verified patterns, pass a
large limit (e.g. `--verified-max-atoms 100000`). Without the flag, no verified filter is applied. Direct models use `eval_cnrs_direct.py`
(`--lat-norm-stats`). `track_snapshot_cosine.py` screens snapshots on the selection slice before
inversion.
