# crystalyze/

Reproducing the **Crystalyze baseline predictions** requires cloning the
Crystalyze repo, <https://github.com/ML-PXRD/Crystalyze>. This folder contains
pre/postprocessing scripts to interface with the Crystalyze model. Predicted
lattice parameters are supplied to `bispectrum/inversion_results_cctbx.py` with
the flag `--crystalyze-csv`.

The baseline is evaluated on the RRUFF dataset from the Crystalyze paper
(reported on the 134 in `training/splits/rruff_crystalyze_eval_ids.txt`), as a
reconstruction run with the true number of atoms given (`--force_num_atoms`).

## Setup

1. Install Crystalyze following their README, and set `PROJECT_ROOT` in `.env`
   to the checkout.
2. Download the pretrained model folder from the
   [Data section of their README](https://github.com/ML-PXRD/Crystalyze#data)
   (checkpoint `epoch=804-step=57154.ckpt`). If loading fails on the GemNet
   `scale_file` path, run their `fix_checkpoint.py`.

## 1. Build the test set

Make a `test.csv` with `material_id` and `cif` columns for the entries in
`training/splits/rruff_crystalyze_ids.txt`, then:

```bash
cd <crystalyze-checkout>
python <this-repo>/crystalyze/prepare_test_data.py \
    --csv_path $POWDERXRD_DATA_ROOT/rruff_crystalyze/test.csv \
    --output_dir $POWDERXRD_DATA_ROOT/rruff_crystalyze
```

Register the directory as a data config: copy `conf_data_template.yaml` to
`<checkout>/conf/data/rruff_cag.yaml` and set `root_path`.

## 2. Run Crystalyze and extract lattices

```bash
export POWDERXRD_DATA_ROOT=/path/to/your/data
export CRYSTALYZE_DIR=/path/to/Crystalyze   # model folder defaults to $CRYSTALYZE_DIR/their_model
CUDA_VISIBLE_DEVICES=0 bash run_crystalyze_predictions.sh
```

This runs `scripts/evaluate.py` (5 samples per mineral), then
`export_preds_csv.py`, which averages the samples into
`their_model/eval_recon_rruff_cag.csv`.

## 3. Score

```bash
cd ../bispectrum

python inversion_results_cctbx.py \
    --input  $POWDERXRD_DATA_ROOT/final_test_results/mp20_final_aug/rruff_crystalyze_direct_results.pkl \
    --output $POWDERXRD_DATA_ROOT/final_test_results_cctbx/rruff_crystalyze/crystalyze/inversion_results_cctbx.pkl \
    --plot-dir $POWDERXRD_DATA_ROOT/final_test_results_cctbx/rruff_crystalyze/crystalyze/plots \
    --true-bravais $POWDERXRD_DATA_ROOT/rruff_true_bravais.pkl \
    --crystalyze-csv <checkout>/their_model/eval_recon_rruff_cag.csv \
    --rruff --make-df --canonicalize

python make_comparison_plots.py --datasets rruff_crystalyze
```

`--input` only provides the ids and true lattices. Keep the output path as
shown, since `make_comparison_plots.py` reads from it.
