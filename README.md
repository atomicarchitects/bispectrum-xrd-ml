# bispectrum-xrd-ml

Code for *Learning Lattice Parameters from Powder X-Ray Diffraction Data Using Invariants*
(Hofgard et al., *Acta Cryst. A*, [doi:10.1107/S2053273326010284](https://doi.org/10.1107/S2053273326010284);
[arXiv:2607.21829](https://arxiv.org/abs/2607.21829)). Trained models and data:
[doi:10.5281/zenodo.24091709](https://doi.org/10.5281/zenodo.24091709). A transformer maps a powder
XRD pattern to a crystallographic bispectrum, which is inverted by L-BFGS to recover the lattice
parameters.

- **`utilities.py`**: XRD simulation, bispectrum computation, neighbor search, lattice comparison.
- **`training/`**: data generation, training and evaluation. See [training/README.md](training/README.md).
- **`bispectrum/`**: inversion, scoring with cctbx, and the paper's analyses. See [bispectrum/README.md](bispectrum/README.md).
- **`crystalyze/`**: reproducing the Crystalyze baseline we compare against. See [crystalyze/README.md](crystalyze/README.md).
- **`tutorials/`**: example notebooks.

## Installation

```bash
pip install -e .                # add ".[dev]" for pytest, ".[notebooks]" for mp-api
export POWDERXRD_DATA_ROOT=/path/to/your/data
```

Every script reads and writes under `$POWDERXRD_DATA_ROOT` (there is no default). Run the scripts
from their own directory, e.g. `cd training && python train.py ...`. Install the `notebooks` extra
in a separate environment: `mp-api` can force a `pymatgen` downgrade.

## Data

The paper uses the data below. The simulated datasets (test and validation sets, training targets and
peak lists, structures) are included in the archive (see
[Pretrained models and data](#pretrained-models-and-data)); the experimental data is not.

- **Materials Project** (full set and MP-20). Split ids are in `training/splits/`:
  `mpfull_{train,val,test}_ids.txt` (92,708 / 31,722 / 30,449; by reduced formula, 60/20/20, seed 42)
  and `mp20_{train,val,test}_ids.txt` (the standard CDVAE splits).
- **RRUFF** (<https://www.rruff.net/>; Lafuente et al., 2015). Entries are listed in
  `training/splits/rruff_alpha_ids.txt` (228) and `rruff_crystalyze_ids.txt` (148). The reported
  Crystalyze-set results use the 134 in `rruff_crystalyze_eval_ids.txt`: 7 entries overlap the MP-20
  training set and 7 have simulated rather than measured powder profiles.
- **opXRD**, CNRS/COD contribution. The 599 quality-controlled patterns used for evaluation are
  listed in `training/splits/cnrs_verified_ids.txt`.

The inversion's lookup database (`materials_project_lat_bispec.h5`) is built by
`bispectrum/build_lookup_database.py`. The paper's copy also contains the 1,922 MP-20 structures that
are not in the full-MP splits (156,801 entries; the script reproduces the other 154,879 exactly), and
is included in the archive.

## Pretrained models and data

Download from Zenodo, [doi:10.5281/zenodo.24091709](https://doi.org/10.5281/zenodo.24091709):
`main.tar`, `cnrs_noise.tar` and `data.tar`, plus the lookup database and `cob_lmax6.pkl`.

- **`main/`**: the manuscript's models, with dense input and RRUFF-fit noise augmentation. These give
  the MP test-set, RRUFF and AlphaDiffract results, and the "Dense" rows of the CNRS tables.

  | directory | training data (`data=`) |
  |---|---|
  | `mp20_cag_{bispec,direct}` | MP-20, Crystalyze Caglioti broadening (`mp20_crystalyze_cag`) |
  | `mp20_aug_{bispec,direct}` | MP-20 Aug: Caglioti + strain + texture (`mp20_aug_final`) |
  | `mpfull_cag_{bispec,direct}` | full MP, Crystalyze Caglioti broadening (`mpfull_crystalyze_cag`) |
  | `mpfull_aug_{bispec,direct}` | Full-MP Aug: Caglioti + strain + texture (`mpfull_aug_final`) |

- **`cnrs_noise/`**: the "+ CNRS noise" models of the opXRD CNRS ablation (see
  [training/README.md](training/README.md#cnrs-noise-models)): `mp20_dense_bispec`,
  `mp20_peaklist_{bispec,direct}` and `mpfull_peaklist_{bispec,direct}`.
  `mp20_peaklist_bispec/epoch51.pt` and `mp20_peaklist_direct/epoch36.pt` are the snapshots behind
  the reported CNRS numbers.
- **`data/`** mirrors the data root: extract it and set `POWDERXRD_DATA_ROOT` to it, and the data
  configs find their files. It holds `mp20_cag/`, `mp20_aug/`, `mp20_cnrs_noise/`, `mpfull_cag/`,
  `mpfull_aug/`, `mpfull_cnrs_noise/` and the split structures in `structures/`. The dense training
  patterns (about 64 GB) are left out; `training/mp_full/generate_augmented_data.py
  --struct-dir <data>/structures/<mp20|mpfull>` regenerates them.

Each model directory has `best_val.pt`, its configs, `xrd_norm_stats.json` (dense input) and
`lat_norm_stats.json` (direct models). Pass these to the eval scripts with `--norm_stats` and
`--lat_norm_stats` (`--norm-stats` and `--lat-norm-stats` in the CNRS scripts), so the training data
is not needed.

## Tests

```bash
pytest tests/
```

## Citation

```bibtex
@article{hofgard2026lattice,
  title   = {Learning Lattice Parameters from Powder X-Ray Diffraction Data Using Invariants},
  author  = {Hofgard, Elyssa and Min, Kyucheol and Segal, Nofit and Mittan-Moreau, David W. and
             Mansouri Tehrani, Aria and Oklejas, Vanessa and Gomez-Bombarelli, Rafael and Nigam, Jigyasa and Paley, Daniel W. and
             Brewster, Aaron S. and Smidt, Tess},
  journal = {Acta Crystallographica Section A},
  year    = {2026},
  doi     = {10.1107/S2053273326010284},
  eprint  = {2607.21829},
  archivePrefix = {arXiv}
}

@dataset{hofgard2026lattice_data,
  title     = {Pretrained models and data for "Learning Lattice Parameters from Powder X-Ray
               Diffraction Data Using Invariants"},
  author    = {Hofgard, Elyssa and Min, Kyucheol and Segal, Nofit and Mittan-Moreau, David W. and
               Mansouri Tehrani, Aria and Oklejas, Vanessa and Gomez-Bombarelli, Rafael and Nigam, Jigyasa and Paley, Daniel W. and
               Brewster, Aaron S. and Smidt, Tess},
  publisher = {Zenodo},
  year      = {2026},
  doi       = {10.5281/zenodo.24091709}
}
```
