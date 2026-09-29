"""
Track cosine similarity (pred vs true bispec, opXRD CNRS selection slice)
across a run's saved snapshots over time, so real-domain overfitting can be
caught even while clean val_loss keeps improving (see eval_opxrd_cnrs_
selection_cosine.py's docstring for the screening methodology this reuses).

Maintains a small JSON log (keyed by run name -> epoch) so re-running this
only evaluates snapshots/live-checkpoints not already logged -- cheap to
call again anytime a new snapshot lands (every snapshot_every_epochs epochs).

Usage:
    python track_snapshot_cosine.py                      # all runs
    python track_snapshot_cosine.py --runs peaklist       # one run
"""
import argparse
import glob
import json
import os
import re

import numpy as np
import torch

from eval_opxrd_cnrs_selection_cosine import load_context, eval_checkpoint

_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
LOG_PATH = f"{_ROOT}/eval_results/opxrd_cnrs/snapshot_cosine_log.json"
CKPT_FILENAME = "nlayers12_nhead8_dm256_hd512_lossL1_smallvalsFalse_masklossTrue.pt"

RUNS = {
    "peaklist": dict(
        model_dir=f"{_ROOT}/mpfull_aug_final_peaklist/transformerbispec/models",
        input_type="sparse",
    ),
    "peaklist_bgmix": dict(
        model_dir=f"{_ROOT}/mpfull_aug_final_peaklist_bgmix/transformerbispec/models",
        input_type="sparse",
    ),
    "cnrs_mix": dict(
        model_dir=f"{_ROOT}/mpfull_aug_final_cnrs_mix/transformerbispec/models",
        input_type="dense",
    ),
}

_EPOCH_RE = re.compile(r"epoch(\d+)_")


def _candidates(model_dir):
    """Yield (epoch_or_None, path) for the live checkpoint and every
    snapshot in model_dir -- epoch is read from the checkpoint itself for
    the live one (unknown up front) and parsed from the filename for
    snapshots (cheap, avoids a torch.load just to list candidates)."""
    live = os.path.join(model_dir, CKPT_FILENAME)
    if os.path.exists(live):
        yield None, live
    for path in sorted(glob.glob(os.path.join(model_dir, "snapshots", "epoch*_*.pt"))):
        m = _EPOCH_RE.search(os.path.basename(path))
        if m:
            yield int(m.group(1)), path


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", default=list(RUNS.keys()), choices=list(RUNS.keys()))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    log = {}
    if os.path.exists(LOG_PATH):
        log = json.load(open(LOG_PATH))

    ctx = load_context(device=args.device)

    for run in args.runs:
        spec = RUNS[run]
        run_log = log.setdefault(run, {})
        print(f"\n=== {run} ===")
        if not os.path.isdir(spec["model_dir"]):
            print(f"  [skip] {spec['model_dir']} does not exist yet")
            continue

        for epoch_hint, path in _candidates(spec["model_dir"]):
            if epoch_hint is not None and str(epoch_hint) in run_log:
                continue  # already logged this exact snapshot epoch
            try:
                epoch, val_loss, sims = eval_checkpoint(path, spec["input_type"], ctx)
            except Exception as e:
                print(f"  [FAIL] {path}: {e}")
                continue
            if epoch is None:
                continue
            key = str(epoch)
            if key in run_log:
                continue  # live checkpoint landed on an epoch already logged via its snapshot
            run_log[key] = dict(
                val_loss=val_loss, mean=float(np.mean(sims)), median=float(np.median(sims)),
                min=float(np.min(sims)), max=float(np.max(sims)), n=len(sims), source=path,
            )
            print(f"  epoch {epoch:>4}  val_loss={val_loss}  "
                  f"mean={run_log[key]['mean']:.4f}  median={run_log[key]['median']:.4f}  "
                  f"min={run_log[key]['min']:.4f}")

        os.makedirs(os.path.dirname(LOG_PATH), exist_ok=True)
        with open(LOG_PATH, "w") as f:
            json.dump(log, f, indent=2)

        if run_log:
            print(f"\n  -- {run} trend (all logged epochs) --")
            print(f"  {'epoch':>6} {'val_loss':>10} {'cos_mean':>9} {'cos_median':>11} {'cos_min':>8}")
            for epoch in sorted(run_log, key=int):
                r = run_log[epoch]
                vl = f"{r['val_loss']:.4f}" if r["val_loss"] is not None else "?"
                print(f"  {epoch:>6} {vl:>10} {r['mean']:>9.4f} {r['median']:>11.4f} {r['min']:>8.4f}")

    print(f"\nLog saved -> {LOG_PATH}")


if __name__ == "__main__":
    main()
