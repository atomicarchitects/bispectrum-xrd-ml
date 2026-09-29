"""
Quantitative timing of the L-BFGS bispectrum-to-lattice inversion (run_alg_inversion.py's
process_row), for the reviewer comment asking for a real distribution rather than the vague
"usually limited to less than 30 seconds per structure".

Runs process_row in PARALLEL across --max-workers processes (same ProcessPoolExecutor +
init_globals pattern run_alg_inversion.py itself uses for production runs), each still
single-threaded internally (OMP/MKL/OPENBLAS pinned to 1). Each material's own wall-clock time
is timed individually inside its worker. This deliberately measures realistic wall-clock cost
AS EXPERIENCED IN PRACTICE (including contention from other workers sharing the node), which is
what "practical cost of inversion" means, and matches how every real run in this project is
actually used -- not an idealized, contention-free single-core number. Use --max-workers 1 to
recover the old fully-isolated-timing behavior if that's ever wanted instead.

Two settings are timed, since they differ substantially in candidate budget:
  default   -num_to_opt 15 -random_restarts 0  -candidate_timeout 10  (run_alg_inversion.py's own
             argparse defaults; confirmed, via training/run_mp_remainder_inversion.sh, to be what
             actually produced the reported MP-Full test-set inversion results)
  wide      -num_to_opt 30 -random_restarts 10 -candidate_timeout 30  (used for the CNRS/RRUFF
             real-data evaluations elsewhere in this project)

Usage:
    python time_inversion.py --n 300 --settings default --max-workers 32
    python time_inversion.py --n 300 --settings default --max-workers 32 \\
        --predictions-file .../mp20_final_aug/bispec_test_results.pkl --tag mp20
"""
import argparse
import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"

import contextlib
import io
import pickle
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import torch
from tqdm import tqdm

import run_alg_inversion as inv

_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
# --predictions-file / --tag select which test set to time; defaults to MP-Full's, the one
# whose settings were confirmed (training/run_mp_remainder_inversion.sh) to match `default` below.
BISPEC_PREDICTIONS = f"{_ROOT}/final_test_results/mpfull_aug/bispec_test_results.pkl"
MAT_PROJ_DF = f"{_ROOT}/materials_project_lat_bispec.h5"
COB_PATH = f"{_ROOT}/cob_lmax6.pkl"

SETTINGS = {
    "default": dict(num_to_opt=15, random_restarts=0, candidate_timeout=10),
    "wide": dict(num_to_opt=30, random_restarts=10, candidate_timeout=30,
                 random_restart_max_abc=55.0),
}


def _time_one_worker(args):
    """Top-level (picklable) function run inside each pool worker. `inv.init_globals` has
    already populated this process's copies of ALL_BISPECS/ALL_MP_IDS/ALL_RECLAT/COB_BASES
    via the pool's initializer, exactly as run_alg_inversion.py's own workers do."""
    mp_id, bispec, wavelength, lmax, n_basis, radial, kw = args
    kw = dict(kw)
    t0 = time.perf_counter()
    res = inv.process_row(
        mp_id, bispec.copy(), radial, n_basis, wavelength, lmax,
        timeout_seconds=kw.pop("candidate_timeout"), **kw,
    )
    elapsed = time.perf_counter() - t0
    return mp_id, elapsed, res["min_loss"]


def _time_one_capped(task):
    """Same path as run_alg_inversion.py's first (parallel) pass: worker_wrapper runs the structure in its
    own process and kills it after `cap` seconds (the script's -timeout), with the same 17-element argument
    tuple that main() builds. Returns (mp_id, wall time, min_loss or nan, timed_out)."""
    mp_id, bispec, wavelength, lmax, n_basis, radial, kw, cap = task
    kw = dict(kw)
    worker_args = (mp_id, bispec.copy(), radial, n_basis, wavelength, lmax, None,
                   kw.get("random_restarts", 0), kw.get("num_to_opt", 15), False, 150, False, 20, 0.1,
                   kw.get("random_restart_min_abc", 3.0), kw.get("random_restart_max_abc", 15.0),
                   kw.get("candidate_timeout", 10))
    t0 = time.perf_counter()
    try:
        # worker() prints every result dict; discard that output (the forked child inherits this redirect)
        with contextlib.redirect_stdout(io.StringIO()):
            res = inv.worker_wrapper(worker_args, cap)
        min_loss = res["min_loss"] if res is not None else float("nan")
        timed_out = False
    except TimeoutError:
        min_loss, timed_out = float("nan"), True
    except Exception:
        min_loss, timed_out = float("nan"), False
    return mp_id, time.perf_counter() - t0, min_loss, timed_out


def _save(path, payload):
    """Atomic write (temp file + rename) so a kill mid-write cannot corrupt an existing pkl."""
    tmp = path + ".tmp"
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    os.replace(tmp, path)


def summarize(label, times, losses, converge_cutoff=1e-2):
    t = np.array(times)
    print(f"\n{label}: n={len(t)}")
    print(f"  mean {t.mean():.2f}s  median {np.median(t):.2f}s  "
          f"p90 {np.percentile(t, 90):.2f}s  p95 {np.percentile(t, 95):.2f}s  "
          f"p99 {np.percentile(t, 99):.2f}s  max {t.max():.2f}s  min {t.min():.2f}s")
    for thresh in (5, 10, 30, 60, 120):
        print(f"  fraction under {thresh:3d}s: {100 * np.mean(t < thresh):.1f}%")
    conv = np.array(losses) < converge_cutoff
    print(f"  converged (min_loss < {converge_cutoff:g}): {100 * conv.mean():.1f}%")
    if conv.any():
        print(f"    converged-only: mean {t[conv].mean():.2f}s  median {np.median(t[conv]):.2f}s")
    if (~conv).any():
        print(f"    not converged:  mean {t[~conv].mean():.2f}s  median {np.median(t[~conv]):.2f}s  "
              f"(these ran the full candidate budget)")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=300)
    p.add_argument("--settings", choices=["default", "wide", "both"], default="both")
    p.add_argument("--max-workers", type=int, default=32,
                   help="parallel worker processes -- the wall-clock times reported reflect "
                        "contention from running this many at once, by design (see module docstring)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--cap", type=float, default=0,
                   help="per-structure wall-clock cap in seconds, enforced exactly as run_alg_inversion.py's "
                        "-timeout does (process killed; structure counts as timed out). 0 = no cap")
    p.add_argument("--ids-file", default=None,
                   help="time exactly these material ids (one per line) instead of a random sample, "
                        "e.g. to rerun structures that did not finish in an earlier run")
    p.add_argument("--wavelength", type=float, default=3.0)
    p.add_argument("--lmax", type=int, default=6)
    p.add_argument("--n-basis", type=int, default=10)
    p.add_argument("--radial", default="bessel")
    p.add_argument("--predictions-file", default=BISPEC_PREDICTIONS,
                   help="raw bispectrum predictions pkl ({material_ids, predictions, targets}), "
                        "e.g. .../final_test_results/mp20_final_aug/bispec_test_results.pkl for MP-20 "
                        "instead of the MP-Full default")
    p.add_argument("--tag", default="mpfull",
                   help="label for the output pkl name (time_inversion_<tag>_<settings>.pkl), "
                        "so timing runs on different datasets don't overwrite each other")
    args = p.parse_args()

    print(f"loading predictions from {args.predictions_file} ...")
    d = pickle.load(open(args.predictions_file, "rb"))
    ids = list(d["material_ids"])
    preds = d["predictions"]
    preds = preds.numpy() if torch.is_tensor(preds) else np.asarray(preds)
    print(f"  {len(ids)} materials available")

    rng = np.random.default_rng(args.seed)
    if args.ids_file:
        pos = {m: i for i, m in enumerate(ids)}
        sel = np.array([pos[l.strip()] for l in open(args.ids_file) if l.strip()])
    else:
        sel = rng.choice(len(ids), size=min(args.n, len(ids)), replace=False)
    sample_ids = [ids[i] for i in sel]
    id_to_bispec = {ids[i]: preds[i] for i in sel}
    print(f"timing {len(sample_ids)} materials, {args.max_workers}-way parallel "
          f"(each worker single-threaded)")

    settings = SETTINGS if args.settings == "both" else {args.settings: SETTINGS[args.settings]}
    for label, kw in settings.items():
        print(f"\n=== settings: {label} {kw} ===")
        out = f"time_inversion_{args.tag}_{label}.pkl"
        done_ids, ids_ordered, times, losses, flags = set(), [], [], [], []
        if os.path.exists(out):
            prev = pickle.load(open(out, "rb"))
            ids_ordered = list(prev["material_ids"])
            done_ids = set(ids_ordered)
            times, losses = list(prev["times"]), list(prev["min_loss"])
            flags = list(prev.get("timed_out", [False] * len(times)))
            print(f"  resuming: {len(done_ids)} already timed (from a previous, interrupted run)")

        todo = [mid for mid in sample_ids if mid not in done_ids]
        if not todo:
            print("  already complete")
        else:
            t_start = time.perf_counter()
            with ProcessPoolExecutor(
                max_workers=args.max_workers, initializer=inv.init_globals,
                initargs=(COB_PATH, MAT_PROJ_DF),
            ) as ex:
                if args.cap > 0:
                    futures = [
                        ex.submit(_time_one_capped,
                                  (mid, id_to_bispec[mid], args.wavelength, args.lmax,
                                   args.n_basis, args.radial, kw, args.cap))
                        for mid in todo
                    ]
                else:
                    futures = [
                        ex.submit(_time_one_worker,
                                  (mid, id_to_bispec[mid], args.wavelength, args.lmax,
                                   args.n_basis, args.radial, kw))
                        for mid in todo
                    ]
                n_since_checkpoint = 0
                for fut in tqdm(as_completed(futures), total=len(futures), desc=f"timing[{label}]"):
                    mp_id, elapsed, min_loss, *extra = fut.result()
                    flags.append(bool(extra[0]) if extra else False)
                    times.append(elapsed)
                    losses.append(min_loss)
                    ids_ordered.append(mp_id)  # aligned with times/losses (completion order)
                    done_ids.add(mp_id)
                    n_since_checkpoint += 1
                    if n_since_checkpoint >= 1:  # every completion: the pkl is small, and slow stragglers must not hold results back
                        n_since_checkpoint = 0
                        # checkpoint periodically -- a killed job (walltime, node failure) loses at
                        # most ~max_workers*10 materials' in-flight work, not the whole settings block
                        _save(out, dict(material_ids=ids_ordered, times=times, min_loss=losses,
                                          settings=kw, max_workers=args.max_workers, cap=args.cap, timed_out=flags))
            print(f"  wall-clock for this batch: {time.perf_counter() - t_start:.0f}s")

        summarize(f"settings={label}", times, losses)
        if args.cap > 0:
            print(f"  hit the {args.cap:g}s cap (killed, would be retried in run_alg_inversion.py): "
                  f"{sum(flags)}/{len(flags)} = {100 * np.mean(flags):.1f}%")
        _save(out, dict(material_ids=ids_ordered, times=times, min_loss=losses,
                          settings=kw, max_workers=args.max_workers, cap=args.cap, timed_out=flags))
        print(f"  saved per-material times -> {out}  ({len(done_ids)}/{len(sample_ids)} total)")

    print(f"\nNote: each material's time is its own wall-clock duration while {args.max_workers} "
          f"materials ran concurrently on this node (contention included by design -- see module "
          f"docstring). Pass --max-workers 1 for an isolated, contention-free per-structure cost instead.")


if __name__ == "__main__":
    main()
