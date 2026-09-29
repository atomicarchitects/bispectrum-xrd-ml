"""
Compute and cache the training-set dense-XRD mean/std used to normalize model
input at train/eval time (see train.py's prepare_data() dense branch and
eval.py, which both recompute this from scratch every run by loading the full
train_sim_xrd_*.pkl -- here it's done once and cached to a small json next to
the data, keyed by data config name, so eval-time scripts (e.g. cosine-
similarity screening) don't have to reload a many-GB pickle each time).

Usage:
    python compute_xrd_norm_stats.py --data mpfull_aug_final
"""
import argparse
import json
import os
import pickle

import numpy as np
from omegaconf import OmegaConf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="conf/data/<name>.yaml (must have DATA_DIR + xrd_train_file)")
    p.add_argument("--out", default=None, help="Output json path (default: <DATA_DIR>/<data>_xrd_norm_stats.json)")
    args = p.parse_args()

    conf_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "conf", "data")
    cfg = OmegaConf.load(os.path.join(conf_dir, f"{args.data}.yaml"))
    data_dir = OmegaConf.to_container(cfg, resolve=True)["DATA_DIR"]
    xrd_path = os.path.join(data_dir, cfg.xrd_train_file)

    out_path = args.out or os.path.join(data_dir, f"{args.data}_xrd_norm_stats.json")

    print(f"loading {xrd_path} (streaming mean/std, avoids holding a second full-size copy)...", flush=True)
    with open(xrd_path, "rb") as f:
        xrd = pickle.load(f)
    print(f"  {len(xrd)} entries loaded", flush=True)

    n = 0
    s = 0.0
    ss = 0.0
    values = xrd.values() if isinstance(xrd, dict) else xrd
    for v in values:
        a = np.asarray(v, dtype=np.float64)
        n += a.size
        s += a.sum()
        ss += (a * a).sum()
    mean = s / n
    var = ss / n - mean * mean
    std = var ** 0.5
    print(f"mean={mean:.6f} std={std:.6f} n={n}", flush=True)

    with open(out_path, "w") as f:
        json.dump({"xrd_mean": mean, "xrd_std": std, "n_entries": len(xrd), "n_values": n,
                    "source": xrd_path, "data_config": args.data}, f)
    print(f"saved -> {out_path}")


if __name__ == "__main__":
    main()
