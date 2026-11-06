"""
Build the change-of-basis tensor (cob_lmax6.pkl) that the bispectrum computation and the
inversion (`run_alg_inversion.py -cob`) read.

It is the e3nn-jax reduced symmetric tensor-product basis of three copies of a spherical
tensor up to l_max, keeping only the scalar (0e) and pseudoscalar (0o) outputs -- the same
tensor `utilities.training_data_gen` builds internally when it is called with cob_bi=None.
Building it takes a while, which is why the scripts load it from a file.

Usage:
    python build_cob.py                                   # -> $POWDERXRD_DATA_ROOT/cob_lmax6.pkl
    python build_cob.py --l-max 6 --out /path/to/cob_lmax6.pkl
"""
import argparse
import os

import torch


def build_cob(l_max):
    import e3nn_jax
    from e3nn import io
    rtp_bi = e3nn_jax.reduced_symmetric_tensor_product_basis(
        io.SphericalTensor(l_max, 1, -1), 3, keep_ir=["0o", "0e"]
    )
    return torch.tensor(rtp_bi.array, dtype=torch.float64)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--l-max", type=int, default=6)
    p.add_argument("--out", default=None,
                   help="output path (default: $POWDERXRD_DATA_ROOT/cob_lmax{l_max}.pkl)")
    args = p.parse_args()

    out = args.out
    if out is None:
        root = os.environ.get("POWDERXRD_DATA_ROOT")
        if root is None:
            p.error("set POWDERXRD_DATA_ROOT or pass --out")
        out = os.path.join(root, f"cob_lmax{args.l_max}.pkl")

    cob = build_cob(args.l_max)
    torch.save(cob, out)
    print(f"wrote {out}  shape={tuple(cob.shape)}")


if __name__ == "__main__":
    main()
