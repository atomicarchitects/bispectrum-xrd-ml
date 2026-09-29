import h5py
import numpy as np
from pymatgen.core.lattice import Lattice
import os


def niggli_params(mat33):
    lat = Lattice(mat33)
    nig = lat.get_niggli_reduced_lattice()
    return nig.a, nig.b, nig.c, nig.alpha, nig.beta, nig.gamma


def analyze_h5(path, label):
    try:
        f = h5py.File(path, "r")
    except Exception as e:
        print(f"\n[SKIP] {label}: {e}")
        return

    minerals = list(f.keys())

    errors = {p: [] for p in ['a','b','c','alpha','beta','gamma']}
    results = []  # (total_error, mn, pred_params, true_params)

    n_valid = 0

    for mn in minerals:
        g = f[mn]
        try:
            pred_lat = g['pred'][()].reshape(3, 3)
            true_lat = g['true_lattice'][()].reshape(3, 3)

            pa, pb, pc, palpha, pbeta, pgamma = niggli_params(pred_lat)
            ta, tb, tc, talpha, tbeta, tgamma = niggli_params(true_lat)

            ea = abs(pa - ta)
            eb = abs(pb - tb)
            ec = abs(pc - tc)
            ealpha = abs(palpha - talpha)
            ebeta = abs(pbeta - tbeta)
            egamma = abs(pgamma - tgamma)

            errors['a'].append(ea)
            errors['b'].append(eb)
            errors['c'].append(ec)
            errors['alpha'].append(ealpha)
            errors['beta'].append(ebeta)
            errors['gamma'].append(egamma)

            total_error = ea + eb + ec + ealpha + ebeta + egamma

            results.append((
                total_error,
                mn,
                (pa, pb, pc, palpha, pbeta, pgamma),
                (ta, tb, tc, talpha, tbeta, tgamma)
            ))

            n_valid += 1

        except Exception:
            pass

    f.close()

    print(f"\n{'='*58}")
    print(f"  {label}  ({n_valid}/{len(minerals)} minerals)")
    print(f"{'='*58}")

    for param in ['a','b','c','alpha','beta','gamma']:
        if errors[param]:
            unit = 'Å' if param in ('a','b','c') else '°'
            print(
                f"  MAE_{param}: {np.mean(errors[param]):.4f} {unit}  "
                f"(med {np.median(errors[param]):.4f})"
            )

    results.sort(key=lambda x: x[0])

    print("\nTop 3 best structures:")
    for err, mn, pred, true in results[:3]:
        print(f"\n  {mn}: {err:.4f}")
        print(f"    pred: a={pred[0]:.3f}, b={pred[1]:.3f}, c={pred[2]:.3f}, "
              f"α={pred[3]:.2f}, β={pred[4]:.2f}, γ={pred[5]:.2f}")
        print(f"    true: a={true[0]:.3f}, b={true[1]:.3f}, c={true[2]:.3f}, "
              f"α={true[3]:.2f}, β={true[4]:.2f}, γ={true[5]:.2f}")

    print("\nTop 3 worst structures:")
    for err, mn, pred, true in results[-3:][::-1]:
        print(f"\n  {mn}: {err:.4f}")
        print(f"    pred: a={pred[0]:.3f}, b={pred[1]:.3f}, c={pred[2]:.3f}, "
              f"α={pred[3]:.2f}, β={pred[4]:.2f}, γ={pred[5]:.2f}")
        print(f"    true: a={true[0]:.3f}, b={true[1]:.3f}, c={true[2]:.3f}, "
              f"α={true[3]:.2f}, β={true[4]:.2f}, γ={true[5]:.2f}")


if __name__ == "__main__":
    base = os.environ.get("POWDERXRD_DATA_ROOT", "./data") + "/rruff_crystalyze"

    for fname in sorted(os.listdir(base)):
        if fname.endswith(".h5"):
            label = fname.replace("rruff_", "").replace(".h5", "")
            analyze_h5(f"{base}/{fname}", label)
