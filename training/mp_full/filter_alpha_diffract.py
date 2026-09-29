import os
import pickle
from tqdm import tqdm

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer


# --------------------------------------------------
# Copy these directly from AlphaDiffract
# --------------------------------------------------

SYMPRECS = [0.01, 0.05, 0.1]
ANGLE_TOLS = [0.1, 1.0, 2.5, 5.0]

MAX_ATOMS = 500
MAX_VOLUME = 100000


def niggli_reduce(
    structure,
    symprec=0.01,
    angle_tolerance=5.0,
):
    conventional = structure.to_conventional(
        symprec=symprec,
        angle_tolerance=angle_tolerance,
    )

    return conventional.get_reduced_structure(
        reduction_algo="niggli"
    )


def compute_spacegroup_grid(
    structure,
    symprecs,
    angle_tolerances,
):
    grid = []

    for sym in symprecs:
        row = []

        for ang in angle_tolerances:
            try:
                sg = SpacegroupAnalyzer(
                    structure,
                    symprec=float(sym),
                    angle_tolerance=float(ang),
                ).get_space_group_number()

                row.append(int(sg))

            except Exception:
                row.append(0)

        grid.append(row)

    return grid


def is_spacegroup_stable(grid):
    flat = [x for row in grid for x in row]

    if not flat:
        return False

    return len(set(flat)) <= 1


# --------------------------------------------------
# Reduced-cell consistency
# --------------------------------------------------

def crystal_system_from_sg(sg_num):
    if sg_num <= 2:
        return "triclinic"
    elif sg_num <= 15:
        return "monoclinic"
    elif sg_num <= 74:
        return "orthorhombic"
    elif sg_num <= 142:
        return "tetragonal"
    elif sg_num <= 167:
        return "trigonal"
    elif sg_num <= 194:
        return "hexagonal"
    else:
        return "cubic"


def check_reduced_cell_consistency(
    structure,
    sg_num,
    symprec=0.01,
    angle_tolerance=5.0,
    rtol=1e-5,
):
    try:
        sga = SpacegroupAnalyzer(
            structure,
            symprec=symprec,
            angle_tolerance=angle_tolerance,
        )

        conv = sga.get_conventional_standard_structure()

        reduced = conv.get_reduced_structure(
            reduction_algo="niggli"
        )

        lat = reduced.lattice

        a, b, c = lat.abc
        alpha, beta, gamma = lat.angles

    except Exception:
        return False

    def eq(x, y):
        return abs(x - y) < rtol

    cs = crystal_system_from_sg(sg_num)

    if cs == "cubic":
        return (
            eq(a, b)
            and eq(b, c)
            and eq(alpha, 90)
            and eq(beta, 90)
            and eq(gamma, 90)
        )

    elif cs == "tetragonal":
        return (
            eq(a, b)
            and eq(alpha, 90)
            and eq(beta, 90)
            and eq(gamma, 90)
        )

    elif cs == "orthorhombic":
        return (
            eq(alpha, 90)
            and eq(beta, 90)
            and eq(gamma, 90)
        )

    elif cs in ["hexagonal", "trigonal"]:
        return eq(a, b)

    elif cs == "monoclinic":
        n90 = sum(eq(x, 90) for x in [alpha, beta, gamma])
        return n90 == 2

    return True


# --------------------------------------------------
# Dataset processing
# --------------------------------------------------

def get_structure(entry):

    if isinstance(entry, Structure):
        return entry

    if isinstance(entry, dict):

        for key in [
            "structure",
            "pmg_structure",
            "crystal",
        ]:
            if key in entry:
                return entry[key]

    raise ValueError("No structure found")


def passes_filters(structure):

    if len(structure) > MAX_ATOMS:
        return False

    if structure.volume > MAX_VOLUME:
        return False

    # spacegroup stability
    '''
    grid = compute_spacegroup_grid(
        structure,
        SYMPRECS,
        ANGLE_TOLS,
    )

    if not is_spacegroup_stable(grid):
        return False

    # reduced cell consistency
    try:
        sg_num = SpacegroupAnalyzer(
            structure,
            symprec=0.01,
            angle_tolerance=5.0,
        ).get_space_group_number()

    except Exception:
        return False

    if not check_reduced_cell_consistency(
        structure,
        sg_num,
    ):
        return False
    '''
    return True


def filter_dataset(path):

    with open(path, "rb") as f:
        data = pickle.load(f)

    kept = {}
    rejected = 0

    for key, structure in tqdm(data.items()):

        try:

            #structure = niggli_reduce(structure)

            if passes_filters(structure):
                kept[key] = structure
            else:
                rejected += 1

        except Exception as e:
            rejected += 1

    outpath = path.replace(
        ".pkl",
        "_filter_num_vol.pkl"
    )

    with open(outpath, "wb") as f:
        pickle.dump(kept, f)

    print()
    print(path)
    print("original:", len(data))
    print("kept:", len(kept))
    print("rejected:", rejected)
    print("saved:", outpath)

_ROOT = os.environ.get("POWDERXRD_DATA_ROOT", "./data")
for path in [
    f"{_ROOT}/train_data.pkl",
    f"{_ROOT}/val_data.pkl",
    f"{_ROOT}/test_data.pkl",
]:
    filter_dataset(path)
