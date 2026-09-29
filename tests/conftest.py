import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

# bispectrum/ and training/mp_full/ are flat script directories (not packages)
# whose modules import each other and utilities.py by bare name, e.g.
# `from utilities import ...` or `from data_gen_full_aug import ...`. They're
# meant to be run with their own directory as cwd/on sys.path, so tests need
# the same directories added here to import them the same way.
for _p in (ROOT, ROOT / "bispectrum", ROOT / "training", ROOT / "training" / "mp_full"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))
