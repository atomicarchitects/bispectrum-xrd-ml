import argparse
import pickle
import os
import sys
import random
from collections import defaultdict
from tqdm import tqdm
import numpy as np

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utilities import training_data_gen, get_sim_xrd_from_preprocessed_results, save_results_pickle
import e3nn_jax
from e3nn import io
import torch

torch.set_default_dtype(torch.float64)

# --------------------------------------------------------------------------
# Worker functions
# --------------------------------------------------------------------------
def bispec_worker(args):
    mpid, struct, cob_bi = args
    try:
        item = {'crystal': struct}
        res = training_data_gen([item], cob_bi=cob_bi)
        if res:
            return (mpid, res[0])
    except Exception as e:
        print(f"⚠️ Error {mpid}: {type(e).__name__} - {e}")
    return None


def xrd_worker(args):
    mpid, struct = args
    try:
        res = get_sim_xrd_from_preprocessed_results([{'crystal': struct}], 0)
        if res is not None:
            return (mpid, res)
    except Exception as e:
        print(f"⚠️ Error in XRD {mpid}: {type(e).__name__} - {e}")
    return None


# --------------------------------------------------------------------------
# Dataset splitting and saving
# --------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Split the Materials Project structures by reduced formula (60/20/20, seed 42) "
                    "and generate, for each split, either simulated XRD patterns or bispectra. "
                    "Reads materials_project_structures.pkl from, and writes to, $POWDERXRD_DATA_ROOT.")
    parser.add_argument("--mode", choices=["xrd", "bispec"], default="xrd",
                        help="what to generate for each split: '{split}_data_sim_xrd.pkl' (xrd) or "
                             "'{split}_data_bispec_list.pkl' (bispec)")
    args = parser.parse_args()
    if "POWDERXRD_DATA_ROOT" not in os.environ:
        sys.exit("Set POWDERXRD_DATA_ROOT to the directory containing materials_project_structures.pkl")
    output_dir = os.environ["POWDERXRD_DATA_ROOT"]
    os.makedirs(output_dir, exist_ok=True)

    split_files = {
        'train': f'{output_dir}/train_data.pkl',
        'val': f'{output_dir}/val_data.pkl',
        'test': f'{output_dir}/test_data.pkl'
    }
    mat_proj_file = f'{output_dir}/materials_project_structures.pkl'
    # --- Check if splits exist ---
    if all(os.path.exists(f) for f in split_files.values()):
        print("📂 Loading existing split files...")
        train_data = pickle.load(open(split_files['train'], 'rb'))
        val_data   = pickle.load(open(split_files['val'], 'rb'))
        test_data  = pickle.load(open(split_files['test'], 'rb'))
    else:
        with open(mat_proj_file, 'rb') as f:
            our_data = pickle.load(f)

        formula_to_keys = defaultdict(list)
        for k, struct in our_data.items():
            formula_to_keys[struct.reduced_formula].append(k)

        formulas = list(formula_to_keys.keys())
        random.seed(42)
        random.shuffle(formulas)

        total = len(formulas)
        train_end = int(0.6 * total)
        val_end = int(0.8 * total)

        train_formulas = formulas[:train_end]
        val_formulas = formulas[train_end:val_end]
        test_formulas = formulas[val_end:]

        train_keys = [k for f in train_formulas for k in formula_to_keys[f]]
        val_keys   = [k for f in val_formulas for k in formula_to_keys[f]]
        test_keys  = [k for f in test_formulas for k in formula_to_keys[f]]

        train_data = {k: our_data[k] for k in train_keys}
        val_data   = {k: our_data[k] for k in val_keys}
        test_data  = {k: our_data[k] for k in test_keys}

        for split_name, split_data in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
            with open(split_files[split_name], 'wb') as f:
                pickle.dump(split_data, f)

    print(f"\nTotal samples: {len(train_data) + len(val_data) + len(test_data)}")
    print(f"Train: {len(train_data)} | Val: {len(val_data)} | Test: {len(test_data)}")

    # --- Generation mode ---
    mode = args.mode

    for split_name, split_data in zip(['train', 'val', 'test'], [train_data, val_data, test_data]):
        print(f"\n=== Processing {split_name.upper()} split ({mode}) ===")

        if mode == "bispec":
            l_max = 6
            rtp_bi = e3nn_jax.reduced_symmetric_tensor_product_basis(
        io.SphericalTensor(l_max, 1, -1), 3, keep_ir=['0o', '0e']
    )
            cob_bi = torch.tensor(rtp_bi.array, dtype=torch.float64)
            results = {}
            for mpid, struct in tqdm(split_data.items(), total=len(split_data), desc=f"Bispectrum {split_name}"):
                out = bispec_worker((mpid, struct, cob_bi))
                if out is not None:
                    mpid_out, val = out
                    results[mpid_out] = val
            with open(f'{output_dir}/{split_name}_data_bispec_list.pkl', 'wb') as f:
                pickle.dump(results, f)

        elif mode == "xrd":
            #results = parallel_xrd_gen(split_data)
            results = {}
            for mpid, struct in tqdm(split_data.items(), total=len(split_data), desc=f"Bispectrum {split_name}"):
                out = xrd_worker((mpid, struct))
                if out is not None:
                    mpid_out, val = out
                    results[mpid_out] = val
            with open(f'{output_dir}/{split_name}_data_sim_xrd.pkl', 'wb') as f:
                pickle.dump(results, f)
