# train.py
import os
import math
import pickle
import torch
import torch.nn as nn
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm
import matplotlib.pyplot as plt
import hydra
from omegaconf import DictConfig

from model import XRDTransformerEncoder
from imports import *
from mp_full.bg_noise_aug import add_bg_noise
import time
from omegaconf import OmegaConf
import random
from torch.utils.data import Dataset

# -------------------
# Scheduler
# -------------------
def get_cosine_schedule_with_warmup(optimizer, warmup_steps, training_steps, min_lr_ratio=0.0, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < warmup_steps:
            return float(current_step) / max(1, warmup_steps)
        progress = float(current_step - warmup_steps) / max(1, training_steps - warmup_steps)
        return max(min_lr_ratio, 0.5 * (1.0 + math.cos(math.pi * progress)))
    return LambdaLR(optimizer, lr_lambda, last_epoch=last_epoch)

def pad_peaks_to_fixed_length(peak_list, max_peaks=30, pad_position=-1.0, pad_intensity=-1.0):
    """Sort ascending by 2theta position and truncate to the first max_peaks --
    low-angle reflections carry the most lattice-geometry information per peak
    (Bragg's law), matching standard peak-based indexing practice (DICVOL/
    TREOR/ITO) and the AIDex peak-list convention (pk0..pk29, sorted by 2theta).
    Pads with a sentinel if there are fewer than max_peaks."""
    peak_list = sorted(peak_list, key=lambda p: p[0])[:max_peaks]
    while len(peak_list) < max_peaks:
        peak_list.append((pad_position, pad_intensity))
    return peak_list

# -------------------
# Small dataset class to keep track of material IDs
# -------------------
class XRDDataset(Dataset):
    def __init__(self, xrd_tensor, target_tensor, material_ids):
        self.xrd_tensor = xrd_tensor
        self.target_tensor = target_tensor
        self.material_ids = material_ids

    def __len__(self):
        return len(self.material_ids)

    def __getitem__(self, idx):
        data = self.xrd_tensor[idx]
        target = self.target_tensor[idx]
        mid = self.material_ids[idx]
        return data, target, mid

def masked_loss(output, target, criterion, mask):
    """
    Compute the loss only for entries where |target| > epsilon.
    Small values are ignored, gradients are zeroed.
    """
    loss = criterion(
    torch.masked_select(output, mask.unsqueeze(0)), 
    torch.masked_select(target, mask.unsqueeze(0)))
    return loss

# -------------------
# Data preparation
# -------------------
def prepare_data(sim_xrd_list, bispec_list, material_ids, batch_size=32, is_normalized=False,
                 indices=False, d_model=256, max_peaks=30, n_basis=10,
                 shuffle=True, structural_mask=None, xrd_mean = None, xrd_std = None,epsilon=1e-12):

    # ------------------------
    # XRD embedding
    # ------------------------
    if indices:
        # Sparse/peak-list input: store the raw padded (position, intensity)
        # tensor as-is. Embedding (position encoding + intensity projection)
        # happens inside XRDTransformerEncoder.forward() so it stays
        # trainable -- it must NOT be precomputed/normalized here.
        padded_xrd_list = [pad_peaks_to_fixed_length(peaks, max_peaks) for peaks in sim_xrd_list]
        xrd_embed = torch.FloatTensor(padded_xrd_list)
    else:
        xrd_array = np.stack(sim_xrd_list).astype(np.float32)
        xrd_tensor = torch.from_numpy(xrd_array).unsqueeze(1)
        if xrd_mean is None or xrd_std is None:
            # compute mean/std from this set (i.e. training set)
            xrd_mean = xrd_tensor.mean()
            xrd_std = xrd_tensor.std()
        xrd_embed = (xrd_tensor - xrd_mean) / (xrd_std + 1e-8)

    # ------------------------
    # Bispectrum tensor
    # ------------------------

    bispec_tensor_raw = torch.stack(bispec_list)

    if epsilon is not False:
    
        bispec_tensor_raw = torch.where(
            torch.abs(bispec_tensor_raw) < epsilon,
            torch.sign(bispec_tensor_raw) * epsilon,  # preserves sign
            bispec_tensor_raw
        )
    #bispec_tensor_raw[torch.abs(bispec_tensor_raw) < 1e-12] = 1e-12  # avoid numerical issues
    bispec_tensor = torch.sign(bispec_tensor_raw) * torch.abs(bispec_tensor_raw).pow(1/3)
    #bispec_tensor = torch.sign(bispec_tensor_raw)*torch.log1p(torch.abs(bispec_tensor_raw))
    if is_normalized:
        bispec_tensor = (bispec_tensor - bispec_tensor.mean()) / (bispec_tensor.std() + 1e-8)

    # ------------------------
    # Apply structural mask if provided
    # ------------------------
    if structural_mask is not None:
        bispec_tensor_new = bispec_tensor[:, structural_mask]
        out_features = bispec_tensor_new.shape[1] // n_basis  # compute automatically
        bispec_tensor = bispec_tensor_new.reshape(bispec_tensor.shape[0], n_basis, out_features)
    
    # ------------------------
    # Dataset and loader
    # ------------------------
    #dataset = torch.utils.data.TensorDataset(xrd_embed, bispec_tensor)
    if material_ids is None:
        material_ids = [None] * len(bispec_tensor)
    dataset = XRDDataset(xrd_tensor=xrd_embed, target_tensor=bispec_tensor, material_ids=material_ids)
    #breakpoint()
        
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

def prepare_data_direct(sim_xrd_list, lattice_param_list, material_ids, batch_size=32, is_normalized=False,
                 indices=False, d_model=256, max_peaks=30, n_basis=10,
                 shuffle=True, structural_mask=None, xrd_mean = None, xrd_std = None,lat_mean=None,lat_std=None,epsilon=1e-12):

    # ------------------------
    # XRD embedding
    # ------------------------
    if indices:
        # See prepare_data(): raw padded peaks, embedded inside the model.
        padded_xrd_list = [pad_peaks_to_fixed_length(peaks, max_peaks) for peaks in sim_xrd_list]
        xrd_embed = torch.FloatTensor(padded_xrd_list)
    else:
        xrd_array = np.stack(sim_xrd_list).astype(np.float32)
        xrd_tensor = torch.from_numpy(xrd_array).unsqueeze(1)
        if xrd_mean is None or xrd_std is None:
            # compute mean/std from this set (i.e. training set)
            xrd_mean = xrd_tensor.mean()
            xrd_std = xrd_tensor.std()
        xrd_embed = (xrd_tensor - xrd_mean) / (xrd_std + 1e-8)

    # ------------------------
    # Bispectrum tensor
    # ------------------------
    if isinstance(lattice_param_list, torch.Tensor):
        lattice_tensor_raw = lattice_param_list.float()
    elif isinstance(lattice_param_list[0], torch.Tensor):
        lattice_tensor_raw = torch.stack(lattice_param_list).float()
    else:
        lattice_tensor_raw = torch.tensor(lattice_param_list, dtype=torch.float32)

    ## assume that parameters are in the order a,b,c,alpha,beta,gammaå
    if is_normalized:
        # normalize cell lengths and angles separately
        if lat_mean is None or lat_std is None:
            lat_mean = lattice_tensor_raw.mean(dim=0)
            lat_std = lattice_tensor_raw.std(dim=0)
        # normalize lengths and angles separately
        lattice_tensor = (lattice_tensor_raw - lat_mean) / (lat_std + 1e-8)
    else:
        lattice_tensor = lattice_tensor_raw
    # ------------------------
    # Dataset and loader
    # ------------------------
    #dataset = torch.utils.data.TensorDataset(xrd_embed, bispec_tensor)
    if material_ids is None:
        material_ids = [None] * len(lattice_tensor)
    dataset = XRDDataset(xrd_tensor=xrd_embed, target_tensor=lattice_tensor, material_ids=material_ids)
    #breakpoint()
        
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

# -------------------
# Model factory
# -------------------
def build_model(cfg, out_dim=None,allowed_indices=None):
    # input_type/max_peaks/use_intensity are optional in older configs --
    # default to the original dense-only behavior when absent.
    input_type = cfg.model.get("input_type", "dense")
    max_peaks = cfg.model.get("max_peaks", 30)
    use_intensity = cfg.model.get("use_intensity", True)

    if cfg.model.type == "transformerbispec":
        if out_dim is None:
            out_dim = cfg.model.out_dim  # fallback to config
        model = XRDTransformerEncoder(
            out_dim=out_dim,
            d_model=cfg.model.d_model,
            h_dim=cfg.model.h_dim,
            transformer_proc=cfg.model.transformer_proc,
            n_self_layer=cfg.model.n_self_layer,
            n_head=cfg.model.n_head,
            attn_pdrop=cfg.model.attn_pdrop,
            resid_pdrop=cfg.model.resid_pdrop,
            input_type=input_type,
            max_peaks=max_peaks,
            use_intensity=use_intensity,
            allowed_indices=allowed_indices
        )

    elif cfg.model.type == "transformerdirect":
        model = XRDTransformerEncoder(
            out_dim=cfg.model.out_dim,
            d_model=cfg.model.d_model,
            h_dim=cfg.model.h_dim,
            transformer_proc=cfg.model.transformer_proc,
            n_self_layer=cfg.model.n_self_layer,
            n_head=cfg.model.n_head,
            attn_pdrop=cfg.model.attn_pdrop,
            resid_pdrop=cfg.model.resid_pdrop,
            input_type=input_type,
            max_peaks=max_peaks,
            use_intensity=use_intensity,
            allowed_indices=None)  # direct prediction does not use structured head

    else:
        raise ValueError(f"Unknown model type {cfg.model.type}")

    # ---- print parameter count ----
    print(f"Total parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    return model.cuda()

# -------------------
# Training loop
# -------------------
def train_model(cfg, model, train_loader, val_loader, loss_mask=None, start_epoch=0, best_val_loss=float("inf"), early_stopping_counter=0, optimizer=None, xrd_std=None):
    if optimizer is None:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
    restart_period = cfg.train.get("scheduler_restart_period", cfg.train.epochs)
    if cfg.train.use_scheduler:
        # Position scheduler at correct point in current cycle without calling step()
        steps_into_cycle = start_epoch % restart_period
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            warmup_steps=int(cfg.train.warmup_ratio * restart_period),
            training_steps=restart_period,
            min_lr_ratio=cfg.train.min_lr / cfg.train.lr,
            last_epoch=steps_into_cycle - 1,
        )
    criterion = (
        nn.L1Loss() if cfg.train.loss == "L1" 
        else nn.MSELoss() if cfg.train.loss == "MSE" 
        else nn.HuberLoss(delta=cfg.train.huber_delta) if cfg.train.loss == "Huber" 
        else None
    )
    if criterion is None:
        raise ValueError(f"Unknown loss type: {cfg.train.loss}")
    train_losses, val_losses = [], []
    _bg_rng = np.random.default_rng(cfg.train.get("bg_noise_seed", 0))
    _bg_noise_std = float(xrd_std) if xrd_std is not None else None

    # Initialize WandB if needed
    if cfg.train.use_wandb:
        import wandb
        if cfg.train.wandb_run_name is None:
            cfg.train.wandb_run_name = f"{cfg.model.type}_{cfg.data.name}_{int(time.time())}"
        wandb.init(
            project=cfg.train.wandb_project,
            name=f"{cfg.train.wandb_run_name}_nlayers{cfg.model.n_self_layer}_nhead{cfg.model.n_head}_dm{cfg.model.d_model}_hd{cfg.model.h_dim}_loss{cfg.train.loss}_smallvals{cfg.train.epsilon}_maskloss{cfg.train.mask_zeros_in_loss}",
            config=OmegaConf.to_container(cfg, resolve=True),
            tags=[cfg.data.name],
            group=cfg.data.name,
        )

    model_dir = os.path.join(cfg.output_dir, cfg.data.name, cfg.model.type, "models")
    os.makedirs(model_dir, exist_ok=True)

    if loss_mask is not None:
        loss_mask = loss_mask.cuda()
    end_epoch = start_epoch + cfg.train.epochs
    for epoch in range(start_epoch,end_epoch):
        model.train()
        train_loss = 0.0
        for batch in tqdm(train_loader):
            if len(batch) == 3:
                data, target, padding_mask = batch
            else:
                data, target = batch
            data, target = data.cuda().float(), target.cuda().float()
            if _bg_noise_std is not None and cfg.train.get("bg_noise_aug", True):
                data = add_bg_noise(data, _bg_noise_std, rng=_bg_rng,
                                     cnrs_mix_prob=cfg.train.get("bg_noise_cnrs_mix_prob", 0.0))
            optimizer.zero_grad()
            output = model(data)
            #if cfg.train.mask_zeros_in_loss and loss_mask is not None:
                # --- Zero symmetry-enforced channels ---
                #output[:, ~loss_mask] = 0.0
            if cfg.train.mask_zeros_in_loss:
                loss = masked_loss(output, target, criterion,mask=loss_mask)
            else:
                loss = criterion(output, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.clip_grad_norm)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)
        model.eval()
        # Validation loop (robust to batch size)
        val_loss_total = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch in val_loader:
                if len(batch) == 3:
                    data, target, padding_mask = batch
                else:
                    data, target = batch

                data, target = data.cuda().float(), target.cuda().float()
                batch_size = data.size(0)

                output = model(data)

                # Compute masked or standard loss
                if cfg.train.mask_zeros_in_loss:
                    batch_loss = masked_loss(output, target, criterion, mask=loss_mask) * batch_size
                else:
                    batch_loss = criterion(output, target).item() * batch_size

                val_loss_total += batch_loss
                total_samples += batch_size

            val_loss = val_loss_total / total_samples

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"Epoch {epoch+1}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

        # Log to WandB
        if cfg.train.use_wandb:
            if cfg.train.use_scheduler:
                wandb.log({
                    "epoch": epoch+1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "lr": scheduler.get_last_lr()[0]
                })
            else:
                wandb.log({
                    "epoch": epoch+1,
                    "train_loss": train_loss,
                    "val_loss": val_loss
                })

        # Save checkpoint if validation improves
        min_delta = cfg.train.get("early_stopping_min_delta", 0.0)
        if val_loss < best_val_loss - min_delta:
            best_val_loss = val_loss
            early_stopping_counter = 0
            checkpoint_path = os.path.join(model_dir, f"nlayers{cfg.model.n_self_layer}_nhead{cfg.model.n_head}_dm{cfg.model.d_model}_hd{cfg.model.h_dim}_loss{cfg.train.loss}_smallvals{cfg.train.epsilon}_maskloss{cfg.train.mask_zeros_in_loss}.pt")
            # Save checkpoint
            if cfg.train.use_scheduler:
                scheduler_state = scheduler.state_dict()
            else:
                scheduler_state = None
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler_state,
                "val_loss": val_loss,
                "early_stopping_counter": early_stopping_counter,
            }, checkpoint_path)
            print(f"Saved best model checkpoint to {checkpoint_path}")
        else:
            early_stopping_counter += 1

        # Periodic, non-overwriting snapshots -- independent of the
        # val-loss-driven "best" checkpoint above, which overwrites the same
        # file every time and so only ever preserves the most recent
        # improvement. Snapshots let you later pick a checkpoint by some
        # other criterion (e.g. real-domain eval on a held-out selection
        # slice, kept disjoint from whatever you report performance on)
        # without needing to rerun already-completed epochs.
        snapshot_every = cfg.train.get("snapshot_every_epochs", None)
        if snapshot_every and (epoch + 1) % snapshot_every == 0:
            snapshot_dir = os.path.join(model_dir, "snapshots")
            os.makedirs(snapshot_dir, exist_ok=True)
            snapshot_path = os.path.join(
                snapshot_dir,
                f"epoch{epoch + 1}_nlayers{cfg.model.n_self_layer}_nhead{cfg.model.n_head}_dm{cfg.model.d_model}_hd{cfg.model.h_dim}.pt",
            )
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.module.state_dict() if isinstance(model, torch.nn.DataParallel) else model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict() if cfg.train.use_scheduler else None,
                "val_loss": val_loss,
                "early_stopping_counter": early_stopping_counter,
            }, snapshot_path)
            print(f"Saved periodic snapshot to {snapshot_path}")

        if cfg.train.use_scheduler:
            scheduler.step()
            if (epoch + 1) % restart_period == 0:
                scheduler = get_cosine_schedule_with_warmup(
                    optimizer,
                    warmup_steps=int(cfg.train.warmup_ratio * restart_period),
                    training_steps=restart_period,
                    min_lr_ratio=cfg.train.min_lr / cfg.train.lr,
                )

        if cfg.train.get("early_stopping", False):
            patience = cfg.train.get("early_stopping_patience", 50)
            if early_stopping_counter >= patience:
                print(f"Early stopping triggered at epoch {epoch+1} (no improvement for {patience} epochs).")
                flag_path = cfg.train.get("early_stopping_flag_path", None)
                if flag_path:
                    open(flag_path, "w").close()
                    print(f"Wrote early stopping flag to {flag_path}")
                break

    if cfg.train.use_wandb:
        wandb.finish()

    return train_losses, val_losses


# -------------------
# Model evaluation
# -------------------
def evaluate_model(model, test_loader, criterion):
    model.eval()
    test_loss = 0.0
    with torch.no_grad():
        for batch in test_loader:
            data, target = batch
            data, target = data.cuda().float(), target.cuda().float()
            output = model(data)
            test_loss += criterion(output, target).item()
    return test_loss / len(test_loader)

def eval_and_save_preds(model, test_loader, criterion, model_dir, filename="eval_results.pkl"):
    model.eval()
    test_loss = 0.0
    all_outputs, all_targets, all_material_ids = [], [], []

    with torch.no_grad():
        for batch in test_loader:
            if len(batch) == 3:
                data, target, material_ids = batch
            else:
                data, target = batch
                material_ids = [None] * len(target)

            data, target = data.cuda().float(), target.cuda().float()
            output = model(data)

            test_loss += criterion(output, target).item()
            all_outputs.append(output.cpu())
            all_targets.append(target.cpu())
            all_material_ids.extend(material_ids)

    test_dict = {
        "predictions": torch.cat(all_outputs),
        "targets": torch.cat(all_targets),
        "material_ids": all_material_ids,
        "average_loss": test_loss / len(test_loader)}
    output_path = os.path.join(model_dir, filename)
    os.makedirs(model_dir, exist_ok=True)
    with open(output_path, "wb") as f:
        pickle.dump(test_dict, f)
    print(f"✅ Saved predictions to {output_path}")
    return test_loss / len(test_loader)
    

def align_dicts(xrd_dict, target_dict):
    common_keys = sorted(set(xrd_dict.keys()) & set(target_dict.keys()))
    print(f"Aligning {len(common_keys)} shared entries by keys.")
    sim_xrd_list = [xrd_dict[k] for k in common_keys]
    target_list = [target_dict[k] for k in common_keys]
    return common_keys, sim_xrd_list, target_list

# -------------------
# Main Hydra entry
# -------------------
@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    # Create output directories
    output_model_dir = os.path.join(cfg.output_dir, cfg.data.name, cfg.model.type, "models")
    output_plot_dir = os.path.join(cfg.output_dir, cfg.data.name, cfg.model.type, "plots")
    os.makedirs(output_model_dir, exist_ok=True)
    os.makedirs(output_plot_dir, exist_ok=True)

    # Load datasets
    data_dir = cfg.data.DATA_DIR
    sim_xrd_train = pickle.load(open(os.path.join(data_dir, cfg.data.xrd_train_file), "rb"))
    target_train = pickle.load(open(os.path.join(data_dir, cfg.data.target_train_file), "rb"))
    sim_xrd_val = pickle.load(open(os.path.join(data_dir, cfg.data.xrd_val_file), "rb"))
    target_val = pickle.load(open(os.path.join(data_dir, cfg.data.target_val_file), "rb"))
    
    sim_xrd_test = pickle.load(open(os.path.join(data_dir, cfg.data.xrd_test_file), "rb"))
    target_test = pickle.load(open(os.path.join(data_dir, cfg.data.target_test_file), "rb"))
    if isinstance(sim_xrd_train, dict) and isinstance(target_train, dict):
        train_ids, sim_xrd_train, target_train = align_dicts(sim_xrd_train, target_train)

    if isinstance(sim_xrd_val, dict) and isinstance(target_val, dict):
        val_ids, sim_xrd_val, target_val = align_dicts(sim_xrd_val, target_val)

    if isinstance(sim_xrd_test, dict) and isinstance(target_test, dict):
        test_ids, sim_xrd_test, target_test = align_dicts(sim_xrd_test, target_test)

    # Prepare loaders

    # mask zeros
    loss_mask = None
    out_features = 35#bispec_train[0].sum().item()  # default
    structural_mask = None # default
    if cfg.train.mask_zeros_in_loss:
        all_targets = torch.stack(target_train)  # [num_samples, target_dim]
        loss_mask = torch.any(torch.abs(all_targets) > cfg.train.mask_loss_threshold, dim=0)
        # loss_mask: (10, 35), torch.bool
        allowed_indices = torch.nonzero(loss_mask, as_tuple=False).tolist()
    else:
        allowed_indices = None
    # sim_xrd_train: list of training XRD arrays
    import numpy as np
    if cfg.data.indices:
        # Sparse/peak-list input: sim_xrd_train is ragged (variable-length lists
        # of (2theta, prominence) pairs, embedded inside the model). There is no
        # dense curve to normalize or to add background noise to, so skip the
        # global mean/std -- xrd_std=None also disables add_bg_noise() in
        # train_model(). prepare_data()'s indices branch ignores xrd_mean/xrd_std.
        xrd_mean = xrd_std = None
        print("Sparse/peak-list input: skipping XRD mean/std normalization and bg-noise aug")
    else:
        xrd_train_tensor = torch.FloatTensor(np.array(sim_xrd_train)).unsqueeze(1)
        xrd_mean = xrd_train_tensor.mean()
        xrd_std = xrd_train_tensor.std()
        print(f"Training XRD mean: {xrd_mean.item():.4f}, std: {xrd_std.item():.4f}")
    # TODO: put everything in one prepare_data function
    if cfg.model.type == "transformerbispec":
        # calculate means and std for lattice parameters from training set
        train_loader = prepare_data(sim_xrd_train, target_train, train_ids, batch_size=cfg.data.batch_size,
                                    indices=cfg.data.indices, d_model=cfg.model.d_model, max_peaks=cfg.model.get("max_peaks", 30),
                                    is_normalized=cfg.data.is_normalized, shuffle=True,structural_mask=structural_mask,xrd_mean=xrd_mean,xrd_std=xrd_std,epsilon=cfg.train.epsilon)
        val_loader = prepare_data(sim_xrd_val, target_val, val_ids, batch_size=cfg.data.batch_size,
                                indices=cfg.data.indices, d_model=cfg.model.d_model, max_peaks=cfg.model.get("max_peaks", 30),
                                is_normalized=cfg.data.is_normalized, shuffle=False,structural_mask=structural_mask,xrd_mean=xrd_mean,xrd_std=xrd_std,epsilon=cfg.train.epsilon)
        test_loader = prepare_data(sim_xrd_test, target_test, test_ids, batch_size=cfg.data.batch_size, indices=cfg.data.indices, d_model=cfg.model.d_model, max_peaks=cfg.model.get("max_peaks", 30),
                                is_normalized=cfg.data.is_normalized, shuffle=False,structural_mask=structural_mask,xrd_mean=xrd_mean,xrd_std=xrd_std,epsilon=cfg.train.epsilon)
    elif cfg.model.type == "transformerdirect":
        target_train = torch.stack(target_train).float() if isinstance(target_train[0], torch.Tensor) else torch.tensor(target_train, dtype=torch.float32)
        lattice_means = target_train.mean(dim=0)
        lattice_stds = target_train.std(dim=0)
        print(f"Training lattice parameter means: {lattice_means.cpu().numpy()}, stds: {lattice_stds.cpu().numpy()}")
        train_loader = prepare_data_direct(sim_xrd_train, target_train, train_ids, batch_size=cfg.data.batch_size,
                                    indices=cfg.data.indices, d_model=cfg.model.d_model, max_peaks=cfg.model.get("max_peaks", 30),
                                    is_normalized=cfg.data.is_normalized, shuffle=True,structural_mask=structural_mask,xrd_mean=xrd_mean,xrd_std=xrd_std,lat_mean=lattice_means,lat_std=lattice_stds,epsilon=cfg.train.epsilon)
        val_loader = prepare_data_direct(sim_xrd_val, target_val, val_ids, batch_size=cfg.data.batch_size,
                                indices=cfg.data.indices, d_model=cfg.model.d_model, max_peaks=cfg.model.get("max_peaks", 30),
                                is_normalized=cfg.data.is_normalized, shuffle=False,structural_mask=structural_mask,xrd_mean=xrd_mean,xrd_std=xrd_std,lat_mean=lattice_means,lat_std=lattice_stds,epsilon=cfg.train.epsilon)
        test_loader = prepare_data_direct(sim_xrd_test, target_test, test_ids, batch_size=cfg.data.batch_size, indices=cfg.data.indices, d_model=cfg.model.d_model, max_peaks=cfg.model.get("max_peaks", 30),
                                is_normalized=cfg.data.is_normalized, shuffle=False,structural_mask=structural_mask,xrd_mean=xrd_mean,xrd_std=xrd_std,lat_mean=lattice_means,lat_std=lattice_stds,epsilon=cfg.train.epsilon)
    # Build model
    model = build_model(cfg,allowed_indices=allowed_indices)

    # optionally resume from a checkpoint
    if cfg.train.load_pretrained and cfg.train.pretrained_path is not None:
        print(f"Resuming training from checkpoint: {cfg.train.pretrained_path}")
        checkpoint = torch.load(cfg.train.pretrained_path, map_location="cuda", weights_only=False)

        # Load model weights
        if isinstance(model, torch.nn.DataParallel):
            model.module.load_state_dict(checkpoint["model_state_dict"])
        else:
            model.load_state_dict(checkpoint["model_state_dict"])

        # Rebuild optimizer and load its state
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        # Scheduler intentionally not restored — per-job cosine restarts are desired behavior

        start_epoch = checkpoint.get("epoch", 0)
        best_val_loss = checkpoint.get("val_loss", float("inf"))
        early_stopping_counter = checkpoint.get("early_stopping_counter", 0)
        print(f"Checkpoint loaded. Resuming from epoch {start_epoch}, best_val_loss={best_val_loss:.4f}, early_stopping_counter={early_stopping_counter}")

    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.wd)
        start_epoch = 0
        best_val_loss = float("inf")
        early_stopping_counter = 0


    # if there is more than one GPU, use DataParallel (should be modified to use DDP for better performance)
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Train
    train_losses, val_losses = train_model(cfg, model, train_loader, val_loader, loss_mask, start_epoch, best_val_loss, early_stopping_counter, optimizer=optimizer, xrd_std=xrd_std)

    # TODO
    '''
    # Load best model checkpoint for evaluation
    #checkpoint_path
    best_model_path = os.path.join(cfg.output_dir, cfg.data.name, cfg.model.type, "models", f"{cfg.model.type}_best.pt")

    # If using DataParallel, load into model.module
    checkpoint = torch.load(best_model_path)
    if isinstance(model, torch.nn.DataParallel):
        model.module.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()
    print(f"Loaded best model from {best_model_path} for evaluation.")


    # Evaluate on test set
    # eval_and_save_preds(model, test_loader, criterion, model_dir, filename="eval_results.pkl")
    output_path = os.path.join(cfg.output_dir, cfg.data.name, cfg.model.type, "outputs")
    criterion = (
        nn.L1Loss() if cfg.train.loss == "L1" 
        else nn.MSELoss() if cfg.train.loss == "MSE" 
        else nn.HuberLoss(delta=cfg.train.huber_delta) if cfg.train.loss == "Huber" 
        else None
    )
    if criterion is None:
        raise ValueError(f"Unknown loss type: {cfg.train.loss}")
    test_loss = eval_and_save_preds(model, test_loader, criterion, output_path)
    print(f"Test Loss: {test_loss:.4f}")

    # Save final loss plot
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='train_loss', color='blue')
    plt.plot(val_losses, label='val_loss', color='red')
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(output_plot_dir, "loss_plot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    '''
if __name__ == "__main__":
    main()
