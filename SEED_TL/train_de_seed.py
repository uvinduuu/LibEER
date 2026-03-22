"""
DE-based CiMamba training on SEED-IV.

Uses Differential Entropy (DE) features instead of raw EEG.
Input: (20, 20) per window — 5 bands × 4 channels × 20 sub-windows of 0.5s.

The CiMamba model processes each (channel × band) pair as a "virtual channel",
so no model changes are needed — just n_channels becomes 20 instead of 4.

Key advantage:
    - Mamba sequence length: 20 steps (vs 125 for raw EEG)
    - Biologically meaningful features (DE encodes spectral power per band)
    - Lightning fast training

Usage:
    python train_de_seed.py --dataset_path /path/to/seed_iv --epochs 50

    Kaggle:
    python train_de_seed.py \
        --dataset_path /kaggle/input/datasets/phhasian0710/seed-iv/seed_iv \
        --epochs 50
"""

import os
import sys
import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Local
sys.path.insert(0, os.path.dirname(__file__))
from ci_mamba_model import CiMambaClassifier
from de_features import compute_de_features, normalize_de, N_BANDS, BAND_NAMES

# Shared data loader from mamba/
mamba_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mamba')
sys.path.insert(0, mamba_dir)
from dataset import load_seediv_clips
from windowed_dataset import split_trial_into_windows, bandpass_filter


CLASS_NAMES = ['neutral', 'sad', 'fear', 'happy']

# Each trial window → 20 virtual channels (4 channels × 5 bands)
N_VIRTUAL_CH = 4 * N_BANDS  # = 20


# ─── DE Dataset ────────────────────────────────────────────────

class DEWindowDataset(Dataset):
    """
    Dataset of DE feature sequences from EEG windows.
    Each sample: (20, n_subwindows) float32 tensor + int label.
    """
    def __init__(self, de_features_list, labels, augment=False):
        self.features = de_features_list   # list of (20, T') arrays
        self.labels   = labels
        self.augment  = augment

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        feat  = self.features[idx].copy()   # (20, T')
        label = self.labels[idx]

        if self.augment:
            # Light noise augmentation on DE features
            feat += np.random.randn(*feat.shape).astype(np.float32) * 0.05

        return torch.from_numpy(feat).float(), label


def build_de_datasets(trials, labels, subject_ids, window_size,
                      subwin_sec, sample_rate, train_idx, val_idx, test_idx):
    """Process trials → DE features → DEWindowDatasets for each split."""

    def process_split(indices, split_name, augment):
        all_feats, all_labels = [], []
        for idx in indices:
            trial = np.array(trials[idx], dtype=np.float32)
            label = int(labels[idx])
            # Split trial into full windows
            windows = split_trial_into_windows(trial, window_size)
            for w in windows:
                feats, _ = compute_de_features(w, sample_rate=sample_rate,
                                               subwin_sec=subwin_sec)
                feats = normalize_de(feats).astype(np.float32)
                all_feats.append(feats)
                all_labels.append(label)

        print(f"    {split_name:5s}: {len(indices):3d} trials → "
              f"{len(all_feats):5d} windows  "
              f"({len(all_feats)/max(len(indices),1):.1f} win/trial)")
        return DEWindowDataset(all_feats, all_labels, augment=augment)

    print(f"\n  Computing DE features (subwin={subwin_sec}s, {N_BANDS} bands × 4 ch = {N_VIRTUAL_CH} virtual channels)...")
    t0 = time.time()
    train_ds = process_split(train_idx, "Train", augment=True)
    val_ds   = process_split(val_idx,   "Val",   augment=False)
    test_ds  = process_split(test_idx,  "Test",  augment=False)
    print(f"  DE extraction done in {time.time()-t0:.1f}s")
    return train_ds, val_ds, test_ds


# ─── Helpers ───────────────────────────────────────────────────

def setup_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.long().to(device) if isinstance(by, torch.Tensor) else \
                 torch.tensor(by, dtype=torch.long, device=device)
            out  = model(bx)
            loss = criterion(out, by)
            all_preds.extend(torch.argmax(out, 1).cpu().numpy())
            all_labels.extend(by.cpu().numpy())
            total_loss += loss.item(); n_batches += 1
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / max(n_batches, 1), acc, f1, all_preds, all_labels


def print_report(y_true, y_pred, class_names, title=""):
    n  = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    print(f"\n  Confusion Matrix{' (' + title + ')' if title else ''}:")
    print(f"  {'':>10}", end="")
    for name in class_names: print(f"{name:>10}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"  {name:>10}", end="")
        for j in range(n): print(f"{cm[i][j]:>10}", end="")
        print()
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


# ─── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DE-based CiMamba on SEED-IV")

    # Data
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--sessions', nargs='+', type=int, default=None)
    parser.add_argument('--window_sec',  type=float, default=10.0,
                        help='Outer window size in seconds (default: 10)')
    parser.add_argument('--subwin_sec',  type=float, default=0.5,
                        help='DE sub-window size in seconds (default: 0.5)')
    parser.add_argument('--mode', choices=['sub_dep', 'sub_indep'], default='sub_dep')
    parser.add_argument('--n_test_subj', type=int, default=3)
    parser.add_argument('--n_val_subj',  type=int, default=2)

    # Model — n_channels=20 (virtual), not 4!
    parser.add_argument('--d_model',     type=int,   default=64)
    parser.add_argument('--n_layers',    type=int,   default=2)
    parser.add_argument('--d_state',     type=int,   default=16)
    parser.add_argument('--dropout',     type=float, default=0.4)
    parser.add_argument('--aggregation', choices=['mean', 'attention'], default='mean')

    # Training
    parser.add_argument('--batch_size',   type=int,   default=64)
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--lr',           type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--patience',     type=int,   default=15)
    parser.add_argument('--seed',         type=int,   default=2024)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    setup_seed(args.seed)
    device      = torch.device(args.device)
    sample_rate = 200
    window_size = int(args.window_sec * sample_rate)
    n_subwindows = int(args.window_sec / args.subwin_sec)

    print(f"\n{'='*60}")
    print(f"DE-CiMamba PRE-TRAINING — SEED-IV")
    print(f"  Mode        : {args.mode}")
    print(f"  Window      : {args.window_sec}s → {n_subwindows} sub-wins of {args.subwin_sec}s")
    print(f"  Input shape : ({N_VIRTUAL_CH} virtual ch, {n_subwindows} time steps)")
    print(f"  Bands       : {BAND_NAMES}")
    print(f"  d_model={args.d_model}, layers={args.n_layers}, agg={args.aggregation}")
    print(f"{'='*60}")

    # Load SEED-IV trials
    print(f"\nLoading SEED-IV...")
    t0 = time.time()
    trials, labels, subject_ids, session_ids = load_seediv_clips(
        args.dataset_path, sessions=args.sessions
    )
    print(f"  Loaded {len(trials)} trials in {time.time()-t0:.1f}s")

    # Trial-level split indices
    rng = np.random.RandomState(args.seed)
    n   = len(trials)
    idx = np.arange(n); rng.shuffle(idx)

    if args.mode == 'sub_dep':
        n_test = int(n * 0.15); n_val = int(n * 0.15)
        test_idx  = idx[:n_test]
        val_idx   = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]
        print(f"  Split (sub_dep): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    else:
        unique_subj = sorted(set(subject_ids)); rng.shuffle(unique_subj)
        test_subj   = set(unique_subj[:args.n_test_subj])
        val_subj    = set(unique_subj[args.n_test_subj:args.n_test_subj + args.n_val_subj])
        train_subj  = set(s for s in unique_subj if s not in test_subj | val_subj)
        train_idx   = [i for i, s in enumerate(subject_ids) if s in train_subj]
        val_idx     = [i for i, s in enumerate(subject_ids) if s in val_subj]
        test_idx    = [i for i, s in enumerate(subject_ids) if s in test_subj]
        print(f"  Split (sub_indep): train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # Build DE datasets
    train_ds, val_ds, test_ds = build_de_datasets(
        trials, labels, subject_ids, window_size,
        args.subwin_sec, sample_rate, train_idx, val_idx, test_idx
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Model — n_channels=20 virtual channels
    model = CiMambaClassifier(
        n_channels=N_VIRTUAL_CH, num_classes=4,
        d_model=args.d_model, n_layers=args.n_layers,
        d_state=args.d_state, dropout=args.dropout,
        aggregation=args.aggregation,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  DE-CiMamba params: {n_params:,}")
    print(f"  Mamba seq len    : {n_subwindows} steps (vs ~125 for raw EEG)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val_f1, best_state, patience_ctr = 0.0, None, 0
    epoch_times = []

    print(f"\n{'='*60}")
    print(f"Training ({args.epochs} epochs, {len(train_loader)} batches/epoch)")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        t0 = time.time()

        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.long().to(device) if isinstance(by, torch.Tensor) else \
                 torch.tensor(by, dtype=torch.long, device=device)
            optimizer.zero_grad()
            out  = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss    += loss.item()
            ep_correct += (torch.argmax(out, 1) == by).sum().item()
            ep_total   += len(by)

        scheduler.step()
        tr_loss = ep_loss / max(len(train_loader), 1)
        tr_acc  = ep_correct / max(ep_total, 1)

        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, device, criterion)
        ep_time = time.time() - t0
        epoch_times.append(ep_time)

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f}, Acc: {va_acc:.4f}, F1: {va_f1:.4f} | "
              f"{ep_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if args.patience > 0 and patience_ctr >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # Test
    if best_state: model.load_state_dict(best_state)
    model = model.to(device)

    te_loss, te_acc, te_f1, te_preds, te_labels = evaluate(model, test_loader, device, criterion)

    print(f"\n{'='*60}")
    print(f"RESULTS — DE-CiMamba on SEED-IV ({args.mode})")
    print(f"{'='*60}")
    print(f"  Best Val F1 : {best_val_f1:.4f}")
    print(f"  Test Acc    : {te_acc:.4f}")
    print(f"  Test F1     : {te_f1:.4f}")
    print(f"  Avg epoch   : {np.mean(epoch_times):.1f}s")
    print(f"  Total time  : {sum(epoch_times)/60:.1f} min")
    print_report(te_labels, te_preds, CLASS_NAMES, title="SEED-IV Test")

    # Save checkpoint
    ckpt_dir  = os.path.join(os.path.dirname(__file__), 'checkpoints', 'de_ci_seed')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
    torch.save({
        'model':     model.state_dict(),
        'model_cfg': {
            'n_channels':  N_VIRTUAL_CH,
            'num_classes': 4,
            'd_model':     args.d_model,
            'n_layers':    args.n_layers,
            'd_state':     args.d_state,
            'dropout':     args.dropout,
            'aggregation': args.aggregation,
        },
        'de_cfg': {
            'subwin_sec':  args.subwin_sec,
            'n_subwindows': n_subwindows,
            'n_virtual_ch': N_VIRTUAL_CH,
        },
        'val_f1':   best_val_f1,
        'test_acc': te_acc,
        'test_f1':  te_f1,
    }, ckpt_path)
    print(f"\n  Checkpoint: {ckpt_path}")
    print(f"  Use with: finetune_de_emognition.py\n")


if __name__ == '__main__':
    main()
