"""
Pre-train Channel-Independent Mamba (CiMamba) on SEED-IV.

Trains with windowed data (no padding). The shared encoder learns
channel-agnostic EEG temporal patterns that transfer to Emognition.

Usage:
    python train_ci_seed.py --dataset_path /path/to/seed_iv --epochs 50

    # Subject-independent split
    python train_ci_seed.py --dataset_path /path/to/seed_iv --epochs 50 --mode sub_indep

    # Kaggle
    python train_ci_seed.py --dataset_path /kaggle/input/datasets/phhasian0710/seed-iv/seed_iv --epochs 50
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
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# Local
sys.path.insert(0, os.path.dirname(__file__))
from ci_mamba_model import CiMambaClassifier

# Import shared data utilities from mamba/
mamba_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mamba')
sys.path.insert(0, mamba_dir)
from dataset import load_seediv_clips
from windowed_dataset import (
    create_windowed_splits, WindowedEEGDataset,
    split_trial_into_windows, bandpass_filter, normalize_trial
)


CLASS_NAMES = ['neutral', 'sad', 'fear', 'happy']


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch_x, batch_y in loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.long().to(device) if isinstance(batch_y, torch.Tensor) \
                      else torch.tensor(batch_y, dtype=torch.long, device=device)
            out  = model(batch_x)
            loss = criterion(out, batch_y)
            all_preds.extend(torch.argmax(out, dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            total_loss += loss.item()
            n_batches  += 1

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / max(n_batches, 1), acc, f1, all_preds, all_labels


def print_report(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    print(f"\n  Confusion Matrix{' (' + title + ')' if title else ''}:")
    print(f"  {'':>10}", end="")
    for n in CLASS_NAMES: print(f"{n:>10}", end="")
    print()
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:>10}", end="")
        for j in range(4): print(f"{cm[i][j]:>10}", end="")
        print()
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))


def make_subject_split_datasets(trials, labels, subject_ids,
                                 window_size, n_test_subj, n_val_subj, seed, sample_rate=200):
    """Split subjects into train/val/test, then window."""
    unique_subjects = sorted(set(subject_ids))
    rng = np.random.RandomState(seed)
    shuffled = list(unique_subjects)
    rng.shuffle(shuffled)
    test_subjects  = set(shuffled[:n_test_subj])
    val_subjects   = set(shuffled[n_test_subj:n_test_subj + n_val_subj])
    train_subjects = set(s for s in unique_subjects if s not in test_subjects and s not in val_subjects)

    print(f"\n  Subject split:")
    print(f"    Train ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"    Val   ({len(val_subjects)}):   {sorted(val_subjects)}")
    print(f"    Test  ({len(test_subjects)}):  {sorted(test_subjects)}")

    def make_ds(subj_set, augment):
        wins, lbls = [], []
        for trial, label, subj in zip(trials, labels, subject_ids):
            if subj not in subj_set: continue
            t = np.array(trial, dtype=np.float32)
            t = bandpass_filter(t, fs=sample_rate)
            t = normalize_trial(t)
            for w in split_trial_into_windows(t, window_size):
                wins.append(w); lbls.append(int(label))
        return WindowedEEGDataset(wins, lbls, augment=augment, sample_rate=sample_rate)

    tr_ds = make_ds(train_subjects, augment=True)
    va_ds = make_ds(val_subjects,   augment=False)
    te_ds = make_ds(test_subjects,  augment=False)
    print(f"    Windows → train={len(tr_ds)}, val={len(va_ds)}, test={len(te_ds)}")
    return tr_ds, va_ds, te_ds


def main():
    parser = argparse.ArgumentParser(description="Pre-train CiMamba on SEED-IV")

    # Data
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--sessions', nargs='+', type=int, default=None)
    parser.add_argument('--window_sec', type=float, default=10.0)
    parser.add_argument('--mode', choices=['sub_dep', 'sub_indep'], default='sub_dep')
    parser.add_argument('--n_test_subj', type=int, default=3)
    parser.add_argument('--n_val_subj',  type=int, default=2)

    # Model
    parser.add_argument('--d_model',     type=int,   default=64)
    parser.add_argument('--n_layers',    type=int,   default=2)
    parser.add_argument('--d_state',     type=int,   default=16)
    parser.add_argument('--dropout',     type=float, default=0.4)
    parser.add_argument('--aggregation', choices=['mean', 'attention'], default='mean')

    # Training
    parser.add_argument('--batch_size',    type=int,   default=64)
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--weight_decay',  type=float, default=0.05)
    parser.add_argument('--patience',      type=int,   default=15)
    parser.add_argument('--seed',          type=int,   default=2024)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)
    sample_rate = 200
    window_size = int(args.window_sec * sample_rate)

    print(f"\n{'='*60}")
    print(f"CiMamba PRE-TRAINING — SEED-IV")
    print(f"  Mode     : {args.mode}")
    print(f"  Window   : {args.window_sec}s ({window_size} samples)")
    print(f"  d_model={args.d_model}, layers={args.n_layers}, agg={args.aggregation}")
    print(f"  batch={args.batch_size}, lr={args.lr}, wd={args.weight_decay}")
    print(f"{'='*60}")

    # Load data
    print(f"\nLoading SEED-IV...")
    t0 = time.time()
    trials, labels, subject_ids, session_ids = load_seediv_clips(
        args.dataset_path, sessions=args.sessions
    )
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Create splits
    if args.mode == 'sub_dep':
        train_ds, val_ds, test_ds, info = create_windowed_splits(
            trials, labels, subject_ids,
            window_size=window_size,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            filter_eeg=True, normalize=True,
            sample_rate=sample_rate, seed=args.seed, augment_train=True,
        )
    else:
        train_ds, val_ds, test_ds = make_subject_split_datasets(
            trials, labels, subject_ids,
            window_size, args.n_test_subj, args.n_val_subj,
            seed=args.seed, sample_rate=sample_rate,
        )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Model
    model = CiMambaClassifier(
        n_channels=4, num_classes=4,
        d_model=args.d_model, n_layers=args.n_layers,
        d_state=args.d_state, dropout=args.dropout,
        aggregation=args.aggregation,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  CiMamba params: {n_params:,}")
    print(f"  Mamba seq len : ~{window_size // 16} steps/channel")

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
            by = by.long().to(device) if isinstance(by, torch.Tensor) \
                 else torch.tensor(by, dtype=torch.long, device=device)
            optimizer.zero_grad()
            out = model(bx)
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
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    te_loss, te_acc, te_f1, te_preds, te_labels = evaluate(model, test_loader, device, criterion)

    print(f"\n{'='*60}")
    print(f"RESULTS — CiMamba on SEED-IV ({args.mode})")
    print(f"{'='*60}")
    print(f"  Best Val F1 : {best_val_f1:.4f}")
    print(f"  Test Acc    : {te_acc:.4f}")
    print(f"  Test F1     : {te_f1:.4f}")
    print(f"  Avg epoch   : {np.mean(epoch_times):.1f}s")
    print(f"  Total time  : {sum(epoch_times)/60:.1f} min")
    print_report(te_labels, te_preds, title="SEED-IV Test")

    # Save checkpoint (includes model config for fine-tuning)
    ckpt_dir  = os.path.join(os.path.dirname(__file__), 'checkpoints', 'ci_seed')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
    torch.save({
        'model':       model.state_dict(),
        'model_cfg': {
            'n_channels':  4,
            'num_classes': 4,
            'd_model':     args.d_model,
            'n_layers':    args.n_layers,
            'd_state':     args.d_state,
            'dropout':     args.dropout,
            'aggregation': args.aggregation,
        },
        'val_f1':    best_val_f1,
        'test_acc':  te_acc,
        'test_f1':   te_f1,
        'window_size': window_size,
    }, ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}")
    print(f"  Use this for fine-tuning with finetune_ci_emognition.py\n")


if __name__ == '__main__':
    main()
