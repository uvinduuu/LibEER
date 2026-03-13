"""
Windowed Mamba Training — Subject-Independent Split for SEED-IV.

Splits SUBJECTS (not trials) into train/val/test so no subject appears
in more than one set. This is faster than LOSO (single run) while still
being truly subject-independent.

Default split for 15 subjects: 10 train / 2 val / 3 test.

Usage:
    python train_subject_split.py --dataset_path /path/to/SEED_IV --epochs 100

    # Custom split
    python train_subject_split.py --dataset_path /path/to/SEED_IV --n_test_subj 3 --n_val_subj 2

    # Specify exact test subjects
    python train_subject_split.py --dataset_path /path/to/SEED_IV --test_subjects 0 1 2
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

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from mamba_model import MambaEEGClassifier
from dataset import load_seediv_clips
from windowed_dataset import (
    WindowedEEGDataset, split_trial_into_windows,
    bandpass_filter, normalize_trial
)


CLASS_NAMES = ['neutral', 'sad', 'fear', 'happy']


def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def evaluate(model, dataloader, device, criterion):
    """Evaluate model. Returns loss, accuracy, macro-F1, predictions, labels."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.long().to(device) if isinstance(batch_y, torch.Tensor) \
                      else torch.tensor(batch_y, dtype=torch.long, device=device)

            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)

            all_preds.extend(torch.argmax(outputs, dim=1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, acc, macro_f1, all_preds, all_labels


def print_test_report(y_true, y_pred, title=""):
    """Print confusion matrix and per-class metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    print(f"\n  Confusion Matrix{' (' + title + ')' if title else ''}:")
    print(f"  {'':>10}", end="")
    for name in CLASS_NAMES:
        print(f"{name:>10}", end="")
    print()
    for i, name in enumerate(CLASS_NAMES):
        print(f"  {name:>10}", end="")
        for j in range(4):
            print(f"{cm[i][j]:>10}", end="")
        print()
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))


def create_subject_split_datasets(
    trials, labels, subject_ids,
    test_subjects, val_subjects, train_subjects,
    window_size, sample_rate=200,
):
    """
    Create windowed datasets split by SUBJECT.

    No subject appears in more than one set → truly subject-independent.
    Windows from the same trial always stay together (no clip leakage).
    """

    def process_subjects(subj_set, split_name, augment=False):
        """Preprocess and window all trials belonging to the given subjects."""
        windows, win_labels = [], []
        trial_count = 0

        for idx, (trial, label, subj) in enumerate(zip(trials, labels, subject_ids)):
            if subj not in subj_set:
                continue
            trial_count += 1

            trial_data = np.array(trial, dtype=np.float32)
            trial_data = bandpass_filter(trial_data, lowcut=1.0, highcut=50.0, fs=sample_rate)
            trial_data = normalize_trial(trial_data)

            wins = split_trial_into_windows(trial_data, window_size)
            for w in wins:
                windows.append(w)
                win_labels.append(int(label))

        print(f"    {split_name:5s}: subjects={sorted(subj_set)} → "
              f"{trial_count:3d} trials → {len(windows):5d} windows")

        ds = WindowedEEGDataset(windows, win_labels, augment=augment, sample_rate=sample_rate)
        return ds, win_labels

    print(f"\n  Creating subject-split windowed datasets:")
    train_ds, train_labels = process_subjects(train_subjects, "Train", augment=True)
    val_ds, val_labels = process_subjects(val_subjects, "Val", augment=False)
    test_ds, test_labels = process_subjects(test_subjects, "Test", augment=False)

    # Label distribution per split
    from collections import Counter
    for name, lbls in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        dist = Counter(lbls)
        print(f"    {name:5s} labels: {dict(sorted(dist.items()))}")

    return train_ds, val_ds, test_ds


def main():
    parser = argparse.ArgumentParser(description="Subject-Split Windowed Mamba — SEED-IV")

    # Dataset
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--sessions', nargs='+', type=int, default=None,
                        help='Sessions to use (1-3). Default: all')
    parser.add_argument('--window_sec', type=float, default=10.0,
                        help='Window size in seconds (default: 10.0)')

    # Subject split
    parser.add_argument('--n_test_subj', type=int, default=3,
                        help='Number of test subjects (default: 3)')
    parser.add_argument('--n_val_subj', type=int, default=2,
                        help='Number of validation subjects (default: 2)')
    parser.add_argument('--test_subjects', nargs='+', type=int, default=None,
                        help='Specific test subject IDs (overrides n_test_subj)')
    parser.add_argument('--val_subjects', nargs='+', type=int, default=None,
                        help='Specific val subject IDs (overrides n_val_subj)')

    # Model
    parser.add_argument('--d_model', type=int, default=64)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.5)

    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--patience', type=int, default=20)

    # Misc
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(args.device)
    sample_rate = 200
    window_size = int(args.window_sec * sample_rate)

    # ── Header ──
    print(f"\n{'='*60}")
    print(f"SUBJECT-SPLIT WINDOWED MAMBA — SEED-IV")
    print(f"  Dataset  : {args.dataset_path}")
    print(f"  Sessions : {args.sessions or 'all'}")
    print(f"  Window   : {args.window_sec}s ({window_size} samples)")
    print(f"  Device   : {device}")
    print(f"  d_model={args.d_model}, layers={args.n_layers}, dropout={args.dropout}")
    print(f"  batch={args.batch_size}, lr={args.lr}, wd={args.weight_decay}")
    print(f"{'='*60}")

    # ── Load trials ──
    print(f"\nLoading SEED-IV raw data...")
    t0 = time.time()
    trials, labels, subject_ids, session_ids = load_seediv_clips(
        args.dataset_path, sessions=args.sessions
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Determine subject split ──
    unique_subjects = sorted(set(subject_ids))
    n_subjects = len(unique_subjects)
    print(f"\n  Total subjects: {n_subjects} → {unique_subjects}")

    rng = np.random.RandomState(args.seed)

    if args.test_subjects is not None:
        test_subjects = set(args.test_subjects)
    else:
        shuffled = list(unique_subjects)
        rng.shuffle(shuffled)
        test_subjects = set(shuffled[:args.n_test_subj])

    if args.val_subjects is not None:
        val_subjects = set(args.val_subjects)
    else:
        remaining = [s for s in unique_subjects if s not in test_subjects]
        rng.shuffle(remaining)
        val_subjects = set(remaining[:args.n_val_subj])

    train_subjects = set(s for s in unique_subjects if s not in test_subjects and s not in val_subjects)

    print(f"\n  Subject split:")
    print(f"    Train ({len(train_subjects)}): {sorted(train_subjects)}")
    print(f"    Val   ({len(val_subjects)}): {sorted(val_subjects)}")
    print(f"    Test  ({len(test_subjects)}): {sorted(test_subjects)}")

    # ── Create datasets ──
    train_ds, val_ds, test_ds = create_subject_split_datasets(
        trials, labels, subject_ids,
        test_subjects, val_subjects, train_subjects,
        window_size, sample_rate=sample_rate,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # ── Model ──
    model = MambaEEGClassifier(
        in_channels=4, num_classes=4,
        d_model=args.d_model, n_layers=args.n_layers,
        d_state=args.d_state, dropout=args.dropout
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params: {n_params:,}")
    print(f"  Mamba seq length: ~{window_size // 16} steps")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training ──
    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0
    epoch_times = []

    print(f"\n{'='*60}")
    print(f"Training ({args.epochs} epochs, {len(train_loader)} batches/epoch)")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t_start = time.time()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.long().to(device) if isinstance(batch_y, torch.Tensor) \
                      else torch.tensor(batch_y, dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            epoch_correct += (preds == batch_y).sum().item()
            epoch_total += len(batch_y)

        scheduler.step()
        train_time = time.time() - t_start
        train_loss = epoch_loss / max(len(train_loader), 1)
        train_acc = epoch_correct / max(epoch_total, 1)

        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, criterion)
        total_time = time.time() - t_start
        epoch_times.append(total_time)

        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f} | "
              f"{total_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # ── Test ──
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model = model.to(device)

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, device, criterion
    )

    avg_epoch = np.mean(epoch_times)
    print(f"\n{'='*60}")
    print(f"RESULTS — Subject-Independent Split")
    print(f"{'='*60}")
    print(f"  Train subjects: {sorted(train_subjects)}")
    print(f"  Val subjects  : {sorted(val_subjects)}")
    print(f"  Test subjects : {sorted(test_subjects)}")
    print(f"  Best Val F1   : {best_val_f1:.4f}")
    print(f"  Test Acc      : {test_acc:.4f}")
    print(f"  Test Macro-F1 : {test_f1:.4f}")
    print(f"  Avg epoch     : {avg_epoch:.1f}s")
    print(f"  Total time    : {sum(epoch_times)/60:.1f} min")

    print_test_report(test_labels, test_preds, title="Subject-Independent")

    # ── Save ──
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', 'subject_split')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
    torch.save({
        'model': model.state_dict(),
        'val_f1': best_val_f1,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'train_subjects': sorted(train_subjects),
        'val_subjects': sorted(val_subjects),
        'test_subjects': sorted(test_subjects),
    }, ckpt_path)
    print(f"\n  Checkpoint: {ckpt_path}")
    print(f"  Done!\n")


if __name__ == '__main__':
    main()
