"""
Windowed Mamba EEG Training for Emognition Dataset (Muse 2, 4 Channels).

Uses fixed-size windows (default 10s at 256Hz = 2560 samples) with:
  - NO zero-padding
  - NO clip leakage (trial-level split before windowing)
  - Subject-dependent or subject-independent (LOSO) modes

Usage:
    Subject-dependent (quick test):
        python train_mamba_emognition.py --data_root /path/to/emognition --epochs 2

    Subject-dependent (full):
        python train_mamba_emognition.py --data_root /path/to/emognition --epochs 100

    Subject-independent (LOSO):
        python train_mamba_emognition.py --data_root /path/to/emognition --mode sub_indep --epochs 50

    Kaggle:
        python train_mamba_emognition.py --data_root /kaggle/input/datasets/uvindukodikara/emognition --epochs 100
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
from emognition_loader import load_emognition_trials, FS, EMOTIONS_4CLASS

# Import from mamba/ folder
mamba_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'mamba')
sys.path.insert(0, mamba_dir)
from mamba_model import MambaEEGClassifier
from windowed_dataset import (
    create_windowed_splits, WindowedEEGDataset,
    split_trial_into_windows, bandpass_filter, normalize_trial
)


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


def print_test_report(y_true, y_pred, class_names, title=""):
    """Print confusion matrix and per-class metrics."""
    n_cls = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n_cls)))
    print(f"\n  Confusion Matrix{' (' + title + ')' if title else ''}:")
    print(f"  {'':>12}", end="")
    for name in class_names:
        print(f"{name:>12}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"  {name:>12}", end="")
        for j in range(n_cls):
            print(f"{cm[i][j]:>12}", end="")
        print()

    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def run_subject_dependent(trials, labels, subject_ids, class_names, args, device):
    """Subject-dependent: pool all trials, random split at trial level."""
    sample_rate = FS
    window_size = int(args.window_sec * sample_rate)

    print(f"\n{'#'*60}")
    print(f"# SUBJECT-DEPENDENT MODE")
    print(f"# {len(trials)} trials, trial-level split, {args.window_sec}s windows")
    print(f"{'#'*60}")

    train_ds, val_ds, test_ds, split_info = create_windowed_splits(
        trials, labels, subject_ids,
        window_size=window_size,
        train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
        filter_eeg=True, normalize=True,
        sample_rate=sample_rate,
        seed=args.seed,
        augment_train=True,
    )

    return _train_and_eval(
        train_ds, val_ds, test_ds, split_info,
        class_names, args, device, split_name="sub_dep"
    )


def run_subject_independent(trials, labels, subject_ids, class_names, args, device):
    """Subject-independent LOSO: train on N-1 subjects, test on 1."""
    sample_rate = FS
    window_size = int(args.window_sec * sample_rate)
    unique_subjects = sorted(set(subject_ids))

    print(f"\n{'#'*60}")
    print(f"# SUBJECT-INDEPENDENT MODE (LOSO)")
    print(f"# {len(unique_subjects)} subjects, {len(trials)} trials")
    print(f"{'#'*60}")

    all_results = []
    all_preds = []
    all_true = []

    for test_subj in unique_subjects:
        print(f"\n{'─'*40}")
        print(f"  LOSO: Test Subject {test_subj}")
        print(f"{'─'*40}")

        # Split by subject
        test_idx = [i for i, s in enumerate(subject_ids) if s == test_subj]
        train_pool = [i for i, s in enumerate(subject_ids) if s != test_subj]

        if len(test_idx) == 0:
            print(f"  Skipping {test_subj} — no test trials")
            continue

        # Split train pool into train + val (by subject for proper independence)
        rng = np.random.RandomState(args.seed)
        remaining_subjects = sorted(set(subject_ids[i] for i in train_pool))
        rng.shuffle(remaining_subjects)
        n_val_subj = max(1, int(len(remaining_subjects) * 0.15))
        val_subjects = set(remaining_subjects[:n_val_subj])

        val_idx = [i for i in train_pool if subject_ids[i] in val_subjects]
        train_idx = [i for i in train_pool if subject_ids[i] not in val_subjects]

        # Preprocess and window each split
        def process_split(indices, split_name):
            windows, win_labels = [], []
            for idx in indices:
                trial = np.array(trials[idx], dtype=np.float32)
                trial = bandpass_filter(trial, lowcut=1.0, highcut=50.0, fs=sample_rate)
                trial = normalize_trial(trial)
                label = int(labels[idx])
                wins = split_trial_into_windows(trial, window_size)
                for w in wins:
                    windows.append(w)
                    win_labels.append(label)
            print(f"    {split_name:5s}: {len(indices):3d} trials → {len(windows):5d} windows")
            return windows, win_labels

        print(f"  Train subjects: {len(remaining_subjects) - n_val_subj}, "
              f"Val subjects: {n_val_subj}, Test subject: {test_subj}")

        train_windows, train_labels = process_split(train_idx, "Train")
        val_windows, val_labels = process_split(val_idx, "Val")
        test_windows, test_labels = process_split(test_idx, "Test")

        train_ds = WindowedEEGDataset(train_windows, train_labels, augment=True, sample_rate=sample_rate)
        val_ds = WindowedEEGDataset(val_windows, val_labels, augment=False, sample_rate=sample_rate)
        test_ds = WindowedEEGDataset(test_windows, test_labels, augment=False, sample_rate=sample_rate)

        split_info = {
            'train_windows': len(train_windows),
            'val_windows': len(val_windows),
            'test_windows': len(test_windows),
        }

        result = _train_and_eval(
            train_ds, val_ds, test_ds, split_info,
            class_names, args, device,
            split_name=f"loso_{test_subj}",
            verbose=False,
        )

        all_results.append(result)
        all_preds.extend(result['test_preds'])
        all_true.extend(result['test_labels'])

        print(f"  Subject {test_subj}: Acc={result['acc']:.4f}, F1={result['macro-f1']:.4f}")

    # Overall
    print(f"\n{'='*60}")
    print("OVERALL LOSO RESULTS")
    print(f"{'='*60}")

    accs = [r['acc'] for r in all_results]
    f1s = [r['macro-f1'] for r in all_results]
    print(f"  Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Macro-F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    print_test_report(all_true, all_preds, class_names, title="Overall LOSO")
    return all_results


def _train_and_eval(train_ds, val_ds, test_ds, split_info,
                    class_names, args, device, split_name="", verbose=True):
    """Train model on one split. Returns result dict."""

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    n_classes = len(class_names)
    model = MambaEEGClassifier(
        in_channels=4, num_classes=n_classes,
        d_model=args.d_model, n_layers=args.n_layers,
        d_state=args.d_state, dropout=args.dropout
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        print(f"\n  Model params: {n_params:,}")
        print(f"  Train: {split_info['train_windows']} windows, "
              f"Val: {split_info['val_windows']}, Test: {split_info['test_windows']}")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    if verbose:
        print(f"\n  Training {split_name} ({args.epochs} epochs)...")

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
        train_loss = epoch_loss / max(len(train_loader), 1)
        train_acc = epoch_correct / max(epoch_total, 1)

        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, criterion)
        epoch_time = time.time() - t_start

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"    Epoch {epoch:3d}/{args.epochs} | "
                  f"Train: {train_loss:.4f}/{train_acc:.4f} | "
                  f"Val: {val_loss:.4f}/{val_acc:.4f}/F1:{val_f1:.4f} | "
                  f"{epoch_time:.1f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if args.patience > 0 and patience_counter >= args.patience:
            if verbose:
                print(f"    Early stopping at epoch {epoch}")
            break

    # Load best model and evaluate on test
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model = model.to(device)

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, device, criterion
    )

    if verbose:
        print(f"\n  Best Val F1: {best_val_f1:.4f}")
        print(f"  Test Acc:    {test_acc:.4f}")
        print(f"  Test F1:     {test_f1:.4f}")
        print_test_report(test_labels, test_preds, class_names, title=split_name)

    # Save checkpoint
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', split_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'val_f1': best_val_f1,
        'test_acc': test_acc,
        'test_f1': test_f1,
    }, os.path.join(ckpt_dir, 'best_model.pt'))

    return {
        'acc': test_acc,
        'macro-f1': test_f1,
        'val_f1': best_val_f1,
        'test_preds': test_preds,
        'test_labels': test_labels,
    }


def main():
    parser = argparse.ArgumentParser(description="Windowed Mamba EEG — Emognition")

    # Dataset
    parser.add_argument('--data_root', type=str, required=True,
                        help='Path to Emognition dataset root')
    parser.add_argument('--mode', type=str, default='sub_dep',
                        choices=['sub_dep', 'sub_indep'],
                        help='sub_dep or sub_indep (LOSO)')
    parser.add_argument('--window_sec', type=float, default=10.0,
                        help='Window size in seconds (default: 10.0)')

    # Model
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.3)

    # Training
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=15)

    # Misc
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(args.device)

    window_size = int(args.window_sec * FS)

    # ── Header ──
    print(f"\n{'='*60}")
    print(f"WINDOWED MAMBA — EMOGNITION (Muse 2, 4ch)")
    print(f"  Dataset  : {args.data_root}")
    print(f"  Mode     : {args.mode}")
    print(f"  Window   : {args.window_sec}s ({window_size} samples at {FS}Hz)")
    print(f"  Device   : {device}")
    print(f"  d_model={args.d_model}, layers={args.n_layers}, d_state={args.d_state}")
    print(f"  batch={args.batch_size}, lr={args.lr}, epochs={args.epochs}")
    print(f"{'='*60}")

    # ── Load trials ──
    print(f"\nLoading Emognition trials...")
    t0 = time.time()
    trials, labels, subject_ids, lab2id, id2lab = load_emognition_trials(args.data_root)
    print(f"  Loaded in {time.time()-t0:.1f}s")

    class_names = [id2lab[i] for i in range(len(id2lab))]

    # ── Run experiment ──
    if args.mode == 'sub_dep':
        results = run_subject_dependent(trials, labels, subject_ids, class_names, args, device)
    elif args.mode == 'sub_indep':
        results = run_subject_independent(trials, labels, subject_ids, class_names, args, device)

    print(f"\n  Checkpoints saved to: {os.path.join(os.path.dirname(__file__), 'checkpoints')}")
    print(f"  Done!\n")


if __name__ == '__main__':
    main()
