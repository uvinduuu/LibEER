"""
Windowed Mamba EEG Training Script for SEED-IV (4 Channels).

Uses fixed-size windows (default 10s) instead of full clips.
  - NO zero-padding (all windows are the same size)
  - NO clip leakage (split at trial level BEFORE windowing)
  - Much faster training (~30-40x vs full-clip approach)

Usage:
    Quick test (2 epochs, 1 session):
        python train_windowed.py --dataset_path /path/to/SEED_IV --epochs 2 --sessions 1

    Full training (all sessions):
        python train_windowed.py --dataset_path /path/to/SEED_IV --epochs 100

    Custom window size:
        python train_windowed.py --dataset_path /path/to/SEED_IV --window_sec 5.0
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
from windowed_dataset import create_windowed_splits


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

            # No lengths needed — no padding!
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


def main():
    parser = argparse.ArgumentParser(description="Windowed Mamba EEG Training")

    # Dataset
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to SEED-IV root')
    parser.add_argument('--sessions', nargs='+', type=int, default=None,
                        help='Sessions to use (1-3). Default: all')
    parser.add_argument('--window_sec', type=float, default=10.0,
                        help='Window size in seconds (default: 10.0)')

    # Model
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.3)

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32, can be larger since windows are short)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience (0=disabled, default: 15)')

    # Split
    parser.add_argument('--train_ratio', type=float, default=0.7)
    parser.add_argument('--val_ratio', type=float, default=0.15)
    parser.add_argument('--test_ratio', type=float, default=0.15)

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
    print(f"WINDOWED MAMBA EEG CLASSIFIER — SEED-IV")
    print(f"  Dataset    : {args.dataset_path}")
    print(f"  Sessions   : {args.sessions or 'all'}")
    print(f"  Window     : {args.window_sec}s ({window_size} samples)")
    print(f"  Device     : {device}")
    print(f"  d_model={args.d_model}, layers={args.n_layers}, d_state={args.d_state}")
    print(f"  batch_size={args.batch_size}, lr={args.lr}, epochs={args.epochs}")
    print(f"{'='*60}")

    # ── Load trials ──
    print(f"\nLoading SEED-IV raw data...")
    t0 = time.time()
    trials, labels, subject_ids, session_ids = load_seediv_clips(
        args.dataset_path, sessions=args.sessions
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # ── Create windowed datasets (trial-level split, no leakage) ──
    print(f"\nCreating windowed datasets...")
    train_ds, val_ds, test_ds, split_info = create_windowed_splits(
        trials, labels, subject_ids,
        window_size=window_size,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        filter_eeg=True, normalize=True,
        sample_rate=sample_rate,
        seed=args.seed,
        augment_train=True,
    )

    print(f"\n  Total windows: train={split_info['train_windows']}, "
          f"val={split_info['val_windows']}, test={split_info['test_windows']}")
    print(f"  No zero-padding! All windows are exactly {window_size} samples.")

    # ── Dataloaders ──
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=False)
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
    mamba_seq_len = window_size // 16  # Approx after conv stem
    print(f"  Mamba sequence length: ~{mamba_seq_len} steps (down from {window_size})")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    # ── Training loop ──
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
        t_epoch = time.time()

        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.long().to(device) if isinstance(batch_y, torch.Tensor) \
                      else torch.tensor(batch_y, dtype=torch.long, device=device)

            optimizer.zero_grad()
            outputs = model(batch_x)  # No lengths — no padding!
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            epoch_correct += (preds == batch_y).sum().item()
            epoch_total += len(batch_y)

        scheduler.step()

        train_time = time.time() - t_epoch
        train_loss = epoch_loss / max(len(train_loader), 1)
        train_acc = epoch_correct / max(epoch_total, 1)

        # Validate
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, criterion)
        val_time = time.time() - t_epoch - train_time
        total_time = train_time + val_time
        epoch_times.append(total_time)

        # Print every epoch (they're fast now!)
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f} | "
              f"{total_time:.1f}s | LR: {scheduler.get_last_lr()[0]:.6f}")

        # Best model tracking
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # ── Load best model and test ──
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model = model.to(device)

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, device, criterion
    )

    avg_epoch_time = np.mean(epoch_times)
    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"  Best Val F1 : {best_val_f1:.4f}")
    print(f"  Test Acc    : {test_acc:.4f}")
    print(f"  Test F1     : {test_f1:.4f}")
    print(f"  Avg epoch   : {avg_epoch_time:.1f}s")
    print(f"  Total time  : {sum(epoch_times)/60:.1f} min")

    print_test_report(test_labels, test_preds, title="Test Set")

    # ── Save checkpoint ──
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', 'windowed')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
    torch.save({
        'model': model.state_dict(),
        'val_f1': best_val_f1,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'window_size': window_size,
        'split_info': split_info,
    }, ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}")
    print(f"  Done!\n")


if __name__ == '__main__':
    main()
