"""
Quick Epoch Test for Mamba EEG Classifier (SEED-IV, 4 Channels).
WITH masked pooling — the model ignores zero-padded positions.

Runs 2 epochs only — measures per-epoch wall-clock time and prints
train/val accuracy + loss so you can sanity-check the pipeline before
launching a full training run.

Usage:
    python epochtest.py --dataset_path /path/to/SEED_IV

    On Kaggle:
        python epochtest.py --dataset_path /kaggle/input/datasets/phhasian0710/seed-iv/seed_iv
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
from sklearn.metrics import f1_score

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from mamba_model import MambaEEGClassifier
from dataset import SeedIVClipDataset, load_seediv_clips


def setup_seed(seed=2024):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Quick 2-epoch timing test (with masking)")
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to SEED-IV root')
    parser.add_argument('--sessions', nargs='+', type=int, default=[1],
                        help='Sessions to load (default: [1] for speed)')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--d_model', type=int, default=128)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--percentile', type=float, default=95,
                        help='Percentile for fixed_length clipping (default: 95)')
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    setup_seed(args.seed)
    device = torch.device(args.device)

    # ── Load data ──
    print(f"\n{'='*60}")
    print(f"EPOCH TEST — Mamba EEG (WITH MASKED POOLING)")
    print(f"  Dataset : {args.dataset_path}")
    print(f"  Sessions: {args.sessions}")
    print(f"  Device  : {device}")
    print(f"{'='*60}\n")

    print("Loading SEED-IV raw data...")
    t0 = time.time()
    trials, labels, subject_ids, session_ids = load_seediv_clips(
        args.dataset_path, sessions=args.sessions
    )
    load_time = time.time() - t0
    print(f"  Loaded in {load_time:.1f}s\n")

    # ── Percentile-based fixed length ──
    lengths = [t.shape[1] for t in trials]
    fixed_length_max = max(lengths)
    fixed_length = int(np.percentile(lengths, args.percentile))
    print(f"  Trial lengths  : min={min(lengths)}, max={max(lengths)}, "
          f"mean={np.mean(lengths):.0f}")
    print(f"  Percentile {args.percentile:.0f}% : {fixed_length}  "
          f"(vs max={fixed_length_max}  → {100*(1-fixed_length/fixed_length_max):.0f}% shorter)")

    # ── Simple 70/15/15 split ──
    n = len(trials)
    indices = np.arange(n)
    np.random.shuffle(indices)
    n_test = int(n * 0.15)
    n_val = int(n * 0.15)
    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    print(f"\n  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ── Build datasets ──
    print(f"\n  Creating datasets (fixed_length={fixed_length})...")
    train_ds = SeedIVClipDataset(
        [trials[i] for i in train_idx],
        [labels[i] for i in train_idx],
        fixed_length=fixed_length, augment=False, filter_eeg=True, normalize=True
    )
    val_ds = SeedIVClipDataset(
        [trials[i] for i in val_idx],
        [labels[i] for i in val_idx],
        fixed_length=fixed_length, augment=False, filter_eeg=True, normalize=True
    )
    test_ds = SeedIVClipDataset(
        [trials[i] for i in test_idx],
        [labels[i] for i in test_idx],
        fixed_length=fixed_length, augment=False, filter_eeg=True, normalize=True
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
    print(f"  Masking: ENABLED (padded positions ignored in pooling)")

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    # ── Train 2 epochs ──
    print(f"\n{'='*60}")
    print(f"  Running 2 training epochs (masked pooling)...")
    print(f"{'='*60}\n")

    for epoch in range(1, 3):
        # --- Training ---
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0
        t_start = time.time()

        for batch_i, (batch_x, batch_y, batch_lengths) in enumerate(train_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.long().to(device) if isinstance(batch_y, torch.Tensor) \
                      else torch.tensor(batch_y, dtype=torch.long, device=device)
            batch_lengths = batch_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x, lengths=batch_lengths)
            loss = criterion(outputs, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            epoch_correct += (preds == batch_y).sum().item()
            epoch_total += len(batch_y)

            # Progress every 5 batches
            if (batch_i + 1) % 5 == 0 or batch_i == 0:
                elapsed = time.time() - t_start
                print(f"    Epoch {epoch} | Batch {batch_i+1}/{len(train_loader)} | "
                      f"Loss: {loss.item():.4f} | "
                      f"Elapsed: {elapsed:.1f}s")

        train_time = time.time() - t_start
        train_loss = epoch_loss / max(len(train_loader), 1)
        train_acc = epoch_correct / max(epoch_total, 1)

        # --- Validation ---
        model.eval()
        val_loss_sum = 0.0
        val_correct = 0
        val_total = 0
        val_preds_all = []
        val_labels_all = []

        t_val = time.time()
        with torch.no_grad():
            for batch_x, batch_y, batch_lengths in val_loader:
                batch_x = batch_x.to(device)
                batch_y = batch_y.long().to(device) if isinstance(batch_y, torch.Tensor) \
                          else torch.tensor(batch_y, dtype=torch.long, device=device)
                batch_lengths = batch_lengths.to(device)

                outputs = model(batch_x, lengths=batch_lengths)
                loss = criterion(outputs, batch_y)
                val_loss_sum += loss.item()
                preds = torch.argmax(outputs, dim=1)
                val_correct += (preds == batch_y).sum().item()
                val_total += len(batch_y)
                val_preds_all.extend(preds.cpu().numpy())
                val_labels_all.extend(batch_y.cpu().numpy())

        val_time = time.time() - t_val
        val_loss = val_loss_sum / max(len(val_loader), 1)
        val_acc = val_correct / max(val_total, 1)
        val_f1 = f1_score(val_labels_all, val_preds_all, average='macro', zero_division=0)

        print(f"\n  ┌─ Epoch {epoch} Summary {'─'*38}")
        print(f"  │ Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f}")
        print(f"  │ Val   Loss: {val_loss:.4f}  Acc: {val_acc:.4f}  F1: {val_f1:.4f}")
        print(f"  │ Train time: {train_time:.1f}s  Val time: {val_time:.1f}s")
        print(f"  │ Total     : {train_time + val_time:.1f}s  per epoch")
        print(f"  └{'─'*50}\n")

    # ── Quick test set eval ──
    model.eval()
    test_correct = 0
    test_total = 0
    test_preds_all = []
    test_labels_all = []

    with torch.no_grad():
        for batch_x, batch_y, batch_lengths in test_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.long().to(device) if isinstance(batch_y, torch.Tensor) \
                      else torch.tensor(batch_y, dtype=torch.long, device=device)
            batch_lengths = batch_lengths.to(device)

            outputs = model(batch_x, lengths=batch_lengths)
            preds = torch.argmax(outputs, dim=1)
            test_correct += (preds == batch_y).sum().item()
            test_total += len(batch_y)
            test_preds_all.extend(preds.cpu().numpy())
            test_labels_all.extend(batch_y.cpu().numpy())

    test_acc = test_correct / max(test_total, 1)
    test_f1 = f1_score(test_labels_all, test_preds_all, average='macro', zero_division=0)

    print(f"{'='*60}")
    print(f"  TEST SET (after 2 epochs, masked pooling)")
    print(f"    Accuracy : {test_acc:.4f}")
    print(f"    Macro-F1 : {test_f1:.4f}")
    print(f"{'='*60}")

    # ── Timing summary ──
    print(f"\n  Estimated time for 100 epochs: "
          f"~{((train_time + val_time) * 100) / 3600:.1f} hours")
    print(f"  Done!\n")


if __name__ == '__main__':
    main()
