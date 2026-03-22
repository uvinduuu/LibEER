"""
BiMamba Training for SEED-IV.

Combines all improvements targeting 50-55% accuracy:
  1. Bidirectional Mamba (BiMamba) — Phase 2 novelty
  2. d_model=128, n_layers=3 — more capacity
  3. Label smoothing — prevents overconfident predictions
  4. Cosine LR with linear warmup — better convergence
  5. Strong regularization — dropout=0.5, weight_decay=0.1

Usage:
    python train_bi_mamba.py --dataset_path /path/to/seed_iv --epochs 100

    Kaggle:
    python train_bi_mamba.py \
        --dataset_path /kaggle/input/datasets/phhasian0710/seed-iv/seed_iv \
        --sessions 1 2 3 \
        --epochs 100

    Subject-independent:
    python train_bi_mamba.py --dataset_path /path/to/seed_iv \
        --mode sub_indep --epochs 100
"""

import os
import sys
import argparse
import random
import math
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from bi_mamba_model import BiMambaEEGClassifier
from dataset import load_seediv_clips
from windowed_dataset import (
    create_windowed_splits, WindowedEEGDataset,
    split_trial_into_windows, bandpass_filter, normalize_trial
)


CLASS_NAMES = ['neutral', 'sad', 'fear', 'happy']


# ─── Label Smoothing Loss ──────────────────────────────────────

class LabelSmoothingCE(nn.Module):
    """Cross-entropy with label smoothing. Prevents overconfident logits."""
    def __init__(self, n_classes, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, logits, target):
        log_prob = torch.log_softmax(logits, dim=-1)
        # Smooth target distribution
        with torch.no_grad():
            smooth_target = torch.full_like(log_prob, self.smoothing / (self.n_classes - 1))
            smooth_target.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth_target * log_prob).sum(dim=-1).mean()


# ─── Warmup Cosine Scheduler ───────────────────────────────────

class WarmupCosineScheduler:
    """Linear warmup then cosine annealing."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        self.base_lrs      = [pg['lr'] for pg in optimizer.param_groups]
        self.current_epoch = 0

    def step(self):
        self.current_epoch += 1
        e = self.current_epoch
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if e <= self.warmup_epochs:
                pg['lr'] = base_lr * e / max(self.warmup_epochs, 1)
            else:
                progress  = (e - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
                pg['lr']  = self.min_lr + (base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


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


# ─── Main ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="BiMamba on SEED-IV — targeting 50-55%")

    # Data
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--sessions', nargs='+', type=int, default=None)
    parser.add_argument('--window_sec', type=float, default=10.0)
    parser.add_argument('--mode', choices=['sub_dep', 'sub_indep'], default='sub_dep')
    parser.add_argument('--n_test_subj', type=int, default=3)
    parser.add_argument('--n_val_subj',  type=int, default=2)

    # Model
    parser.add_argument('--d_model',  type=int,   default=128)
    parser.add_argument('--n_layers', type=int,   default=3)
    parser.add_argument('--d_state',  type=int,   default=16)
    parser.add_argument('--dropout',  type=float, default=0.5)

    # Training
    parser.add_argument('--batch_size',    type=int,   default=64)
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--weight_decay',  type=float, default=0.1)
    parser.add_argument('--warmup_epochs', type=int,   default=5)
    parser.add_argument('--label_smooth',  type=float, default=0.1)
    parser.add_argument('--patience',      type=int,   default=20)
    parser.add_argument('--seed',          type=int,   default=2024)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    setup_seed(args.seed)
    device      = torch.device(args.device)
    sample_rate = 200
    window_size = int(args.window_sec * sample_rate)

    print(f"\n{'='*60}")
    print(f"BiMamba — SEED-IV  (targeting 50-55%)")
    print(f"  Mode       : {args.mode}")
    print(f"  Window     : {args.window_sec}s → {window_size//16} Mamba steps")
    print(f"  Model      : d_model={args.d_model}, layers={args.n_layers}, BiMamba")
    print(f"  Regularize : dropout={args.dropout}, wd={args.weight_decay}, smooth={args.label_smooth}")
    print(f"  Schedule   : warmup={args.warmup_epochs} + cosine, lr={args.lr}")
    print(f"  Device     : {device}")
    print(f"{'='*60}")

    # Load SEED-IV
    print(f"\nLoading SEED-IV...")
    t0 = time.time()
    trials, labels, subject_ids, session_ids = load_seediv_clips(
        args.dataset_path, sessions=args.sessions
    )
    print(f"  Loaded {len(trials)} trials in {time.time()-t0:.1f}s")

    # Splits
    if args.mode == 'sub_dep':
        train_ds, val_ds, test_ds, info = create_windowed_splits(
            trials, labels, subject_ids,
            window_size=window_size,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            filter_eeg=True, normalize=True,
            sample_rate=sample_rate, seed=args.seed, augment_train=True,
        )
        print(f"  sub_dep: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)} windows")
    else:
        rng  = np.random.RandomState(args.seed)
        subj = sorted(set(subject_ids)); rng.shuffle(subj)
        test_s  = set(subj[:args.n_test_subj])
        val_s   = set(subj[args.n_test_subj:args.n_test_subj + args.n_val_subj])
        train_s = set(s for s in subj if s not in test_s | val_s)
        print(f"  sub_indep: train={sorted(train_s)} | val={sorted(val_s)} | test={sorted(test_s)}")

        def make_ds(subj_set, aug):
            wins, lbls = [], []
            for tri, lbl, sid in zip(trials, labels, subject_ids):
                if sid not in subj_set: continue
                t = np.array(tri, dtype=np.float32)
                t = bandpass_filter(t, fs=sample_rate)
                t = normalize_trial(t)
                for w in split_trial_into_windows(t, window_size):
                    wins.append(w); lbls.append(int(lbl))
            return WindowedEEGDataset(wins, lbls, augment=aug, sample_rate=sample_rate)

        train_ds = make_ds(train_s, True)
        val_ds   = make_ds(val_s,   False)
        test_ds  = make_ds(test_s,  False)
        print(f"  Windows: train={len(train_ds)}, val={len(val_ds)}, test={len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Model
    model = BiMambaEEGClassifier(
        in_channels=4, num_classes=4,
        d_model=args.d_model, n_layers=args.n_layers,
        d_state=args.d_state, dropout=args.dropout,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  BiMamba params : {n_params:,}")
    print(f"  Mamba seq len  : ~{window_size // 16} steps")

    criterion = LabelSmoothingCE(n_classes=4, smoothing=args.label_smooth)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, eps=1e-8)
    scheduler = WarmupCosineScheduler(optimizer,
                                       warmup_epochs=args.warmup_epochs,
                                       total_epochs=args.epochs,
                                       min_lr=1e-7)

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

        # Use standard CE for val (no smoothing — measures real accuracy)
        va_loss, va_acc, va_f1, _, _ = evaluate(
            model, val_loader, device, nn.CrossEntropyLoss()
        )
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
    te_loss, te_acc, te_f1, te_preds, te_labels = evaluate(
        model, test_loader, device, nn.CrossEntropyLoss()
    )

    print(f"\n{'='*60}")
    print(f"RESULTS — BiMamba on SEED-IV ({args.mode})")
    print(f"{'='*60}")
    print(f"  Best Val F1 : {best_val_f1:.4f}")
    print(f"  Test Acc    : {te_acc:.4f}")
    print(f"  Test F1     : {te_f1:.4f}")
    print(f"  Avg epoch   : {np.mean(epoch_times):.1f}s")
    print(f"  Total time  : {sum(epoch_times)/60:.1f} min")
    print_report(te_labels, te_preds, title="SEED-IV Test")

    ckpt_dir  = os.path.join(os.path.dirname(__file__), 'checkpoints', 'bi_mamba')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'best_model.pt')
    torch.save({
        'model': model.state_dict(),
        'model_cfg': {
            'in_channels': 4, 'num_classes': 4,
            'd_model': args.d_model, 'n_layers': args.n_layers,
            'd_state': args.d_state, 'dropout': args.dropout,
        },
        'val_f1':   best_val_f1,
        'test_acc': te_acc,
        'test_f1':  te_f1,
    }, ckpt_path)
    print(f"\n  Checkpoint: {ckpt_path}\n")


if __name__ == '__main__':
    main()
