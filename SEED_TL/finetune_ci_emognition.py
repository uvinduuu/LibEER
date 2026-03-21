"""
Fine-tune CiMamba (pre-trained on SEED-IV) on Emognition.

Two-phase fine-tuning:
    Phase A — Head only (5 epochs):  Freeze encoder, train new classifier head
    Phase B — Full model (N epochs): Unfreeze all, train with low LR

The channel-independent architecture allows transfer despite different
channel positions: SEED-IV (TP7/F7/F8/TP8) → Emognition (TP9/AF7/AF8/TP10)

Usage:
    python finetune_ci_emognition.py \
        --data_root /path/to/emognition \
        --pretrained /path/to/ci_seed/best_model.pt \
        --epochs 50

    # Kaggle
    python finetune_ci_emognition.py \
        --data_root /kaggle/input/datasets/uvindukodikara/emognition \
        --pretrained /kaggle/working/LibEER/SEED_TL/checkpoints/ci_seed/best_model.pt \
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
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from collections import Counter

# Local
sys.path.insert(0, os.path.dirname(__file__))
from ci_mamba_model import CiMambaClassifier

# Emognition loader
emog_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'emognition', 'emognition_mamba')
sys.path.insert(0, emog_dir)
from emognition_loader import load_emognition_trials, FS

# Windowed dataset utilities
mamba_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mamba')
sys.path.insert(0, mamba_dir)
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


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.long().to(device) if isinstance(by, torch.Tensor) \
                 else torch.tensor(by, dtype=torch.long, device=device)
            out  = model(bx)
            loss = criterion(out, by)
            all_preds.extend(torch.argmax(out, 1).cpu().numpy())
            all_labels.extend(by.cpu().numpy())
            total_loss += loss.item()
            n_batches  += 1

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / max(n_batches, 1), acc, f1, all_preds, all_labels


def print_report(y_true, y_pred, class_names, title=""):
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    print(f"\n  Confusion Matrix{' (' + title + ')' if title else ''}:")
    print(f"  {'':>14}", end="")
    for name in class_names: print(f"{name:>14}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"  {name:>14}", end="")
        for j in range(n): print(f"{cm[i][j]:>14}", end="")
        print()
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def train_one_phase(model, loader, optimizer, criterion, device, n_epochs,
                    phase_name, scheduler=None):
    """Run a training phase. Returns per-epoch (train_loss, train_acc) list."""
    history = []
    for epoch in range(1, n_epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for bx, by in loader:
            bx = bx.to(device)
            by = by.long().to(device) if isinstance(by, torch.Tensor) \
                 else torch.tensor(by, dtype=torch.long, device=device)
            optimizer.zero_grad()
            out  = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            ep_loss    += loss.item()
            ep_correct += (torch.argmax(out, 1) == by).sum().item()
            ep_total   += len(by)
        if scheduler: scheduler.step()
        ep_loss /= max(len(loader), 1)
        ep_acc   = ep_correct / max(ep_total, 1)
        history.append((ep_loss, ep_acc))
        print(f"  [{phase_name}] Epoch {epoch:3d}/{n_epochs} | "
              f"Loss: {ep_loss:.4f}, Acc: {ep_acc:.4f}")
    return history


def main():
    parser = argparse.ArgumentParser(description="Fine-tune CiMamba on Emognition")

    # Data
    parser.add_argument('--data_root',  required=True)
    parser.add_argument('--pretrained', required=True,
                        help='Path to SEED-IV pretrained checkpoint (.pt)')
    parser.add_argument('--mode', choices=['sub_dep', 'sub_indep'], default='sub_dep')
    parser.add_argument('--window_sec', type=float, default=10.0)

    # Fine-tuning
    parser.add_argument('--head_epochs',  type=int,   default=5,
                        help='Phase A: epochs to train head only (frozen backbone)')
    parser.add_argument('--epochs',       type=int,   default=50,
                        help='Phase B: epochs for full fine-tuning')
    parser.add_argument('--head_lr',      type=float, default=1e-3,
                        help='LR for head-only phase')
    parser.add_argument('--finetune_lr',  type=float, default=5e-5,
                        help='LR for full fine-tuning phase')
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--patience',     type=int,   default=15)
    parser.add_argument('--seed',         type=int,   default=2024)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    setup_seed(args.seed)
    device     = torch.device(args.device)
    sample_rate = FS
    window_size = int(args.window_sec * sample_rate)

    print(f"\n{'='*60}")
    print(f"CiMamba FINE-TUNING — SEED-IV → EMOGNITION")
    print(f"  Pretrained : {args.pretrained}")
    print(f"  Data root  : {args.data_root}")
    print(f"  Mode       : {args.mode}")
    print(f"  Window     : {args.window_sec}s ({window_size} samples at {sample_rate}Hz)")
    print(f"  Phase A    : {args.head_epochs} epochs (head only, lr={args.head_lr})")
    print(f"  Phase B    : {args.epochs} epochs (full fine-tune, lr={args.finetune_lr})")
    print(f"{'='*60}")

    # ── Load Emognition data ──
    print(f"\nLoading Emognition trials...")
    t0 = time.time()
    trials, labels, subject_ids, lab2id, id2lab = load_emognition_trials(args.data_root)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    class_names = [id2lab[i] for i in range(len(id2lab))]
    n_classes   = len(class_names)

    # ── Create windowed datasets ──
    if args.mode == 'sub_dep':
        train_ds, val_ds, test_ds, info = create_windowed_splits(
            trials, labels, subject_ids,
            window_size=window_size,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            filter_eeg=True, normalize=True,
            sample_rate=sample_rate, seed=args.seed, augment_train=True,
        )
    else:
        # Subject-independent: split by subject
        unique_subj = sorted(set(subject_ids))
        rng = np.random.RandomState(args.seed)
        shuffled = list(unique_subj)
        rng.shuffle(shuffled)
        n_test = max(1, int(len(unique_subj) * 0.15))
        n_val  = max(1, int(len(unique_subj) * 0.15))
        test_subj  = set(shuffled[:n_test])
        val_subj   = set(shuffled[n_test:n_test + n_val])
        train_subj = set(s for s in unique_subj if s not in test_subj | val_subj)

        print(f"\n  Subject split:")
        print(f"    Train: {sorted(train_subj)}")
        print(f"    Val  : {sorted(val_subj)}")
        print(f"    Test : {sorted(test_subj)}")

        def make_ds(subj_set, augment):
            wins, lbls = [], []
            for trial, label, subj in zip(trials, labels, subject_ids):
                if subj not in subj_set: continue
                t = np.array(trial, dtype=np.float32)
                t = bandpass_filter(t, fs=sample_rate)
                t = normalize_trial(t)
                for w in split_trial_into_windows(t, window_size):
                    wins.append(w); lbls.append(int(label))
            ds = WindowedEEGDataset(wins, lbls, augment=augment, sample_rate=sample_rate)
            print(f"    {'Train' if augment else 'Val/Test'}: {len(ds)} windows")
            return ds

        train_ds = make_ds(train_subj, True)
        val_ds   = make_ds(val_subj,   False)
        test_ds  = make_ds(test_subj,  False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ── Load pre-trained model ──
    print(f"\nLoading pre-trained CiMamba from: {args.pretrained}")
    ckpt     = torch.load(args.pretrained, map_location='cpu')
    cfg      = ckpt['model_cfg']

    # Build model with the same encoder but NEW classifier head for Emognition
    model = CiMambaClassifier(
        n_channels=cfg['n_channels'],
        num_classes=n_classes,           # ← NEW: Emognition classes
        d_model=cfg['d_model'],
        n_layers=cfg['n_layers'],
        d_state=cfg['d_state'],
        dropout=cfg['dropout'],
        aggregation=cfg['aggregation'],
    )

    # Load encoder weights, skip classifier head (different class count)
    state_dict      = ckpt['model']
    model_state     = model.state_dict()
    matched, skipped = [], []
    for k, v in state_dict.items():
        if k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            matched.append(k)
        else:
            skipped.append(k)

    model.load_state_dict(model_state)
    model = model.to(device)

    print(f"  Loaded {len(matched)} weight tensors from checkpoint")
    print(f"  Skipped {len(skipped)} (re-initialized): {skipped}")
    print(f"  Pretrained SEED-IV performance: "
          f"val_f1={ckpt.get('val_f1', '?'):.4f}, "
          f"test_acc={ckpt.get('test_acc', '?'):.4f}")

    criterion = nn.CrossEntropyLoss()

    # ════════════════════════════════════════════════
    # Phase A — Head only
    # ════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"Phase A: Head-only training ({args.head_epochs} epochs)")
    print(f"  Encoder is FROZEN. Only classifier head trains.")
    print(f"{'─'*60}")

    # Freeze encoder
    for param in model.encoder.parameters():
        param.requires_grad = False
    if hasattr(model, 'ch_attn'):
        for param in model.ch_attn.parameters():
            param.requires_grad = False

    head_optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.head_lr, weight_decay=args.weight_decay
    )
    train_one_phase(model, train_loader, head_optimizer, criterion,
                    device, args.head_epochs, "Head")

    va_loss_a, va_acc_a, va_f1_a, _, _ = evaluate(model, val_loader, device, criterion)
    print(f"  Phase A Val → Acc: {va_acc_a:.4f}, F1: {va_f1_a:.4f}")

    # ════════════════════════════════════════════════
    # Phase B — Full fine-tuning
    # ════════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"Phase B: Full fine-tuning ({args.epochs} epochs, lr={args.finetune_lr})")
    print(f"  All layers unfrozen. Encoder fine-tunes with low LR.")
    print(f"{'─'*60}")

    # Unfreeze all
    for param in model.parameters():
        param.requires_grad = True

    ft_optimizer = optim.AdamW(model.parameters(),
                                lr=args.finetune_lr, weight_decay=args.weight_decay)
    ft_scheduler = optim.lr_scheduler.CosineAnnealingLR(
        ft_optimizer, T_max=args.epochs, eta_min=1e-7
    )

    best_val_f1, best_state, patience_ctr = 0.0, None, 0
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        t0 = time.time()

        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.long().to(device) if isinstance(by, torch.Tensor) \
                 else torch.tensor(by, dtype=torch.long, device=device)
            ft_optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ft_optimizer.step()
            ep_loss    += loss.item()
            ep_correct += (torch.argmax(out, 1) == by).sum().item()
            ep_total   += len(by)

        ft_scheduler.step()
        tr_loss = ep_loss / max(len(train_loader), 1)
        tr_acc  = ep_correct / max(ep_total, 1)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, device, criterion)
        ep_time = time.time() - t0
        epoch_times.append(ep_time)

        print(f"  [FineTune] Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f}, Acc: {va_acc:.4f}, F1: {va_f1:.4f} | "
              f"{ep_time:.1f}s")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if args.patience > 0 and patience_ctr >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # ── Test ──
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    te_loss, te_acc, te_f1, te_preds, te_labels = evaluate(model, test_loader, device, criterion)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — CiMamba (SEED→Emognition Transfer)")
    print(f"{'='*60}")
    print(f"  Phase A Val F1   : {va_f1_a:.4f}")
    print(f"  Best Finetune F1 : {best_val_f1:.4f}")
    print(f"  Test Acc         : {te_acc:.4f}")
    print(f"  Test Macro-F1    : {te_f1:.4f}")
    print(f"  Avg epoch (B)    : {np.mean(epoch_times):.1f}s")
    print_report(te_labels, te_preds, class_names, title="Emognition Test")

    # Save
    ckpt_dir  = os.path.join(os.path.dirname(__file__), 'checkpoints', 'ci_emognition')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'finetuned_model.pt')
    torch.save({
        'model':      model.state_dict(),
        'test_acc':   te_acc,
        'test_f1':    te_f1,
        'class_names': class_names,
    }, ckpt_path)
    print(f"\n  Checkpoint saved: {ckpt_path}\n")


if __name__ == '__main__':
    main()
