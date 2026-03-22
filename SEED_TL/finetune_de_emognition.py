"""
Fine-tune DE-based CiMamba (pre-trained on SEED-IV) on Emognition.

Key difference from raw CiMamba fine-tuning:
- Input: DE features (20, n_subwindows) instead of raw EEG (4, T)
- Much faster (20 Mamba steps vs 160+) and richer features
- Same two-phase strategy: head-only → full fine-tune

Usage:
    python finetune_de_emognition.py \
        --data_root /path/to/emognition \
        --pretrained /path/to/de_ci_seed/best_model.pt \
        --epochs 50

    Kaggle:
    python finetune_de_emognition.py \
        --data_root /kaggle/input/datasets/uvindukodikara/emognition \
        --pretrained /kaggle/working/LibEER/SEED_TL/checkpoints/de_ci_seed/best_model.pt \
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

# Local
sys.path.insert(0, os.path.dirname(__file__))
from ci_mamba_model import CiMambaClassifier
from de_features import compute_de_features, normalize_de, N_BANDS, BAND_NAMES
from train_de_seed import DEWindowDataset, build_de_datasets

# Emognition loader
emog_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                         '..', 'emognition', 'emognition_mamba')
sys.path.insert(0, emog_dir)
from emognition_loader import load_emognition_trials, FS

# Shared windowing utils
mamba_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mamba')
sys.path.insert(0, mamba_dir)
from windowed_dataset import split_trial_into_windows

N_VIRTUAL_CH = 4 * N_BANDS  # 20


def setup_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False


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
    print(f"  {'':>14}", end="")
    for name in class_names: print(f"{name:>14}", end="")
    print()
    for i, name in enumerate(class_names):
        print(f"  {name:>14}", end="")
        for j in range(n): print(f"{cm[i][j]:>14}", end="")
        print()
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


def build_emognition_de_datasets(trials, labels, subject_ids,
                                  window_size, subwin_sec, sample_rate,
                                  train_ratio, val_ratio, seed, mode):
    """Split trials/subjects and compute DE features for Emognition."""
    rng = np.random.RandomState(seed)
    n   = len(trials)

    if mode == 'sub_dep':
        idx = np.arange(n); rng.shuffle(idx)
        n_test = int(n * val_ratio); n_val = int(n * val_ratio)
        test_idx  = idx[:n_test]
        val_idx   = idx[n_test:n_test + n_val]
        train_idx = idx[n_test + n_val:]
        print(f"  Trial split: train={len(train_idx)}, "
              f"val={len(val_idx)}, test={len(test_idx)}")
    else:
        unique_subj = sorted(set(subject_ids)); rng.shuffle(unique_subj)
        n_test_s = max(1, int(len(unique_subj) * val_ratio))
        n_val_s  = max(1, int(len(unique_subj) * val_ratio))
        test_subj  = set(unique_subj[:n_test_s])
        val_subj   = set(unique_subj[n_test_s:n_test_s + n_val_s])
        train_subj = set(s for s in unique_subj if s not in test_subj | val_subj)
        test_idx   = [i for i, s in enumerate(subject_ids) if s in test_subj]
        val_idx    = [i for i, s in enumerate(subject_ids) if s in val_subj]
        train_idx  = [i for i, s in enumerate(subject_ids) if s in train_subj]
        print(f"  Subject split: train={len(train_subj)}, "
              f"val={len(val_subj)}, test={len(test_subj)} subjects")

    return build_de_datasets(trials, labels, subject_ids, window_size,
                              subwin_sec, sample_rate, train_idx, val_idx, test_idx)


def main():
    parser = argparse.ArgumentParser(description="Fine-tune DE-CiMamba on Emognition")

    parser.add_argument('--data_root',   required=True)
    parser.add_argument('--pretrained',  required=True)
    parser.add_argument('--mode', choices=['sub_dep', 'sub_indep'], default='sub_dep')
    parser.add_argument('--window_sec',  type=float, default=10.0)
    parser.add_argument('--subwin_sec',  type=float, default=0.5)

    parser.add_argument('--head_epochs',  type=int,   default=10)
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--head_lr',      type=float, default=1e-3)
    parser.add_argument('--finetune_lr',  type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--patience',     type=int,   default=15)
    parser.add_argument('--seed',         type=int,   default=2024)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    setup_seed(args.seed)
    device       = torch.device(args.device)
    sample_rate  = FS
    window_size  = int(args.window_sec * sample_rate)
    n_subwindows = int(args.window_sec / args.subwin_sec)

    print(f"\n{'='*60}")
    print(f"DE-CiMamba FINE-TUNING — SEED-IV → EMOGNITION")
    print(f"  Pretrained : {args.pretrained}")
    print(f"  Window     : {args.window_sec}s → {n_subwindows} sub-wins of {args.subwin_sec}s")
    print(f"  Input      : ({N_VIRTUAL_CH} virtual ch, {n_subwindows} steps)")
    print(f"  Phase A    : {args.head_epochs} epochs (head, lr={args.head_lr})")
    print(f"  Phase B    : {args.epochs} epochs (full, lr={args.finetune_lr})")
    print(f"{'='*60}")

    # Load Emognition
    print(f"\nLoading Emognition trials...")
    t0 = time.time()
    trials, labels, subject_ids, lab2id, id2lab = load_emognition_trials(args.data_root)
    print(f"  Loaded in {time.time()-t0:.1f}s")
    class_names = [id2lab[i] for i in range(len(id2lab))]
    n_classes   = len(class_names)

    # Build DE datasets
    train_ds, val_ds, test_ds = build_emognition_de_datasets(
        trials, labels, subject_ids,
        window_size, args.subwin_sec, sample_rate,
        train_ratio=0.7, val_ratio=0.15,
        seed=args.seed, mode=args.mode
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    # Load checkpoint
    print(f"\nLoading pre-trained DE-CiMamba from: {args.pretrained}")
    ckpt = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    cfg  = ckpt['model_cfg']

    model = CiMambaClassifier(
        n_channels=cfg['n_channels'],
        num_classes=n_classes,
        d_model=cfg['d_model'],
        n_layers=cfg['n_layers'],
        d_state=cfg['d_state'],
        dropout=cfg['dropout'],
        aggregation=cfg['aggregation'],
    )

    # Load encoder weights only — skip head by name
    HEAD_KEYS = {'head.0.weight', 'head.0.bias', 'head.2.weight', 'head.2.bias'}
    state_dict  = ckpt['model']
    model_state = model.state_dict()
    matched, skipped = [], []
    for k, v in state_dict.items():
        if any(k.startswith(hk) or k == hk for hk in HEAD_KEYS):
            skipped.append(k)
        elif k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v; matched.append(k)
        else:
            skipped.append(k)
    model.load_state_dict(model_state)
    model = model.to(device)

    print(f"  Loaded {len(matched)} tensors | Skipped (head/mismatch): {skipped}")

    criterion = nn.CrossEntropyLoss()

    # ── Phase A: Head only ──
    print(f"\n{'─'*60}")
    print(f"Phase A: Head-only ({args.head_epochs} epochs)")
    for p in model.encoder.parameters(): p.requires_grad = False
    if hasattr(model, 'ch_attn'):
        for p in model.ch_attn.parameters(): p.requires_grad = False

    head_opt = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                            lr=args.head_lr, weight_decay=args.weight_decay)
    for epoch in range(1, args.head_epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.long().to(device) if isinstance(by, torch.Tensor) else \
                 torch.tensor(by, dtype=torch.long, device=device)
            head_opt.zero_grad()
            out = model(bx); loss = criterion(out, by)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            head_opt.step()
            ep_loss += loss.item()
            ep_correct += (torch.argmax(out, 1) == by).sum().item()
            ep_total   += len(by)
        ep_loss /= max(len(train_loader), 1)
        ep_acc   = ep_correct / max(ep_total, 1)
        print(f"  [Head] Epoch {epoch:2d}/{args.head_epochs} | Loss: {ep_loss:.4f}, Acc: {ep_acc:.4f}")

    _, va_acc_a, va_f1_a, _, _ = evaluate(model, val_loader, device, criterion)
    print(f"  Phase A Val → Acc: {va_acc_a:.4f}, F1: {va_f1_a:.4f}")

    # ── Phase B: Full fine-tune ──
    print(f"\n{'─'*60}")
    print(f"Phase B: Full fine-tuning ({args.epochs} epochs, lr={args.finetune_lr})")
    for p in model.parameters(): p.requires_grad = True

    ft_opt  = optim.AdamW(model.parameters(), lr=args.finetune_lr,
                           weight_decay=args.weight_decay)
    ft_sch  = optim.lr_scheduler.CosineAnnealingLR(ft_opt, T_max=args.epochs, eta_min=1e-7)

    best_val_f1, best_state, patience_ctr = 0.0, None, 0
    epoch_times = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        t0 = time.time()
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.long().to(device) if isinstance(by, torch.Tensor) else \
                 torch.tensor(by, dtype=torch.long, device=device)
            ft_opt.zero_grad()
            out = model(bx); loss = criterion(out, by)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            ft_opt.step()
            ep_loss    += loss.item()
            ep_correct += (torch.argmax(out, 1) == by).sum().item()
            ep_total   += len(by)
        ft_sch.step()
        tr_loss = ep_loss / max(len(train_loader), 1)
        tr_acc  = ep_correct / max(ep_total, 1)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, device, criterion)
        ep_time = time.time() - t0
        epoch_times.append(ep_time)

        print(f"  [FT] Epoch {epoch:3d}/{args.epochs} | "
              f"Train: {tr_loss:.4f}/{tr_acc:.4f} | "
              f"Val: {va_loss:.4f}/{va_acc:.4f}/F1:{va_f1:.4f} | {ep_time:.1f}s")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if args.patience > 0 and patience_ctr >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}"); break

    # Test
    if best_state: model.load_state_dict(best_state)
    model = model.to(device)
    te_loss, te_acc, te_f1, te_preds, te_labels = evaluate(model, test_loader, device, criterion)

    import sys as _sys; _sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — DE-CiMamba (SEED→Emognition Transfer)")
    print(f"{'='*60}")
    print(f"  Phase A Val F1   : {va_f1_a:.4f}")
    print(f"  Best Finetune F1 : {best_val_f1:.4f}")
    print(f"  Test Acc         : {te_acc:.4f}")
    print(f"  Test Macro-F1    : {te_f1:.4f}")
    print(f"  Avg epoch (B)    : {np.mean(epoch_times):.1f}s")
    print_report(te_labels, te_preds, class_names, title="Emognition Test")

    # Save
    ckpt_dir  = os.path.join(os.path.dirname(__file__), 'checkpoints', 'de_ci_emognition')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'finetuned_model.pt')
    torch.save({
        'model':       model.state_dict(),
        'model_cfg':   cfg,
        'de_cfg':      ckpt.get('de_cfg', {}),
        'class_names': class_names,
        'test_acc':    te_acc,
        'test_f1':     te_f1,
        'val_f1':      best_val_f1,
    }, ckpt_path)
    print(f"\n  Checkpoint: {ckpt_path}\n")


if __name__ == '__main__':
    main()
