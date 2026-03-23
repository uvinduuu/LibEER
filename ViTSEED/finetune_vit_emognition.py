"""
Fine-tune ViT (pre-trained on SEED-IV) on Emognition.

Two-phase:
    Phase A — Head only (10 epochs): Freeze transformer, train classifier
    Phase B — Full model (N epochs): Unfreeze all, train with low LR

Since both SEED-IV and Emognition use 4-channel EEG at similar positions,
the patch embeddings and transformer attention patterns learned on SEED-IV
should transfer well.

Usage:
    python finetune_vit_emognition.py \
        --data_root /path/to/emognition \
        --pretrained /path/to/ViTSEED/checkpoints/best_vit_4ch.pt \
        --epochs 50

    Kaggle:
    python finetune_vit_emognition.py \
        --data_root /kaggle/input/datasets/uvindukodikara/emognition \
        --pretrained /kaggle/working/LibEER/ViTSEED/checkpoints/best_vit_4ch.pt \
        --epochs 50
"""

import os
import sys
import argparse
import random
import time
import math

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

sys.path.insert(0, os.path.dirname(__file__))
from vit_eeg_model import EEGViT

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


def main():
    parser = argparse.ArgumentParser(description="Fine-tune ViT on Emognition")

    parser.add_argument('--data_root',   required=True)
    parser.add_argument('--pretrained',  required=True)
    parser.add_argument('--mode', choices=['sub_dep', 'sub_indep'], default='sub_dep')

    # Fine-tuning
    parser.add_argument('--head_epochs',  type=int,   default=10)
    parser.add_argument('--epochs',       type=int,   default=50)
    parser.add_argument('--head_lr',      type=float, default=1e-3)
    parser.add_argument('--finetune_lr',  type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--patience',     type=int,   default=15)
    parser.add_argument('--warmup',       type=int,   default=5)
    parser.add_argument('--seed',         type=int,   default=2024)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)

    # ── Load checkpoint to get model config ──
    print(f"\nLoading pre-trained ViT from: {args.pretrained}")
    ckpt = torch.load(args.pretrained, map_location='cpu', weights_only=False)
    cfg  = ckpt['model_cfg']

    sample_rate  = FS   # Emognition sample rate (256Hz)
    window_size  = cfg['n_samples']   # match pre-trained window size

    # Emognition may have different sample rate — adjust window
    # SEED-IV is 200Hz, Emognition is 256Hz
    # To keep same temporal duration: window_sec = n_samples / 200
    window_sec   = cfg['n_samples'] / 200.0
    emog_window  = int(window_sec * sample_rate)
    # Ensure patch divisibility
    emog_window  = (emog_window // cfg['patch_size']) * cfg['patch_size']
    n_patches    = emog_window // cfg['patch_size']

    print(f"\n{'='*60}")
    print(f"ViT FINE-TUNING — SEED-IV → EMOGNITION")
    print(f"  Pretrained : {args.pretrained}")
    print(f"  SEED-IV cfg: {cfg['n_samples']} samples @ 200Hz = {window_sec}s")
    print(f"  Emognition : {emog_window} samples @ {sample_rate}Hz = {emog_window/sample_rate:.2f}s")
    print(f"  Patches    : {n_patches}")
    print(f"  Phase A    : {args.head_epochs} epochs (head only, lr={args.head_lr})")
    print(f"  Phase B    : {args.epochs} epochs (full, lr={args.finetune_lr})")
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
            window_size=emog_window,
            train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
            filter_eeg=True, normalize=True,
            sample_rate=sample_rate, seed=args.seed, augment_train=True,
        )
    else:
        unique_subj = sorted(set(subject_ids))
        rng = np.random.RandomState(args.seed)
        shuffled = list(unique_subj); rng.shuffle(shuffled)
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
                for w in split_trial_into_windows(t, emog_window):
                    wins.append(w); lbls.append(int(label))
            return WindowedEEGDataset(wins, lbls, augment=augment, sample_rate=sample_rate)

        train_ds = make_ds(train_subj, True)
        val_ds   = make_ds(val_subj,   False)
        test_ds  = make_ds(test_subj,  False)

    print(f"  Train: {len(train_ds)}, Val: {len(val_ds)}, Test: {len(test_ds)} windows")

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds,   args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  args.batch_size, shuffle=False,
                              num_workers=0, pin_memory=True)

    # ── Build model ──
    model = EEGViT(
        n_channels=cfg['n_channels'],
        n_samples=emog_window,        # Emognition window size
        patch_size=cfg['patch_size'],
        num_classes=n_classes,         # Emognition classes
        dim=cfg['dim'],
        depth=cfg['depth'],
        n_heads=cfg['n_heads'],
        mlp_dim=cfg['mlp_dim'],
        dropout=cfg['dropout'],
        emb_dropout=cfg['dropout'],
    )

    # Load pre-trained weights — skip head by name
    HEAD_KEYS = {'head.', 'norm.'}  # classifier head + final norm
    state_dict  = ckpt['model']
    model_state = model.state_dict()
    matched, skipped = [], []

    for k, v in state_dict.items():
        if any(k.startswith(hk) for hk in HEAD_KEYS):
            skipped.append(k)
        elif k in model_state and model_state[k].shape == v.shape:
            model_state[k] = v
            matched.append(k)
        elif k == 'pos_embed' and model_state[k].shape != v.shape:
            # Different window → different n_patches → interpolate pos_embed
            print(f"  Interpolating pos_embed: {v.shape} → {model_state[k].shape}")
            # CLS token stays, interpolate patch positions
            cls_tok = v[:, :1, :]        # (1, 1, dim)
            patch_pe = v[:, 1:, :]       # (1, N_old, dim)
            N_new = model_state[k].shape[1] - 1
            patch_pe = torch.nn.functional.interpolate(
                patch_pe.transpose(1, 2), size=N_new, mode='linear', align_corners=False
            ).transpose(1, 2)
            model_state[k] = torch.cat([cls_tok, patch_pe], dim=1)
            matched.append(k)
        else:
            skipped.append(k)

    model.load_state_dict(model_state)
    model = model.to(device)

    print(f"  Loaded {len(matched)} tensors | Skipped: {skipped}")

    criterion = nn.CrossEntropyLoss()

    # ═══════════════════════════════════════════════
    # Phase A: Head only
    # ═══════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"Phase A: Head-only ({args.head_epochs} epochs)")

    # Freeze everything except head
    for name, p in model.named_parameters():
        if not (name.startswith('head.') or name.startswith('norm.')):
            p.requires_grad = False

    head_opt = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.head_lr, weight_decay=args.weight_decay
    )
    for epoch in range(1, args.head_epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.long().to(device) if isinstance(by, torch.Tensor) else \
                 torch.tensor(by, dtype=torch.long, device=device)
            head_opt.zero_grad()
            out = model(bx); loss = criterion(out, by)
            loss.backward(); head_opt.step()
            ep_loss += loss.item()
            ep_correct += (torch.argmax(out, 1) == by).sum().item()
            ep_total   += len(by)
        print(f"  [Head] Epoch {epoch:2d}/{args.head_epochs} | "
              f"Loss: {ep_loss/len(train_loader):.4f}, "
              f"Acc: {ep_correct/ep_total:.4f}")

    _, va_acc_a, va_f1_a, _, _ = evaluate(model, val_loader, device, criterion)
    print(f"  Phase A Val → Acc: {va_acc_a:.4f}, F1: {va_f1_a:.4f}")

    # ═══════════════════════════════════════════════
    # Phase B: Full fine-tuning
    # ═══════════════════════════════════════════════
    print(f"\n{'─'*60}")
    print(f"Phase B: Full fine-tuning ({args.epochs} epochs, lr={args.finetune_lr})")

    for p in model.parameters():
        p.requires_grad = True

    ft_opt = optim.AdamW(model.parameters(), lr=args.finetune_lr,
                          weight_decay=args.weight_decay)

    def lr_fn(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / max(args.warmup, 1)
        progress = (epoch - args.warmup) / max(args.epochs - args.warmup, 1)
        return max(1e-7 / args.finetune_lr, 0.5 * (1 + math.cos(math.pi * progress)))

    ft_sch = optim.lr_scheduler.LambdaLR(ft_opt, lr_fn)

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
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
        if args.patience > 0 and patience_ctr >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}"); break

    # Test
    if best_state: model.load_state_dict(best_state)
    model = model.to(device)
    te_loss, te_acc, te_f1, te_preds, te_labels = evaluate(
        model, test_loader, device, criterion
    )

    sys.stdout.flush()

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — ViT (SEED→Emognition Transfer)")
    print(f"{'='*60}")
    print(f"  Phase A Val F1   : {va_f1_a:.4f}")
    print(f"  Best Finetune F1 : {best_val_f1:.4f}")
    print(f"  Test Acc         : {te_acc:.4f}")
    print(f"  Test Macro-F1    : {te_f1:.4f}")
    print(f"  Avg epoch (B)    : {np.mean(epoch_times):.1f}s")
    print_report(te_labels, te_preds, class_names, title="Emognition Test")

    # Save
    ckpt_dir  = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'finetuned_vit_emognition.pt')
    torch.save({
        'model':       model.state_dict(),
        'model_cfg':   {**cfg, 'n_samples': emog_window, 'num_classes': n_classes},
        'class_names': class_names,
        'test_acc':    te_acc,
        'test_f1':     te_f1,
        'val_f1':      best_val_f1,
    }, ckpt_path)
    print(f"\n  Checkpoint: {ckpt_path}\n")


if __name__ == '__main__':
    main()
