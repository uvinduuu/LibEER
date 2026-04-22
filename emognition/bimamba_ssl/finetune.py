#!/usr/bin/env python3
"""
finetune.py  —  Supervised Fine-Tuning with Pre-trained EEG Encoder
═══════════════════════════════════════════════════════════════════════════════

Loads a pre-trained MB-BiMamba encoder checkpoint (from pretrain.py) and
fine-tunes a 4-class multimodal emotion classifier (EEG + BVP) on the
Emognition dataset.

Key differences vs train_mb_invbase_bimamba.py:
  • --pretrained path/to/pretrained_encoder.pt  (from pretrain.py)
  • Two-LR optimizer: encoder at encoder_lr (low) + head at lr (normal)
  • --freeze_encoder: fully freeze encoder, train only BVP head (fastest)

Without --pretrained, this behaves identically to the original training script
(useful as a random-init ablation baseline to measure pre-training gain).

Usage (Kaggle):
    # Step 1 — pre-train first:
    python emognition/bimamba_ssl/pretrain.py \\
        --data_root /kaggle/input/.../emognition-processed \\
        --save_dir  /kaggle/working

    # Step 2 — fine-tune with pre-trained encoder:
    python emognition/bimamba_ssl/finetune.py \\
        --pretrained   /kaggle/working/pretrained_encoder.pt \\
        --data_root    /kaggle/input/.../emognition-processed \\
        --samsung_root /kaggle/input/.../samsung-data \\
        --d_model 32 --n_layers 2 --dropout 0.6 \\
        --epochs 100 --lr 8e-5 --encoder_lr 1e-5 \\
        --weight_decay 0.05 --label_smooth 0.20 --patience 30 --seed 42

    # Ablation — random init (no pre-training):
    python emognition/bimamba_ssl/finetune.py \\
        --data_root    /kaggle/input/.../emognition-processed \\
        --samsung_root /kaggle/input/.../samsung-data \\
        --d_model 32 --n_layers 2 --dropout 0.6 \\
        --epochs 100 --lr 8e-5 --patience 30 --seed 42
"""

import os, sys, argparse, time, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ── path setup ───────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EMOG_DIR   = os.path.dirname(_SCRIPT_DIR)               # emognition/
_MAMBA_DIR  = os.path.join(_EMOG_DIR, 'emognition_mamba')
sys.path.insert(0, _MAMBA_DIR)
sys.path.insert(0, _EMOG_DIR)

# Safe to import — train_mb_invbase_bimamba.py has `if __name__ == '__main__'` guard
from train_mb_invbase_bimamba import (
    clip_artefacts, apply_band_stack, process_trial,
    load_baselines_raw, euclidean_align_subjects,
    subject_split, window_trials,
    MultimodalMBModel, EmognitionMBDataset,
    LabelSmoothingCE, WarmupCosineScheduler,
    setup_seed, evaluate, print_report,
    build_bvp_lookup, BVP_DIM,
    FS, NUM_CLASSES, CLASS_NAMES, NUM_BANDS,
)
from mb_invbase_bimamba_model import MBInvBaseBiMamba, IN_CHANNELS
from invbase import load_baselines_processed
from emognition_processed_loader import load_emognition_processed
from sklearn.metrics import f1_score


# ══════════════════════════════════════════════════════════════════════════════
#  Training loop  (equivalent to original)
# ══════════════════════════════════════════════════════════════════════════════

def train_epoch(model, loader, optimizer, scheduler, criterion, device,
                use_bvp=False):
    model.train()
    total_loss, n_correct, n_total = 0.0, 0, 0
    for batch in loader:
        if use_bvp and len(batch) == 4:
            bx, bb, by, _ = batch
            bb = bb.to(device)
        else:
            bx, by, _ = batch[0], batch[-2], batch[-1]
            bb = None
        bx = bx.to(device)
        by = (by.long().to(device) if isinstance(by, torch.Tensor)
              else torch.tensor(by, dtype=torch.long, device=device))

        optimizer.zero_grad()
        out  = model(bx, bb) if use_bvp else model(bx)
        loss = criterion(out, by)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_correct  += (out.argmax(1) == by).sum().item()
        n_total    += by.size(0)
    scheduler.step()
    return total_loss / max(len(loader), 1), n_correct / max(n_total, 1)


# The 4 target emotions — must match train_mb_invbase_bimamba.py's TARGET_EMOT
_TARGET_EMOTIONS = ['ENTHUSIASM', 'FEAR', 'NEUTRAL', 'SADNESS']

def main():
    parser = argparse.ArgumentParser(
        description='MB-BiMamba Fine-Tuning with Pre-trained Encoder')

    # ── data ──────────────────────────────────────────────────────────────────
    parser.add_argument('--data_root',    required=True)
    parser.add_argument('--samsung_root', default=None,
                        help='Samsung Watch root (defaults to data_root)')
    parser.add_argument('--window_sec',   type=float, default=4.0)
    parser.add_argument('--val_size',     type=float, default=0.15)
    parser.add_argument('--test_size',    type=float, default=0.15)
    parser.add_argument('--no_bvp',       action='store_true')
    parser.add_argument('--norm_mode',    default='invbase',
                        choices=['invbase', 'zscore'])
    parser.add_argument('--use_ea',       action='store_true',
                        help='Euclidean Alignment (needs ≥10 trials/subj; '
                             'not recommended for Emognition with 4 trials/subj)')

    # ── pre-trained encoder ───────────────────────────────────────────────────
    parser.add_argument('--pretrained',     default=None,
                        help='Path to pretrained_encoder.pt (from pretrain.py). '
                             'Omit for random-init ablation baseline.')
    parser.add_argument('--freeze_encoder', action='store_true',
                        help='Freeze encoder completely — only head trains. '
                             'Fastest option; useful when data is very limited.')
    parser.add_argument('--encoder_lr',    type=float, default=1e-5,
                        help='LR for pre-trained encoder (keep < head LR). '
                             'Ignored when --freeze_encoder is set.')

    # ── model ─────────────────────────────────────────────────────────────────
    parser.add_argument('--d_model',        type=int,   default=32,
                        help='MUST match the value used in pretrain.py')
    parser.add_argument('--n_layers',       type=int,   default=2)
    parser.add_argument('--d_state',        type=int,   default=16)
    parser.add_argument('--dropout',        type=float, default=0.60)
    parser.add_argument('--attn_reduction', type=int,   default=4)

    # ── training ──────────────────────────────────────────────────────────────
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--epochs',        type=int,   default=100)
    parser.add_argument('--lr',            type=float, default=8e-5,
                        help='Head / BVP LR; encoder uses --encoder_lr')
    parser.add_argument('--weight_decay',  type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int,   default=5)
    parser.add_argument('--label_smooth',  type=float, default=0.20)
    parser.add_argument('--patience',      type=int,   default=30)

    # ── misc ──────────────────────────────────────────────────────────────────
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    setup_seed(args.seed)
    device        = torch.device(args.device)
    window_size   = int(args.window_sec * FS)
    step_tr       = window_size // 2    # 50% overlap for training windows
    step_ev       = window_size         # non-overlapping for eval/test
    samsung_root  = args.samsung_root or args.data_root
    use_bvp       = not args.no_bvp

    print()
    print('=' * 70)
    print('  MB-BiMamba  —  Supervised Fine-Tuning  /  SUB_INDEP')
    print('=' * 70)
    print(f'  data_root   : {args.data_root}')
    print(f'  pretrained  : {args.pretrained or "None (random init — ablation)"}')
    if args.pretrained:
        print(f'  freeze_enc  : {args.freeze_encoder}')
        enc_lr_str = "frozen" if args.freeze_encoder else str(args.encoder_lr)
        print(f'  LR (enc/head): {enc_lr_str} / {args.lr}')
    print(f'  BVP fusion  : {"ON" if use_bvp else "OFF (EEG-only)"}')
    print(f'  window      : {args.window_sec}s → {window_size} samples')
    print(f'  model       : d_model={args.d_model}, n_layers={args.n_layers}, '
          f'dropout={args.dropout}')
    print(f'  training    : epochs={args.epochs}, lr={args.lr}, '
          f'wd={args.weight_decay}, smooth={args.label_smooth}, '
          f'patience={args.patience}')
    print(f'  device      : {args.device}')
    print('=' * 70)

    # ── Step 1: Load trials ──────────────────────────────────────────────────
    print('\nStep 1 — Loading trials...')
    t0 = time.time()
    trials, labels, subject_ids, lab2id, id2lab = load_emognition_processed(
        args.data_root,
        emotions=_TARGET_EMOTIONS,   # filter to 4 target emotions only
        verbose=True)
    emot_strs = [id2lab[l] for l in labels]   # emotion string per trial
    print(f'  Done in {time.time()-t0:.1f}s  ({len(trials)} trials)\n')

    # ── Step 2: Baselines ────────────────────────────────────────────────────
    print('Step 2 — Loading baselines...')
    t0 = time.time()
    if args.norm_mode == 'zscore':
        baseline_info = load_baselines_raw(args.data_root)
    else:
        baseline_info = load_baselines_processed(args.data_root, fs=FS)
    n_cov = sum(1 for s in set(subject_ids) if s in baseline_info)
    print(f'  Coverage: {n_cov}/{len(set(subject_ids))} subjects  '
          f'(norm_mode={args.norm_mode})')
    print(f'  Done in {time.time()-t0:.1f}s\n')

    # ── Step 2b: Optional Euclidean Alignment ────────────────────────────────
    if args.use_ea:
        print('Step 2b — Euclidean Alignment (per-subject)...')
        t0 = time.time()
        trials = euclidean_align_subjects(trials, subject_ids)
        print(f'  Done in {time.time()-t0:.1f}s\n')

    # ── Step 3: Pre-process trials ───────────────────────────────────────────
    print(f'Step 3 — Pre-processing (clip → {args.norm_mode} → band-stack)...')
    t0 = time.time()
    processed_trials = []
    for i, (trial, subj) in enumerate(zip(trials, subject_ids)):
        binfo = baseline_info.get(subj, None)
        proc  = process_trial(trial, binfo, fs=FS, norm_mode=args.norm_mode)
        processed_trials.append(proc)
        if (i + 1) % 20 == 0 or (i + 1) == len(trials):
            print(f'  {i+1}/{len(trials)} processed...', end='\r')
    print(f'\n  Done in {time.time()-t0:.1f}s\n')

    # ── Step 4: BVP features + global z-score normalization ──────────────────
    bvp_lookup = None
    bvp_mean = bvp_std = None

    if use_bvp:
        print('Step 4 — Loading Samsung Watch BVP features...')
        bvp_lookup = build_bvp_lookup(samsung_root)
        # Global BVP normalization stats (computed over ALL available trials,
        # identical to train_mb_invbase_bimamba.py to ensure compatibility)
        vecs = [bvp_lookup.get((s, e)) for s, e in zip(subject_ids, emot_strs)
                if bvp_lookup.get((s, e)) is not None]
        if vecs:
            arr      = np.stack(vecs)
            bvp_mean = arr.mean(0).astype(np.float32)
            bvp_std  = (arr.std(0) + 1e-8).astype(np.float32)
        print(f'  BVP stats: mean (first 4)={bvp_mean[:4].round(2)}, '
              f'std={bvp_std[:4].round(2)}\n')

    def get_bvp_per_window(subj_list, emot_list, n_wins_per_trial):
        """
        Replicate the clip-level BVP feature vector for every window of
        that clip, then apply global z-score normalization.
        Mirrors train_mb_invbase_bimamba.py:get_bvp_per_window() exactly.
        """
        if not use_bvp or bvp_lookup is None:
            return None
        out = []
        for subj, emot, nw in zip(subj_list, emot_list, n_wins_per_trial):
            vec = bvp_lookup.get((subj, emot), np.zeros(BVP_DIM, np.float32))
            if vec is None:
                vec = np.zeros(BVP_DIM, np.float32)
            if bvp_mean is not None:
                vec = (vec - bvp_mean) / bvp_std
            out.extend([vec] * nw)
        return out

    def count_windows(proc_list, step):
        """Count how many windows each trial produces at the given step."""
        return [len(list(range(0, max(t.shape[1] - window_size + 1, 1), step)))
                for t in proc_list]

    # ── Step 5: Build model ──────────────────────────────────────────────────
    pretrain_tag = 'PretrainedEnc' if args.pretrained else 'RandomInit'
    bvp_tag      = 'EEG+BVP' if use_bvp else 'EEG-only'
    print(f'Step 5 — Model: MBInvBaseBiMamba [{pretrain_tag}] + {bvp_tag}')

    backbone = MBInvBaseBiMamba(
        in_channels    = IN_CHANNELS,
        num_classes    = NUM_CLASSES,
        d_model        = args.d_model,
        n_layers       = args.n_layers,
        d_state        = args.d_state,
        dropout        = args.dropout,
        attn_reduction = args.attn_reduction,
    )

    if args.pretrained:
        print(f'  Loading: {args.pretrained}')
        state = torch.load(args.pretrained, map_location='cpu')
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        if missing:
            print(f'  [warn] missing keys   : {missing}')
        if unexpected:
            print(f'  [warn] unexpected keys: {unexpected}')
        print(f'Pre-trained encoder loaded')
    else:
        print(f'Random init (ablation — no --pretrained given)')

    if use_bvp:
        model = MultimodalMBModel(backbone, bvp_dim=BVP_DIM,
                                  n_classes=NUM_CLASSES, dropout=0.5)
    else:
        model = backbone
    model = model.to(device)

    enc_p = sum(p.numel() for p in backbone.parameters())
    print(f'  EEG encoder params: {enc_p:,}\n')

    # ── Step 6: Subject-independent split (70/15/15) ─────────────────────────
    print('Step 6 — Subject-independent split (70/15/15)...')
    tr_subjs, va_subjs, te_subjs = subject_split(
        subject_ids, seed=args.seed,
        val_frac=args.val_size, test_frac=args.test_size)
    print(f'  Train:{len(tr_subjs)}  Val:{len(va_subjs)}  Test:{len(te_subjs)}\n')

    def gather(subj_set):
        idx = [i for i, s in enumerate(subject_ids) if s in subj_set]
        return ([processed_trials[i] for i in idx],
                [labels[i] for i in idx],
                [subject_ids[i] for i in idx],
                [emot_strs[i] for i in idx])

    tr_p, tr_l, tr_s, tr_e = gather(tr_subjs)
    va_p, va_l, va_s, va_e = gather(va_subjs)
    te_p, te_l, te_s, te_e = gather(te_subjs)

    # Window trials — training uses 50% overlap, eval/test uses no overlap
    tr_wins, tr_wl, _, tr_cids = window_trials(tr_p, tr_l, tr_s, window_size, step_tr)
    va_wins, va_wl, _, va_cids = window_trials(va_p, va_l, va_s, window_size, step_ev)
    te_wins, te_wl, _, te_cids = window_trials(te_p, te_l, te_s, window_size, step_ev)

    tr_bvp = get_bvp_per_window(tr_s, tr_e, count_windows(tr_p, step_tr))
    va_bvp = get_bvp_per_window(va_s, va_e, count_windows(va_p, step_ev))
    te_bvp = get_bvp_per_window(te_s, te_e, count_windows(te_p, step_ev))

    print(f'  Windows: tr={len(tr_wins)}, va={len(va_wins)}, te={len(te_wins)}')
    print(f'  Test clips: {len(set(te_cids))}\n')

    def make_loader(wins, wls, bvp, cids, augment=False):
        ds = EmognitionMBDataset(wins, wls, bvp, cids, augment=augment)
        return DataLoader(ds, batch_size=args.batch_size,
                          shuffle=augment, num_workers=0, pin_memory=True)

    tr_dl = make_loader(tr_wins, tr_wl, tr_bvp, tr_cids, augment=True)
    va_dl = make_loader(va_wins, va_wl, va_bvp, va_cids)
    te_dl = make_loader(te_wins, te_wl, te_bvp, te_cids)

    # ── Step 7: Optimizer (differential LR for pre-trained encoder) ───────────
    if args.freeze_encoder:
        for p in backbone.parameters():
            p.requires_grad_(False)
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = optim.AdamW(trainable, lr=args.lr,
                                weight_decay=args.weight_decay)
        print(f'  Optimizer: encoder FROZEN, head lr={args.lr}')
    elif args.pretrained:
        enc_ids    = set(id(p) for p in backbone.parameters())
        head_pars  = [p for p in model.parameters()
                      if id(p) not in enc_ids and p.requires_grad]
        enc_pars   = [p for p in backbone.parameters() if p.requires_grad]
        optimizer  = optim.AdamW(
            [{'params': enc_pars,  'lr': args.encoder_lr},
             {'params': head_pars, 'lr': args.lr}],
            weight_decay=args.weight_decay, eps=1e-8)
        print(f'  Optimizer: encoder lr={args.encoder_lr}, '
              f'head lr={args.lr}')
    else:
        optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay, eps=1e-8)
        print(f'  Optimizer: all params lr={args.lr}')
    print()

    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs,
                                      args.epochs, min_lr=1e-7)
    criterion = LabelSmoothingCE(NUM_CLASSES, args.label_smooth)
    eval_crit = nn.CrossEntropyLoss()

    # ── Step 8: Training loop ────────────────────────────────────────────────
    best_val_f1 = -1.0
    best_state  = None
    no_improve  = 0

    for ep in range(1, args.epochs + 1):
        tr_loss, tr_acc = train_epoch(
            model, tr_dl, optimizer, scheduler, criterion, device, use_bvp)
        _, va_wf, va_wf1, va_cf, va_cf1, *_ = evaluate(
            model, va_dl, device, eval_crit, use_bvp)
        lr = optimizer.param_groups[0]['lr']

        if ep % 10 == 0 or ep == 1:
            print(f'   Ep {ep:3d} | Tr:{tr_acc:.3f} | '
                  f'Va-win:{va_wf:.3f} Va-clip:{va_cf:.3f} F1:{va_cf1:.3f} | '
                  f'lr:{lr:.1e}')

        if va_cf1 > best_val_f1:
            best_val_f1 = va_cf1
            best_state  = {k: (v.clone() if torch.is_tensor(v) else v)
                           for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1
        if no_improve >= args.patience:
            print(f'  Early stop at epoch {ep}')
            break

    # ── Step 9: Test evaluation ───────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)

    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        tag  = f'ft_{"pre" if args.pretrained else "rand"}_{("bvp" if use_bvp else "eeg")}_seed{args.seed}'
        path = os.path.join(args.save_dir, f'{tag}.pt')
        torch.save({'model': model.state_dict(), 'args': vars(args)}, path)
        print(f'  Checkpoint saved → {path}')

    _, w_acc, w_f1, c_acc, c_f1, w_preds, w_true, c_preds, c_true = evaluate(
        model, te_dl, device, eval_crit, use_bvp)

    print()
    print('=' * 70)
    pre_label = (f'PT({os.path.basename(args.pretrained)})'
                 if args.pretrained else 'RandomInit')
    print(f'  RESULTS — {bvp_tag} / {pre_label} / seed={args.seed}')
    print('=' * 70)
    print(f'  Window Acc : {w_acc:.4f}  ({w_acc*100:.1f}%)')
    print(f'  Clip   Acc : {c_acc:.4f}  ({c_acc*100:.1f}%)  ← KEY METRIC')
    print(f'  Clip   F1  : {c_f1:.4f}')
    print(f'  Chance     : {100/NUM_CLASSES:.1f}%')
    print_report(c_true, c_preds, title=f'{pre_label} Fine-tuned Clip-Level')


if __name__ == '__main__':
    main()
