#!/usr/bin/env python3
"""
train_window_leakage.py
=======================
Window-leakage ablation experiment for MB-InvBase-BiMamba.

Many EEG papers accidentally inflate accuracy by windowing recordings first
and then splitting windows randomly — so windows from the SAME clip appear in
both train and test.  This script reproduces that scenario deliberately so you
can quantify the inflation vs the clean subject-independent split.

Two leakage modes
-----------------
sequential   First train_frac of EACH CLIP'S TIME → training windows (50 %
             overlap).  Last (1-train_frac) → test windows (non-overlapping).
             Leakage source: (a) filtfilt / InvBase preprocessing uses the
             full recording, (b) temporal proximity means train and test windows
             come from the same emotion episode.

random_pool  All clips windowed (50 % overlap), pooled, and RANDOMLY split by
             window count.  This is the classic paper mistake: neighboring
             overlapping windows may end up in different splits simultaneously.
             Gives the maximum possible inflation.

Tunable leakage
---------------
--train_frac  controls how much of each clip feeds training
              0.5 → moderate leakage   0.8 → high leakage   0.9 → very high

Compare the numbers here against the clean sub_indep baseline from
  train_mb_invbase_bimamba.py --mode sub_indep
to see how much the leakage inflates the reported accuracy.

IMPORTANT: model defaults are set for the leakage demo (low regularisation,
no augmentation).  Do NOT override with the clean-baseline settings or the
leakage inflation will be suppressed.

  Baseline (clean sub_indep)  : ~38% window / ~50% clip
  Sequential leakage 80 %     : ~47% window / ~55% clip
  Random-pool leakage 80 %    : ~63% window / ~71% clip

Usage
-----
# Sequential leakage, 80 % of each clip's TIME → train  (use defaults):
python emognition/emognition_mamba/train_window_leakage.py \\
    --data_root    /kaggle/input/datasets/sasinduabewickrema/emognition-processed \\
    --samsung_root /kaggle/input/datasets/uvindukodikara/emognition \\
    --train_frac 0.80 --leak_mode sequential \\
    --d_model 32 --n_layers 2 --seed 42

# Maximum leakage (random window split per clip):
python emognition/emognition_mamba/train_window_leakage.py \\
    --data_root    /kaggle/input/datasets/sasinduabewickrema/emognition-processed \\
    --samsung_root /kaggle/input/datasets/uvindukodikara/emognition \\
    --train_frac 0.80 --leak_mode random_pool \\
    --d_model 32 --n_layers 2 --seed 42

# Sweep leakage fractions (bash):
for frac in 0.5 0.7 0.8 0.9; do
    python emognition/emognition_mamba/train_window_leakage.py \\
        --data_root    /kaggle/input/... \\
        --samsung_root /kaggle/input/... \\
        --train_frac $frac --leak_mode sequential \\
        --d_model 32 --n_layers 2 --seed 42
done
"""

import os
import sys
import time
import argparse

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))   # emognition/emognition_mamba
_EMOG_DIR   = os.path.dirname(_SCRIPT_DIR)                 # emognition/
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _EMOG_DIR)

from mb_invbase_bimamba_model    import MBInvBaseBiMamba, IN_CHANNELS
from invbase                     import load_baselines_processed
from emognition_processed_loader import load_emognition_processed
from train_mb_invbase_bimamba    import (
    MultimodalMBModel, EmognitionMBDataset,
    WarmupCosineScheduler, LabelSmoothingCE,
    evaluate, build_bvp_lookup, process_trial,
    BVP_DIM, NUM_CLASSES, CLASS_NAMES, FS,
    setup_seed, print_report,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Window-leakage data split
# ══════════════════════════════════════════════════════════════════════════════

def make_leakage_split(
        processed_trials,
        labels,
        bvp_feats,
        window_size: int,
        step_train:  int,
        step_test:   int,
        train_frac:  float = 0.80,
        val_frac:    float = 0.15,
        leak_mode:   str   = 'sequential',
        seed:        int   = 42,
):
    """
    Build train / val / test window sets with intentional cross-clip leakage.

    Parameters
    ----------
    processed_trials : list of (20, T) float32 arrays
    labels           : list of int  (one per trial)
    bvp_feats        : list of (BVP_DIM,) float32 arrays (one per trial)
    window_size      : number of samples per window
    step_train       : window step for training side (use overlap, e.g. window//2)
    step_test        : window step for test side (non-overlapping = window_size)
    train_frac       : fraction of each clip assigned to the training pool
                         sequential  → fraction of TIME
                         random_pool → fraction of WINDOWS per clip
    val_frac         : fraction of the training pool held for validation
    leak_mode        : 'sequential' | 'random_pool'
    seed             : random seed for reproducibility

    Returns
    -------
    (tr_wins, tr_lbls, tr_bvps, tr_cids,
     va_wins, va_lbls, va_bvps, va_cids,
     te_wins, te_lbls, te_bvps, te_cids)
    Each element is a plain Python list.
    """
    rng = np.random.RandomState(seed)

    # Accumulators
    pool_wins, pool_lbls, pool_bvps, pool_cids = [], [], [], []
    te_wins,   te_lbls,   te_bvps,   te_cids   = [], [], [], []

    for cid, (trial, label, bvp) in enumerate(
            zip(processed_trials, labels, bvp_feats)):
        C, T = trial.shape

        if leak_mode == 'sequential':
            # ── temporal boundary: first train_frac of recording → training ──
            T_split = max(window_size, int(T * train_frac))

            # Training windows (with overlap)
            for s in range(0, max(T_split - window_size + 1, 1), step_train):
                w = trial[:, s:s + window_size]
                if w.shape[1] < window_size:
                    w = np.pad(w, ((0, 0), (0, window_size - w.shape[1])))
                pool_wins.append(w.astype(np.float32))
                pool_lbls.append(label)
                pool_bvps.append(bvp)
                pool_cids.append(cid)

            # Test windows (non-overlapping, from the tail of the clip)
            for s in range(T_split,
                           max(T - window_size + 1, T_split + 1),
                           step_test):
                w = trial[:, s:s + window_size]
                if w.shape[1] < window_size:
                    w = np.pad(w, ((0, 0), (0, window_size - w.shape[1])))
                te_wins.append(w.astype(np.float32))
                te_lbls.append(label)
                te_bvps.append(bvp)
                te_cids.append(cid)

        else:  # 'random_pool'
            # ── generate all windows from this clip, then split randomly ──
            clip_wins = []
            for s in range(0, max(T - window_size + 1, 1), step_train):
                w = trial[:, s:s + window_size]
                if w.shape[1] < window_size:
                    w = np.pad(w, ((0, 0), (0, window_size - w.shape[1])))
                clip_wins.append(w.astype(np.float32))

            n_clip  = len(clip_wins)
            perm_c  = rng.permutation(n_clip)
            n_tr_c  = max(1, int(n_clip * train_frac))
            tr_idx_c = perm_c[:n_tr_c].tolist()
            te_idx_c = perm_c[n_tr_c:].tolist()

            for i in tr_idx_c:
                pool_wins.append(clip_wins[i])
                pool_lbls.append(label)
                pool_bvps.append(bvp)
                pool_cids.append(cid)
            for i in te_idx_c:
                te_wins.append(clip_wins[i])
                te_lbls.append(label)
                te_bvps.append(bvp)
                te_cids.append(cid)

    # ── split training pool into train + val ──────────────────────────────────
    n_pool = len(pool_wins)
    perm   = rng.permutation(n_pool)
    n_val  = max(1, int(n_pool * val_frac))
    va_idx = perm[:n_val].tolist()
    tr_idx = perm[n_val:].tolist()

    def sel(lst, idx):
        return [lst[i] for i in idx]

    return (
        sel(pool_wins, tr_idx), sel(pool_lbls, tr_idx),
        sel(pool_bvps, tr_idx), sel(pool_cids, tr_idx),
        sel(pool_wins, va_idx), sel(pool_lbls, va_idx),
        sel(pool_bvps, va_idx), sel(pool_cids, va_idx),
        te_wins, te_lbls, te_bvps, te_cids,
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='MB-InvBase-BiMamba — Window-Leakage Experiment'
    )

    # ── data ──────────────────────────────────────────────────────────────────
    parser.add_argument('--data_root',     required=True,
                        help='Emognition processed dataset root (EEG JSON files)')
    parser.add_argument('--samsung_root',  default=None,
                        help='Samsung Watch data root (default: same as data_root)')
    parser.add_argument('--no_bvp',        action='store_true',
                        help='Disable BVP fusion — EEG-only ablation')
    parser.add_argument('--emotions',      nargs='+',
                        default=['ENTHUSIASM', 'FEAR', 'NEUTRAL', 'SADNESS'])
    parser.add_argument('--min_trial_sec', type=float, default=5.0)
    parser.add_argument('--norm_mode',     default='invbase',
                        choices=['invbase', 'zscore'],
                        help='Normalisation: invbase (default) or zscore')

    # ── leakage split ──────────────────────────────────────────────────────────
    parser.add_argument('--train_frac', type=float, default=0.80,
        help='Fraction of each clip assigned to training.  '
             'For sequential: fraction of TIME per clip (e.g. 0.80 → first 80%% → train).  '
             'For random_pool: fraction of WINDOWS per clip.  '
             'Range: 0.0–1.0.  Default: 0.80')
    parser.add_argument('--val_frac',   type=float, default=0.15,
        help='Fraction of training-pool windows held for validation.  Default: 0.15')
    parser.add_argument('--leak_mode',  default='sequential',
        choices=['sequential', 'random_pool'],
        help='sequential   : first train_frac of each clip\'s TIME → train  '
             '(RECOMMENDED — easier to interpret).  '
             'random_pool  : all windows pooled and randomly split  '
             '(maximum leakage — classic paper mistake).')

    # ── windowing ─────────────────────────────────────────────────────────────
    parser.add_argument('--window_sec', type=float, default=4.0,
        help='Window length in seconds.  Default: 4.0 (1024 samples at 256 Hz)')

    # ── model ─────────────────────────────────────────────────────────────────
    parser.add_argument('--d_model',        type=int,   default=32)
    parser.add_argument('--n_layers',       type=int,   default=2)
    parser.add_argument('--d_state',        type=int,   default=16)
    parser.add_argument('--dropout',        type=float, default=0.10,
        help='Dropout. LOW default (0.10) is intentional for the leakage demo: '
             'allows the model to overfit training clips so leakage inflation is '
             'visible. Use 0.60 only if you want to replicate the clean baseline.')
    parser.add_argument('--attn_reduction', type=int,   default=4)

    # ── training ──────────────────────────────────────────────────────────────
    parser.add_argument('--epochs',        type=int,   default=150,
        help='Training epochs. More epochs let the model overfit. Default: 150.')
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--lr',            type=float, default=3e-4,
        help='Learning rate. Higher than clean baseline (3e-4 vs 8e-5) to '
             'accelerate overfitting to training clips.')
    parser.add_argument('--weight_decay',  type=float, default=1e-3,
        help='Weight decay. LOW default (0.001) is intentional for leakage demo.')
    parser.add_argument('--warmup_epochs', type=int,   default=5)
    parser.add_argument('--label_smooth',  type=float, default=0.05,
        help='Label smoothing. LOW default (0.05) is intentional: model needs '
             'confident predictions to demonstrate leakage inflation.')
    parser.add_argument('--patience',      type=int,   default=50,
        help='Early stopping patience. Larger than clean baseline to allow more '
             'overfitting time.')
    parser.add_argument('--augment',       action='store_true', default=False,
        help='Enable data augmentation on training windows (default: OFF). '
             'Augmentation (Gaussian noise, band-dropout, time-masking) fights '
             'memorisation and SUPPRESSES leakage inflation — keep it OFF for the '
             'leakage demo. Pass --augment only for a regularised-leakage ablation.')

    # ── misc ──────────────────────────────────────────────────────────────────
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--save_dir', default=None,
                        help='Directory to save the trained checkpoint.  Optional.')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    setup_seed(args.seed)

    device       = torch.device(args.device)
    window_size  = int(args.window_sec * FS)   # e.g. 4.0 * 256 = 1024
    step_train   = window_size // 2            # 50 % overlap for training
    step_test    = window_size                 # non-overlapping for test
    samsung_root = args.samsung_root or args.data_root
    use_bvp      = not args.no_bvp

    # ── banner ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MB-InvBase-BiMamba  —  Window-Leakage Experiment")
    print(f"{'='*70}")
    print(f"  data_root  : {args.data_root}")
    print(f"  BVP fusion : {'ON' if use_bvp else 'OFF'}")
    print(f"  window     : {args.window_sec}s → {window_size} samples")
    print(f"  leak_mode  : {args.leak_mode}")
    if args.leak_mode == 'sequential':
        print(f"  train_frac : {args.train_frac:.0%}  "
              f"(first {args.train_frac:.0%} of each clip's TIME → train, "
              f"last {(1 - args.train_frac)*100:.0f}% → test)")
    else:
        print(f"  train_frac : {args.train_frac:.0%}  "
              f"({args.train_frac:.0%} of each clip's windows randomly → train)")
    print(f"  val_frac   : {args.val_frac:.0%} of training pool → validation")
    print(f"  model      : d_model={args.d_model}, n_layers={args.n_layers}, "
          f"dropout={args.dropout}")
    print(f"  training   : lr={args.lr}, wd={args.weight_decay}, "
          f"epochs={args.epochs}, patience={args.patience}")
    print(f"  augment    : {'ON' if args.augment else 'OFF (allows memorisation)'}")
    print(f"  device     : {device}")
    print(f"{'='*70}\n")
    if args.dropout > 0.30 or args.label_smooth > 0.10 or args.weight_decay > 0.01:
        print("  WARNING: High regularisation detected — leakage inflation will be "
              "suppressed.")
        print("  For the leakage demo use defaults: "
              "dropout=0.10, label_smooth=0.05, weight_decay=0.001\n")

    # ── 1. Load trials ─────────────────────────────────────────────────────────
    print("Step 1 — Loading trials...")
    trials, labels, subject_ids, lab2id, id2lab = load_emognition_processed(
        args.data_root, emotions=args.emotions,
        min_trial_sec=args.min_trial_sec, verbose=True)
    print(f"  {len(trials)} trials  |  "
          f"{len(set(subject_ids))} subjects  |  "
          f"{len(set(labels))} classes\n")
    if not trials:
        print("ERROR: No trials loaded.  Check --data_root.")
        return

    # ── 2. Load baselines ──────────────────────────────────────────────────────
    print("Step 2 — Loading baselines...")
    if args.norm_mode == 'zscore':
        from train_mb_invbase_bimamba import load_baselines_raw
        baseline_info = load_baselines_raw(args.data_root)
        print(f"  {len(baseline_info)} z-score baselines loaded\n")
    else:
        baseline_info = load_baselines_processed(args.data_root, fs=FS)
        print(f"  {len(baseline_info)} InvBase spectra loaded\n")

    # ── 3. Pre-process trials ──────────────────────────────────────────────────
    print(f"Step 3 — Pre-processing ({args.norm_mode})...")
    processed = []
    for i, (trial, sid) in enumerate(zip(trials, subject_ids)):
        proc = process_trial(trial, baseline_info.get(sid), fs=FS,
                             norm_mode=args.norm_mode)
        processed.append(proc)
        if (i + 1) % 20 == 0 or (i + 1) == len(trials):
            print(f"  {i+1}/{len(trials)} processed...", end="\r")
    print(f"  {len(processed)} trials pre-processed  "
          f"[shape per trial: (20, T_i)]\n")

    # ── 4. BVP features ────────────────────────────────────────────────────────
    emot_strs = [id2lab[l] for l in labels]
    bvp_list  = [np.zeros(BVP_DIM, np.float32)] * len(trials)
    if use_bvp:
        print("Step 4 — Loading Samsung Watch BVP features...")
        bvp_lookup = build_bvp_lookup(samsung_root)
        vecs = [bvp_lookup.get((s, e))
                for s, e in zip(subject_ids, emot_strs)
                if bvp_lookup.get((s, e)) is not None]
        bvp_mean = bvp_std = None
        if vecs:
            arr      = np.stack(vecs)
            bvp_mean = arr.mean(0).astype(np.float32)
            bvp_std  = (arr.std(0) + 1e-8).astype(np.float32)
        bvp_list = []
        n_found  = 0
        for sid, emot in zip(subject_ids, emot_strs):
            vec = bvp_lookup.get((sid, emot), np.zeros(BVP_DIM, np.float32))
            if bvp_lookup.get((sid, emot)) is not None:
                n_found += 1
            if bvp_mean is not None:
                vec = (vec - bvp_mean) / bvp_std
            bvp_list.append(vec)
        print(f"  BVP features: {n_found}/{len(trials)} trials\n")

    # ── 5. Window-leakage split ────────────────────────────────────────────────
    print(f"Step 5 — Window-leakage split "
          f"(mode={args.leak_mode}, train_frac={args.train_frac:.0%})...")
    (tr_wins, tr_lbls, tr_bvps, tr_cids,
     va_wins, va_lbls, va_bvps, va_cids,
     te_wins, te_lbls, te_bvps, te_cids) = make_leakage_split(
        processed, labels, bvp_list,
        window_size, step_train, step_test,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        leak_mode=args.leak_mode,
        seed=args.seed,
    )
    print(f"  Train : {len(tr_wins):5d} windows  from {len(set(tr_cids)):3d} clips")
    print(f"  Val   : {len(va_wins):5d} windows  from {len(set(va_cids)):3d} clips")
    print(f"  Test  : {len(te_wins):5d} windows  from {len(set(te_cids)):3d} clips\n")
    if not te_wins:
        print("ERROR: No test windows.  Reduce --train_frac or increase clip length.")
        return

    # ── 6. Build datasets & loaders ────────────────────────────────────────────
    # NOTE: augment=False by default — augmentation (noise, band-dropout,
    # time-masking) fights memorisation of clip patterns and suppresses the
    # leakage effect we want to demonstrate. Enable with --augment only for
    # a regularised-leakage ablation comparison.
    tr_ds = EmognitionMBDataset(
        tr_wins, tr_lbls, tr_bvps if use_bvp else None, tr_cids,
        augment=args.augment)
    va_ds = EmognitionMBDataset(
        va_wins, va_lbls, va_bvps if use_bvp else None, va_cids, augment=False)
    te_ds = EmognitionMBDataset(
        te_wins, te_lbls, te_bvps if use_bvp else None, te_cids, augment=False)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       drop_last=False, num_workers=0, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)
    te_dl = DataLoader(te_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)

    # ── 7. Build model ─────────────────────────────────────────────────────────
    backbone = MBInvBaseBiMamba(
        in_channels    = IN_CHANNELS,
        num_classes    = NUM_CLASSES,
        d_model        = args.d_model,
        n_layers       = args.n_layers,
        d_state        = args.d_state,
        dropout        = args.dropout,
        attn_reduction = args.attn_reduction,
    )
    model = (MultimodalMBModel(backbone, BVP_DIM, NUM_CLASSES, dropout=args.dropout)
             if use_bvp else backbone)
    model = model.to(device)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model: {n_params:,} trainable parameters")

    crit      = LabelSmoothingCE(NUM_CLASSES, args.label_smooth)
    eval_crit = nn.CrossEntropyLoss()
    opt       = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, eps=1e-8)
    sched     = WarmupCosineScheduler(opt, args.warmup_epochs, args.epochs,
                                      min_lr=1e-7)

    # ── 8. Training loop ───────────────────────────────────────────────────────
    print("\nStep 6 — Training...")
    best_f1, best_st, pat = 0.0, None, 0
    t0 = time.time()

    for ep in range(1, args.epochs + 1):
        model.train()
        tr_loss_sum, tr_n = 0.0, 0
        tr_ok, tr_tot = 0, 0      # track training accuracy to monitor overfitting
        for batch in tr_dl:
            if use_bvp and len(batch) == 4:
                bx, bb, by, _ = batch
                bb = bb.to(device)
            else:
                bx, by, _ = batch[0], batch[-2], batch[-1]
                bb = None
            bx = bx.to(device)
            by = by.long().to(device)
            opt.zero_grad()
            out  = model(bx, bb) if use_bvp else model(bx)
            loss = crit(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tr_loss_sum += loss.item()
            tr_n += 1
            tr_ok  += (out.argmax(1) == by).sum().item()
            tr_tot += len(by)
        sched.step()

        ret    = evaluate(model, va_dl, device, eval_crit, use_bvp)
        va_f1  = ret[4]
        lr_now = opt.param_groups[0]['lr']
        tr_acc = tr_ok / max(tr_tot, 1)

        if ep % 10 == 0 or ep == 1:
            print(f'   Ep{ep:4d} | Tr-acc:{tr_acc:.3f} loss:{tr_loss_sum/max(tr_n,1):.3f} | '
                  f'Va-win:{ret[1]:.3f} Va-clip:{ret[3]:.3f} F1:{va_f1:.3f} | '
                  f'lr:{lr_now:.1e}')

        if va_f1 > best_f1:
            best_f1 = va_f1
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat = 0
        else:
            pat += 1
            if args.patience > 0 and pat >= args.patience:
                print(f'   Early stop at epoch {ep}  (best val F1: {best_f1:.4f})')
                break

    elapsed = time.time() - t0
    if best_st:
        model.load_state_dict(best_st)
    print(f'\n  Training done in {elapsed:.0f}s  |  Best val F1: {best_f1:.4f}')

    # ── 9. Test evaluation ─────────────────────────────────────────────────────
    print("\nStep 7 — Test evaluation...")
    ret = evaluate(model, te_dl, device, eval_crit, use_bvp)
    win_acc    = ret[1]
    clip_acc   = ret[3]
    clip_f1    = ret[4]
    clip_preds = ret[7]
    clip_true  = ret[8]

    print(f"\n{'='*70}")
    print(f"  RESULTS — leak_mode={args.leak_mode}  "
          f"train_frac={args.train_frac:.0%}  seed={args.seed}")
    print(f"{'='*70}")
    print(f"  Window Acc : {win_acc*100:.1f}%")
    print(f"  Clip   Acc : {clip_acc*100:.1f}%  ← KEY METRIC")
    print(f"  Clip   F1  : {clip_f1:.4f}")
    print(f"  Chance     : {100/NUM_CLASSES:.1f}%")
    if args.leak_mode == 'sequential':
        print(f"\n  Leakage note: model trained on first {args.train_frac:.0%} of "
              f"EACH clip's recording time.")
        print(f"  Test is the last {(1-args.train_frac)*100:.0f}% of each clip.")
        print(f"  Compare with sub_indep baseline to see leakage inflation.")
    else:
        print(f"\n  ⚠  Maximum leakage: test windows randomly sampled from all clips.")
        print(f"  Adjacent overlapping windows may be in both train and test.")
    print(f"{'='*70}\n")

    if clip_preds and clip_true:
        print_report(clip_true, clip_preds,
                     title=f'{args.leak_mode} train_frac={args.train_frac:.0%}')

    # ── 10. Save checkpoint ────────────────────────────────────────────────────
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        name = (f'model_leak_{args.leak_mode}_'
                f'frac{int(args.train_frac*100)}_seed{args.seed}.pt')
        ckpt_path = os.path.join(args.save_dir, name)
        torch.save({
            'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
            'args':        vars(args),
            'class_names': CLASS_NAMES,
            'bvp_dim':     BVP_DIM if use_bvp else 0,
            'win_acc':     win_acc,
            'clip_acc':    clip_acc,
            'clip_f1':     clip_f1,
        }, ckpt_path)
        print(f"  [checkpoint] saved → {ckpt_path}")


if __name__ == '__main__':
    main()
