#!/usr/bin/env python3
"""
train_clip_leakage.py
=====================
Clip-leakage ablation experiment for MB-InvBase-BiMamba.

Leakage Hierarchy (three scripts, three levels)
------------------------------------------------
1. Clean sub_indep  (train_mb_invbase_bimamba.py)
   Split at the SUBJECT level.  A subject's clips only ever appear in ONE
   of train / val / test.  No subject-identity or temporal leakage.

2. Clip leakage  ← THIS SCRIPT
   Split at the WINDOW level, but windows are kept non-overlapping so no raw
   samples are shared.  For each clip (trial), its non-overlapping windows are
   RANDOMLY ASSIGNED to train / val / test pools.
   Leakage source: the same clip's temporal context appears in all three splits.
   The model implicitly sees the recording it will be tested on.

3. Window leakage  (train_window_leakage.py --overlap_frac 0.70)
   70 % overlapping windows are randomly split.  Adjacent windows share 70 %
   of their raw samples → near-duplicate data in train and test.

Why clip leakage inflates accuracy
-----------------------------------
The model is tested on windows from the same recording session it trained on.
  - Same subject → no subject-generalisation required.
  - Same emotional episode → the model can partly memorise the EEG pattern of
    that specific clip rather than learning general features.
  - BVP features are identical across all windows of a clip, reinforcing the
    memorisation pathway.

Expected results vs clean baseline (~38% window / ~50% clip):
  Clip leakage  ≈ 55-65% window / 65-75% clip

Usage
-----
python emognition/emognition_mamba/train_clip_leakage.py \\
    --data_root    /kaggle/input/datasets/sasinduabewickrema/emognition-processed \\
    --samsung_root /kaggle/input/datasets/uvindukodikara/emognition \\
    --train_frac 0.70 --d_model 32 --n_layers 2 --seed 42
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
    load_baselines_raw,
    BVP_DIM, NUM_CLASSES, CLASS_NAMES, FS,
    setup_seed, print_report,
)


# ══════════════════════════════════════════════════════════════════════════════
#  Clip-leakage data split
# ══════════════════════════════════════════════════════════════════════════════

def make_clip_leakage_split(
        processed_trials,
        labels,
        bvp_feats,
        window_size: int,
        train_frac:  float = 0.70,
        val_frac:    float = 0.15,
        seed:        int   = 42,
):
    """
    Clip-leakage split: each clip's NON-OVERLAPPING windows are randomly
    assigned to train / val / test pools.

    Leakage mechanism:
      - No subject-level separation: the same subject's clips feed all splits.
      - No window duplication: each window appears in exactly ONE split.
      - But the same clip's temporal context appears in MULTIPLE splits, so
        the model is trained and evaluated on different moments of the very
        same emotional recording session.

    Parameters
    ----------
    processed_trials : list of (20, T) float32 arrays
    labels           : list of int  (one per trial / clip)
    bvp_feats        : list of (BVP_DIM,) float32 arrays (one per trial)
    window_size      : samples per window (e.g. 1024 for 4 s at 256 Hz)
    train_frac       : fraction of each clip's windows → training pool
    val_frac         : fraction of each clip's windows → validation pool
    seed             : random seed

    Returns
    -------
    (tr_wins, tr_lbls, tr_bvps, tr_cids,
     va_wins, va_lbls, va_bvps, va_cids,
     te_wins, te_lbls, te_bvps, te_cids)
    """
    rng = np.random.RandomState(seed)

    tr_wins, tr_lbls, tr_bvps, tr_cids = [], [], [], []
    va_wins, va_lbls, va_bvps, va_cids = [], [], [], []
    te_wins, te_lbls, te_bvps, te_cids = [], [], [], []

    for cid, (trial, label, bvp) in enumerate(
            zip(processed_trials, labels, bvp_feats)):
        C, T = trial.shape

        # Extract NON-overlapping windows (step = window_size)
        # Non-overlapping ensures no sample-level leakage, only clip-level.
        windows = []
        for s in range(0, max(T - window_size + 1, 1), window_size):
            w = trial[:, s:s + window_size]
            if w.shape[1] < window_size:
                w = np.pad(w, ((0, 0), (0, window_size - w.shape[1])))
            windows.append(w.astype(np.float32))

        n = len(windows)
        if n == 0:
            continue

        perm = rng.permutation(n)
        n_tr = max(1, int(n * train_frac))
        n_va = max(0, int(n * val_frac))
        # Guarantee at least 1 window in test when n > 1
        if n_tr + n_va >= n and n > 1:
            n_tr = max(1, n - 2)
            n_va = 1

        tr_idx = perm[:n_tr].tolist()
        va_idx = perm[n_tr:n_tr + n_va].tolist()
        te_idx = perm[n_tr + n_va:].tolist()

        for i in tr_idx:
            tr_wins.append(windows[i]); tr_lbls.append(label)
            tr_bvps.append(bvp);        tr_cids.append(cid)
        for i in va_idx:
            va_wins.append(windows[i]); va_lbls.append(label)
            va_bvps.append(bvp);        va_cids.append(cid)
        for i in te_idx:
            te_wins.append(windows[i]); te_lbls.append(label)
            te_bvps.append(bvp);        te_cids.append(cid)

    return (tr_wins, tr_lbls, tr_bvps, tr_cids,
            va_wins, va_lbls, va_bvps, va_cids,
            te_wins, te_lbls, te_bvps, te_cids)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='MB-InvBase-BiMamba — Clip-Leakage Experiment'
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
    parser.add_argument('--norm_mode',     default='zscore',
                        choices=['invbase', 'zscore'])

    # ── clip leakage split ────────────────────────────────────────────────────
    parser.add_argument('--train_frac', type=float, default=0.70,
        help='Fraction of each clip\'s windows assigned to training. '
             'Default: 0.70 (70 %% train / 15 %% val / 15 %% test per clip).')
    parser.add_argument('--val_frac',   type=float, default=0.15,
        help='Fraction of each clip\'s windows assigned to validation. '
             'Default: 0.15')

    # ── windowing ─────────────────────────────────────────────────────────────
    parser.add_argument('--window_sec', type=float, default=4.0,
        help='Window length in seconds.  Default: 4.0 (1024 samples at 256 Hz)')

    # ── model ─────────────────────────────────────────────────────────────────
    parser.add_argument('--d_model',        type=int,   default=32)
    parser.add_argument('--n_layers',       type=int,   default=2)
    parser.add_argument('--d_state',        type=int,   default=16)
    parser.add_argument('--dropout',        type=float, default=0.10,
        help='Dropout. LOW default (0.10) is intentional for the leakage demo.')
    parser.add_argument('--attn_reduction', type=int,   default=4)

    # ── training ──────────────────────────────────────────────────────────────
    parser.add_argument('--epochs',        type=int,   default=150)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--weight_decay',  type=float, default=1e-3)
    parser.add_argument('--warmup_epochs', type=int,   default=5)
    parser.add_argument('--label_smooth',  type=float, default=0.05)
    parser.add_argument('--patience',      type=int,   default=50)
    parser.add_argument('--augment',       action='store_true', default=False,
        help='Enable augmentation (default: OFF — keeps leakage inflation visible).')

    # ── misc ──────────────────────────────────────────────────────────────────
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--save_dir', default=None)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    setup_seed(args.seed)

    device       = torch.device(args.device)
    window_size  = int(args.window_sec * FS)   # 4.0 * 256 = 1024
    samsung_root = args.samsung_root or args.data_root
    use_bvp      = not args.no_bvp

    # ── banner ────────────────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"  MB-InvBase-BiMamba  —  Clip-Leakage Experiment")
    print(f"{'='*70}")
    print(f"  data_root  : {args.data_root}")
    print(f"  BVP fusion : {'ON' if use_bvp else 'OFF'}")
    print(f"  window     : {args.window_sec}s = {window_size} samples (non-overlapping)")
    print(f"  split      : {args.train_frac:.0%} train / {args.val_frac:.0%} val /"
          f" {1-args.train_frac-args.val_frac:.0%} test  per clip")
    print(f"  leakage    : clip-level (same clip windows in all splits; "
          f"no sample duplication)")
    print(f"  model      : d_model={args.d_model}, n_layers={args.n_layers}, "
          f"dropout={args.dropout}")
    print(f"  training   : lr={args.lr}, wd={args.weight_decay}, "
          f"epochs={args.epochs}, patience={args.patience}")
    print(f"  device     : {device}")
    print(f"{'='*70}\n")

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
    print(f"  {len(processed)} trials pre-processed  [shape: (20, T_i)]\n")

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

    # ── 5. Clip-leakage split ──────────────────────────────────────────────────
    print(f"Step 5 — Clip-leakage split "
          f"(train_frac={args.train_frac:.0%}, val_frac={args.val_frac:.0%})...")
    (tr_wins, tr_lbls, tr_bvps, tr_cids,
     va_wins, va_lbls, va_bvps, va_cids,
     te_wins, te_lbls, te_bvps, te_cids) = make_clip_leakage_split(
        processed, labels, bvp_list,
        window_size,
        train_frac=args.train_frac,
        val_frac=args.val_frac,
        seed=args.seed,
    )
    print(f"  Train : {len(tr_wins):5d} windows  from {len(set(tr_cids)):3d} clips")
    print(f"  Val   : {len(va_wins):5d} windows  from {len(set(va_cids)):3d} clips")
    print(f"  Test  : {len(te_wins):5d} windows  from {len(set(te_cids)):3d} clips")
    n_shared = len(set(tr_cids) & set(te_cids))
    print(f"  Clips shared between train and test : {n_shared} "
          f"(leakage = {n_shared}/{len(set(te_cids))} test clips)\n")
    if not te_wins:
        print("ERROR: No test windows.  Reduce --train_frac or increase trial length.")
        return

    # ── 6. Build datasets & loaders ────────────────────────────────────────────
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
        tr_ok, tr_tot = 0, 0
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
    print(f"  RESULTS — Clip Leakage  "
          f"train_frac={args.train_frac:.0%}  seed={args.seed}")
    print(f"{'='*70}")
    print(f"  Window Acc : {win_acc*100:.1f}%")
    print(f"  Clip   Acc : {clip_acc*100:.1f}%  <- KEY METRIC")
    print(f"  Clip   F1  : {clip_f1:.4f}")
    print(f"  Chance     : {100/NUM_CLASSES:.1f}%")
    print(f"\n  Leakage note: same clip's windows appear in BOTH train and test.")
    print(f"  Each window appears in exactly ONE split (no sample duplication).")
    print(f"  Compare with sub_indep baseline to quantify clip-level leakage.")
    print(f"{'='*70}\n")

    if clip_preds and clip_true:
        print_report(clip_true, clip_preds,
                     title=f'clip_leakage train_frac={args.train_frac:.0%}')

    # ── 10. Save checkpoint ────────────────────────────────────────────────────
    if args.save_dir:
        os.makedirs(args.save_dir, exist_ok=True)
        ckpt_path = os.path.join(
            args.save_dir,
            f'clip_leakage_f{args.train_frac:.0f}_s{args.seed}.pt')
        torch.save({
            'model_state': model.state_dict(),
            'args':        vars(args),
            'win_acc':     win_acc,
            'clip_acc':    clip_acc,
            'clip_f1':     clip_f1,
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")


if __name__ == '__main__':
    main()
