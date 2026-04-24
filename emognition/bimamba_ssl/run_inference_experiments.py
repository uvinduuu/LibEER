#!/usr/bin/env python3
"""
run_inference_experiments.py
════════════════════════════════════════════════════════════════════════════
Three-setup inference experiment for Anushka (Participant 1).

Setup 1 — Clean
    Train on original Emognition 70/15/15 split (no Anushka data).
    Test  on ALL 8 Anushka clips.
    → Baseline: how well does the model generalise to a completely new subject?

Setup 2 — Partial (1 clip/participant/emotion added to training)
    Train on original data  +  1 clip per emotion from Anushka
           (the lowest-indexed clip: enthusiasm2, fear4, neutral2, sad1).
    Test  on the remaining Anushka clips
           (enthusiasm5, fear5, neutral3, SAD4).
    → Realistic "personalisation": subject's resting data added at enrolment.

Setup 3 — Full (all inference clips in training)
    Train on original data  +  ALL inference clips (all windows included).
    Test  on ALL 8 Anushka clips (intentional data overlap).
    → Upper-bound: what accuracy is possible when the exact test recording
      appeared during training?

For each setup the script:
  • Trains the MB-InvBase-BiMamba + BVP multimodal model from scratch
  • Saves a checkpoint to --save_dir
  • Runs inference on the designated Anushka test clips
  • Prints window-level and clip-level accuracy + per-clip predictions
Finally prints a side-by-side comparison table for all three setups.

Usage (Kaggle):
    python emognition/bimamba_ssl/run_inference_experiments.py \\
        --data_root    /kaggle/input/datasets/sasinduabewickrema/emognition-processed \\
        --samsung_root /kaggle/input/datasets/uvindukodikara/emognition \\
        --inference_dir /kaggle/input/datasets/sasinduabewickrema/processed-participant/Output_KNN_ASR_InferenceCSV \\
        --save_dir     /kaggle/working \\
        --epochs 80 --d_model 32 --n_layers 2 --seed 42
"""

import os
import sys
import glob
import json
import math
import time
import random
import argparse
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EMOG_DIR   = os.path.dirname(_SCRIPT_DIR)
_MAMBA_DIR  = os.path.join(_EMOG_DIR, 'emognition_mamba')
sys.path.insert(0, _MAMBA_DIR)
sys.path.insert(0, _EMOG_DIR)
sys.path.insert(0, _SCRIPT_DIR)

from mb_invbase_bimamba_model  import MBInvBaseBiMamba, IN_CHANNELS
from invbase                   import (load_baselines_processed,
                                       apply_invbase_to_raw)
from emognition_processed_loader import load_emognition_processed
from anushka_loader            import (load_participant_clips,
                                       load_all_participants, split_clips,
                                       CLASS_NAMES, EMOT2ID, EEG_BAND_COLS)

# Re-use training utilities from the main training script
from train_mb_invbase_bimamba import (
    MultimodalMBModel, EmognitionMBDataset,
    WarmupCosineScheduler, LabelSmoothingCE,
    evaluate, window_trials, build_bvp_lookup,
    clip_artefacts, apply_band_stack,
    subject_split,
    BVP_DIM, NUM_CLASSES,
)

FS           = 256
WINDOW_SEC   = 4.0
WINDOW_SIZE  = int(WINDOW_SEC * FS)   # 1024


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers
# ══════════════════════════════════════════════════════════════════════════════

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


def process_orig_trial(trial: np.ndarray, baseline_spectrum,
                        fs: float = FS) -> np.ndarray:
    """clip → invbase → band-stack → (20, T)"""
    trial = clip_artefacts(trial)
    trial = apply_invbase_to_raw(trial, baseline_spectrum, fs=fs)
    proc  = apply_band_stack(trial, fs=fs)
    if not np.isfinite(proc).all():
        proc = np.nan_to_num(proc, nan=0.0, posinf=0.0, neginf=0.0)
    return proc


def window_clip(eeg_20ch: np.ndarray, label: int, bvp: np.ndarray,
                clip_id: int, window_size: int, step: int):
    """Slice a single (20, T) clip into overlapping windows."""
    T        = eeg_20ch.shape[1]
    windows  = []
    labels   = []
    bvps     = []
    clip_ids = []
    for s in range(0, max(T - window_size + 1, 1), step):
        w = eeg_20ch[:, s:s + window_size]
        if w.shape[1] < window_size:
            w = np.pad(w, ((0, 0), (0, window_size - w.shape[1])))
        windows.append(w.astype(np.float32))
        labels.append(label)
        bvps.append(bvp.astype(np.float32))
        clip_ids.append(clip_id)
    return windows, labels, bvps, clip_ids


# ══════════════════════════════════════════════════════════════════════════════
#  Training loop (compact, self-contained)
# ══════════════════════════════════════════════════════════════════════════════

def train_model(tr_wins, tr_lbls, tr_bvps, tr_cids,
                va_wins, va_lbls, va_bvps, va_cids,
                args, device, use_bvp: bool):
    """
    Build a fresh model, train, return (model, best_clip_f1).
    """
    tr_ds = EmognitionMBDataset(tr_wins, tr_lbls, tr_bvps, tr_cids, augment=True)
    va_ds = EmognitionMBDataset(va_wins, va_lbls, va_bvps, va_cids, augment=False)

    tr_dl = DataLoader(tr_ds, batch_size=args.batch_size, shuffle=True,
                       drop_last=False, num_workers=0, pin_memory=True)
    va_dl = DataLoader(va_ds, batch_size=args.batch_size, shuffle=False,
                       num_workers=0, pin_memory=True)

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

    crit      = LabelSmoothingCE(NUM_CLASSES, args.label_smooth)
    eval_crit = nn.CrossEntropyLoss()
    opt       = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, eps=1e-8)
    sched     = WarmupCosineScheduler(opt, args.warmup_epochs,
                                      args.epochs, min_lr=1e-7)

    best_f1, best_st, pat = 0.0, None, 0

    for ep in range(1, args.epochs + 1):
        model.train()
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
        sched.step()

        ret       = evaluate(model, va_dl, device, eval_crit, use_bvp)
        va_clip_f1 = ret[4]

        if ep % 10 == 0 or ep == 1:
            tr_acc = ret[1]  # window acc
            print(f'      Ep{ep:3d} | Va-win:{ret[1]:.3f} '
                  f'Va-clip:{ret[3]:.3f} F1:{va_clip_f1:.3f}')

        if va_clip_f1 > best_f1:
            best_f1 = va_clip_f1
            best_st = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            pat     = 0
        else:
            pat += 1
            if args.patience > 0 and pat >= args.patience:
                print(f'      Early stop at epoch {ep}')
                break

    if best_st:
        model.load_state_dict(best_st)
    return model.to(device), best_f1


# ══════════════════════════════════════════════════════════════════════════════
#  Inference on Anushka clips
# ══════════════════════════════════════════════════════════════════════════════

def infer_on_clips(model, clips, use_bvp: bool, device: torch.device,
                   batch_size: int = 64):
    """
    Run model on each clip in clips independently.
    Returns (win_acc, clip_acc, clip_f1, per_clip_results).
    per_clip_results: list of dicts with keys: emotion, clip_idx, pred, correct, probs
    """
    model.eval()
    all_win_preds, all_win_labels = [], []
    per_clip = []

    with torch.no_grad():
        for clip in clips:
            # Window non-overlapping for inference
            step    = WINDOW_SIZE
            T       = clip.eeg.shape[1]
            windows = []
            for s in range(0, max(T - WINDOW_SIZE + 1, 1), step):
                w = clip.eeg[:, s:s + WINDOW_SIZE]
                if w.shape[1] < WINDOW_SIZE:
                    w = np.pad(w, ((0, 0), (0, WINDOW_SIZE - w.shape[1])))
                windows.append(w.astype(np.float32))

            if not windows:
                continue

            x    = torch.from_numpy(np.stack(windows)).float()
            N    = x.shape[0]
            bvp  = (torch.from_numpy(clip.bvp).float().unsqueeze(0).expand(N, -1)
                    if use_bvp else None)

            all_logits = []
            for i in range(0, N, batch_size):
                xb  = x[i:i + batch_size].to(device)
                bb  = bvp[i:i + batch_size].to(device) if use_bvp else None
                out = model(xb, bb) if use_bvp else model(xb)
                all_logits.append(out.cpu())

            logits  = torch.cat(all_logits, dim=0)              # (N, 4)
            win_pred = logits.argmax(dim=1).numpy().tolist()

            # Window-level
            all_win_preds .extend(win_pred)
            all_win_labels.extend([clip.label] * N)

            # Clip-level: softmax average
            probs     = F.softmax(logits, dim=-1).mean(dim=0).numpy()
            clip_pred = int(probs.argmax())

            per_clip.append({
                'emotion':    clip.emotion,
                'clip_idx':   clip.clip_idx,
                'true_label': clip.label,
                'pred_label': clip_pred,
                'correct':    clip_pred == clip.label,
                'probs':      probs,
                'pid':        clip.pid,
                'folder':     clip.folder,
            })

    win_acc  = float(np.mean(
        np.array(all_win_preds) == np.array(all_win_labels))) if all_win_labels else 0.0
    clip_acc = float(np.mean([r['correct'] for r in per_clip])) if per_clip else 0.0
    clip_f1  = f1_score([r['true_label'] for r in per_clip],
                         [r['pred_label'] for r in per_clip],
                         average='macro', zero_division=0) if per_clip else 0.0
    return win_acc, clip_acc, clip_f1, per_clip


def infer_on_windowed_records(model, test_records: list, use_bvp: bool,
                              device: torch.device, batch_size: int = 64):
    """
    Inference for Setup 3: each test_record holds pre-sliced windows
    (the last 20% of a clip's windows).  Returns (win_acc, clip_acc, clip_f1, per_clip).
    """
    model.eval()
    all_win_preds, all_win_labels = [], []
    per_clip = []
    with torch.no_grad():
        for rec in test_records:
            wins = rec['wins']
            if not wins:
                continue
            x   = torch.from_numpy(np.stack(wins)).float()
            N   = x.shape[0]
            bvp = (torch.from_numpy(rec['bvp']).float().unsqueeze(0).expand(N, -1)
                   if use_bvp else None)
            all_logits = []
            for i in range(0, N, batch_size):
                xb  = x[i:i + batch_size].to(device)
                bb  = bvp[i:i + batch_size].to(device) if use_bvp else None
                out = model(xb, bb) if use_bvp else model(xb)
                all_logits.append(out.cpu())
            logits   = torch.cat(all_logits, dim=0)
            win_pred = logits.argmax(dim=1).numpy().tolist()
            all_win_preds.extend(win_pred)
            all_win_labels.extend([rec['lbl']] * N)
            probs     = F.softmax(logits, dim=-1).mean(dim=0).numpy()
            clip_pred = int(probs.argmax())
            per_clip.append({
                'emotion':    rec['emotion'],
                'clip_idx':   rec['clip_idx'],
                'true_label': rec['lbl'],
                'pred_label': clip_pred,
                'correct':    clip_pred == rec['lbl'],
                'probs':      probs,
                'pid':        rec['pid'],
                'folder':     rec['folder'],
            })
    win_acc  = float(np.mean(
        np.array(all_win_preds) == np.array(all_win_labels))) if all_win_labels else 0.0
    clip_acc = float(np.mean([r['correct'] for r in per_clip])) if per_clip else 0.0
    clip_f1  = f1_score([r['true_label'] for r in per_clip],
                         [r['pred_label'] for r in per_clip],
                         average='macro', zero_division=0) if per_clip else 0.0
    return win_acc, clip_acc, clip_f1, per_clip


def print_clip_results(per_clip: list, title: str):
    """Print a formatted per-clip prediction table, grouped by participant."""
    print(f'\n  {"─"*66}')
    print(f'  {title}')
    print(f'  {"─"*66}')
    # Group by folder/pid if available
    pids = sorted(set(r.get('pid', 0) for r in per_clip))
    for pid in pids:
        group = [r for r in per_clip if r.get('pid', 0) == pid]
        if not group:
            continue
        folder = group[0].get('folder', f'Participant {pid}')
        print(f'  [{folder}]')
        print(f'  {"Clip":<22} {"True":>12} {"Predicted":>12} {"✓/✗":>5}')
        print(f'  {"─"*54}')
        for r in group:
            key  = f"{r['emotion'].lower()}{r['clip_idx']}"
            true = CLASS_NAMES[r['true_label']]
            pred = CLASS_NAMES[r['pred_label']]
            mark = '✓' if r['correct'] else '✗'
            print(f'  {key:<22} {true:>12} {pred:>12} {mark:>5}')
        print()

    print(f'  {"─"*66}')

    # Confidence scores
    print(f'\n  Confidence scores:')
    for r in per_clip:
        key  = f"P{r.get('pid',0)}/{r['emotion'].lower()}{r['clip_idx']}"
        bars = '  '.join(f'{CLASS_NAMES[i]}:{r["probs"][i]*100:.0f}%'
                         for i in range(4))
        print(f'  {key:<26} {bars}')


# ══════════════════════════════════════════════════════════════════════════════
#  Load a checkpoint back into a model (for --load_checkpoints mode)
# ══════════════════════════════════════════════════════════════════════════════

def load_model_from_checkpoint(ckpt_path: str, device: torch.device):
    """
    Reconstruct a model from a checkpoint saved by this script.
    Returns (model, use_bvp).
    """
    ckpt     = torch.load(ckpt_path, map_location=device)
    saved_args = ckpt.get('args', {})
    use_bvp    = ckpt.get('use_bvp', True)
    d_model    = ckpt.get('d_model',  saved_args.get('d_model',  32))
    n_layers   = ckpt.get('n_layers', saved_args.get('n_layers', 2))
    d_state    = saved_args.get('d_state',        16)
    dropout    = saved_args.get('dropout',        0.55)
    attn_red   = saved_args.get('attn_reduction', 4)

    backbone = MBInvBaseBiMamba(
        in_channels    = IN_CHANNELS,
        num_classes    = NUM_CLASSES,
        d_model        = d_model,
        n_layers       = n_layers,
        d_state        = d_state,
        dropout        = dropout,
        attn_reduction = attn_red,
    )
    model = (MultimodalMBModel(backbone, BVP_DIM, NUM_CLASSES, dropout=dropout)
             if use_bvp else backbone)
    model.load_state_dict(ckpt['model_state'])
    model = model.to(device)
    model.eval()
    print(f'  [checkpoint] loaded ← {ckpt_path}  (use_bvp={use_bvp})')
    return model, use_bvp


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Three-setup inference experiment (all inference participants)')

    # ── data ──────────────────────────────────────────────────────────────────────────
    parser.add_argument('--data_root',    required=True,
                        help='Emognition Processed dataset root (EEG JSONs)')
    parser.add_argument('--samsung_root', default=None,
                        help='Samsung Watch data root (default: data_root)')
    parser.add_argument('--inference_dir', required=True,
                        help='Root of Output_KNN_ASR_InferenceCSV containing '
                             'Participant N sub-folders')
    parser.add_argument('--no_bvp',       action='store_true')

    # ── model ─────────────────────────────────────────────────────────────────
    parser.add_argument('--d_model',        type=int,   default=32)
    parser.add_argument('--n_layers',       type=int,   default=2)
    parser.add_argument('--d_state',        type=int,   default=16)
    parser.add_argument('--dropout',        type=float, default=0.55)
    parser.add_argument('--attn_reduction', type=int,   default=4)

    # ── training ──────────────────────────────────────────────────────────────
    parser.add_argument('--epochs',        type=int,   default=80)
    parser.add_argument('--batch_size',    type=int,   default=32)
    parser.add_argument('--lr',            type=float, default=1e-4)
    parser.add_argument('--weight_decay',  type=float, default=0.05)
    parser.add_argument('--warmup_epochs', type=int,   default=5)
    parser.add_argument('--label_smooth',  type=float, default=0.20)
    parser.add_argument('--patience',      type=int,   default=20)
    parser.add_argument('--val_frac',      type=float, default=0.15)
    parser.add_argument('--test_frac',     type=float, default=0.15)

    # ── misc ──────────────────────────────────────────────────────────────────
    parser.add_argument('--seed',     type=int, default=42)
    parser.add_argument('--save_dir', default='/kaggle/working')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    # ── inference-only mode ────────────────────────────────────────────────────────
    parser.add_argument(
        '--load_checkpoints', action='store_true',
        help='Skip training entirely and load the three checkpoint files from '
             '--save_dir instead.  Useful when training already finished but '
             'inference failed (e.g. Kaggle path issues).')
    parser.add_argument(
        '--ckpt_setup1', default=None,
        help='Override checkpoint path for Setup 1 (default: <save_dir>/model_setup1_clean.pt)')
    parser.add_argument(
        '--ckpt_setup2', default=None,
        help='Override checkpoint path for Setup 2 (default: <save_dir>/model_setup2_partial.pt)')
    parser.add_argument(
        '--ckpt_setup3', default=None,
        help='Override checkpoint path for Setup 3 (default: <save_dir>/model_setup3_full.pt)')
    args = parser.parse_args()

    setup_seed(args.seed)
    device       = torch.device(args.device)
    samsung_root = args.samsung_root or args.data_root
    use_bvp      = not args.no_bvp
    os.makedirs(args.save_dir, exist_ok=True)

    # Default checkpoint paths (can be overridden individually)
    default_ckpts = {
        'setup1': args.ckpt_setup1 or os.path.join(args.save_dir, 'model_setup1_clean.pt'),
        'setup2': args.ckpt_setup2 or os.path.join(args.save_dir, 'model_setup2_partial.pt'),
        'setup3': args.ckpt_setup3 or os.path.join(args.save_dir, 'model_setup3_full.pt'),
    }

    print()
    print('=' * 70)
    print('  MB-InvBase-BiMamba  —  Three-Setup Inference Experiment')
    print('=' * 70)
    print(f'  data_root    : {args.data_root}')
    print(f'  inference_dir: {args.inference_dir}')
    print(f'  BVP fusion   : {"ON" if use_bvp else "OFF"}')
    print(f'  window       : {WINDOW_SEC}s → {WINDOW_SIZE} samples')
    print(f'  model        : d_model={args.d_model}, n_layers={args.n_layers}')
    if args.load_checkpoints:
        print(f'  MODE         : INFERENCE ONLY (--load_checkpoints)')
        for k, p in default_ckpts.items():
            print(f'    {k}: {p}')
    else:
        print(f'  epochs       : {args.epochs}, patience={args.patience}')
    print(f'  device       : {device}')
    print('=' * 70)

    # ══════════════════════════════════════════════════════════════════════════
    #  A. Load and preprocess original Emognition dataset
    # ══════════════════════════════════════════════════════════════════════════
    print('\n[A] Loading original Emognition dataset...')
    trials_raw, labels, subject_ids, lab2id, id2lab = load_emognition_processed(
        args.data_root, emotions=['ENTHUSIASM', 'FEAR', 'NEUTRAL', 'SADNESS'],
        min_trial_sec=5.0, verbose=True)

    print('\n[B] Loading InvBase baselines...')
    baselines = load_baselines_processed(args.data_root, fs=FS)
    print(f'    {len(baselines)} subject baselines loaded')

    print('\n[C] Pre-processing (clip → invbase → band-stack)...')
    processed = []
    for i, (trial, sid) in enumerate(zip(trials_raw, subject_ids)):
        proc = process_orig_trial(trial, baselines.get(sid), fs=FS)
        processed.append(proc)
    print(f'    {len(processed)} trials processed')

    print('\n[D] Loading Samsung Watch BVP features...')
    bvp_lookup = build_bvp_lookup(samsung_root)

    # BVP features per trial
    emot_strs = [id2lab[l] for l in labels]
    def get_bvp(subjs, emots):
        return [bvp_lookup.get((s, e), np.zeros(BVP_DIM, np.float32))
                for s, e in zip(subjs, emots)]

    orig_bvp = get_bvp(subject_ids, emot_strs)

    # Subject split (deterministic)
    tr_subjs, va_subjs, _ = subject_split(
        subject_ids, seed=args.seed,
        val_frac=args.val_frac, test_frac=args.test_frac)
    print(f'\n    Subject split: train={len(tr_subjs)} val={len(va_subjs)} '
          f'test={len(set(subject_ids)-tr_subjs-va_subjs)} (test held out)')

    def get_split(subj_set):
        idx = [i for i, s in enumerate(subject_ids) if s in subj_set]
        return ([processed[i] for i in idx], [labels[i] for i in idx],
                [subject_ids[i] for i in idx], [emot_strs[i] for i in idx],
                [orig_bvp[i] for i in idx])

    tr_proc, tr_lbl, tr_sub, tr_emot, tr_bvp_list = get_split(tr_subjs)
    va_proc, va_lbl, va_sub, va_emot, va_bvp_list = get_split(va_subjs)

    # Window original data
    step_tr = WINDOW_SIZE // 2   # 50% overlap for training
    step_ev = WINDOW_SIZE        # non-overlapping for validation

    tr_wins, tr_wlbls, tr_wsubs, tr_cids = window_trials(
        tr_proc, tr_lbl, tr_sub, WINDOW_SIZE, step_tr)
    va_wins, va_wlbls, va_wsubs, va_cids = window_trials(
        va_proc, va_lbl, va_sub, WINDOW_SIZE, step_ev)

    # Per-window BVP (replicate trial-level BVP to each window)
    def wins_per_trial(procs, step):
        return [max(len(range(0, max(t.shape[1]-WINDOW_SIZE+1, 1), step)), 1)
                for t in procs]

    def expand_bvp(bvp_list, procs, step):
        out = []
        for feat, n in zip(bvp_list, wins_per_trial(procs, step)):
            out.extend([feat] * n)
        return out

    tr_wbvp = expand_bvp(tr_bvp_list, tr_proc, step_tr)
    va_wbvp = expand_bvp(va_bvp_list, va_proc, step_ev)

    print(f'\n    Windows: train={len(tr_wins)}, val={len(va_wins)}')

    # ══════════════════════════════════════════════════════════════════════════
    #  E. Load all inference participant clips
    # ══════════════════════════════════════════════════════════════════════════
    # Compute population-average baseline spectrum from training subjects.
    # This is passed to load_all_participants so inference clips go through the
    # EXACT same InvBase normalisation as the original training data:
    #   clip → clip_artefacts → InvBase(pop_baseline) → band_stack
    # Without this, inference clips have a different spectral distribution
    # (raw 1/f slope vs InvBase-flattened training data) → model can't generalise.
    print('\n[E] Computing population-average InvBase baseline...')
    valid_spectra = [v for v in baselines.values()
                     if v is not None and hasattr(v, 'shape') and v.ndim == 2]
    if valid_spectra:
        ref_len = valid_spectra[0].shape[1]
        aligned = []
        for sp in valid_spectra:
            if sp.shape[1] == ref_len:
                aligned.append(sp.astype(np.float64))
            else:
                # Interpolate to common frequency resolution
                old_f = np.linspace(0.0, FS / 2.0, sp.shape[1])
                new_f = np.linspace(0.0, FS / 2.0, ref_len)
                new_sp = np.zeros((sp.shape[0], ref_len), dtype=np.float64)
                for c in range(sp.shape[0]):
                    new_sp[c] = np.interp(new_f, old_f, sp[c])
                aligned.append(new_sp)
        pop_baseline = np.stack(aligned).mean(axis=0).astype(np.float32)  # (4, ref_len)
        print(f'    Population baseline: shape={pop_baseline.shape} '
              f'from {len(aligned)} subjects')
    else:
        pop_baseline = None
        print('    WARNING: No baselines found — inference clips will use '
              'self-whitening fallback (poor approximation)')

    print('\n[F] Loading inference participant clips...')
    all_clips              = load_all_participants(args.inference_dir,
                                                  baseline_spectrum=pop_baseline)
    lower_clips, upper_clips = split_clips(all_clips)

    # Summarise split
    pids = sorted(set(c.pid for c in all_clips))
    print(f'\n    Participants loaded: {pids}')
    print(f'    Lower clips ({len(lower_clips)}) → Setup 2 train: '
          + ', '.join(f'P{c.pid}/{c.emotion.lower()}{c.clip_idx}'
                      for c in lower_clips))
    print(f'    Upper clips ({len(upper_clips)}) → Setup 2 test : '
          + ', '.join(f'P{c.pid}/{c.emotion.lower()}{c.clip_idx}'
                      for c in upper_clips))
    print(f'    All clips   ({len(all_clips)}) → Setup 1 & 3  : '
          + ', '.join(f'P{c.pid}/{c.emotion.lower()}{c.clip_idx}'
                      for c in all_clips))

    def clips_to_windows(clips, step):
        """Convert ClipRecord list → windows, labels, bvps, clip_ids."""
        wins, lbls, bvps, cids = [], [], [], []
        base_cid = 10000   # offset to avoid collisions with training clip IDs
        for cid_offset, clip in enumerate(clips):
            w, l, b, c = window_clip(clip.eeg, clip.label, clip.bvp,
                                     base_cid + cid_offset, WINDOW_SIZE, step)
            wins.extend(w); lbls.extend(l)
            bvps.extend(b); cids.extend(c)
        return wins, lbls, bvps, cids

    def split_clip_windows_leakage(clips, train_frac: float = 0.8,
                                   step_train: int = None):
        """
        Setup 3 leakage split: for each clip, window with overlap and assign
        first train_frac fraction of windows to training, the rest to test.
        Returns (tr_wins, tr_lbls, tr_bvps, tr_cids, test_records).
        test_records is a list of dicts for infer_on_windowed_records().
        """
        if step_train is None:
            step_train = WINDOW_SIZE // 2
        tr_wins, tr_lbls, tr_bvps, tr_cids = [], [], [], []
        test_records = []
        base_cid = 20000   # different offset from clips_to_windows (10000)
        for cid_off, clip in enumerate(clips):
            T      = clip.eeg.shape[1]
            starts = list(range(0, max(T - WINDOW_SIZE + 1, 1), step_train))
            n_tr   = max(1, int(len(starts) * train_frac))
            cid    = base_cid + cid_off
            for s in starts[:n_tr]:
                w = clip.eeg[:, s:s + WINDOW_SIZE]
                if w.shape[1] < WINDOW_SIZE:
                    w = np.pad(w, ((0, 0), (0, WINDOW_SIZE - w.shape[1])))
                tr_wins.append(w.astype(np.float32))
                tr_lbls.append(clip.label)
                tr_bvps.append(clip.bvp.astype(np.float32))
                tr_cids.append(cid)
            test_ws = []
            for s in (starts[n_tr:] if len(starts) > n_tr else starts[-1:]):
                w = clip.eeg[:, s:s + WINDOW_SIZE]
                if w.shape[1] < WINDOW_SIZE:
                    w = np.pad(w, ((0, 0), (0, WINDOW_SIZE - w.shape[1])))
                test_ws.append(w.astype(np.float32))
            test_records.append({
                'wins':     test_ws,
                'lbl':      clip.label,
                'bvp':      clip.bvp.astype(np.float32),
                'pid':      clip.pid,
                'folder':   clip.folder,
                'emotion':  clip.emotion,
                'clip_idx': clip.clip_idx,
            })
        return tr_wins, tr_lbls, tr_bvps, tr_cids, test_records

    # ══════════════════════════════════════════════════════════════════════════
    #  Run all three setups
    # ══════════════════════════════════════════════════════════════════════════

    # Pre-compute Setup 3 leakage training windows (first 50% per clip, same overlap step)
    lk_tr_wins, lk_tr_lbls, lk_tr_bvps, lk_tr_cids, _lk_unused = \
        split_clip_windows_leakage(all_clips, train_frac=0.5, step_train=step_tr)
    print(f'\n    Leakage pre-split (Setup 3): '
          f'{len(lk_tr_wins)} train windows from first 50% of each clip '
          f'({len(all_clips)} clips); test = all {len(all_clips)} clips (full)')

    results_summary = {}   # setup_name → (win_acc, clip_acc, clip_f1)

    setups = [
        {
            'name':         'Setup 1 — Clean (no inference participants in train)',
            'key':          'setup1',
            'extra_clips':  [],                    # no inference data in training
            'test_clips':   all_clips,             # test on ALL inference clips
            'ckpt_file':    'model_setup1_clean.pt',
        },
        {
            'name':         'Setup 2 — Partial (1 clip/participant/emotion in train)',
            'key':          'setup2',
            'extra_clips':  lower_clips,           # add 1 clip/participant/emotion
            'test_clips':   upper_clips,           # test on remaining clips
            'ckpt_file':    'model_setup2_partial.pt',
        },
        {
            'name':       'Setup 3 — Leakage (first 50% windows in train, test all clips)',
            'key':        'setup3',
            'leakage':    True,       # first 50% of each clip's windows added to training
            'test_clips': all_clips,  # test on ALL clips (full evaluation)
            'ckpt_file':  'model_setup3_full.pt',
        },
    ]

    for setup in setups:
        print(f'\n{"="*70}')
        print(f'  {setup["name"]}')
        print(f'{"="*70}')
        t0 = time.time()

        ckpt_path = default_ckpts[setup['key']]

        if args.load_checkpoints:
            # ── Inference-only: load saved checkpoint, skip training ───────
            if not os.path.exists(ckpt_path):
                print(f'  [ERROR] Checkpoint not found: {ckpt_path}')
                print(f'  Skipping {setup["name"]}')
                results_summary[setup['key']] = {
                    'name': setup['name'], 'win_acc': 0, 'clip_acc': 0,
                    'clip_f1': 0, 'per_clip': [], 'elapsed': 0,
                }
                continue
            model, use_bvp = load_model_from_checkpoint(ckpt_path, device)
        else:
            # ── Normal: build training set, train, save checkpoint ────────
            if setup.get('leakage'):
                final_tr_wins = tr_wins + lk_tr_wins
                final_tr_lbls = tr_wlbls + lk_tr_lbls
                final_tr_bvps = tr_wbvp  + lk_tr_bvps
                final_tr_cids = tr_cids  + lk_tr_cids
                n_test_parts  = len(set(c.pid for c in setup['test_clips']))
                print(f'  Training: {len(tr_wins)} orig + '
                      f'{len(lk_tr_wins)} inference (first 50% windows/clip) '
                      f'= {len(final_tr_wins)} windows')
                print(f'  Val    : {len(va_wins)} windows')
                print(f'  Test   : {len(setup["test_clips"])} inference clips (all, '
                      f'{n_test_parts} participant(s)) ← intentional leakage')
            elif setup.get('extra_clips'):
                extra_wins, extra_lbls, extra_bvps, extra_cids = clips_to_windows(
                    setup['extra_clips'], step=step_tr)
                final_tr_wins = tr_wins + extra_wins
                final_tr_lbls = tr_wlbls + extra_lbls
                final_tr_bvps = tr_wbvp  + extra_bvps
                final_tr_cids = tr_cids  + extra_cids
                print(f'  Training: {len(tr_wins)} orig + '
                      f'{len(extra_wins)} inference = {len(final_tr_wins)} windows')
                print(f'  Val    : {len(va_wins)} windows')
                print(f'  Test   : {len(setup["test_clips"])} inference clips '
                      f'from {len(set(c.pid for c in setup["test_clips"]))} participant(s)')
            else:
                final_tr_wins = tr_wins
                final_tr_lbls = tr_wlbls
                final_tr_bvps = tr_wbvp
                final_tr_cids = tr_cids
                print(f'  Training: {len(tr_wins)} orig windows (no inference participants)')
                print(f'  Val    : {len(va_wins)} windows')
                print(f'  Test   : {len(setup["test_clips"])} inference clips '
                      f'from {len(set(c.pid for c in setup["test_clips"]))} participant(s)')
            print()

            model, best_val_f1 = train_model(
                final_tr_wins, final_tr_lbls, final_tr_bvps, final_tr_cids,
                va_wins,        va_wlbls,      va_wbvp,        va_cids,
                args, device, use_bvp)

            torch.save({
                'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
                'args':        vars(args),
                'class_names': CLASS_NAMES,
                'bvp_dim':     BVP_DIM if use_bvp else 0,
                'use_bvp':     use_bvp,
                'd_model':     args.d_model,
                'n_layers':    args.n_layers,
                'setup':       setup['key'],
            }, ckpt_path)
            print(f'\n  [checkpoint] saved → {ckpt_path}')

        # Inference on test clips (Setup 3 also uses infer_on_clips — full clip eval)
        win_acc, clip_acc, clip_f1, per_clip = infer_on_clips(
            model, setup['test_clips'], use_bvp, device)

        results_summary[setup['key']] = {
            'name':      setup['name'],
            'win_acc':   win_acc,
            'clip_acc':  clip_acc,
            'clip_f1':   clip_f1,
            'per_clip':  per_clip,
            'elapsed':   time.time() - t0,
        }

        # Print per-clip table
        print_clip_results(per_clip, title=setup['name'])
        print(f'\n  Window Acc : {win_acc*100:.1f}%')
        print(f'  Clip   Acc : {clip_acc*100:.1f}%  ← KEY METRIC')
        print(f'  Clip   F1  : {clip_f1:.4f}')
        print(f'  Time       : {time.time()-t0:.0f}s')

    # ══════════════════════════════════════════════════════════════════════════
    #  Final comparison table
    # ══════════════════════════════════════════════════════════════════════════
    print(f'\n\n{"="*70}')
    print('  FINAL COMPARISON — All Inference Participants')
    n_parts = len(set(c.pid for c in all_clips))
    print(f'  ({n_parts} participant(s), {len(all_clips)} total clips)')
    print(f'{"="*70}')
    print(f'  {"Setup":<45} {"Win%":>6} {"Clip%":>6} {"F1":>6}')
    print(f'  {"─"*65}')
    for key in ('setup1', 'setup2', 'setup3'):
        r = results_summary[key]
        print(f'  {r["name"]:<45} {r["win_acc"]*100:>5.1f}% '
              f'{r["clip_acc"]*100:>5.1f}% {r["clip_f1"]:>6.4f}')
    print(f'  {"─"*65}')
    print(f'  Chance: {100/NUM_CLASSES:.1f}%')
    print()

    # Per-setup confusion matrices
    for key in ('setup1', 'setup2', 'setup3'):
        r = results_summary[key]
        y_true = [x['true_label'] for x in r['per_clip']]
        y_pred = [x['pred_label'] for x in r['per_clip']]
        if not y_true:
            continue
        print(f'\n  {r["name"]}')
        print(f'  Confusion matrix (clip-level):')
        cm = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
        print(f'  {"":>14}', end='')
        for n in CLASS_NAMES:
            print(f'{n:>14}', end='')
        print()
        for i, n in enumerate(CLASS_NAMES):
            print(f'  {n:>14}', end='')
            for j in range(NUM_CLASSES):
                print(f'{cm[i][j]:>14}', end='')
            print()
        print()
        print(classification_report(y_true, y_pred,
              target_names=CLASS_NAMES, digits=4, zero_division=0))


if __name__ == '__main__':
    main()
