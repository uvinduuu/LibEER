"""
train_mb_invbase_bimamba.py
===========================
Multi-Band InvBase BiMamba — Subject-Independent Training for Emognition.

Full pipeline:
  1. Load variable-length raw EEG trials (emognition_processed_loader)
  2. Load per-subject baseline spectra  (invbase.load_baselines_processed)
  3. Per-trial pre-processing
       a. Clip artefacts: ±5 σ per channel
       b. InvBase normalization (time-domain, phase-preserving)
       c. 5-band Butterworth bandpass filter → stack → (20, T)
  4. Subject-independent 70/15/15 split (disjoint subjects in each set)
  5. Window trials into fixed-size chunks
       • Train: 50 % overlap (more samples, heavier augmentation)
       • Val / Test: no overlap (clean evaluation)
  6. Train MBInvBaseBiMamba with:
       • AdamW + warmup-cosine LR schedule
       • Label smoothing cross-entropy
       • Gradient clipping
  7. Evaluate on test set, print confusion matrix + per-class F1

Usage (Kaggle / local):
    python train_mb_invbase_bimamba.py \\
        --data_root /kaggle/input/emognition-processed/Emognition\\ Processed \\
        --epochs 150 --d_model 64 --n_layers 3 --dropout 0.5 --seed 42

    # Quick smoke-test (8 trials, 50 epochs):
    python train_mb_invbase_bimamba.py --data_root ... --overfit_test
"""

import os
import sys
import math
import time
import random
import argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ── local imports via sys.path ───────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EMOG_DIR   = os.path.dirname(_SCRIPT_DIR)           # emognition/
sys.path.insert(0, _SCRIPT_DIR)
sys.path.insert(0, _EMOG_DIR)

from emognition_processed_loader import load_emognition_processed
from invbase import (load_baselines_processed, apply_invbase_to_raw,
                     INVBASE_BAND_HZ, NUM_BANDS)
from mb_invbase_bimamba_model import MBInvBaseBiMamba, IN_CHANNELS

# scipy is used only during data preprocessing (not in the training loop)
from scipy.signal import butter, filtfilt


# ── constants ────────────────────────────────────────────────────────────────

FS           = 256           # Muse 2 sampling rate (Hz)
NUM_CLASSES  = 4
CLASS_NAMES  = ["ENTHUSIASM", "FEAR", "NEUTRAL", "SADNESS"]  # alphabetical


# ════════════════════════════════════════════════════════════════════════════
#  Signal Pre-processing Utilities
# ════════════════════════════════════════════════════════════════════════════

def clip_artefacts(trial: np.ndarray, n_sigma: float = 5.0) -> np.ndarray:
    """
    Clip per-channel artefacts to ±n_sigma × channel std.

    Applied BEFORE InvBase so that extreme spikes do not corrupt the
    spectral division.

    Args:
        trial:   (C, T) float  — raw EEG trial
        n_sigma: clip threshold in units of per-channel std (default: 5.0)
    Returns:
        (C, T) float32 — clipped trial
    """
    trial = trial.astype(np.float64, copy=True)
    for c in range(trial.shape[0]):
        σ = trial[c].std()
        if σ > 1e-8:
            trial[c] = np.clip(trial[c], -n_sigma * σ, n_sigma * σ)
    return trial.astype(np.float32)


def _butter_bandpass(lo: float, hi: float, fs: float, order: int = 4):
    """Design a zero-phase Butterworth bandpass filter."""
    nyq  = fs / 2.0
    low  = np.clip(lo / nyq, 1e-6, 1.0 - 1e-6)
    high = np.clip(hi / nyq, 1e-6, 1.0 - 1e-6)
    return butter(order, [low, high], btype="band")


def apply_band_stack(trial: np.ndarray, fs: float = FS,
                     order: int = 4) -> np.ndarray:
    """
    Bandpass-filter a trial into 5 frequency bands and stack as channels.

    For each of the 5 bands (delta, theta, alpha, beta, gamma) a zero-phase
    Butterworth filter is applied to the 4 EEG channels.  The 5 filtered
    copies are then stacked along the channel axis, giving a 20-channel
    representation where each group of 4 channels corresponds to one band.

    Order: [delta_ch0..3, theta_ch0..3, alpha_ch0..3, beta_ch0..3, gamma_ch0..3]

    Args:
        trial: (4, T) float — InvBase-normalised EEG trial
        fs:    sampling rate in Hz
        order: Butterworth filter order (default: 4)

    Returns:
        (20, T) float32 — stacked band-channel signal
    """
    C, T = trial.shape
    bands_out = []
    for (_, lo, hi) in INVBASE_BAND_HZ:
        b, a     = _butter_bandpass(lo, hi, fs, order)
        filtered = filtfilt(b, a, trial, axis=1)          # (C, T)
        bands_out.append(filtered.astype(np.float32))
    return np.concatenate(bands_out, axis=0)              # (C*5, T) = (20, T)


def process_trial(trial: np.ndarray, baseline_spectrum, fs: float = FS) -> np.ndarray:
    """
    Full pre-processing pipeline for one raw EEG trial.

    Steps:
      1. Clip artefacts (±5 σ per channel)
      2. InvBase normalization (time-domain, per subject)
      3. Band-filter into 5 bands → stack → (20, T)

    Args:
        trial:             (4, T) float — raw EEG trial
        baseline_spectrum: (4, freq_bins) or None — from load_baselines_processed
        fs:                sampling rate in Hz

    Returns:
        (20, T) float32 — pre-processed trial ready for windowing
    """
    # 1 — clip
    trial = clip_artefacts(trial, n_sigma=5.0)

    # 2 — InvBase (graceful fallback if no baseline for this subject)
    trial = apply_invbase_to_raw(trial, baseline_spectrum, fs=fs)

    # 3 — band filter + stack
    return apply_band_stack(trial, fs=fs)


# ════════════════════════════════════════════════════════════════════════════
#  Subject-Independent Split
# ════════════════════════════════════════════════════════════════════════════

def subject_split(subject_ids, seed: int = 42,
                  val_frac: float = 0.15, test_frac: float = 0.15):
    """
    Split subjects into train / val / test sets (disjoint by subject).

    All trials of a given subject land in exactly ONE set — no subject
    can appear in two splits.  This is the most stringent generalisation
    test: the model must work on people it has never seen.

    Args:
        subject_ids: sequence of subject ID strings (one per trial)
        seed:        random seed for reproducibility
        val_frac:    fraction of subjects for validation  (default: 0.15)
        test_frac:   fraction of subjects for test        (default: 0.15)

    Returns:
        (train_set, val_set, test_set) — each a Python set of subject ID strings
    """
    subjects = sorted(set(subject_ids))
    rng      = np.random.RandomState(seed)
    rng.shuffle(subjects)                          # in-place, seeded

    n        = len(subjects)
    n_test   = max(1, round(n * test_frac))
    n_val    = max(1, round(n * val_frac))

    test_subjects  = set(subjects[:n_test])
    val_subjects   = set(subjects[n_test:n_test + n_val])
    train_subjects = set(subjects[n_test + n_val:])

    # Sanity check — assert disjoint
    assert not (train_subjects & val_subjects), "Overlap between train and val!"
    assert not (train_subjects & test_subjects), "Overlap between train and test!"
    assert not (val_subjects   & test_subjects), "Overlap between val and test!"

    return train_subjects, val_subjects, test_subjects


# ════════════════════════════════════════════════════════════════════════════
#  Windowing
# ════════════════════════════════════════════════════════════════════════════

def window_trials(processed_trials, labels, subject_ids,
                  window_size: int, step: int):
    """
    Slice pre-processed trials into fixed-size windows.

    The last partial window is zero-padded on the right so that no data
    is discarded from short trials.

    Args:
        processed_trials: list of (20, T_i) arrays
        labels:           list of int labels
        subject_ids:      list of str subject IDs
        window_size:      number of time-steps per window
        step:             stride between consecutive windows
                          (use window_size for no overlap, window_size//2 for 50 %)

    Returns:
        windows:     list of (20, window_size) float32 arrays
        win_labels:  list of int labels
        win_subjs:   list of str subject IDs
    """
    windows, win_labels, win_subjs = [], [], []

    for trial, label, subj in zip(processed_trials, labels, subject_ids):
        C, T = trial.shape
        # At least one window per trial even if T < window_size
        starts = list(range(0, max(T - window_size + 1, 1), step))
        for s in starts:
            win = trial[:, s:s + window_size]
            if win.shape[1] < window_size:
                pad = window_size - win.shape[1]
                win = np.pad(win, ((0, 0), (0, pad)))
            windows.append(win.astype(np.float32))
            win_labels.append(label)
            win_subjs.append(subj)

    return windows, win_labels, win_subjs


# ════════════════════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ════════════════════════════════════════════════════════════════════════════

class EmognitionMBDataset(Dataset):
    """
    Dataset of fixed-size (20, window_size) EEG windows.

    Optional augmentations (training only):
      • Gaussian noise     — adds small zero-mean noise scaled to signal std
      • Amplitude scaling  — multiplies all channels by a random scalar
      • Band dropout       — zeros one entire band (4 channels) with prob p
      • Time masking       — zeros one random time segment
    """

    def __init__(self, windows, labels, augment: bool = False,
                 noise_ratio: float = 0.03,
                 scale_range: tuple = (0.85, 1.15),
                 band_drop_p: float = 0.15,
                 time_mask_p: float = 0.40,
                 time_mask_frac: float = 0.10):
        self.windows       = windows
        self.labels        = labels
        self.augment       = augment
        self.noise_ratio   = noise_ratio
        self.scale_range   = scale_range
        self.band_drop_p   = band_drop_p
        self.time_mask_p   = time_mask_p
        self.time_mask_frac = time_mask_frac

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x     = self.windows[idx].copy()   # (20, window_size)
        label = self.labels[idx]

        if self.augment:
            x = self._augment(x)

        return torch.from_numpy(x), label

    def _augment(self, x: np.ndarray) -> np.ndarray:
        """Apply random augmentations to (20, T) window."""
        # 1. Gaussian noise (always applied in training)
        σ = x.std()
        if σ > 1e-8:
            x = x + (np.random.randn(*x.shape).astype(np.float32)
                     * σ * self.noise_ratio)

        # 2. Amplitude scaling  (single scalar — preserves relative channel ratios)
        scale = np.random.uniform(*self.scale_range)
        x     = x * scale

        # 3. Band dropout — zero one full band (all 4 channels of that band)
        if np.random.random() < self.band_drop_p:
            band_idx = np.random.randint(0, NUM_BANDS)
            x[band_idx * 4:(band_idx + 1) * 4, :] = 0.0

        # 4. Time masking — zero a contiguous time segment
        if np.random.random() < self.time_mask_p:
            T        = x.shape[1]
            mask_len = max(1, int(T * self.time_mask_frac))
            start    = np.random.randint(0, max(T - mask_len, 1) + 1)
            x[:, start:start + mask_len] = 0.0

        return x


# ════════════════════════════════════════════════════════════════════════════
#  Training Utilities
# ════════════════════════════════════════════════════════════════════════════

class LabelSmoothingCE(nn.Module):
    """Cross-entropy loss with label smoothing."""

    def __init__(self, n_classes: int, smoothing: float = 0.1):
        super().__init__()
        self.smoothing = smoothing
        self.n_classes = n_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor):
        log_prob = torch.log_softmax(logits, dim=-1)
        with torch.no_grad():
            smooth   = torch.full_like(log_prob,
                                       self.smoothing / (self.n_classes - 1))
            smooth.scatter_(1, target.unsqueeze(1), 1.0 - self.smoothing)
        return -(smooth * log_prob).sum(dim=-1).mean()


class WarmupCosineScheduler:
    """Linear warmup followed by cosine annealing."""

    def __init__(self, optimizer, warmup_epochs: int,
                 total_epochs: int, min_lr: float = 1e-7):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        self.base_lrs      = [pg["lr"] for pg in optimizer.param_groups]
        self._epoch        = 0

    def step(self):
        self._epoch += 1
        e = self._epoch
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if e <= self.warmup_epochs:
                pg["lr"] = base_lr * e / max(self.warmup_epochs, 1)
            else:
                prog     = ((e - self.warmup_epochs) /
                            max(self.total_epochs - self.warmup_epochs, 1))
                pg["lr"] = (self.min_lr +
                            (base_lr - self.min_lr) * 0.5 *
                            (1.0 + math.cos(math.pi * prog)))

    def get_last_lr(self):
        return [pg["lr"] for pg in self.optimizer.param_groups]


def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ════════════════════════════════════════════════════════════════════════════
#  Evaluation
# ════════════════════════════════════════════════════════════════════════════

def evaluate(model, loader, device, criterion):
    """Evaluate model. Returns (loss, accuracy, macro-F1, preds, labels)."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for bx, by in loader:
            bx  = bx.to(device)
            by  = (by.long().to(device) if isinstance(by, torch.Tensor)
                   else torch.tensor(by, dtype=torch.long, device=device))
            out = model(bx)
            total_loss += criterion(out, by).item()
            all_preds.extend(torch.argmax(out, 1).cpu().numpy())
            all_labels.extend(by.cpu().numpy())
            n_batches  += 1

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1  = f1_score(all_labels, all_preds, average="macro", zero_division=0)
    return total_loss / max(n_batches, 1), acc, f1, all_preds, all_labels


def print_report(y_true, y_pred, title: str = ""):
    """Print confusion matrix and per-class F1."""
    cm  = confusion_matrix(y_true, y_pred, labels=list(range(NUM_CLASSES)))
    hdr = " " if not title else f" ({title})"
    print(f"\n  Confusion Matrix{hdr}:")
    print(f"  {'':>12}", end="")
    for n in CLASS_NAMES:
        print(f"{n:>12}", end="")
    print()
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:>12}", end="")
        for j in range(NUM_CLASSES):
            print(f"{cm[i][j]:>12}", end="")
        print()
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred,
                                target_names=CLASS_NAMES, digits=4))


# ════════════════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="MB-InvBase-BiMamba — Emognition Subject-Independent Training"
    )

    # ── data ──
    parser.add_argument("--data_root", required=True,
                        help="Path to Emognition Processed dataset root")
    parser.add_argument("--emotions", nargs="+",
                        default=["ENTHUSIASM", "FEAR", "NEUTRAL", "SADNESS"],
                        help="Emotion classes to use (default: 4-class)")
    parser.add_argument("--min_trial_sec", type=float, default=5.0,
                        help="Minimum trial length in seconds (shorter skipped)")

    # ── windowing ──
    parser.add_argument("--window_sec", type=float, default=10.0,
                        help="Window length in seconds (default: 10)")
    parser.add_argument("--val_size", type=float, default=0.15,
                        help="Fraction of subjects for validation (default: 0.15)")
    parser.add_argument("--test_size", type=float, default=0.15,
                        help="Fraction of subjects for test (default: 0.15)")

    # ── model ──
    parser.add_argument("--d_model",  type=int,   default=64)
    parser.add_argument("--n_layers", type=int,   default=3)
    parser.add_argument("--d_state",  type=int,   default=16)
    parser.add_argument("--dropout",  type=float, default=0.5)
    parser.add_argument("--attn_reduction", type=int, default=4)

    # ── training ──
    parser.add_argument("--batch_size",    type=int,   default=16)
    parser.add_argument("--epochs",        type=int,   default=150)
    parser.add_argument("--lr",            type=float, default=2e-4)
    parser.add_argument("--weight_decay",  type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int,   default=5)
    parser.add_argument("--label_smooth",  type=float, default=0.1)
    parser.add_argument("--patience",      type=int,   default=30,
                        help="Early stopping patience (0 = disabled)")

    # ── misc ──
    parser.add_argument("--seed",    type=int,  default=42)
    parser.add_argument("--device",  type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overfit_test", action="store_true",
                        help="Quick sanity check: train on 8 trials → expect >90 %%")
    parser.add_argument("--save_dir", type=str, default=None,
                        help="Directory for model checkpoints (default: ./checkpoints)")

    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)

    window_size = int(args.window_sec * FS)
    save_dir    = args.save_dir or os.path.join(_SCRIPT_DIR, "checkpoints",
                                                "mb_invbase_bimamba")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  MB-InvBase-BiMamba  /  Emognition  /  Subject-Independent")
    print(f"{'='*70}")
    print(f"  data_root   : {args.data_root}")
    print(f"  emotions    : {args.emotions}")
    print(f"  window      : {args.window_sec}s  →  {window_size} samples"
          f"  →  {window_size // 16} Mamba steps")
    print(f"  model       : d_model={args.d_model}, n_layers={args.n_layers},"
          f" d_state={args.d_state}, dropout={args.dropout}")
    print(f"  training    : lr={args.lr}, wd={args.weight_decay},"
          f" warmup={args.warmup_epochs}, smooth={args.label_smooth}")
    print(f"  device      : {device}")
    print(f"{'='*70}\n")

    # ── 1. Load raw trials ───────────────────────────────────────────────────
    print("Step 1 — Loading trials...")
    t0 = time.time()
    trials, labels, subject_ids, lab2id, id2lab = load_emognition_processed(
        args.data_root, emotions=args.emotions,
        min_trial_sec=args.min_trial_sec, verbose=True
    )
    print(f"  Done in {time.time() - t0:.1f}s\n")

    if len(trials) == 0:
        print("ERROR: No trials loaded. Check --data_root and emotion labels.")
        return

    # ── 2. Load baselines ────────────────────────────────────────────────────
    print("Step 2 — Loading baseline spectra...")
    t0       = time.time()
    baselines = load_baselines_processed(args.data_root, fs=FS)
    n_covered = sum(1 for s in set(subject_ids) if s in baselines)
    n_total   = len(set(subject_ids))
    print(f"  InvBase coverage: {n_covered}/{n_total} subjects have a baseline\n"
          f"  ({n_total - n_covered} will use pass-through normalization)\n"
          f"  Done in {time.time() - t0:.1f}s\n")

    # ── 3. Pre-process trials ────────────────────────────────────────────────
    print("Step 3 — Pre-processing (clip → InvBase → band-stack)...")
    t0 = time.time()
    processed_trials = []
    for i, (trial, subj) in enumerate(zip(trials, subject_ids)):
        baseline_spectrum = baselines.get(subj, None)
        proc = process_trial(trial, baseline_spectrum, fs=FS)
        processed_trials.append(proc)
        if (i + 1) % 20 == 0 or (i + 1) == len(trials):
            print(f"  {i + 1}/{len(trials)} trials processed...", end="\r")
    print(f"\n  Done in {time.time() - t0:.1f}s\n"
          f"  Output shape per trial: (20, T_i) "
          f"[5 bands × 4 channels = {IN_CHANNELS} channels]")

    # Print trial length stats post-processing
    lengths = [t.shape[1] for t in processed_trials]
    print(f"  Trial lengths: min={min(lengths)/FS:.1f}s, "
          f"max={max(lengths)/FS:.1f}s, "
          f"mean={np.mean(lengths)/FS:.1f}s  (at {FS} Hz)\n")

    # ── Quick overfit test ───────────────────────────────────────────────────
    if args.overfit_test:
        print("=" * 50)
        print("OVERFIT TEST — 8 trials, 50 epochs, no early stop")
        print("=" * 50)
        sel = []
        for cls in range(NUM_CLASSES):
            sel.extend([i for i, l in enumerate(labels) if l == cls][:2])
        sel = sel[:8]
        sub_proc   = [processed_trials[i] for i in sel]
        sub_labels = [labels[i] for i in sel]
        sub_subjs  = [subject_ids[i] for i in sel]
        # tiny windows for speed
        wins, wlbls, _ = window_trials(sub_proc, sub_labels, sub_subjs,
                                       window_size, window_size)
        ds = EmognitionMBDataset(wins, wlbls, augment=False)
        loader = DataLoader(ds, batch_size=min(8, len(ds)), shuffle=True)
        model  = MBInvBaseBiMamba(IN_CHANNELS, NUM_CLASSES,
                                  args.d_model, args.n_layers,
                                  args.d_state, dropout=0.0).to(device)
        opt = optim.AdamW(model.parameters(), lr=1e-3)
        crit = nn.CrossEntropyLoss()
        for ep in range(50):
            model.train()
            for bx, by in loader:
                bx = bx.to(device)
                by = by.long().to(device)
                opt.zero_grad()
                loss = crit(model(bx), by)
                loss.backward()
                opt.step()
            if (ep + 1) % 10 == 0:
                _, acc, _, _, _ = evaluate(model, loader, device, crit)
                print(f"  Epoch {ep+1:3d} | Train acc: {acc:.4f}")
        _, acc, _, _, _ = evaluate(model, loader, device, crit)
        status = "✓ PASSED" if acc > 0.9 else f"✗ FAILED (acc={acc:.4f}, expected >0.9)"
        print(f"  Overfit test: {status}")
        return

    # ── 4. Subject-independent split ─────────────────────────────────────────
    print("Step 4 — Subject-independent split (70/15/15)...")
    train_subjs, val_subjs, test_subjs = subject_split(
        subject_ids, seed=args.seed,
        val_frac=args.val_size, test_frac=args.test_size
    )
    print(f"  Train subjects ({len(train_subjs)}): {sorted(train_subjs)}")
    print(f"  Val   subjects ({len(val_subjs)}): {sorted(val_subjs)}")
    print(f"  Test  subjects ({len(test_subjs)}): {sorted(test_subjs)}")
    print(f"  Overlap check: ✓ zero overlap\n")

    def gather(subj_set):
        idx = [i for i, s in enumerate(subject_ids) if s in subj_set]
        return ([processed_trials[i] for i in idx],
                [labels[i]           for i in idx],
                [subject_ids[i]      for i in idx])

    tr_proc, tr_lbl, tr_sub = gather(train_subjs)
    va_proc, va_lbl, va_sub = gather(val_subjs)
    te_proc, te_lbl, te_sub = gather(test_subjs)

    # ── 5. Window trials ─────────────────────────────────────────────────────
    print("Step 5 — Windowing trials...")
    step_train = window_size // 2    # 50 % overlap for train
    step_eval  = window_size         # no overlap for val / test

    tr_wins, tr_wlbls, _ = window_trials(tr_proc, tr_lbl, tr_sub,
                                          window_size, step_train)
    va_wins, va_wlbls, _ = window_trials(va_proc, va_lbl, va_sub,
                                          window_size, step_eval)
    te_wins, te_wlbls, _ = window_trials(te_proc, te_lbl, te_sub,
                                          window_size, step_eval)

    print(f"  Train windows : {len(tr_wins)}")
    print(f"  Val   windows : {len(va_wins)}")
    print(f"  Test  windows : {len(te_wins)}")

    # Class balance check
    for split_name, wlbls in [("Train", tr_wlbls), ("Val", va_wlbls),
                               ("Test",  te_wlbls)]:
        dist = Counter(wlbls)
        dist_str = ", ".join(f"{id2lab[k]}:{dist[k]}" for k in sorted(dist))
        print(f"  {split_name} class dist: {dist_str}")
    print()

    # ── 6. Datasets & Loaders ────────────────────────────────────────────────
    train_ds = EmognitionMBDataset(tr_wins, tr_wlbls, augment=True)
    val_ds   = EmognitionMBDataset(va_wins, va_wlbls, augment=False)
    test_ds  = EmognitionMBDataset(te_wins, te_wlbls, augment=False)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True, num_workers=0, pin_memory=True,
                              drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=args.batch_size,
                              shuffle=False, num_workers=0, pin_memory=True)

    # ── 7. Model ─────────────────────────────────────────────────────────────
    model = MBInvBaseBiMamba(
        in_channels    = IN_CHANNELS,
        num_classes    = NUM_CLASSES,
        d_model        = args.d_model,
        n_layers       = args.n_layers,
        d_state        = args.d_state,
        dropout        = args.dropout,
        attn_reduction = args.attn_reduction,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Step 6 — Model: MBInvBaseBiMamba")
    print(f"  Parameters   : {n_params:,}")
    print(f"  Input shape  : (B, {IN_CHANNELS}, {window_size})")
    print(f"  Mamba seq len: {window_size // 16}  (after conv stem ×16)\n")

    # ── 8. Optimiser & scheduler ──────────────────────────────────────────────
    criterion = LabelSmoothingCE(NUM_CLASSES, smoothing=args.label_smooth)
    eval_crit = nn.CrossEntropyLoss()   # unsmoothed, for true val/test loss
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, eps=1e-8)
    scheduler = WarmupCosineScheduler(optimizer,
                                      warmup_epochs=args.warmup_epochs,
                                      total_epochs=args.epochs,
                                      min_lr=1e-7)

    # ── 9. Training loop ──────────────────────────────────────────────────────
    best_val_f1   = 0.0
    best_state    = None
    patience_ctr  = 0
    epoch_times   = []

    print(f"Step 7 — Training ({args.epochs} epochs, "
          f"patience={args.patience if args.patience > 0 else 'off'})\n"
          f"{'='*70}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        t_ep = time.time()

        for bx, by in train_loader:
            bx  = bx.to(device)
            by  = (by.long().to(device) if isinstance(by, torch.Tensor)
                   else torch.tensor(by, dtype=torch.long, device=device))
            optimizer.zero_grad()
            out  = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            ep_loss    += loss.item()
            ep_correct += (torch.argmax(out, 1) == by).sum().item()
            ep_total   += len(by)

        scheduler.step()
        tr_acc  = ep_correct / max(ep_total, 1)
        tr_loss = ep_loss    / max(len(train_loader), 1)

        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, device,
                                                  eval_crit)
        ep_time = time.time() - t_ep
        epoch_times.append(ep_time)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:4d}/{args.epochs} | "
                  f"Tr loss {tr_loss:.4f}  acc {tr_acc:.4f} | "
                  f"Va loss {va_loss:.4f}  acc {va_acc:.4f}  F1 {va_f1:.4f} | "
                  f"{ep_time:.1f}s | lr {scheduler.get_last_lr()[0]:.2e}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if args.patience > 0 and patience_ctr >= args.patience:
            print(f"\n  Early stopping at epoch {epoch} "
                  f"(patience={args.patience})")
            break

    # ── 10. Test evaluation ───────────────────────────────────────────────────
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    te_loss, te_acc, te_f1, te_preds, te_labels = evaluate(
        model, test_loader, device, eval_crit
    )

    print(f"\n{'='*70}")
    print(f"  RESULTS  —  MB-InvBase-BiMamba  /  Emognition")
    print(f"{'='*70}")
    print(f"  Best Val F1  : {best_val_f1:.4f}")
    print(f"  Test Acc     : {te_acc:.4f}")
    print(f"  Test Macro-F1: {te_f1:.4f}")
    print(f"  Avg epoch    : {np.mean(epoch_times):.1f}s")
    print(f"  Total time   : {sum(epoch_times)/60:.1f} min")
    print_report(te_labels, te_preds, title="Emognition Test Set")

    # ── 11. Save checkpoint ───────────────────────────────────────────────────
    ckpt_path = os.path.join(save_dir, "best_model.pt")
    torch.save({
        "model_state": model.state_dict(),
        "model_cfg": {
            "in_channels":    IN_CHANNELS,
            "num_classes":    NUM_CLASSES,
            "d_model":        args.d_model,
            "n_layers":       args.n_layers,
            "d_state":        args.d_state,
            "dropout":        args.dropout,
            "attn_reduction": args.attn_reduction,
        },
        "id2lab":   id2lab,
        "lab2id":   lab2id,
        "val_f1":   best_val_f1,
        "test_acc": te_acc,
        "test_f1":  te_f1,
        "args":     vars(args),
    }, ckpt_path)
    print(f"\n  Checkpoint saved → {ckpt_path}\n")


if __name__ == "__main__":
    main()
