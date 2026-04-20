"""
train_mb_invbase_bimamba.py
===========================
Multi-Band InvBase BiMamba — Multimodal Emognition Training.

Supports two evaluation modes:
  --mode loso       : Leave-One-Subject-Out (default, strictest)
  --mode sub_indep  : 70/15/15 subject-independent split

BVP Fusion (Samsung Watch):
  By default, 4 clip-level HRV features [HR_mean, RMSSD, pNN50, IBI_range]
  are loaded from Samsung Watch JSONs and concatenated to the EEG embedding
  before the classifier head. Disable with --no_bvp.

Anti-overfitting:
  Label smoothing (0.20), Dropout (0.55), Weight decay (0.05),
  Band dropout + time masking augmentation, Early stopping, LOSO.

Usage (Kaggle):
    # Multimodal LOSO (recommended):
    python train_mb_invbase_bimamba.py \\
        --data_root /kaggle/input/.../emognition \\
        --samsung_root /kaggle/input/.../emognition \\
        --mode loso --epochs 120

    # EEG-only ablation:
    python train_mb_invbase_bimamba.py \\
        --data_root /kaggle/input/.../emognition \\
        --mode loso --no_bvp --epochs 120
"""

import os
import sys
import glob
import json
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

# ════════════════════════════════════════════════════════════════════════════
#  BVP / Samsung Watch Feature Extraction
# ════════════════════════════════════════════════════════════════════════════

BVP_DIM      = 4                    # [HR_mean, RMSSD, pNN50, IBI_range]
TARGET_EMOT  = {'ENTHUSIASM', 'FEAR', 'NEUTRAL', 'SADNESS'}


def _parse_paired(raw):
    """[[timestamp, value], ...] → numpy 1-D array of values."""
    if not isinstance(raw, list) or len(raw) < 5:
        return None
    try:
        return np.array([r[1] for r in raw], dtype=np.float64)
    except Exception:
        return None


def load_bvp_features_one(fp):
    """
    Extract 4 HRV features from one Samsung Watch STIMULUS JSON.
    Returns float32 array [HR_mean, RMSSD, pNN50, IBI_range] or None.
    """
    try:
        with open(fp) as f:
            obj = json.load(f)
    except Exception:
        return None

    ibi = _parse_paired(obj.get('PPInterval'))
    hr  = _parse_paired(obj.get('heartRate'))

    if ibi is not None:
        ibi = ibi[(ibi > 300) & (ibi < 2000) & np.isfinite(ibi)]
    if hr is not None:
        hr  = hr[(hr > 30)   & (hr  < 220)  & np.isfinite(hr)]

    if ibi is None or len(ibi) < 5:
        return None

    hr_mean   = float(np.mean(hr))   if (hr is not None and len(hr) >= 3) \
                else float(np.mean(60000.0 / ibi))
    rmssd     = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
    pnn50     = float(np.mean(np.abs(np.diff(ibi)) > 50))
    ibi_range = float(ibi.max() - ibi.min())

    feat = np.array([hr_mean, rmssd, pnn50, ibi_range], dtype=np.float32)
    return feat if np.all(np.isfinite(feat)) else None


def build_bvp_lookup(samsung_root):
    """
    Scan samsung_root for *_STIMULUS_SAMSUNG_WATCH.json and build
    dict: (subject_str, EMOTION_STR) → float32[4].
    """
    patterns = [
        os.path.join(samsung_root, '*_STIMULUS_SAMSUNG_WATCH.json'),
        os.path.join(samsung_root, '*', '*_STIMULUS_SAMSUNG_WATCH.json'),
        os.path.join(samsung_root, '**', '*_STIMULUS_SAMSUNG_WATCH.json'),
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})

    lookup = {}
    n_ok = n_fail = 0
    for fp in files:
        name  = os.path.splitext(os.path.basename(fp))[0].split('_')
        if len(name) < 2:
            continue
        subj, emot = name[0], name[1].upper()
        if emot not in TARGET_EMOT:
            continue
        feat = load_bvp_features_one(fp)
        if feat is not None:
            lookup[(subj, emot)] = feat
            n_ok += 1
        else:
            n_fail += 1

    print(f"  BVP lookup: {n_ok} loaded, {n_fail} failed "
          f"({len(set(s for s,_ in lookup))} subjects)")
    return lookup


# ════════════════════════════════════════════════════════════════════════════
#  Multimodal Model Wrapper
# ════════════════════════════════════════════════════════════════════════════

class MultimodalMBModel(nn.Module):
    """
    Wraps MBInvBaseBiMamba and concatenates BVP features before
    the final classification head.

    EEG embedding (d_model) → concat [HR_mean, RMSSD, pNN50, IBI_range]
    → LayerNorm → Dropout(0.5) → Linear(d_model+4 → 32) → ELU
    → Dropout(0.3) → Linear(32 → n_classes)
    """
    def __init__(self, backbone: MBInvBaseBiMamba, bvp_dim: int, n_classes: int,
                 dropout: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.bvp_dim  = bvp_dim
        d_emb         = backbone.d_model         # embedding dimension

        # Remove backbone's original head; replace with multimodal one
        in_dim = d_emb + bvp_dim
        self.head = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 32),
            nn.ELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(32, n_classes),
        )

    def forward(self, x_eeg, x_bvp=None):
        """
        x_eeg : (B, 20, T)
        x_bvp : (B, 4)  or None
        """
        emb = self.backbone.get_embedding(x_eeg)   # (B, d_model)
        if self.bvp_dim > 0 and x_bvp is not None:
            emb = torch.cat([emb, x_bvp], dim=-1)  # (B, d_model+4)
        return self.head(emb)


# ════════════════════════════════════════════════════════════════════════════
#  Dataset
# ════════════════════════════════════════════════════════════════════════════

class EmognitionMBDataset(Dataset):
    """
    Dataset of (20, window_size) EEG windows + optional BVP feature vector.
    Augmentations applied only when augment=True (training).
    """

    def __init__(self, windows, labels, bvp_feats=None, augment: bool = False,
                 noise_ratio: float = 0.03,
                 scale_range: tuple = (0.85, 1.15),
                 band_drop_p: float = 0.15,
                 time_mask_p: float = 0.40,
                 time_mask_frac: float = 0.10):
        self.windows        = windows
        self.labels         = labels
        self.bvp_feats      = (torch.tensor(np.array(bvp_feats), dtype=torch.float32)
                               if bvp_feats is not None else None)
        self.augment        = augment
        self.noise_ratio    = noise_ratio
        self.scale_range    = scale_range
        self.band_drop_p    = band_drop_p
        self.time_mask_p    = time_mask_p
        self.time_mask_frac = time_mask_frac

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x     = self.windows[idx].copy()
        label = self.labels[idx]
        if self.augment:
            x = self._augment(x)
        x_t = torch.from_numpy(x)
        if self.bvp_feats is not None:
            return x_t, self.bvp_feats[idx], label
        return x_t, label

    def _augment(self, x: np.ndarray) -> np.ndarray:
        σ = x.std()
        if σ > 1e-8:
            x = x + np.random.randn(*x.shape).astype(np.float32) * σ * self.noise_ratio
        x = x * np.random.uniform(*self.scale_range)
        if np.random.random() < self.band_drop_p:
            b = np.random.randint(0, NUM_BANDS)
            x[b*4:(b+1)*4, :] = 0.0
        if np.random.random() < self.time_mask_p:
            T   = x.shape[1]
            ml  = max(1, int(T * self.time_mask_frac))
            s   = np.random.randint(0, max(T - ml, 1) + 1)
            x[:, s:s+ml] = 0.0
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

def evaluate(model, loader, device, criterion, use_bvp=False):
    """Evaluate model. Returns (loss, accuracy, macro-F1, preds, labels)."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            if use_bvp and len(batch) == 3:
                bx, bb, by = batch
                bb = bb.to(device)
            else:
                bx, by = batch[0], batch[-1]
                bb = None
            bx = bx.to(device)
            by = (by.long().to(device) if isinstance(by, torch.Tensor)
                  else torch.tensor(by, dtype=torch.long, device=device))
            out = model(bx, bb) if use_bvp else model(bx)
            total_loss += criterion(out, by).item()
            all_preds.extend(torch.argmax(out, 1).cpu().numpy())
            all_labels.extend(by.cpu().numpy())
            n_batches += 1

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
        description="MB-InvBase-BiMamba — Multimodal Emognition Training (LOSO)"
    )

    # ── data ──
    parser.add_argument("--data_root",    required=True,
                        help="Emognition Processed dataset root (EEG JSON files)")
    parser.add_argument("--samsung_root", default=None,
                        help="Samsung Watch data root (default: same as data_root)")
    parser.add_argument("--no_bvp",       action="store_true",
                        help="Disable BVP fusion — run EEG-only ablation")
    parser.add_argument("--emotions",     nargs="+",
                        default=["ENTHUSIASM", "FEAR", "NEUTRAL", "SADNESS"])
    parser.add_argument("--min_trial_sec",type=float, default=5.0)

    # ── evaluation mode ──
    parser.add_argument("--mode", choices=["loso", "sub_indep"],
                        default="loso",
                        help="loso = leave-one-subject-out (default)")

    # ── windowing ──
    parser.add_argument("--window_sec",   type=float, default=10.0)
    parser.add_argument("--val_size",     type=float, default=0.15)
    parser.add_argument("--test_size",    type=float, default=0.15)

    # ── model (reduced defaults to fight overfitting) ──
    parser.add_argument("--d_model",         type=int,   default=32)
    parser.add_argument("--n_layers",        type=int,   default=2)
    parser.add_argument("--d_state",         type=int,   default=16)
    parser.add_argument("--dropout",         type=float, default=0.55)
    parser.add_argument("--attn_reduction",  type=int,   default=4)

    # ── training ──
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--epochs",        type=int,   default=120)
    parser.add_argument("--lr",            type=float, default=1e-4)
    parser.add_argument("--weight_decay",  type=float, default=0.05)
    parser.add_argument("--warmup_epochs", type=int,   default=5)
    parser.add_argument("--label_smooth",  type=float, default=0.20)
    parser.add_argument("--patience",      type=int,   default=25)

    # ── misc ──
    parser.add_argument("--seed",    type=int,  default=42)
    parser.add_argument("--device",  type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overfit_test", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)

    args = parser.parse_args()
    setup_seed(args.seed)
    device        = torch.device(args.device)
    window_size   = int(args.window_sec * FS)
    samsung_root  = args.samsung_root or args.data_root
    use_bvp       = not args.no_bvp
    save_dir      = args.save_dir or os.path.join(_SCRIPT_DIR, "checkpoints",
                                                  "mb_invbase_bimamba")
    os.makedirs(save_dir, exist_ok=True)

    print(f"\n{'='*70}")
    print(f"  MB-InvBase-BiMamba  /  Emognition  /  {args.mode.upper()}")
    print(f"{'='*70}")
    print(f"  data_root   : {args.data_root}")
    print(f"  BVP fusion  : {'✅ ON' if use_bvp else '❌ OFF (EEG-only)'}")
    print(f"  mode        : {args.mode}")
    print(f"  window      : {args.window_sec}s → {window_size} samples")
    print(f"  model       : d_model={args.d_model}, n_layers={args.n_layers},"
          f" dropout={args.dropout}")
    print(f"  training    : lr={args.lr}, wd={args.weight_decay},"
          f" smooth={args.label_smooth}, patience={args.patience}")
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

    # ── 4. BVP lookup ────────────────────────────────────────────────────────
    bvp_lookup = None
    bvp_mean   = bvp_std = None
    emot_strs  = [id2lab[l] for l in labels]   # emotion string per trial

    if use_bvp:
        print("Step 4 — Loading Samsung Watch BVP features...")
        bvp_lookup = build_bvp_lookup(samsung_root)
        # Global BVP normalisation stats (across all trials)
        vecs = [bvp_lookup.get((s, e)) for s, e in zip(subject_ids, emot_strs)
                if bvp_lookup.get((s, e)) is not None]
        if vecs:
            arr       = np.stack(vecs)
            bvp_mean  = arr.mean(0).astype(np.float32)
            bvp_std   = (arr.std(0) + 1e-8).astype(np.float32)
            print(f"  BVP stats: mean={bvp_mean.round(2)}, std={bvp_std.round(2)}")
        print()

    def get_bvp_per_window(subj_list, emot_str_list, n_wins_list):
        """Replicate clip-level BVP feature for every window of that clip."""
        if not use_bvp or bvp_lookup is None:
            return None
        out = []
        for subj, emot, nw in zip(subj_list, emot_str_list, n_wins_list):
            vec = bvp_lookup.get((subj, emot), np.zeros(BVP_DIM, np.float32))
            if bvp_mean is not None:
                vec = (vec - bvp_mean) / bvp_std
            out.extend([vec] * nw)
        return out

    def run_one_split(tr_proc, tr_lbl, tr_sub, tr_emot,
                      va_proc, va_lbl, va_sub, va_emot,
                      te_proc, te_lbl, te_sub, te_emot,
                      fold_name=""):
        """
        Build loaders, create model, train, and return test metrics.
        Returns (te_acc, te_f1, te_preds, te_labels).
        """
        step_tr = window_size // 2
        step_ev = window_size

        tr_wins, tr_wlbls, tr_wsubs = window_trials(
            tr_proc, tr_lbl, tr_sub, window_size, step_tr)
        va_wins, va_wlbls, va_wsubs = window_trials(
            va_proc, va_lbl, va_sub, window_size, step_ev)
        te_wins, te_wlbls, te_wsubs = window_trials(
            te_proc, te_lbl, te_sub, window_size, step_ev)

        # Count windows per trial for BVP replication
        def count_wins(proc, step):
            return [len(list(range(0, max(t.shape[1]-window_size+1,1), step)))
                    for t in proc]

        tr_bvp = get_bvp_per_window(tr_sub, tr_emot, count_wins(tr_proc, step_tr))
        va_bvp = get_bvp_per_window(va_sub, va_emot, count_wins(va_proc, step_ev))
        te_bvp = get_bvp_per_window(te_sub, te_emot, count_wins(te_proc, step_ev))

        if fold_name:
            print(f"  Windows: tr={len(tr_wins)}, va={len(va_wins)}, te={len(te_wins)}")

        tr_ds = EmognitionMBDataset(tr_wins, tr_wlbls, tr_bvp, augment=True)
        va_ds = EmognitionMBDataset(va_wins, va_wlbls, va_bvp, augment=False)
        te_ds = EmognitionMBDataset(te_wins, te_wlbls, te_bvp, augment=False)

        tr_dl = DataLoader(tr_ds, args.batch_size, shuffle=True,
                           drop_last=False, num_workers=0, pin_memory=True)
        va_dl = DataLoader(va_ds, args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)
        te_dl = DataLoader(te_ds, args.batch_size, shuffle=False,
                           num_workers=0, pin_memory=True)

        # Build model fresh for each fold
        backbone = MBInvBaseBiMamba(
            in_channels=IN_CHANNELS, num_classes=NUM_CLASSES,
            d_model=args.d_model,    n_layers=args.n_layers,
            d_state=args.d_state,    dropout=args.dropout,
            attn_reduction=args.attn_reduction,
        )
        if use_bvp:
            fold_model = MultimodalMBModel(
                backbone, BVP_DIM, NUM_CLASSES, dropout=args.dropout
            ).to(device)
        else:
            fold_model = backbone.to(device)

        crit      = LabelSmoothingCE(NUM_CLASSES, args.label_smooth)
        eval_crit = nn.CrossEntropyLoss()
        opt       = optim.AdamW(fold_model.parameters(), lr=args.lr,
                                weight_decay=args.weight_decay, eps=1e-8)
        sched     = WarmupCosineScheduler(opt, args.warmup_epochs,
                                          args.epochs, min_lr=1e-7)

        best_f1 = 0.0; best_st = None; pat_ctr = 0

        for epoch in range(1, args.epochs + 1):
            fold_model.train()
            ep_loss = ep_n = ep_ok = ep_tot = 0

            for batch in tr_dl:
                if use_bvp and len(batch) == 3:
                    bx, bb, by = batch
                    bb = bb.to(device)
                else:
                    bx, by = batch[0], batch[-1]
                    bb = None
                bx = bx.to(device)
                by = by.long().to(device)
                opt.zero_grad()
                out  = fold_model(bx, bb) if use_bvp else fold_model(bx)
                loss = crit(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(fold_model.parameters(), 1.0)
                opt.step()
                ep_loss += loss.item(); ep_n += 1
                ep_ok   += (out.argmax(1) == by).sum().item()
                ep_tot  += len(by)

            sched.step()
            _, va_acc, va_f1, _, _ = evaluate(fold_model, va_dl, device,
                                               eval_crit, use_bvp)

            if epoch % 10 == 0 or epoch == 1:
                tr_acc = ep_ok / max(ep_tot, 1)
                print(f"  {fold_name} Ep{epoch:3d} | "
                      f"Tr:{tr_acc:.3f} | Va:{va_acc:.3f} F1:{va_f1:.3f} | "
                      f"lr:{sched.get_last_lr()[0]:.1e}")

            if va_f1 > best_f1:
                best_f1 = va_f1
                best_st = {k: v.cpu().clone() for k, v in fold_model.state_dict().items()}
                pat_ctr = 0
            else:
                pat_ctr += 1
                if args.patience > 0 and pat_ctr >= args.patience:
                    print(f"  Early stop at epoch {epoch}")
                    break

        if best_st:
            fold_model.load_state_dict(best_st)
            fold_model = fold_model.to(device)

        _, te_acc, te_f1, te_preds, te_lbls = evaluate(
            fold_model, te_dl, device, eval_crit, use_bvp)
        return te_acc, te_f1, te_preds, te_lbls

    # ── 5+6+7: Split, window, train ──────────────────────────────────────────
    n_params_est = sum(p.numel() for p in MBInvBaseBiMamba(
        IN_CHANNELS, NUM_CLASSES, args.d_model, args.n_layers,
        args.d_state, args.dropout, args.attn_reduction).parameters())
    bvp_params = (args.d_model + BVP_DIM) * 32 + 32 * NUM_CLASSES + 32 + NUM_CLASSES \
                 if use_bvp else 0
    print(f"Step 5 — Model: MBInvBaseBiMamba{'+ BVP head' if use_bvp else ''}")
    print(f"  EEG params : {n_params_est:,}")
    print(f"  BVP head   : {'~'+str(bvp_params) if use_bvp else 'N/A'}")
    print()

    if args.mode == 'loso':
        # ── LOSO ─────────────────────────────────────────────────────────────
        unique_subjs = sorted(set(subject_ids))
        all_preds, all_true, fold_accs, fold_f1s = [], [], [], []

        for fi, test_subj in enumerate(unique_subjs):
            print(f"\n{'='*70}")
            print(f"  FOLD {fi+1}/{len(unique_subjs)}  —  Test subject: {test_subj}")
            print(f"{'='*70}")
            setup_seed(args.seed + fi)

            te_idx = [i for i in range(len(labels)) if subject_ids[i] == test_subj]
            tr_all = [i for i in range(len(labels)) if subject_ids[i] != test_subj]

            # 15% of remaining subjects → val
            rem_subjs  = sorted(set(subject_ids[i] for i in tr_all))
            n_va       = max(1, int(0.15 * len(rem_subjs)))
            va_subjs_f = set(np.random.RandomState(args.seed).choice(
                             rem_subjs, n_va, replace=False))
            tr_idx = [i for i in tr_all if subject_ids[i] not in va_subjs_f]
            va_idx = [i for i in tr_all if subject_ids[i] in va_subjs_f]

            def gd(idx):
                return ([processed_trials[i] for i in idx],
                        [labels[i]            for i in idx],
                        [subject_ids[i]       for i in idx],
                        [emot_strs[i]         for i in idx])

            ta, tl, ts, te_ = gd(tr_idx)
            va, vl, vs, ve  = gd(va_idx)
            xa, xl, xs, xe  = gd(te_idx)

            acc, f1, preds, true = run_one_split(
                ta, tl, ts, te_, va, vl, vs, ve, xa, xl, xs, xe,
                fold_name=f"Fold{fi+1}")

            print(f"  → Fold {fi+1} ({test_subj}): Acc={acc:.4f}  F1={f1:.4f}")
            all_preds.extend(preds); all_true.extend(true)
            fold_accs.append(acc);   fold_f1s.append(f1)

        # ── Aggregate ─────────────────────────────────────────────────────────
        loso_acc = np.mean(np.array(all_preds) == np.array(all_true))
        loso_f1  = f1_score(all_true, all_preds, average='macro', zero_division=0)
        print(f"\n{'='*70}")
        print(f"  LOSO FINAL — {'Multimodal EEG+BVP' if use_bvp else 'EEG-only'}")
        print(f"{'='*70}")
        print(f"  Global Acc : {loso_acc:.4f}  ({loso_acc*100:.1f}%)")
        print(f"  Global F1  : {loso_f1:.4f}")
        print(f"  Per-fold   : mean={np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
        print(f"  Chance     : {100/NUM_CLASSES:.1f}%")
        print_report(all_true, all_preds,
                     title=f"LOSO {'EEG+BVP' if use_bvp else 'EEG-only'}")

    else:
        # ── 70/15/15 subject-independent split ────────────────────────────────
        print("Step — Subject-independent split (70/15/15)...")
        tr_subjs, va_subjs, te_subjs = subject_split(
            subject_ids, seed=args.seed,
            val_frac=args.val_size, test_frac=args.test_size)
        print(f"  Train:{len(tr_subjs)}  Val:{len(va_subjs)}  Test:{len(te_subjs)}\n")

        def gd(subj_set):
            idx = [i for i, s in enumerate(subject_ids) if s in subj_set]
            return ([processed_trials[i] for i in idx], [labels[i] for i in idx],
                    [subject_ids[i] for i in idx],      [emot_strs[i] for i in idx])

        ta, tl, ts, te_ = gd(tr_subjs)
        va, vl, vs, ve  = gd(va_subjs)
        xa, xl, xs, xe  = gd(te_subjs)

        acc, f1, preds, true = run_one_split(
            ta, tl, ts, te_, va, vl, vs, ve, xa, xl, xs, xe, fold_name="")

        print(f"\n{'='*70}")
        print(f"  RESULTS — {'EEG+BVP' if use_bvp else 'EEG-only'} / sub_indep")
        print(f"{'='*70}")
        print(f"  Test Acc : {acc:.4f}  ({acc*100:.1f}%)")
        print(f"  Test F1  : {f1:.4f}")
        print_report(true, preds, title="sub_indep")


if __name__ == "__main__":
    main()

