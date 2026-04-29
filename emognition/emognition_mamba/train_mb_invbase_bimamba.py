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
from mb_invbase_bimamba_model import (MBInvBaseBiMamba, IN_CHANNELS,
                                      BVPEncoder, FiLMLayer, IBI_MAX_BEATS)

# scipy is used only during data preprocessing (not in the training loop)
from scipy.signal import butter, filtfilt


# ── constants ────────────────────────────────────────────────────────────────

FS           = 256           # Muse 2 sampling rate (Hz)
NUM_CLASSES  = 4
CLASS_NAMES  = ["ENTHUSIASM", "FEAR", "NEUTRAL", "SADNESS"]  # alphabetical

# L-R electrode symmetry flip index for the stacked (20, T) band representation.
# MUSE 2 electrode order per band: [TP9(L), AF7(L), AF8(R), TP10(R)]
# Swapping left ↔ right hemispheres: [TP10, AF8, AF7, TP9] = indices [3,2,1,0] per band.
# This is a valid label-preserving augmentation because:
#   (a) emotions activate bilateral brain networks,
#   (b) headband fit varies slightly left/right across subjects,
#   (c) inflates effective training set size by ~2× for free.
# Pattern: reverse within each 4-channel block across all 5 bands.
_LR_FLIP_IDX = np.array([
    3, 2, 1, 0,          # band 0 (delta):  TP10, AF8, AF7, TP9
    7, 6, 5, 4,          # band 1 (theta)
    11, 10, 9, 8,        # band 2 (alpha)
    15, 14, 13, 12,      # band 3 (beta)
    19, 18, 17, 16,      # band 4 (gamma)
], dtype=np.intp)


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


# ════════════════════════════════════════════════════════════════════════════
#  Baseline Normalisation
# ════════════════════════════════════════════════════════════════════════════

def apply_zscore_baseline(trial: np.ndarray, baseline_stats: dict | None,
                          fs: float = FS) -> np.ndarray:
    """
    Z-score normalise using per-channel statistics from the resting baseline.

    Z(t) = (x(t) - μ_baseline) / σ_baseline   per channel

    This removes each subject's amplitude bias AND variance mismatch in the
    time domain.  If no baseline is available, falls back to within-trial
    z-score.

    Args:
        trial:           (4, T) float32 — clipped raw EEG
        baseline_stats:  dict {'μ': (4,), 'σ': (4,)} or None
        fs:              sampling rate (unused, kept for API symmetry)
    Returns:
        (4, T) float32 — z-scored trial
    """
    trial = trial.astype(np.float32)
    if baseline_stats is not None:
        μ = baseline_stats['μ'][:, np.newaxis]          # (4, 1)
        σ = baseline_stats['σ'][:, np.newaxis] + 1e-8   # (4, 1)
    else:
        # fallback: within-trial normalisation
        μ = trial.mean(axis=1, keepdims=True)
        σ = trial.std(axis=1, keepdims=True) + 1e-8
    return ((trial - μ) / σ).astype(np.float32)


def extract_zscore_baselines(baselines_raw: dict) -> dict:
    """
    Convert raw baseline time-series (4, T) per subject into
    {subj: {'μ': (4,), 'σ': (4,)}} dicts for Z-score normalisation.

    Args:
        baselines_raw: dict subj → (4, T) raw baseline EEG array
                       (as returned by load_baselines_processed when
                        raw=True is supported, or pre-extracted here)
    Returns:
        dict subj → {'μ': np.float32 (4,), 'σ': np.float32 (4,)}
    """
    stats = {}
    for subj, raw in baselines_raw.items():
        if raw is None:
            continue
        arr = np.asarray(raw, dtype=np.float32)     # (4, T)
        stats[subj] = {
            'μ': arr.mean(axis=1).astype(np.float32),
            'σ': arr.std(axis=1).astype(np.float32),
        }
    return stats


def process_trial(trial: np.ndarray, baseline_info, fs: float = FS,
                  norm_mode: str = 'zscore') -> np.ndarray:
    """
    Full pre-processing pipeline for one raw EEG trial.

    Steps:
      1. Clip artefacts (±5 σ per channel)
      2. Normalise:
           'zscore'  : Z(t) = (x(t) - μ_base) / σ_base  [default]
           'invbase' : spectral inverse-baseline (original)
      3. Band-filter into 5 bands → stack → (20, T)

    Args:
        trial:         (4, T) float — raw EEG trial
        baseline_info: for 'zscore': {'μ':(4,), 'σ':(4,)} or None
                       for 'invbase': (4, freq_bins) spectrum or None
        fs:            sampling rate in Hz
        norm_mode:     'zscore' | 'invbase'
    Returns:
        (20, T) float32
    """
    trial = clip_artefacts(trial, n_sigma=5.0)

    if norm_mode == 'zscore':
        trial = apply_zscore_baseline(trial, baseline_info, fs=fs)
    else:
        trial = apply_invbase_to_raw(trial, baseline_info, fs=fs)

    return apply_band_stack(trial, fs=fs)


# ════════════════════════════════════════════════════════════════════════════
#  Z-score Baseline Loader  (loads raw EEG  —  NOT spectral data)
# ════════════════════════════════════════════════════════════════════════════

# Raw EEG channel names — must match what load_baselines_processed uses
# (imported from invbase.CHANNELS so both functions scan the same JSON keys)
from invbase import CHANNELS as _RAW_EEG_CHANNELS


def load_baselines_raw(data_root: str) -> dict:
    """
    Load raw EEG time-series from BASELINE_MUSE_cleaned.json files and compute
    per-channel (μ, σ) for z-score normalisation.

    Uses the SAME glob patterns as load_baselines_processed() in invbase.py,
    which is proven to find all 41 baseline files on the Kaggle dataset.
    Reads channel data as direct top-level JSON keys (obj.get(ch, [])),
    matching how invbase.py reads the MUSE JSON format.

    Args:
        data_root: path to the processed Emognition dataset root.
    Returns:
        dict: subj_str -> {'μ': float32(4,), 'σ': float32(4,)}
              Empty dict if no baseline files are found.
    """
    # Same patterns as load_baselines_processed — proven to find 41 files
    patterns = [
        os.path.join(data_root, "*", "*_BASELINE_STIMULUS_MUSE_cleaned",
                     "*_BASELINE_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "*", "*_BASELINE_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "**", "*_BASELINE_STIMULUS_MUSE_cleaned.json"),
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})

    # Fallback to non-cleaned pattern
    if not files:
        orig = [
            os.path.join(data_root, "*_BASELINE_STIMULUS_MUSE.json"),
            os.path.join(data_root, "*", "*_BASELINE_STIMULUS_MUSE.json"),
            os.path.join(data_root, "**", "*_BASELINE_STIMULUS_MUSE.json"),
        ]
        files = sorted({p for pat in orig for p in glob.glob(pat, recursive=True)})

    print(f"[load_baselines_raw] Found {len(files)} BASELINE files")
    stats = {}

    for fp in files:
        # Extract subject ID from filename (same as load_baselines_processed)
        name = os.path.splitext(os.path.basename(fp))[0]
        sid  = name.split("_")[0]   # e.g. "22_BASELINE_..." -> "22"
        try:
            with open(fp) as fh:
                obj = json.load(fh)
            # Read raw EEG channels directly from top-level JSON keys
            # (same format as invbase.py — NOT wrapped in a DataFrame)
            raw_ch = []
            for ch in _RAW_EEG_CHANNELS:
                arr = np.asarray(obj.get(ch, []), dtype=np.float64)
                if len(arr) == 0:
                    raw_ch = []
                    break
                raw_ch.append(arr)
            if len(raw_ch) != len(_RAW_EEG_CHANNELS):
                print(f"  [load_baselines_raw] {sid}: missing channels in JSON, skipping")
                continue
            L   = min(len(a) for a in raw_ch)
            sig = np.stack([a[:L] for a in raw_ch], axis=0)   # (4, T)
            sig = np.nan_to_num(sig, nan=0.0, posinf=0.0, neginf=0.0)
            mu  = sig.mean(axis=1).astype(np.float32)
            sd  = sig.std(axis=1).astype(np.float32)
            sd  = np.where(sd < 1e-6, 1.0, sd)               # clamp flat channels
            stats[sid] = {'μ': mu, 'σ': sd}
        except Exception as e:
            print(f"  [load_baselines_raw] {sid}: {e}")

    print(f"[load_baselines_raw] Loaded raw baseline for {len(stats)} subjects")
    return stats


# ════════════════════════════════════════════════════════════════════════════
#  Euclidean Alignment  (He & Wu 2019)  -- per-subject EEG domain adaptation
# ════════════════════════════════════════════════════════════════════════════

def _reg_cov(X: np.ndarray, reg: float = 1e-4) -> np.ndarray:
    """Regularised covariance of (C, T) array."""
    C, T = X.shape
    Xc   = X - X.mean(axis=1, keepdims=True)
    cov  = (Xc @ Xc.T) / max(T - 1, 1)
    cov  = (1 - reg) * cov + reg * (np.trace(cov) / C) * np.eye(C)
    return cov


def _sqrt_inv(M: np.ndarray) -> np.ndarray:
    """Symmetric matrix square-root inverse via eigendecomposition."""
    v, U = np.linalg.eigh(M)
    return U @ np.diag(1.0 / np.sqrt(np.maximum(v, 1e-10))) @ U.T


def euclidean_align_subjects(trials: list, subject_ids: list) -> list:
    """
    Euclidean Alignment (EA) per subject.

    For each subject s, compute the arithmetic mean covariance R_s across all
    their raw EEG trials, then apply:  x_aligned = R_s^{-1/2} @ x_raw

    This re-centres every subject's EEG covariance to the identity matrix on
    the SPD manifold, dramatically reducing inter-subject variability before
    band-filtering.  Fully unsupervised -- requires no labels.

    Reference:
        He & Wu, "Transfer Learning for Brain-Computer Interfaces:
        A Euclidean Space Data Alignment Approach", IEEE TNSRE 2019.

    Args:
        trials:      list of (4, T_i) float32 raw EEG arrays
        subject_ids: list of str, same length as trials
    Returns:
        list of (4, T_i) float32 EA-aligned EEG arrays (same order)
    """
    from collections import defaultdict as _dd
    subj_idx = _dd(list)
    for i, sid in enumerate(subject_ids):
        subj_idx[sid].append(i)

    aligned = list(trials)             # shallow copy, entries replaced below
    n_aligned = 0
    for sid, idxs in subj_idx.items():
        if len(idxs) < 2:              # single trial -- EA undefined, skip
            continue
        trs  = [trials[i].astype(np.float64) for i in idxs]
        covs = [_reg_cov(t) for t in trs]
        R    = np.stack(covs, axis=0).mean(axis=0)
        Rinv = _sqrt_inv(R)
        for i, t in zip(idxs, trs):
            aligned[i] = (Rinv @ t).astype(np.float32)
        n_aligned += len(idxs)
    print(f"  [EA] Aligned {n_aligned}/{len(trials)} trials across "
          f"{len(subj_idx)} subjects")
    return aligned




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
                  window_size: int, step: int,
                  emot_strs: list = None):
    """
    Slice pre-processed trials into fixed-size windows.
    Returns clip_ids alongside windows so clip-level evaluation is possible.
    A clip_id is a unique integer per (subject, emotion) trial.
    """
    windows, win_labels, win_subjs, win_clip_ids = [], [], [], []
    clip_id = 0

    for idx, (trial, label, subj) in enumerate(
            zip(processed_trials, labels, subject_ids)):
        C, T  = trial.shape
        starts = list(range(0, max(T - window_size + 1, 1), step))
        for s in starts:
            win = trial[:, s:s + window_size]
            if win.shape[1] < window_size:
                pad = window_size - win.shape[1]
                win = np.pad(win, ((0, 0), (0, pad)))
            windows.append(win.astype(np.float32))
            win_labels.append(label)
            win_subjs.append(subj)
            win_clip_ids.append(clip_id)
        clip_id += 1

    return windows, win_labels, win_subjs, win_clip_ids


# ════════════════════════════════════════════════════════════════════════════
#  PyTorch Dataset
# ════════════════════════════════════════════════════════════════════════════

# ════════════════════════════════════════════════════════════════════════════
#  BVP / Samsung Watch Feature Extraction
# ════════════════════════════════════════════════════════════════════════════

# BVP feature layout (stored as a single vector per clip):
#   First BVP_HAND_DIM dims : handcrafted HRV features (global z-score in training loop)
#   Last  IBI_MAX_BEATS dims : per-clip z-scored IBI series (already normalised)
BVP_HAND_DIM = 11                   # HR_mean, RMSSD, pNN50, pNN20,
                                    # IBI_range, SDNN, mean_IBI,
                                    # LF_power, HF_power, LF_HF_ratio, IBI_cv
BVP_DIM      = BVP_HAND_DIM + IBI_MAX_BEATS   # 11 + 64 = 75
TARGET_EMOT  = {'ENTHUSIASM', 'FEAR', 'NEUTRAL', 'SADNESS'}


def _parse_paired(raw):
    """[[timestamp, value], ...] → numpy 1-D array of values."""
    if not isinstance(raw, list) or len(raw) < 5:
        return None
    try:
        return np.array([r[1] for r in raw], dtype=np.float64)
    except Exception:
        return None


def _lf_hf_proxy(ibi: np.ndarray, fs_ibi: float = 4.0):
    """
    Approximate LF (0.04–0.15 Hz) and HF (0.15–0.4 Hz) power from IBI.
    Uses Welch PSD on uniformly resampled IBI signal.
    Returns (lf_power, hf_power) or (0, 0) if too short.
    """
    try:
        from scipy.signal import welch, resample
        if len(ibi) < 10:
            return 0.0, 0.0
        # Resample to uniform grid at fs_ibi Hz (standard for HRV)
        duration = len(ibi) / fs_ibi
        n_samp   = max(int(duration * fs_ibi), 8)
        ibi_uni  = resample(ibi, n_samp)
        f, pxx   = welch(ibi_uni, fs=fs_ibi, nperseg=min(64, n_samp))
        lf = float(np.trapz(pxx[(f >= 0.04) & (f < 0.15)],
                             f[(f >= 0.04) & (f < 0.15)] + 1e-12))
        hf = float(np.trapz(pxx[(f >= 0.15) & (f < 0.40)],
                             f[(f >= 0.15) & (f < 0.40)] + 1e-12))
        return max(lf, 0.0), max(hf, 0.0)
    except Exception:
        return 0.0, 0.0


def load_bvp_features_one(fp):
    """
    Extract a 75-dim BVP feature vector from one Samsung Watch STIMULUS JSON.

    Layout: [handcrafted(11) | ibi_series_zscored(64)]

    Handcrafted HRV features (11, indices 0-10):
      Time-domain (7):
        0  HR_mean     — mean heart rate in BPM
        1  RMSSD       — root-mean-square successive RR differences (vagal tone)
        2  pNN50       — fraction of RR diffs > 50 ms (parasympathetic index)
        3  pNN20       — fraction of RR diffs > 20 ms (more sensitive short-term HRV)
        4  IBI_range   — max − min IBI (overall beat-interval spread)
        5  SDNN        — std-dev of all RR intervals (global HRV)
        6  mean_IBI    — mean RR interval (inversely proportional to HR)
      Frequency-domain (3):
        7  LF_power    — 0.04–0.15 Hz band (sympathovagal balance)
        8  HF_power    — 0.15–0.40 Hz band (vagal/respiratory)
        9  LF_HF_ratio — LF/HF ratio (autonomic balance; elevated under stress)
      Normalised (1):
        10 IBI_cv      — SDNN / mean_IBI (HR-normalised HRV, coefficient of variation)

    IBI series (64, indices 11-74):
        Per-clip z-scored IBI values, padded/truncated to IBI_MAX_BEATS (64).
        Per-clip normalisation: (ibi - ibi.mean()) / (ibi.std() + eps)
        Zero-padding for clips shorter than 64 beats.
        The CNN learns temporal dynamics (deceleration shape, RSA oscillations)
        that scalar HRV statistics compress away.

    Note on double-normalisation:
        The training loop applies global z-score to all BVP_DIM=75 dimensions.
        The IBI series is already per-clip z-scored (mean≈0, std≈1 across clips),
        so the global z-score is approximately identity for those 64 dims — safe.

    Returns float32 array (75,) or None if IBI data is insufficient.
    """
    try:
        with open(fp) as f:
            obj = json.load(f)
    except Exception:
        return None

    ibi_raw = _parse_paired(obj.get('PPInterval'))
    hr      = _parse_paired(obj.get('heartRate'))

    if ibi_raw is not None:
        ibi_raw = ibi_raw[(ibi_raw > 300) & (ibi_raw < 2000) & np.isfinite(ibi_raw)]
    if hr is not None:
        hr = hr[(hr > 30) & (hr < 220) & np.isfinite(hr)]

    if ibi_raw is None or len(ibi_raw) < 5:
        return None

    ibi       = ibi_raw
    diff_ibi  = np.diff(ibi)
    hr_mean   = (float(np.mean(hr)) if (hr is not None and len(hr) >= 3)
                 else float(np.mean(60000.0 / ibi)))
    rmssd       = float(np.sqrt(np.mean(diff_ibi ** 2)))
    pnn50       = float(np.mean(np.abs(diff_ibi) > 50))
    pnn20       = float(np.mean(np.abs(diff_ibi) > 20))
    ibi_range   = float(ibi.max() - ibi.min())
    sdnn        = float(ibi.std())
    mean_ibi    = float(ibi.mean())
    lf, hf      = _lf_hf_proxy(ibi)
    lf_hf_ratio = float(lf / (hf + 1e-8))
    ibi_cv      = float(sdnn / (mean_ibi + 1e-8))

    hand = np.array(
        [hr_mean, rmssd, pnn50, pnn20, ibi_range, sdnn, mean_ibi,
         lf, hf, lf_hf_ratio, ibi_cv],
        dtype=np.float32)
    if not np.all(np.isfinite(hand)):
        return None

    # ── per-clip z-score the IBI series and pad/truncate to IBI_MAX_BEATS ──
    ibi_f   = ibi.astype(np.float32)
    ibi_mu  = ibi_f.mean()
    ibi_sig = ibi_f.std() + 1e-8
    ibi_z   = (ibi_f - ibi_mu) / ibi_sig          # z-scored IBI series
    ibi_pad = np.zeros(IBI_MAX_BEATS, dtype=np.float32)
    n       = min(len(ibi_z), IBI_MAX_BEATS)
    ibi_pad[:n] = ibi_z[:n]                        # truncate or zero-pad

    return np.concatenate([hand, ibi_pad])         # (75,)


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
    FiLM-conditioned multimodal EEG + BVP emotion classifier.

    Architecture
    ────────────
    EEG path:
        (B, 20, T) → MBInvBaseBiMamba backbone → eeg_emb (B, d_model)

    BVP path (hybrid encoder):
        x_bvp[:, :11]  — handcrafted HRV → Linear(11→d_model) → GELU
        x_bvp[:, 11:]  — IBI series      → 2-layer 1D CNN → AdaptiveAvgPool
        Concat → Linear(2*d_model → d_model) → LayerNorm → GELU → Dropout
        → bvp_emb (B, d_model)

    FiLM conditioning:
        γ = Linear(d_model → d_model) initialised near 1
        β = Linear(d_model → d_model) initialised near 0
        modulated = LayerNorm(γ(bvp_emb) ⊙ eeg_emb + β(bvp_emb))

    MLP head:
        modulated → Dropout → Linear(d_model→d_model) → GELU
                 → Dropout → Linear(d_model→n_classes) → logits

    Fallback (--no_bvp):
        eeg_emb passed directly to MLP head without any FiLM modulation.
    """
    def __init__(self, backbone: MBInvBaseBiMamba, bvp_dim: int, n_classes: int,
                 dropout: float = 0.5):
        super().__init__()
        self.backbone = backbone
        self.bvp_dim  = bvp_dim
        d_model       = backbone.d_model

        if bvp_dim > 0:
            self.bvp_encoder = BVPEncoder(
                hand_dim = BVP_HAND_DIM,
                ibi_len  = IBI_MAX_BEATS,
                d_bvp    = d_model,
                dropout  = dropout * 0.4,
            )
            self.film = FiLMLayer(d_feat=d_model, d_cond=d_model)
        else:
            self.bvp_encoder = None
            self.film        = None

        # MLP head: non-linear decision boundary after FiLM modulation
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model, n_classes),
        )

    def forward(self, x_eeg, x_bvp=None):
        """
        x_eeg : (B, 20, T)
        x_bvp : (B, BVP_DIM=75) or None — [handcrafted(11) | ibi_series(64)]
        """
        eeg_emb = self.backbone.get_embedding(x_eeg)          # (B, d_model)
        if self.bvp_encoder is not None and x_bvp is not None:
            hand    = x_bvp[:, :BVP_HAND_DIM]                 # (B, 11)
            ibi     = x_bvp[:, BVP_HAND_DIM:]                 # (B, 64)
            bvp_emb = self.bvp_encoder(hand, ibi)             # (B, d_model)
            eeg_emb = self.film(eeg_emb, bvp_emb)             # (B, d_model)
        return self.head(eeg_emb)


# ════════════════════════════════════════════════════════════════════════════
#  Dataset
# ════════════════════════════════════════════════════════════════════════════

class EmognitionMBDataset(Dataset):
    """
    Dataset of (20, window_size) EEG windows + optional BVP feature vector.
    Stores clip_ids for clip-level aggregation at evaluation time.
    """

    def __init__(self, windows, labels, bvp_feats=None, clip_ids=None,
                 augment: bool = False,
                 noise_ratio: float = 0.05,         # bumped from 0.03 → 0.05
                 scale_range: tuple = (0.85, 1.15),
                 flip_lr_p: float = 0.50,           # NEW: L-R electrode flip
                 band_drop_p: float = 0.15,
                 time_mask_p: float = 0.40,
                 time_mask_frac: float = 0.10):
        self.windows        = windows
        self.labels         = labels
        self.clip_ids       = clip_ids        # list[int] or None
        self.bvp_feats      = (torch.tensor(np.array(bvp_feats), dtype=torch.float32)
                               if bvp_feats is not None else None)
        self.augment        = augment
        self.noise_ratio    = noise_ratio
        self.scale_range    = scale_range
        self.flip_lr_p      = flip_lr_p
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
        cid = self.clip_ids[idx] if self.clip_ids is not None else -1
        if self.bvp_feats is not None:
            return x_t, self.bvp_feats[idx], label, cid
        return x_t, label, cid

    def _augment(self, x: np.ndarray) -> np.ndarray:
        # ① Gaussian noise  (5% of signal std)
        σ = x.std()
        if σ > 1e-8:
            x = x + np.random.randn(*x.shape).astype(np.float32) * σ * self.noise_ratio

        # ② Amplitude scaling  (±15%)
        x = x * np.random.uniform(*self.scale_range)

        # ③ L-R electrode symmetry flip
        # Swaps [TP9(L), AF7(L), AF8(R), TP10(R)] → [TP10(R), AF8(R), AF7(L), TP9(L)]
        # per band — valid because emotion activates bilateral networks and
        # headband fit varies L/R across subjects.
        if np.random.random() < self.flip_lr_p:
            x = x[_LR_FLIP_IDX]

        # ④ Band dropout  (zero one entire frequency band)
        if np.random.random() < self.band_drop_p:
            b = np.random.randint(0, NUM_BANDS)
            x[b*4:(b+1)*4, :] = 0.0

        # ⑤ Time masking  (mask 10% of temporal window)
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
    """
    Window-level evaluation.
    Dataset must return (eeg, [bvp,] label, clip_id) tuples.
    Returns (loss, win_acc, win_f1, clip_acc, clip_f1, win_preds, win_labels).
    """
    model.eval()
    all_preds, all_labels, all_logits, all_cids = [], [], [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch in loader:
            if use_bvp and len(batch) == 4:
                bx, bb, by, cid = batch
                bb = bb.to(device)
            else:
                bx, by, cid = batch[0], batch[-2], batch[-1]
                bb = None
            bx  = bx.to(device)
            by  = (by.long().to(device) if isinstance(by, torch.Tensor)
                   else torch.tensor(by, dtype=torch.long, device=device))
            out = model(bx, bb) if use_bvp else model(bx)
            total_loss += criterion(out, by).item()
            probs = torch.softmax(out, dim=-1)
            all_logits.extend(probs.cpu().numpy())
            all_preds.extend(out.argmax(1).cpu().numpy())
            all_labels.extend(by.cpu().numpy())
            all_cids.extend(cid.cpu().numpy() if isinstance(cid, torch.Tensor)
                            else cid)
            n_batches += 1

    # Window-level
    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_logits = np.array(all_logits)
    all_cids   = np.array(all_cids)
    win_acc = float(np.mean(all_preds == all_labels))
    win_f1  = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Clip-level: equal-weight average of softmax probs over all windows per clip.
    # Uniform averaging is more robust than confidence-weighting on small test sets
    # (24 clips), where high-confidence wrong windows can dominate and hurt accuracy.
    clip_preds, clip_true = [], []
    for cid in np.unique(all_cids):
        mask       = all_cids == cid
        avg_prob   = all_logits[mask].mean(axis=0)   # (n_classes,)  equal-weight
        clip_pred  = int(avg_prob.argmax())
        clip_label = int(all_labels[mask][0])
        clip_preds.append(clip_pred)
        clip_true.append(clip_label)
    clip_acc = float(np.mean(np.array(clip_preds) == np.array(clip_true)))
    clip_f1  = f1_score(clip_true, clip_preds, average='macro', zero_division=0)

    return (total_loss / max(n_batches, 1),
            win_acc, win_f1, clip_acc, clip_f1,
            all_preds.tolist(), all_labels.tolist(),
            clip_preds, clip_true)


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
    parser.add_argument("--seed",       type=int,  default=42)
    parser.add_argument("--n_seeds",    type=int,  default=5,
                        help="Number of seeds for ensemble (1 = no ensemble)")
    parser.add_argument("--swa_start",  type=int,  default=40,
                        help="Epoch to start SWA averaging (0 = disable)")
    parser.add_argument("--norm_mode",  type=str,  default='invbase',
                        choices=['zscore', 'invbase'],
                        help="Baseline normalisation: invbase (default) or zscore")
    parser.add_argument("--use_ea", action="store_true", default=False,
                        help="Apply Euclidean Alignment per-subject before band-stack "
                             "(He & Wu 2019). NOTE: requires ≥10 trials per subject to "
                             "estimate stable mean covariance. DO NOT use with Emognition "
                             "(only 4 trials per subject) — will harm performance.")
    parser.add_argument("--device",  type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--overfit_test", action="store_true")
    parser.add_argument("--save_dir", type=str, default=None)
    parser.add_argument("--save_model", type=str, default=None,
                        help="Path to save the final trained model checkpoint (.pt). "
                             "Saves a dict with keys: model_state, args, class_names, bvp_dim. "
                             "Use emognition/bimamba_ssl/inference.py to load and predict.")

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
    print("Step 2 — Loading baselines...")
    t0 = time.time()

    if args.norm_mode == 'zscore':
        # Load RAW EEG time-series from baseline JSON to get proper μ/σ
        baselines_raw  = load_baselines_raw(args.data_root)
        baseline_info  = baselines_raw   # already {subj: {'μ': (4,), 'σ': (4,)}}
        n_covered      = sum(1 for v in baseline_info.values() if v is not None)
    else:
        baselines_raw  = load_baselines_processed(args.data_root, fs=FS)
        baseline_info  = baselines_raw
        n_covered      = sum(1 for s in set(subject_ids) if s in baseline_info)

    n_total = len(set(subject_ids))
    print(f"  norm_mode   : {args.norm_mode}")
    print(f"  Coverage    : {n_covered}/{n_total} subjects have a baseline")
    print(f"  Done in {time.time() - t0:.1f}s\n")

    # ── 2b. Euclidean Alignment (optional) ─────────────────────────────────
    if args.use_ea:
        print("Step 2b — Euclidean Alignment (per-subject covariance re-centring)...")
        t0 = time.time()
        trials = euclidean_align_subjects(trials, subject_ids)
        print(f"  Done in {time.time() - t0:.1f}s\n")

    # ── 3. Pre-process trials ────────────────────────────────────────────────
    print(f"Step 3 — Pre-processing (clip → {args.norm_mode} → band-stack)...")
    t0 = time.time()
    processed_trials = []
    for i, (trial, subj) in enumerate(zip(trials, subject_ids)):
        binfo = baseline_info.get(subj, None)
        proc  = process_trial(trial, binfo, fs=FS, norm_mode=args.norm_mode)
        processed_trials.append(proc)
        if (i + 1) % 20 == 0 or (i + 1) == len(trials):
            print(f"  {i + 1}/{len(trials)} trials processed...", end="\r")
    print(f"\n  Done in {time.time() - t0:.1f}s\n"
          f"  Output shape per trial: (20, T_i) "
          f"[5 bands × 4 channels = {IN_CHANNELS} channels]")

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

        tr_wins, tr_wlbls, tr_wsubs, tr_cids = window_trials(
            tr_proc, tr_lbl, tr_sub, window_size, step_tr)
        va_wins, va_wlbls, va_wsubs, va_cids = window_trials(
            va_proc, va_lbl, va_sub, window_size, step_ev)
        te_wins, te_wlbls, te_wsubs, te_cids = window_trials(
            te_proc, te_lbl, te_sub, window_size, step_ev)

        # Count windows per trial for BVP replication
        def count_wins(proc, step):
            return [len(list(range(0, max(t.shape[1]-window_size+1,1), step)))
                    for t in proc]

        tr_bvp = get_bvp_per_window(tr_sub, tr_emot, count_wins(tr_proc, step_tr))
        va_bvp = get_bvp_per_window(va_sub, va_emot, count_wins(va_proc, step_ev))
        te_bvp = get_bvp_per_window(te_sub, te_emot, count_wins(te_proc, step_ev))

        if fold_name:
            n_clips = len(set(te_cids))
            print(f"  Windows: tr={len(tr_wins)}, va={len(va_wins)}, te={len(te_wins)} "
                  f"({n_clips} test clips)")

        tr_ds = EmognitionMBDataset(tr_wins, tr_wlbls, tr_bvp, tr_cids, augment=True)
        va_ds = EmognitionMBDataset(va_wins, va_wlbls, va_bvp, va_cids, augment=False)
        te_ds = EmognitionMBDataset(te_wins, te_wlbls, te_bvp, te_cids, augment=False)

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
        # SWA: accumulate averaged weights
        swa_state   = None
        swa_count   = 0
        swa_active  = (args.swa_start > 0)

        for epoch in range(1, args.epochs + 1):
            fold_model.train()
            ep_loss = ep_n = ep_ok = ep_tot = 0

            for batch in tr_dl:
                if use_bvp and len(batch) == 4:   # (eeg, bvp, label, clip_id)
                    bx, bb, by, _ = batch
                    bb = bb.to(device)
                else:                              # (eeg, label, clip_id)
                    bx, by, _ = batch[0], batch[1], batch[2]
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

            # SWA: accumulate running average after swa_start
            if swa_active and epoch >= args.swa_start:
                swa_count += 1
                curr_state = fold_model.state_dict()
                if swa_state is None:
                    swa_state = {k: v.cpu().float().clone() for k, v in curr_state.items()}
                else:
                    for k in swa_state:
                        swa_state[k] += (curr_state[k].cpu().float() - swa_state[k]) / swa_count

            ret = evaluate(fold_model, va_dl, device, eval_crit, use_bvp)
            _, va_acc, va_f1, va_clip_acc, va_clip_f1, _, _, _, _ = ret
            # Use clip-level F1 for early stopping — aligns with the key test metric
            monitor_f1 = va_clip_f1

            if epoch % 10 == 0 or epoch == 1:
                tr_acc = ep_ok / max(ep_tot, 1)
                swa_tag = f" SWA×{swa_count}" if swa_count > 0 else ""
                print(f"  {fold_name} Ep{epoch:3d} | "
                      f"Tr:{tr_acc:.3f} | "
                      f"Va-win:{va_acc:.3f} Va-clip:{va_clip_acc:.3f} F1:{va_clip_f1:.3f} | "
                      f"lr:{sched.get_last_lr()[0]:.1e}{swa_tag}")

            if monitor_f1 > best_f1:
                best_f1 = monitor_f1
                best_st = {k: v.cpu().clone() for k, v in fold_model.state_dict().items()}
                pat_ctr = 0
            else:
                pat_ctr += 1
                if args.patience > 0 and pat_ctr >= args.patience:
                    print(f"  Early stop at epoch {epoch}")
                    break

        # Use SWA weights if available (better generalization), else best-val
        if swa_state is not None and swa_count >= 3:
            print(f"  Using SWA weights ({swa_count} epochs averaged)")
            ref = fold_model.state_dict()
            swa_state_cast = {k: swa_state[k].to(ref[k].dtype) for k in swa_state}
            fold_model.load_state_dict(swa_state_cast)
        elif best_st:
            fold_model.load_state_dict(best_st)
        fold_model = fold_model.to(device)

        # Save model checkpoint if requested (sub_indep only — saves the single split model)
        if getattr(args, 'save_model', None) and fold_name == "":
            ckpt = {
                'model_state': {k: v.cpu() for k, v in fold_model.state_dict().items()},
                'args':        vars(args),
                'class_names': CLASS_NAMES,
                'bvp_dim':     BVP_DIM if use_bvp else 0,
                'use_bvp':     use_bvp,
                'd_model':     args.d_model,
                'n_layers':    args.n_layers,
            }
            os.makedirs(os.path.dirname(os.path.abspath(args.save_model)), exist_ok=True)
            torch.save(ckpt, args.save_model)
            print(f"  [checkpoint] Model saved → {args.save_model}")

        ret = evaluate(fold_model, te_dl, device, eval_crit, use_bvp)
        _, te_win_acc, te_win_f1, te_clip_acc, te_clip_f1, \
            te_win_preds, te_win_lbls, te_clip_preds, te_clip_lbls = ret
        return (te_win_acc, te_win_f1, te_win_preds, te_win_lbls,
                te_clip_acc, te_clip_f1, te_clip_preds, te_clip_lbls)

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
        all_w_preds, all_w_true = [], []
        all_c_preds, all_c_true = [], []
        fold_accs = []

        for fi, test_subj in enumerate(unique_subjs):
            print(f"\n{'='*70}")
            print(f"  FOLD {fi+1}/{len(unique_subjs)}  —  Test subject: {test_subj}")
            print(f"{'='*70}")
            setup_seed(args.seed + fi)

            te_idx = [i for i in range(len(labels)) if subject_ids[i] == test_subj]
            tr_all = [i for i in range(len(labels)) if subject_ids[i] != test_subj]

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

            (w_acc, w_f1, w_preds, w_true,
             c_acc, c_f1, c_preds, c_true) = run_one_split(
                ta, tl, ts, te_, va, vl, vs, ve, xa, xl, xs, xe,
                fold_name=f"Fold{fi+1}")

            print(f"  → Fold {fi+1} ({test_subj}): "
                  f"Win-Acc={w_acc:.4f}  Clip-Acc={c_acc:.4f}  Clip-F1={c_f1:.4f}")
            all_w_preds.extend(w_preds); all_w_true.extend(w_true)
            all_c_preds.extend(c_preds); all_c_true.extend(c_true)
            fold_accs.append(c_acc)

        loso_win_acc = np.mean(np.array(all_w_preds) == np.array(all_w_true))
        loso_clip_acc = np.mean(np.array(all_c_preds) == np.array(all_c_true))
        loso_clip_f1  = f1_score(all_c_true, all_c_preds, average='macro', zero_division=0)
        print(f"\n{'='*70}")
        print(f"  LOSO FINAL — {'Multimodal EEG+BVP' if use_bvp else 'EEG-only'}")
        print(f"{'='*70}")
        print(f"  Window Acc : {loso_win_acc:.4f}  ({loso_win_acc*100:.1f}%)")
        print(f"  Clip Acc   : {loso_clip_acc:.4f}  ({loso_clip_acc*100:.1f}%)  ← KEY METRIC")
        print(f"  Clip F1    : {loso_clip_f1:.4f}")
        print(f"  Per-fold   : mean={np.mean(fold_accs):.4f} ± {np.std(fold_accs):.4f}")
        print(f"  Chance     : {100/NUM_CLASSES:.1f}%")
        print_report(all_c_true, all_c_preds,
                     title=f"LOSO Clip-Level {'EEG+BVP' if use_bvp else 'EEG-only'}")

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

        (w_acc, w_f1, w_preds, w_true,
         c_acc, c_f1, c_preds, c_true) = run_one_split(
            ta, tl, ts, te_, va, vl, vs, ve, xa, xl, xs, xe, fold_name="")

        print(f"\n{'='*70}")
        print(f"  RESULTS — {'EEG+BVP' if use_bvp else 'EEG-only'} / sub_indep")
        print(f"{'='*70}")
        print(f"  Window Acc : {w_acc:.4f}  ({w_acc*100:.1f}%)")
        print(f"  Clip   Acc : {c_acc:.4f}  ({c_acc*100:.1f}%)  ← KEY METRIC")
        print(f"  Clip   F1  : {c_f1:.4f}")
        print(f"  Chance     : {100/NUM_CLASSES:.1f}%")
        print_report(c_true, c_preds, title="sub_indep Clip-Level")


if __name__ == "__main__":
    main()


