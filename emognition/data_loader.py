"""
Emognition dataset loader for dual-branch EEGNet-BiLSTM.

Loads Muse 2 JSON files, extracts raw EEG windows AND
precomputes handcrafted features for the fusion model.
"""

import os
import glob
import json
import numpy as np
import pandas as pd
from collections import Counter

import torch
from torch.utils.data import TensorDataset, DataLoader, WeightedRandomSampler

from config import (
    DATA_ROOT, FS, CHANNELS, QUALITY_CHANNELS,
    WIN_SEC, OVERLAP, SAMPLE_LENGTH, NUM_CHANNELS,
    LABEL_MODE, EMOTIONS_USED,
    EXPERIMENT_MODE, TEST_SIZE, VAL_SIZE, SEED, BATCH_SIZE,
    USE_INVBASE
)
from features import extract_handcrafted_features


# ===================== HELPERS =====================

def _to_num(x):
    """Convert JSON field to float64 numpy array."""
    if isinstance(x, list):
        if not x:
            return np.array([], np.float64)
        if isinstance(x[0], str):
            return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(np.float64)
        return np.asarray(x, np.float64)
    return np.asarray([x], np.float64)


def _interp_nan(a):
    """Interpolate NaN values in a 1D array."""
    a = a.astype(np.float64, copy=True)
    m = np.isfinite(a)
    if m.all():
        return a
    if not m.any():
        return np.zeros_like(a)
    idx = np.arange(len(a))
    a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a


def _make_windows(sig_2d):
    """Slice (T, 4) signal into overlapping windows → (n_win, win_samples, 4)."""
    win = SAMPLE_LENGTH
    step = int(round(win * (1.0 - OVERLAP)))
    windows = []
    for s in range(0, max(0, len(sig_2d) - win + 1), step):
        chunk = sig_2d[s:s + win]
        if len(chunk) == win:
            windows.append(chunk)
    if windows:
        return np.stack(windows).astype(np.float32)
    return np.zeros((0, win, sig_2d.shape[1]), np.float32)


def _load_one_file(filepath):
    """Load one Emognition JSON file → windowed raw EEG."""
    name = os.path.basename(filepath)
    parts = name.split("_")
    subject = parts[0]
    label = parts[1].upper()

    with open(filepath, "r") as f:
        obj = json.load(f)

    raw = {}
    for ch in CHANNELS:
        raw[ch] = _interp_nan(_to_num(obj.get(ch, [])))

    L = min(len(raw[ch]) for ch in CHANNELS)
    if L == 0:
        return np.zeros((0, SAMPLE_LENGTH, NUM_CHANNELS), np.float32), label, subject

    for ch in CHANNELS:
        raw[ch] = raw[ch][:L]

    # Quality mask
    mask = np.ones(L, dtype=bool)
    for ch in CHANNELS:
        mask &= np.isfinite(raw[ch])

    head_on = _to_num(obj.get("HeadBandOn", []))[:L]
    if len(head_on) == L:
        mask &= (head_on == 1)
        for qch in QUALITY_CHANNELS:
            hsi = _to_num(obj.get(qch, []))[:L]
            if len(hsi) == L:
                mask &= np.isfinite(hsi) & (hsi <= 2)

    for ch in CHANNELS:
        raw[ch] = raw[ch][mask]

    L = min(len(raw[ch]) for ch in CHANNELS)
    if L < SAMPLE_LENGTH:
        return np.zeros((0, SAMPLE_LENGTH, NUM_CHANNELS), np.float32), label, subject

    sig = np.stack([raw[ch][:L] for ch in CHANNELS], axis=1)  # (T, 4)
    sig = sig - np.nanmean(sig, axis=0, keepdims=True)

    X = _make_windows(sig)  # (n_win, sample_length, 4)
    return X, label, subject


# ===================== DATASET BUILDER =====================

def _get_label_map():
    """Build label string -> integer mapping from EMOTIONS_USED."""
    classes = sorted(EMOTIONS_USED)
    return {c: i for i, c in enumerate(classes)}


def load_emognition():
    """
    Load the full Emognition dataset with both raw EEG and handcrafted features.

    Returns:
        X_raw:      (N, 4, sample_length) float32 — channels first
        X_features: (N, 4, 26) float32 — handcrafted features
        y:          (N,) int64 labels
        subjects:   (N,) str subject IDs
        lab2id:     dict label_name -> int
        id2lab:     dict int -> label_name
    """
    patterns = [
        os.path.join(DATA_ROOT, "*_STIMULUS_MUSE.json"),
        os.path.join(DATA_ROOT, "*", "*_STIMULUS_MUSE.json"),
        os.path.join(DATA_ROOT, "**", "*_STIMULUS_MUSE.json"),
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})
    print(f"[DataLoader] Found {len(files)} STIMULUS_MUSE files")

    lab2id = _get_label_map()
    id2lab = {v: k for k, v in lab2id.items()}
    valid_emotions = set(EMOTIONS_USED)

    all_X, all_y, all_subj = [], [], []

    for fp in files:
        name = os.path.basename(fp)
        emotion = name.split("_")[1].upper()
        if emotion not in valid_emotions:
            continue

        X, label, subject = _load_one_file(fp)
        if len(X) == 0:
            continue

        if label not in lab2id:
            continue

        label_int = lab2id[label]
        n = len(X)

        all_X.append(X)
        all_y.append(np.full(n, label_int, dtype=np.int64))
        all_subj.append(np.array([subject] * n))

    X_raw = np.concatenate(all_X, axis=0)            # (N, sample_length, 4)
    y = np.concatenate(all_y, axis=0)
    subjects = np.concatenate(all_subj, axis=0)

    # Transpose to channels first: (N, 4, sample_length)
    X_raw = X_raw.transpose(0, 2, 1)

    # Extract handcrafted features: (N, 4, 26)
    print("[DataLoader] Extracting handcrafted features (26 per channel)...")
    X_features = extract_handcrafted_features(X_raw)

    # Optionally add InvBase features
    if USE_INVBASE:
        from invbase import load_baselines, extract_invbase_features
        baselines = load_baselines(DATA_ROOT)
        X_invbase = extract_invbase_features(X_raw, subjects, baselines)
        # Concatenate: (N, 4, 26) + (N, 4, 10) → (N, 4, 36)
        X_features = np.concatenate([X_features, X_invbase], axis=2)
        print(f"[DataLoader] InvBase features added: {X_invbase.shape[2]} per channel")

    print(f"[DataLoader] Total windows: {len(X_raw)}")
    print(f"[DataLoader] Raw shape: {X_raw.shape} | Features shape: {X_features.shape}")
    print(f"[DataLoader] Classes: {lab2id}")
    print(f"[DataLoader] Label distribution:")
    for lid, cnt in sorted(Counter(y).items()):
        print(f"    {id2lab[lid]}: {cnt}")

    return X_raw, X_features, y, subjects, lab2id, id2lab


# ===================== SPLIT =====================

def split_data(X_raw, X_features, y, subjects):
    """Split data into train/val/test with both raw and features."""
    np.random.seed(SEED)

    if EXPERIMENT_MODE == "subject-independent":
        unique_subj = np.unique(subjects)
        np.random.shuffle(unique_subj)
        n = len(unique_subj)
        n_test = max(1, int(n * TEST_SIZE))
        n_val = max(1, int(n * VAL_SIZE))

        test_subj = set(unique_subj[:n_test])
        val_subj = set(unique_subj[n_test:n_test + n_val])
        train_subj = set(unique_subj[n_test + n_val:])

        train_idx = np.array([i for i, s in enumerate(subjects) if s in train_subj])
        val_idx = np.array([i for i, s in enumerate(subjects) if s in val_subj])
        test_idx = np.array([i for i, s in enumerate(subjects) if s in test_subj])

        print(f"[Split] Subject-independent: {len(train_subj)} train / "
              f"{len(val_subj)} val / {len(test_subj)} test subjects")
    else:
        indices = np.arange(len(y))
        np.random.shuffle(indices)
        n = len(indices)
        n_test = int(n * TEST_SIZE)
        n_val = int(n * VAL_SIZE)

        test_idx = indices[:n_test]
        val_idx = indices[n_test:n_test + n_val]
        train_idx = indices[n_test + n_val:]

        print(f"[Split] Subject-dependent: {len(train_idx)} train / "
              f"{len(val_idx)} val / {len(test_idx)} test windows")

    # Standardize handcrafted features using training set statistics
    Xf_tr = X_features[train_idx]
    mu_f = Xf_tr.mean(axis=0, keepdims=True)  # (1, 4, 26)
    sd_f = Xf_tr.std(axis=0, keepdims=True) + 1e-6

    Xf_tr = (Xf_tr - mu_f) / sd_f
    Xf_va = (X_features[val_idx] - mu_f) / sd_f
    Xf_te = (X_features[test_idx] - mu_f) / sd_f

    print(f"[Split] Handcrafted features standardized (train set stats)")

    return (X_raw[train_idx], Xf_tr, y[train_idx],
            X_raw[val_idx],   Xf_va, y[val_idx],
            X_raw[test_idx],  Xf_te, y[test_idx])


# ===================== DATALOADERS =====================

def make_loaders(Xr_tr, Xf_tr, y_tr, Xr_va, Xf_va, y_va, Xr_te, Xf_te, y_te):
    """Create DataLoaders returning (x_raw, x_features, y) triples."""

    n_classes = len(np.unique(y_tr))
    class_counts = np.bincount(y_tr, minlength=n_classes).astype(np.float32)
    weights_per_class = 1.0 / np.clip(class_counts, 1.0, None)
    sample_weights = weights_per_class[y_tr]
    sampler = WeightedRandomSampler(
        torch.from_numpy(sample_weights.astype(np.float32)),
        num_samples=len(sample_weights), replacement=True
    )

    train_ds = TensorDataset(
        torch.from_numpy(Xr_tr).float(),
        torch.from_numpy(Xf_tr).float(),
        torch.from_numpy(y_tr).long()
    )
    val_ds = TensorDataset(
        torch.from_numpy(Xr_va).float(),
        torch.from_numpy(Xf_va).float(),
        torch.from_numpy(y_va).long()
    )
    test_ds = TensorDataset(
        torch.from_numpy(Xr_te).float(),
        torch.from_numpy(Xf_te).float(),
        torch.from_numpy(y_te).long()
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, sampler=sampler,
                              num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=256, shuffle=False,
                            num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=256, shuffle=False,
                             num_workers=2, pin_memory=True)

    return train_loader, val_loader, test_loader
