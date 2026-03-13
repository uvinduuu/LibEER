"""
Emognition Raw Trial Loader for Mamba Classifier.

Loads Muse 2 JSON files and returns full-length raw EEG trials (not windowed).
Windowing is handled separately by the training script.

Data format:
    Each JSON file = one trial (~60s) for one subject + one emotion.
    Filename: {subject}_{emotion}_STIMULUS_MUSE.json
    Channels: RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10 (4 channels at 256Hz)
    Quality:  HSI_TP9, HSI_AF7, HSI_AF8, HSI_TP10 + HeadBandOn
"""

import os
import glob
import json
import numpy as np
import pandas as pd


# ── Constants ──
FS = 256  # Muse 2 sampling rate
CHANNELS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
QUALITY_CHANNELS = ["HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10"]
EMOTIONS_4CLASS = ["ENTHUSIASM", "FEAR", "SADNESS", "NEUTRAL"]


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


def _load_one_trial(filepath):
    """
    Load one Emognition JSON file → full raw EEG trial.

    Returns:
        trial: (4, T) float32 — full-length cleaned EEG
        emotion: str — e.g. "FEAR"
        subject: str — e.g. "01"
    """
    name = os.path.basename(filepath)
    parts = name.split("_")
    subject = parts[0]
    emotion = parts[1].upper()

    with open(filepath, "r") as f:
        obj = json.load(f)

    # Read raw channels
    raw = {}
    for ch in CHANNELS:
        raw[ch] = _interp_nan(_to_num(obj.get(ch, [])))

    L = min(len(raw[ch]) for ch in CHANNELS)
    if L == 0:
        return None, emotion, subject

    for ch in CHANNELS:
        raw[ch] = raw[ch][:L]

    # Quality mask: finite values + headband on + HSI quality
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

    # Apply mask
    for ch in CHANNELS:
        raw[ch] = raw[ch][mask]

    L = min(len(raw[ch]) for ch in CHANNELS)
    if L < FS:  # Less than 1 second of data → skip
        return None, emotion, subject

    # Stack channels: (4, T) and remove DC offset
    trial = np.stack([raw[ch][:L] for ch in CHANNELS], axis=0).astype(np.float32)
    trial = trial - trial.mean(axis=1, keepdims=True)  # DC removal per channel

    return trial, emotion, subject


def load_emognition_trials(data_root, emotions=None, min_trial_sec=2.0):
    """
    Load all Emognition trials as full-length raw EEG.

    Args:
        data_root: Path to Emognition dataset root
        emotions: List of emotion labels to include (default: 4-class)
        min_trial_sec: Minimum trial duration in seconds to keep

    Returns:
        trials: list of numpy arrays, each (4, T_i)
        labels: list of int labels (0-indexed)
        subject_ids: list of str subject IDs
        lab2id: dict emotion → int
        id2lab: dict int → emotion
    """
    if emotions is None:
        emotions = EMOTIONS_4CLASS

    # Build label map
    lab2id = {e: i for i, e in enumerate(sorted(emotions))}
    id2lab = {v: k for k, v in lab2id.items()}

    # Find files
    patterns = [
        os.path.join(data_root, "*_STIMULUS_MUSE.json"),
        os.path.join(data_root, "*", "*_STIMULUS_MUSE.json"),
        os.path.join(data_root, "**", "*_STIMULUS_MUSE.json"),
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})
    print(f"  Found {len(files)} STIMULUS_MUSE files")

    valid_emotions = set(emotions)
    min_samples = int(min_trial_sec * FS)

    trials = []
    labels = []
    subject_ids = []
    skipped = 0

    for fp in files:
        name = os.path.basename(fp)
        emotion = name.split("_")[1].upper()
        if emotion not in valid_emotions:
            continue

        trial, label_str, subject = _load_one_trial(fp)

        if trial is None or trial.shape[1] < min_samples:
            skipped += 1
            continue

        if label_str not in lab2id:
            continue

        trials.append(trial)
        labels.append(lab2id[label_str])
        subject_ids.append(subject)

    print(f"  Loaded {len(trials)} trials ({skipped} skipped for quality/length)")
    print(f"  Subjects: {len(set(subject_ids))} unique")
    print(f"  Classes: {lab2id}")

    # Length stats
    lengths = [t.shape[1] for t in trials]
    if lengths:
        print(f"  Trial lengths: min={min(lengths)} ({min(lengths)/FS:.1f}s), "
              f"max={max(lengths)} ({max(lengths)/FS:.1f}s), "
              f"mean={np.mean(lengths):.0f} ({np.mean(lengths)/FS:.1f}s)")

    # Label distribution
    from collections import Counter
    dist = Counter(labels)
    for lid in sorted(dist.keys()):
        print(f"    {id2lab[lid]}: {dist[lid]} trials")

    return trials, labels, subject_ids, lab2id, id2lab
