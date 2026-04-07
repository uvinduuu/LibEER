"""
Emognition Processed Dataset Loader for Mamba Classifier.

Handles the NEW preprocessed dataset structure:
    <root>/
        <subject>/
            <subject>_<EMOTION>_STIMULUS_MUSE_cleaned/
                <subject>_<EMOTION>_STIMULUS_MUSE_cleaned.json

Returns full-length raw EEG trials (C, T) — windowing is done separately
by the training script (or not at all for full-clip mode).

Sampling rate: 256 Hz (Muse 2)
Channels: RAW_TP9, RAW_AF7, RAW_AF8, RAW_TP10 (4 channels)
"""

import os
import glob
import json
import numpy as np
import pandas as pd


# ── Constants ──────────────────────────────────────────────────────────────
FS = 256   # Muse 2 sampling rate (Hz)
CHANNELS = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
QUALITY_CHANNELS = ["HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10"]

# Default 4-class setup matching your experiment
EMOTIONS_4CLASS = ["ENTHUSIASM", "FEAR", "SADNESS", "NEUTRAL"]


# ── Internal helpers ────────────────────────────────────────────────────────

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


def _parse_filename(filepath):
    """
    Extract (subject, emotion) from a _cleaned filename.

    New format: {subject}_{EMOTION}_STIMULUS_MUSE_cleaned.json
    Example:    22_ENTHUSIASM_STIMULUS_MUSE_cleaned.json
                → subject='22', emotion='ENTHUSIASM'
    """
    name = os.path.splitext(os.path.basename(filepath))[0]   # strip .json
    parts = name.split("_")
    # parts[0] = subject, parts[1] = emotion
    if len(parts) < 2:
        return None, None
    subject = parts[0]
    emotion = parts[1].upper()
    return subject, emotion


def _load_one_trial(filepath):
    """
    Load one cleaned Emognition JSON → full raw EEG trial.

    Returns:
        trial:   (4, T) float32 — EEG channels first — or None on failure
        emotion: str, e.g. 'FEAR'
        subject: str, e.g. '22'
    """
    subject, emotion = _parse_filename(filepath)
    if subject is None:
        return None, None, None

    try:
        with open(filepath, "r") as f:
            obj = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  [warn] Cannot read {filepath}: {e}")
        return None, emotion, subject

    # ── Read raw channels ──
    raw = {}
    for ch in CHANNELS:
        raw[ch] = _interp_nan(_to_num(obj.get(ch, [])))

    L = min(len(raw[ch]) for ch in CHANNELS)
    if L == 0:
        return None, emotion, subject

    for ch in CHANNELS:
        raw[ch] = raw[ch][:L]

    # ── Quality mask: finite + headband on + HSI ≤ 2 ──
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
    if L < FS:  # less than 1 second — skip
        return None, emotion, subject

    # ── Stack channels + DC removal ──
    trial = np.stack([raw[ch][:L] for ch in CHANNELS], axis=0).astype(np.float32)
    trial = trial - trial.mean(axis=1, keepdims=True)

    return trial, emotion, subject


# ── Public API ──────────────────────────────────────────────────────────────

def find_processed_files(data_root):
    """
    Find all *_STIMULUS_MUSE_cleaned.json files under data_root.

    Handles both:
      - Nested:  <root>/<subject>/<subject>_<EMOTION>_STIMULUS_MUSE_cleaned/<file>.json
      - Flat:    <root>/**/*_STIMULUS_MUSE_cleaned.json

    Returns sorted list of absolute file paths.
    """
    patterns = [
        os.path.join(data_root, "*_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "*", "*_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "*", "*_STIMULUS_MUSE_cleaned", "*_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "**", "*_STIMULUS_MUSE_cleaned.json"),
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})
    return files


def load_emognition_processed(
    data_root,
    emotions=None,
    min_trial_sec=2.0,
    verbose=True,
):
    """
    Load the new preprocessed Emognition dataset as full-length raw EEG trials.

    Args:
        data_root:      Path to dataset root (folder containing subject subdirs)
        emotions:       List of emotion labels to use (default: 4-class)
        min_trial_sec:  Minimum trial length in seconds (shorter trials skipped)
        verbose:        Print progress/statistics

    Returns:
        trials:      list of numpy arrays, each (4, T_i) float32
        labels:      list of int labels (0-indexed, sorted alphabetically)
        subject_ids: list of str subject IDs
        lab2id:      dict  emotion_str → int
        id2lab:      dict  int → emotion_str
    """
    if emotions is None:
        emotions = EMOTIONS_4CLASS

    # Alphabetical label mapping (matches SEED-IV style)
    lab2id = {e: i for i, e in enumerate(sorted(emotions))}
    id2lab = {v: k for k, v in lab2id.items()}
    valid_emotions = set(emotions)
    min_samples = int(min_trial_sec * FS)

    files = find_processed_files(data_root)
    if verbose:
        print(f"  Found {len(files)} *_STIMULUS_MUSE_cleaned.json files under '{data_root}'")

    trials, labels, subject_ids = [], [], []
    skipped_quality = 0
    skipped_emotion = 0

    for fp in files:
        _, emotion, _ = _parse_filename(fp)
        if emotion not in valid_emotions:
            skipped_emotion += 1
            continue

        trial, emotion_str, subject = _load_one_trial(fp)

        if trial is None or trial.shape[1] < min_samples:
            skipped_quality += 1
            continue

        if emotion_str not in lab2id:
            skipped_emotion += 1
            continue

        trials.append(trial)
        labels.append(lab2id[emotion_str])
        subject_ids.append(subject)

    if verbose:
        print(f"  Loaded {len(trials)} trials "
              f"({skipped_quality} skipped: quality/length, "
              f"{skipped_emotion} skipped: wrong emotion class)")
        print(f"  Subjects: {len(set(subject_ids))} unique  →  {sorted(set(subject_ids))}")
        print(f"  Label map: {lab2id}")

        if trials:
            lengths = [t.shape[1] for t in trials]
            print(f"  Trial length: min={min(lengths)/FS:.1f}s  "
                  f"max={max(lengths)/FS:.1f}s  "
                  f"mean={np.mean(lengths)/FS:.1f}s  "
                  f"(at {FS}Hz)")

        from collections import Counter
        dist = Counter(labels)
        for lid in sorted(dist.keys()):
            print(f"    {id2lab[lid]:>12}: {dist[lid]:3d} trials")

    return trials, labels, subject_ids, lab2id, id2lab


# ── Optional standalone test ────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True,
                        help="Path to the Emognition Processed dataset root")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"Testing emognition_processed_loader.py")
    print(f"  data_root: {args.data_root}")
    print(f"{'='*60}\n")

    trials, labels, subject_ids, lab2id, id2lab = load_emognition_processed(
        args.data_root, verbose=True
    )

    if trials:
        # Show first trial's JSON keys to verify channel names
        files = find_processed_files(args.data_root)
        if files:
            with open(files[0]) as f:
                obj = json.load(f)
            avail_keys = [k for k in CHANNELS if k in obj]
            missing_keys = [k for k in CHANNELS if k not in obj]
            print(f"\n  Channel check in '{os.path.basename(files[0])}':")
            print(f"    Found   : {avail_keys}")
            print(f"    Missing : {missing_keys}")
