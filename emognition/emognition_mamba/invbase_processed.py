"""
InvBase Baseline Removal for Preprocessed Emognition Dataset.

Loads *_BASELINE_STIMULUS_MUSE_cleaned.json files (same _cleaned structure
as emotion trials) and computes per-subject baseline DE-LDS features that
are then used to normalise emotion trial DE features.

How it works (in log/DE space):
  DE_normalised[t, ch, band] = DE_emotion[t, ch, band] - DE_baseline_mean[ch, band]

  Since DE = 0.5 * log(2πe * σ²), subtraction = log division:
    → equivalent to dividing emotion spectral power by baseline spectral power
    → removes subject-specific resting-state brain rhythms
    → leaves only emotion-driven spectral changes

Why this helps for cross-subject generalisation:
  Subject A resting alpha:  high    Subject B resting alpha: low
  Subject A fear alpha:     medium  Subject B fear alpha:    very low
  Raw DE:      A_fear ≠ B_fear     (different brains)
  InvBase DE:  A_fear - A_rest ≈ B_fear - B_rest  (same emotion response)

Usage:
  from invbase_processed import load_invbase_baselines, apply_invbase

  baselines = load_invbase_baselines(data_root, delds_win_sec=1.0)
  normalised_feats = apply_invbase(features, subject_ids, baselines)
"""

import os
import glob
import json
import numpy as np
import pandas as pd

# ── Constants (same as emognition_processed_loader) ────────────────────────
FS              = 256
CHANNELS        = ["RAW_TP9", "RAW_AF7", "RAW_AF8", "RAW_TP10"]
QUALITY_CHANNELS = ["HSI_TP9", "HSI_AF7", "HSI_AF8", "HSI_TP10"]


# ── Private helpers (copy-free, no cross-import dependency) ────────────────

def _to_num(x):
    if isinstance(x, list):
        if not x:
            return np.array([], np.float64)
        if isinstance(x[0], str):
            return pd.to_numeric(pd.Series(x), errors="coerce").to_numpy(np.float64)
        return np.asarray(x, np.float64)
    return np.asarray([x], np.float64)


def _interp_nan(a):
    a = a.astype(np.float64, copy=True)
    m = np.isfinite(a)
    if m.all():
        return a
    if not m.any():
        return np.zeros_like(a)
    idx = np.arange(len(a))
    a[~m] = np.interp(idx[~m], idx[m], a[m])
    return a


def _load_raw_trial(filepath):
    """Load a single JSON → (4, T) float32 or None."""
    try:
        with open(filepath, "r") as f:
            obj = json.load(f)
    except (json.JSONDecodeError, OSError):
        return None

    raw = {ch: _interp_nan(_to_num(obj.get(ch, []))) for ch in CHANNELS}
    L = min(len(raw[ch]) for ch in CHANNELS)
    if L == 0:
        return None

    for ch in CHANNELS:
        raw[ch] = raw[ch][:L]

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
    if L < FS:   # < 1 second
        return None

    trial = np.stack([raw[ch][:L] for ch in CHANNELS], axis=0).astype(np.float32)
    trial = trial - trial.mean(axis=1, keepdims=True)   # DC removal
    return trial


# ── Public API ─────────────────────────────────────────────────────────────

def find_baseline_files(data_root):
    """
    Find all *_BASELINE_STIMULUS_MUSE_cleaned.json files.

    Handles both flat and nested _cleaned folder structures:
      <root>/<subject>/<subject>_BASELINE_STIMULUS_MUSE_cleaned/
          <subject>_BASELINE_STIMULUS_MUSE_cleaned.json
    """
    patterns = [
        os.path.join(data_root, "*_BASELINE_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "*", "*_BASELINE_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "*", "*_BASELINE_STIMULUS_MUSE_cleaned",
                     "*_BASELINE_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "**", "*_BASELINE_STIMULUS_MUSE_cleaned.json"),
    ]
    return sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})


def load_invbase_baselines(data_root, delds_win_sec=1.0, verbose=True):
    """
    Load and compute per-subject baseline DE-LDS features.

    The baseline DE is the MEAN across all time windows — a (4, 5) matrix
    representing the subject's resting-state spectral profile.

    Args:
        data_root:     Path to dataset root
        delds_win_sec: Same window size as used for emotion DE features
        verbose:       Print progress

    Returns:
        baselines: dict { subject_id (str) → np.ndarray (4, 5) float32 }
                   Returns empty dict if no baseline files found.
    """
    # Import here to keep this module dependency-light
    from delds_extractor import compute_delds

    files = find_baseline_files(data_root)

    if verbose:
        print(f"[InvBase] Found {len(files)} BASELINE_STIMULUS_MUSE_cleaned files")

    if not files:
        if verbose:
            print("[InvBase] WARNING: No baseline files found — InvBase disabled.")
            print("          Expected pattern: *_BASELINE_STIMULUS_MUSE_cleaned.json")
        return {}

    baselines = {}
    skipped   = 0

    for fp in files:
        name      = os.path.splitext(os.path.basename(fp))[0]
        parts     = name.split("_")
        subject   = parts[0]

        raw = _load_raw_trial(fp)
        if raw is None:
            skipped += 1
            continue

        # Compute DE-LDS on the baseline raw EEG: (T_baseline, 4, 5)
        de = compute_delds(raw, fs=FS,
                           window_sec=delds_win_sec,
                           step_sec=delds_win_sec)

        if de.shape[0] == 0:
            skipped += 1
            continue

        # Baseline DE = mean across time → (4, 5)
        baselines[subject] = de.mean(axis=0).astype(np.float32)

    if verbose:
        print(f"[InvBase] Loaded baselines for {len(baselines)} subjects "
              f"({skipped} skipped: too short / unreadable)")
        if baselines:
            covered = list(baselines.keys())
            print(f"[InvBase] Subjects with baseline: {sorted(covered)}")

    return baselines


def apply_invbase(features, subject_ids, baselines, verbose=True):
    """
    Subtract each subject's resting-state DE from their emotion DE features.

    In log space: subtraction = spectral ratio → removes subject brainprint.

    Args:
        features:    list of N arrays, each (T_i, 4, 5) float32
        subject_ids: list of N subject ID strings matching features
        baselines:   dict { subject_id → (4, 5) baseline DE mean }
        verbose:     Print coverage stats

    Returns:
        normalised: list of N arrays, each (T_i, 4, 5) float32
                    Trials whose subject has no baseline are returned unchanged.
    """
    normalised  = []
    n_applied   = 0
    n_fallback  = 0

    for feat, subj in zip(features, subject_ids):
        if subj in baselines:
            # feat: (T, 4, 5)   baselines[subj]: (4, 5)
            # Broadcast subtraction over time axis
            normalised.append(feat - baselines[subj][np.newaxis, :, :])
            n_applied += 1
        else:
            normalised.append(feat.copy())
            n_fallback += 1

    if verbose:
        total = n_applied + n_fallback
        print(f"[InvBase] Applied to {n_applied}/{total} trials "
              f"({n_fallback} trials had no baseline → used raw DE)")

    return normalised


# ── Standalone test ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    args = parser.parse_args()

    print(f"\n{'='*55}")
    print(f"InvBase Baseline Check — Processed Emognition")
    print(f"  data_root: {args.data_root}")
    print(f"{'='*55}\n")

    baselines = load_invbase_baselines(args.data_root, verbose=True)

    if baselines:
        sample_subj = sorted(baselines.keys())[0]
        print(f"\n  Sample baseline DE for subject {sample_subj}:")
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        chs   = ['TP9', 'AF7', 'AF8', 'TP10']
        b     = baselines[sample_subj]   # (4, 5)
        print(f"  {'CH':>6}  " + "  ".join(f"{bd:>7}" for bd in bands))
        for i, ch in enumerate(chs):
            vals = "  ".join(f"{b[i, j]:>7.3f}" for j in range(5))
            print(f"  {ch:>6}  {vals}")
    else:
        print("\n  No baselines found. InvBase cannot be applied.")
        print("  Check that BASELINE files exist in the dataset.")
