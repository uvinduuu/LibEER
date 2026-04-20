"""
Samsung Watch BVP Data Sanity Check
====================================
Scans the raw Emognition dataset for *_STIMULUS_SAMSUNG_WATCH.json files,
inspects what BVP-related keys exist, checks signal lengths, and cross-references
with Muse EEG availability for the 4 target emotions.

Usage:
    python samsung_bvp_sanity_check.py --data_root /path/to/emognition

Output:
    - How many Samsung Watch files exist per subject/emotion
    - What keys are in the JSON (BVP, HR, accelerometer, etc.)
    - Signal lengths and sampling rates
    - Missing data: subjects that have EEG but no Samsung Watch (or vice versa)
    - BVP data quality (NaN rate, zero-rate)
"""

import os
import sys
import glob
import json
import argparse
from collections import defaultdict, Counter

import numpy as np
import pandas as pd


# ── Target emotions ─────────────────────────────────────────────────────────
TARGET_EMOTIONS = {'ENTHUSIASM', 'FEAR', 'NEUTRAL', 'SADNESS'}

# ── Known Samsung Watch signal keys to look for ──────────────────────────────
# Samsung Health SDK field names (may vary by firmware version)
KNOWN_BVP_KEYS = [
    'BVP', 'bvp',
    'HeartRate', 'heart_rate', 'HR',
    'IBI', 'ibi',                          # inter-beat interval
    'PPG', 'ppg',
    'GreenChannel', 'green_channel',        # raw PPG green channel
    'EDA', 'eda', 'GSR',                   # electrodermal (Empatica overlap)
    'TEMP', 'temp', 'Temperature',
    'ACC_X', 'ACC_Y', 'ACC_Z',            # accelerometer
    'GYRO_X', 'GYRO_Y', 'GYRO_Z',
]


def _to_arr(x):
    """Convert JSON field to float64 numpy array (safe)."""
    if isinstance(x, list):
        if not x:
            return np.array([], np.float64)
        try:
            if isinstance(x[0], str):
                return pd.to_numeric(pd.Series(x), errors='coerce').to_numpy(np.float64)
            return np.asarray(x, np.float64)
        except Exception:
            return np.array([], np.float64)
    if isinstance(x, (int, float)):
        return np.array([x], np.float64)
    return np.array([], np.float64)


def inspect_samsung_file(filepath):
    """
    Inspect one Samsung Watch JSON file.

    Returns a dict with:
        keys_found      : list of top-level keys
        bvp_key         : which key contains the BVP/PPG signal (or None)
        bvp_length      : number of samples
        bvp_nan_rate    : fraction NaN
        bvp_zero_rate   : fraction zeros
        all_keys        : all JSON keys
        ts_key          : timestamp key (for sampling rate estimation)
        estimated_fs    : estimated sampling rate (Hz) if timestamps present
    """
    try:
        with open(filepath, 'r') as f:
            obj = json.load(f)
    except Exception as e:
        return {'error': str(e)}

    all_keys    = list(obj.keys())
    result      = {
        'all_keys':    all_keys,
        'n_keys':      len(all_keys),
        'bvp_key':     None,
        'bvp_length':  0,
        'bvp_nan_rate':0.0,
        'bvp_zero_rate':0.0,
        'estimated_fs': None,
        'error':       None,
    }

    # Search for BVP signal
    for key in KNOWN_BVP_KEYS:
        if key in obj:
            arr = _to_arr(obj[key])
            if len(arr) > 0:
                result['bvp_key']       = key
                result['bvp_length']    = len(arr)
                result['bvp_nan_rate']  = float(np.isnan(arr).mean())
                result['bvp_zero_rate'] = float((arr == 0).mean())
                break  # use first match

    # Try to estimate FS from a timestamp array
    for ts_key in ['timestamps', 'Timestamp', 'timestamp', 'time', 'Time']:
        if ts_key in obj:
            ts_arr = _to_arr(obj[ts_key])
            if len(ts_arr) > 2:
                diffs = np.diff(ts_arr[np.isfinite(ts_arr)])
                if len(diffs) > 0:
                    median_dt = np.median(diffs[diffs > 0])
                    if median_dt > 0:
                        result['estimated_fs'] = round(1.0 / median_dt, 2)
            break

    return result


def find_samsung_files(data_root):
    """Find all *_STIMULUS_SAMSUNG_WATCH.json files."""
    patterns = [
        os.path.join(data_root, '*_STIMULUS_SAMSUNG_WATCH.json'),
        os.path.join(data_root, '*', '*_STIMULUS_SAMSUNG_WATCH.json'),
        os.path.join(data_root, '**', '*_STIMULUS_SAMSUNG_WATCH.json'),
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})
    return files


def find_muse_files(data_root):
    """Find all *_STIMULUS_MUSE.json and *_STIMULUS_MUSE_cleaned.json files."""
    patterns = [
        os.path.join(data_root, '*_STIMULUS_MUSE*.json'),
        os.path.join(data_root, '*', '*_STIMULUS_MUSE*.json'),
        os.path.join(data_root, '**', '*_STIMULUS_MUSE*.json'),
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})
    return files


def parse_filename(filepath):
    """Extract (subject, emotion) from Emognition filename."""
    name  = os.path.splitext(os.path.basename(filepath))[0]
    parts = name.split('_')
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1].upper()


def main():
    parser = argparse.ArgumentParser(
        description='Sanity check Samsung Watch BVP data in Emognition dataset'
    )
    parser.add_argument('--data_root', required=True,
                        help='Root directory of the raw Emognition dataset '
                             '(the folder containing subject subdirs like 22/, 23/ ...)')
    parser.add_argument('--inspect_sample', type=int, default=3,
                        help='Number of Samsung Watch files to fully inspect for key names')
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"Samsung Watch BVP Sanity Check — Emognition Dataset")
    print(f"  data_root: {args.data_root}")
    print(f"  Target emotions: {sorted(TARGET_EMOTIONS)}")
    print(f"{'='*65}\n")

    # ── 1. Find files ─────────────────────────────────────────────────────────
    samsung_files = find_samsung_files(args.data_root)
    muse_files    = find_muse_files(args.data_root)

    print(f"Files found:")
    print(f"  Samsung Watch  : {len(samsung_files)}")
    print(f"  Muse EEG       : {len(muse_files)}")

    if not samsung_files:
        print("\n[ERROR] No Samsung Watch files found!")
        print("  Check that --data_root points to the ROOT of the Emognition dataset")
        print("  (the folder where subject subdirs like 22/, 23/ live).")
        sys.exit(1)

    # ── 2. Parse all Samsung files ────────────────────────────────────────────
    samsung_by_subj_emot = {}   # (subject, emotion) → filepath
    samsung_skip         = []

    for fp in samsung_files:
        subj, emot = parse_filename(fp)
        if subj is None or emot not in TARGET_EMOTIONS:
            samsung_skip.append(fp)
            continue
        samsung_by_subj_emot[(subj, emot)] = fp

    samsung_subjs  = sorted(set(s for s, e in samsung_by_subj_emot.keys()))
    samsung_emots  = Counter(e for s, e in samsung_by_subj_emot.keys())

    print(f"\n  Samsung files for target emotions: {len(samsung_by_subj_emot)}")
    print(f"  Samsung subjects (target emotions): {len(samsung_subjs)} — {samsung_subjs}")
    print(f"  Per emotion:")
    for emot in sorted(samsung_emots):
        print(f"    {emot:>12}: {samsung_emots[emot]:3d} files")

    # ── 3. Parse all Muse (EEG) files ────────────────────────────────────────
    muse_by_subj_emot = {}
    for fp in muse_files:
        subj, emot = parse_filename(fp)
        if subj is None or emot not in TARGET_EMOTIONS:
            continue
        muse_by_subj_emot[(subj, emot)] = fp

    muse_subjs = sorted(set(s for s, e in muse_by_subj_emot.keys()))

    print(f"\n  Muse (EEG) files for target emotions: {len(muse_by_subj_emot)}")
    print(f"  Muse subjects: {len(muse_subjs)}")

    # ── 4. Pairing: EEG ↔ Samsung ────────────────────────────────────────────
    paired_keys   = set(samsung_by_subj_emot.keys()) & set(muse_by_subj_emot.keys())
    eeg_only_keys = set(muse_by_subj_emot.keys()) - set(samsung_by_subj_emot.keys())
    sam_only_keys = set(samsung_by_subj_emot.keys()) - set(muse_by_subj_emot.keys())

    print(f"\n  {'='*55}")
    print(f"  Pairing (EEG ↔ Samsung, target emotions only):")
    print(f"  {'='*55}")
    print(f"  Paired (both available)  : {len(paired_keys)}")
    print(f"  EEG only (no Samsung)    : {len(eeg_only_keys)}")
    print(f"  Samsung only (no EEG)    : {len(sam_only_keys)}")

    # Per-subject pair counts
    subj_pair_counts = Counter(s for s, e in paired_keys)
    print(f"\n  Paired trials per subject:")
    for subj in sorted(subj_pair_counts.keys()):
        cnt = subj_pair_counts[subj]
        bar = '█' * cnt
        print(f"    Subject {subj:>3}: {cnt}/4 emotions  {bar}")

    # Subjects missing Samsung entirely
    muse_only_subjs = set(muse_subjs) - set(samsung_subjs)
    if muse_only_subjs:
        print(f"\n  Subjects with EEG but NO Samsung Watch data: {sorted(muse_only_subjs)}")
    else:
        print(f"\n  ✓ All {len(muse_subjs)} EEG subjects also have Samsung Watch files")

    # ── 5. Inspect sample Samsung files for key names ─────────────────────────
    sample_fps = list(samsung_by_subj_emot.values())[:args.inspect_sample]

    print(f"\n  {'='*55}")
    print(f"  Inspecting {len(sample_fps)} sample Samsung Watch files:")
    print(f"  {'='*55}")

    all_seen_keys = set()
    bvp_keys_seen = set()

    for fp in sample_fps:
        subj, emot = parse_filename(fp)
        print(f"\n  File: {os.path.basename(fp)}")
        info = inspect_samsung_file(fp)

        if info.get('error'):
            print(f"    [ERROR] {info['error']}")
            continue

        all_seen_keys.update(info['all_keys'])
        print(f"    All JSON keys ({info['n_keys']} total):")
        print(f"      {info['all_keys']}")

        if info['bvp_key']:
            bvp_keys_seen.add(info['bvp_key'])
            print(f"    → BVP key found     : '{info['bvp_key']}'")
            print(f"      Length            : {info['bvp_length']} samples")
            print(f"      NaN rate          : {info['bvp_nan_rate']*100:.1f}%")
            print(f"      Zero rate         : {info['bvp_zero_rate']*100:.1f}%")
            if info['estimated_fs']:
                dur = info['bvp_length'] / info['estimated_fs']
                print(f"      Estimated FS      : {info['estimated_fs']} Hz  "
                      f"(→ {dur:.1f}s duration)")
            else:
                print(f"      Estimated FS      : unknown (no timestamp key found)")
        else:
            print(f"    ✗ No BVP key found among known keys")
            print(f"      Searched for: {KNOWN_BVP_KEYS}")

    # ── 6. Full scan of BVP signal quality across ALL paired files ────────────
    print(f"\n  {'='*55}")
    print(f"  Full BVP scan across all {len(samsung_by_subj_emot)} Samsung files:")
    print(f"  {'='*55}")

    lengths_by_emot  = defaultdict(list)
    nan_rates        = []
    zero_rates       = []
    fs_estimates     = []
    no_bvp_files     = []

    for (subj, emot), fp in sorted(samsung_by_subj_emot.items()):
        info = inspect_samsung_file(fp)
        if info.get('error') or info['bvp_key'] is None:
            no_bvp_files.append((subj, emot))
            continue
        lengths_by_emot[emot].append(info['bvp_length'])
        nan_rates.append(info['bvp_nan_rate'])
        zero_rates.append(info['bvp_zero_rate'])
        if info['estimated_fs']:
            fs_estimates.append(info['estimated_fs'])

    if no_bvp_files:
        print(f"\n  ⚠ Files with NO BVP signal ({len(no_bvp_files)}):")
        for subj, emot in no_bvp_files:
            print(f"    Subject {subj}, {emot}")
    else:
        print(f"\n  ✓ BVP signal found in all files")

    print(f"\n  Signal length by emotion (samples):")
    for emot in sorted(lengths_by_emot.keys()):
        ls = np.array(lengths_by_emot[emot])
        print(f"    {emot:>12}: n={len(ls):3d}  "
              f"min={ls.min():6.0f}  mean={ls.mean():8.0f}  max={ls.max():6.0f}")

    if nan_rates:
        print(f"\n  NaN rate   : mean={np.mean(nan_rates)*100:.2f}%  "
              f"max={np.max(nan_rates)*100:.2f}%")
        print(f"  Zero rate  : mean={np.mean(zero_rates)*100:.2f}%  "
              f"max={np.max(zero_rates)*100:.2f}%")

    if fs_estimates:
        print(f"\n  Sampling rate estimates (from timestamps):")
        print(f"    min={min(fs_estimates)} Hz  mean={np.mean(fs_estimates):.1f} Hz  "
              f"max={max(fs_estimates)} Hz")
        if len(set(fs_estimates)) > 1:
            print(f"    [NOTE] Inconsistent FS across files — will need per-trial resampling")

    # ── 7. Alignment check: paired valid counts ───────────────────────────────
    valid_paired = len(paired_keys) - len(no_bvp_files)
    per_emot_valid = Counter(e for s, e in paired_keys if (s, e) not in no_bvp_files)

    print(f"\n  {'='*55}")
    print(f"  SUMMARY — Data Available for EEG + BVP Fusion:")
    print(f"  {'='*55}")
    print(f"  Total valid paired trials: {valid_paired}")
    print(f"  Per emotion:")
    for emot in sorted(TARGET_EMOTIONS):
        cnt = per_emot_valid.get(emot, 0)
        bar = '█' * cnt
        balanced = "✓" if cnt >= 30 else "⚠ LOW"
        print(f"    {emot:>12}: {cnt:3d}  {bar}  {balanced}")

    print(f"\n  Recommended fusion approach based on data size:")
    if valid_paired >= 100:
        print(f"  → Feature-level fusion (safe: enough paired samples)")
        print(f"     Concatenate BVP HRV features with EEG DE-LDS per window")
    else:
        print(f"  → Late/decision-level fusion (safer: small dataset)")
        print(f"     Train EEG model + BVP model separately, average predictions")
    print()


if __name__ == '__main__':
    main()
