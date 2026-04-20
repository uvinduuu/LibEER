"""
BVP-Only Emotion Classifier — Sanity Check
============================================
Tests whether Samsung Watch BVP features alone can beat random chance
(25% for 4-class) on Emognition under LOSO evaluation.

Extracts 8 hand-crafted HRV features per trial:
    HR_mean, HR_std, RMSSD, pNN50, LF_power, HF_power, LF_HF_ratio, IBI_range

Classifiers tested: LDA, SVM (RBF), MLP (3-layer)

Why this matters:
  - If BVP beats chance (>28%), it adds value to multimodal fusion
  - If BVP is at chance (~25%), skip BVP and go EEG-only
  - Gives us a principled decision in <30 minutes on Kaggle

Usage (Kaggle / Colab):
    python bvp_only_classifier.py --data_root /kaggle/input/emognition/raw
    python bvp_only_classifier.py --data_root /kaggle/input/emognition/raw --verbose
"""

import os
import glob
import json
import argparse
import warnings
import numpy as np
from collections import defaultdict, Counter
from scipy import signal as scipy_signal
from scipy.stats import zscore

warnings.filterwarnings('ignore')

# ── Target emotions (4-class) ─────────────────────────────────────────────────
TARGET_EMOTIONS = ['ENTHUSIASM', 'FEAR', 'NEUTRAL', 'SADNESS']
LABEL_MAP       = {e: i for i, e in enumerate(sorted(TARGET_EMOTIONS))}
# ENTHUSIASM=0, FEAR=1, NEUTRAL=2, SADNESS=3

# ── Samsung Watch JSON keys (actual format discovered from data)
# Each entry is [timestamp_string, value] — NOT plain floats
BVP_KEY = 'BVPProcessed'   # filtered BVP waveform
HR_KEY  = 'heartRate'      # instantaneous HR in bpm
IBI_KEY = 'PPInterval'     # pulse-to-pulse interval in ms (= IBI)


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def find_samsung_files(data_root):
    """Find all STIMULUS Samsung Watch JSON files."""
    patterns = [
        os.path.join(data_root, '*_STIMULUS_SAMSUNG_WATCH.json'),
        os.path.join(data_root, '*', '*_STIMULUS_SAMSUNG_WATCH.json'),
        os.path.join(data_root, '**', '*_STIMULUS_SAMSUNG_WATCH.json'),
    ]
    return sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})


def parse_filename(fp):
    """Extract (subject, emotion) from filename like 22_FEAR_STIMULUS_SAMSUNG_WATCH.json"""
    name  = os.path.splitext(os.path.basename(fp))[0]
    parts = name.split('_')
    if len(parts) < 2:
        return None, None
    return parts[0], parts[1].upper()


def _parse_paired_list(raw):
    """
    Parse Samsung Watch paired format: [[timestamp_str, value], ...]
    Returns (timestamps_sec, values) as numpy arrays, or (None, None).
    Format: timestamp like '2020-07-27T09:23:37:057716'
    """
    if not isinstance(raw, list) or len(raw) < 5:
        return None, None
    try:
        values = np.array([row[1] for row in raw], dtype=np.float64)
    except Exception:
        return None, None

    # Estimate timestamps in seconds from the string timestamps
    try:
        def ts_to_sec(ts_str):
            # Format: '2020-07-27T09:23:37:057716'
            # Last part after final colon is microseconds
            parts = str(ts_str).replace('T', ':').split(':')
            # parts: [date, hour, min, sec, microsec]
            h, m, s = int(parts[1]), int(parts[2]), int(parts[3])
            us = int(parts[4]) if len(parts) > 4 else 0
            return h * 3600 + m * 60 + s + us * 1e-6

        t0 = ts_to_sec(raw[0][0])
        timestamps = np.array([ts_to_sec(row[0]) - t0 for row in raw])
    except Exception:
        # Fall back to uniform spacing
        timestamps = np.arange(len(values))

    return timestamps, values


def load_samsung_signals(fp):
    """
    Load HR, IBI (PPInterval), and BVP signals from Samsung Watch JSON.

    Returns dict with keys:
        'ibi_ms':   numpy array of PP intervals in ms  (the IBI sequence)
        'ibi_t':    timestamps in seconds for each IBI sample
        'hr':       numpy array of heart rate values in bpm
        'bvp':      numpy array of BVPProcessed signal
        'bvp_fs':   estimated BVP sampling rate in Hz
    Returns None if data is insufficient.
    """
    try:
        with open(fp, 'r') as f:
            obj = json.load(f)
    except Exception:
        return None

    result = {}

    # ── PPInterval (IBI in ms) — most direct HRV source ──
    if IBI_KEY in obj:
        t, v = _parse_paired_list(obj[IBI_KEY])
        if v is not None and len(v) >= 5:
            # Filter physiologically valid IBIs (300–2000 ms = 30–200 bpm)
            valid = (v > 300) & (v < 2000) & np.isfinite(v)
            result['ibi_ms'] = v[valid]
            result['ibi_t']  = t[valid] if t is not None else np.arange(valid.sum())

    # ── heartRate (bpm) ──
    if HR_KEY in obj:
        t, v = _parse_paired_list(obj[HR_KEY])
        if v is not None and len(v) >= 3:
            valid = (v > 30) & (v < 220) & np.isfinite(v)
            result['hr'] = v[valid]

    # ── BVPProcessed (filtered waveform) ──
    if BVP_KEY in obj:
        t, v = _parse_paired_list(obj[BVP_KEY])
        if v is not None and len(v) >= 20 and np.isfinite(v).mean() > 0.5:
            # Estimate fs from timestamps
            if t is not None and t[-1] > 0:
                fs = len(v) / t[-1]
                fs = float(np.clip(fs, 15.0, 30.0))
            else:
                fs = 20.0
            # Interpolate NaNs
            if not np.all(np.isfinite(v)):
                idx = np.arange(len(v))
                mask = np.isfinite(v)
                v[~mask] = np.interp(idx[~mask], idx[mask], v[mask])
            result['bvp']    = v
            result['bvp_fs'] = fs

    # Need at least IBI or HR to be useful
    if 'ibi_ms' not in result and 'hr' not in result:
        return None
    if 'ibi_ms' in result and len(result['ibi_ms']) < 4:
        return None

    return result


# ─────────────────────────────────────────────────────────────────────────────
# HRV feature extraction (NO peak detection needed — PPInterval is direct IBI)
# ─────────────────────────────────────────────────────────────────────────────

def extract_hrv_features(signals):
    """
    Extract 8 HRV features directly from PPInterval + heartRate.

    Samsung Watch already provides:
      - PPInterval : pulse-to-pulse intervals in ms  (= IBI sequence)
      - heartRate  : instantaneous HR in bpm

    No peak detection needed — far more reliable than detecting peaks
    from the raw BVP waveform.

    Features:
        HR_mean   : mean heart rate (bpm)
        HR_std    : std of heart rate
        RMSSD     : sqrt(mean(diff(IBI)^2))  — parasympathetic tone
        pNN50     : fraction of consecutive IBI diffs > 50ms
        LF_power  : LF band HRV power (0.04–0.15 Hz) via Lomb-Scargle
        HF_power  : HF band HRV power (0.15–0.4 Hz)
        LF_HF     : LF/HF ratio — sympatho-vagal balance
        IBI_range : max – min IBI (ms) — overall variability range
    """
    FAIL = {
        'HR_mean': 70.0, 'HR_std': 5.0,  'RMSSD': 30.0,
        'pNN50':   0.1,  'LF_power': 0.5, 'HF_power': 0.5,
        'LF_HF':   1.0,  'IBI_range': 100.0,
    }

    ibi_ms = signals.get('ibi_ms', None)
    hr_arr = signals.get('hr',     None)

    if ibi_ms is None or len(ibi_ms) < 5:
        return FAIL

    # ── Time-domain features from IBI ──
    rmssd     = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))
    pnn50     = float(np.mean(np.abs(np.diff(ibi_ms)) > 50))
    ibi_range = float(ibi_ms.max() - ibi_ms.min())

    # HR from IBI (or use direct HR if available)
    hr_from_ibi = 60000.0 / ibi_ms
    if hr_arr is not None and len(hr_arr) >= 3:
        hr_mean = float(np.mean(hr_arr))
        hr_std  = float(np.std(hr_arr))
    else:
        hr_mean = float(np.mean(hr_from_ibi))
        hr_std  = float(np.std(hr_from_ibi))

    # ── Frequency-domain: Lomb-Scargle on IBI time series ──
    try:
        ibi_t = signals.get('ibi_t', None)
        if ibi_t is None or len(ibi_t) != len(ibi_ms):
            # Reconstruct timestamps from cumulative IBI
            ibi_t = np.cumsum(ibi_ms) / 1000.0  # convert ms → seconds

        ibi_t = ibi_t - ibi_t[0]   # start at 0
        duration = ibi_t[-1]

        if duration > 10.0 and len(ibi_ms) >= 8:
            freqs = np.linspace(0.01, 0.5, 500)
            ibi_norm = ibi_ms - np.mean(ibi_ms)
            pgram = scipy_signal.lombscargle(
                ibi_t.astype(np.float64),
                ibi_norm.astype(np.float64),
                2 * np.pi * freqs,
                normalize=True
            )
            lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
            hf_mask = (freqs >= 0.15) & (freqs <= 0.40)
            lf_p = float(np.trapz(pgram[lf_mask], freqs[lf_mask])) + 1e-8
            hf_p = float(np.trapz(pgram[hf_mask], freqs[hf_mask])) + 1e-8
        else:
            lf_p, hf_p = 0.5, 0.5
    except Exception:
        lf_p, hf_p = 0.5, 0.5

    lf_hf = float(lf_p / hf_p)

    return {
        'HR_mean':   hr_mean,
        'HR_std':    hr_std,
        'RMSSD':     rmssd,
        'pNN50':     pnn50,
        'LF_power':  lf_p,
        'HF_power':  hf_p,
        'LF_HF':     lf_hf,
        'IBI_range': ibi_range,
    }


# ─────────────────────────────────────────────────────────────────────────────
# LOSO evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_loso(X, y, subjects, clf_name='LDA', verbose=False):
    """
    Leave-One-Subject-Out cross-validation.
    Returns list of per-fold accuracies.
    """
    from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
    from sklearn.svm import SVC
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline

    def make_clf(name):
        if name == 'LDA':
            return Pipeline([('scaler', StandardScaler()),
                             ('clf', LinearDiscriminantAnalysis())])
        elif name == 'SVM':
            return Pipeline([('scaler', StandardScaler()),
                             ('clf', SVC(kernel='rbf', C=10, gamma='scale',
                                         class_weight='balanced'))])
        elif name == 'MLP':
            return Pipeline([('scaler', StandardScaler()),
                             ('clf', MLPClassifier(
                                 hidden_layer_sizes=(64, 32),
                                 activation='relu',
                                 max_iter=500,
                                 early_stopping=True,
                                 validation_fraction=0.15,
                                 random_state=42,
                             ))])
        raise ValueError(f"Unknown classifier: {name}")

    unique_subjects = sorted(set(subjects))
    fold_accs = []
    X = np.array(X)
    y = np.array(y)
    subjects = np.array(subjects)

    for test_subj in unique_subjects:
        tr_mask = subjects != test_subj
        te_mask = subjects == test_subj

        X_tr, y_tr = X[tr_mask], y[tr_mask]
        X_te, y_te = X[te_mask], y[te_mask]

        if len(np.unique(y_tr)) < 2 or len(X_te) == 0:
            continue

        try:
            clf = make_clf(clf_name)
            clf.fit(X_tr, y_tr)
            preds = clf.predict(X_te)
            acc = np.mean(preds == y_te)
            fold_accs.append(acc)
            if verbose:
                print(f"  Subject {test_subj}: acc={acc:.3f} "
                      f"(n_test={len(y_te)}, classes={sorted(Counter(y_te).items())})")
        except Exception as e:
            if verbose:
                print(f"  Subject {test_subj}: FAILED — {e}")

    return fold_accs


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description='BVP-only emotion classification sanity check (Emognition LOSO)'
    )
    parser.add_argument('--data_root', required=True,
                        help='Root of raw Emognition dataset '
                             '(folder containing subject subdirs or raw JSON files)')
    parser.add_argument('--verbose', action='store_true',
                        help='Print per-subject fold results')
    args = parser.parse_args()

    print(f"\n{'='*65}")
    print(f"  BVP-Only Emotion Classifier — Emognition Sanity Check")
    print(f"  data_root : {args.data_root}")
    print(f"  Chance level: {100/len(TARGET_EMOTIONS):.1f}%")
    print(f"{'='*65}\n")

    # ── 1. Find Samsung Watch files ──────────────────────────────────────────
    files = find_samsung_files(args.data_root)
    print(f"Found {len(files)} Samsung Watch JSON files\n")

    if not files:
        print("❌ No Samsung Watch files found!")
        print("   Double-check --data_root points to the Emognition raw dataset.")
        return

    # ── 2. Extract HRV features per trial ────────────────────────────────────
    X_list, y_list, subj_list = [], [], []
    feature_names = ['HR_mean', 'HR_std', 'RMSSD', 'pNN50',
                     'LF_power', 'HF_power', 'LF_HF', 'IBI_range']

    failed = 0
    for fp in files:
        subj, emot = parse_filename(fp)
        if subj is None or emot not in TARGET_EMOTIONS:
            continue

        signals = load_samsung_signals(fp)
        if signals is None:
            failed += 1
            if args.verbose:
                print(f"  [skip] {os.path.basename(fp)} — no usable signals")
            continue

        feats = extract_hrv_features(signals)
        feat_vec = np.array([feats[k] for k in feature_names], dtype=np.float32)

        # Guard against Infs/NaNs
        if not np.all(np.isfinite(feat_vec)):
            failed += 1
            continue

        X_list.append(feat_vec)
        y_list.append(LABEL_MAP[emot])
        subj_list.append(subj)

    n_total = len(X_list)
    print(f"Extracted features for {n_total} trials  ({failed} failed/skipped)")
    print(f"Subjects: {len(set(subj_list))} unique — {sorted(set(subj_list))}")

    label_dist = Counter(y_list)
    print(f"Label distribution:")
    rev_map = {v: k for k, v in LABEL_MAP.items()}
    for lid in sorted(label_dist):
        print(f"  {rev_map[lid]:>12}: {label_dist[lid]:3d} trials")

    if n_total < 20:
        print("\n❌ Too few trials for LOSO evaluation. Check data_root path.")
        return

    print(f"\nFeature summary (mean ± std across all trials):")
    X_arr = np.array(X_list)
    for i, fname in enumerate(feature_names):
        print(f"  {fname:>12}: {X_arr[:, i].mean():8.3f} ± {X_arr[:, i].std():.3f}")

    # ── 3. LOSO classification ────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  LOSO Results  (chance = {100/len(TARGET_EMOTIONS):.1f}%)")
    print(f"{'='*65}")

    target = 100 / len(TARGET_EMOTIONS)  # 25.0%

    for clf_name in ['LDA', 'SVM', 'MLP']:
        print(f"\n── {clf_name} ──")
        fold_accs = run_loso(X_list, y_list, subj_list,
                              clf_name=clf_name, verbose=args.verbose)
        if not fold_accs:
            print("  No valid folds")
            continue
        mean_acc = np.mean(fold_accs) * 100
        std_acc  = np.std(fold_accs)  * 100
        beats    = "✅ beats chance!" if mean_acc > (target + 3) else \
                   "⚠️  marginally above chance" if mean_acc > target else \
                   "❌ at/below chance"
        print(f"  LOSO Acc: {mean_acc:.1f}% ± {std_acc:.1f}%   {beats}")
        print(f"  n_folds : {len(fold_accs)}")

    # ── 4. Decision ───────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"  DECISION GUIDE")
    print(f"{'='*65}")
    print(f"  BVP ≥ 30%  → Include BVP features in multimodal fusion (worth it)")
    print(f"  BVP 26-29% → Include as lightweight concat only (low risk, small gain)")
    print(f"  BVP ≤ 25%  → Skip BVP, go EEG-only (BVP has no discriminative signal)")
    print()


if __name__ == '__main__':
    main()
