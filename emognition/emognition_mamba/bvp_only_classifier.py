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

# ── BVP signal key names to search in Samsung JSON ───────────────────────────
BVP_KEYS = ['BVPProcessed', 'BVP', 'bvp', 'PPG', 'ppg', 'GreenChannel', 'green_channel']
TS_KEYS  = ['Timestamp', 'timestamps', 'timestamp', 'time', 'Time']


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def find_samsung_files(data_root):
    """Find all Samsung Watch JSON files for target emotions."""
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


def load_bvp_signal(fp):
    """
    Load BVP signal and estimated sampling rate from one Samsung JSON.
    Returns (bvp_arr, fs_hz) or (None, None) on failure.
    """
    try:
        with open(fp, 'r') as f:
            obj = json.load(f)
    except Exception:
        return None, None

    bvp = None
    for key in BVP_KEYS:
        if key in obj:
            raw = obj[key]
            if isinstance(raw, list) and len(raw) > 10:
                try:
                    arr = np.array(raw, dtype=np.float64)
                    if np.isfinite(arr).mean() > 0.5:
                        bvp = arr
                        break
                except Exception:
                    continue

    if bvp is None:
        return None, None

    # Estimate sampling rate from timestamps
    fs = 20.0  # default Samsung BVP sampling rate
    for ts_key in TS_KEYS:
        if ts_key in obj:
            try:
                ts = np.array(obj[ts_key], dtype=np.float64)
                ts = ts[np.isfinite(ts)]
                if len(ts) > 10:
                    diffs = np.diff(ts)
                    diffs = diffs[(diffs > 0) & (diffs < 1.0)]
                    if len(diffs) > 5:
                        fs = round(1.0 / np.median(diffs), 1)
            except Exception:
                pass
            break

    # Clamp to reasonable range
    fs = float(np.clip(fs, 15.0, 30.0))

    # Interpolate NaNs
    mask = np.isfinite(bvp)
    if mask.sum() < 20:
        return None, None
    if not mask.all():
        idx = np.arange(len(bvp))
        bvp[~mask] = np.interp(idx[~mask], idx[mask], bvp[mask])

    return bvp, fs


# ─────────────────────────────────────────────────────────────────────────────
# BVP feature extraction
# ─────────────────────────────────────────────────────────────────────────────

def bandpass(sig, lo, hi, fs, order=4):
    """Butterworth bandpass filter."""
    nyq = fs / 2.0
    lo_n = max(lo / nyq, 0.01)
    hi_n = min(hi / nyq, 0.99)
    if lo_n >= hi_n:
        return sig
    sos = scipy_signal.butter(order, [lo_n, hi_n], btype='band', output='sos')
    return scipy_signal.sosfiltfilt(sos, sig)


def detect_peaks(bvp, fs, min_hr=40, max_hr=200):
    """
    Simple pan-tompkins-style peak detector for BVP.
    Returns array of peak indices.
    """
    min_dist = int(fs * 60.0 / max_hr)
    max_dist = int(fs * 60.0 / min_hr)

    filtered = bandpass(bvp, 0.7, 3.5, fs)
    # Z-score normalise
    std = filtered.std()
    if std < 1e-8:
        return np.array([], dtype=int)
    filtered = (filtered - filtered.mean()) / std

    peaks, props = scipy_signal.find_peaks(
        filtered,
        distance=min_dist,
        height=0.3,
    )
    return peaks


def extract_hrv_features(bvp, fs):
    """
    Extract 8 HRV features from a BVP signal.

    Returns a dict with:
        HR_mean   : mean heart rate (bpm)
        HR_std    : std of instantaneous HR
        RMSSD     : root mean square of successive IBI differences (ms)
        pNN50     : fraction of successive diffs > 50ms
        LF_power  : low-frequency HRV power (0.04–0.15 Hz)
        HF_power  : high-frequency HRV power (0.15–0.4 Hz)
        LF_HF     : LF/HF ratio
        IBI_range : max – min IBI (ms)
    """
    FAIL = {
        'HR_mean': 70.0, 'HR_std': 5.0, 'RMSSD': 30.0,
        'pNN50': 0.1, 'LF_power': 0.5, 'HF_power': 0.5,
        'LF_HF': 1.0, 'IBI_range': 100.0
    }

    peaks = detect_peaks(bvp, fs)
    if len(peaks) < 5:
        return FAIL

    # IBI sequence in milliseconds
    ibi_ms = np.diff(peaks) / fs * 1000.0

    # Remove physiologically impossible IBIs (< 300ms = 200bpm, > 2000ms = 30bpm)
    ibi_ms = ibi_ms[(ibi_ms > 300) & (ibi_ms < 2000)]
    if len(ibi_ms) < 4:
        return FAIL

    # Time-domain features
    hr      = 60000.0 / ibi_ms           # instantaneous HR (bpm)
    hr_mean = float(np.mean(hr))
    hr_std  = float(np.std(hr))
    rmssd   = float(np.sqrt(np.mean(np.diff(ibi_ms) ** 2)))
    pnn50   = float(np.mean(np.abs(np.diff(ibi_ms)) > 50))
    ibi_range = float(ibi_ms.max() - ibi_ms.min())

    # Frequency-domain features using Lomb-Scargle on IBI time series
    # (handles unevenly-sampled IBI sequences)
    try:
        # Timestamps of R-peaks in seconds
        t_peaks = peaks[1:] / fs          # time of each IBI measurement
        t_peaks_norm = t_peaks - t_peaks[0]

        if t_peaks_norm[-1] > 10.0:
            freqs = np.linspace(0.01, 0.5, 500)
            pgram = scipy_signal.lombscargle(
                t_peaks_norm, ibi_ms - np.mean(ibi_ms),
                2 * np.pi * freqs, normalize=True
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

    # ── 2. Extract BVP features per trial ────────────────────────────────────
    X_list, y_list, subj_list = [], [], []
    feature_names = ['HR_mean', 'HR_std', 'RMSSD', 'pNN50',
                     'LF_power', 'HF_power', 'LF_HF', 'IBI_range']

    failed = 0
    for fp in files:
        subj, emot = parse_filename(fp)
        if subj is None or emot not in TARGET_EMOTIONS:
            continue

        bvp, fs = load_bvp_signal(fp)
        if bvp is None:
            failed += 1
            if args.verbose:
                print(f"  [skip] {os.path.basename(fp)} — no BVP signal")
            continue

        feats = extract_hrv_features(bvp, fs)
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
