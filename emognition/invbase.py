"""
InvBase — Inverse Filtering Baseline Removal for EEG.

Removes subject-specific baseline brain activity by dividing
the frequency spectrum of each emotion trial by the subject's
resting-state (baseline) spectrum.

Reference: "InvBase" paradigm for EEG emotion recognition.

Why this helps:
  - Each person has unique baseline brain rhythms
  - Without baseline removal, features are dominated by inter-subject
    differences rather than emotion differences
  - Dividing (not subtracting) normalizes the scale properly
"""

import os
import glob
import json
import numpy as np
import pandas as pd

from config import DATA_ROOT, FS, CHANNELS, QUALITY_CHANNELS, SAMPLE_LENGTH


# Frequency bands for InvBase feature extraction
INVBASE_BANDS = [
    ("delta", 1, 3),
    ("theta", 4, 7),
    ("alpha", 8, 13),
    ("beta",  14, 30),
    ("gamma", 31, 45),
]

# 5 bands × 2 stats (mean, var) = 10 features per channel
NUM_INVBASE_FEATURES = len(INVBASE_BANDS) * 2


# ===================== HELPERS =====================

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


# ===================== BASELINE LOADING =====================

def load_baselines(data_root=DATA_ROOT):
    """
    Load resting-state BASELINE recordings for all subjects.

    Looks for files named: {subject_id}_BASELINE_STIMULUS_MUSE.json

    Returns:
        baselines: dict {subject_id: baseline_spectrum}
                   baseline_spectrum shape: (4, freq_bins)
                   This is the average power spectrum across the
                   entire baseline recording for each channel.
    """
    patterns = [
        os.path.join(data_root, "*_BASELINE_STIMULUS_MUSE.json"),
        os.path.join(data_root, "*", "*_BASELINE_STIMULUS_MUSE.json"),
        os.path.join(data_root, "**", "*_BASELINE_STIMULUS_MUSE.json"),
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})
    print(f"[InvBase] Found {len(files)} BASELINE files")

    baselines = {}

    for fp in files:
        name = os.path.basename(fp)
        subject_id = name.split("_")[0]

        with open(fp, "r") as f:
            obj = json.load(f)

        # Load raw channels
        raw = {}
        for ch in CHANNELS:
            raw[ch] = _interp_nan(_to_num(obj.get(ch, [])))

        L = min(len(raw[ch]) for ch in CHANNELS)
        if L < SAMPLE_LENGTH:
            print(f"[InvBase] WARNING: Baseline too short for subject {subject_id} ({L} samples), skipping")
            continue

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
            print(f"[InvBase] WARNING: Baseline too short after filtering for {subject_id}, skipping")
            continue

        # Stack channels: (L, 4)
        sig = np.stack([raw[ch][:L] for ch in CHANNELS], axis=1)
        sig = sig - np.mean(sig, axis=0, keepdims=True)  # remove DC

        # Compute average power spectrum over sliding windows
        win = SAMPLE_LENGTH
        step = win // 2  # 50% overlap for baseline averaging
        spectra = []

        for s in range(0, max(0, L - win + 1), step):
            chunk = sig[s:s + win]  # (win, 4)
            fft_chunk = np.fft.rfft(chunk, axis=0)  # (freq_bins, 4)
            power = np.abs(fft_chunk) ** 2  # power spectrum
            spectra.append(power)

        if not spectra:
            continue

        # Average power spectrum: (freq_bins, 4) → transpose to (4, freq_bins)
        avg_spectrum = np.mean(spectra, axis=0).T  # (4, freq_bins)

        # Add small epsilon to avoid division by zero
        avg_spectrum = np.maximum(avg_spectrum, 1e-10)

        baselines[subject_id] = avg_spectrum.astype(np.float64)

    print(f"[InvBase] Loaded baselines for {len(baselines)} subjects")
    return baselines


# ===================== INVBASE FEATURE EXTRACTION =====================

def extract_invbase_features(X_raw, subjects, baselines, fs=FS):
    """
    Apply InvBase normalization and extract band features.

    For each window:
      1. FFT of the window
      2. Divide by subject's baseline FFT (point-wise)
      3. Extract mean & variance per frequency band

    Args:
        X_raw:      (N, 4, T) raw EEG windows, channels first
        subjects:   (N,) subject IDs
        baselines:  dict {subject_id: baseline_spectrum (4, freq_bins)}
        fs:         sampling frequency

    Returns:
        invbase_features: (N, 4, 10) float32
                         5 bands × 2 stats (mean, var)
    """
    N, C, T = X_raw.shape
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)
    n_freq = len(freqs)

    features = np.zeros((N, C, NUM_INVBASE_FEATURES), dtype=np.float32)

    # Track how many windows get InvBase vs fallback
    n_invbase = 0
    n_fallback = 0

    for i in range(N):
        subj = subjects[i]
        window = X_raw[i]  # (4, T)

        # FFT of the window
        fft_window = np.fft.rfft(window, axis=1)  # (4, freq_bins)
        power_window = np.abs(fft_window) ** 2

        # Apply InvBase if baseline available
        if subj in baselines:
            baseline_spectrum = baselines[subj]  # (4, freq_bins)

            # Ensure same freq bins
            if baseline_spectrum.shape[1] == n_freq:
                # Core InvBase: divide window spectrum by baseline spectrum
                normalized = power_window / baseline_spectrum
                n_invbase += 1
            else:
                # Resample baseline to match window freq bins
                from scipy.interpolate import interp1d
                new_baseline = np.zeros((C, n_freq))
                old_freqs = np.linspace(0, fs / 2, baseline_spectrum.shape[1])
                new_freqs = freqs
                for ch in range(C):
                    f_interp = interp1d(old_freqs, baseline_spectrum[ch],
                                        kind='linear', fill_value='extrapolate')
                    new_baseline[ch] = np.maximum(f_interp(new_freqs), 1e-10)
                normalized = power_window / new_baseline
                n_invbase += 1
        else:
            # No baseline available — use raw power spectrum
            normalized = power_window
            n_fallback += 1

        # Extract band features from normalized spectrum
        feat_idx = 0
        for _, lo, hi in INVBASE_BANDS:
            band_mask = (freqs >= lo) & (freqs < hi)
            if band_mask.sum() == 0:
                feat_idx += 2
                continue

            band_power = normalized[:, band_mask]  # (4, band_bins)
            features[i, :, feat_idx] = np.log(band_power.mean(axis=1) + 1e-10)    # log mean
            features[i, :, feat_idx + 1] = np.log(band_power.var(axis=1) + 1e-10)  # log var
            feat_idx += 2

    print(f"[InvBase] Applied to {n_invbase}/{N} windows "
          f"({n_fallback} fallback, no baseline)")

    return features


# ===================== TIME-DOMAIN INVBASE APPLICATION =====================

# Frequency bands used for band-filtering after InvBase
INVBASE_BAND_HZ = [
    ("delta", 1.0,   3.0),
    ("theta", 4.0,   7.0),
    ("alpha", 8.0,  13.0),
    ("beta",  14.0, 30.0),
    ("gamma", 31.0, 45.0),
]
NUM_BANDS = len(INVBASE_BAND_HZ)  # 5


def apply_invbase_to_raw(trial, baseline_spectrum, fs=FS):
    """
    Apply InvBase normalization to a raw EEG trial, returning a time-domain signal.

    Divides the trial's FFT amplitude by the square-root of the subject's baseline
    power spectrum, then reconstructs via iFFT (phase is preserved).  The result
    is a time-domain signal whose frequency content is expressed as a ratio to the
    subject's resting-state activity — emotion-driven deviations stand out.

    Different trial lengths are handled automatically via interpolation of the
    baseline spectrum to the trial's frequency resolution.

    Args:
        trial:             (C, T) float  — raw EEG trial (C channels, T samples)
        baseline_spectrum: (C, n_base)   — average power spectrum from
                           load_baselines() or load_baselines_processed()
        fs:                sampling rate in Hz (default: FS from config)

    Returns:
        normalized: (C, T) float32 — InvBase-normalized trial in time domain.
                    Falls back to the original trial (as float32) if the baseline
                    is None or has a channel-count mismatch.
    """
    C, T = trial.shape

    if baseline_spectrum is None or baseline_spectrum.shape[0] != C:
        return trial.astype(np.float32)

    # FFT of the trial along the time axis
    fft    = np.fft.rfft(trial.astype(np.float64), axis=1)  # (C, n_freq) complex
    n_freq = fft.shape[1]
    freqs  = np.fft.rfftfreq(T, d=1.0 / fs)                 # (n_freq,)

    # Match baseline frequency bins to the trial's frequency resolution
    n_base = baseline_spectrum.shape[1]
    if n_base == n_freq:
        baseline = baseline_spectrum.astype(np.float64).copy()
    else:
        from scipy.interpolate import interp1d
        old_freqs = np.linspace(0.0, fs / 2.0, n_base)
        baseline  = np.zeros((C, n_freq), dtype=np.float64)
        for c in range(C):
            f_interp    = interp1d(old_freqs, baseline_spectrum[c].astype(np.float64),
                                   kind="linear", fill_value="extrapolate")
            baseline[c] = f_interp(freqs)

    # Relative floor: clamp each channel's baseline to 1% of its own mean power.
    #
    # WHY: The old floor was an absolute constant (1e-10). Brain signals exist
    # only in 1-45 Hz; above 45 Hz the baseline power is naturally near zero.
    # Dividing by sqrt(1e-10) ≈ 3e-6 amplified out-of-band noise by ~1e5,
    # producing float32-overflow / inf values in the time-domain signal.
    # Those inf values made augmentor std = inf, and 0.0 * inf = NaN (IEEE 754),
    # silently corrupting every batch that contained one of these windows.
    #
    # A relative floor = 1% of mean power ensures out-of-band bins where the
    # brain genuinely has no activity are normalised to ~sqrt(100)=10× at most,
    # rather than millions-fold. In-band bins (1-45 Hz) are unaffected because
    # their power >> 1% of the mean.
    for c in range(C):
        floor_c     = 0.01 * np.mean(baseline[c])        # 1% of mean per channel
        floor_c     = max(floor_c, 1e-10)                 # absolute safety net
        baseline[c] = np.maximum(baseline[c], floor_c)

    # InvBase: divided amplitude, preserved phase
    #   normalized_amplitude = |fft| / sqrt(baseline_power)
    #   normalized_fft       = normalized_amplitude * exp(j * phase)
    amplitude = np.abs(fft)                           # (C, n_freq)
    phase     = np.angle(fft)                         # (C, n_freq)
    norm_amp  = amplitude / np.sqrt(baseline)         # (C, n_freq)
    norm_fft  = norm_amp * np.exp(1j * phase)         # (C, n_freq) complex

    normalized = np.fft.irfft(norm_fft, n=T, axis=1) # (C, T) real
    return normalized.astype(np.float32)


# ===================== BASELINE LOADER FOR PROCESSED (_CLEANED) DATASET =====================

def load_baselines_processed(data_root, fs=FS, sample_length=None):
    """
    Load per-subject resting-state baselines from the cleaned/processed dataset.

    Handles the nested folder format produced by the Emognition preprocessing:
        {data_root}/{subj}/{subj}_BASELINE_STIMULUS_MUSE_cleaned/
            {subj}_BASELINE_STIMULUS_MUSE_cleaned.json

    Falls back to the original (non-cleaned) file pattern if no cleaned files
    are found, so the function works with both dataset versions.

    Args:
        data_root:     path to the Emognition Processed dataset root.
        fs:            sampling rate in Hz (default: 256 for Muse 2).
        sample_length: FFT window length for spectrum estimation.
                       Default: int(4.0 * fs) = 1024 @ 256 Hz.

    Returns:
        baselines: dict {subject_id (str): np.ndarray of shape (4, freq_bins)}
                   Returns an empty dict if no baseline files are found.
    """
    if sample_length is None:
        sample_length = int(4.0 * fs)   # 4-second windows (= 1024 @ 256 Hz)

    # Glob patterns for the cleaned layout (most-specific first)
    patterns = [
        os.path.join(data_root, "*", "*_BASELINE_STIMULUS_MUSE_cleaned",
                     "*_BASELINE_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "*", "*_BASELINE_STIMULUS_MUSE_cleaned.json"),
        os.path.join(data_root, "**", "*_BASELINE_STIMULUS_MUSE_cleaned.json"),
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})

    # Fall back to original format if nothing found
    if not files:
        orig_patterns = [
            os.path.join(data_root, "*_BASELINE_STIMULUS_MUSE.json"),
            os.path.join(data_root, "*", "*_BASELINE_STIMULUS_MUSE.json"),
            os.path.join(data_root, "**", "*_BASELINE_STIMULUS_MUSE.json"),
        ]
        files = sorted({p for pat in orig_patterns
                        for p in glob.glob(pat, recursive=True)})
        if files:
            print("[InvBase] Using original (non-cleaned) baseline format")

    print(f"[InvBase] Found {len(files)} BASELINE files")

    baselines = {}

    for fp in files:
        name       = os.path.splitext(os.path.basename(fp))[0]  # strip .json
        subject_id = name.split("_")[0]                          # e.g. "22"

        try:
            with open(fp, "r") as f:
                obj = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            print(f"[InvBase] WARNING: Cannot read {fp}: {e}")
            continue

        # Read raw channels
        raw = {}
        for ch in CHANNELS:
            raw[ch] = _interp_nan(_to_num(obj.get(ch, [])))

        L = min(len(raw[ch]) for ch in CHANNELS)
        if L < sample_length:
            print(f"[InvBase] WARNING: Baseline too short for {subject_id} "
                  f"({L} samples < {sample_length}), skipping")
            continue

        for ch in CHANNELS:
            raw[ch] = raw[ch][:L]

        # Quality mask: finite values + headband on + HSI ≤ 2
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
        if L < sample_length:
            print(f"[InvBase] WARNING: Baseline too short after quality filter "
                  f"for {subject_id}, skipping")
            continue

        # Stack channels, remove DC offset
        sig = np.stack([raw[ch][:L] for ch in CHANNELS], axis=1)  # (L, 4)
        sig = sig - np.mean(sig, axis=0, keepdims=True)

        # Average power spectrum over 50%-overlapping windows
        step    = sample_length // 2
        spectra = []
        for s in range(0, max(0, L - sample_length + 1), step):
            chunk   = sig[s:s + sample_length]           # (sample_length, 4)
            fft_c   = np.fft.rfft(chunk, axis=0)         # (freq_bins, 4)
            spectra.append(np.abs(fft_c) ** 2)

        if not spectra:
            continue

        avg_spectrum = np.mean(spectra, axis=0).T         # (4, freq_bins)
        avg_spectrum = np.maximum(avg_spectrum, 1e-10)

        baselines[subject_id] = avg_spectrum.astype(np.float64)

    print(f"[InvBase] Loaded baselines for {len(baselines)} subjects: "
          f"{sorted(baselines.keys())}")
    return baselines
