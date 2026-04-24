"""
anushka_loader.py  —  CSV loader for inference participant data
═══════════════════════════════════════════════════════════════════════════════

Input file formats
──────────────────
EEG CSV  :  {emotion}{clip_id}_{name}_eeg_cleaned.csv
  Columns: TimeStamp, Delta_TP9..Gamma_TP10 (20 band-ch), RAW_TP9..TP10,
           HeadBandOn, HSI_TP9..TP10, ...
  → Reads RAW_TP9/AF7/AF8/TP10, applies artefact clipping +
    InvBase normalisation (using population-average resting baseline from
    training subjects) + Butterworth band-stack. Matches training pipeline.

PPG CSV  :  {emotion}{clip_id}_{name}_ppg_hr_ibi_cleaned.csv
  Columns: ppg_timestamp_ms, ppg_timestamp_readable, ppg_green, time_s
  → Peak-detects heartbeats on ppg_green to compute IBI series,
    then extracts the same 8 HRV features used in training.

Directory layout expected
─────────────────────────
  <root>/
    Participant 1/
      SAD4_Anushka_eeg_cleaned.csv
      SAD4_Anushka_ppg_hr_ibi_cleaned.csv
      ...
    Participant 2/
      ...

Output
──────
load_all_participants(root_dir) → list of ClipRecord, one per EEG file:
  .emotion      str   ENTHUSIASM / FEAR / NEUTRAL / SADNESS
  .clip_idx     int   numeric clip index (2, 4, 5, ...)
  .eeg          (20, T)  float32 band-filtered EEG
  .bvp          (8,)     float32 HRV features (zeros if unavailable)
  .label        int   0-3
  .key          str   e.g. 'enthusiasm2'   (for setup assignment)
  .participant  str   name extracted from filename (e.g. 'Anushka')
  .folder       str   subdirectory name (e.g. 'Participant 1')
  .pid          int   numeric participant ID parsed from folder name
"""

import os
import re
import glob

import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# ── Raw EEG channel order (must match training pipeline: CHANNELS in config.py)
RAW_EEG_COLS = ['RAW_TP9', 'RAW_AF7', 'RAW_AF8', 'RAW_TP10']

# ── Band ranges matching invbase.INVBASE_BAND_HZ (training pipeline) ────────
_BAND_HZ = [
    ('delta',  1.0,  3.0),
    ('theta',  4.0,  7.0),
    ('alpha',  8.0, 13.0),
    ('beta',  14.0, 30.0),
    ('gamma', 31.0, 45.0),
]


def _butter_band(lo: float, hi: float, fs: float, order: int = 4):
    nyq  = fs / 2.0
    low  = float(np.clip(lo / nyq, 1e-6, 1.0 - 1e-6))
    high = float(np.clip(hi / nyq, 1e-6, 1.0 - 1e-6))
    return butter(order, [low, high], btype='band')


def _clip_artefacts(trial: np.ndarray, n_sigma: float = 5.0) -> np.ndarray:
    """Clip per-channel outliers to ±n_sigma × std (same as training)."""
    trial = trial.astype(np.float64, copy=True)
    for c in range(trial.shape[0]):
        s = trial[c].std()
        if s > 1e-8:
            trial[c] = np.clip(trial[c], -n_sigma * s, n_sigma * s)
    return trial.astype(np.float32)


def _band_stack(trial: np.ndarray, fs: float = 256.0,
               order: int = 4) -> np.ndarray:
    """
    Butterworth band-stack: matches apply_band_stack() in training pipeline.
    Input:  (4, T)  artefact-clipped raw EEG
    Output: (20, T) band-filtered amplitude time series
    """
    bands = []
    for (_, lo, hi) in _BAND_HZ:
        b, a = _butter_band(lo, hi, fs, order)
        bands.append(filtfilt(b, a, trial, axis=1).astype(np.float32))
    return np.concatenate(bands, axis=0)   # (4*5, T) = (20, T)


def _zscore_clip(arr: np.ndarray) -> np.ndarray:
    """
    Per-channel z-score normalization over the full clip.
    Used as a final step after spectral whitening + band-stack.
    """
    m = arr.mean(axis=1, keepdims=True)
    s = arr.std(axis=1, keepdims=True)
    s = np.where(s < 1e-8, 1.0, s)
    return ((arr - m) / s).astype(np.float32)


def _invbase_whiten(trial: np.ndarray,
                    baseline_power: np.ndarray = None,
                    fs: float = 256.0) -> np.ndarray:
    """
    Apply InvBase spectral normalisation — mirrors apply_invbase_to_raw() in
    invbase.py exactly.

    Divides each FFT bin's amplitude by sqrt(baseline_power), so the output
    represents deviations from the subject's (or population's) resting state.

    baseline_power : (C, n_base) float — power spectrum of the resting baseline.
        If None, falls back to using the clip's own power (poor approximation —
        only use when no baseline is available at all).

    Floor strategy: 1% of per-channel mean power (same as invbase.py), which
    prevents catastrophic amplification in out-of-band bins.

    Input:  (C, T)  artefact-clipped raw EEG
    Output: (C, T)  InvBase-normalised EEG
    """
    C, T   = trial.shape
    fft    = np.fft.rfft(trial.astype(np.float64), axis=1)  # (C, n_freq) complex
    n_freq = fft.shape[1]

    if baseline_power is not None:
        # ── proper InvBase: use provided power spectrum (matches training) ──
        n_base = baseline_power.shape[1]
        if n_base == n_freq:
            base = baseline_power.astype(np.float64).copy()
        else:
            # Interpolate baseline to clip's FFT frequency resolution
            from scipy.interpolate import interp1d
            old_freqs = np.linspace(0.0, fs / 2.0, n_base)
            new_freqs = np.fft.rfftfreq(T, d=1.0 / fs)
            base = np.zeros((C, n_freq), dtype=np.float64)
            for c in range(C):
                fi = interp1d(old_freqs, baseline_power[c].astype(np.float64),
                              kind='linear', fill_value='extrapolate')
                base[c] = fi(new_freqs)
    else:
        # ── fallback: clip's own power (no resting baseline available) ──
        base = np.abs(fft) ** 2

    # Per-channel floor = 1% of mean power (same as invbase.py)
    mean_p    = base.mean(axis=1, keepdims=True)
    floor     = np.maximum(mean_p * 0.01, 1e-10)
    base_safe = np.maximum(base, floor)

    fft_norm = fft / np.sqrt(base_safe)
    result   = np.fft.irfft(fft_norm, n=T, axis=1)
    return result.astype(np.float32)


# ── EEG band column order kept for reference (not used for loading any more) ─
EEG_BAND_COLS = [
    'Delta_TP9',  'Delta_AF7',  'Delta_AF8',  'Delta_TP10',
    'Theta_TP9',  'Theta_AF7',  'Theta_AF8',  'Theta_TP10',
    'Alpha_TP9',  'Alpha_AF7',  'Alpha_AF8',  'Alpha_TP10',
    'Beta_TP9',   'Beta_AF7',   'Beta_AF8',   'Beta_TP10',
    'Gamma_TP9',  'Gamma_AF7',  'Gamma_AF8',  'Gamma_TP10',
]

CLASS_NAMES = ['ENTHUSIASM', 'FEAR', 'NEUTRAL', 'SADNESS']
EMOT2ID     = {e: i for i, e in enumerate(CLASS_NAMES)}

_EMOT_MAP = {
    'sad':        'SADNESS',
    'sadness':    'SADNESS',
    'enthusiasm': 'ENTHUSIASM',
    'fear':       'FEAR',
    'neutral':    'NEUTRAL',
}

# ── ClipRecord ────────────────────────────────────────────────────────────────

class ClipRecord:
    __slots__ = ('emotion', 'clip_idx', 'eeg', 'bvp', 'label', 'key',
                 'participant', 'folder', 'pid')

    def __init__(self, emotion, clip_idx, eeg, bvp, participant,
                 folder='', pid=0):
        self.emotion     = emotion
        self.clip_idx    = clip_idx
        self.eeg         = eeg        # (20, T) float32
        self.bvp         = bvp        # (8,) float32
        self.label       = EMOT2ID[emotion]
        self.participant = participant
        self.folder      = folder     # e.g. 'Participant 1'
        self.pid         = pid        # numeric, e.g. 1
        self.key         = f'{emotion.lower()}{clip_idx}'  # e.g. 'enthusiasm2'


# ══════════════════════════════════════════════════════════════════════════════
#  Filename parsing
# ══════════════════════════════════════════════════════════════════════════════

def parse_clip_info(filename: str):
    """
    Parse emotion, clip index, and participant name from a CSV filename.

    Handles mixed-case prefixes:
        enthusiasm2_Anushka_eeg_cleaned.csv  → ('ENTHUSIASM', 2, 'Anushka')
        SAD4_Anushka_ppg_hr_ibi_cleaned.csv  → ('SADNESS', 4, 'Anushka')
        fear5_Anushka_eeg_cleaned.csv        → ('FEAR', 5, 'Anushka')
        sad1_Anushka_eeg_cleaned.csv         → ('SADNESS', 1, 'Anushka')

    Returns (emotion_str, clip_idx, participant) or None.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    # Strip known suffixes
    base = re.sub(r'_(eeg_cleaned|ppg_hr_ibi_cleaned)$', '', base,
                  flags=re.IGNORECASE)
    # Pattern: {letters}{digits}_{Participant}
    m = re.match(r'^([a-zA-Z]+)(\d+)_(.+)$', base)
    if not m:
        return None
    emot_raw    = m.group(1).lower()
    clip_idx    = int(m.group(2))
    participant = m.group(3)
    emotion     = _EMOT_MAP.get(emot_raw)
    if emotion is None:
        return None
    return emotion, clip_idx, participant


# ══════════════════════════════════════════════════════════════════════════════
#  EEG CSV loading
# ══════════════════════════════════════════════════════════════════════════════

def load_eeg_csv(fp: str, min_sec: float = 4.0, fs: float = 256.0,
                 baseline_spectrum: np.ndarray = None) -> np.ndarray:
    """
    Load and process raw EEG from a CSV file into the model's expected format.

    Pipeline (matches training exactly when baseline_spectrum is provided):
      1. Read RAW_TP9/AF7/AF8/TP10 with quality mask (HeadBandOn==1, HSI≤2)
      2. Fill NaN/Inf in raw EEG (signal dropouts, connection glitches)
      3. Artefact clipping (±5σ per channel)
      4. InvBase normalisation — divides FFT amplitudes by sqrt(baseline_power)
           • baseline_spectrum provided (recommended): population-average resting
             baseline from training subjects → exact match to training pipeline
           • baseline_spectrum=None (fallback): uses clip's own power spectrum
             (poor approximation, only when no baseline is available)
      5. Butterworth band-stack → (20, T) band-filtered amplitude time series
      6. Per-channel z-score (only applied when baseline_spectrum=None)

    Args:
        fp:                 path to *_eeg_cleaned.csv
        min_sec:            minimum duration after quality masking
        fs:                 sampling rate in Hz (default: 256)
        baseline_spectrum:  (4, n_freq) power spectrum of resting baseline.
                            Pass the population-average spectrum from training
                            subjects for distribution-matched inference.

    Returns (20, T) float32.
    Raises ValueError if too few quality samples remain or output contains NaN.
    """
    needed = RAW_EEG_COLS + ['HeadBandOn', 'HSI_TP9', 'HSI_AF7', 'HSI_AF8', 'HSI_TP10']
    try:
        df = pd.read_csv(fp, usecols=lambda c: c in needed)
    except Exception as e:
        raise ValueError(f'Cannot read {fp}: {e}')

    # ── quality mask ──────────────────────────────────────────────────────────
    mask = np.ones(len(df), dtype=bool)
    if 'HeadBandOn' in df.columns:
        mask &= (pd.to_numeric(df['HeadBandOn'], errors='coerce').fillna(0) == 1).values
    for hsi in ('HSI_TP9', 'HSI_AF7', 'HSI_AF8', 'HSI_TP10'):
        if hsi in df.columns:
            mask &= (pd.to_numeric(df[hsi], errors='coerce').fillna(99) <= 2).values

    df_q = df[mask]
    if len(df_q) < int(fs * min_sec):
        raise ValueError(
            f'Only {len(df_q)} quality samples ({len(df_q)/fs:.1f}s) in {fp}, '
            f'need ≥{min_sec}s')

    # ── extract raw EEG in fixed channel order ────────────────────────────────
    missing = [c for c in RAW_EEG_COLS if c not in df_q.columns]
    if missing:
        raise ValueError(f'Missing raw EEG columns in {fp}: {missing}')

    # Fill NaN/Inf in raw EEG (signal dropouts, connection glitches)
    raw_df = df_q[RAW_EEG_COLS].ffill().bfill().fillna(0.0)
    raw    = raw_df.values.T.astype(np.float32)              # (4, T)
    n_qual = int(mask.sum())

    # ── process: clip → InvBase → band-stack (→ z-score if no baseline) ──────
    raw  = _clip_artefacts(raw)                          # (4, T) artefact-clipped
    raw  = _invbase_whiten(raw, baseline_spectrum, fs)   # (4, T) InvBase-normalised
    data = _band_stack(raw, fs=fs)                       # (20, T) band-filtered
    if baseline_spectrum is None:
        data = _zscore_clip(data)                        # fallback z-score only

    if not np.isfinite(data).all():
        raise ValueError(
            f'NaN/Inf in processed EEG from {os.path.basename(fp)} '
            f'({n_qual} quality samples)')
    return data


# ══════════════════════════════════════════════════════════════════════════════
#  PPG CSV → HRV features
# ══════════════════════════════════════════════════════════════════════════════

_ZERO_BVP = np.zeros(8, dtype=np.float32)


def extract_bvp_from_ppg_csv(fp: str) -> np.ndarray:
    """
    Extract 8 HRV features from raw PPG CSV.

    CSV columns: ppg_timestamp_ms, ppg_timestamp_readable, ppg_green, time_s
    Uses scipy peak detection on ppg_green → IBI series → HRV features:
        [HR_mean, RMSSD, pNN50, IBI_range, SDNN, mean_IBI, LF_proxy, HF_proxy]

    Returns float32 (8,) or zeros on failure.
    """
    try:
        df = pd.read_csv(fp)
    except Exception as e:
        print(f'    [bvp warn] Cannot read {fp}: {e}')
        return _ZERO_BVP.copy()

    if 'ppg_green' not in df.columns or 'time_s' not in df.columns:
        print(f'    [bvp warn] Missing ppg_green/time_s columns in {fp}')
        return _ZERO_BVP.copy()

    ppg    = pd.to_numeric(df['ppg_green'], errors='coerce').fillna(0).values.astype(np.float64)
    time_s = pd.to_numeric(df['time_s'],    errors='coerce').ffill().values.astype(np.float64)

    if len(ppg) < 30:
        return _ZERO_BVP.copy()

    # Compute PPG sampling rate from time column
    diffs = np.diff(time_s)
    diffs = diffs[(diffs > 0) & (diffs < 1.0)]
    if len(diffs) == 0:
        return _ZERO_BVP.copy()
    fs_ppg = float(1.0 / np.median(diffs))

    try:
        from scipy.ndimage import uniform_filter1d
        from scipy.signal  import find_peaks

        # Smooth to reduce noise before peak detection
        smooth_w  = max(3, int(fs_ppg / 8))
        ppg_s     = uniform_filter1d(ppg, size=smooth_w)

        # Minimum distance between peaks: 60/220 bpm
        min_dist  = max(int(fs_ppg * 60.0 / 220.0), 3)
        peaks, _  = find_peaks(ppg_s, distance=min_dist,
                               prominence=np.std(ppg_s) * 0.25)

        if len(peaks) < 5:
            print(f'    [bvp warn] Only {len(peaks)} peaks found in {os.path.basename(fp)}')
            return _ZERO_BVP.copy()

        # IBI in milliseconds from peak timestamps
        peak_times_s = time_s[peaks]
        ibi_ms       = np.diff(peak_times_s) * 1000.0

        # Physiological filter: 300–2000 ms = 30–200 bpm
        ibi_ms = ibi_ms[(ibi_ms > 300) & (ibi_ms < 2000)]
        if len(ibi_ms) < 5:
            return _ZERO_BVP.copy()

        diff_ibi  = np.diff(ibi_ms)
        hr_mean   = float(np.mean(60000.0 / ibi_ms))
        rmssd     = float(np.sqrt(np.mean(diff_ibi ** 2)))
        pnn50     = float(np.mean(np.abs(diff_ibi) > 50))
        ibi_range = float(ibi_ms.max() - ibi_ms.min())
        sdnn      = float(ibi_ms.std())
        mean_ibi  = float(ibi_ms.mean())

        # LF/HF proxy via Welch PSD on resampled IBI
        try:
            from scipy.signal import welch, resample
            n_samp  = max(len(ibi_ms), 8)
            ibi_uni = resample(ibi_ms, n_samp)
            f, pxx  = welch(ibi_uni, fs=4.0, nperseg=min(64, n_samp))
            lf = float(np.trapz(pxx[(f >= 0.04) & (f < 0.15)],
                                 f[(f >= 0.04) & (f < 0.15)] + 1e-12))
            hf = float(np.trapz(pxx[(f >= 0.15) & (f < 0.40)],
                                 f[(f >= 0.15) & (f < 0.40)] + 1e-12))
        except Exception:
            lf, hf = 0.0, 0.0

        feat = np.array([hr_mean, rmssd, pnn50, ibi_range, sdnn, mean_ibi, lf, hf],
                        dtype=np.float32)
        return feat if np.all(np.isfinite(feat)) else _ZERO_BVP.copy()

    except Exception as e:
        print(f'    [bvp warn] Feature extraction failed for {os.path.basename(fp)}: {e}')
        return _ZERO_BVP.copy()


# ══════════════════════════════════════════════════════════════════════════════
#  Main loaders
# ══════════════════════════════════════════════════════════════════════════════

def _parse_pid(folder_name: str) -> int:
    """'Participant 10' → 10, 'Participant 2' → 2, etc."""
    m = re.search(r'(\d+)', folder_name)
    return int(m.group(1)) if m else 0


def load_participant_clips(data_dir: str, min_sec: float = 4.0,
                           folder_name: str = '',
                           baseline_spectrum: np.ndarray = None) -> list:
    """
    Scan data_dir for *_eeg_cleaned.csv files and load all clips.

    For each EEG file, looks for a matching *_ppg_hr_ibi_cleaned.csv in the
    same directory to extract BVP features.

    Returns: list of ClipRecord (sorted by emotion then clip_idx).
    """
    patterns = [
        os.path.join(data_dir, '*_eeg_cleaned.csv'),
        os.path.join(data_dir, '**', '*_eeg_cleaned.csv'),
    ]
    eeg_files = sorted({p for pat in patterns
                        for p in glob.glob(pat, recursive=True)})

    if not eeg_files:
        raise FileNotFoundError(f'No *_eeg_cleaned.csv files found under {data_dir}')

    pid    = _parse_pid(folder_name or os.path.basename(data_dir))
    clips  = []
    n_skip = 0

    for fp in eeg_files:
        info = parse_clip_info(fp)
        if info is None:
            print(f'    [skip] Cannot parse filename: {os.path.basename(fp)}')
            n_skip += 1
            continue

        emotion, clip_idx, participant = info
        if emotion not in EMOT2ID:
            print(f'    [skip] Unknown emotion "{emotion}" in {os.path.basename(fp)}')
            n_skip += 1
            continue

        # Load EEG
        try:
            eeg = load_eeg_csv(fp, min_sec=min_sec, baseline_spectrum=baseline_spectrum)
        except ValueError as e:
            print(f'    [skip] {e}')
            n_skip += 1
            continue

        # Find matching PPG file
        base   = os.path.splitext(os.path.basename(fp))[0]
        base   = re.sub(r'_eeg_cleaned$', '', base, flags=re.IGNORECASE)
        ppg_fp = os.path.join(os.path.dirname(fp),
                              f'{base}_ppg_hr_ibi_cleaned.csv')
        if os.path.exists(ppg_fp):
            bvp = extract_bvp_from_ppg_csv(ppg_fp)
        else:
            print(f'    [bvp warn] No PPG file for {os.path.basename(fp)} — using zeros')
            bvp = _ZERO_BVP.copy()

        clips.append(ClipRecord(emotion, clip_idx, eeg, bvp, participant,
                                folder=folder_name, pid=pid))
        print(f'    ✓  {emotion:>12}  clip={clip_idx}  '
              f'shape={eeg.shape}  bvp_ok={not np.all(bvp == 0)}')

    if n_skip:
        print(f'    {n_skip} file(s) skipped')
    clips.sort(key=lambda c: (c.emotion, c.clip_idx))
    return clips


def load_all_participants(root_dir: str, min_sec: float = 4.0,
                          baseline_spectrum: np.ndarray = None) -> list:
    """
    Load clips from ALL participant subdirectories under root_dir.

    Expected layout:
        root_dir/
            Participant 1/  ...eeg_cleaned.csv ...
            Participant 2/  ...
            ...

    Args:
        baseline_spectrum: (4, n_freq) population-average resting power spectrum
            from training subjects.  Pass this to apply exact InvBase normalisation
            so inference data matches the training distribution.  If None, falls
            back to clip-self-whitening (poor approximation).

    Returns: list of ClipRecord across all participants,
             sorted by (pid, emotion, clip_idx).
    """
    # Find immediate subdirectories that look like 'Participant N'
    try:
        entries = os.listdir(root_dir)
    except FileNotFoundError:
        raise FileNotFoundError(f'root_dir not found: {root_dir}')

    subdirs = sorted(
        [e for e in entries
         if os.path.isdir(os.path.join(root_dir, e))
         and re.search(r'participant', e, re.IGNORECASE)],
        key=lambda e: _parse_pid(e)
    )

    if not subdirs:
        # Fall back: treat root_dir itself as a single participant folder
        print(f'  No Participant N subdirs found — treating root as single folder')
        return load_participant_clips(root_dir, min_sec=min_sec,
                                     folder_name=os.path.basename(root_dir),
                                     baseline_spectrum=baseline_spectrum)

    print(f'  Found {len(subdirs)} participant folder(s): {subdirs}')
    all_clips = []
    for folder in subdirs:
        path = os.path.join(root_dir, folder)
        try:
            eeg_count = len(glob.glob(os.path.join(path, '*_eeg_cleaned.csv')))
        except Exception:
            eeg_count = 0
        if eeg_count == 0:
            print(f'  [{folder}] no EEG CSVs — skipping')
            continue
        print(f'\n  [{folder}]  ({eeg_count} EEG file(s))')
        clips = load_participant_clips(path, min_sec=min_sec, folder_name=folder,
                                      baseline_spectrum=baseline_spectrum)
        all_clips.extend(clips)
        print(f'    → {len(clips)} clips loaded')

    all_clips.sort(key=lambda c: (c.pid, c.emotion, c.clip_idx))
    print(f'\n  Total: {len(all_clips)} clips from {len(subdirs)} participant(s)')
    by_emot = {e: sum(1 for c in all_clips if c.emotion == e) for e in CLASS_NAMES}
    print('  Per-emotion: ' + ', '.join(f'{e}={n}' for e, n in by_emot.items()))
    return all_clips


def split_clips(clips: list):
    """
    Split clips into lower-indexed and upper-indexed per (participant, emotion).

    For each (pid, emotion) group:
      • Lower = clip with the smallest clip_idx  → Setup 2 training augmentation
      • Upper = all remaining clips              → Setup 2 test

    Returns (lower_clips, upper_clips).
    """
    lower, upper = [], []
    # Group by (pid, emotion)
    from collections import defaultdict
    groups = defaultdict(list)
    for c in clips:
        groups[(c.pid, c.emotion)].append(c)

    for key, group in groups.items():
        group_sorted = sorted(group, key=lambda c: c.clip_idx)
        lower.append(group_sorted[0])   # smallest index → training
        upper.extend(group_sorted[1:])  # the rest       → test

    return lower, upper
