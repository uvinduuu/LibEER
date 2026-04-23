"""
anushka_loader.py  —  CSV loader for inference participant data
═══════════════════════════════════════════════════════════════════════════════

Input file formats
──────────────────
EEG CSV  :  {emotion}{clip_id}_{name}_eeg_cleaned.csv
  Columns: TimeStamp, Delta_TP9..Gamma_TP10 (20 band-ch), RAW_TP9..TP10,
           HeadBandOn, HSI_TP9..TP10, ...
  → Reads the 20 already-filtered band columns directly (no InvBase needed).

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

# ── EEG band column order (matches training pipeline output) ─────────────────
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

def load_eeg_csv(fp: str, min_sec: float = 4.0, fs: float = 256.0) -> np.ndarray:
    """
    Load the 20 band columns from an EEG CSV.

    Applies quality mask (HeadBandOn==1, HSI ≤ 2) to remove headband-off
    periods, then returns (20, T) float32.

    Raises ValueError if too few quality samples remain.
    """
    usecols = EEG_BAND_COLS.copy()
    for col in ('HeadBandOn', 'HSI_TP9', 'HSI_AF7', 'HSI_AF8', 'HSI_TP10'):
        usecols.append(col)

    try:
        df = pd.read_csv(fp, usecols=lambda c: c in usecols)
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

    # ── extract band columns in fixed order ───────────────────────────────────
    missing = [c for c in EEG_BAND_COLS if c not in df_q.columns]
    if missing:
        raise ValueError(f'Missing EEG band columns in {fp}: {missing}')

    data = df_q[EEG_BAND_COLS].values.T.astype(np.float32)  # (20, T)
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
    time_s = pd.to_numeric(df['time_s'],    errors='coerce').fillna(method='ffill').values.astype(np.float64)

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
                           folder_name: str = '') -> list:
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
            eeg = load_eeg_csv(fp, min_sec=min_sec)
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


def load_all_participants(root_dir: str, min_sec: float = 4.0) -> list:
    """
    Load clips from ALL participant subdirectories under root_dir.

    Expected layout:
        root_dir/
            Participant 1/  ...eeg_cleaned.csv ...
            Participant 2/  ...
            ...

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
                                     folder_name=os.path.basename(root_dir))

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
        clips = load_participant_clips(path, min_sec=min_sec, folder_name=folder)
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
