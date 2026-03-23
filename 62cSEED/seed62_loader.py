"""
SEED-IV Full 62-Channel Data Loader.

Loads all 62 EEG channels from SEED-IV .mat files.
Uses the same structure as seed_loader.py but without any channel selection.

Returns: trials as list of (62, T) arrays, labels, subject_ids, session_ids.
"""

import numpy as np
import multiprocessing as mp
from functools import partial
from scipy.io import loadmat

NUM_CHANNELS = 62  # Full SEED-IV channel count

# SEED-IV labels per session
_SES_LABELS = [
    [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3],  # session 1
    [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1],  # session 2
    [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0],  # session 3
]

_EEG_FILES = [
    ['1_20160518.mat','2_20150915.mat','3_20150919.mat','4_20151111.mat',
     '5_20160406.mat','6_20150507.mat','7_20150715.mat','8_20151103.mat',
     '9_20151028.mat','10_20151014.mat','11_20150916.mat','12_20150725.mat',
     '13_20151115.mat','14_20151205.mat','15_20150508.mat'],
    ['1_20161125.mat','2_20150920.mat','3_20151018.mat','4_20151118.mat',
     '5_20160413.mat','6_20150511.mat','7_20150717.mat','8_20151110.mat',
     '9_20151119.mat','10_20151021.mat','11_20150921.mat','12_20150804.mat',
     '13_20151125.mat','14_20151208.mat','15_20150514.mat'],
    ['1_20161126.mat','2_20151012.mat','3_20151101.mat','4_20151123.mat',
     '5_20160420.mat','6_20150512.mat','7_20150721.mat','8_20151117.mat',
     '9_20151209.mat','10_20151023.mat','11_20151011.mat','12_20150807.mat',
     '13_20161130.mat','14_20151215.mat','15_20150527.mat'],
]


def _load_subject_file(dir_path, file_relpath):
    """Load a single subject .mat file, return list of 24 trials (each 62, T)."""
    subject_data = loadmat(f"{dir_path}/{file_relpath}")
    keys = list(subject_data.keys())[3:]  # skip __header__, __version__, __globals__
    trials = []
    for i in range(24):
        trial = subject_data[keys[i]]  # shape: (62, T)
        trials.append(trial[:, 1:])    # skip first sample (matching original repo)
    return trials


def load_seediv_62ch(dataset_path, sessions=None):
    """
    Load full 62-channel SEED-IV raw EEG data.

    Args:
        dataset_path: Root path containing eeg_raw_data/
        sessions: List of session indices (1-based). Default: all 3.

    Returns:
        trials:      List of (62, T) float32 arrays
        labels:      List of int labels (0=neutral,1=sad,2=fear,3=happy)
        subject_ids: List of int subject IDs (0-indexed)
        session_ids: List of int session IDs (0-indexed)
    """
    if sessions is None:
        sessions = [1, 2, 3]

    raw_dir = f"{dataset_path}/eeg_raw_data"
    trials, labels, subject_ids, session_ids = [], [], [], []

    for ses_1based in sessions:
        ses_idx = ses_1based - 1
        ses_labels = _SES_LABELS[ses_idx]
        ses_files  = [f"{ses_1based}/{f}" for f in _EEG_FILES[ses_idx]]

        print(f"  Session {ses_1based}: loading {len(ses_files)} subjects...")
        with mp.Pool(processes=min(5, len(ses_files))) as pool:
            subjects_data = pool.map(
                partial(_load_subject_file, raw_dir),
                ses_files
            )

        for subj_idx, subj_trials in enumerate(subjects_data):
            for trial_idx, trial_data in enumerate(subj_trials):
                trials.append(trial_data.astype(np.float32))
                labels.append(ses_labels[trial_idx])
                subject_ids.append(subj_idx)    # 0-14 per session
                session_ids.append(ses_idx)

    print(f"  Total: {len(trials)} trials, {NUM_CHANNELS} channels")
    from collections import Counter
    print(f"  Label dist: {dict(sorted(Counter(labels).items()))}")

    return trials, labels, subject_ids, session_ids


if __name__ == '__main__':
    import sys, time
    if len(sys.argv) < 2:
        print("Usage: python seed62_loader.py <SEED_IV_PATH>")
        sys.exit(1)
    t0 = time.time()
    trials, labels, sids, sesids = load_seediv_62ch(sys.argv[1])
    print(f"  Loaded in {time.time()-t0:.1f}s")
    print(f"  Trial shape example: {trials[0].shape}")
    print(f"  Sessions used: {sorted(set(sesids))}")
