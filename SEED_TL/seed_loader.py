"""
SEED-IV 4-Channel Data Loader
Loads SEED-IV raw EEG data with only 4 channels matching Muse positions:
    TP7 (idx 32), F7 (idx 5), F8 (idx 13), TP8 (idx 40)
Returns data in LibEER's standard format for direct use with preprocess/label_process.
"""

import numpy as np
import multiprocessing as mp
from functools import partial
from scipy.io import loadmat

# Channel indices in SEED's 62-channel layout that approximate Muse positions
# TP9 → TP7 (idx 32), AF7 → F7 (idx 5), AF8 → F8 (idx 13), TP10 → TP8 (idx 40)
SEED_4CH_INDICES = [32, 5, 13, 40]
SEED_4CH_NAMES = ["TP7", "F7", "F8", "TP8"]
NUM_CHANNELS = 4


def _parallel_read_seedIV_raw_4ch(dir_path, channel_indices, file):
    """Read a single SEED-IV .mat file and extract only specified channels."""
    subject_data = loadmat("{}/{}".format(dir_path, file))
    keys = list(subject_data.keys())[3:]
    trail_datas = []
    for i in range(24):
        trail_data = subject_data[keys[i]]
        # Select only the 4 channels, skip first sample point (matching original repo)
        trail_datas.append(trail_data[channel_indices, 1:])
    return trail_datas


def read_seedIV_raw_4ch(dir_path, channel_indices=None):
    """
    Load SEED-IV raw EEG data with only 4 selected channels.
    
    Args:
        dir_path: Root path to SEED-IV dataset (containing eeg_raw_data/)
        channel_indices: List of 4 channel indices to extract (default: Muse-matched)
    
    Returns:
        eeg_data: (session=3, subject=15, trial=24, channels=4, raw_samples)
        baseline: None
        label: (3, 15, 24) integer labels 0-3
        sample_rate: 200
        channels: 4
    """
    if channel_indices is None:
        channel_indices = SEED_4CH_INDICES

    dir_path_raw = dir_path + "/eeg_raw_data"

    # SEED-IV file list (3 sessions × 15 subjects)
    eeg_files = [
        ['1_20160518.mat', '2_20150915.mat', '3_20150919.mat',
         '4_20151111.mat', '5_20160406.mat', '6_20150507.mat',
         '7_20150715.mat', '8_20151103.mat', '9_20151028.mat',
         '10_20151014.mat', '11_20150916.mat', '12_20150725.mat',
         '13_20151115.mat', '14_20151205.mat', '15_20150508.mat'],
        ['1_20161125.mat', '2_20150920.mat', '3_20151018.mat',
         '4_20151118.mat', '5_20160413.mat', '6_20150511.mat',
         '7_20150717.mat', '8_20151110.mat', '9_20151119.mat',
         '10_20151021.mat', '11_20150921.mat', '12_20150804.mat',
         '13_20151125.mat', '14_20151208.mat', '15_20150514.mat'],
        ['1_20161126.mat', '2_20151012.mat', '3_20151101.mat',
         '4_20151123.mat', '5_20160420.mat', '6_20150512.mat',
         '7_20150721.mat', '8_20151117.mat', '9_20151209.mat',
         '10_20151023.mat', '11_20151011.mat', '12_20150807.mat',
         '13_20161130.mat', '14_20151215.mat', '15_20150527.mat']
    ]

    # SEED-IV labels: 0=neutral, 1=sad, 2=fear, 3=happy
    label = np.zeros((3, 15, 24), dtype=int)
    ses_label1 = [1, 2, 3, 0, 2, 0, 0, 1, 0, 1, 2, 1, 1, 1, 2, 3, 2, 2, 3, 3, 0, 3, 0, 3]
    ses_label2 = [2, 1, 3, 0, 0, 2, 0, 2, 3, 3, 2, 3, 2, 0, 1, 1, 2, 1, 0, 3, 0, 1, 3, 1]
    ses_label3 = [1, 2, 2, 1, 3, 3, 3, 1, 1, 2, 1, 0, 2, 3, 3, 0, 2, 3, 0, 0, 2, 0, 1, 0]
    label[0] = np.tile(ses_label1, (15, 1))
    label[1] = np.tile(ses_label2, (15, 1))
    label[2] = np.tile(ses_label3, (15, 1))

    # Add session folder prefix to each file
    for i, session in enumerate(eeg_files):
        eeg_files[i] = [f"{i + 1}/{sub_file}" for sub_file in session]

    # Load data using multiprocessing (matching repo pattern)
    eeg_data = [[[[] for _ in range(24)] for _ in range(15)] for _ in range(3)]
    for session_files, session_id in zip(eeg_files, range(3)):
        with mp.Pool(processes=5) as pool:
            eeg_data[session_id] = pool.map(
                partial(_parallel_read_seedIV_raw_4ch, dir_path_raw, channel_indices),
                eeg_files[session_id]
            )

    return eeg_data, None, label, 200, NUM_CHANNELS


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python seed_loader.py <SEED_IV_PATH>")
        sys.exit(1)
    
    path = sys.argv[1]
    print(f"Loading SEED-IV from: {path}")
    print(f"Channels: {SEED_4CH_NAMES} (indices: {SEED_4CH_INDICES})")
    
    data, _, label, sr, ch = read_seedIV_raw_4ch(path)
    print(f"\nLoaded successfully!")
    print(f"  Sessions: {len(data)}")
    print(f"  Subjects per session: {len(data[0])}")
    print(f"  Trials per subject: {len(data[0][0])}")
    print(f"  Channels: {ch}")
    print(f"  Sample rate: {sr} Hz")
    print(f"  Trial 0 shape: {np.array(data[0][0][0]).shape}")
    print(f"  Label shape: {label.shape}")
    print(f"  Unique labels: {np.unique(label)}")
