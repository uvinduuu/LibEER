"""
SEED-IV Full-Clip Dataset for Mamba classifier.

Loads raw EEG trials (4 channels, ~60s each at 200Hz) and returns them
as fixed-length clips for batched training. Applies bandpass filtering
and optional data augmentation.

Data structure:
    SEED-IV has 3 sessions × 15 subjects × 24 trials = 1,080 trials total.
    Each trial is shape (4, ~12000) at 200Hz.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(data, lowcut=1.0, highcut=50.0, fs=200, order=4):
    """
    Apply zero-phase Butterworth bandpass filter to EEG data.
    
    Args:
        data: (channels, time_samples)
        lowcut: Low frequency cutoff in Hz
        highcut: High frequency cutoff in Hz
        fs: Sampling frequency in Hz
        order: Filter order
    
    Returns:
        Filtered data, same shape as input
    """
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = butter(order, [low, high], btype='band', output='sos')
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = sosfiltfilt(sos, data[ch])
    return filtered


def normalize_trial(data):
    """
    Z-score normalize each channel independently.
    
    Args:
        data: (channels, time_samples)
    Returns:
        Normalized data
    """
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0  # avoid division by zero
    return (data - mean) / std


class SeedIVClipDataset(Dataset):
    """
    PyTorch Dataset for full-clip SEED-IV EEG trials.
    
    Each item is a single trial (~60 seconds) represented as a fixed-length
    tensor of shape (4, fixed_len).
    
    Args:
        raw_data: List of trials, each shape (4, T_i) — variable length
        labels: List of integer labels (0-3)
        fixed_length: Pad/crop all trials to this length. If None, uses max length.
        augment: Whether to apply data augmentation
        transform: Custom transform pipeline (overrides default augmentation)
        filter_eeg: Whether to apply bandpass filtering (default: True)
        normalize: Whether to z-score normalize each channel (default: True)
        sample_rate: Sampling rate in Hz (default: 200)
    """

    def __init__(
        self,
        raw_data,
        labels,
        fixed_length=None,
        augment=False,
        transform=None,
        filter_eeg=True,
        normalize=True,
        sample_rate=200,
    ):
        assert len(raw_data) == len(labels), \
            f"Data ({len(raw_data)}) and labels ({len(labels)}) length mismatch"

        self.sample_rate = sample_rate
        self.augment = augment
        self.normalize = normalize

        # Preprocess: filter + normalize all trials
        self.trials = []
        self.labels = []
        for trial, label in zip(raw_data, labels):
            trial = np.array(trial, dtype=np.float32)
            if filter_eeg:
                trial = bandpass_filter(trial, lowcut=1.0, highcut=50.0, fs=sample_rate)
            if normalize:
                trial = normalize_trial(trial)
            self.trials.append(trial)
            self.labels.append(int(label))

        # Determine fixed length
        lengths = [t.shape[1] for t in self.trials]
        if fixed_length is not None:
            self.fixed_length = fixed_length
        else:
            self.fixed_length = max(lengths)

        print(f"  Dataset: {len(self.trials)} trials, fixed_len={self.fixed_length} "
              f"(min={min(lengths)}, max={max(lengths)}, mean={np.mean(lengths):.0f})")

        # Set up augmentation
        if transform is not None:
            self.transform = transform
        elif augment:
            from augmentations import get_train_augmentations
            self.transform = get_train_augmentations(
                target_length=self.fixed_length,
                sample_rate=sample_rate
            )
        else:
            self.transform = None

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        trial = self.trials[idx].copy()  # (4, T_i)
        label = self.labels[idx]

        # Apply augmentation transforms
        if self.transform is not None:
            trial = self.transform(trial)

        # Record actual length before padding (for masking)
        C, T = trial.shape
        actual_length = min(T, self.fixed_length)

        # Pad or crop to fixed length
        if T < self.fixed_length:
            pad = np.zeros((C, self.fixed_length - T), dtype=trial.dtype)
            trial = np.concatenate([trial, pad], axis=1)
        elif T > self.fixed_length:
            trial = trial[:, :self.fixed_length]

        return torch.from_numpy(trial), label, actual_length


def load_seediv_clips(dataset_path, sessions=None):
    """
    Load all SEED-IV raw trials for the 4 channels.
    Returns flat lists of (trial_data, label) with session/subject info.
    
    Args:
        dataset_path: Path to SEED-IV root folder
        sessions: List of sessions to use (1-indexed), or None for all
    
    Returns:
        trials: list of np.array, each (4, T_i)
        labels: list of int (0-3)
        subject_ids: list of int (0-14, repeated per session)
        session_ids: list of int (0-2)
    """
    import sys, os
    # Import from SEED_TL
    seed_tl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'SEED_TL')
    sys.path.insert(0, seed_tl_dir)
    from seed_loader import read_seedIV_raw_4ch

    raw_data, _, raw_labels, sample_rate, channels = read_seedIV_raw_4ch(dataset_path)

    if sessions is None:
        session_range = range(3)
    else:
        session_range = [s - 1 for s in sessions]  # Convert 1-indexed to 0-indexed

    trials = []
    labels = []
    subject_ids = []
    session_ids = []

    for ses in session_range:
        for subj in range(15):
            for trial_idx in range(24):
                trial_data = np.array(raw_data[ses][subj][trial_idx], dtype=np.float32)
                trial_label = int(raw_labels[ses, subj, trial_idx])
                trials.append(trial_data)
                labels.append(trial_label)
                subject_ids.append(subj)
                session_ids.append(ses)

    print(f"Loaded {len(trials)} trials from {len(session_range)} session(s)")
    print(f"  Label distribution: {dict(zip(*np.unique(labels, return_counts=True)))}")

    return trials, labels, subject_ids, session_ids
