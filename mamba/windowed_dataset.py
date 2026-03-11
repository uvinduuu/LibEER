"""
Windowed EEG Dataset for Mamba Classifier.

Splits full-length EEG trials into fixed-size non-overlapping windows.
All windows have the SAME length → NO zero-padding needed.

Key design: Split at TRIAL level first, THEN create windows.
This prevents clip leakage (no windows from the same trial in different splits).

Usage:
    from windowed_dataset import create_windowed_splits, WindowedEEGDataset
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from scipy.signal import butter, sosfiltfilt


def bandpass_filter(data, lowcut=1.0, highcut=50.0, fs=200, order=4):
    """Apply zero-phase Butterworth bandpass filter. Input: (C, T)."""
    nyq = 0.5 * fs
    sos = butter(order, [lowcut / nyq, highcut / nyq], btype='band', output='sos')
    filtered = np.zeros_like(data)
    for ch in range(data.shape[0]):
        filtered[ch] = sosfiltfilt(sos, data[ch])
    return filtered


def normalize_trial(data):
    """Z-score normalize each channel independently. Input: (C, T)."""
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    return (data - mean) / std


def split_trial_into_windows(trial, window_size, min_window_ratio=0.5):
    """
    Split a single trial (C, T) into non-overlapping windows of fixed size.

    Args:
        trial: numpy array (C, T)
        window_size: int, number of samples per window
        min_window_ratio: float, minimum fraction of window_size to keep
                         the last partial window (default: 0.5 = drop if <50%)

    Returns:
        List of numpy arrays, each (C, window_size)
    """
    C, T = trial.shape
    windows = []

    n_full_windows = T // window_size
    remainder = T % window_size

    # Full windows
    for i in range(n_full_windows):
        start = i * window_size
        end = start + window_size
        windows.append(trial[:, start:end])

    # Handle remainder: keep if large enough, discard if too small
    # No padding — we simply discard small remainders
    if remainder >= int(window_size * min_window_ratio) and remainder > 0:
        # Take the LAST window_size samples (overlaps slightly with previous window)
        # This avoids padding while using the remaining data
        windows.append(trial[:, T - window_size:T])

    return windows


class WindowedEEGDataset(Dataset):
    """
    PyTorch Dataset of fixed-size EEG windows. No padding needed.

    Args:
        windows: List of numpy arrays, each (C, window_size)
        labels: List of int labels, same length as windows
        augment: Whether to apply data augmentation
    """

    def __init__(self, windows, labels, augment=False, sample_rate=200):
        assert len(windows) == len(labels), \
            f"Windows ({len(windows)}) and labels ({len(labels)}) mismatch"

        self.windows = windows
        self.labels = labels
        self.augment = augment
        self.window_size = windows[0].shape[1] if len(windows) > 0 else 0

        # Set up augmentation
        if augment:
            from augmentations import GaussianNoise, AmplitudeScaling, TimeMasking, Compose
            self.transform = Compose([
                GaussianNoise(noise_ratio=0.05, p=0.5),
                AmplitudeScaling(min_scale=0.8, max_scale=1.2, p=0.5),
                TimeMasking(num_masks=(1, 2), min_duration=0.3, max_duration=1.0,
                            sample_rate=sample_rate, p=0.3),
            ])
        else:
            self.transform = None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        window = self.windows[idx].copy()  # (C, window_size)
        label = self.labels[idx]

        if self.transform is not None:
            window = self.transform(window)

        return torch.from_numpy(window).float(), label


def create_windowed_splits(
    trials, labels, subject_ids,
    window_size=2000,
    train_ratio=0.7, val_ratio=0.15, test_ratio=0.15,
    filter_eeg=True, normalize=True, sample_rate=200,
    seed=2024, augment_train=True,
):
    """
    Create train/val/test WindowedEEGDatasets with NO clip leakage.

    Process:
        1. Split TRIALS into train/val/test (trial-level, no leakage)
        2. Preprocess each trial (filter + normalize)
        3. Split each trial into fixed-size windows
        4. Create datasets from the windows

    Args:
        trials: list of numpy arrays, each (C, T_i)
        labels: list of int labels
        subject_ids: list of subject IDs (for logging)
        window_size: samples per window (default: 2000 = 10s at 200Hz)
        train/val/test_ratio: split ratios
        filter_eeg: whether to bandpass filter
        normalize: whether to z-score normalize
        sample_rate: Hz
        seed: random seed
        augment_train: whether to augment training data

    Returns:
        train_ds, val_ds, test_ds: WindowedEEGDataset instances
        split_info: dict with statistics
    """
    rng = np.random.RandomState(seed)

    n = len(trials)
    indices = np.arange(n)
    rng.shuffle(indices)

    n_test = int(n * test_ratio)
    n_val = int(n * val_ratio)

    test_trial_idx = indices[:n_test]
    val_trial_idx = indices[n_test:n_test + n_val]
    train_trial_idx = indices[n_test + n_val:]

    def process_and_window(trial_indices, split_name):
        """Preprocess trials and split into windows."""
        all_windows = []
        all_labels = []
        trial_count = len(trial_indices)
        total_discarded = 0

        for idx in trial_indices:
            trial = np.array(trials[idx], dtype=np.float32)

            # Preprocess
            if filter_eeg:
                trial = bandpass_filter(trial, lowcut=1.0, highcut=50.0, fs=sample_rate)
            if normalize:
                trial = normalize_trial(trial)

            label = int(labels[idx])

            # Split into windows
            windows = split_trial_into_windows(trial, window_size)

            for w in windows:
                all_windows.append(w)
                all_labels.append(label)

        print(f"    {split_name:5s}: {trial_count:3d} trials → "
              f"{len(all_windows):5d} windows  "
              f"({len(all_windows)/max(trial_count,1):.1f} windows/trial)")

        return all_windows, all_labels

    print(f"\n  Window size: {window_size} samples ({window_size/sample_rate:.1f}s at {sample_rate}Hz)")
    print(f"  Splitting {n} trials → train/val/test at TRIAL level (no leakage):")
    print(f"    Trial split: train={len(train_trial_idx)}, "
          f"val={len(val_trial_idx)}, test={len(test_trial_idx)}")
    print()

    train_windows, train_labels = process_and_window(train_trial_idx, "Train")
    val_windows, val_labels = process_and_window(val_trial_idx, "Val")
    test_windows, test_labels = process_and_window(test_trial_idx, "Test")

    # Check label distribution
    for name, lbls in [("Train", train_labels), ("Val", val_labels), ("Test", test_labels)]:
        unique, counts = np.unique(lbls, return_counts=True)
        dist = dict(zip(unique, counts))
        print(f"    {name:5s} label dist: {dist}")

    train_ds = WindowedEEGDataset(train_windows, train_labels,
                                   augment=augment_train, sample_rate=sample_rate)
    val_ds = WindowedEEGDataset(val_windows, val_labels,
                                 augment=False, sample_rate=sample_rate)
    test_ds = WindowedEEGDataset(test_windows, test_labels,
                                  augment=False, sample_rate=sample_rate)

    split_info = {
        'n_trials': n,
        'train_trials': len(train_trial_idx),
        'val_trials': len(val_trial_idx),
        'test_trials': len(test_trial_idx),
        'train_windows': len(train_windows),
        'val_windows': len(val_windows),
        'test_windows': len(test_windows),
        'window_size': window_size,
        'window_seconds': window_size / sample_rate,
    }

    return train_ds, val_ds, test_ds, split_info
