"""
EEG Data Augmentation transforms for full-clip training.

All transforms operate on numpy arrays of shape (channels, time_samples).
Designed for 4-channel EEG at 200Hz.
"""

import numpy as np


class Compose:
    """Apply a list of transforms sequentially."""

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, x):
        for t in self.transforms:
            x = t(x)
        return x


class RandomCrop:
    """
    Randomly crop a portion of the trial.
    
    Args:
        min_ratio: Minimum fraction of trial to keep (default: 0.7 = 70%)
        max_ratio: Maximum fraction of trial to keep (default: 1.0 = 100%)
        target_length: If set, always output this length (pad if needed)
    """

    def __init__(self, min_ratio=0.7, max_ratio=1.0, target_length=None):
        self.min_ratio = min_ratio
        self.max_ratio = max_ratio
        self.target_length = target_length

    def __call__(self, x):
        C, T = x.shape
        crop_ratio = np.random.uniform(self.min_ratio, self.max_ratio)
        crop_len = int(T * crop_ratio)
        crop_len = max(crop_len, 1)

        start = np.random.randint(0, max(T - crop_len, 1) + 1)
        x = x[:, start:start + crop_len]

        # Pad to target length if specified
        if self.target_length is not None and x.shape[1] < self.target_length:
            pad_len = self.target_length - x.shape[1]
            x = np.pad(x, ((0, 0), (0, pad_len)), mode='constant', constant_values=0)

        return x


class GaussianNoise:
    """
    Add Gaussian noise scaled to signal standard deviation.
    
    Args:
        noise_ratio: Noise std as fraction of signal std (default: 0.05)
        p: Probability of applying (default: 0.5)
    """

    def __init__(self, noise_ratio=0.05, p=0.5):
        self.noise_ratio = noise_ratio
        self.p = p

    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        std = np.std(x)
        if std < 1e-8:
            return x
        noise = np.random.randn(*x.shape).astype(x.dtype) * std * self.noise_ratio
        return x + noise


class AmplitudeScaling:
    """
    Randomly scale signal amplitude.
    
    Args:
        min_scale: Minimum scale factor (default: 0.8)
        max_scale: Maximum scale factor (default: 1.2)
        p: Probability of applying (default: 0.5)
    """

    def __init__(self, min_scale=0.8, max_scale=1.2, p=0.5):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.p = p

    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        scale = np.random.uniform(self.min_scale, self.max_scale)
        return x * scale


class ChannelDropout:
    """
    Randomly zero out one channel.
    
    Args:
        p: Probability of dropping a channel (default: 0.15)
    """

    def __init__(self, p=0.15):
        self.p = p

    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        C, T = x.shape
        ch = np.random.randint(0, C)
        x = x.copy()
        x[ch, :] = 0.0
        return x


class TimeMasking:
    """
    Zero out random time segments (like SpecAugment for time).
    
    Args:
        num_masks: Number of masks to apply (default: 1-3 random)
        min_duration: Minimum mask duration in seconds (default: 0.5)
        max_duration: Maximum mask duration in seconds (default: 2.0)
        sample_rate: Sampling rate in Hz (default: 200)
        p: Probability of applying (default: 0.5)
    """

    def __init__(self, num_masks=(1, 3), min_duration=0.5, max_duration=2.0,
                 sample_rate=200, p=0.5):
        self.num_masks = num_masks
        self.min_samples = int(min_duration * sample_rate)
        self.max_samples = int(max_duration * sample_rate)
        self.p = p

    def __call__(self, x):
        if np.random.random() > self.p:
            return x
        C, T = x.shape
        x = x.copy()

        n_masks = np.random.randint(self.num_masks[0], self.num_masks[1] + 1)
        for _ in range(n_masks):
            mask_len = np.random.randint(self.min_samples, min(self.max_samples, T) + 1)
            start = np.random.randint(0, max(T - mask_len, 1) + 1)
            x[:, start:start + mask_len] = 0.0

        return x


def get_train_augmentations(target_length=None, sample_rate=200):
    """Get the default training augmentation pipeline."""
    return Compose([
        RandomCrop(min_ratio=0.7, max_ratio=1.0, target_length=target_length),
        GaussianNoise(noise_ratio=0.05, p=0.5),
        AmplitudeScaling(min_scale=0.8, max_scale=1.2, p=0.5),
        ChannelDropout(p=0.15),
        TimeMasking(num_masks=(1, 3), min_duration=0.5, max_duration=2.0,
                    sample_rate=sample_rate, p=0.5),
    ])


def get_eval_augmentations(target_length=None):
    """Get evaluation transforms (just pad/crop to fixed length, no augmentation)."""
    return Compose([
        RandomCrop(min_ratio=1.0, max_ratio=1.0, target_length=target_length),
    ])
