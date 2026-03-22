"""
Differential Entropy (DE) Feature Extraction for EEG.

DE = 0.5 * log(2πe * σ²) = 0.5 * log(2πe) + log(σ)
     [for Gaussian-distributed EEG within each frequency band]

Usage:
    from de_features import compute_de_features

    # trial: (n_channels, T) numpy array
    # returns: (n_channels * n_bands, n_subwindows) numpy array
    features = compute_de_features(trial, sample_rate=200)
"""

import numpy as np
from scipy.signal import butter, sosfiltfilt


# Standard EEG frequency bands for emotion recognition
BANDS = {
    'delta': (1,   4),
    'theta': (4,   8),
    'alpha': (8,  14),
    'beta':  (14, 31),
    'gamma': (31, 51),
}
BAND_NAMES = list(BANDS.keys())  # Consistent ordering
N_BANDS    = len(BANDS)          # 5


def _bandpass(data, low, high, fs, order=4):
    """Zero-phase Butterworth bandpass filter. Input: (n_channels, T)."""
    nyq = 0.5 * fs
    low_n  = max(low  / nyq, 1e-4)
    high_n = min(high / nyq, 1.0 - 1e-4)
    sos = butter(order, [low_n, high_n], btype='band', output='sos')
    return np.stack([sosfiltfilt(sos, ch) for ch in data], axis=0)


def _de_from_signal(signal_1d):
    """
    Compute Differential Entropy of a 1D signal.
    Assumes Gaussian distribution: DE = 0.5 * log(2πe * variance)
    """
    var = np.var(signal_1d)
    if var < 1e-10:
        return 0.0
    return 0.5 * np.log(2 * np.pi * np.e * var)


def compute_de_features(trial, sample_rate=200, subwin_sec=0.5):
    """
    Compute DE features for a single EEG trial.

    Splits the trial into non-overlapping sub-windows of length `subwin_sec`,
    then computes DE per frequency band per channel per sub-window.

    Args:
        trial:       (n_channels, T) float32 numpy array — raw EEG
        sample_rate: Hz (200 for SEED-IV, 256 for Emognition)
        subwin_sec:  Duration of each sub-window in seconds (default: 0.5s)

    Returns:
        features: (n_channels * n_bands, n_subwindows) float32
                  Channel-band ordering: ch0_delta, ch0_theta, ..., ch3_gamma
                  i.e. all bands for channel 0, then all bands for channel 1, etc.
        n_subwindows: int — how many sub-windows were computed
    """
    n_channels, T = trial.shape
    subwin_len = int(subwin_sec * sample_rate)

    if T < subwin_len:
        # Fallback: single sub-window from entire trial
        subwin_len = T

    n_subwindows = T // subwin_len

    # Pre-filter each band
    band_signals = {}
    for band_name, (low, high) in BANDS.items():
        band_signals[band_name] = _bandpass(trial, low, high, sample_rate)

    # Compute DE per sub-window per band per channel
    # Shape: (n_channels, n_bands, n_subwindows)
    de = np.zeros((n_channels, N_BANDS, n_subwindows), dtype=np.float32)

    for sw in range(n_subwindows):
        start = sw * subwin_len
        end   = start + subwin_len
        for bi, band_name in enumerate(BAND_NAMES):
            band_seg = band_signals[band_name][:, start:end]  # (n_channels, subwin_len)
            for ch in range(n_channels):
                de[ch, bi, sw] = _de_from_signal(band_seg[ch])

    # Reshape to (n_channels * n_bands, n_subwindows)
    # Each "virtual channel" is one (channel, band) pair
    features = de.reshape(n_channels * N_BANDS, n_subwindows)
    return features, n_subwindows


def normalize_de(features):
    """
    Z-score normalize DE features across the time (sub-window) axis.
    Input:  (n_channels * n_bands, n_subwindows)
    Output: (n_channels * n_bands, n_subwindows)
    """
    mean = features.mean(axis=1, keepdims=True)
    std  = features.std(axis=1, keepdims=True)
    std[std < 1e-8] = 1.0
    return (features - mean) / std


if __name__ == '__main__':
    # Quick sanity test
    import time
    np.random.seed(42)
    sample_rate = 200
    trial = np.random.randn(4, 2000).astype(np.float32)  # 4ch, 10s @ 200Hz

    t0 = time.time()
    features, n_sw = compute_de_features(trial, sample_rate=sample_rate, subwin_sec=0.5)
    print(f"Input : {trial.shape}  (4 channels, 10s @ {sample_rate}Hz)")
    print(f"Output: {features.shape}  ({N_BANDS} bands × 4 channels = {N_BANDS*4} virtual channels, {n_sw} sub-windows)")
    print(f"Time  : {(time.time()-t0)*1000:.1f}ms")
    print(f"Bands : {BAND_NAMES}")
    print(f"DE range: [{features.min():.3f}, {features.max():.3f}]")
