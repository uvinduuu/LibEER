"""
DE-LDS Feature Extractor for Raw EEG.

Computes Differential Entropy (DE) with Linear Dynamic System (LDS) smoothing
from raw EEG signals — the same feature type used in SEED-IV's eeg_feature_smooth/.

This makes Emognition features directly compatible with models trained on SEED-IV.

Pipeline per trial:
  Raw EEG (4ch, T samples)
    → bandpass filter into 5 frequency bands
    → compute DE per 1-second window per band
    → apply LDS (Kalman) smoothing across windows
    → output: (T_windows, 4_channels, 5_bands)

Frequency bands (standard for emotion EEG):
  delta:  1-4  Hz
  theta:  4-8  Hz
  alpha:  8-14 Hz
  beta:  14-31 Hz
  gamma: 31-50 Hz
"""

import numpy as np
from scipy import signal as sp_signal


# ─────────────────────────────────────────────────
# Band definitions (matches SEED-IV DE-LDS convention)
# ─────────────────────────────────────────────────

BANDS = {
    'delta': (1,  4),
    'theta': (4,  8),
    'alpha': (8,  14),
    'beta':  (14, 31),
    'gamma': (31, 50),
}
BAND_NAMES = list(BANDS.keys())  # fixed order: δ, θ, α, β, γ


# ─────────────────────────────────────────────────
# Signal utilities
# ─────────────────────────────────────────────────

def bandpass(x, low, high, fs, order=4):
    """Butterworth bandpass filter. x: (n_channels, T)"""
    nyq = fs / 2.0
    b, a = sp_signal.butter(order, [low / nyq, high / nyq], btype='band')
    return sp_signal.filtfilt(b, a, x, axis=1).astype(np.float32)


def differential_entropy(x):
    """Compute DE from a 1D signal segment.
    DE = 0.5 * log(2πe * σ²)  for Gaussian signal.
    For band-limited EEG: DE ≈ 0.5 * log(2πe * var(x)).
    Clipped to avoid log(0).
    """
    var = np.var(x)
    var = max(var, 1e-10)
    return float(0.5 * np.log(2 * np.pi * np.e * var))


# ─────────────────────────────────────────────────
# LDS Smoothing (Kalman filter)
# ─────────────────────────────────────────────────

def lds_smooth(seq, process_noise=0.1, obs_noise=1.0):
    """Apply Kalman filter smoothing to a 1D sequence of DE values.
    
    This matches the LDS smoothing applied in the SEED-IV preprocessing:
    y_t = x_t + noise  (observation model)
    x_t = x_{t-1}      (state transition: random walk)
    
    Args:
        seq: (T,) array of DE values
        process_noise: Q — state transition noise variance
        obs_noise: R — observation noise variance
    
    Returns:
        smoothed: (T,) array
    """
    T = len(seq)
    if T == 0:
        return seq
    
    # Forward Kalman pass
    x_est = np.zeros(T)
    p_est = np.zeros(T)
    
    x_est[0] = seq[0]
    p_est[0] = 1.0
    
    for t in range(1, T):
        # Predict
        x_pred = x_est[t - 1]
        p_pred = p_est[t - 1] + process_noise
        # Update
        K = p_pred / (p_pred + obs_noise)
        x_est[t] = x_pred + K * (seq[t] - x_pred)
        p_est[t] = (1 - K) * p_pred
    
    return x_est.astype(np.float32)


# ─────────────────────────────────────────────────
# Main DE-LDS Computation
# ─────────────────────────────────────────────────

def compute_delds(raw_eeg, fs, window_sec=1.0, step_sec=None,
                  process_noise=0.1, obs_noise=1.0):
    """
    Compute DE-LDS features from raw EEG.
    
    Args:
        raw_eeg: (n_channels, T) float array — raw EEG signal
        fs: Sampling rate in Hz (e.g., 256 for Muse 2, 200 for SEED-IV)
        window_sec: Window length for DE computation in seconds (default: 1.0)
        step_sec: Stride in seconds (default: same as window_sec = no overlap)
        process_noise: Kalman Q
        obs_noise: Kalman R
    
    Returns:
        features: (n_windows, n_channels, 5) float32 array
                  Compatible with SEED-IV eeg_feature_smooth format after transpose
    
    Example:
        raw = load_muse2_eeg(...)   # (4, T) at 256Hz
        feats = compute_delds(raw, fs=256)
        # feats.shape = (n_windows, 4, 5)
    """
    n_channels, T = raw_eeg.shape
    win_len = int(window_sec * fs)
    step_len = int((step_sec or window_sec) * fs)
    
    if T < win_len:
        # Too short — pad and return single window
        pad = np.zeros((n_channels, win_len - T), dtype=np.float32)
        raw_eeg = np.concatenate([raw_eeg, pad], axis=1)
        T = win_len
    
    # Step 1: bandpass filter into 5 bands
    band_signals = {}
    for band, (low, high) in BANDS.items():
        # Clip high to Nyquist
        high = min(high, fs / 2.0 - 0.5)
        if low >= high:
            band_signals[band] = np.zeros_like(raw_eeg)
        else:
            band_signals[band] = bandpass(raw_eeg, low, high, fs)
    
    # Step 2: compute DE per window per channel per band
    windows = list(range(0, T - win_len + 1, step_len))
    n_windows = len(windows)
    
    # de_raw[band][channel] = list of DE values over windows
    de_raw = {band: np.zeros((n_channels, n_windows), dtype=np.float32)
              for band in BAND_NAMES}
    
    for w_idx, start in enumerate(windows):
        end = start + win_len
        for band in BAND_NAMES:
            seg = band_signals[band][:, start:end]  # (n_ch, win_len)
            for ch in range(n_channels):
                de_raw[band][ch, w_idx] = differential_entropy(seg[ch])
    
    # Step 3: LDS smoothing per channel per band
    de_smooth = np.zeros((n_windows, n_channels, len(BAND_NAMES)), dtype=np.float32)
    for b_idx, band in enumerate(BAND_NAMES):
        for ch in range(n_channels):
            de_smooth[:, ch, b_idx] = lds_smooth(
                de_raw[band][ch], process_noise, obs_noise
            )
    
    return de_smooth  # (n_windows, n_channels, 5)


def compute_delds_batch(trials, fs, window_sec=1.0, step_sec=None,
                        process_noise=0.1, obs_noise=1.0, verbose=True):
    """
    Compute DE-LDS for a list of raw EEG trials.
    
    Args:
        trials: list of (n_channels, T_i) arrays
        fs: Sampling rate
        ...
    
    Returns:
        features: list of (n_windows_i, n_channels, 5) arrays
    """
    features = []
    lengths = []
    for i, trial in enumerate(trials):
        feat = compute_delds(trial, fs=fs, window_sec=window_sec,
                             step_sec=step_sec,
                             process_noise=process_noise,
                             obs_noise=obs_noise)
        features.append(feat)
        lengths.append(feat.shape[0])
        if verbose and (i % 50 == 0 or i == len(trials) - 1):
            print(f"  DE-LDS: {i+1}/{len(trials)} trials | "
                  f"shape: {feat.shape}")
    
    if verbose:
        print(f"  Window count stats: "
              f"min={min(lengths)}, mean={np.mean(lengths):.1f}, max={max(lengths)}")
    
    return features


if __name__ == '__main__':
    # Quick verification test
    fs = 256
    raw = np.random.randn(4, fs * 30).astype(np.float32)  # 30s Muse 2 trial
    feat = compute_delds(raw, fs=fs, window_sec=1.0)
    print(f"Input : {raw.shape}  (4ch, {raw.shape[1]} samples = 30s @ 256Hz)")
    print(f"Output: {feat.shape} (windows, channels, bands)")
    print(f"  δ mean: {feat[:, :, 0].mean():.3f}")
    print(f"  Bands: delta, theta, alpha, beta, gamma")
    print("DE-LDS extractor OK!")
