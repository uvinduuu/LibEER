"""
Handcrafted EEG feature extraction (26 features per channel).

Ported from the user's original BiLSTM codebase.
Runs during data loading (pre-computed, not in training loop).

Features per channel (26 total):
    - 5 band Differential Entropy (DE)
    - 5 band log-PSD
    - 4 temporal statistics (mean, std, skew, kurtosis)
    - 5 hemispheric asymmetry (DE left - right)
    - 3 bandpower ratios (theta/alpha, beta/alpha, gamma/beta)
    - 2 Hjorth parameters (mobility, complexity)
    - 2 time-domain (log-variance, zero-crossing rate)
"""

import numpy as np
from scipy.stats import skew, kurtosis

from config import FS

# Frequency bands
BANDS = [
    ("delta", (1, 3)),
    ("theta", (4, 7)),
    ("alpha", (8, 13)),
    ("beta",  (14, 30)),
    ("gamma", (31, 45)),
]

NUM_HANDCRAFTED_FEATURES = 26


def extract_handcrafted_features(X, fs=FS, eps=1e-12):
    """
    Extract 26 handcrafted features per channel per window.

    Args:
        X: (N, 4, T) raw EEG windows — channels first
        fs: sampling frequency
        eps: small constant for numerical stability

    Returns:
        features: (N, 4, 26) float32
    """
    # Transpose to (N, T, C) for easier per-timepoint ops
    X_tc = X.transpose(0, 2, 1)  # (N, T, C)
    N, T, C = X_tc.shape

    # FFT for spectral features
    P = (np.abs(np.fft.rfft(X_tc, axis=1)) ** 2) / T  # (N, F, C)
    freqs = np.fft.rfftfreq(T, d=1.0 / fs)

    # ===== 1) Differential Entropy per band (5) =====
    de_list = []
    band_bp_list = []
    for _, (lo, hi) in BANDS:
        m = (freqs >= lo) & (freqs < hi)
        bp = P[:, m, :].mean(axis=1)  # (N, C)
        band_bp_list.append(bp)
        de = 0.5 * np.log(2 * np.pi * np.e * (bp + eps))  # (N, C)
        de_list.append(de[:, :, np.newaxis])  # (N, C, 1)
    de_all = np.concatenate(de_list, axis=2)  # (N, C, 5)

    # ===== 2) Log-PSD per band (5) =====
    psd_list = []
    for bp in band_bp_list:
        log_psd = np.log(bp + eps)  # (N, C)
        psd_list.append(log_psd[:, :, np.newaxis])
    psd_all = np.concatenate(psd_list, axis=2)  # (N, C, 5)

    # ===== 3) Temporal statistics (4) =====
    temp_mean = X_tc.mean(axis=1)[:, :, np.newaxis]       # (N, C, 1)
    temp_std = X_tc.std(axis=1)[:, :, np.newaxis]          # (N, C, 1)
    temp_skew = skew(X_tc, axis=1)[:, :, np.newaxis]       # (N, C, 1)
    temp_kurt = kurtosis(X_tc, axis=1)[:, :, np.newaxis]   # (N, C, 1)
    temp_all = np.concatenate([temp_mean, temp_std, temp_skew, temp_kurt], axis=2)

    # ===== 4) DE hemispheric asymmetry (5) =====
    # Channels: 0=TP9(left), 1=AF7(left), 2=AF8(right), 3=TP10(right)
    de_left = (de_all[:, 0, :] + de_all[:, 1, :]) / 2   # (N, 5)
    de_right = (de_all[:, 2, :] + de_all[:, 3, :]) / 2  # (N, 5)
    de_asym = de_left - de_right                          # (N, 5)
    # Replicate across all channels
    de_asym_full = np.tile(de_asym[:, np.newaxis, :], (1, C, 1))  # (N, C, 5)

    # ===== 5) Bandpower ratios (3) =====
    delta_bp, theta_bp, alpha_bp, beta_bp, gamma_bp = band_bp_list
    ratio_theta_alpha = (theta_bp + eps) / (alpha_bp + eps)  # (N, C)
    ratio_beta_alpha = (beta_bp + eps) / (alpha_bp + eps)
    ratio_gamma_beta = (gamma_bp + eps) / (beta_bp + eps)
    ratio_all = np.stack([ratio_theta_alpha, ratio_beta_alpha,
                          ratio_gamma_beta], axis=2)  # (N, C, 3)

    # ===== 6) Hjorth parameters (2) =====
    Xc = X_tc - X_tc.mean(axis=1, keepdims=True)
    dx = np.diff(Xc, axis=1)
    var_x = (Xc ** 2).mean(axis=1) + eps  # (N, C)
    var_dx = (dx ** 2).mean(axis=1) + eps
    mobility = np.sqrt(var_dx / var_x)
    ddx = np.diff(dx, axis=1)
    var_ddx = (ddx ** 2).mean(axis=1) + eps
    mobility_dx = np.sqrt(var_ddx / var_dx)
    complexity = mobility_dx / (mobility + eps)
    hjorth_all = np.stack([mobility, complexity], axis=2)  # (N, C, 2)

    # ===== 7) Time-domain extras (2) =====
    log_var = np.log(var_x + eps)  # (N, C)
    sign_x = np.sign(Xc)
    zc = (np.diff(sign_x, axis=1) != 0).sum(axis=1) / float(T - 1 + eps)
    td_extras = np.stack([log_var, zc], axis=2)  # (N, C, 2)

    # ===== Concatenate all 26 features =====
    features = np.concatenate([
        de_all,           # 5
        psd_all,          # 5
        temp_all,         # 4
        de_asym_full,     # 5
        ratio_all,        # 3
        hjorth_all,       # 2
        td_extras,        # 2
    ], axis=2)  # (N, C, 26)

    return features.astype(np.float32)
