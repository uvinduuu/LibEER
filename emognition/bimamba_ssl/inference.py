#!/usr/bin/env python3
"""
inference.py  —  Emotion Inference with MB-InvBase-BiMamba
═══════════════════════════════════════════════════════════════════════════════

Predicts the emotion expressed during a single EEG recording using the
trained MB-InvBase-BiMamba + BVP multimodal model.

Input:
    - EEG JSON : Muse 2 recording  (*_STIMULUS_MUSE_cleaned.json format)
    - BVP JSON : Samsung Watch recording  (*_STIMULUS_SAMSUNG_WATCH.json format)
    - Baseline : either a subject baseline JSON for InvBase normalisation,
                 OR the --data_root directory to auto-find it by subject ID

Output:
    Predicted emotion + per-class confidence scores.

Usage:
    python emognition/bimamba_ssl/inference.py \\
        --model   /kaggle/working/best_model.pt \\
        --eeg     /path/to/22_FEAR_STIMULUS_MUSE_cleaned.json \\
        --bvp     /path/to/22_FEAR_STIMULUS_SAMSUNG_WATCH.json \\
        --baseline_root /kaggle/input/datasets/sasinduabewickrema/emognition-processed

    # EEG-only (no BVP):
    python emognition/bimamba_ssl/inference.py \\
        --model   /kaggle/working/best_model.pt \\
        --eeg     /path/to/22_FEAR_STIMULUS_MUSE_cleaned.json \\
        --baseline_root /path/to/processed_dataset

Training (to get the checkpoint):
    python emognition/emognition_mamba/train_mb_invbase_bimamba.py \\
        --data_root /kaggle/input/.../emognition-processed \\
        --samsung_root /kaggle/input/.../emognition \\
        --mode sub_indep --epochs 120 \\
        --save_model /kaggle/working/best_model.pt
"""

import os
import sys
import json
import glob
import argparse

import numpy as np
import torch
import torch.nn.functional as F

# ── path setup ────────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EMOG_DIR   = os.path.dirname(_SCRIPT_DIR)
_MAMBA_DIR  = os.path.join(_EMOG_DIR, 'emognition_mamba')
sys.path.insert(0, _MAMBA_DIR)
sys.path.insert(0, _EMOG_DIR)

from mb_invbase_bimamba_model import MBInvBaseBiMamba, IN_CHANNELS
from invbase import (load_baselines_processed, apply_invbase_to_raw,
                     INVBASE_BAND_HZ, CHANNELS as _EEG_CHANNELS)
from scipy.signal import butter, filtfilt

FS          = 256
NUM_CLASSES = 4
CLASS_NAMES = ['ENTHUSIASM', 'FEAR', 'NEUTRAL', 'SADNESS']


# ══════════════════════════════════════════════════════════════════════════════
#  Pre-processing  (identical to training pipeline)
# ══════════════════════════════════════════════════════════════════════════════

def clip_artefacts(x: np.ndarray, n_sigma: float = 5.0) -> np.ndarray:
    x = x.astype(np.float64, copy=True)
    for c in range(x.shape[0]):
        s = x[c].std()
        if s > 1e-8:
            x[c] = np.clip(x[c], -n_sigma * s, n_sigma * s)
    return x.astype(np.float32)


def _butter_bandpass(lo, hi, fs, order=4):
    nyq  = fs / 2.0
    low  = np.clip(lo / nyq, 1e-6, 1 - 1e-6)
    high = np.clip(hi / nyq, 1e-6, 1 - 1e-6)
    return butter(order, [low, high], btype='band')


def apply_band_stack(trial: np.ndarray, fs: float = FS) -> np.ndarray:
    bands = []
    for (_, lo, hi) in INVBASE_BAND_HZ:
        b, a = _butter_bandpass(lo, hi, fs)
        filtered = filtfilt(b, a, trial, axis=1)
        bands.append(filtered.astype(np.float32))
    return np.concatenate(bands, axis=0)   # (20, T)


def preprocess_trial(trial: np.ndarray, baseline_spectrum,
                     fs: float = FS) -> np.ndarray:
    """(4, T) raw EEG → (20, T) processed."""
    trial = clip_artefacts(trial)
    trial = apply_invbase_to_raw(trial, baseline_spectrum, fs=fs)
    proc  = apply_band_stack(trial, fs=fs)
    if not np.isfinite(proc).all():
        proc = np.nan_to_num(proc, nan=0.0, posinf=0.0, neginf=0.0)
    return proc


def slice_windows(processed: np.ndarray, window_size: int,
                  step: int) -> np.ndarray:
    """(20, T) → (N, 20, window_size) float32."""
    T = processed.shape[1]
    windows = []
    for s in range(0, max(T - window_size + 1, 1), step):
        w = processed[:, s:s + window_size]
        if w.shape[1] < window_size:
            w = np.pad(w, ((0, 0), (0, window_size - w.shape[1])))
        windows.append(w.astype(np.float32))
    return np.stack(windows, axis=0)   # (N, 20, window_size)


# ══════════════════════════════════════════════════════════════════════════════
#  EEG loading
# ══════════════════════════════════════════════════════════════════════════════

def load_eeg_json(fp: str, fs: int = FS, min_sec: float = 4.0):
    """
    Load a Muse 2 STIMULUS JSON → (4, T) float32 raw EEG array.
    Applies quality mask (HeadBandOn==1, HSI≤2) and DC removal.
    Returns (trial, subject_id) or raises ValueError.
    """
    with open(fp) as f:
        obj = json.load(f)

    name   = os.path.splitext(os.path.basename(fp))[0]
    parts  = name.split('_')
    sid    = parts[0] if len(parts) >= 1 else 'unknown'

    raw_ch = [np.asarray(obj.get(ch, []), dtype=np.float64)
              for ch in _EEG_CHANNELS]

    if any(len(a) == 0 for a in raw_ch):
        raise ValueError(f'Missing EEG channels in {fp}')

    L = min(len(a) for a in raw_ch)
    if L < fs * min_sec:
        raise ValueError(f'Trial too short: {L/fs:.1f}s < {min_sec}s')

    _QUALITY_CHANNELS = ['HSI_TP9', 'HSI_AF7', 'HSI_AF8', 'HSI_TP10']
    mask = np.ones(L, dtype=bool)
    for arr in raw_ch:
        mask &= np.isfinite(arr[:L])

    head_on = np.asarray(obj.get('HeadBandOn', []), dtype=np.float64)[:L]
    if len(head_on) == L:
        mask &= (head_on == 1)
        for qch in _QUALITY_CHANNELS:
            hsi = np.asarray(obj.get(qch, []), dtype=np.float64)[:L]
            if len(hsi) == L:
                mask &= np.isfinite(hsi) & (hsi <= 2)

    raw_ch = [a[:L][mask] for a in raw_ch]
    L = min(len(a) for a in raw_ch)
    if L < fs * min_sec:
        raise ValueError(
            f'Trial too short after quality filter: {L/fs:.1f}s < {min_sec}s '
            f'(headband off or poor contact most of the recording)')

    trial = np.stack(raw_ch, axis=0).astype(np.float32)     # (4, L)
    trial = trial - trial.mean(axis=1, keepdims=True)        # DC removal
    print(f'  EEG loaded  : {L} samples ({L/fs:.1f}s), subject={sid}')
    return trial, sid


# ══════════════════════════════════════════════════════════════════════════════
#  BVP feature extraction (identical to training)
# ══════════════════════════════════════════════════════════════════════════════

def _parse_paired(raw):
    if not isinstance(raw, list) or len(raw) < 5:
        return None
    try:
        return np.array([r[1] for r in raw], dtype=np.float64)
    except Exception:
        return None


def load_bvp_features(fp: str):
    """
    Extract 8 HRV features from a Samsung Watch STIMULUS JSON.
    Returns float32 (8,) or None.
    Features: [HR_mean, RMSSD, pNN50, IBI_range, SDNN, mean_IBI, LF_proxy, HF_proxy]
    """
    try:
        with open(fp) as f:
            obj = json.load(f)
    except Exception as e:
        print(f'  [warn] Cannot read BVP file: {e}')
        return None

    ibi = _parse_paired(obj.get('PPInterval'))
    hr  = _parse_paired(obj.get('heartRate'))

    if ibi is not None:
        ibi = ibi[(ibi > 300) & (ibi < 2000) & np.isfinite(ibi)]
    if hr is not None:
        hr  = hr[(hr > 30)   & (hr  < 220)  & np.isfinite(hr)]

    if ibi is None or len(ibi) < 5:
        print('  [warn] BVP: insufficient IBI data — using zeros')
        return None

    try:
        from scipy.signal import welch, resample
        diff_ibi  = np.diff(ibi)
        hr_mean   = float(np.mean(hr)) if (hr is not None and len(hr) >= 3) \
                    else float(np.mean(60000.0 / ibi))
        rmssd     = float(np.sqrt(np.mean(diff_ibi ** 2)))
        pnn50     = float(np.mean(np.abs(diff_ibi) > 50))
        ibi_range = float(ibi.max() - ibi.min())
        sdnn      = float(ibi.std())
        mean_ibi  = float(ibi.mean())

        # LF/HF proxy
        try:
            fs_ibi   = 4.0
            n_samp   = max(int(len(ibi)), 8)
            ibi_uni  = resample(ibi, n_samp)
            f, pxx   = welch(ibi_uni, fs=fs_ibi, nperseg=min(64, n_samp))
            lf = float(np.trapz(pxx[(f >= 0.04) & (f < 0.15)],
                                 f[(f >= 0.04) & (f < 0.15)] + 1e-12))
            hf = float(np.trapz(pxx[(f >= 0.15) & (f < 0.40)],
                                 f[(f >= 0.15) & (f < 0.40)] + 1e-12))
        except Exception:
            lf, hf = 0.0, 0.0

        feat = np.array([hr_mean, rmssd, pnn50, ibi_range, sdnn, mean_ibi, lf, hf],
                        dtype=np.float32)
        if not np.all(np.isfinite(feat)):
            return None
        print(f'  BVP loaded  : HR_mean={hr_mean:.1f}, RMSSD={rmssd:.1f}, '
              f'pNN50={pnn50:.3f}, SDNN={sdnn:.1f}')
        return feat
    except Exception as e:
        print(f'  [warn] BVP feature extraction failed: {e}')
        return None


# ══════════════════════════════════════════════════════════════════════════════
#  Model loading
# ══════════════════════════════════════════════════════════════════════════════

def load_model(ckpt_path: str, device: torch.device):
    """
    Load a checkpoint saved by train_mb_invbase_bimamba.py --save_model.
    Returns (model, use_bvp, bvp_dim, class_names).
    """
    ckpt = torch.load(ckpt_path, map_location='cpu')

    # Support both formats:
    #   (a) full dict with 'model_state', 'args', 'class_names', etc.
    #   (b) plain state_dict (flat, from pretrain_supcon.py)
    if isinstance(ckpt, dict) and 'model_state' in ckpt:
        state      = ckpt['model_state']
        use_bvp    = ckpt.get('use_bvp', True)
        bvp_dim    = ckpt.get('bvp_dim', 8) if use_bvp else 0
        class_names = ckpt.get('class_names', CLASS_NAMES)
        d_model    = ckpt.get('d_model', 32)
        n_layers   = ckpt.get('n_layers', 2)
        ckpt_args  = ckpt.get('args', {})
        d_state    = ckpt_args.get('d_state', 16)
        dropout    = ckpt_args.get('dropout', 0.55)
        attn_red   = ckpt_args.get('attn_reduction', 4)
    else:
        raise ValueError(
            f'Checkpoint format not recognised. Expected a dict with '
            f"'model_state' key (saved by --save_model). Got keys: "
            f"{list(ckpt.keys()) if isinstance(ckpt, dict) else type(ckpt)}"
        )

    n_classes = len(class_names)
    backbone  = MBInvBaseBiMamba(
        in_channels    = IN_CHANNELS,
        num_classes    = n_classes,
        d_model        = d_model,
        n_layers       = n_layers,
        d_state        = d_state,
        dropout        = dropout,
        attn_reduction = attn_red,
    )

    if use_bvp:
        # Reconstruct MultimodalMBModel locally to avoid circular imports
        import torch.nn as nn
        class _MultimodalMBModel(nn.Module):
            def __init__(self, backbone, bvp_dim, n_classes, dropout):
                super().__init__()
                self.backbone = backbone
                self.bvp_dim  = bvp_dim
                in_dim        = backbone.d_model + bvp_dim
                self.head     = nn.Sequential(
                    nn.LayerNorm(in_dim),
                    nn.Dropout(dropout),
                    nn.Linear(in_dim, 32),
                    nn.ELU(),
                    nn.Dropout(dropout * 0.6),
                    nn.Linear(32, n_classes),
                )
            def forward(self, x_eeg, x_bvp=None):
                emb = self.backbone.get_embedding(x_eeg)
                if self.bvp_dim > 0 and x_bvp is not None:
                    emb = torch.cat([emb, x_bvp], dim=-1)
                return self.head(emb)

        model = _MultimodalMBModel(backbone, bvp_dim, n_classes,
                                   dropout=ckpt_args.get('dropout', 0.55))
    else:
        model = backbone

    model.load_state_dict(state, strict=True)
    model = model.to(device).eval()

    n_params = sum(p.numel() for p in model.parameters())
    print(f'  Model loaded: {n_params:,} params, '
          f'BVP={"ON" if use_bvp else "OFF"}, '
          f'classes={class_names}')
    return model, use_bvp, bvp_dim, class_names


# ══════════════════════════════════════════════════════════════════════════════
#  Inference
# ══════════════════════════════════════════════════════════════════════════════

def predict(model, windows: np.ndarray, bvp_feat, use_bvp: bool,
            bvp_dim: int, device: torch.device, batch_size: int = 64):
    """
    Run model on N windows → clip-level prediction via softmax averaging.

    Args:
        windows:  (N, 20, window_size) float32
        bvp_feat: (8,) float32 or None
        use_bvp:  whether the model uses BVP
        bvp_dim:  expected BVP feature dimension

    Returns:
        probs:      (num_classes,) float  — averaged softmax probabilities
        pred_class: int                   — argmax prediction
    """
    model.eval()
    x = torch.from_numpy(windows).float()           # (N, 20, T)
    N = x.shape[0]

    # Replicate BVP feature to every window (same clip-level features)
    if use_bvp and bvp_feat is not None:
        bvp_t = torch.from_numpy(bvp_feat).float().unsqueeze(0).expand(N, -1)
    else:
        bvp_t = torch.zeros(N, bvp_dim, dtype=torch.float32) if use_bvp else None

    all_logits = []
    with torch.no_grad():
        for i in range(0, N, batch_size):
            xb = x[i:i + batch_size].to(device)
            if use_bvp:
                bb = bvp_t[i:i + batch_size].to(device)
                logits = model(xb, bb)
            else:
                logits = model(xb)
            all_logits.append(logits.cpu())

    logits = torch.cat(all_logits, dim=0)           # (N, num_classes)
    probs  = F.softmax(logits, dim=-1).mean(dim=0)  # average over windows
    pred   = int(probs.argmax().item())
    return probs.numpy(), pred


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='Emotion inference with MB-InvBase-BiMamba')
    parser.add_argument('--model',          required=True,
                        help='Path to checkpoint saved by --save_model')
    parser.add_argument('--eeg',            required=True,
                        help='Path to *_STIMULUS_MUSE_cleaned.json')
    parser.add_argument('--bvp',            default=None,
                        help='Path to *_STIMULUS_SAMSUNG_WATCH.json (optional)')
    parser.add_argument('--baseline_root',  default=None,
                        help='Dataset root to auto-find subject baseline '
                             '(needed for InvBase normalisation)')
    parser.add_argument('--baseline_file',  default=None,
                        help='Explicit path to subject baseline JSON '
                             '(alternative to --baseline_root)')
    parser.add_argument('--window_sec',     type=float, default=4.0)
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')
    args = parser.parse_args()

    device      = torch.device(args.device)
    window_size = int(args.window_sec * FS)
    step        = window_size  # no overlap for inference (non-overlapping windows)

    print()
    print('=' * 60)
    print('  MB-InvBase-BiMamba  —  Emotion Inference')
    print('=' * 60)

    # ── Step 1: Load model ────────────────────────────────────────────────────
    print('\nStep 1 — Loading model...')
    model, use_bvp, bvp_dim, class_names = load_model(args.model, device)

    # ── Step 2: Load EEG ─────────────────────────────────────────────────────
    print('\nStep 2 — Loading EEG...')
    trial, sid = load_eeg_json(args.eeg, fs=FS)

    # ── Step 3: Load baseline ─────────────────────────────────────────────────
    print('\nStep 3 — Loading InvBase baseline...')
    baseline_spectrum = None

    if args.baseline_file:
        # Load a single baseline JSON directly
        baselines = load_baselines_processed(
            os.path.dirname(args.baseline_file), fs=FS)
        name = os.path.splitext(os.path.basename(args.baseline_file))[0]
        key  = name.split('_')[0]
        baseline_spectrum = baselines.get(key)
    elif args.baseline_root:
        baselines = load_baselines_processed(args.baseline_root, fs=FS)
        baseline_spectrum = baselines.get(sid)

    if baseline_spectrum is not None:
        print(f'  Baseline found for subject {sid}')
    else:
        print(f'  [warn] No baseline for subject {sid} — InvBase disabled '
              f'(raw EEG will be band-stacked without normalisation)')

    # ── Step 4: Load BVP ──────────────────────────────────────────────────────
    bvp_feat = None
    if args.bvp:
        print('\nStep 4 — Loading BVP features...')
        bvp_feat = load_bvp_features(args.bvp)
        if bvp_feat is None:
            print('  [warn] BVP features unavailable — using zeros')
    else:
        print('\nStep 4 — BVP: not provided (EEG-only inference)')

    # ── Step 5: Preprocess ────────────────────────────────────────────────────
    print('\nStep 5 — Pre-processing (clip → invbase → band-stack)...')
    processed = preprocess_trial(trial, baseline_spectrum, fs=FS)
    print(f'  Output shape: {processed.shape}  (20 band-channels × {processed.shape[1]} samples)')

    # ── Step 6: Window ────────────────────────────────────────────────────────
    print('\nStep 6 — Windowing...')
    windows = slice_windows(processed, window_size, step)
    print(f'  {len(windows)} windows of {window_size} samples ({args.window_sec}s each)')

    # ── Step 7: Predict ───────────────────────────────────────────────────────
    print('\nStep 7 — Running inference...')
    probs, pred_idx = predict(model, windows, bvp_feat, use_bvp,
                              bvp_dim, device)
    pred_label = class_names[pred_idx]

    # ── Results ───────────────────────────────────────────────────────────────
    print()
    print('=' * 60)
    print(f'  PREDICTED EMOTION : {pred_label}')
    print('=' * 60)
    print()
    print('  Per-class confidence:')
    sorted_idx = np.argsort(probs)[::-1]
    for i in sorted_idx:
        bar    = '█' * int(probs[i] * 30)
        marker = ' ← predicted' if i == pred_idx else ''
        print(f'    {class_names[i]:>12}  {probs[i]*100:5.1f}%  {bar}{marker}')
    print()


if __name__ == '__main__':
    main()
