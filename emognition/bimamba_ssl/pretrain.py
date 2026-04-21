#!/usr/bin/env python3
"""
pretrain.py  —  Self-Supervised EEG Pre-training via Reconstruction Autoencoder
═══════════════════════════════════════════════════════════════════════════════

Trains the MB-BiMamba encoder on ALL available EEG data using MSE
reconstruction loss — NO emotion labels required.

Architecture:
    EEG (B, 20, T) ─→ ConvStem ─→ N×BiMambaBlock ─→ sequence (B, T//16, d)
                                                           │
                                              LinearDecoder │
                                                           ↓
                                            reconstructed (B, 20, T)
    Loss: MSE( reconstructed, original )

Pre-training data:
  • All *_STIMULUS_MUSE_cleaned.json (ALL 11 emotion classes, ~451 files)
  • Optional: EmoKey dataset (same MUSE 2 channels, different subjects)

After training:
  • pretrained_encoder.pt  ← encoder state_dict only  (load with finetune.py)
  • pretrained_full.pt     ← full checkpoint (encoder + decoder + optimizer)

Usage (Kaggle):
    python emognition/bimamba_ssl/pretrain.py \\
        --data_root    /kaggle/input/.../emognition-processed \\
        --save_dir     /kaggle/working \\
        --d_model 32 --n_layers 2 --dropout 0.1 \\
        --epochs 60 --batch_size 64 --lr 3e-4

    # also include EmoKey for more subjects:
    python emognition/bimamba_ssl/pretrain.py \\
        --data_root    /kaggle/input/.../emognition-processed \\
        --emokey_root  /kaggle/input/.../emokey \\
        --save_dir     /kaggle/working \\
        --epochs 80 --batch_size 64
"""

import os, sys, glob, json, argparse, time, math, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ── path setup ───────────────────────────────────────────────────────────────
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_EMOG_DIR   = os.path.dirname(_SCRIPT_DIR)               # emognition/
_MAMBA_DIR  = os.path.join(_EMOG_DIR, 'emognition_mamba')
sys.path.insert(0, _MAMBA_DIR)
sys.path.insert(0, _EMOG_DIR)

from mb_invbase_bimamba_model import MBInvBaseBiMamba, IN_CHANNELS
from invbase import (load_baselines_processed, apply_invbase_to_raw,
                     INVBASE_BAND_HZ, CHANNELS as _EEG_CHANNELS)
from scipy.signal import butter, filtfilt

# ── constants ─────────────────────────────────────────────────────────────────
FS          = 256
STEM_STRIDE = 16   # ConvStem total downsampling: stride 4 × stride 4


# ══════════════════════════════════════════════════════════════════════════════
#  Pre-processing  (identical to train_mb_invbase_bimamba.py)
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


def process_trial(trial: np.ndarray, baseline_spectrum,
                  fs: float = FS) -> np.ndarray:
    """clip artefacts → invbase (or pass-through) → band-stack → (20, T)"""
    trial = clip_artefacts(trial)
    trial = apply_invbase_to_raw(trial, baseline_spectrum, fs=fs)
    return apply_band_stack(trial, fs=fs)


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading  (ALL STIMULUS files — no emotion label filtering)
# ══════════════════════════════════════════════════════════════════════════════

def load_all_eeg_trials(data_root: str, fs: int = FS, min_sec: float = 4.0,
                        sid_prefix: str = ''):
    """
    Load ALL *_STIMULUS_MUSE_cleaned.json files under data_root.
    No emotion filtering — uses everything available for pre-training.

    Returns:
        trials:      list of (4, T) float32 raw EEG arrays
        subject_ids: list of str (prefixed by sid_prefix to avoid ID collision)
    """
    patterns = [
        os.path.join(data_root, '*_STIMULUS_MUSE_cleaned.json'),
        os.path.join(data_root, '*', '*_STIMULUS_MUSE_cleaned.json'),
        os.path.join(data_root, '**', '*_STIMULUS_MUSE_cleaned.json'),
    ]
    files = sorted({p for pat in patterns
                    for p in glob.glob(pat, recursive=True)})
    print(f"  Found {len(files)} *_STIMULUS_MUSE_cleaned.json files")

    trials, sids = [], []
    n_skip = 0
    for fp in files:
        name = os.path.splitext(os.path.basename(fp))[0]
        sid  = sid_prefix + name.split('_')[0]
        try:
            with open(fp) as f:
                obj = json.load(f)
            raw_ch = [np.asarray(obj.get(ch, []), dtype=np.float32)
                      for ch in _EEG_CHANNELS]
            if any(len(a) == 0 for a in raw_ch):
                n_skip += 1
                continue
            L = min(len(a) for a in raw_ch)
            if L < fs * min_sec:
                n_skip += 1
                continue
            trial = np.stack(raw_ch, axis=0)[:, :L]   # (4, L)
            trial = np.nan_to_num(trial, nan=0.0, posinf=0.0, neginf=0.0)
            trials.append(trial)
            sids.append(sid)
        except Exception as e:
            print(f"    [skip] {os.path.basename(fp)}: {e}")
            n_skip += 1

    print(f"  Loaded {len(trials)} trials ({n_skip} skipped), "
          f"{len(set(sids))} unique subjects")
    return trials, sids


def window_trials(processed_trials: list, window_size: int, step: int) -> list:
    """Slice (20, T) trials into (20, window_size) windows."""
    windows = []
    for trial in processed_trials:
        T = trial.shape[1]
        for s in range(0, max(T - window_size + 1, 1), step):
            w = trial[:, s:s + window_size]
            if w.shape[1] < window_size:
                w = np.pad(w, ((0, 0), (0, window_size - w.shape[1])))
            windows.append(w.astype(np.float32))
    return windows


class ReconDataset(Dataset):
    """Unsupervised dataset: returns only EEG windows, no labels."""

    def __init__(self, windows: list):
        self.windows = windows

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return torch.from_numpy(self.windows[i])   # (20, window_size)


# ══════════════════════════════════════════════════════════════════════════════
#  Reconstruction Autoencoder
# ══════════════════════════════════════════════════════════════════════════════

class LinearDecoder(nn.Module):
    """
    Lightweight temporal decoder using linear projection + time interleaving.

    Maps BiMamba sequence (B, T_enc, d_model) → reconstructed signal (B, C, T)
    where T = T_enc × stem_stride.

    This is the 1-D analogue of pixel-shuffle:
        Linear: d_model → C × stem_stride  (e.g. 32 → 20×16 = 320)
        Reshape: (B, 64, 320) → (B, 1024, 20) → transpose → (B, 20, 1024)

    No transpose convolutions — simple, fast, and parameter-efficient.
    Total decoder params: d_model × (C × stride) + C×stride  ≈  10 K
    """

    def __init__(self, d_model: int, out_channels: int = IN_CHANNELS,
                 stride: int = STEM_STRIDE):
        super().__init__()
        self.out_channels = out_channels
        self.stride       = stride
        self.proj = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, out_channels * stride),
        )

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        """
        Args:
            seq: (B, T_enc, d_model) — BiMamba output before pooling
        Returns:
            (B, out_channels, T_enc × stride)  e.g. (B, 20, 1024)
        """
        B, T_enc, _ = seq.shape
        proj = self.proj(seq)                                      # (B, T_enc, C*S)
        proj = proj.view(B, T_enc * self.stride, self.out_channels)  # (B, T, C)
        return proj.transpose(1, 2)                                # (B, C, T)


class MBPretrainModel(nn.Module):
    """
    Encoder + Decoder for self-supervised EEG reconstruction.

    Pre-training objective:
        MSE( decoder(encoder_sequence(x)), x )   x ∈ (B, 20, 1024)

    Key method: save_encoder(path) — saves only encoder state_dict,
    which is loaded by finetune.py to initialise the emotion classifier.
    """

    def __init__(self, encoder: MBInvBaseBiMamba,
                 out_channels: int = IN_CHANNELS,
                 stride: int = STEM_STRIDE):
        super().__init__()
        self.encoder = encoder
        self.decoder = LinearDecoder(encoder.d_model, out_channels, stride)

    def _encode_sequence(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward through encoder WITHOUT global average pooling.
        Returns full sequence (B, T//16, d_model) for the decoder to use.
        """
        x = self.encoder.channel_attn(x)          # (B, 20, T)
        x = self.encoder.conv_stem(x)              # (B, d_model, T//16)
        x = x.transpose(1, 2)                      # (B, T//16, d_model)
        for layer in self.encoder.bi_layers:
            x = layer(x)                           # (B, T//16, d_model)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Returns reconstructed (B, 20, T)."""
        seq = self._encode_sequence(x)             # (B, T//16, d_model)
        return self.decoder(seq)                   # (B, 20, T)

    def save_encoder(self, path: str):
        """Save only the encoder state_dict — this is what finetune.py loads."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.encoder.state_dict(), path)
        print(f"  [checkpoint] Encoder saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def train_one_epoch(model, loader, optimizer, scheduler, device):
    model.train()
    total_loss, n = 0.0, 0
    for x in loader:
        x = x.to(device)                          # (B, 20, 1024)
        optimizer.zero_grad()
        recon = model(x)                           # (B, 20, 1024) reconstructed
        loss  = F.mse_loss(recon, x)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
    scheduler.step()
    return total_loss / max(n, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='MB-BiMamba Self-Supervised EEG Pre-training')

    # ── data ──────────────────────────────────────────────────────────────────
    parser.add_argument('--data_root',   required=True,
                        help='Emognition processed dataset root '
                             '(ALL emotion classes are used — no filtering)')
    parser.add_argument('--emokey_root', default=None,
                        help='Optional EmoKey dataset root '
                             '(same MUSE 2 channels, adds more subjects)')
    parser.add_argument('--window_sec',  type=float, default=4.0,
                        help='Window length in seconds (default: 4.0 → 1024 samples)')
    parser.add_argument('--overlap',     type=float, default=0.5,
                        help='Sliding window overlap fraction (default: 0.5)')

    # ── model ─────────────────────────────────────────────────────────────────
    parser.add_argument('--d_model',        type=int,   default=32,
                        help='BiMamba hidden dimension (must match fine-tuning)')
    parser.add_argument('--n_layers',       type=int,   default=2)
    parser.add_argument('--d_state',        type=int,   default=16)
    parser.add_argument('--dropout',        type=float, default=0.10,
                        help='Low dropout for pre-training (default: 0.1)')
    parser.add_argument('--attn_reduction', type=int,   default=4)

    # ── training ──────────────────────────────────────────────────────────────
    parser.add_argument('--epochs',       type=int,   default=60)
    parser.add_argument('--batch_size',   type=int,   default=64)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--seed',         type=int,   default=42)

    # ── output ────────────────────────────────────────────────────────────────
    parser.add_argument('--save_dir',  default='/kaggle/working',
                        help='Directory to write pretrained_encoder.pt')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    setup_seed(args.seed)
    device      = torch.device(args.device)
    window_size = int(args.window_sec * FS)
    step        = int(window_size * (1.0 - args.overlap))

    print()
    print('=' * 70)
    print('  MB-BiMamba  —  Self-Supervised EEG Pre-training')
    print('=' * 70)
    print(f'  data_root   : {args.data_root}')
    if args.emokey_root:
        print(f'  emokey_root : {args.emokey_root}')
    print(f'  window      : {args.window_sec}s → {window_size} samples  '
          f'(overlap={args.overlap})')
    print(f'  model       : d_model={args.d_model}, n_layers={args.n_layers}, '
          f'dropout={args.dropout}')
    print(f'  training    : epochs={args.epochs}, lr={args.lr}, '
          f'bs={args.batch_size}')
    print(f'  device      : {args.device}')
    print('=' * 70)

    # ── Step 1: Load ALL EEG trials ──────────────────────────────────────────
    print('\nStep 1 — Loading EEG trials (ALL emotion classes)...')
    t0 = time.time()
    trials, sids = load_all_eeg_trials(args.data_root, fs=FS)

    if args.emokey_root:
        print(f'\n  + EmoKey: {args.emokey_root}')
        ek_trials, ek_sids = load_all_eeg_trials(
            args.emokey_root, fs=FS, sid_prefix='ek_')
        trials += ek_trials
        sids   += ek_sids
        print(f'  Combined: {len(trials)} trials total\n')
    print(f'  Done in {time.time()-t0:.1f}s\n')

    # ── Step 2: InvBase baselines ────────────────────────────────────────────
    print('Step 2 — Loading InvBase baselines...')
    t0 = time.time()
    baselines = load_baselines_processed(args.data_root, fs=FS)
    # For EmoKey subjects (no Emognition baseline), apply_invbase_to_raw
    # automatically falls back to returning the trial unchanged (no normalization)
    print(f'  {len(baselines)} subject baselines loaded  '
          f'(EmoKey subjects use no-baseline fallback)\n'
          f'  Done in {time.time()-t0:.1f}s\n')

    # ── Step 3: Pre-process trials ───────────────────────────────────────────
    print('Step 3 — Pre-processing (clip → invbase → band-stack)...')
    t0 = time.time()
    processed = []
    for i, (trial, sid) in enumerate(zip(trials, sids)):
        bspec = baselines.get(sid, None)
        proc  = process_trial(trial, bspec, fs=FS)
        processed.append(proc)
        if (i + 1) % 50 == 0 or (i + 1) == len(trials):
            print(f'  {i+1}/{len(trials)} processed...', end='\r')
    print(f'\n  Done in {time.time()-t0:.1f}s\n')

    # ── Step 4: Window ───────────────────────────────────────────────────────
    print('Step 4 — Windowing...')
    t0 = time.time()
    windows = window_trials(processed, window_size, step)
    print(f'  {len(windows)} training windows of shape (20, {window_size})')
    print(f'  Done in {time.time()-t0:.1f}s\n')

    dataset = ReconDataset(windows)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=2, pin_memory=True,
                         drop_last=True)

    # ── Step 5: Build model ──────────────────────────────────────────────────
    print('Step 5 — Building encoder + decoder...')
    encoder = MBInvBaseBiMamba(
        in_channels    = IN_CHANNELS,
        num_classes    = 4,             # unused in pre-training
        d_model        = args.d_model,
        n_layers       = args.n_layers,
        d_state        = args.d_state,
        dropout        = args.dropout,
        attn_reduction = args.attn_reduction,
    )
    model = MBPretrainModel(encoder, out_channels=IN_CHANNELS, stride=STEM_STRIDE)
    model = model.to(device)

    enc_p = sum(p.numel() for p in encoder.parameters())
    dec_p = sum(p.numel() for p in model.decoder.parameters())
    print(f'  Encoder params : {enc_p:,}')
    print(f'  Decoder params : {dec_p:,}  (discarded after pre-training)')
    print(f'  Total          : {enc_p + dec_p:,}\n')

    # Verify decoder output shape
    with torch.no_grad():
        dummy = torch.zeros(2, IN_CHANNELS, window_size, device=device)
        recon = model(dummy)
        assert recon.shape == dummy.shape, \
            f"Decoder shape mismatch: {recon.shape} vs {dummy.shape}"
    print(f'  Shape check OK: (2, {IN_CHANNELS}, {window_size}) → '
          f'{recon.shape} ✓\n')

    # ── Step 6: Optimizer & scheduler ───────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01)

    # ── Step 7: Pre-training loop ────────────────────────────────────────────
    print('Step 6 — Pre-training...')
    print(f'  {"Epoch":>6}  {"MSE Loss":>10}  {"LR":>10}')
    print('  ' + '─' * 32)

    save_enc  = os.path.join(args.save_dir, 'pretrained_encoder.pt')
    save_full = os.path.join(args.save_dir, 'pretrained_full.pt')
    best_loss = float('inf')
    t_total   = time.time()

    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, scheduler, device)
        lr   = optimizer.param_groups[0]['lr']

        if ep % 5 == 0 or ep == 1 or ep == args.epochs:
            print(f'  {ep:>6}  {loss:>10.6f}  {lr:>10.2e}')

        if loss < best_loss:
            best_loss = loss
            model.save_encoder(save_enc)
            torch.save({
                'epoch' : ep,
                'loss'  : loss,
                'model' : model.state_dict(),
                'args'  : vars(args),
            }, save_full)

    elapsed = (time.time() - t_total) / 60
    print(f'\n  Pre-training complete in {elapsed:.1f} min')
    print(f'  Best MSE loss   : {best_loss:.6f}')
    print(f'  Encoder saved   → {save_enc}')
    print(f'  Full ckpt saved → {save_full}')
    print()
    print('  Next step — fine-tune the emotion classifier:')
    print(f'    python emognition/bimamba_ssl/finetune.py \\')
    print(f'      --pretrained   {save_enc} \\')
    print(f'      --data_root    <emognition-processed-path> \\')
    print(f'      --samsung_root <samsung-watch-path> \\')
    print(f'      --d_model {args.d_model} --n_layers {args.n_layers}')


if __name__ == '__main__':
    main()
