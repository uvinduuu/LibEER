#!/usr/bin/env python3
"""
pretrain_contrastive.py  —  Contrastive Self-Supervised EEG Pre-training
═══════════════════════════════════════════════════════════════════════════════

Trains the MB-BiMamba encoder on ALL available EEG data using NT-Xent
(SimCLR-style) contrastive loss — NO emotion labels required.

Objective:
    Given a 4s EEG window x, produce two differently-augmented views x1, x2.
    Train the encoder so that embeddings of x1 and x2 are similar (pulled
    together), while embeddings of different windows are dissimilar (pushed
    apart).

    This forces the encoder to learn features that are INVARIANT to noise,
    amplitude drift, and single-channel/band failures — which are exactly the
    nuisance factors that hurt cross-subject generalization.

Architecture:
    EEG (B, 20, T)
        → Encoder (MBInvBaseBiMamba._encode)  → (B, d_model)
        → ProjectionHead (MLP)               → (B, proj_dim)
        → L2-normalize                       → z  (B, proj_dim)
    Loss: NT-Xent(z_view1, z_view2)          (InfoNCE, symmetric)

After training:
    • pretrained_encoder_contrastive.pt  ← encoder state_dict
      (compatible with finetune.py — same format as reconstruction pretrain)
    ProjectionHead is discarded.

Augmentations (conservative, EEG-safe):
    • Gaussian noise     (σ = 0.05 × per-channel std)
    • Amplitude jitter   (scale ∈ [0.8, 1.2])
    • Time masking       (zero up to 10% of window)
    • Channel dropout    (zero 1 random channel)
    • Band dropout       (zero all 4 channels of 1 random frequency band)
    Each applied independently with prob=0.5.  Two views are created from
    independent draws, so no two augmented pairs are identical.

Usage (Kaggle):
    python emognition/bimamba_ssl/pretrain_contrastive.py \\
        --data_root  /kaggle/input/.../emognition-processed \\
        --save_dir   /kaggle/working \\
        --d_model 32 --n_layers 2 --epochs 60 --batch_size 128

    # Then fine-tune with the contrastive encoder:
    python emognition/bimamba_ssl/finetune.py \\
        --pretrained   /kaggle/working/pretrained_encoder_contrastive.pt \\
        --data_root    /kaggle/input/.../emognition-processed \\
        --samsung_root /kaggle/input/.../samsung-data \\
        --d_model 32 --n_layers 2 --dropout 0.6 \\
        --epochs 100 --lr 8e-5 --encoder_lr 1e-5 \\
        --weight_decay 0.05 --label_smooth 0.20 --patience 30 --seed 42
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
NUM_BANDS   = 5          # delta, theta, alpha, beta, gamma
NUM_EEG_CH  = 4          # MUSE 2 electrodes
# Band indices in the stacked (20, T) representation:
# band b → channels [b*4 : b*4+4]  for b in 0..4


# ══════════════════════════════════════════════════════════════════════════════
#  Pre-processing  (same as pretrain.py / train_mb_invbase_bimamba.py)
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
#  Data loading  (identical to pretrain.py — all emotion classes)
# ══════════════════════════════════════════════════════════════════════════════

def load_all_eeg_trials(data_root: str, fs: int = FS, min_sec: float = 4.0,
                        sid_prefix: str = ''):
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


def slice_windows(processed_trials: list, window_size: int, step: int) -> list:
    windows = []
    for trial in processed_trials:
        T = trial.shape[1]
        for s in range(0, max(T - window_size + 1, 1), step):
            w = trial[:, s:s + window_size]
            if w.shape[1] < window_size:
                w = np.pad(w, ((0, 0), (0, window_size - w.shape[1])))
            windows.append(w.astype(np.float32))
    return windows


# ══════════════════════════════════════════════════════════════════════════════
#  EEG Augmentations
# ══════════════════════════════════════════════════════════════════════════════

class EEGAugmentor:
    """
    Conservative EEG-safe augmentations for contrastive learning.

    All operations are applied to a single window tensor of shape (C, T)
    where C=20 (5 bands × 4 channels).

    Each augmentation is applied independently with probability `p`.
    Two independent draws of this augmentor on the same window produce
    different but semantically equivalent views.

    Design principle: augmentations should corrupt nuisance variation
    (noise, amplitude scale, single-channel dropout) while preserving
    emotion-relevant spectral content (relative band power ratios,
    temporal envelope of oscillations).
    """

    def __init__(
        self,
        p:               float = 0.5,    # probability of applying each augmentation
        noise_std:       float = 0.05,   # Gaussian noise σ relative to channel std
        amp_range:       tuple = (0.8, 1.2),  # amplitude scale range
        time_mask_frac:  float = 0.10,   # max fraction of T to zero out
        n_ch_drop:       int   = 1,      # number of channels to zero
        band_drop:       bool  = True,   # whether to allow band-level dropout
    ):
        self.p              = p
        self.noise_std      = noise_std
        self.amp_lo, self.amp_hi = amp_range
        self.time_mask_frac = time_mask_frac
        self.n_ch_drop      = n_ch_drop
        self.band_drop      = band_drop

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (C, T) float32 tensor
        Returns:
            augmented (C, T) float32 tensor
        """
        x = x.clone()
        C, T = x.shape

        # 1. Gaussian noise: add small random noise proportional to each
        #    channel's own standard deviation.  Corrupts measurement noise
        #    without changing the signal's frequency content.
        if random.random() < self.p:
            for c in range(C):
                std_c = x[c].std().item()
                if std_c > 1e-8:
                    x[c] = x[c] + torch.randn_like(x[c]) * (self.noise_std * std_c)

        # 2. Amplitude jitter: scale the entire window by a random factor.
        #    Simulates inter-trial amplitude variability.  Preserves relative
        #    band-power ratios.
        if random.random() < self.p:
            scale = random.uniform(self.amp_lo, self.amp_hi)
            x = x * scale

        # 3. Time masking: zero out a random contiguous time segment.
        #    Forces the encoder to not rely on any single temporal region,
        #    encouraging learning of global oscillatory features.
        if random.random() < self.p:
            mask_len = random.randint(1, max(1, int(T * self.time_mask_frac)))
            start    = random.randint(0, T - mask_len)
            x[:, start:start + mask_len] = 0.0

        # 4. Channel dropout: zero out one random channel (one band-electrode
        #    pair).  Forces the encoder to use all 20 channels redundantly
        #    rather than relying on a single strong channel.
        if random.random() < self.p and C > 1:
            drop_ch = random.randint(0, C - 1)
            x[drop_ch, :] = 0.0

        # 5. Band dropout: zero out all 4 channels of one frequency band.
        #    Models a scenario where one frequency range is informative but
        #    noisy.  Encourages multi-band feature extraction.
        #    Band b → channels [b*4 : b*4+4].
        if self.band_drop and random.random() < self.p:
            drop_band = random.randint(0, NUM_BANDS - 1)
            x[drop_band * NUM_EEG_CH:(drop_band + 1) * NUM_EEG_CH, :] = 0.0

        return x


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset — returns two augmented views per window
# ══════════════════════════════════════════════════════════════════════════════

class ContrastiveDataset(Dataset):
    """
    Returns two independently-augmented views of each EEG window.

    For a window x:
        view1 = augmentor(x)   ← drawn independently
        view2 = augmentor(x)   ← drawn independently (different random ops)

    (view1, view2) is the positive pair: they came from the same window.
    All other pairs in the batch are negatives.
    """

    def __init__(self, windows: list, augmentor: EEGAugmentor):
        self.windows   = [torch.from_numpy(w) for w in windows]
        self.augmentor = augmentor

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        x     = self.windows[i]   # (20, T)
        view1 = self.augmentor(x)
        view2 = self.augmentor(x)
        return view1, view2


# ══════════════════════════════════════════════════════════════════════════════
#  Projection Head
# ══════════════════════════════════════════════════════════════════════════════

class ProjectionHead(nn.Module):
    """
    Non-linear projection head from SimCLR.

    Maps the d_model-dim encoder embedding to a proj_dim-dim space where
    the NT-Xent loss is applied.

    Using a separate projection space (rather than applying loss directly
    on encoder embeddings) is important: it lets the encoder retain richer
    features than those needed for the contrastive objective alone, which
    transfers better to downstream classification.

    The projection head is DISCARDED after pre-training; only the encoder
    is kept for fine-tuning.

    Architecture: Linear → BN → ReLU → Linear → L2-normalize
    """

    def __init__(self, d_model: int, proj_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.BatchNorm1d(d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, proj_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, d_model) → L2-normalized (B, proj_dim)"""
        z = self.net(x)
        return F.normalize(z, dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  Contrastive Model  (Encoder + Projection Head)
# ══════════════════════════════════════════════════════════════════════════════

class MBContrastiveModel(nn.Module):
    """
    Wraps MBInvBaseBiMamba encoder + ProjectionHead for contrastive pre-training.

    forward() takes a view (B, 20, T) and returns normalized projection z (B, proj_dim).
    The classification head of the encoder is never used here.
    """

    def __init__(self, encoder: MBInvBaseBiMamba, proj_dim: int = 64):
        super().__init__()
        self.encoder    = encoder
        self.projector  = ProjectionHead(encoder.d_model, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T) — one augmented view
        Returns:
            z: (B, proj_dim) — L2-normalized projection
        """
        emb = self.encoder._encode(x)    # (B, d_model) — GAP embedding, no classifier
        return self.projector(emb)       # (B, proj_dim) normalized

    def save_encoder(self, path: str):
        """Save only the encoder state_dict — loadable by finetune.py."""
        os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
        torch.save(self.encoder.state_dict(), path)
        print(f"  [checkpoint] Encoder saved → {path}")


# ══════════════════════════════════════════════════════════════════════════════
#  NT-Xent Loss  (SimCLR InfoNCE, symmetric)
# ══════════════════════════════════════════════════════════════════════════════

class NTXentLoss(nn.Module):
    """
    Normalized Temperature-scaled Cross-Entropy Loss (NT-Xent).

    For a batch of N windows, each producing two views → 2N embeddings total.
    The positive pair for sample i is (z_i^a, z_i^b).
    All other 2(N-1) pairs in the batch are negatives.

    Loss for a single positive pair (i, j):
        ℓ(i, j) = -log[ exp(sim(zi, zj)/τ) / Σ_{k≠i} exp(sim(zi, zk)/τ) ]

    Total loss = mean over all 2N positive pairs (symmetric).

    Args:
        temperature: τ (default 0.5 — lower = harder negatives, more discriminative)
    """

    def __init__(self, temperature: float = 0.5):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1: (N, proj_dim) — L2-normalized projections of view 1
            z2: (N, proj_dim) — L2-normalized projections of view 2
        Returns:
            scalar loss
        """
        N      = z1.shape[0]
        device = z1.device

        # Concatenate both views: (2N, proj_dim)
        z = torch.cat([z1, z2], dim=0)

        # Full cosine similarity matrix (2N, 2N)
        # Since z is L2-normalized, sim(a,b) = a·b
        sim = torch.mm(z, z.T) / self.tau   # (2N, 2N)

        # Mask out self-similarities on the diagonal
        mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim  = sim.masked_fill(mask, float('-inf'))

        # Labels: for index i in [0, N), positive is at index i+N
        #         for index i in [N, 2N), positive is at index i-N
        labels = torch.cat([torch.arange(N, 2 * N),
                            torch.arange(0, N)], dim=0).to(device)

        # Cross-entropy treats each row as a classification problem:
        # "which of the 2N-1 other embeddings is the positive pair?"
        loss = F.cross_entropy(sim, labels)
        return loss


# ══════════════════════════════════════════════════════════════════════════════
#  Training helpers
# ══════════════════════════════════════════════════════════════════════════════

def setup_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class WarmupCosineScheduler:
    """Linear warmup then cosine decay."""

    def __init__(self, optimizer, warmup_epochs: int, total_epochs: int,
                 min_lr: float = 1e-7):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        self.base_lrs      = [pg['lr'] for pg in optimizer.param_groups]
        self._epoch        = 0

    def step(self):
        self._epoch += 1
        e  = self._epoch
        T  = self.total_epochs
        W  = self.warmup_epochs
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if e <= W:
                lr = base_lr * e / max(W, 1)
            else:
                progress = (e - W) / max(T - W, 1)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * progress))
            pg['lr'] = lr

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, n = 0.0, 0
    for view1, view2 in loader:
        view1, view2 = view1.to(device), view2.to(device)

        optimizer.zero_grad()
        z1 = model(view1)   # (B, proj_dim)
        z2 = model(view2)   # (B, proj_dim)
        loss = criterion(z1, z2)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n          += 1

    scheduler.step()
    return total_loss / max(n, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='MB-BiMamba Contrastive Self-Supervised EEG Pre-training')

    # ── data ──────────────────────────────────────────────────────────────────
    parser.add_argument('--data_root',   required=True,
                        help='Emognition processed dataset root '
                             '(ALL emotion classes used — no filtering)')
    parser.add_argument('--emokey_root', default=None,
                        help='Optional EmoKey dataset root for additional subjects')
    parser.add_argument('--window_sec',  type=float, default=4.0)
    parser.add_argument('--overlap',     type=float, default=0.5,
                        help='Sliding window overlap fraction (default 0.5)')

    # ── augmentation ──────────────────────────────────────────────────────────
    parser.add_argument('--aug_p',           type=float, default=0.5,
                        help='Probability of applying each augmentation')
    parser.add_argument('--noise_std',       type=float, default=0.05,
                        help='Gaussian noise σ relative to channel std')
    parser.add_argument('--amp_lo',          type=float, default=0.8,
                        help='Amplitude jitter lower bound')
    parser.add_argument('--amp_hi',          type=float, default=1.2,
                        help='Amplitude jitter upper bound')
    parser.add_argument('--time_mask_frac',  type=float, default=0.10,
                        help='Max fraction of window to zero out (time masking)')
    parser.add_argument('--no_band_drop',    action='store_true',
                        help='Disable band-level dropout augmentation')

    # ── model ─────────────────────────────────────────────────────────────────
    parser.add_argument('--d_model',        type=int,   default=32,
                        help='MUST match the d_model used in finetune.py')
    parser.add_argument('--n_layers',       type=int,   default=2)
    parser.add_argument('--d_state',        type=int,   default=16)
    parser.add_argument('--dropout',        type=float, default=0.10,
                        help='Low dropout during pre-training (default 0.1)')
    parser.add_argument('--attn_reduction', type=int,   default=4)
    parser.add_argument('--proj_dim',       type=int,   default=64,
                        help='Projection head output dimension')

    # ── contrastive loss ──────────────────────────────────────────────────────
    parser.add_argument('--temperature',  type=float, default=0.5,
                        help='NT-Xent temperature τ (default 0.5)')

    # ── training ──────────────────────────────────────────────────────────────
    parser.add_argument('--epochs',        type=int,   default=60)
    parser.add_argument('--batch_size',    type=int,   default=128,
                        help='Larger batches = more negatives = better contrastive signal')
    parser.add_argument('--lr',            type=float, default=3e-4)
    parser.add_argument('--weight_decay',  type=float, default=1e-4)
    parser.add_argument('--warmup_epochs', type=int,   default=5)
    parser.add_argument('--seed',          type=int,   default=42)

    # ── output ────────────────────────────────────────────────────────────────
    parser.add_argument('--save_dir', default='/kaggle/working')
    parser.add_argument('--device',
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    setup_seed(args.seed)
    device      = torch.device(args.device)
    window_size = int(args.window_sec * FS)
    step        = int(window_size * (1.0 - args.overlap))

    print()
    print('=' * 70)
    print('  MB-BiMamba  —  Contrastive Self-Supervised Pre-training (NT-Xent)')
    print('=' * 70)
    print(f'  data_root   : {args.data_root}')
    if args.emokey_root:
        print(f'  emokey_root : {args.emokey_root}')
    print(f'  window      : {args.window_sec}s → {window_size} samples  '
          f'(overlap={args.overlap})')
    print(f'  augmentation: p={args.aug_p}, noise_std={args.noise_std}, '
          f'amp=[{args.amp_lo},{args.amp_hi}], '
          f'time_mask={args.time_mask_frac}, '
          f'band_drop={not args.no_band_drop}')
    print(f'  model       : d_model={args.d_model}, n_layers={args.n_layers}, '
          f'proj_dim={args.proj_dim}')
    print(f'  contrastive : temperature={args.temperature}')
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
        print(f'  Combined: {len(trials)} trials total')
    print(f'  Done in {time.time()-t0:.1f}s\n')

    # ── Step 2: InvBase baselines ────────────────────────────────────────────
    print('Step 2 — Loading InvBase baselines...')
    t0 = time.time()
    baselines = load_baselines_processed(args.data_root, fs=FS)
    print(f'  {len(baselines)} subject baselines loaded  '
          f'(EmoKey subjects use no-baseline fallback)')
    print(f'  Done in {time.time()-t0:.1f}s\n')

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
    windows = slice_windows(processed, window_size, step)
    print(f'  {len(windows)} training windows of shape (20, {window_size})')
    print(f'  Done in {time.time()-t0:.1f}s\n')

    augmentor = EEGAugmentor(
        p              = args.aug_p,
        noise_std      = args.noise_std,
        amp_range      = (args.amp_lo, args.amp_hi),
        time_mask_frac = args.time_mask_frac,
        n_ch_drop      = 1,
        band_drop      = not args.no_band_drop,
    )
    dataset = ContrastiveDataset(windows, augmentor)
    loader  = DataLoader(dataset, batch_size=args.batch_size,
                         shuffle=True, num_workers=2, pin_memory=True,
                         drop_last=True)   # drop_last: keep batch size stable for NT-Xent
    print(f'  Batches/epoch : {len(loader)}  '
          f'(batch={args.batch_size}, negatives per sample={2*(args.batch_size-1)})\n')

    # ── Step 5: Build model ──────────────────────────────────────────────────
    print('Step 5 — Building encoder + projection head...')
    encoder = MBInvBaseBiMamba(
        in_channels    = IN_CHANNELS,
        num_classes    = 4,              # unused during pre-training
        d_model        = args.d_model,
        n_layers       = args.n_layers,
        d_state        = args.d_state,
        dropout        = args.dropout,
        attn_reduction = args.attn_reduction,
    )
    model = MBContrastiveModel(encoder, proj_dim=args.proj_dim)
    model = model.to(device)

    enc_p  = sum(p.numel() for p in encoder.parameters())
    proj_p = sum(p.numel() for p in model.projector.parameters())
    print(f'  Encoder params    : {enc_p:,}')
    print(f'  Projector params  : {proj_p:,}  (discarded after pre-training)')
    print(f'  Total             : {enc_p + proj_p:,}\n')

    # Sanity check shapes
    with torch.no_grad():
        dummy       = torch.zeros(2, IN_CHANNELS, window_size, device=device)
        z_dummy     = model(dummy)
        assert z_dummy.shape == (2, args.proj_dim), \
            f'Shape mismatch: {z_dummy.shape}'
    print(f'  Shape check OK: (2, {IN_CHANNELS}, {window_size}) '
          f'→ z: {tuple(z_dummy.shape)} ✓\n')

    # ── Step 6: Optimizer + scheduler ────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs,
                                      args.epochs, min_lr=1e-7)
    criterion = NTXentLoss(temperature=args.temperature)

    # ── Step 7: Training loop ────────────────────────────────────────────────
    print('Step 6 — Contrastive pre-training...')
    print(f'   {"Epoch":>6}    {"NT-Xent":>10}    {"LR":>10}')
    print('  ' + '─' * 34)

    best_loss   = float('inf')
    best_enc_sd = None
    t0_total    = time.time()

    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, scheduler,
                               criterion, device)
        lr   = scheduler.get_last_lr()[0]

        if ep % 5 == 0 or ep == 1:
            print(f'  {ep:6d}    {loss:10.6f}    {lr:.2e}')

        if loss < best_loss:
            best_loss   = loss
            best_enc_sd = {k: v.cpu().clone()
                           for k, v in encoder.state_dict().items()}
            # Save checkpoint immediately
            enc_path = os.path.join(args.save_dir, 'pretrained_encoder_contrastive.pt')
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(best_enc_sd, enc_path)
            print(f'  [checkpoint] Encoder saved → {enc_path}')

    elapsed = time.time() - t0_total
    print(f'\n  Pre-training complete in {elapsed/60:.1f} min')
    print(f'  Best NT-Xent loss : {best_loss:.6f}')

    # Save final best encoder (already saved in loop, but save full checkpoint too)
    enc_path  = os.path.join(args.save_dir, 'pretrained_encoder_contrastive.pt')
    full_path = os.path.join(args.save_dir, 'pretrained_contrastive_full.pt')
    torch.save(best_enc_sd, enc_path)
    torch.save({
        'encoder':    best_enc_sd,
        'projector':  model.projector.state_dict(),
        'args':       vars(args),
        'best_loss':  best_loss,
    }, full_path)

    print(f'  Encoder saved   → {enc_path}')
    print(f'  Full ckpt saved → {full_path}')
    print()
    print('  Next step — fine-tune the emotion classifier:')
    print(f'    python emognition/bimamba_ssl/finetune.py \\')
    print(f'      --pretrained   {enc_path} \\')
    print(f'      --data_root    <emognition-processed-path> \\')
    print(f'      --samsung_root <samsung-watch-path> \\')
    print(f'      --d_model {args.d_model} --n_layers {args.n_layers}')


if __name__ == '__main__':
    main()
