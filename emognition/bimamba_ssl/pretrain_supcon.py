#!/usr/bin/env python3
"""
pretrain_supcon.py  —  Supervised Contrastive EEG Pre-training (SupCon)
═══════════════════════════════════════════════════════════════════════════════

Why NT-Xent (SimCLR) failed:
    With 41 subjects and 27k windows, NT-Xent treats two augmented views of
    the SAME WINDOW as the only positive pair. With ~3 windows/subject per
    batch, the easiest shortcut is learning subject identity — subjects have
    unique skull geometry, electrode impedance, resting alpha rhythms. The
    encoder became an excellent subject-fingerprinter instead of an emotion
    discriminator.

Why SupCon works here:
    SupCon uses EMOTION LABELS as the similarity criterion.
    Any two windows with the SAME EMOTION (even from DIFFERENT subjects) are
    positives. To pull subject-22-FEAR together with subject-45-FEAR, the
    encoder MUST suppress subject-specific features (they conflict) and learn
    the shared emotion-related spectral structure.

    This is exactly the invariance needed for subject-independent classification.

Pre-training data:
    ALL 451 trials (all 11 emotion classes) — more labels = richer supervision.
    The fine-tuning step later restricts to 4 target emotions.

Architecture:
    EEG (B, 20, T)
        → Encoder._encode()    → (B, d_model)   [GAP, no classifier head]
        → ProjectionHead (MLP) → (B, proj_dim)
        → L2-normalize         → z
    Loss: SupCon(z, emotion_labels)

SupCon Loss:
    For each anchor i:
        - Positives P(i): all other windows in batch with same emotion label
        - Negatives: all windows with different emotion labels

        ℓ_i = -1/|P(i)| Σ_{j∈P(i)} log[
                exp(sim(zi,zj)/τ) / Σ_{k≠i} exp(sim(zi,zk)/τ)
              ]

    This is identical to NT-Xent when |P(i)|=1 (one positive per anchor),
    but generalises to multiple positives — all same-emotion windows in batch.

After training:
    pretrained_encoder_supcon.pt  ← encoder state_dict only
    loadable by finetune.py with --pretrained flag (same format)

Usage (Kaggle):
    # Step A — SupCon pre-train on ALL 451 trials (11 emotion classes)
    python emognition/bimamba_ssl/pretrain_supcon.py \\
        --data_root  /kaggle/input/datasets/sasinduabewickrema/emognition-processed \\
        --save_dir   /kaggle/working \\
        --d_model 32 --n_layers 2 --epochs 60 --batch_size 128

    # Step B — fine-tune on 4-class
    python emognition/bimamba_ssl/finetune.py \\
        --pretrained   /kaggle/working/pretrained_encoder_supcon.pt \\
        --data_root    /kaggle/input/datasets/sasinduabewickrema/emognition-processed \\
        --samsung_root /kaggle/input/datasets/uvindukodikara/emognition \\
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
_EMOG_DIR   = os.path.dirname(_SCRIPT_DIR)
_MAMBA_DIR  = os.path.join(_EMOG_DIR, 'emognition_mamba')
sys.path.insert(0, _MAMBA_DIR)
sys.path.insert(0, _EMOG_DIR)

from mb_invbase_bimamba_model import MBInvBaseBiMamba, IN_CHANNELS
from invbase import (load_baselines_processed, apply_invbase_to_raw,
                     INVBASE_BAND_HZ, CHANNELS as _EEG_CHANNELS)
from scipy.signal import butter, filtfilt

FS       = 256
NUM_BANDS   = 5
NUM_EEG_CH  = 4


# ══════════════════════════════════════════════════════════════════════════════
#  Pre-processing  (identical to all other scripts)
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
    trial = clip_artefacts(trial)
    trial = apply_invbase_to_raw(trial, baseline_spectrum, fs=fs)
    return apply_band_stack(trial, fs=fs)


# ══════════════════════════════════════════════════════════════════════════════
#  Data loading  — ALL 451 trials with emotion labels
# ══════════════════════════════════════════════════════════════════════════════

def load_all_labeled_trials(data_root: str, fs: int = FS, min_sec: float = 4.0):
    """
    Load ALL *_STIMULUS_MUSE_cleaned.json files under data_root.
    Parses emotion label from filename: {subject}_{EMOTION}_STIMULUS_MUSE_cleaned.json

    Returns:
        trials:      list of (4, T) float32 arrays
        emotions:    list of str  (e.g. 'FEAR', 'ENTHUSIASM', ...)
        subject_ids: list of str
    """
    patterns = [
        os.path.join(data_root, '*_STIMULUS_MUSE_cleaned.json'),
        os.path.join(data_root, '*', '*_STIMULUS_MUSE_cleaned.json'),
        os.path.join(data_root, '**', '*_STIMULUS_MUSE_cleaned.json'),
    ]
    files = sorted({p for pat in patterns
                    for p in glob.glob(pat, recursive=True)})
    print(f"  Found {len(files)} *_STIMULUS_MUSE_cleaned.json files")

    trials, emotions, sids = [], [], []
    n_skip = 0
    for fp in files:
        name   = os.path.splitext(os.path.basename(fp))[0]
        parts  = name.split('_')
        if len(parts) < 2:
            n_skip += 1
            continue
        sid    = parts[0]
        emot   = parts[1].upper()
        try:
            with open(fp) as f:
                obj = json.load(f)
            raw_ch = [np.asarray(obj.get(ch, []), dtype=np.float64)
                      for ch in _EEG_CHANNELS]
            if any(len(a) == 0 for a in raw_ch):
                n_skip += 1
                continue
            L = min(len(a) for a in raw_ch)
            if L < fs * min_sec:
                n_skip += 1
                continue

            # ── Quality mask (same as supervised emognition_processed_loader) ──
            # Keeps only samples where headband is on AND all HSI channels ≤ 2.
            # Without this, SURPRISE/DISGUST trials include headset-movement
            # artifacts with huge high-frequency noise that overflows InvBase.
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
                n_skip += 1
                continue

            trial = np.stack(raw_ch, axis=0).astype(np.float32)   # (4, L)
            trial = trial - trial.mean(axis=1, keepdims=True)       # DC removal
            trials.append(trial)
            emotions.append(emot)
            sids.append(sid)
        except Exception as e:
            print(f"    [skip] {os.path.basename(fp)}: {e}")
            n_skip += 1

    emotion_set = sorted(set(emotions))
    emot2id     = {e: i for i, e in enumerate(emotion_set)}
    print(f"  Loaded {len(trials)} trials ({n_skip} skipped), "
          f"{len(set(sids))} unique subjects")
    print(f"  Emotion classes ({len(emotion_set)}): {emotion_set}")
    return trials, emotions, sids, emot2id


def slice_windows_labeled(processed_trials, emotions, window_size, step):
    """
    Slice trials into (20, window_size) windows, propagating the
    trial-level emotion label to every window.

    Returns:
        windows:       list of (20, window_size) float32 arrays
        window_emots:  list of str — emotion label for each window
    """
    windows, window_emots = [], []
    for trial, emot in zip(processed_trials, emotions):
        T = trial.shape[1]
        for s in range(0, max(T - window_size + 1, 1), step):
            w = trial[:, s:s + window_size]
            if w.shape[1] < window_size:
                w = np.pad(w, ((0, 0), (0, window_size - w.shape[1])))
            windows.append(w.astype(np.float32))
            window_emots.append(emot)
    return windows, window_emots


# ══════════════════════════════════════════════════════════════════════════════
#  EEG Augmentor  (same conservative set as pretrain_contrastive.py)
# ══════════════════════════════════════════════════════════════════════════════

class EEGAugmentor:
    """
    Conservative EEG-safe augmentations.
    Each applied independently with probability p.
    """

    def __init__(self, p=0.5, noise_std=0.05, amp_range=(0.8, 1.2),
                 time_mask_frac=0.10, band_drop=True):
        self.p              = p
        self.noise_std      = noise_std
        self.amp_lo, self.amp_hi = amp_range
        self.time_mask_frac = time_mask_frac
        self.band_drop      = band_drop

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        x = x.clone()
        C, T = x.shape

        # Gaussian noise
        if random.random() < self.p:
            for c in range(C):
                std_c = x[c].std().item()
                if std_c > 1e-8:
                    x[c] = x[c] + torch.randn_like(x[c]) * (self.noise_std * std_c)

        # Amplitude jitter
        if random.random() < self.p:
            x = x * random.uniform(self.amp_lo, self.amp_hi)

        # Time masking
        if random.random() < self.p:
            mask_len = random.randint(1, max(1, int(T * self.time_mask_frac)))
            start    = random.randint(0, T - mask_len)
            x[:, start:start + mask_len] = 0.0

        # Channel dropout
        if random.random() < self.p and C > 1:
            x[random.randint(0, C - 1), :] = 0.0

        # Band dropout: zero all 4 ch of one frequency band
        if self.band_drop and random.random() < self.p:
            b = random.randint(0, NUM_BANDS - 1)
            x[b * NUM_EEG_CH:(b + 1) * NUM_EEG_CH, :] = 0.0

        return x


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset  — returns two augmented views + integer emotion label
# ══════════════════════════════════════════════════════════════════════════════

class SupConDataset(Dataset):
    """
    Each item: (view1, view2, label_int)
    view1, view2 are independently-augmented views of the same window.
    label_int is the integer emotion class index (used by SupCon loss).
    """

    def __init__(self, windows: list, labels: list, augmentor: EEGAugmentor):
        self.windows   = [torch.from_numpy(w.astype(np.float32)) for w in windows]
        self.labels    = labels   # list of int
        self.augmentor = augmentor

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        x  = self.windows[i]
        v1 = self.augmentor(x)
        v2 = self.augmentor(x)
        return v1, v2, self.labels[i]


# ══════════════════════════════════════════════════════════════════════════════
#  Projection Head
# ══════════════════════════════════════════════════════════════════════════════

class ProjectionHead(nn.Module):
    """
    Non-linear projection head (SimCLR / SupCon standard).
    Applied after encoder GAP. Discarded after pre-training.
    Linear → BN → ReLU → Linear → L2-normalize
    """

    def __init__(self, d_model: int, proj_dim: int = 64):
        super().__init__()
        # Use LayerNorm instead of BatchNorm1d:
        # - BN depends on batch statistics → unstable with heterogeneous EEG classes
        # - LN normalises per-sample → no batch-size or class-distribution sensitivity
        self.net = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.ReLU(inplace=True),
            nn.Linear(d_model * 2, proj_dim),
            nn.LayerNorm(proj_dim),   # ensures |z| ≈ sqrt(proj_dim) before F.normalize
                                      # → prevents div-by-~0 at random init
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ══════════════════════════════════════════════════════════════════════════════
#  SupCon Loss
# ══════════════════════════════════════════════════════════════════════════════

class SupConLoss(nn.Module):
    """
    Supervised Contrastive Loss — Khosla et al. (NeurIPS 2020).

    For each anchor i:
        Positives P(i) = all OTHER samples in the batch with the same label
        Negatives       = all samples with different labels

        ℓ_i = -1/|P(i)| Σ_{j∈P(i)} log[
                  exp(sim(zi, zj) / τ)
                  ─────────────────────────────────────────
                  Σ_{k≠i} exp(sim(zi, zk) / τ)
              ]

    When called with two views (z1, z2) from the same batch of N samples:
        Concatenate into 2N embeddings.
        Labels are also duplicated: [y, y] so each view of sample i can
        find the other view plus any same-emotion windows as positives.

    Args:
        temperature: τ  (default 0.1 — lower is harder, better for SupCon)
    """

    def __init__(self, temperature: float = 0.1):
        super().__init__()
        self.tau = temperature

    def forward(self, z1: torch.Tensor, z2: torch.Tensor,
                labels: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z1:     (N, proj_dim) L2-normalized  — view 1
            z2:     (N, proj_dim) L2-normalized  — view 2
            labels: (N,) int64 — emotion class per sample
        Returns:
            scalar loss
        """
        N      = z1.shape[0]
        device = z1.device

        # Concatenate views: (2N, proj_dim)
        z      = torch.cat([z1, z2], dim=0)                       # (2N, D)
        labels = torch.cat([labels, labels], dim=0)               # (2N,)

        # Cosine similarity matrix
        sim    = torch.mm(z, z.T) / self.tau                      # (2N, 2N)

        # Remove self-similarity
        self_mask = torch.eye(2 * N, dtype=torch.bool, device=device)
        sim       = sim.masked_fill(self_mask, float('-inf'))

        # Positive mask: same label, not self
        label_eq  = labels.unsqueeze(0) == labels.unsqueeze(1)    # (2N, 2N)
        pos_mask  = label_eq & ~self_mask                         # (2N, 2N)

        # For samples with no positive in the batch, skip (avoid nan)
        has_pos   = pos_mask.any(dim=1)
        if not has_pos.any():
            return torch.tensor(0.0, device=device, requires_grad=True)

        # Log-softmax denominator over all non-self pairs
        log_denom = torch.logsumexp(sim, dim=1)                   # (2N,)

        # For each anchor i, sum log p over positives
        # log p(i,j) = sim(i,j)/τ - log_denom[i]
        log_prob  = sim - log_denom.unsqueeze(1)                  # (2N, 2N)

        # Mean log-prob over positives per anchor
        n_pos     = pos_mask.float().sum(dim=1).clamp(min=1.0)    # (2N,)
        loss_per  = -(log_prob * pos_mask.float()).sum(dim=1) / n_pos  # (2N,)

        # Average only over anchors that have at least one positive
        loss = loss_per[has_pos].mean()
        return loss


# ══════════════════════════════════════════════════════════════════════════════
#  Contrastive Model  (Encoder + Projection Head)
# ══════════════════════════════════════════════════════════════════════════════

class MBSupConModel(nn.Module):
    def __init__(self, encoder: MBInvBaseBiMamba, proj_dim: int = 64):
        super().__init__()
        self.encoder   = encoder
        self.projector = ProjectionHead(encoder.d_model, proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        emb = self.encoder._encode(x)   # (B, d_model) — GAP, no classifier
        return self.projector(emb)      # (B, proj_dim) L2-normalized

    def save_encoder(self, path: str):
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


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-7):
        self.optimizer     = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.min_lr        = min_lr
        self.base_lrs      = [pg['lr'] for pg in optimizer.param_groups]
        self._epoch        = 0

    def step(self):
        self._epoch += 1
        e = self._epoch
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            if e <= self.warmup_epochs:
                lr = base_lr * e / max(self.warmup_epochs, 1)
            else:
                prog = (e - self.warmup_epochs) / max(
                    self.total_epochs - self.warmup_epochs, 1)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + math.cos(math.pi * prog))
            pg['lr'] = lr

    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


def train_one_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, n, n_skipped = 0.0, 0, 0
    for v1, v2, labels in loader:
        v1, v2  = v1.to(device), v2.to(device)
        labels  = labels.long().to(device)

        # BatchNorm1d in ConvStem will corrupt ALL 128 samples in a batch
        # if even one sample has NaN. Guard here as a final safety net.
        if not (torch.isfinite(v1).all() and torch.isfinite(v2).all()):
            n_skipped += 1
            continue

        optimizer.zero_grad()
        z1 = model(v1)
        z2 = model(v2)

        if not (torch.isfinite(z1).all() and torch.isfinite(z2).all()):
            n_skipped += 1
            optimizer.zero_grad()
            continue

        loss = criterion(z1, z2, labels)
        if not torch.isfinite(loss):
            n_skipped += 1
            optimizer.zero_grad()
            continue

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n          += 1

    scheduler.step()
    if n_skipped > 0:
        print(f'    [warn] {n_skipped}/{n+n_skipped} batches skipped (NaN/inf)')
    return total_loss / max(n, 1)


# ══════════════════════════════════════════════════════════════════════════════
#  Batch composition helper — balanced sampling across emotion classes
# ══════════════════════════════════════════════════════════════════════════════

class BalancedEmotionSampler(torch.utils.data.Sampler):
    """
    Each batch contains `n_classes_per_batch` emotion classes,
    with `n_samples_per_class` windows per class.

    batch_size = n_classes_per_batch × n_samples_per_class

    Why: SupCon needs at least 2 samples per class in each batch to form
    positives. Random sampling with 11 classes and 27k windows works but
    can produce batches with no positives for rare classes. Balanced
    sampling guarantees positives exist for every class in the batch.
    """

    def __init__(self, labels: list, n_classes_per_batch: int,
                 n_samples_per_class: int, n_batches: int):
        self.labels               = np.array(labels)
        self.n_classes_per_batch  = n_classes_per_batch
        self.n_samples_per_class  = n_samples_per_class
        self.n_batches            = n_batches
        self.unique_classes       = np.unique(self.labels)

        # Index of windows per class
        self.class_indices = {
            c: np.where(self.labels == c)[0]
            for c in self.unique_classes
        }

    def __iter__(self):
        for _ in range(self.n_batches):
            # pick n_classes_per_batch classes for this batch
            chosen_classes = np.random.choice(
                self.unique_classes,
                size=min(self.n_classes_per_batch, len(self.unique_classes)),
                replace=False)
            batch = []
            for c in chosen_classes:
                idx = self.class_indices[c]
                chosen = np.random.choice(
                    idx, size=self.n_samples_per_class, replace=len(idx) < self.n_samples_per_class)
                batch.extend(chosen.tolist())
            random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.n_batches * self.n_classes_per_batch * self.n_samples_per_class


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description='MB-BiMamba Supervised Contrastive EEG Pre-training (SupCon)')

    # ── data ──────────────────────────────────────────────────────────────────
    parser.add_argument('--data_root',   required=True)
    parser.add_argument('--window_sec',  type=float, default=4.0)
    parser.add_argument('--overlap',     type=float, default=0.5)

    # ── augmentation ──────────────────────────────────────────────────────────
    parser.add_argument('--aug_p',          type=float, default=0.5)
    parser.add_argument('--noise_std',      type=float, default=0.05)
    parser.add_argument('--amp_lo',         type=float, default=0.8)
    parser.add_argument('--amp_hi',         type=float, default=1.2)
    parser.add_argument('--time_mask_frac', type=float, default=0.10)
    parser.add_argument('--no_band_drop',   action='store_true')

    # ── model ─────────────────────────────────────────────────────────────────
    parser.add_argument('--d_model',        type=int,   default=32,
                        help='MUST match d_model in finetune.py')
    parser.add_argument('--n_layers',       type=int,   default=2)
    parser.add_argument('--d_state',        type=int,   default=16)
    parser.add_argument('--dropout',        type=float, default=0.10)
    parser.add_argument('--attn_reduction', type=int,   default=4)
    parser.add_argument('--proj_dim',       type=int,   default=64)

    # ── SupCon loss ───────────────────────────────────────────────────────────
    parser.add_argument('--temperature',      type=float, default=0.3,
                        help='SupCon temperature τ — lower=harder (default 0.3; '
                             'use 0.1 only after a stable warm-up; '
                             'τ=0.1 on random-init EEG causes NaN via BatchNorm instability)')

    # ── batch composition ─────────────────────────────────────────────────────
    parser.add_argument('--n_classes_per_batch',  type=int, default=8,
                        help='Emotion classes per batch (default 8 of 11)')
    parser.add_argument('--n_samples_per_class',  type=int, default=16,
                        help='Windows per class per batch (default 16). '
                             'batch_size = n_classes × n_samples = 128')

    # ── training ──────────────────────────────────────────────────────────────
    parser.add_argument('--epochs',        type=int,   default=60)
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
    batch_size  = args.n_classes_per_batch * args.n_samples_per_class

    print()
    print('=' * 70)
    print('  MB-BiMamba  —  Supervised Contrastive Pre-training (SupCon)')
    print('=' * 70)
    print(f'  data_root   : {args.data_root}')
    print(f'  window      : {args.window_sec}s → {window_size} samples  '
          f'(overlap={args.overlap})')
    print(f'  augmentation: p={args.aug_p}, noise_std={args.noise_std}, '
          f'amp=[{args.amp_lo},{args.amp_hi}], '
          f'time_mask={args.time_mask_frac}, '
          f'band_drop={not args.no_band_drop}')
    print(f'  model       : d_model={args.d_model}, n_layers={args.n_layers}, '
          f'proj_dim={args.proj_dim}')
    print(f'  SupCon      : temperature={args.temperature}')
    print(f'  batch       : {args.n_classes_per_batch} classes × '
          f'{args.n_samples_per_class} samples = {batch_size} per batch')
    print(f'  training    : epochs={args.epochs}, lr={args.lr}')
    print(f'  device      : {args.device}')
    print('=' * 70)

    # ── Step 1: Load ALL labeled EEG trials ──────────────────────────────────
    print('\nStep 1 — Loading EEG trials (ALL emotion classes, with labels)...')
    t0 = time.time()
    trials, emotions, sids, emot2id = load_all_labeled_trials(args.data_root, fs=FS)
    n_classes = len(emot2id)
    print(f'  Done in {time.time()-t0:.1f}s\n')

    # ── Step 2: InvBase baselines ────────────────────────────────────────────
    print('Step 2 — Loading InvBase baselines...')
    t0 = time.time()
    baselines = load_baselines_processed(args.data_root, fs=FS)
    print(f'  {len(baselines)} subject baselines loaded')
    print(f'  Done in {time.time()-t0:.1f}s\n')

    # ── Step 3: Pre-process trials ───────────────────────────────────────────
    print('Step 3 — Pre-processing (clip → invbase → band-stack)...')
    t0 = time.time()
    processed = []
    n_nan_trials = 0
    for i, (trial, sid) in enumerate(zip(trials, sids)):
        proc = process_trial(trial, baselines.get(sid, None), fs=FS)
        # Hard sanitization: nan/inf from filtfilt edge effects or invbase overflow
        # get zeroed here so they never reach the encoder
        if not np.isfinite(proc).all():
            n_nan_trials += 1
            proc = np.nan_to_num(proc, nan=0.0, posinf=0.0, neginf=0.0)
        processed.append(proc)
        if (i + 1) % 50 == 0 or (i + 1) == len(trials):
            print(f'  {i+1}/{len(trials)} processed...', end='\r')
    if n_nan_trials:
        print(f'\n  ⚠  {n_nan_trials} trials had NaN/inf after preprocessing — zeroed out')
    print(f'\n  Done in {time.time()-t0:.1f}s\n')

    # ── Step 4: Window with labels ───────────────────────────────────────────
    print('Step 4 — Windowing...')
    t0 = time.time()
    windows, window_emots = slice_windows_labeled(processed, emotions,
                                                   window_size, step)
    window_labels = [emot2id[e] for e in window_emots]

    # Post-windowing NaN scan — reports any survivors so we know the data is clean
    n_nan_windows = sum(1 for w in windows if not np.isfinite(w).all())
    if n_nan_windows:
        print(f'  ⚠  {n_nan_windows}/{len(windows)} windows still have NaN/inf — zeroing')
        windows = [np.nan_to_num(w, nan=0.0, posinf=0.0, neginf=0.0) for w in windows]
    else:
        print(f'  ✓  All {len(windows)} windows are finite')

    # Print per-class window counts
    from collections import Counter
    cnt = Counter(window_emots)
    print(f'  {len(windows)} windows total:')
    for emot in sorted(cnt):
        print(f'    {emot:>20}: {cnt[emot]:>5} windows')
    print(f'  Done in {time.time()-t0:.1f}s\n')

    # ── Step 5: Build model ──────────────────────────────────────────────────
    print('Step 5 — Building encoder + projection head...')
    encoder = MBInvBaseBiMamba(
        in_channels    = IN_CHANNELS,
        num_classes    = n_classes,   # unused during pre-training
        d_model        = args.d_model,
        n_layers       = args.n_layers,
        d_state        = args.d_state,
        dropout        = args.dropout,
        attn_reduction = args.attn_reduction,
    )
    model = MBSupConModel(encoder, proj_dim=args.proj_dim)
    model = model.to(device)

    enc_p  = sum(p.numel() for p in encoder.parameters())
    proj_p = sum(p.numel() for p in model.projector.parameters())
    print(f'  Encoder params    : {enc_p:,}')
    print(f'  Projector params  : {proj_p:,}  (discarded after pre-training)')
    print(f'  Total             : {enc_p + proj_p:,}\n')

    # Shape check
    with torch.no_grad():
        dummy = torch.zeros(4, IN_CHANNELS, window_size, device=device)
        z_out = model(dummy)
        assert z_out.shape == (4, args.proj_dim), f'Shape mismatch: {z_out.shape}'
    print(f'  Shape check OK: (4, {IN_CHANNELS}, {window_size}) '
          f'→ z: {tuple(z_out.shape)} ✓\n')

    # ── Step 6: DataLoader with balanced sampler ──────────────────────────────
    augmentor = EEGAugmentor(
        p              = args.aug_p,
        noise_std      = args.noise_std,
        amp_range      = (args.amp_lo, args.amp_hi),
        time_mask_frac = args.time_mask_frac,
        band_drop      = not args.no_band_drop,
    )
    dataset = SupConDataset(windows, window_labels, augmentor)

    # n_batches: aim for roughly same iterations as before
    n_batches = max(100, len(windows) // batch_size)

    sampler = BalancedEmotionSampler(
        labels                = window_labels,
        n_classes_per_batch   = min(args.n_classes_per_batch, n_classes),
        n_samples_per_class   = args.n_samples_per_class,
        n_batches             = n_batches,
    )
    loader = DataLoader(dataset, batch_sampler=sampler,
                        num_workers=2, pin_memory=True)

    # Positives per anchor in a typical batch
    avg_pos = args.n_samples_per_class * 2 - 1  # both views count as positives
    print(f'  Batches/epoch   : {n_batches}')
    print(f'  Batch size      : {batch_size}')
    print(f'  Avg positives   : ~{avg_pos} per anchor (same-emotion windows)\n')

    # ── Step 7: Optimizer + scheduler ────────────────────────────────────────
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                  weight_decay=args.weight_decay)
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs,
                                      args.epochs, min_lr=1e-7)
    criterion = SupConLoss(temperature=args.temperature)

    # ── Step 8: Training loop ────────────────────────────────────────────────
    print('Step 6 — SupCon pre-training...')
    print(f'   {"Epoch":>6}    {"SupCon Loss":>12}    {"LR":>10}')
    print('  ' + '─' * 38)

    best_loss   = float('inf')
    best_enc_sd = None
    t0_total    = time.time()
    enc_path    = os.path.join(args.save_dir, 'pretrained_encoder_supcon.pt')

    for ep in range(1, args.epochs + 1):
        loss = train_one_epoch(model, loader, optimizer, scheduler,
                               criterion, device)
        lr   = scheduler.get_last_lr()[0]

        # Guard: nan loss means weights are corrupt — print warning and skip save
        if math.isnan(loss):
            if ep % 5 == 0 or ep == 1:
                print(f'  {ep:6d}    {"NaN — skipped":>12}    {lr:.2e}  ⚠ check temperature/BN')
            continue

        if ep % 5 == 0 or ep == 1:
            print(f'  {ep:6d}    {loss:12.6f}    {lr:.2e}')

        if loss < best_loss:
            best_loss   = loss
            best_enc_sd = {k: v.cpu().clone()
                           for k, v in encoder.state_dict().items()}
            os.makedirs(args.save_dir, exist_ok=True)
            torch.save(best_enc_sd, enc_path)
            print(f'  [checkpoint] Encoder saved → {enc_path}  (loss={loss:.6f})')

    elapsed = time.time() - t0_total
    print(f'\n  Pre-training complete in {elapsed/60:.1f} min')

    # Fallback: if loss was nan every epoch, best_enc_sd is None.
    # Save the final model weights as a last resort so finetune.py gets real tensors.
    if best_enc_sd is None:
        print('  ⚠  WARNING: loss was NaN for all epochs — encoder was never checkpointed.')
        print('  ⚠  Saving final (random-init) encoder as fallback.')
        print('  ⚠  Fix: re-run with higher --temperature (e.g. 0.3) and LayerNorm in head.')
        best_enc_sd = {k: v.cpu().clone() for k, v in encoder.state_dict().items()}
        best_loss   = float('nan')
        os.makedirs(args.save_dir, exist_ok=True)
        torch.save(best_enc_sd, enc_path)

    print(f'  Best SupCon loss : {best_loss:.6f}')

    # Verify no NaN in saved encoder
    nan_count = sum(int(torch.isnan(v).sum()) for v in best_enc_sd.values()
                    if isinstance(v, torch.Tensor))
    total_params = sum(v.numel() for v in best_enc_sd.values()
                       if isinstance(v, torch.Tensor))
    nan_status = '✓ CLEAN' if nan_count == 0 else f'✗ {nan_count}/{total_params} NaN — weights corrupt!'
    print(f'  Encoder NaN check: {nan_status}')

    # Save full checkpoint
    full_path = os.path.join(args.save_dir, 'pretrained_supcon_full.pt')
    torch.save({
        'encoder':    best_enc_sd,
        'projector':  model.projector.state_dict(),
        'emot2id':    emot2id,
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
    print(f'      --d_model {args.d_model} --n_layers {args.n_layers} \\')
    print(f'      --dropout 0.6 --epochs 100 --lr 8e-5 --encoder_lr 1e-5 \\')
    print(f'      --weight_decay 0.05 --label_smooth 0.20 --patience 30')


if __name__ == '__main__':
    main()
