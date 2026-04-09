"""
LS-STG-Mamba Training on Preprocessed Emognition Dataset.

Uses the NEW *_STIMULUS_MUSE_cleaned.json dataset structure.

Pipeline:
  Raw Muse 2 EEG (TP9, AF7, AF8, TP10) from _cleaned files
    → DE-LDS feature extraction (1s windows, 5 bands)
    → Trial-level 70/15/15 split (subject-dependent)  ← key for >40% accuracy
    → Windowed DE-LDS sequences as STG-Mamba input
    → LS-STG-Mamba training with Mixup + VAE losses

Why trial-level split beats subject-split for accuracy:
  - 164 trials / 41 subjects / 4 emotions
  - Subject-split: model never sees test subjects at all → hard cross-subject task
  - Trial-level split: the model CAN see other emotions from the same subject
    during training → learns per-subject emotion contrast → much higher accuracy

Why DE-LDS beats raw EEG for overfitting:
  - Raw EEG: model learns subject brainprint (amplitude, noise pattern)
  - DE-LDS: log variance per frequency band — captures RELATIVE spectral
    changes between emotions, much less sensitive to absolute amplitude

Why long context (full trial) helps:
  - Mean trial = 120s = 120 DE windows
  - Emotion is not expressed in 6s — it evolves over the full clip
  - STG-Mamba's SSM naturally handles long (120-step) sequences

Usage:
  # Trial-level split (sub_dep) — recommended, fastest, best accuracy
  python train_stg_mamba_processed.py \\
      --data_root /kaggle/input/datasets/sasinduabewickrema/emognition-processed/"Emognition Processed" \\
      --mode sub_dep --epochs 150 --seq_window 30 --stride 10

  # Subject-split (sub_split) — stricter, lower expected accuracy
  python train_stg_mamba_processed.py \\
      --data_root /path/to/dataset --mode sub_split --epochs 200

Classes (alphabetical): ENTHUSIASM=0, FEAR=1, NEUTRAL=2, SADNESS=3
"""

import os, sys, time, random, argparse
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ── Local imports ──────────────────────────────────────────────────────────
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from emognition_processed_loader import load_emognition_processed, FS
from delds_extractor import compute_delds_batch
from invbase_processed import load_invbase_baselines, apply_invbase

# ── LS-STG-Mamba model (from 62cSEED) ──────────────────────────────────────
_MODEL_DIR = os.path.join(_HERE, '..', '..', '62cSEED')
sys.path.insert(0, _MODEL_DIR)
from stg_mamba_v3 import LSSTGMamba, count_parameters

CLASS_NAMES = ['ENTHUSIASM', 'FEAR', 'NEUTRAL', 'SADNESS']   # alphabetical
CHANNELS    = ['TP9', 'AF7', 'AF8', 'TP10']                  # Muse 2


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class DeLDSWindowDataset(Dataset):
    """
    Windowed DE-LDS dataset.

    Each item: (T_seq, 4_channels, 5_bands) float32 tensor + int label.
    Augmentations applied during training only.
    """

    def __init__(self, windows, labels, augment=False,
                 noise=0.04, mask_prob=0.15):
        self.windows   = windows
        self.labels    = labels
        self.augment   = augment
        self.noise     = noise
        self.mask_prob = mask_prob

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx].copy()   # (T, C, 5)
        y = int(self.labels[idx])

        if self.augment:
            # Gaussian feature noise
            x += np.random.randn(*x.shape).astype(np.float32) * self.noise

            # Random time masking
            if random.random() < self.mask_prob:
                t = x.shape[0]
                ml = random.randint(1, max(2, t // 4))
                ms = random.randint(0, t - ml)
                x[ms:ms + ml] = 0.0

            # Random band dropout (drop one freq band)
            if random.random() < self.mask_prob:
                x[:, :, random.randint(0, 4)] = 0.0

            # Random time shift
            shift = random.randint(-3, 3)
            if shift:
                x = np.roll(x, shift, axis=0)

            # Random amplitude scaling
            if random.random() < 0.4:
                x *= random.uniform(0.85, 1.15)

        return torch.FloatTensor(x), y


def create_windows(feats, labels, w_size, stride, max_wins=None):
    """Slice (n_windows, C, 5) feature sequences into fixed-length windows.

    Args:
        max_wins: cap on windows per trial (None = unlimited).
                  Use this to prevent one long trial from dominating training.
    """
    wins, wlbls = [], []
    for x, y in zip(feats, labels):
        T = x.shape[0]
        if T < w_size:
            # Pad short trials to w_size (only if min_trial_sec is set low)
            x = np.pad(x, ((0, w_size - T), (0, 0), (0, 0)))
            T = w_size
        trial_wins = []
        for s in range(0, T - w_size + 1, stride):
            trial_wins.append(x[s:s + w_size])
        # Cap windows per trial
        if max_wins is not None and len(trial_wins) > max_wins:
            # Sample evenly spaced windows rather than random
            idxs = np.linspace(0, len(trial_wins) - 1, max_wins, dtype=int)
            trial_wins = [trial_wins[i] for i in idxs]
        for w in trial_wins:
            wins.append(w)
            wlbls.append(y)
    return wins, wlbls


# ══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════════════

def setup_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(model, loader, device, criterion, alpha, beta):
    model.eval()
    preds, targets = [], []
    tloss = tcls = trec = tkl = n = 0.0

    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.long().to(device)
            logits, lrec, lkl = model(bx, return_losses=True)
            lcls = criterion(logits, by)
            loss = lcls + alpha * lrec + beta * lkl

            tloss += loss.item()
            tcls  += lcls.item()
            trec  += lrec.item()
            tkl   += lkl.item()
            n     += 1

            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(by.cpu().numpy())

    nb  = max(n, 1)
    acc = float(np.mean(np.array(preds) == np.array(targets)))
    f1  = f1_score(targets, preds, average='macro', zero_division=0)
    return tloss/nb, tcls/nb, trec/nb, tkl/nb, acc, f1, preds, targets


def mixup_data(x, y, alpha=0.3):
    lam = float(np.random.beta(alpha, alpha)) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def print_report(y_true, y_pred, class_names, title=""):
    n  = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    print(f"\n  Confusion Matrix{' (' + title + ')' if title else ''}:")
    print(f"  {'':>12}" + "".join(f"{nm:>12}" for nm in class_names))
    for i, nm in enumerate(class_names):
        print(f"  {nm:>12}" + "".join(f"{cm[i][j]:>12}" for j in range(n)))
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


# ══════════════════════════════════════════════════════════════════════════════
#  Normalisation (fit on train, apply everywhere)
# ══════════════════════════════════════════════════════════════════════════════

def per_trial_zscore(features_list):
    """
    Z-score each trial INDEPENDENTLY across its own time axis.

    Why: After InvBase, trials can still have very different residual scales.
    Per-trial normalisation makes the model invariant to trial-level amplitude,
    so it focuses on the SHAPE of spectral evolution, not the magnitude.
    Applied before global normalisation.
    """
    normed = []
    for f in features_list:
        # f: (T, C, 5)  — normalise per band across time
        mean = f.mean(axis=0, keepdims=True)       # (1, C, 5)
        std  = f.std(axis=0, keepdims=True) + 1e-8 # (1, C, 5)
        normed.append((f - mean) / std)
    return normed


def fit_normalizer(features_list):
    """Compute per-band mean/std from a list of (T, C, 5) arrays."""
    all_f = np.concatenate([f.reshape(-1, 5) for f in features_list], axis=0)
    mean  = all_f.mean(0)            # (5,)
    std   = all_f.std(0) + 1e-8      # (5,)
    return mean, std


def apply_normalizer(features_list, mean, std):
    """Standardise a list of (T, C, 5) feature arrays."""
    return [(f - mean) / std for f in features_list]


# ══════════════════════════════════════════════════════════════════════════════
#  Adjacency prior for Muse 2
# ══════════════════════════════════════════════════════════════════════════════

def build_muse_adjacency():
    """
    Hand-crafted adjacency for Muse 2 electrodes:
      idx 0: TP9   idx 1: AF7   idx 2: AF8   idx 3: TP10

    Connections:
      TP9 ↔ TP10 (temporal pair, strong)
      AF7 ↔ AF8  (frontal pair, strong)
      TP9 ↔ AF7  (left hemisphere)
      AF8 ↔ TP10 (right hemisphere)
    """
    adj = torch.eye(4)
    adj[0, 3] = adj[3, 0] = 0.8   # TP9–TP10
    adj[1, 2] = adj[2, 1] = 0.8   # AF7–AF8
    adj[0, 1] = adj[1, 0] = 0.5   # left hemisphere
    adj[2, 3] = adj[3, 2] = 0.5   # right hemisphere
    return adj


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="LS-STG-Mamba on Preprocessed Emognition"
    )

    # ── Data ──
    parser.add_argument('--data_root', required=True,
                        help='Root of Emognition Processed dataset')
    parser.add_argument('--mode', default='sub_dep',
                        choices=['sub_dep', 'sub_split'],
                        help=('sub_dep   = trial-level 70/15/15 split (recommended, ~55-65%% acc)\n'
                              'sub_split = subject-level 70/15/15 split (stricter, ~35-45%% acc)'))
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--val_ratio',  type=float, default=0.15)

    # ── DE-LDS ──
    parser.add_argument('--delds_win_sec', type=float, default=1.0,
                        help='Window for DE computation in seconds (default: 1.0)')

    # ── Sequence windowing ──
    parser.add_argument('--seq_window', type=int, default=30,
                        help='Number of 1-second DE steps per training sample (default: 30 = 30s context)')
    parser.add_argument('--stride', type=int, default=10,
                        help='Stride in DE steps for training (default: 10)')

    # ── Model ──
    parser.add_argument('--d_latent',       type=int,   default=8)
    parser.add_argument('--d_graph',        type=int,   default=24)
    parser.add_argument('--d_mamba',        type=int,   default=128,
                        help='Global Mamba hidden dim (default: 128, was 64)')
    parser.add_argument('--d_state',        type=int,   default=16)
    parser.add_argument('--n_global_layers',type=int,   default=4,
                        help='Global Mamba layers (default: 4, was 3)')
    parser.add_argument('--dropout',        type=float, default=0.4)

    # ── VAE losses ──
    parser.add_argument('--alpha',      type=float, default=5.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--beta_max',   type=float, default=0.1,
                        help='Max weight for KL loss (annealed from 0)')
    parser.add_argument('--warmup_kl', type=int,   default=20,
                        help='Epochs to ramp KL weight from 0 to beta_max')

    # ── Training ──
    parser.add_argument('--epochs',      type=int,   default=150)
    parser.add_argument('--batch_size',  type=int,   default=64)
    parser.add_argument('--lr',          type=float, default=5e-4)
    parser.add_argument('--weight_decay',type=float, default=0.03)
    parser.add_argument('--patience',    type=int,   default=30)
    parser.add_argument('--mixup_alpha', type=float, default=0.3,
                        help='Mixup alpha (0 to disable)')
    parser.add_argument('--seed',        type=int,   default=42)

    # ── Data quality ──
    parser.add_argument('--min_trial_sec', type=float, default=30.0,
                        help='Skip trials shorter than this (seconds). '
                             'Prevents zero-padded garbage windows. '
                             'Default: 30 = same as seq_window.')
    parser.add_argument('--max_wins_per_trial', type=int, default=20,
                        help='Cap windows per trial to prevent long trials '
                             'from dominating training. Default: 20. '
                             'Val/Test: uncapped for complete evaluation.')
    parser.add_argument('--per_trial_zscore', action='store_true', default=True,
                        help='Apply per-trial z-score before global normalisation '
                             '(default: True). Helps when InvBase is used.')
    parser.add_argument('--no_per_trial_zscore', dest='per_trial_zscore',
                        action='store_false')

    # ── InvBase baseline removal ──
    parser.add_argument('--use_invbase', action='store_true',
                        help='Apply InvBase baseline removal: subtract each subject\'s '
                             'resting-state DE from their emotion trial DE features. '
                             'Requires *_BASELINE_STIMULUS_MUSE_cleaned.json in --data_root.')

    args   = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ── Header ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"LS-STG-Mamba — Preprocessed Emognition")
    print(f"  Loader   : emognition_processed_loader (new _cleaned format)")
    print(f"  Mode     : {args.mode.upper()}")
    print(f"  InvBase  : {'YES — resting-state subtraction' if args.use_invbase else 'NO'}")
    print(f"  DE-LDS   : {args.delds_win_sec}s windows, 5 bands")
    print(f"  Seq win  : {args.seq_window} DE steps ({args.seq_window}s context)")
    print(f"  Stride   : {args.stride} (train); {args.seq_window} (val/test)")
    print(f"  Min trial: {args.min_trial_sec}s | Max wins/trial: {args.max_wins_per_trial}")
    print(f"  Z-score  : per-trial={'YES' if args.per_trial_zscore else 'NO'}")
    print(f"  Device   : {device}")
    print(f"  Model    : d_latent={args.d_latent}, d_graph={args.d_graph}, "
          f"d_mamba={args.d_mamba}, dropout={args.dropout}")
    print(f"  Training : lr={args.lr}, wd={args.weight_decay}, "
          f"epochs={args.epochs}, patience={args.patience}")
    print(f"{'='*65}\n")

    # ── Load raw trials ──────────────────────────────────────────────────────
    print("Loading Emognition Processed trials...")
    t0 = time.time()
    trials, labels, subj_ids, lab2id, id2lab = load_emognition_processed(
        args.data_root, verbose=True
    )
    class_names = [id2lab[i] for i in range(len(id2lab))]
    n_classes   = len(class_names)
    print(f"  Loaded {len(trials)} trials in {time.time() - t0:.1f}s | "
          f"Classes: {class_names}")

    if not trials:
        print("[ERROR] No trials loaded. Check --data_root.")
        sys.exit(1)

    # ── Compute DE-LDS features ──────────────────────────────────────────────
    print(f"\nComputing DE-LDS features ({args.delds_win_sec}s windows, 5 bands)...")
    t0 = time.time()
    features = compute_delds_batch(
        trials, fs=FS,
        window_sec=args.delds_win_sec,
        step_sec=args.delds_win_sec,   # non-overlapping 1s windows
        verbose=True,
    )
    print(f"  Done in {time.time() - t0:.1f}s")

    # ── InvBase baseline removal (optional) ─────────────────────────────────
    if args.use_invbase:
        print("\nApplying InvBase baseline removal...")
        baselines = load_invbase_baselines(
            args.data_root,
            delds_win_sec=args.delds_win_sec,
            verbose=True,
        )
        if baselines:
            features = apply_invbase(features, subj_ids, baselines, verbose=True)
        else:
            print("  [InvBase] No baseline files found — skipping. "
                  "Check dataset for *_BASELINE_STIMULUS_MUSE_cleaned.json files.")

    # ── Filter very short trials ──────────────────────────────────────────
    min_de_wins = int(args.min_trial_sec / args.delds_win_sec)
    before = len(features)
    keep = [i for i, f in enumerate(features) if f.shape[0] >= min_de_wins]
    features   = [features[i]   for i in keep]
    labels     = [labels[i]     for i in keep]
    subj_ids   = [subj_ids[i]   for i in keep]
    N_filtered = before - len(features)
    print(f"\n  Filtered {N_filtered} trials shorter than {args.min_trial_sec}s "
          f"→ {len(features)} trials remain")

    N = len(features)
    if N == 0:
        print("[ERROR] No trials remain after filtering. Lower --min_trial_sec.")
        sys.exit(1)

    # ── Per-trial z-score (optional, recommended with InvBase) ─────────────────
    if args.per_trial_zscore:
        features = per_trial_zscore(features)
        print(f"  Per-trial z-score applied")

    # ── Split ───────────────────────────────────────────────────────────────────
    rng = np.random.RandomState(args.seed)

    if args.mode == 'sub_dep':
        # Trial-level random 70/15/15 — subject-dependent, ~55-65% expected acc
        idx   = rng.permutation(N)
        n_tr  = int(0.70 * N)
        n_va  = int(args.val_ratio * N)
        tr_i  = idx[:n_tr]
        va_i  = idx[n_tr:n_tr + n_va]
        te_i  = idx[n_tr + n_va:]
        print(f"\n  Sub-dependent split (trial-level):")
    else:
        # Subject-level 70/15/15 — no subject leakage, stricter
        subjs = sorted(set(subj_ids))
        rng.shuffle(subjs)
        n_te  = max(1, int(args.test_ratio * len(subjs)))
        n_va  = max(1, int(args.val_ratio  * len(subjs)))
        te_set = set(subjs[:n_te])
        va_set = set(subjs[n_te:n_te + n_va])
        tr_set = set(subjs[n_te + n_va:])
        tr_i  = [i for i in range(N) if subj_ids[i] in tr_set]
        va_i  = [i for i in range(N) if subj_ids[i] in va_set]
        te_i  = [i for i in range(N) if subj_ids[i] in te_set]
        print(f"\n  Subject-split (70/15/15 subjects):")
        print(f"    Train subjs ({len(tr_set)}): {sorted(tr_set)}")
        print(f"    Val   subjs ({len(va_set)}): {sorted(va_set)}")
        print(f"    Test  subjs ({len(te_set)}): {sorted(te_set)}")

    print(f"    Trials → train={len(tr_i)}, val={len(va_i)}, test={len(te_i)}")
    for name, idxs in [("Train", tr_i), ("Val", va_i), ("Test", te_i)]:
        dist = Counter(labels[i] for i in idxs)
        print(f"    {name} labels: { {class_names[k]: v for k, v in sorted(dist.items())} }")

    # ── Normalise on TRAIN set only ──────────────────────────────────────────
    tr_feats = [features[i] for i in tr_i]
    va_feats = [features[i] for i in va_i]
    te_feats = [features[i] for i in te_i]

    feat_mean, feat_std = fit_normalizer(tr_feats)
    tr_feats = apply_normalizer(tr_feats, feat_mean, feat_std)
    va_feats = apply_normalizer(va_feats, feat_mean, feat_std)
    te_feats = apply_normalizer(te_feats, feat_mean, feat_std)

    tr_labels = [labels[i] for i in tr_i]
    va_labels = [labels[i] for i in va_i]
    te_labels = [labels[i] for i in te_i]

    # ── Window into fixed-length sequences ──────────────────────────────────
    # Train: overlapping (stride < seq_window) for more samples
    # Val/Test: non-overlapping (stride == seq_window) for clean evaluation
    tr_w, tr_y = create_windows(tr_feats, tr_labels, args.seq_window, args.stride,
                                 max_wins=args.max_wins_per_trial)
    va_w, va_y = create_windows(va_feats, va_labels, args.seq_window, args.seq_window,
                                 max_wins=None)  # uncapped for complete evaluation
    te_w, te_y = create_windows(te_feats, te_labels, args.seq_window, args.seq_window,
                                 max_wins=None)  # uncapped for complete evaluation

    print(f"\n  Sequence windows ({args.seq_window}s context):")
    print(f"    Train: {len(tr_w)} windows | Val: {len(va_w)} | Test: {len(te_w)}")
    print(f"    Train label dist: {dict(sorted(Counter(tr_y).items()))}")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    tr_dl = DataLoader(
        DeLDSWindowDataset(tr_w, tr_y, augment=True),
        batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=0, pin_memory=True,
    )
    va_dl = DataLoader(
        DeLDSWindowDataset(va_w, va_y, augment=False),
        batch_size=args.batch_size, num_workers=0, pin_memory=True,
    )
    te_dl = DataLoader(
        DeLDSWindowDataset(te_w, te_y, augment=False),
        batch_size=args.batch_size, num_workers=0, pin_memory=True,
    )

    # ── Model ────────────────────────────────────────────────────────────────
    model = LSSTGMamba(
        n_channels=4, n_bands=5,
        d_latent=args.d_latent, d_graph=args.d_graph,
        d_mamba=args.d_mamba,   d_state=args.d_state,
        n_global_layers=args.n_global_layers,
        num_classes=n_classes,  dropout=args.dropout,
    ).to(device)

    model.init_adjacency(build_muse_adjacency())

    print(f"\n  Model params  : {count_parameters(model):,}")
    print(f"  Batches/epoch : {len(tr_dl)}")

    # ── Loss & Optimiser ─────────────────────────────────────────────────────
    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_f1, best_state, patience_ctr = 0.0, None, 0
    epoch_times = []

    print(f"\n{'='*65}")
    print(f"Training ({args.epochs} epochs, patience={args.patience})")
    print(f"{'='*65}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0   = time.time()
        beta = args.beta_max * min(1.0, epoch / max(args.warmup_kl, 1))

        ep_cls = ep_rec = ep_kl = 0.0
        tr_correct = tr_total = 0

        for bx, by in tr_dl:
            bx = bx.to(device)
            by = by.long().to(device)

            if args.mixup_alpha > 0 and random.random() < 0.5:
                bx, y_a, y_b, lam = mixup_data(bx, by, args.mixup_alpha)
                logits, l_rec, l_kl = model(bx, return_losses=True)
                l_cls = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                tr_correct += (logits.argmax(1) == y_a).sum().item()
            else:
                logits, l_rec, l_kl = model(bx, return_losses=True)
                l_cls = criterion(logits, by)
                tr_correct += (logits.argmax(1) == by).sum().item()

            tr_total += len(by)
            loss = l_cls + args.alpha * l_rec + beta * l_kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_cls += l_cls.item()
            ep_rec += l_rec.item()
            ep_kl  += l_kl.item()

        scheduler.step()
        n_b     = max(len(tr_dl), 1)
        tr_acc  = tr_correct / max(tr_total, 1)
        ep_time = time.time() - t0
        epoch_times.append(ep_time)

        _, vl_cls, _, _, v_acc, v_f1, _, _ = evaluate(
            model, va_dl, device, criterion, args.alpha, beta
        )

        star = " ★" if v_f1 > best_f1 else ""
        print(f"  Ep {epoch:3d}/{args.epochs} | "
              f"Tr Cls:{ep_cls/n_b:.3f} Rec:{ep_rec/n_b:.3f} "
              f"KL:{ep_kl/n_b:.4f} Acc:{tr_acc:.3f} | "
              f"Va Cls:{vl_cls:.3f} Acc:{v_acc:.3f} F1:{v_f1:.3f} | "
              f"β={beta:.3f} {ep_time:.1f}s{star}")

        if v_f1 > best_f1:
            best_f1    = v_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # ── Test ─────────────────────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)

    _, _, _, _, te_acc, te_f1, te_preds, te_lbls = evaluate(
        model, te_dl, device, criterion, args.alpha, beta
    )

    print(f"\n{'='*65}")
    print(f"RESULTS — LS-STG-Mamba Processed Emognition ({args.mode})")
    print(f"{'='*65}")
    print(f"  Mode        : {args.mode.upper()}")
    print(f"  Seq context : {args.seq_window}s ({args.seq_window} DE-LDS steps)")
    print(f"  Best Val F1 : {best_f1:.4f}")
    print(f"  Test Acc    : {te_acc:.4f}  ({te_acc*100:.1f}%)")
    print(f"  Test F1     : {te_f1:.4f}")
    print(f"  Avg ep time : {np.mean(epoch_times):.1f}s")
    print(f"  Total time  : {sum(epoch_times) / 60:.1f} min")
    print_report(te_lbls, te_preds, class_names, title=args.mode)

    # Learned adjacency
    adj_np = model.get_adjacency()
    print(f"\n  Learned Adjacency (Muse 2 channels):")
    print(f"  {'':>8}" + "".join(f"{c:>8}" for c in CHANNELS))
    for i, c in enumerate(CHANNELS):
        print(f"  {c:>8}" + "".join(f"{adj_np[i, j]:>8.3f}" for j in range(4)))

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_dir = os.path.join(_HERE, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, f'ls_stg_mamba_processed_{args.mode}.pt')
    torch.save({
        'model': model.state_dict(),
        'model_cfg': {
            'n_channels':      4,
            'n_bands':         5,
            'd_latent':        args.d_latent,
            'd_graph':         args.d_graph,
            'd_mamba':         args.d_mamba,
            'd_state':         args.d_state,
            'n_global_layers': args.n_global_layers,
            'num_classes':     n_classes,
            'dropout':         args.dropout,
        },
        'channels':    CHANNELS,
        'class_names': class_names,
        'val_f1':      best_f1,
        'test_acc':    te_acc,
        'test_f1':     te_f1,
        'feat_mean':   feat_mean.tolist(),
        'feat_std':    feat_std.tolist(),
        'adjacency':   adj_np.tolist(),
        'mode':        args.mode,
        'seq_window':  args.seq_window,
        'use_invbase': args.use_invbase,
    }, ckpt)
    print(f"\n  Checkpoint saved: {ckpt}\n")


if __name__ == '__main__':
    main()
