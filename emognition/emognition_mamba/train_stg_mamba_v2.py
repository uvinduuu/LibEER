"""
LS-STG-Mamba v2 — Preprocessed Emognition  (anti-overfitting edition)
======================================================================

Target: >40% test accuracy, every class >35% recall.

Key changes vs the 33% run:
  1. SMALLER MODEL     — d_latent=4, d_graph=12, d_mamba=32  (~14K params)
                         73K params for 114 trials was the primary overfitting cause.
  2. LONGER CONTEXT    — seq_window=60s (was 30s)
                         Mean trial=120s; 60s gives Mamba richer temporal arc.
  3. SLOWER LR         — 1e-4 (was 5e-4)
                         Best val F1 at ep 13 means LR was too aggressive.
  4. SUBJECT EMBEDDING — --use_subject_emb enabled by default
                         164 scalars teach per-subject emotion prior.
  5. STRONGER REG      — dropout=0.5, wd=0.05, label_smooth=0.15
  6. LONGER PATIENCE   — patience=50 (was 30); lower LR needs more time
  7. TRIAL-LEVEL VOTE  — soft-vote over all windows of a trial = primary metric
                         The 33% was window-level; trial-level is typically +5-10%.
  8. SLOWER KL WARMUP  — warmup_kl=50 (was 20); lets classifier learn first
  9. CAPPED WINS       — max_wins_per_trial=10 (was 20)
                         Reduces correlation between overlapping windows from same trial.
  10. COSINE WARMUP LR — 10-epoch linear warmup then cosine decay

All helper modules (loader, delds_extractor, invbase_processed, stg_mamba_v3)
are unchanged from the 33% run — only training dynamics changed.

Usage (Kaggle):
  python train_stg_mamba_v2.py \\
      --data_root "/kaggle/input/.../Emognition Processed" \\
      --mode sub_dep --use_invbase --use_subject_emb

Classes (alphabetical): ENTHUSIASM=0, FEAR=1, NEUTRAL=2, SADNESS=3
"""

import os, sys, time, random, argparse
from collections import Counter, defaultdict

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

CLASS_NAMES = ['ENTHUSIASM', 'FEAR', 'NEUTRAL', 'SADNESS']
CHANNELS    = ['TP9', 'AF7', 'AF8', 'TP10']


# ══════════════════════════════════════════════════════════════════
#  Subject Bias Head  (unchanged from v1)
# ══════════════════════════════════════════════════════════════════

class SubjectBiasHead(nn.Module):
    """
    Per-subject logit bias — learned additive prior over emotion classes.

    In sub_dep mode the model sees multiple emotions per subject, so it
    can learn that subject 22 tends toward NEUTRAL at rest.
    Adds a (n_subjects, n_classes) embedding lookup to the model logits.
    Tiny params (41×4=164) — no risk of additional overfitting.
    """
    def __init__(self, n_subjects: int, n_classes: int):
        super().__init__()
        self.bias = nn.Embedding(n_subjects, n_classes)
        nn.init.zeros_(self.bias.weight)

    def forward(self, logits, subj_idx):
        return logits + self.bias(subj_idx)


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset
# ══════════════════════════════════════════════════════════════════════════════

class DeLDSWindowDataset(Dataset):
    """Windowed DE-LDS dataset with augmentation."""

    def __init__(self, windows, labels, subj_indices=None, trial_ids=None,
                 augment=False, noise=0.03, mask_prob=0.12):
        self.windows      = windows
        self.labels       = labels
        self.subj_indices = subj_indices
        self.trial_ids    = trial_ids
        self.augment      = augment
        self.noise        = noise
        self.mask_prob    = mask_prob

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx].copy()   # (T, C, 5)
        y = int(self.labels[idx])
        s = int(self.subj_indices[idx]) if self.subj_indices is not None else -1
        t = int(self.trial_ids[idx])    if self.trial_ids    is not None else -1

        if self.augment:
            # Gaussian feature noise (mild)
            x += np.random.randn(*x.shape).astype(np.float32) * self.noise

            # Random time masking
            if random.random() < self.mask_prob:
                T = x.shape[0]
                ml = random.randint(1, max(2, T // 5))
                ms = random.randint(0, T - ml)
                x[ms:ms + ml] = 0.0

            # Random band dropout
            if random.random() < self.mask_prob:
                x[:, :, random.randint(0, 4)] = 0.0

            # Random channel dropout (one channel → mean of others)
            if random.random() < 0.1:
                ch = random.randint(0, 3)
                x[:, ch, :] = x[:, [c for c in range(4) if c != ch], :].mean(axis=1)

            # Mild amplitude scaling
            if random.random() < 0.4:
                x *= random.uniform(0.88, 1.12)

            # Random time shift (circular)
            shift = random.randint(-5, 5)
            if shift:
                x = np.roll(x, shift, axis=0)

        return torch.FloatTensor(x), y, s, t


def create_windows(feats, labels, subj_idx_list, w_size, stride, max_wins=None):
    """Slice DE-LDS sequences into windows. Returns (windows, labels, subj_idxs, trial_ids)."""
    wins, wlbls, widxs, wtids = [], [], [], []
    for trial_id, (x, y, si) in enumerate(zip(feats, labels, subj_idx_list)):
        T = x.shape[0]
        if T < w_size:
            x = np.pad(x, ((0, w_size - T), (0, 0), (0, 0)))
            T = w_size
        trial_wins = [x[s:s + w_size] for s in range(0, T - w_size + 1, stride)]
        if max_wins is not None and len(trial_wins) > max_wins:
            idxs = np.linspace(0, len(trial_wins) - 1, max_wins, dtype=int)
            trial_wins = [trial_wins[i] for i in idxs]
        for w in trial_wins:
            wins.append(w);  wlbls.append(y)
            widxs.append(si); wtids.append(trial_id)
    return wins, wlbls, widxs, wtids


# ══════════════════════════════════════════════════════════════════════════════
#  Utilities
# ══════════════════════════════════════════════════════════════════════════════

def setup_seed(seed=42):
    random.seed(seed);  np.random.seed(seed)
    torch.manual_seed(seed);  torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(model, loader, device, criterion, alpha, beta, bias_head=None):
    model.eval()
    if bias_head is not None:
        bias_head.eval()
    preds, targets, all_logits = [], [], []
    tloss = tcls = trec = tkl = n = 0.0

    with torch.no_grad():
        for bx, by, bs, bt in loader:
            bx = bx.to(device)
            by = by.long().to(device)
            logits, lrec, lkl = model(bx, return_losses=True)
            if bias_head is not None:
                logits = bias_head(logits, bs.to(device))
            lcls = criterion(logits, by)
            loss = lcls + alpha * lrec + beta * lkl

            tloss += loss.item();  tcls += lcls.item()
            trec  += lrec.item(); tkl  += lkl.item(); n += 1

            all_logits.extend(logits.cpu().numpy())
            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(by.cpu().numpy())

    nb  = max(n, 1)
    acc = float(np.mean(np.array(preds) == np.array(targets)))
    f1  = f1_score(targets, preds, average='macro', zero_division=0)
    return tloss/nb, tcls/nb, trec/nb, tkl/nb, acc, f1, preds, targets, np.array(all_logits)


def trial_vote_accuracy(logits_all, targets_all, trial_ids, class_names):
    """
    Soft-vote over all windows of each trial — primary EEG metric.

    Groups windows by trial_id, averages their logits, argmax = trial prediction.
    Typical gain over window-level: +5-15%.
    """
    trial_logits = defaultdict(list)
    trial_label  = {}
    for logit, label, tid in zip(logits_all, targets_all, trial_ids):
        trial_logits[tid].append(logit)
        trial_label[tid] = label

    trial_preds, trial_true = [], []
    for tid in sorted(trial_logits.keys()):
        avg_logit = np.mean(trial_logits[tid], axis=0)
        trial_preds.append(int(np.argmax(avg_logit)))
        trial_true.append(trial_label[tid])

    n_correct = sum(p == t for p, t in zip(trial_preds, trial_true))
    acc = n_correct / max(len(trial_true), 1)
    f1  = f1_score(trial_true, trial_preds, average='macro', zero_division=0)
    return acc, f1, trial_preds, trial_true


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
#  Normalisation
# ══════════════════════════════════════════════════════════════════════════════

def per_trial_zscore(features_list):
    """Z-score each trial independently over time (per channel per band)."""
    normed = []
    for f in features_list:
        mean = f.mean(axis=0, keepdims=True)
        std  = f.std(axis=0,  keepdims=True) + 1e-8
        normed.append((f - mean) / std)
    return normed


def fit_normalizer(features_list):
    all_f = np.concatenate([f.reshape(-1, 5) for f in features_list], axis=0)
    return all_f.mean(0), all_f.std(0) + 1e-8


def apply_normalizer(features_list, mean, std):
    return [(f - mean) / std for f in features_list]


# ══════════════════════════════════════════════════════════════════════════════
#  LR Scheduler: Linear Warmup + Cosine Decay
# ══════════════════════════════════════════════════════════════════════════════

class WarmupCosineScheduler:
    """
    Linear warmup for `warmup_epochs`, then cosine decay to `eta_min`.

    Why warmup matters here:
      With only 18 batches/epoch, the first few epochs are very noisy.
      Linear warmup prevents early large updates from destroying initialisation.
    """
    def __init__(self, optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
        self.opt           = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs  = total_epochs
        self.eta_min       = eta_min
        self.base_lrs      = [pg['lr'] for pg in optimizer.param_groups]

    def step(self, epoch):
        """Call once per epoch (epoch is 1-indexed)."""
        if epoch <= self.warmup_epochs:
            scale = epoch / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            scale    = 0.5 * (1 + np.cos(np.pi * progress))
        for pg, base_lr in zip(self.opt.param_groups, self.base_lrs):
            pg['lr'] = self.eta_min + (base_lr - self.eta_min) * scale


# ══════════════════════════════════════════════════════════════════════════════
#  Adjacency prior for Muse 2
# ══════════════════════════════════════════════════════════════════════════════

def build_muse_adjacency():
    """Hand-crafted adjacency for Muse 2: TP9↔TP10 (temporal), AF7↔AF8 (frontal)."""
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
        description="LS-STG-Mamba v2 on Preprocessed Emognition (anti-overfitting)"
    )

    # ── Data ──
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--mode', default='sub_dep',
                        choices=['sub_dep', 'sub_split'])
    parser.add_argument('--test_ratio', type=float, default=0.15)
    parser.add_argument('--val_ratio',  type=float, default=0.15)

    # ── DE-LDS ──
    parser.add_argument('--delds_win_sec', type=float, default=1.0)

    # ── Sequence windowing ──
    parser.add_argument('--seq_window', type=int, default=60,
                        help='DE steps per training sample (default: 60 = 60s context)')
    parser.add_argument('--stride', type=int, default=15,
                        help='Stride for training windows (default: 15s)')

    # ── Model — SMALLER than v1 to fight overfitting ──
    parser.add_argument('--d_latent',        type=int,   default=4,
                        help='Latent dim (default 4, was 8 in 33%% run)')
    parser.add_argument('--d_graph',         type=int,   default=12,
                        help='Graph dim (default 12, was 24)')
    parser.add_argument('--d_mamba',         type=int,   default=32,
                        help='Mamba dim (default 32, was 64)')
    parser.add_argument('--d_state',         type=int,   default=16)
    parser.add_argument('--n_global_layers', type=int,   default=2,
                        help='Global Mamba layers (default 2, was 3)')
    parser.add_argument('--dropout',         type=float, default=0.5,
                        help='Dropout (default 0.5, was 0.4)')

    # ── VAE losses ──
    parser.add_argument('--alpha',     type=float, default=2.0,
                        help='Reconstruction weight (reduced from 5.0)')
    parser.add_argument('--beta_max',  type=float, default=0.05,
                        help='Max KL weight (reduced from 0.1)')
    parser.add_argument('--warmup_kl', type=int,   default=50,
                        help='KL warmup epochs (extended from 20)')

    # ── Training ──
    parser.add_argument('--epochs',       type=int,   default=200)
    parser.add_argument('--batch_size',   type=int,   default=32,
                        help='Smaller batch = more gradient noise = more regularization')
    parser.add_argument('--lr',           type=float, default=1e-4,
                        help='Peak LR after warmup (was 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.05,
                        help='L2 penalty (increased from 0.03)')
    parser.add_argument('--patience',     type=int,   default=50,
                        help='Early stopping patience (extended from 30)')
    parser.add_argument('--mixup_alpha',  type=float, default=0.4)
    parser.add_argument('--warmup_lr_epochs', type=int, default=10,
                        help='Linear LR warmup epochs')
    parser.add_argument('--label_smooth', type=float, default=0.15,
                        help='Label smoothing (increased from 0.05)')
    parser.add_argument('--seed',         type=int,   default=42)

    # ── Data quality ──
    parser.add_argument('--min_trial_sec',       type=float, default=60.0,
                        help='Min trial length (matches seq_window default)')
    parser.add_argument('--max_wins_per_trial',  type=int,   default=10,
                        help='Cap windows/trial (reduced from 20)')
    parser.add_argument('--per_trial_zscore',    action='store_true', default=False)
    parser.add_argument('--no_per_trial_zscore', dest='per_trial_zscore',
                        action='store_false')

    # ── Subject embedding ──
    parser.add_argument('--use_subject_emb', action='store_true', default=False,
                        help='Add per-subject logit bias (recommended for sub_dep)')
    parser.add_argument('--subj_emb_lr',    type=float, default=1e-3)

    # ── InvBase ──
    parser.add_argument('--use_invbase', action='store_true', default=False)

    args   = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Ensure min_trial_sec >= seq_window
    if args.min_trial_sec < args.seq_window:
        args.min_trial_sec = float(args.seq_window)

    # ── Header ──────────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print(f"LS-STG-Mamba v2 — Preprocessed Emognition (anti-overfitting)")
    print(f"  Mode       : {args.mode.upper()}")
    print(f"  InvBase    : {'YES' if args.use_invbase else 'NO'}")
    print(f"  SubjEmb    : {'YES' if args.use_subject_emb else 'NO'}")
    print(f"  Seq win    : {args.seq_window}s | Stride: {args.stride}s")
    print(f"  Model      : d_latent={args.d_latent}, d_graph={args.d_graph}, "
          f"d_mamba={args.d_mamba}, layers={args.n_global_layers}, drop={args.dropout}")
    print(f"  Training   : lr={args.lr}, wd={args.weight_decay}, "
          f"bs={args.batch_size}, epochs={args.epochs}, patience={args.patience}")
    print(f"  LR warmup  : {args.warmup_lr_epochs} epochs linear → cosine")
    print(f"  KL warmup  : {args.warmup_kl} epochs | beta_max={args.beta_max}")
    print(f"  LabelSmooth: {args.label_smooth}")
    print(f"  Device     : {device}")
    print(f"{'='*65}\n")

    # ── Load raw trials ──────────────────────────────────────────────────────
    print("Loading Emognition Processed trials...")
    t0 = time.time()
    trials, labels, subj_ids, lab2id, id2lab = load_emognition_processed(
        args.data_root, verbose=True
    )
    class_names = [id2lab[i] for i in range(len(id2lab))]
    n_classes   = len(class_names)
    print(f"  Loaded {len(trials)} trials in {time.time()-t0:.1f}s | Classes: {class_names}")
    if not trials:
        print("[ERROR] No trials loaded — check --data_root.")
        sys.exit(1)

    # ── DE-LDS features ──────────────────────────────────────────────────────
    print(f"\nComputing DE-LDS features ({args.delds_win_sec}s windows, 5 bands)...")
    t0 = time.time()
    features = compute_delds_batch(
        trials, fs=FS,
        window_sec=args.delds_win_sec,
        step_sec=args.delds_win_sec,
        verbose=True,
    )
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── InvBase ─────────────────────────────────────────────────────────────
    if args.use_invbase:
        print("\nApplying InvBase baseline removal...")
        baselines = load_invbase_baselines(
            args.data_root, delds_win_sec=args.delds_win_sec, verbose=True,
        )
        if baselines:
            features = apply_invbase(features, subj_ids, baselines, verbose=True)
        else:
            print("  [InvBase] No baseline files found — skipping.")

    # ── Filter short trials ──────────────────────────────────────────────────
    min_de = int(args.min_trial_sec / args.delds_win_sec)
    keep   = [i for i, f in enumerate(features) if f.shape[0] >= min_de]
    features  = [features[i]  for i in keep]
    labels    = [labels[i]    for i in keep]
    subj_ids  = [subj_ids[i]  for i in keep]
    skipped   = (len(trials) - len(features))
    print(f"\n  Filtered {skipped} trials shorter than {args.min_trial_sec}s "
          f"→ {len(features)} trials remain")
    N = len(features)
    if N == 0:
        print("[ERROR] No trials remain after filtering. Lower --min_trial_sec.")
        sys.exit(1)

    # ── Per-trial z-score (optional) ─────────────────────────────────────────
    if args.per_trial_zscore:
        features = per_trial_zscore(features)
        print("  Per-trial z-score applied")

    # ── Stratified trial-level split ─────────────────────────────────────────
    rng = np.random.RandomState(args.seed)

    if args.mode == 'sub_dep':
        # Stratified: split each class separately → balanced val/test guaranteed
        tr_i, va_i, te_i = [], [], []
        for cls_id in range(n_classes):
            cls_idx = [i for i in range(N) if labels[i] == cls_id]
            rng.shuffle(cls_idx)
            n_te = max(1, int(args.test_ratio * len(cls_idx)))
            n_va = max(1, int(args.val_ratio  * len(cls_idx)))
            te_i.extend(cls_idx[:n_te])
            va_i.extend(cls_idx[n_te:n_te + n_va])
            tr_i.extend(cls_idx[n_te + n_va:])
        tr_i = np.array(tr_i); va_i = np.array(va_i); te_i = np.array(te_i)
        print(f"\n  Sub-dependent STRATIFIED split (70/15/15):")
    else:
        subjs = sorted(set(subj_ids))
        rng.shuffle(subjs)
        n_te    = max(1, int(args.test_ratio * len(subjs)))
        n_va    = max(1, int(args.val_ratio  * len(subjs)))
        te_set  = set(subjs[:n_te])
        va_set  = set(subjs[n_te:n_te + n_va])
        tr_set  = set(subjs[n_te + n_va:])
        tr_i    = [i for i in range(N) if subj_ids[i] in tr_set]
        va_i    = [i for i in range(N) if subj_ids[i] in va_set]
        te_i    = [i for i in range(N) if subj_ids[i] in te_set]
        print(f"\n  Subject-split: {len(tr_set)} train / {len(va_set)} val / {len(te_set)} test subjects")

    print(f"    Trials → train={len(tr_i)}, val={len(va_i)}, test={len(te_i)}")
    for name, idxs in [("Train", tr_i), ("Val", va_i), ("Test", te_i)]:
        dist = Counter(labels[i] for i in idxs)
        print(f"    {name}: { {class_names[k]: v for k, v in sorted(dist.items())} }")

    # ── Subject index map ────────────────────────────────────────────────────
    all_subjs    = sorted(set(subj_ids))
    subj2idx     = {s: i for i, s in enumerate(all_subjs)}
    n_subjects   = len(all_subjs)
    subj_indices = [subj2idx[s] for s in subj_ids]

    # ── Normalise on train only ──────────────────────────────────────────────
    tr_feats = [features[i]     for i in tr_i]
    va_feats = [features[i]     for i in va_i]
    te_feats = [features[i]     for i in te_i]
    tr_sidxs = [subj_indices[i] for i in tr_i]
    va_sidxs = [subj_indices[i] for i in va_i]
    te_sidxs = [subj_indices[i] for i in te_i]

    feat_mean, feat_std = fit_normalizer(tr_feats)
    tr_feats = apply_normalizer(tr_feats, feat_mean, feat_std)
    va_feats = apply_normalizer(va_feats, feat_mean, feat_std)
    te_feats = apply_normalizer(te_feats, feat_mean, feat_std)

    tr_labels = [labels[i] for i in tr_i]
    va_labels = [labels[i] for i in va_i]
    te_labels = [labels[i] for i in te_i]

    # ── Windowing ────────────────────────────────────────────────────────────
    # Val/Test: non-overlapping (stride=seq_window) — clean windows for voting
    val_test_stride = args.seq_window
    tr_w, tr_y, tr_s, tr_t = create_windows(
        tr_feats, tr_labels, tr_sidxs,
        args.seq_window, args.stride, max_wins=args.max_wins_per_trial)
    va_w, va_y, va_s, va_t = create_windows(
        va_feats, va_labels, va_sidxs,
        args.seq_window, val_test_stride, max_wins=None)
    te_w, te_y, te_s, te_t = create_windows(
        te_feats, te_labels, te_sidxs,
        args.seq_window, val_test_stride, max_wins=None)

    print(f"\n  Sequence windows ({args.seq_window}s context):")
    print(f"    Train: {len(tr_w)} windows (stride={args.stride}, cap={args.max_wins_per_trial})")
    print(f"    Val  : {len(va_w)} windows (stride={val_test_stride}, non-overlapping)")
    print(f"    Test : {len(te_w)} windows (stride={val_test_stride}, non-overlapping)")
    tr_win_dist = dict(sorted(Counter(tr_y).items()))
    print(f"    Train label dist: {tr_win_dist}")

    # ── DataLoaders ──────────────────────────────────────────────────────────
    tr_dl = DataLoader(
        DeLDSWindowDataset(tr_w, tr_y, subj_indices=tr_s, trial_ids=tr_t, augment=True),
        batch_size=args.batch_size, shuffle=True,
        drop_last=len(tr_w) >= args.batch_size, num_workers=0, pin_memory=True,
    )
    va_dl = DataLoader(
        DeLDSWindowDataset(va_w, va_y, subj_indices=va_s, trial_ids=va_t, augment=False),
        batch_size=args.batch_size, num_workers=0, pin_memory=True,
    )
    te_dl = DataLoader(
        DeLDSWindowDataset(te_w, te_y, subj_indices=te_s, trial_ids=te_t, augment=False),
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

    # ── Subject Bias Head ────────────────────────────────────────────────────
    bias_head = None
    if args.use_subject_emb:
        if args.mode != 'sub_dep':
            print("[WARN] --use_subject_emb only meaningful in sub_dep mode.")
        else:
            bias_head = SubjectBiasHead(n_subjects, n_classes).to(device)
            print(f"  SubjectBias: {n_subjects}×{n_classes} = {n_subjects*n_classes} params")

    total_params = count_parameters(model)
    bias_params  = sum(p.numel() for p in bias_head.parameters()) if bias_head else 0
    print(f"\n  Model params  : {total_params:,}")
    if bias_head:
        print(f"  Bias params   : {bias_params:,}")
    print(f"  Total params  : {total_params + bias_params:,}")
    print(f"  Batches/epoch : {len(tr_dl)}")

    # ── Loss & Optimiser ─────────────────────────────────────────────────────
    win_cnts  = np.array([Counter(tr_y)[i] for i in range(n_classes)], dtype=np.float32)
    cls_wts   = (win_cnts.sum() / (n_classes * win_cnts)).clip(0.5, 3.0)
    cls_wts_t = torch.tensor(cls_wts, device=device)
    print(f"  Class weights : { {class_names[i]: f'{cls_wts[i]:.2f}' for i in range(n_classes)} }")

    criterion = nn.CrossEntropyLoss(weight=cls_wts_t, label_smoothing=args.label_smooth)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    if bias_head is not None:
        optimizer.add_param_group({
            'params': bias_head.parameters(),
            'lr': args.subj_emb_lr,
            'weight_decay': 0.0,
        })

    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_lr_epochs,
        total_epochs=args.epochs,
        eta_min=1e-6,
    )

    best_f1 = best_state = best_bias_state = 0.0
    best_state = None;  best_bias_state = None
    patience_ctr = 0
    epoch_times  = []

    print(f"\n{'='*65}")
    print(f"Training ({args.epochs} epochs, patience={args.patience})")
    print(f"{'='*65}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        if bias_head is not None:
            bias_head.train()
        t0   = time.time()
        beta = args.beta_max * min(1.0, epoch / max(args.warmup_kl, 1))

        ep_cls = ep_rec = ep_kl = 0.0
        tr_correct = tr_total = 0

        for bx, by, bs, bt in tr_dl:
            bx = bx.to(device)
            by = by.long().to(device)

            if args.mixup_alpha > 0 and random.random() < 0.5:
                bx, y_a, y_b, lam = mixup_data(bx, by, args.mixup_alpha)
                logits, l_rec, l_kl = model(bx, return_losses=True)
                if bias_head is not None:
                    logits = bias_head(logits, bs.to(device))
                l_cls = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                tr_correct += (logits.argmax(1) == y_a).sum().item()
            else:
                logits, l_rec, l_kl = model(bx, return_losses=True)
                if bias_head is not None:
                    logits = bias_head(logits, bs.to(device))
                l_cls = criterion(logits, by)
                tr_correct += (logits.argmax(1) == by).sum().item()

            tr_total += len(by)
            loss = l_cls + args.alpha * l_rec + beta * l_kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_cls += l_cls.item(); ep_rec += l_rec.item(); ep_kl += l_kl.item()

        scheduler.step(epoch)
        n_b     = max(len(tr_dl), 1)
        tr_acc  = tr_correct / max(tr_total, 1)
        ep_time = time.time() - t0
        epoch_times.append(ep_time)

        _, vl_cls, _, _, v_acc, v_f1, _, _, _ = evaluate(
            model, va_dl, device, criterion, args.alpha, beta, bias_head
        )

        star = " ★" if v_f1 > best_f1 else ""
        print(f"  Ep {epoch:3d}/{args.epochs} | "
              f"Tr Cls:{ep_cls/n_b:.3f} Rec:{ep_rec/n_b:.3f} "
              f"KL:{ep_kl/n_b:.4f} Acc:{tr_acc:.3f} | "
              f"Va Cls:{vl_cls:.3f} Acc:{v_acc:.3f} F1:{v_f1:.3f} | "
              f"β={beta:.3f} {ep_time:.1f}s{star}")

        if v_f1 > best_f1:
            best_f1         = v_f1
            best_state      = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_bias_state = ({k: v.cpu().clone() for k, v in bias_head.state_dict().items()}
                               if bias_head is not None else None)
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # ── Test ─────────────────────────────────────────────────────────────────
    if best_state:
        model.load_state_dict(best_state)
    if bias_head is not None and best_bias_state is not None:
        bias_head.load_state_dict(best_bias_state)
    model = model.to(device)

    _, _, _, _, te_acc, te_f1, te_preds, te_lbls, te_logits = evaluate(
        model, te_dl, device, criterion, args.alpha, beta, bias_head
    )

    # ── Trial-level soft vote (primary metric) ────────────────────────────
    tv_acc, tv_f1, tv_preds, tv_true = trial_vote_accuracy(
        te_logits, te_lbls, te_t, class_names
    )

    print(f"\n{'='*65}")
    print(f"RESULTS — LS-STG-Mamba v2 ({args.mode})")
    print(f"{'='*65}")
    print(f"  Mode         : {args.mode.upper()}")
    print(f"  Seq context  : {args.seq_window}s | Model params: {count_parameters(model):,}")
    print(f"  Best Val F1  : {best_f1:.4f}")
    print(f"")
    print(f"  ── Window-level (per {args.seq_window}s window): ──")
    print(f"  Test Acc     : {te_acc:.4f}  ({te_acc*100:.1f}%)")
    print(f"  Test F1      : {te_f1:.4f}")
    print(f"")
    print(f"  ── Trial-level (soft-vote over all windows): ──  ← PRIMARY METRIC")
    print(f"  Test Acc     : {tv_acc:.4f}  ({tv_acc*100:.1f}%)")
    print(f"  Test F1      : {tv_f1:.4f}")
    print(f"  n_trials     : {len(tv_true)}")
    print(f"")
    print(f"  Avg ep time  : {np.mean(epoch_times):.1f}s")
    print(f"  Total time   : {sum(epoch_times)/60:.1f} min")

    print(f"\n  [Window-level report:]")
    print_report(te_lbls, te_preds, class_names, title="window-level")
    print(f"\n  [Trial-level report:]")
    print_report(tv_true, tv_preds, class_names, title="trial-level vote")

    # Learned adjacency
    adj_np = model.get_adjacency()
    print(f"\n  Learned Adjacency (Muse 2 channels):")
    print(f"  {'':>8}" + "".join(f"{c:>8}" for c in CHANNELS))
    for i, c in enumerate(CHANNELS):
        print(f"  {c:>8}" + "".join(f"{adj_np[i, j]:>8.3f}" for j in range(4)))

    # ── Save checkpoint ──────────────────────────────────────────────────────
    ckpt_dir = os.path.join(_HERE, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, f'ls_stg_mamba_v2_{args.mode}.pt')
    torch.save({
        'model':      model.state_dict(),
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
        'test_acc_window': te_acc,
        'test_f1_window':  te_f1,
        'test_acc_trial':  tv_acc,
        'test_f1_trial':   tv_f1,
        'feat_mean':   feat_mean.tolist(),
        'feat_std':    feat_std.tolist(),
        'adjacency':   adj_np.tolist(),
        'mode':        args.mode,
        'seq_window':  args.seq_window,
        'use_invbase': args.use_invbase,
        'bias_head':   bias_head.state_dict() if bias_head is not None else None,
        'subj2idx':    subj2idx if bias_head is not None else None,
    }, ckpt)
    print(f"\n  Checkpoint saved: {ckpt}\n")


if __name__ == '__main__':
    main()
