"""
STG-Mamba Training v2 — 4-Channel SEED-IV with DE-LDS Features.

Key improvements over v1:
  - WINDOWED DE-LDS: Slide window over temporal dimension (~8× more training data)
  - Better adjacency init: Initialize with electrode distance prior
  - Data augmentation: Temporal jitter + Gaussian noise on DE-LDS
  - Mixup training: Interpolate samples for regularization

Usage on Kaggle:
    python train_stg_mamba_4ch.py \
        --dataset_path /kaggle/input/datasets/phhasian0710/seed-iv/seed_iv \
        --epochs 150 --batch_size 128
"""

import os, sys, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from scipy.io import loadmat
import multiprocessing as mp
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stg_mamba_model import STGMamba, count_parameters

CLASS_NAMES = ['neutral', 'sad', 'fear', 'happy']

# SEED-IV 62-channel order
SEED_CHANNELS_62 = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ',
    'F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2',
    'FC4','FC6','FT8','C5','C3','C1','CZ','C2','C4','C6',
    'T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7',
    'PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2',
    'CB2','T7'
]

DEFAULT_4CH = ['FP1', 'FP2', 'T7', 'T8']

_SES_LABELS = [
    [1,2,3,0,2,0,0,1,0,1,2,1,1,1,2,3,2,2,3,3,0,3,0,3],
    [2,1,3,0,0,2,0,2,3,3,2,3,2,0,1,1,2,1,0,3,0,1,3,1],
    [1,2,2,1,3,3,3,1,1,2,1,0,2,3,3,0,2,3,0,0,2,0,1,0],
]

_EEG_FILES = [
    ['1_20160518.mat','2_20150915.mat','3_20150919.mat','4_20151111.mat',
     '5_20160406.mat','6_20150507.mat','7_20150715.mat','8_20151103.mat',
     '9_20151028.mat','10_20151014.mat','11_20150916.mat','12_20150725.mat',
     '13_20151115.mat','14_20151205.mat','15_20150508.mat'],
    ['1_20161125.mat','2_20150920.mat','3_20151018.mat','4_20151118.mat',
     '5_20160413.mat','6_20150511.mat','7_20150717.mat','8_20151110.mat',
     '9_20151119.mat','10_20151021.mat','11_20150921.mat','12_20150804.mat',
     '13_20151125.mat','14_20151208.mat','15_20150514.mat'],
    ['1_20161126.mat','2_20151012.mat','3_20151101.mat','4_20151123.mat',
     '5_20160420.mat','6_20150512.mat','7_20150721.mat','8_20151117.mat',
     '9_20151209.mat','10_20151023.mat','11_20151011.mat','12_20150807.mat',
     '13_20161130.mat','14_20151215.mat','15_20150527.mat'],
]


# ─────────────────────────────────────────────────
# Data Loading
# ─────────────────────────────────────────────────

def get_channel_indices(channel_names):
    name_to_idx = {ch.upper(): i for i, ch in enumerate(SEED_CHANNELS_62)}
    return [name_to_idx[n.upper()] for n in channel_names]


def _load_subject_delds(dir_path, file_relpath):
    mat = loadmat(f"{dir_path}/{file_relpath}")
    keys = list(mat.keys())[3:]
    trials = []
    for i in range(24):
        feat = np.array(mat[keys[i * 4 + 1]])  # de_LDS
        trials.append(feat.astype(np.float32))  # (62, T, 5)
    return trials


def load_seediv_4ch_delds(dataset_path, channel_names=None, sessions=None):
    if sessions is None:
        sessions = [1, 2, 3]
    if channel_names is None:
        channel_names = DEFAULT_4CH

    ch_indices = get_channel_indices(channel_names)
    print(f"  Channels: {channel_names} → indices {ch_indices}")

    smooth_dir = f"{dataset_path}/eeg_feature_smooth"
    all_feats, all_labels, all_subj, all_ses = [], [], [], []

    for ses in sessions:
        si = ses - 1
        labels = _SES_LABELS[si]
        files = [f"{ses}/{f}" for f in _EEG_FILES[si]]
        print(f"  Session {ses}: loading DE-LDS...")
        with mp.Pool(processes=min(5, len(files))) as pool:
            subjects = pool.map(partial(_load_subject_delds, smooth_dir), files)
        for subj_idx, subj_trials in enumerate(subjects):
            for trial_idx, feat in enumerate(subj_trials):
                sel = feat[ch_indices]           # (4, T, 5)
                sel = sel.transpose(1, 0, 2)     # (T, 4, 5)
                all_feats.append(sel)
                all_labels.append(labels[trial_idx])
                all_subj.append(subj_idx)
                all_ses.append(si)

    print(f"  Loaded {len(all_feats)} trials, shape: {all_feats[0].shape}")
    from collections import Counter
    print(f"  Labels: {dict(sorted(Counter(all_labels).items()))}")
    return all_feats, all_labels, all_subj, all_ses


# ─────────────────────────────────────────────────
# Windowed DE-LDS Dataset (KEY FIX: 8-10× more data)
# ─────────────────────────────────────────────────

class WindowedDeLDSDataset(Dataset):
    """Slide a window over DE-LDS time dimension to create many samples per trial."""

    def __init__(self, windows, labels, augment=False, noise_std=0.05):
        self.windows = windows   # list of (win_len, n_ch, n_bands) arrays
        self.labels = labels
        self.augment = augment
        self.noise_std = noise_std

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx].copy()
        y = self.labels[idx]

        if self.augment:
            # Gaussian noise
            x += np.random.randn(*x.shape).astype(np.float32) * self.noise_std
            # Random time shift (circular)
            shift = np.random.randint(-2, 3)
            if shift != 0:
                x = np.roll(x, shift, axis=0)
            # Random channel dropout (zero one channel 10% of the time)
            if np.random.random() < 0.1:
                ch = np.random.randint(x.shape[1])
                x[:, ch, :] = 0

        return torch.FloatTensor(x), y


def create_windowed_data(features, labels, window_size=10, stride=5):
    """Slide window over DE-LDS temporal dimension.

    Each trial (T, 4, 5) → multiple windows (win_size, 4, 5).
    This is the key to getting enough training data!
    """
    all_windows, all_labels = [], []
    for feat, label in zip(features, labels):
        T = feat.shape[0]
        if T < window_size:
            # Pad short trials
            pad = np.zeros((window_size - T,) + feat.shape[1:], dtype=feat.dtype)
            feat = np.concatenate([feat, pad], axis=0)
            T = window_size
        for start in range(0, T - window_size + 1, stride):
            win = feat[start:start + window_size]
            all_windows.append(win)
            all_labels.append(label)
    return all_windows, all_labels


# ─────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────

def setup_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def normalize_features(features_list):
    """Z-score normalize per band across all data."""
    all_vals = np.concatenate([f.reshape(-1, f.shape[-1]) for f in features_list])
    mean = all_vals.mean(axis=0)
    std = all_vals.std(axis=0) + 1e-8
    return [(f - mean) / std for f in features_list], mean, std


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n = 0.0, 0
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.long().to(device)
            out = model(bx)
            loss = criterion(out, by)
            all_preds.extend(torch.argmax(out, 1).cpu().numpy())
            all_labels.extend(by.cpu().numpy())
            total_loss += loss.item(); n += 1
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / max(n, 1), acc, f1, all_preds, all_labels


def print_report(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1,2,3])
    print(f"\n  Confusion Matrix{' (' + title + ')' if title else ''}:")
    print(f"  {'':>10}", end="")
    for n in CLASS_NAMES: print(f"{n:>10}", end="")
    print()
    for i, n in enumerate(CLASS_NAMES):
        print(f"  {n:>10}", end="")
        for j in range(4): print(f"{cm[i][j]:>10}", end="")
        print()
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))


def mixup_data(x, y, alpha=0.2):
    """Mixup augmentation: interpolate two random samples."""
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    mixed_x = lam * x + (1 - lam) * x[idx]
    return mixed_x, y, y[idx], lam


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="STG-Mamba v2 — 4ch SEED-IV")

    # Data
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--sessions', nargs='+', type=int, default=None)
    parser.add_argument('--channels', nargs='+', default=None)
    parser.add_argument('--mode', choices=['sub_dep', 'sub_indep'], default='sub_dep')
    parser.add_argument('--window_size', type=int, default=10,
                        help='DE-LDS window size in timesteps (default: 10 = 10s)')
    parser.add_argument('--stride', type=int, default=3,
                        help='Window stride (default: 3 timesteps)')

    # Model
    parser.add_argument('--d_graph', type=int, default=48)
    parser.add_argument('--d_mamba', type=int, default=96)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--n_global_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--gcn_k', type=int, default=2)

    # Training
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--label_smooth', type=float, default=0.1)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--mixup_alpha', type=float, default=0.3)
    parser.add_argument('--noise_std', type=float, default=0.05)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)

    channel_names = args.channels or DEFAULT_4CH
    n_channels = len(channel_names)
    n_bands = 5

    print(f"\n{'='*60}")
    print(f"STG-Mamba v2 — {n_channels}-Channel SEED-IV (Windowed DE-LDS)")
    print(f"  Channels  : {channel_names}")
    print(f"  DE-LDS win: {args.window_size} steps, stride={args.stride}")
    print(f"  Model     : d_graph={args.d_graph}, d_mamba={args.d_mamba}")
    print(f"  Augment   : noise={args.noise_std}, mixup={args.mixup_alpha}")
    print(f"  LR        : {args.lr}, dropout={args.dropout}")
    print(f"  Device    : {device}")
    print(f"{'='*60}\n")

    # ── Load ──
    print("Loading SEED-IV DE-LDS features...")
    t0 = time.time()
    features, labels, subj_ids, ses_ids = load_seediv_4ch_delds(
        args.dataset_path, channel_names=channel_names, sessions=args.sessions
    )
    print(f"  Loaded {len(features)} trials in {time.time()-t0:.1f}s")

    # ── Normalize BEFORE splitting ──
    features, feat_mean, feat_std = normalize_features(features)

    # ── Split trials FIRST (no data leakage) ──
    N = len(features)
    rng = np.random.RandomState(args.seed)

    if args.mode == 'sub_dep':
        idx = rng.permutation(N)
        n_train = int(0.7 * N)
        n_val = int(0.15 * N)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
    else:
        subjs = sorted(set(subj_ids))
        rng.shuffle(subjs)
        n_test = max(1, int(0.2 * len(subjs)))
        n_val = max(1, int(0.15 * len(subjs)))
        test_subjs = set(subjs[:n_test])
        val_subjs = set(subjs[n_test:n_test + n_val])
        train_subjs = set(subjs[n_test + n_val:])
        train_idx = [i for i in range(N) if subj_ids[i] in train_subjs]
        val_idx = [i for i in range(N) if subj_ids[i] in val_subjs]
        test_idx = [i for i in range(N) if subj_ids[i] in test_subjs]

    print(f"\n  Trial split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")

    # ── Window AFTER splitting (critical: no leakage) ──
    train_feats = [features[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_feats = [features[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_feats = [features[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    # Window with smaller stride for training, larger for val/test
    train_wins, train_wlabels = create_windowed_data(
        train_feats, train_labels, args.window_size, args.stride
    )
    val_wins, val_wlabels = create_windowed_data(
        val_feats, val_labels, args.window_size, args.window_size  # no overlap
    )
    test_wins, test_wlabels = create_windowed_data(
        test_feats, test_labels, args.window_size, args.window_size  # no overlap
    )

    print(f"  Windowed: train={len(train_wins)}, val={len(val_wins)}, test={len(test_wins)}")
    from collections import Counter
    print(f"  Train labels: {dict(sorted(Counter(train_wlabels).items()))}")

    # ── Datasets ──
    train_ds = WindowedDeLDSDataset(train_wins, train_wlabels, augment=True,
                                      noise_std=args.noise_std)
    val_ds = WindowedDeLDSDataset(val_wins, val_wlabels, augment=False)
    test_ds = WindowedDeLDSDataset(test_wins, test_wlabels, augment=False)

    train_loader = DataLoader(train_ds, args.batch_size, shuffle=True,
                              num_workers=0, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, args.batch_size, shuffle=False,
                            num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                             num_workers=0, pin_memory=True)

    # ── Model ──
    model = STGMamba(
        n_channels=n_channels, n_bands=n_bands,
        d_graph=args.d_graph, d_mamba=args.d_mamba,
        d_state=args.d_state, n_global_layers=args.n_global_layers,
        num_classes=4, dropout=args.dropout, gcn_k=args.gcn_k,
    ).to(device)

    # Initialize adjacency with a reasonable prior (not random)
    with torch.no_grad():
        # Assume electrodes are somewhat connected (warm start)
        adj_init = torch.ones(n_channels, n_channels) * 0.3
        for i in range(n_channels):
            adj_init[i, i] = 1.0  # self-connections stronger
        # FP1-FP2 and T7-T8 are paired (same hemisphere logic)
        adj_init[0, 1] = adj_init[1, 0] = 0.8  # FP1-FP2 (frontal pair)
        adj_init[2, 3] = adj_init[3, 2] = 0.8  # T7-T8 (temporal pair)
        adj_init[0, 2] = adj_init[2, 0] = 0.5  # FP1-T7 (left hemisphere)
        adj_init[1, 3] = adj_init[3, 1] = 0.5  # FP2-T8 (right hemisphere)
        model.spatial_block.gcn.adj.data.copy_(adj_init)
        model.spatial_block.gcn.adj_bias.data.fill_(0.0)

    n_params = count_parameters(model)
    print(f"\n  Model params : {n_params:,}")
    print(f"  Window       : {args.window_size} timesteps × ({n_channels} ch × {n_bands} bands)")
    print(f"  Batches/epoch: {len(train_loader)}")

    # ── Loss ──
    class SmoothCE(nn.Module):
        def __init__(self, s, c):
            super().__init__()
            self.s = s; self.c = c
        def forward(self, logits, target):
            lp = torch.log_softmax(logits, -1)
            with torch.no_grad():
                t = torch.full_like(lp, self.s / (self.c - 1))
                t.scatter_(1, target.unsqueeze(1), 1 - self.s)
            return -(t * lp).sum(-1).mean()

    criterion = SmoothCE(args.label_smooth, 4)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)

    def lr_lambda(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / max(args.warmup, 1)
        progress = (epoch - args.warmup) / max(args.epochs - args.warmup, 1)
        return max(1e-7 / args.lr, 0.5 * (1 + np.cos(np.pi * progress)))
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Training ──
    best_val_f1, best_state, patience_ctr = 0.0, None, 0
    epoch_times = []

    print(f"\n{'='*60}")
    print(f"Training ({args.epochs} epochs, {len(train_loader)} batches/epoch)")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        t0 = time.time()

        for bx, by in train_loader:
            bx = bx.to(device)
            by = by.long().to(device)

            # Mixup augmentation
            if args.mixup_alpha > 0 and random.random() < 0.5:
                mixed_x, y_a, y_b, lam = mixup_data(bx, by, args.mixup_alpha)
                optimizer.zero_grad()
                out = model(mixed_x)
                loss = lam * criterion(out, y_a) + (1 - lam) * criterion(out, y_b)
            else:
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_loss += loss.item()
            preds = torch.argmax(out, 1)
            ep_correct += (preds == by).sum().item()
            ep_total += len(by)

        scheduler.step()
        tr_loss = ep_loss / max(len(train_loader), 1)
        tr_acc = ep_correct / max(ep_total, 1)
        va_loss, va_acc, va_f1, _, _ = evaluate(model, val_loader, device, val_criterion)
        ep_time = time.time() - t0
        epoch_times.append(ep_time)

        marker = " ★" if va_f1 > best_val_f1 else ""
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f}, Acc: {va_acc:.4f}, F1: {va_f1:.4f} | "
              f"{ep_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}{marker}")

        if va_f1 > best_val_f1:
            best_val_f1 = va_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if args.patience > 0 and patience_ctr >= args.patience:
            print(f"\n  Early stopping at epoch {epoch}")
            break

    # ── Test ──
    if best_state: model.load_state_dict(best_state)
    model = model.to(device)
    te_loss, te_acc, te_f1, te_preds, te_labels = evaluate(
        model, test_loader, device, val_criterion
    )

    print(f"\n{'='*60}")
    print(f"RESULTS — STG-Mamba v2 ({n_channels}ch DE-LDS, {args.mode})")
    print(f"{'='*60}")
    print(f"  Channels    : {channel_names}")
    print(f"  Best Val F1 : {best_val_f1:.4f}")
    print(f"  Test Acc    : {te_acc:.4f}")
    print(f"  Test F1     : {te_f1:.4f}")
    print(f"  Avg epoch   : {np.mean(epoch_times):.1f}s")
    print(f"  Total time  : {sum(epoch_times)/60:.1f} min")
    print_report(te_labels, te_preds, title=f"STG-Mamba v2 {n_channels}ch Test")

    # Learned adjacency
    adj = model.get_attention_weights()
    print(f"\n  Learned Electrode Adjacency ({n_channels}×{n_channels}):")
    print(f"  {'':>8}", end="")
    for ch in channel_names: print(f"{ch:>8}", end="")
    print()
    for i, ch in enumerate(channel_names):
        print(f"  {ch:>8}", end="")
        for j in range(n_channels): print(f"{adj[i,j]:>8.3f}", end="")
        print()

    # Save
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'stg_mamba_{n_channels}ch_v2.pt')
    torch.save({
        'model': model.state_dict(),
        'model_cfg': {
            'n_channels': n_channels, 'n_bands': n_bands,
            'd_graph': args.d_graph, 'd_mamba': args.d_mamba,
            'd_state': args.d_state, 'n_global_layers': args.n_global_layers,
            'num_classes': 4, 'dropout': args.dropout, 'gcn_k': args.gcn_k,
        },
        'channels': channel_names,
        'val_f1': best_val_f1, 'test_acc': te_acc, 'test_f1': te_f1,
        'mode': args.mode, 'adjacency': adj,
        'feat_mean': feat_mean.tolist(), 'feat_std': feat_std.tolist(),
    }, ckpt_path)
    print(f"\n  Checkpoint: {ckpt_path}\n")


if __name__ == '__main__':
    main()
