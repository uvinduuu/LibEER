"""
STG-Mamba Training — 4-Channel SEED-IV with DE-LDS Features.

Novel architecture that jointly models spatial (graph), temporal (Mamba),
and spectral (DE-LDS) dimensions of EEG for emotion recognition.

Uses 4 channels selected from SEED-IV's 62-channel DE-LDS features:
  Default: FP1, FP2, T7, T8 (frontal + temporal — key emotion regions)

Usage on Kaggle:
    python train_stg_mamba_4ch.py \
        --dataset_path /kaggle/input/datasets/phhasian0710/seed-iv/seed_iv \
        --epochs 150 --batch_size 64

Expected: 40-48% (4-channel, 4-class SEED-IV)
"""

import os, sys, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from scipy.io import loadmat
import multiprocessing as mp
from functools import partial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from stg_mamba_model import STGMamba, count_parameters


CLASS_NAMES = ['neutral', 'sad', 'fear', 'happy']

# SEED-IV 62-channel order (standard 10-20 montage)
SEED_CHANNELS_62 = [
    'FP1','FPZ','FP2','AF3','AF4','F7','F5','F3','F1','FZ',
    'F2','F4','F6','F8','FT7','FC5','FC3','FC1','FCZ','FC2',
    'FC4','FC6','FT8','C5','C3','C1','CZ','C2','C4','C6',
    'T8','TP7','CP5','CP3','CP1','CPZ','CP2','CP4','CP6','TP8',
    'P7','P5','P3','P1','PZ','P2','P4','P6','P8','PO7',
    'PO5','PO3','POZ','PO4','PO6','PO8','CB1','O1','OZ','O2',
    'CB2', 'T7'   # T7 is often listed as the 62nd channel
]

# Default 4 channels for emotion recognition (frontal + temporal)
# These are the most informative for emotion based on neuroscience literature
DEFAULT_4CH = ['FP1', 'FP2', 'T7', 'T8']

# Session labels
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
# Data loading: 4-channel DE-LDS from eeg_feature_smooth/
# ─────────────────────────────────────────────────

def get_channel_indices(channel_names, all_channels=SEED_CHANNELS_62):
    """Map channel names to indices in the 62-channel array."""
    name_to_idx = {ch.upper(): i for i, ch in enumerate(all_channels)}
    indices = []
    for name in channel_names:
        name_upper = name.upper()
        if name_upper in name_to_idx:
            indices.append(name_to_idx[name_upper])
        else:
            raise ValueError(f"Channel '{name}' not found. Available: {list(name_to_idx.keys())}")
    return indices


def _load_subject_delds(dir_path, file_relpath):
    """Load one subject's DE-LDS features from eeg_feature_smooth/.
    Returns list of 24 trials, each (62, T, 5)."""
    mat = loadmat(f"{dir_path}/{file_relpath}")
    keys = list(mat.keys())[3:]
    trials = []
    for i in range(24):
        feat = np.array(mat[keys[i * 4 + 1]])  # de_LDS index = 1
        trials.append(feat.astype(np.float32))  # (62, T, 5)
    return trials


def load_seediv_4ch_delds(dataset_path, channel_names=None, sessions=None):
    """Load SEED-IV DE-LDS features, selecting only specified channels.
    
    Returns:
        features: list of (T, n_ch, 5) arrays
        labels: list of int
        subj_ids, ses_ids: list of int
    """
    if sessions is None:
        sessions = [1, 2, 3]
    if channel_names is None:
        channel_names = DEFAULT_4CH
    
    ch_indices = get_channel_indices(channel_names)
    print(f"  Selecting channels: {channel_names} → indices {ch_indices}")
    
    smooth_dir = f"{dataset_path}/eeg_feature_smooth"
    all_features, all_labels, all_subj_ids, all_ses_ids = [], [], [], []
    
    for ses_1based in sessions:
        ses_idx = ses_1based - 1
        ses_labels = _SES_LABELS[ses_idx]
        ses_files = [f"{ses_1based}/{f}" for f in _EEG_FILES[ses_idx]]
        
        print(f"  Session {ses_1based}: loading DE-LDS features...")
        with mp.Pool(processes=min(5, len(ses_files))) as pool:
            subjects_data = pool.map(
                partial(_load_subject_delds, smooth_dir),
                ses_files
            )
        
        for subj_idx, subj_trials in enumerate(subjects_data):
            for trial_idx, trial_feat in enumerate(subj_trials):
                # trial_feat: (62, T, 5) → select channels → (4, T, 5)
                selected = trial_feat[ch_indices]  # (4, T, 5)
                selected = selected.transpose(1, 0, 2)  # → (T, 4, 5)
                all_features.append(selected)
                all_labels.append(ses_labels[trial_idx])
                all_subj_ids.append(subj_idx)
                all_ses_ids.append(ses_idx)
    
    print(f"  Total: {len(all_features)} trials")
    print(f"  Feature shape: {all_features[0].shape}")  # (T, 4, 5)
    from collections import Counter
    print(f"  Label dist: {dict(sorted(Counter(all_labels).items()))}")
    
    return all_features, all_labels, all_subj_ids, all_ses_ids


# ─────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────

def setup_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_or_truncate(feat, target_len):
    """Pad/truncate to uniform sequence length."""
    T = feat.shape[0]
    if T >= target_len:
        return feat[:target_len]
    pad = np.zeros((target_len - T,) + feat.shape[1:], dtype=feat.dtype)
    return np.concatenate([feat, pad], axis=0)


def normalize_features(features_list):
    """Z-score normalize across all trials. Normalize per-band."""
    # Compute stats: (n_bands,)
    all_vals = np.concatenate([f.reshape(-1, f.shape[-1]) for f in features_list])
    mean = all_vals.mean(axis=0)
    std = all_vals.std(axis=0) + 1e-8
    normalized = [(f - mean) / std for f in features_list]
    return normalized, mean, std


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.to(device)
            out = model(bx)
            loss = criterion(out, by)
            all_preds.extend(torch.argmax(out, 1).cpu().numpy())
            all_labels.extend(by.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / max(n_batches, 1), acc, f1, all_preds, all_labels


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


# ─────────────────────────────────────────────────
# Main training
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="STG-Mamba — 4ch SEED-IV DE-LDS")
    
    # Data
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--sessions', nargs='+', type=int, default=None)
    parser.add_argument('--channels', nargs='+', default=None,
                        help='Channel names to use (default: FP1 FP2 T7 T8)')
    parser.add_argument('--mode', choices=['sub_dep', 'sub_indep'], default='sub_dep')
    
    # Model  
    parser.add_argument('--d_graph', type=int, default=32,
                        help='Graph conv hidden dim per channel')
    parser.add_argument('--d_mamba', type=int, default=64,
                        help='Global Mamba hidden dim')
    parser.add_argument('--d_state', type=int, default=16,
                        help='SSM state dimension')
    parser.add_argument('--n_global_layers', type=int, default=3,
                        help='Number of global Mamba layers')
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--gcn_k', type=int, default=2,
                        help='Chebyshev polynomial order')
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--label_smooth', type=float, default=0.1)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)
    
    channel_names = args.channels or DEFAULT_4CH
    n_channels = len(channel_names)
    n_bands = 5
    
    print(f"\n{'='*60}")
    print(f"STG-Mamba — {n_channels}-Channel SEED-IV (DE-LDS)")
    print(f"  Channels : {channel_names}")
    print(f"  Features : DE-LDS ({n_channels}ch × {n_bands} bands)")
    print(f"  Model    : d_graph={args.d_graph}, d_mamba={args.d_mamba}")
    print(f"             layers={args.n_global_layers}, GCN k={args.gcn_k}")
    print(f"  Dropout  : {args.dropout}, smooth={args.label_smooth}")
    print(f"  LR       : {args.lr} (warmup={args.warmup}ep, cosine)")
    print(f"  Mode     : {args.mode}")
    print(f"  Device   : {device}")
    print(f"{'='*60}\n")
    
    # ── Load DE-LDS features (4ch) ──
    print("Loading SEED-IV DE-LDS features...")
    t0 = time.time()
    features, labels, subj_ids, ses_ids = load_seediv_4ch_delds(
        args.dataset_path, channel_names=channel_names, sessions=args.sessions
    )
    print(f"  Loaded {len(features)} trials in {time.time()-t0:.1f}s")
    
    # ── Determine target sequence length ──
    lengths = [f.shape[0] for f in features]
    target_len = int(np.median(lengths))
    print(f"  Seq lengths: min={min(lengths)}, median={target_len}, max={max(lengths)}")
    
    # ── Normalize and pad ──
    features, feat_mean, feat_std = normalize_features(features)
    padded = [pad_or_truncate(f, target_len) for f in features]
    
    X = np.stack(padded)  # (N, target_len, n_channels, n_bands)
    y = np.array(labels)
    print(f"  Data shape: {X.shape}, labels: {y.shape}")
    
    # ── Train/Val/Test split ──
    N = len(X)
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
        print(f"  Train subjs: {sorted(train_subjs)}")
        print(f"  Val subjs:   {sorted(val_subjs)}")
        print(f"  Test subjs:  {sorted(test_subjs)}")
        train_idx = [i for i in range(N) if subj_ids[i] in train_subjs]
        val_idx = [i for i in range(N) if subj_ids[i] in val_subjs]
        test_idx = [i for i in range(N) if subj_ids[i] in test_subjs]
    
    print(f"  Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}")
    from collections import Counter
    print(f"  Train labels: {dict(sorted(Counter(y[train_idx]).items()))}")
    print(f"  Val labels:   {dict(sorted(Counter(y[val_idx]).items()))}")
    print(f"  Test labels:  {dict(sorted(Counter(y[test_idx]).items()))}")
    
    # ── DataLoaders ──
    def make_loader(indices, shuffle):
        ds = TensorDataset(
            torch.FloatTensor(X[indices]),
            torch.LongTensor(y[indices])
        )
        return DataLoader(ds, batch_size=args.batch_size, shuffle=shuffle,
                          num_workers=0, pin_memory=True)
    
    train_loader = make_loader(train_idx, True)
    val_loader = make_loader(val_idx, False)
    test_loader = make_loader(test_idx, False)
    
    # ── Model ──
    model = STGMamba(
        n_channels=n_channels,
        n_bands=n_bands,
        d_graph=args.d_graph,
        d_mamba=args.d_mamba,
        d_state=args.d_state,
        n_global_layers=args.n_global_layers,
        num_classes=4,
        dropout=args.dropout,
        gcn_k=args.gcn_k,
    ).to(device)
    
    n_params = count_parameters(model)
    print(f"\n  Model params : {n_params:,}")
    print(f"  Seq length   : {target_len} time-steps")
    print(f"  Input        : ({n_channels}, {n_bands}) per step")
    
    # ── Loss & Optimizer ──
    class SmoothCE(nn.Module):
        def __init__(self, smoothing, n_classes):
            super().__init__()
            self.smoothing = smoothing
            self.n_classes = n_classes
        def forward(self, logits, target):
            lp = torch.log_softmax(logits, -1)
            with torch.no_grad():
                smooth = torch.full_like(lp, self.smoothing / (self.n_classes - 1))
                smooth.scatter_(1, target.unsqueeze(1), 1 - self.smoothing)
            return -(smooth * lp).sum(-1).mean()
    
    criterion = SmoothCE(args.label_smooth, 4)
    val_criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, eps=1e-8)
    
    # Warmup + cosine
    def lr_lambda(epoch):
        if epoch < args.warmup:
            return (epoch + 1) / max(args.warmup, 1)
        progress = (epoch - args.warmup) / max(args.epochs - args.warmup, 1)
        return max(1e-7 / args.lr, 0.5 * (1 + np.cos(np.pi * progress)))
    
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    
    # ── Training loop ──
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
            bx = bx.to(device)  # (B, T, 4, 5)
            by = by.to(device)
            
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            ep_loss += loss.item()
            ep_correct += (torch.argmax(out, 1) == by).sum().item()
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
    if best_state:
        model.load_state_dict(best_state)
    model = model.to(device)
    te_loss, te_acc, te_f1, te_preds, te_labels = evaluate(
        model, test_loader, device, val_criterion
    )
    
    print(f"\n{'='*60}")
    print(f"RESULTS — STG-Mamba ({n_channels}ch DE-LDS, {args.mode})")
    print(f"{'='*60}")
    print(f"  Channels    : {channel_names}")
    print(f"  Best Val F1 : {best_val_f1:.4f}")
    print(f"  Test Acc    : {te_acc:.4f}")
    print(f"  Test F1     : {te_f1:.4f}")
    print(f"  Avg epoch   : {np.mean(epoch_times):.1f}s")
    print(f"  Total time  : {sum(epoch_times)/60:.1f} min")
    print_report(te_labels, te_preds, title=f"STG-Mamba {n_channels}ch Test")
    
    # ── Show learned adjacency ──
    adj = model.get_attention_weights()
    print(f"\n  Learned Electrode Adjacency ({n_channels}×{n_channels}):")
    print(f"  {'':>8}", end="")
    for ch in channel_names: print(f"{ch:>8}", end="")
    print()
    for i, ch in enumerate(channel_names):
        print(f"  {ch:>8}", end="")
        for j in range(n_channels):
            print(f"{adj[i,j]:>8.3f}", end="")
        print()
    
    # ── Save checkpoint ──
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, f'stg_mamba_{n_channels}ch.pt')
    torch.save({
        'model': model.state_dict(),
        'model_cfg': {
            'n_channels': n_channels, 'n_bands': n_bands,
            'd_graph': args.d_graph, 'd_mamba': args.d_mamba,
            'd_state': args.d_state, 'n_global_layers': args.n_global_layers,
            'num_classes': 4, 'dropout': args.dropout, 'gcn_k': args.gcn_k,
        },
        'channels': channel_names,
        'val_f1': best_val_f1,
        'test_acc': te_acc,
        'test_f1': te_f1,
        'mode': args.mode,
        'adjacency': adj,
        'feat_mean': feat_mean.tolist(),
        'feat_std': feat_std.tolist(),
    }, ckpt_path)
    print(f"\n  Checkpoint: {ckpt_path}\n")


if __name__ == '__main__':
    main()
