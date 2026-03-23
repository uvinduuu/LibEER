"""
Mamba on SEED-IV DE-LDS Features — 62 Channels.

Instead of training on raw EEG (which requires windowing + bandpass),
this uses the pre-computed DE-LDS features from SEED-IV's eeg_feature_smooth/.
Each trial becomes a sequence of time-steps, each with shape (62, 5) = 310-dim,
giving Mamba a feature-rich temporal sequence to model.

DE-LDS = Differential Entropy with Linear Dynamic System smoothing.
This is what SOTA papers use to achieve 50-55% on SEED-IV (4-class).

Usage on Kaggle:
    python train_mamba_delds.py \
        --dataset_path /kaggle/input/datasets/phhasian0710/seed-iv/seed_iv \
        --epochs 150 --batch_size 64

Expected:
    ~50-55% accuracy (matching LibEER DGCNN benchmark)
    Potentially higher due to Mamba's temporal modeling
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

# Reuse MambaEEGClassifier — we'll create a lightweight wrapper
mamba_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'mamba')
sys.path.insert(0, mamba_dir)

CLASS_NAMES = ['neutral', 'sad', 'fear', 'happy']
NUM_CHANNELS = 62
NUM_BANDS = 5  # delta, theta, alpha, beta, gamma

# SEED-IV session labels
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
# Data loading: DE-LDS features from eeg_feature_smooth/
# ─────────────────────────────────────────────────

def _load_subject_delds(dir_path, file_relpath):
    """Load one subject's DE-LDS features.
    
    In eeg_feature_smooth/, each .mat has keys like:
        de_movingAve1, de_LDS1, psd_movingAve1, psd_LDS1,
        de_movingAve2, de_LDS2, ...  (4 features × 24 trials)
    
    de_LDS index = 1 (0=de_movingAve, 1=de_LDS, 2=psd_movingAve, 3=psd_LDS)
    
    Each trial's DE-LDS: shape (channels=62, time_steps, bands=5)
    We transpose to (time_steps, channels, bands) for our model.
    """
    mat = loadmat(f"{dir_path}/{file_relpath}")
    keys = list(mat.keys())[3:]  # skip __header__, __version__, __globals__
    
    trials = []
    for i in range(24):
        # de_LDS is at index 1 within each group of 4
        feat = np.array(mat[keys[i * 4 + 1]])  # (channels, time_steps, bands)
        # Transpose to (time_steps, channels × bands) — flatten ch×band
        # Or keep as (time_steps, channels, bands) for more flexible models
        feat = feat.transpose(1, 0, 2)  # → (time_steps, 62, 5)
        trials.append(feat.astype(np.float32))
    return trials


def load_seediv_delds(dataset_path, sessions=None):
    """Load SEED-IV DE-LDS features.
    
    Returns:
        all_features: list of (T, 62, 5) arrays — one per trial
        all_labels:   list of int labels
        all_subj_ids: list of subject IDs
        all_ses_ids:  list of session IDs
    """
    if sessions is None:
        sessions = [1, 2, 3]
    
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
                all_features.append(trial_feat)
                all_labels.append(ses_labels[trial_idx])
                all_subj_ids.append(subj_idx)
                all_ses_ids.append(ses_idx)
    
    print(f"  Total: {len(all_features)} trials")
    print(f"  Feature shape example: {all_features[0].shape}")  # (T, 62, 5)
    from collections import Counter
    print(f"  Label dist: {dict(sorted(Counter(all_labels).items()))}")
    
    return all_features, all_labels, all_subj_ids, all_ses_ids


# ─────────────────────────────────────────────────
# Mamba Model for DE-LDS features  
# ─────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """Simplified selective state-space block (Mamba-style)."""
    def __init__(self, d_model, d_state=16, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        # Selective scan approximation using conv + gating
        self.conv1d = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, groups=d_model)
        self.gate_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
        # SSM params
        self.A = nn.Parameter(torch.randn(d_model, d_state) * 0.01)
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.dt_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Softplus()
        )
        
    def forward(self, x):
        # x: (B, T, D)
        residual = x
        x = self.norm(x)
        
        # Conv branch
        conv_out = self.conv1d(x.transpose(1, 2)).transpose(1, 2)
        
        # Gate
        gate = self.gate_proj(x)
        g1, g2 = gate.chunk(2, dim=-1)
        x = conv_out * torch.sigmoid(g1) + x * torch.sigmoid(g2)
        
        # SSM-style recurrence (parallel scan approximation)
        dt = self.dt_proj(x)
        B = self.B_proj(x)
        C = self.C_proj(x)
        
        # Discretized SSM
        dA = torch.exp(-dt.unsqueeze(-1) * self.A.abs())  # (B, T, D, N)
        dB = dt.unsqueeze(-1) * B.unsqueeze(2)  # (B, T, D, N)
        
        # Scan (simplified — use cumulative for speed)
        x_ssm = x.unsqueeze(-1) * dB  # (B, T, D, N)
        x_ssm = x_ssm.cumsum(dim=1)  # approximate scan
        y = (x_ssm * C.unsqueeze(2)).sum(-1)  # (B, T, D)
        
        x = self.out_proj(y)
        x = self.dropout(x)
        return x + residual


class MambaDeLDS(nn.Module):
    """Mamba classifier for DE-LDS EEG features.
    
    Input: (batch, time_steps, 62 * 5) = (batch, T, 310)
    Uses Mamba blocks for temporal modeling, then classification.
    """
    def __init__(self, input_dim=310, d_model=256, n_layers=4, d_state=16,
                 num_classes=4, dropout=0.3):
        super().__init__()
        
        # Project input features to model dimension
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Mamba blocks
        self.blocks = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout) for _ in range(n_layers)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )
        
    def forward(self, x):
        # x: (B, T, 310)
        x = self.input_proj(x)  # → (B, T, d_model)
        
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        # Global average pooling over time
        x = x.mean(dim=1)  # → (B, d_model)
        return self.head(x)


# ─────────────────────────────────────────────────
# Training utilities
# ─────────────────────────────────────────────────

def setup_seed(seed):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def pad_or_truncate(features, target_len):
    """Pad/truncate sequences to uniform length."""
    T = features.shape[0]
    if T >= target_len:
        return features[:target_len]
    else:
        pad = np.zeros((target_len - T,) + features.shape[1:], dtype=features.dtype)
        return np.concatenate([features, pad], axis=0)


def normalize_features(features_list):
    """Z-score normalize DE-LDS features across all trials."""
    # Compute mean/std from all trials
    all_feats = np.concatenate([f.reshape(-1, f.shape[-1]) for f in features_list])
    mean = all_feats.mean(axis=0)
    std = all_feats.std(axis=0) + 1e-8
    return [(f - mean) / std for f in features_list], mean, std


def evaluate(model, loader, device, criterion):
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.long().to(device)
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
# Main
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Mamba + DE-LDS on SEED-IV (62ch)")
    
    # Data
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--sessions', nargs='+', type=int, default=None)
    parser.add_argument('--mode', choices=['sub_dep', 'sub_indep'], default='sub_dep')
    
    # Model
    parser.add_argument('--d_model', type=int, default=256)
    parser.add_argument('--n_layers', type=int, default=4)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--dropout', type=float, default=0.3)
    
    # Training
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--label_smooth', type=float, default=0.1)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    
    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)
    
    print(f"\n{'='*60}")
    print(f"Mamba + DE-LDS — 62ch SEED-IV")
    print(f"  Features : DE-LDS (62ch × 5 bands = 310-dim per timestep)")
    print(f"  Model    : d_model={args.d_model}, layers={args.n_layers}")
    print(f"  Dropout  : {args.dropout}, smooth={args.label_smooth}")
    print(f"  LR       : {args.lr} (warmup={args.warmup}ep, cosine)")
    print(f"  Mode     : {args.mode}")
    print(f"  Device   : {device}")
    print(f"{'='*60}\n")
    
    # ── Load DE-LDS features ──
    print("Loading SEED-IV DE-LDS features...")
    t0 = time.time()
    features, labels, subj_ids, ses_ids = load_seediv_delds(
        args.dataset_path, sessions=args.sessions
    )
    print(f"  Loaded {len(features)} trials in {time.time()-t0:.1f}s")
    
    # ── Determine target sequence length ──
    lengths = [f.shape[0] for f in features]
    med_len = int(np.median(lengths))
    target_len = med_len
    print(f"  Sequence lengths: min={min(lengths)}, median={med_len}, max={max(lengths)}")
    print(f"  Using target_len = {target_len}")
    
    # ── Flatten 62×5 → 310 per timestep, normalize, pad ──
    flat_features = [f.reshape(f.shape[0], -1) for f in features]  # (T, 310)
    flat_features, feat_mean, feat_std = normalize_features(flat_features)
    padded = [pad_or_truncate(f, target_len) for f in flat_features]
    
    X = np.stack(padded)  # (N, target_len, 310)
    y = np.array(labels)
    
    print(f"  Final data shape: {X.shape}, labels: {y.shape}")
    
    # ── Train/Val/Test split ──
    N = len(X)
    rng = np.random.RandomState(args.seed)
    
    if args.mode == 'sub_dep':
        # Shuffle trials, split 70/15/15
        idx = rng.permutation(N)
        n_train = int(0.7 * N)
        n_val = int(0.15 * N)
        train_idx = idx[:n_train]
        val_idx = idx[n_train:n_train + n_val]
        test_idx = idx[n_train + n_val:]
    else:
        # Subject-independent split
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
    
    # Check label distribution
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
    input_dim = NUM_CHANNELS * NUM_BANDS  # 310
    model = MambaDeLDS(
        input_dim=input_dim,
        d_model=args.d_model,
        n_layers=args.n_layers,
        d_state=args.d_state,
        num_classes=4,
        dropout=args.dropout,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\n  Model params: {n_params:,}")
    print(f"  Sequence len: {target_len} time-steps")
    print(f"  Input dim:    {input_dim}")
    
    # ── Loss & Optimizer ──
    class SmoothCE(nn.Module):
        def forward(self, logits, target):
            lp = torch.log_softmax(logits, -1)
            with torch.no_grad():
                smooth = torch.full_like(lp, args.label_smooth / 3)
                smooth.scatter_(1, target.unsqueeze(1), 1 - args.label_smooth)
            return -(smooth * lp).sum(-1).mean()
    
    criterion = SmoothCE()
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
            bx = bx.to(device)
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
        
        print(f"  Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {tr_loss:.4f}, Acc: {tr_acc:.4f} | "
              f"Val Loss: {va_loss:.4f}, Acc: {va_acc:.4f}, F1: {va_f1:.4f} | "
              f"{ep_time:.1f}s | LR: {optimizer.param_groups[0]['lr']:.6f}")
        
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
    print(f"RESULTS — Mamba + DE-LDS on SEED-IV (62ch, {args.mode})")
    print(f"{'='*60}")
    print(f"  Best Val F1 : {best_val_f1:.4f}")
    print(f"  Test Acc    : {te_acc:.4f}")
    print(f"  Test F1     : {te_f1:.4f}")
    print(f"  Avg epoch   : {np.mean(epoch_times):.1f}s")
    print(f"  Total time  : {sum(epoch_times)/60:.1f} min")
    print_report(te_labels, te_preds, title="SEED-IV 62ch DE-LDS Test")
    
    # Save checkpoint
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(ckpt_dir, 'best_mamba_delds.pt')
    torch.save({
        'model': model.state_dict(),
        'model_cfg': {
            'input_dim': input_dim, 'num_classes': 4,
            'd_model': args.d_model, 'n_layers': args.n_layers,
            'd_state': args.d_state, 'dropout': args.dropout,
        },
        'val_f1': best_val_f1,
        'test_acc': te_acc,
        'test_f1': te_f1,
        'mode': args.mode,
        'feat_mean': feat_mean.tolist(),
        'feat_std': feat_std.tolist(),
    }, ckpt_path)
    print(f"\n  Checkpoint: {ckpt_path}\n")


if __name__ == '__main__':
    main()
