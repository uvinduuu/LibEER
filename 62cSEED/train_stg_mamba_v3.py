"""
LS-STG-Mamba v3 Training — 4-Channel SEED-IV
Latent-Space Spatio-Temporal Graph Mamba

Multi-task loss: L = L_cls + α * L_recon + β * L_kl

Usage on Kaggle:
    python train_stg_mamba_v3.py \
        --dataset_path /kaggle/input/datasets/phhasian0710/seed-iv/seed_iv \
        --epochs 150 --batch_size 128 \
        --window_size 10 --stride 3 \
        --alpha 5.0 --beta_max 0.1
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
from stg_mamba_v3 import LSSTGMamba, count_parameters

CLASS_NAMES = ['neutral', 'sad', 'fear', 'happy']

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

def get_channel_indices(channels):
    n2i = {ch.upper(): i for i, ch in enumerate(SEED_CHANNELS_62)}
    return [n2i[c.upper()] for c in channels]


def _load_subj(dir_path, fpath):
    mat = loadmat(f"{dir_path}/{fpath}")
    keys = list(mat.keys())[3:]
    return [np.array(mat[keys[i * 4 + 1]], dtype=np.float32) for i in range(24)]


def load_seediv(data_path, channels=None, sessions=None):
    if not sessions: sessions = [1, 2, 3]
    if not channels: channels = DEFAULT_4CH
    c_idx = get_channel_indices(channels)

    smooth_dir = f"{data_path}/eeg_feature_smooth"
    feats, lbls, subjs = [], [], []

    for ses in sessions:
        si = ses - 1
        labels = _SES_LABELS[si]
        files = [f"{ses}/{f}" for f in _EEG_FILES[si]]
        print(f"  Session {ses}: loading DE-LDS ...")
        with mp.Pool(min(5, len(files))) as pool:
            data = pool.map(partial(_load_subj, smooth_dir), files)
        for s_idx, s_trials in enumerate(data):
            for t_idx, feat in enumerate(s_trials):
                # feat: (62, T, 5) → select channels → (4, T, 5) → (T, 4, 5)
                feats.append(feat[c_idx].transpose(1, 0, 2))
                lbls.append(labels[t_idx])
                subjs.append(s_idx)

    from collections import Counter
    print(f"  Loaded {len(feats)} trials | Labels: {dict(sorted(Counter(lbls).items()))}")
    return feats, lbls, subjs


# ─────────────────────────────────────────────────
# Dataset with Augmentations
# ─────────────────────────────────────────────────

class WindowDataset(Dataset):
    """Windowed DE-LDS dataset with SpecAugment-style augmentations."""
    def __init__(self, windows, labels, augment=False, noise=0.05, mask_prob=0.15):
        self.windows = windows
        self.labels = labels
        self.augment = augment
        self.noise = noise
        self.mask_prob = mask_prob

    def __len__(self): return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx].copy()   # (T, C, bands)
        y = self.labels[idx]

        if self.augment:
            # 1. Gaussian noise
            x += np.random.randn(*x.shape).astype(np.float32) * self.noise

            # 2. Time masking (SpecAugment)
            if np.random.rand() < self.mask_prob:
                t_len = x.shape[0]
                mask_len = np.random.randint(1, max(2, t_len // 3))
                mask_start = np.random.randint(0, t_len - mask_len)
                x[mask_start:mask_start + mask_len, :, :] = 0.0

            # 3. Frequency-band masking (SpecAugment)
            if np.random.rand() < self.mask_prob:
                b = np.random.randint(0, x.shape[2])  # mask one DE band
                x[:, :, b] = 0.0

            # 4. Temporal circular shift
            shift = np.random.randint(-2, 3)
            if shift != 0:
                x = np.roll(x, shift, axis=0)

        return torch.FloatTensor(x), int(y)


def create_windows(feats, labels, w_size, stride):
    """Slide a window over each trial's time axis."""
    wins, w_lbls = [], []
    for x, y in zip(feats, labels):
        T = x.shape[0]
        if T < w_size:                       # pad if too short
            x = np.pad(x, ((0, w_size - T), (0, 0), (0, 0)))
            T = w_size
        for s in range(0, T - w_size + 1, stride):
            wins.append(x[s:s + w_size])
            w_lbls.append(y)
    return wins, w_lbls


# ─────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────

def setup_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def evaluate(model, loader, device, criterion, alpha, beta):
    model.eval()
    preds, targets = [], []
    t_loss = t_cls = t_rec = t_kl = 0.0
    n_batches = 0
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device)
            by = by.long().to(device)
            logits, loss_rec, loss_kl = model(bx, return_losses=True)
            loss_cls = criterion(logits, by)
            loss = loss_cls + alpha * loss_rec + beta * loss_kl

            t_loss += loss.item()
            t_cls  += loss_cls.item()
            t_rec  += loss_rec.item()
            t_kl   += loss_kl.item()
            n_batches += 1

            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(by.cpu().numpy())

    n = max(n_batches, 1)
    acc = np.mean(np.array(preds) == np.array(targets))
    f1  = f1_score(targets, preds, average='macro', zero_division=0)
    return t_loss/n, t_cls/n, t_rec/n, t_kl/n, acc, f1, preds, targets


def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def print_report(y_true, y_pred, title=""):
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2, 3])
    print(f"\n  Confusion Matrix{' (' + title + ')' if title else ''}:")
    header = f"  {'':>10}" + "".join(f"{n:>10}" for n in CLASS_NAMES)
    print(header)
    for i, n in enumerate(CLASS_NAMES):
        row = f"  {n:>10}" + "".join(f"{cm[i][j]:>10}" for j in range(4))
        print(row)
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES, digits=4))


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LS-STG-Mamba v3 — 4ch SEED-IV")

    # Data
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--channels', nargs='+', default=None,
                        help='Channels to use (default: FP1 FP2 T7 T8)')
    parser.add_argument('--sessions', nargs='+', type=int, default=None)
    parser.add_argument('--mode', choices=['sub_dep', 'sub_indep'], default='sub_dep')
    parser.add_argument('--window_size', type=int, default=10)
    parser.add_argument('--stride', type=int, default=3,
                        help='Training window stride (val/test uses window_size = no overlap)')

    # Model
    parser.add_argument('--d_latent', type=int, default=8,
                        help='Bottleneck latent dim per channel')
    parser.add_argument('--d_graph', type=int, default=24)
    parser.add_argument('--d_mamba', type=int, default=64)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--n_global_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.35)

    # Loss weights
    parser.add_argument('--alpha', type=float, default=5.0,
                        help='Reconstruction loss weight')
    parser.add_argument('--beta_max', type=float, default=0.1,
                        help='Max KL-divergence weight (warmed up over 30 epochs)')

    # Training
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=6e-4)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--patience', type=int, default=35)
    parser.add_argument('--warmup_kl', type=int, default=30,
                        help='Epochs to warm up KL weight from 0 → beta_max')
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    channel_names = args.channels or DEFAULT_4CH
    n_channels = len(channel_names)

    print(f"\n{'='*60}")
    print(f"LS-STG-Mamba v3 — {n_channels}-Channel SEED-IV (Windowed DE-LDS)")
    print(f"  Channels  : {channel_names}")
    print(f"  Latent    : {args.d_latent}-dim bottleneck per channel")
    print(f"  Loss      : Cls + {args.alpha}*Recon + KL(0→{args.beta_max} over {args.warmup_kl}ep)")
    print(f"  Window    : {args.window_size} steps, stride={args.stride}")
    print(f"  Mode      : {args.mode}")
    print(f"  Device    : {device}")
    print(f"{'='*60}\n")

    # ── Load ──
    print("Loading SEED-IV DE-LDS features...")
    t0 = time.time()
    feats, lbls, subjs = load_seediv(args.dataset_path, channel_names, args.sessions)
    print(f"  Loaded in {time.time()-t0:.1f}s\n")

    # ── Normalize per band (fit on all data, no leakage since normalization is global) ──
    all_f = np.concatenate([f.reshape(-1, 5) for f in feats])
    feat_mean = all_f.mean(0)
    feat_std  = all_f.std(0) + 1e-8
    feats = [(f - feat_mean) / feat_std for f in feats]

    # ── Split at TRIAL level (no leakage) ──
    N = len(feats)
    rng = np.random.RandomState(args.seed)

    if args.mode == 'sub_dep':
        idx = rng.permutation(N)
        n_tr = int(0.70 * N)
        n_va = int(0.15 * N)
        tr_i, va_i, te_i = idx[:n_tr], idx[n_tr:n_tr+n_va], idx[n_tr+n_va:]
    else:
        all_subjs = sorted(set(subjs))
        rng.shuffle(all_subjs)
        n_te = max(1, int(0.20 * len(all_subjs)))
        n_va = max(1, int(0.15 * len(all_subjs)))
        te_set = set(all_subjs[:n_te])
        va_set = set(all_subjs[n_te:n_te+n_va])
        tr_set = set(all_subjs[n_te+n_va:])
        tr_i = [i for i in range(N) if subjs[i] in tr_set]
        va_i = [i for i in range(N) if subjs[i] in va_set]
        te_i = [i for i in range(N) if subjs[i] in te_set]
        print(f"  Sub-indep: train={sorted(tr_set)}, val={sorted(va_set)}, test={sorted(te_set)}")

    print(f"  Trial split → train={len(tr_i)}, val={len(va_i)}, test={len(te_i)}")

    # ── Window (after split → no leakage) ──
    tr_wins, tr_y = create_windows(
        [feats[i] for i in tr_i], [lbls[i] for i in tr_i],
        args.window_size, stride=args.stride
    )
    va_wins, va_y = create_windows(
        [feats[i] for i in va_i], [lbls[i] for i in va_i],
        args.window_size, stride=args.window_size   # no overlap for val/test
    )
    te_wins, te_y = create_windows(
        [feats[i] for i in te_i], [lbls[i] for i in te_i],
        args.window_size, stride=args.window_size
    )

    print(f"  Windowed   → train={len(tr_wins)}, val={len(va_wins)}, test={len(te_wins)}")
    from collections import Counter
    print(f"  Train labels: {dict(sorted(Counter(tr_y).items()))}\n")

    # ── DataLoaders ──
    tr_dl = DataLoader(
        WindowDataset(tr_wins, tr_y, augment=True,
                      noise=0.05, mask_prob=0.15),
        batch_size=args.batch_size, shuffle=True,
        drop_last=True, num_workers=0, pin_memory=True
    )
    va_dl = DataLoader(
        WindowDataset(va_wins, va_y, augment=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    te_dl = DataLoader(
        WindowDataset(te_wins, te_y, augment=False),
        batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    # ── Model ──
    model = LSSTGMamba(
        n_channels=n_channels, n_bands=5,
        d_latent=args.d_latent, d_graph=args.d_graph,
        d_mamba=args.d_mamba, d_state=args.d_state,
        n_global_layers=args.n_global_layers,
        num_classes=4, dropout=args.dropout,
    ).to(device)

    # Electrode-prior adjacency (must come AFTER model.to(device) for init_adjacency)
    adj_prior = torch.zeros(n_channels, n_channels)
    for i in range(n_channels): adj_prior[i, i] = 1.0
    if n_channels == 4:
        adj_prior[0, 1] = adj_prior[1, 0] = 0.8   # FP1-FP2 (frontal pair)
        adj_prior[2, 3] = adj_prior[3, 2] = 0.8   # T7-T8  (temporal pair)
        adj_prior[0, 2] = adj_prior[2, 0] = 0.5   # FP1-T7 (left hemi)
        adj_prior[1, 3] = adj_prior[3, 1] = 0.5   # FP2-T8 (right hemi)
    model.init_adjacency(adj_prior)   # sets adj.data directly (no gradient issue)

    n_params = count_parameters(model)
    print(f"  Params     : {n_params:,}")
    print(f"  Batches/ep : {len(tr_dl)}\n")

    # ── Loss & Optimizer ──
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # ── Training ──
    best_val_f1 = 0.0
    best_state  = None
    patience_ctr = 0
    epoch_times = []

    print(f"{'='*60}")
    print(f"Training ({args.epochs} epochs, {len(tr_dl)} batches/epoch)")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_t0 = time.time()

        # KL weight warms up from 0 → beta_max over warmup_kl epochs
        beta = args.beta_max * min(1.0, epoch / max(args.warmup_kl, 1))

        ep_cls = ep_rec = ep_kl = 0.0
        tr_correct = tr_total = 0

        for bx, by in tr_dl:
            bx = bx.to(device)
            by = by.long().to(device)

            # Mixup 50% of the time
            if random.random() < 0.5:
                bx, y_a, y_b, lam = mixup_data(bx, by)
                logits, l_rec, l_kl = model(bx, return_losses=True)
                l_cls = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                # Approx train acc from the non-mixed labels
                preds_b = logits.argmax(1)
                tr_correct += (preds_b == y_a).sum().item()  # approx
                tr_total   += len(by)
            else:
                logits, l_rec, l_kl = model(bx, return_losses=True)
                l_cls = criterion(logits, by)
                tr_correct += (logits.argmax(1) == by).sum().item()
                tr_total   += len(by)

            loss = l_cls + args.alpha * l_rec + beta * l_kl

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_cls += l_cls.item()
            ep_rec += l_rec.item()
            ep_kl  += l_kl.item()

        scheduler.step()

        n_b    = len(tr_dl)
        tr_acc = tr_correct / max(tr_total, 1)
        ep_time = time.time() - ep_t0
        epoch_times.append(ep_time)

        vl_loss, vl_cls, vl_rec, vl_kl, v_acc, v_f1, _, _ = evaluate(
            model, va_dl, device, criterion, args.alpha, beta
        )

        star = " ★" if v_f1 > best_val_f1 else ""
        print(
            f"  Ep {epoch:3d}/{args.epochs} | "
            f"Tr  Cls:{ep_cls/n_b:.3f} Rec:{ep_rec/n_b:.3f} KL:{ep_kl/n_b:.4f} Acc:{tr_acc:.3f} | "
            f"Va  Cls:{vl_cls:.3f} Acc:{v_acc:.3f} F1:{v_f1:.3f} | "
            f"β={beta:.3f} {ep_time:.1f}s{star}"
        )

        if v_f1 > best_val_f1:
            best_val_f1 = v_f1
            best_state  = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # ── Test ──
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    _, _, _, _, te_acc, te_f1, te_preds, te_lbl = evaluate(
        model, te_dl, device, criterion, args.alpha, beta
    )

    print(f"\n{'='*60}")
    print(f"RESULTS — LS-STG-Mamba v3 ({n_channels}ch DE-LDS, {args.mode})")
    print(f"{'='*60}")
    print(f"  Channels    : {channel_names}")
    print(f"  Best Val F1 : {best_val_f1:.4f}")
    print(f"  Test Acc    : {te_acc:.4f}")
    print(f"  Test F1     : {te_f1:.4f}")
    print(f"  Avg ep time : {np.mean(epoch_times):.1f}s")
    print(f"  Total time  : {sum(epoch_times)/60:.1f} min")
    print_report(te_lbl, te_preds, title=f"LS-STG-Mamba v3 {n_channels}ch")

    # ── Learned adjacency ──
    adj = model.get_adjacency()
    print(f"\n  Learned Electrode Adjacency ({n_channels}×{n_channels}):")
    print(f"  {'':>8}" + "".join(f"{c:>8}" for c in channel_names))
    for i, c in enumerate(channel_names):
        print(f"  {c:>8}" + "".join(f"{adj[i,j]:>8.3f}" for j in range(n_channels)))

    # ── Save checkpoint ──
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, f'ls_stg_mamba_{n_channels}ch_v3.pt')
    torch.save({
        'model': model.state_dict(),
        'model_cfg': {
            'n_channels': n_channels, 'n_bands': 5,
            'd_latent': args.d_latent, 'd_graph': args.d_graph,
            'd_mamba': args.d_mamba, 'd_state': args.d_state,
            'n_global_layers': args.n_global_layers,
            'num_classes': 4, 'dropout': args.dropout,
        },
        'channels': channel_names,
        'val_f1': best_val_f1,
        'test_acc': te_acc,
        'test_f1': te_f1,
        'mode': args.mode,
        'adjacency': adj.tolist(),
        'feat_mean': feat_mean.tolist(),
        'feat_std': feat_std.tolist(),
    }, ckpt)
    print(f"\n  Checkpoint saved: {ckpt}\n")


if __name__ == '__main__':
    main()
