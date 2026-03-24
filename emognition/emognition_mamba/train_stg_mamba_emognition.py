"""
LS-STG-Mamba Direct Training on Emognition Dataset.

Trains the novel Latent-Space Spatio-Temporal Graph Mamba architecture
from scratch on the Emognition dataset (Muse 2, 4 channels).

Pipeline:
  Raw Muse 2 EEG (TP9, AF7, AF8, TP10)
    → DE-LDS feature extraction (1s windows, 5 bands) 
    → Windowed DE-LDS sequences
    → LS-STG-Mamba training

This is the BASELINE (no transfer learning) experiment.
Compare with finetune_stg_mamba_emognition.py to measure TL benefit.

Usage on Kaggle:
    python train_stg_mamba_emognition.py \
        --data_root /kaggle/input/datasets/uvindukodikara/emognition \
        --epochs 150 --batch_size 64 --mode sub_dep

Classes (sorted alphabetically → integer IDs):
    ENTHUSIASM=0, FEAR=1, NEUTRAL=2, SADNESS=3
"""

import os, sys, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ── Local imports ──
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from emognition_loader import load_emognition_trials, FS
from delds_extractor import compute_delds_batch

# ── LS-STG-Mamba model ──
_MODEL_DIR = os.path.join(_HERE, '..', '..', '62cSEED')
sys.path.insert(0, _MODEL_DIR)
from stg_mamba_v3 import LSSTGMamba, count_parameters

CLASS_NAMES_SORTED = ['ENTHUSIASM', 'FEAR', 'NEUTRAL', 'SADNESS']  # alphabetical
CHANNELS = ['TP9', 'AF7', 'AF8', 'TP10']   # Muse 2 channels


# ─────────────────────────────────────────────────
# Dataset
# ─────────────────────────────────────────────────

class DeLDSWindowDataset(Dataset):
    """Windowed DE-LDS dataset with augmentations."""
    def __init__(self, windows, labels, augment=False, noise=0.05, mask_prob=0.15):
        self.windows = windows
        self.labels = labels
        self.augment = augment
        self.noise = noise
        self.mask_prob = mask_prob

    def __len__(self): return len(self.windows)

    def __getitem__(self, idx):
        x = self.windows[idx].copy()   # (T, C, 5)
        y = int(self.labels[idx])
        if self.augment:
            x += np.random.randn(*x.shape).astype(np.float32) * self.noise
            if np.random.rand() < self.mask_prob:
                t = x.shape[0]
                ml = np.random.randint(1, max(2, t // 3))
                ms = np.random.randint(0, t - ml)
                x[ms:ms+ml] = 0.0
            if np.random.rand() < self.mask_prob:
                x[:, :, np.random.randint(5)] = 0.0
            shift = np.random.randint(-2, 3)
            if shift: x = np.roll(x, shift, axis=0)
        return torch.FloatTensor(x), y


def create_windows(feats, labels, w_size, stride):
    wins, wlbls = [], []
    for x, y in zip(feats, labels):
        T = x.shape[0]
        if T < w_size:
            x = np.pad(x, ((0, w_size - T), (0, 0), (0, 0)))
            T = w_size
        for s in range(0, T - w_size + 1, stride):
            wins.append(x[s:s + w_size])
            wlbls.append(y)
    return wins, wlbls


# ─────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────

def setup_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
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
            tloss += loss.item(); tcls += lcls.item()
            trec += lrec.item(); tkl += lkl.item(); n += 1
            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(by.cpu().numpy())
    nb = max(n, 1)
    acc = np.mean(np.array(preds) == np.array(targets))
    f1  = f1_score(targets, preds, average='macro', zero_division=0)
    return tloss/nb, tcls/nb, trec/nb, tkl/nb, acc, f1, preds, targets


def mixup_data(x, y, alpha=0.3):
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.
    idx = torch.randperm(x.size(0), device=x.device)
    return lam * x + (1 - lam) * x[idx], y, y[idx], lam


def print_report(y_true, y_pred, class_names, title=""):
    cm = confusion_matrix(y_true, y_pred, labels=list(range(len(class_names))))
    print(f"\n  Confusion Matrix{' (' + title + ')' if title else ''}:")
    print(f"  {'':>12}" + "".join(f"{n:>12}" for n in class_names))
    for i, n in enumerate(class_names):
        print(f"  {n:>12}" + "".join(f"{cm[i][j]:>12}" for j in range(len(class_names))))
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="LS-STG-Mamba on Emognition (from scratch)")

    # Data
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--mode', choices=['sub_dep', 'sub_indep'], default='sub_dep')
    parser.add_argument('--delds_window_sec', type=float, default=1.0,
                        help='Window size for DE-LDS computation (seconds)')
    parser.add_argument('--seq_window', type=int, default=10,
                        help='Number of DE-LDS windows per training sample')
    parser.add_argument('--stride', type=int, default=3)

    # Model
    parser.add_argument('--d_latent', type=int, default=8)
    parser.add_argument('--d_graph', type=int, default=24)
    parser.add_argument('--d_mamba', type=int, default=64)
    parser.add_argument('--d_state', type=int, default=16)
    parser.add_argument('--n_global_layers', type=int, default=3)
    parser.add_argument('--dropout', type=float, default=0.35)

    # Loss
    parser.add_argument('--alpha', type=float, default=5.0)
    parser.add_argument('--beta_max', type=float, default=0.1)
    parser.add_argument('--warmup_kl', type=int, default=20)

    # Training
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--patience', type=int, default=30)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"LS-STG-Mamba — Emognition (Direct, No Transfer Learning)")
    print(f"  Channels : {CHANNELS}")
    print(f"  DE-LDS   : {args.delds_window_sec}s windows, 5 bands")
    print(f"  Seq win  : {args.seq_window} DE steps × stride={args.stride}")
    print(f"  Mode     : {args.mode}")
    print(f"  Device   : {device}")
    print(f"{'='*60}\n")

    # ── Load raw trials ──
    print("Loading Emognition trials...")
    t0 = time.time()
    trials, labels, subj_ids, lab2id, id2lab = load_emognition_trials(args.data_root)
    class_names = [id2lab[i] for i in range(len(id2lab))]
    n_classes = len(class_names)
    print(f"  Loaded {len(trials)} trials in {time.time()-t0:.1f}s")
    print(f"  Classes: {class_names}")

    # ── Compute DE-LDS from raw EEG ──
    print(f"\nComputing DE-LDS features (1s windows, 5 bands)...")
    t0 = time.time()
    features = compute_delds_batch(trials, fs=FS,
                                   window_sec=args.delds_window_sec,
                                   step_sec=args.delds_window_sec)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Normalize globally per band ──
    all_f = np.concatenate([f.reshape(-1, 5) for f in features])
    feat_mean = all_f.mean(0)
    feat_std  = all_f.std(0) + 1e-8
    features  = [(f - feat_mean) / feat_std for f in features]

    # ── Split at trial level ──
    N = len(features)
    rng = np.random.RandomState(args.seed)

    if args.mode == 'sub_dep':
        idx = rng.permutation(N)
        n_tr = int(0.70 * N); n_va = int(0.15 * N)
        tr_i = idx[:n_tr]; va_i = idx[n_tr:n_tr+n_va]; te_i = idx[n_tr+n_va:]
    else:
        subjs = sorted(set(subj_ids)); rng.shuffle(subjs)
        n_te = max(1, int(0.20 * len(subjs)))
        n_va = max(1, int(0.15 * len(subjs)))
        te_set = set(subjs[:n_te]); va_set = set(subjs[n_te:n_te+n_va])
        tr_set = set(subjs[n_te+n_va:])
        tr_i = [i for i in range(N) if subj_ids[i] in tr_set]
        va_i = [i for i in range(N) if subj_ids[i] in va_set]
        te_i = [i for i in range(N) if subj_ids[i] in te_set]

    print(f"\n  Trial split → train={len(tr_i)}, val={len(va_i)}, test={len(te_i)}")

    # ── Window ──
    tr_w, tr_y = create_windows([features[i] for i in tr_i],
                                 [labels[i] for i in tr_i], args.seq_window, args.stride)
    va_w, va_y = create_windows([features[i] for i in va_i],
                                 [labels[i] for i in va_i], args.seq_window, args.seq_window)
    te_w, te_y = create_windows([features[i] for i in te_i],
                                 [labels[i] for i in te_i], args.seq_window, args.seq_window)

    print(f"  Windowed   → train={len(tr_w)}, val={len(va_w)}, test={len(te_w)}")
    from collections import Counter
    print(f"  Train labels: {dict(sorted(Counter(tr_y).items()))}")

    # ── DataLoaders ──
    tr_dl = DataLoader(DeLDSWindowDataset(tr_w, tr_y, augment=True),
                       args.batch_size, shuffle=True, drop_last=True,
                       num_workers=0, pin_memory=True)
    va_dl = DataLoader(DeLDSWindowDataset(va_w, va_y), args.batch_size,
                       num_workers=0, pin_memory=True)
    te_dl = DataLoader(DeLDSWindowDataset(te_w, te_y), args.batch_size,
                       num_workers=0, pin_memory=True)

    # ── Model ──
    model = LSSTGMamba(
        n_channels=4, n_bands=5,
        d_latent=args.d_latent, d_graph=args.d_graph,
        d_mamba=args.d_mamba, d_state=args.d_state,
        n_global_layers=args.n_global_layers,
        num_classes=n_classes, dropout=args.dropout,
    ).to(device)

    # Adjacency prior for Muse 2 electrode pairs
    # TP9↔TP10 (temporal pair), AF7↔AF8 (frontal pair)
    adj = torch.zeros(4, 4)
    for i in range(4): adj[i, i] = 1.0
    adj[0, 3] = adj[3, 0] = 0.8   # TP9-TP10 (temporal pair)
    adj[1, 2] = adj[2, 1] = 0.8   # AF7-AF8 (frontal pair)
    adj[0, 1] = adj[1, 0] = 0.5   # TP9-AF7 (left hemisphere)
    adj[2, 3] = adj[3, 2] = 0.5   # AF8-TP10 (right hemisphere)
    model.init_adjacency(adj)

    print(f"\n  Params      : {count_parameters(model):,}")
    print(f"  Batches/ep  : {len(tr_dl)}")

    # ── Loss & Optimizer ──
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    best_f1, best_state, patience_ctr = 0.0, None, 0
    epoch_times = []

    print(f"\n{'='*60}")
    print(f"Training ({args.epochs} epochs)")
    print(f"{'='*60}\n")

    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        beta = args.beta_max * min(1.0, epoch / max(args.warmup_kl, 1))

        ep_cls = ep_rec = ep_kl = 0.0
        tr_correct = tr_total = 0

        for bx, by in tr_dl:
            bx = bx.to(device); by = by.long().to(device)
            if random.random() < 0.5:
                bx, y_a, y_b, lam = mixup_data(bx, by)
                logits, l_rec, l_kl = model(bx, return_losses=True)
                l_cls = lam * criterion(logits, y_a) + (1-lam) * criterion(logits, y_b)
                tr_correct += (logits.argmax(1) == y_a).sum().item(); tr_total += len(by)
            else:
                logits, l_rec, l_kl = model(bx, return_losses=True)
                l_cls = criterion(logits, by)
                tr_correct += (logits.argmax(1) == by).sum().item(); tr_total += len(by)

            loss = l_cls + args.alpha * l_rec + beta * l_kl
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_cls += l_cls.item(); ep_rec += l_rec.item(); ep_kl += l_kl.item()

        scheduler.step()
        n_b = len(tr_dl)
        tr_acc = tr_correct / max(tr_total, 1)
        ep_time = time.time() - t0
        epoch_times.append(ep_time)

        _, vl_cls, _, _, v_acc, v_f1, _, _ = evaluate(
            model, va_dl, device, criterion, args.alpha, beta
        )

        star = " ★" if v_f1 > best_f1 else ""
        print(f"  Ep {epoch:3d}/{args.epochs} | "
              f"Tr Cls:{ep_cls/n_b:.3f} Rec:{ep_rec/n_b:.3f} KL:{ep_kl/n_b:.4f} Acc:{tr_acc:.3f} | "
              f"Va Cls:{vl_cls:.3f} Acc:{v_acc:.3f} F1:{v_f1:.3f} | "
              f"β={beta:.3f} {ep_time:.1f}s{star}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= args.patience:
                print(f"\n  Early stopping at epoch {epoch}")
                break

    # ── Test ──
    if best_state: model.load_state_dict(best_state)
    model = model.to(device)
    _, _, _, _, te_acc, te_f1, te_preds, te_lbls = evaluate(
        model, te_dl, device, criterion, args.alpha, beta
    )

    print(f"\n{'='*60}")
    print(f"RESULTS — LS-STG-Mamba Emognition (Direct, {args.mode})")
    print(f"{'='*60}")
    print(f"  Channels    : {CHANNELS}")
    print(f"  Best Val F1 : {best_f1:.4f}")
    print(f"  Test Acc    : {te_acc:.4f}")
    print(f"  Test F1     : {te_f1:.4f}")
    print(f"  Avg ep time : {np.mean(epoch_times):.1f}s")
    print(f"  Total time  : {sum(epoch_times)/60:.1f} min")
    print_report(te_lbls, te_preds, class_names, title="Direct Training")

    adj_np = model.get_adjacency()
    print(f"\n  Learned Adjacency (Muse 2 channels):")
    print(f"  {'':>8}" + "".join(f"{c:>8}" for c in CHANNELS))
    for i, c in enumerate(CHANNELS):
        print(f"  {c:>8}" + "".join(f"{adj_np[i,j]:>8.3f}" for j in range(4)))

    # ── Save ──
    ckpt_dir = os.path.join(_HERE, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt = os.path.join(ckpt_dir, 'ls_stg_mamba_emognition_direct.pt')
    torch.save({
        'model': model.state_dict(),
        'model_cfg': {
            'n_channels': 4, 'n_bands': 5,
            'd_latent': args.d_latent, 'd_graph': args.d_graph,
            'd_mamba': args.d_mamba, 'd_state': args.d_state,
            'n_global_layers': args.n_global_layers,
            'num_classes': n_classes, 'dropout': args.dropout,
        },
        'channels': CHANNELS, 'class_names': class_names,
        'val_f1': best_f1, 'test_acc': te_acc, 'test_f1': te_f1,
        'feat_mean': feat_mean.tolist(), 'feat_std': feat_std.tolist(),
        'adjacency': adj_np.tolist(), 'mode': args.mode,
    }, ckpt)
    print(f"\n  Checkpoint: {ckpt}\n")


if __name__ == '__main__':
    main()
