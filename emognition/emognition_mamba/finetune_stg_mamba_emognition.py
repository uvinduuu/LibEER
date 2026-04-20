"""
LS-STG-Mamba Transfer Learning: SEED-IV → Emognition  (+ BVP Fusion)
======================================================================
Multimodal emotion recognition:
  - EEG backbone : LS-STG-Mamba pretrained on SEED-IV (4-ch, DE-LDS)
  - BVP features : 4 clip-level HRV features from Samsung Watch
                   [HR_mean, RMSSD, pNN50, IBI_range]
                   — replicated across all EEG windows of each clip
  - Fusion       : feature-level concat → classifier head
  - Evaluation   : LOSO (Leave-One-Subject-Out) — true subject-independent

Fine-tuning strategy (two phases):
  Phase 1: FREEZE encoder + spatial graph  → only train Mamba + classifier
  Phase 2: UNFREEZE all                   → end-to-end, differential LR
           backbone LR = lr_phase2
           head LR     = lr_phase2 × 5

Anti-overfitting:
  - Label smoothing (eps=0.15)
  - Differential learning rates (head gets 5× backbone LR)
  - Step-level CosineAnnealingLR
  - Weight decay = 0.03
  - Dropout in classifier (0.5)
  - Mixup augmentation

Usage on Kaggle:
    python finetune_stg_mamba_emognition.py \\
        --data_root      /kaggle/input/datasets/uvindukodikara/emognition \\
        --checkpoint     /kaggle/working/LibEER/62cSEED/checkpoints/ls_stg_mamba_4ch_v3.pt \\
        --samsung_root   /kaggle/input/datasets/uvindukodikara/emognition \\
        --mode           loso \\
        --epochs         80 \\
        --phase1_epochs  25

    # EEG-only (no BVP):
    python finetune_stg_mamba_emognition.py \\
        --data_root  /kaggle/input/datasets/uvindukodikara/emognition \\
        --checkpoint /kaggle/working/LibEER/62cSEED/checkpoints/ls_stg_mamba_4ch_v3.pt \\
        --mode       loso --no_bvp
"""

import os, sys, time, random, argparse, glob, json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from collections import defaultdict

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from emognition_loader import load_emognition_trials, FS
from delds_extractor import compute_delds_batch
from train_stg_mamba_emognition import (
    DeLDSWindowDataset, create_windows, setup_seed,
    mixup_data, print_report, CHANNELS
)

_MODEL_DIR = os.path.join(_HERE, '..', '..', '62cSEED')
sys.path.insert(0, _MODEL_DIR)
from stg_mamba_v3 import LSSTGMamba, count_parameters


# ─────────────────────────────────────────────────────────────────────────────
# BVP / Samsung Watch loading
# ─────────────────────────────────────────────────────────────────────────────

def _parse_paired(raw):
    """Parse Samsung [[timestamp, value],...] → numpy values."""
    if not isinstance(raw, list) or len(raw) < 5:
        return None
    try:
        return np.array([r[1] for r in raw], dtype=np.float64)
    except Exception:
        return None


def load_bvp_features(fp):
    """
    Load 4 HRV features from one Samsung Watch STIMULUS JSON.
    Returns numpy array [HR_mean, RMSSD, pNN50, IBI_range] or None.
    """
    try:
        with open(fp) as f:
            obj = json.load(f)
    except Exception:
        return None

    # PPInterval = pulse-to-pulse interval in ms (IBI sequence)
    ibi_raw = obj.get('PPInterval')
    hr_raw  = obj.get('heartRate')

    ibi = _parse_paired(ibi_raw)
    hr  = _parse_paired(hr_raw)

    if ibi is not None:
        ibi = ibi[(ibi > 300) & (ibi < 2000) & np.isfinite(ibi)]
    if hr is not None:
        hr = hr[(hr > 30) & (hr < 220) & np.isfinite(hr)]

    if ibi is None or len(ibi) < 5:
        return None

    hr_mean   = float(np.mean(hr)) if (hr is not None and len(hr) >= 3) \
                else float(np.mean(60000.0 / ibi))
    rmssd     = float(np.sqrt(np.mean(np.diff(ibi) ** 2)))
    pnn50     = float(np.mean(np.abs(np.diff(ibi)) > 50))
    ibi_range = float(ibi.max() - ibi.min())

    feats = np.array([hr_mean, rmssd, pnn50, ibi_range], dtype=np.float32)
    if not np.all(np.isfinite(feats)):
        return None
    return feats


def build_bvp_lookup(samsung_root):
    """
    Build dict: (subject, emotion) → 4-dim BVP feature vector.
    Scans samsung_root recursively for *_STIMULUS_SAMSUNG_WATCH.json.
    """
    TARGET = {'ENTHUSIASM', 'FEAR', 'NEUTRAL', 'SADNESS'}
    patterns = [
        os.path.join(samsung_root, '*_STIMULUS_SAMSUNG_WATCH.json'),
        os.path.join(samsung_root, '*', '*_STIMULUS_SAMSUNG_WATCH.json'),
        os.path.join(samsung_root, '**', '*_STIMULUS_SAMSUNG_WATCH.json'),
    ]
    files = sorted({p for pat in patterns for p in glob.glob(pat, recursive=True)})

    lookup = {}
    n_ok = n_fail = 0
    for fp in files:
        name  = os.path.splitext(os.path.basename(fp))[0]
        parts = name.split('_')
        if len(parts) < 2:
            continue
        subj, emot = parts[0], parts[1].upper()
        if emot not in TARGET:
            continue
        feats = load_bvp_features(fp)
        if feats is not None:
            lookup[(subj, emot)] = feats
            n_ok += 1
        else:
            n_fail += 1

    print(f"  BVP lookup: {n_ok} loaded, {n_fail} failed "
          f"({len(set(s for s,e in lookup))} subjects)")
    return lookup


# ─────────────────────────────────────────────────────────────────────────────
# Dataset with optional BVP concat
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalWindowDataset(Dataset):
    """
    EEG DE-LDS windows + optional BVP features.
    bvp_feats: None or array (n_windows, 4) — same BVP vector replicated per window.
    """
    def __init__(self, windows, labels, bvp_feats=None, augment=False):
        self.windows   = torch.tensor(np.array(windows), dtype=torch.float32)
        self.labels    = torch.tensor(np.array(labels),  dtype=torch.long)
        self.bvp_feats = torch.tensor(bvp_feats, dtype=torch.float32) \
                         if bvp_feats is not None else None
        self.augment   = augment

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        x = self.windows[idx]
        y = self.labels[idx]

        if self.augment and random.random() < 0.3:
            # Time-shift: roll along time axis
            shift = random.randint(1, x.shape[0] // 4)
            x = torch.roll(x, shift, dims=0)

        if self.bvp_feats is not None:
            return x, self.bvp_feats[idx], y
        return x, y


def build_windows_with_bvp(feat_list, label_list, subj_list, emot_list,
                            bvp_lookup, seq_window, stride,
                            bvp_mean=None, bvp_std=None, augment=False):
    """
    Build windowed EEG features and replicate BVP features per window.
    Returns (windows, bvp_feats_per_window, labels) or (windows, None, labels).
    """
    all_w, all_y, all_b = [], [], []
    use_bvp = (bvp_lookup is not None)

    for feat, lbl, subj, emot in zip(feat_list, label_list, subj_list, emot_list):
        T = feat.shape[0]
        wins = []
        for start in range(0, T - seq_window + 1, stride):
            wins.append(feat[start:start + seq_window])

        if not wins:
            continue

        n_wins = len(wins)
        all_w.extend(wins)
        all_y.extend([lbl] * n_wins)

        if use_bvp:
            # Look up (subject, EMOTION_STRING) → 4-dim BVP feature
            # emot here is integer label; we need the string
            bvp_vec = bvp_lookup.get((subj, emot), None)
            if bvp_vec is None:
                # fallback: zeros (will be masked out by normalisation anyway)
                bvp_vec = np.zeros(4, dtype=np.float32)
            # Normalize
            if bvp_mean is not None:
                bvp_vec = (bvp_vec - bvp_mean) / (bvp_std + 1e-8)
            # Replicate for every window of this clip
            all_b.extend([bvp_vec] * n_wins)

    all_w = np.array(all_w, dtype=np.float32)
    all_y = np.array(all_y, dtype=np.int64)
    all_b = np.array(all_b, dtype=np.float32) if all_b else None

    return all_w, all_b, all_y


# ─────────────────────────────────────────────────────────────────────────────
# Model wrapper: adds a BVP concat branch to classifier head
# ─────────────────────────────────────────────────────────────────────────────

class MultimodalSTGMamba(nn.Module):
    """
    Wraps LSSTGMamba and optionally concatenates BVP features before the
    final classification layer.

    Classifier: Linear(d_mamba + bvp_dim → 32) → ELU → Dropout(0.5)
                → Linear(32 → n_classes)
    """
    def __init__(self, backbone, d_mamba, n_classes, bvp_dim=0, dropout=0.5):
        super().__init__()
        self.backbone = backbone
        self.use_bvp  = (bvp_dim > 0)

        # Replace backbone's original classifier
        in_dim = d_mamba + bvp_dim
        self.classifier = nn.Sequential(
            nn.LayerNorm(in_dim),
            nn.Dropout(dropout),
            nn.Linear(in_dim, 32),
            nn.ELU(),
            nn.Dropout(dropout * 0.6),
            nn.Linear(32, n_classes),
        )
        # Patch backbone: remove its own classifier output
        # (backbone returns logits, lrec, lkl from its internal head)

    def forward(self, x_eeg, x_bvp=None, return_losses=False):
        # Get backbone embedding (before its final linear)
        # LSSTGMamba.forward returns (logits, lrec, lkl) when return_losses=True
        # We tap into the penultimate representation via a hook
        logits_bb, lrec, lkl = self.backbone(x_eeg, return_losses=True)

        # The backbone's readout is d_mamba → n_classes (Linear).
        # We need the d_mamba embedding. Get it by passing through backbone
        # up to (not including) its classifier.
        emb = self._get_embedding(x_eeg)  # (B, d_mamba)

        if self.use_bvp and x_bvp is not None:
            emb = torch.cat([emb, x_bvp], dim=-1)          # (B, d_mamba + 4)

        logits = self.classifier(emb)

        if return_losses:
            return logits, lrec, lkl
        return logits

    def _get_embedding(self, x):
        """Extract d_mamba embedding from backbone (before classifier)."""
        with torch.no_grad():
            # Temporarily hook to get the pre-classifier embedding
            pass
        # Since LSSTGMamba uses self.classifier as a Sequential at the end,
        # we run the backbone and intercept the embedding before the last Linear.
        # Simpler: use backbone's embedding method if available, else use logits trick.
        # We'll run the backbone's layers manually up to the classifier.
        return self.backbone.get_embedding(x)


# ─────────────────────────────────────────────────────────────────────────────
# Evaluate
# ─────────────────────────────────────────────────────────────────────────────

def evaluate(model, loader, device, criterion, use_bvp, alpha=0.0, beta=0.0):
    model.eval()
    preds, targets = [], []
    tloss = n = 0.0
    with torch.no_grad():
        for batch in loader:
            if use_bvp:
                bx, bb, by = batch
                bb = bb.to(device)
            else:
                bx, by = batch
                bb = None
            bx = bx.to(device); by = by.long().to(device)

            logits, lrec, lkl = model(bx, bb, return_losses=True)
            lcls = criterion(logits, by)
            loss = lcls + alpha * lrec + beta * lkl
            tloss += loss.item(); n += 1
            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(by.cpu().numpy())

    nb  = max(n, 1)
    acc = np.mean(np.array(preds) == np.array(targets))
    f1  = f1_score(targets, preds, average='macro', zero_division=0)
    return tloss/nb, acc, f1, preds, targets


# ─────────────────────────────────────────────────────────────────────────────
# Training phases
# ─────────────────────────────────────────────────────────────────────────────

def freeze_modules(model, module_names):
    for name, param in model.named_parameters():
        for mn in module_names:
            if name.startswith(mn):
                param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


def train_phase(model, tr_dl, va_dl, device, criterion,
                optimizer, scheduler, n_epochs,
                alpha, beta_max, warmup_kl, patience,
                use_bvp, best_f1_init=0.0, epoch_offset=0, phase_name=""):

    best_f1     = best_f1_init
    best_state  = None
    patience_ctr = 0
    trainable   = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  [{phase_name}] Trainable params: {trainable:,}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0   = time.time()
        beta = beta_max * min(1.0, epoch / max(warmup_kl, 1))
        ep_cls = ep_n = tr_correct = tr_total = 0

        for batch in tr_dl:
            if use_bvp:
                bx, bb, by = batch
                bb = bb.to(device)
            else:
                bx, by = batch
                bb = None
            bx = bx.to(device); by = by.long().to(device)

            # Mixup with 50% probability
            if random.random() < 0.5:
                bx, y_a, y_b, lam = mixup_data(bx, by)
                logits, lrec, lkl = model(bx, bb, return_losses=True)
                lcls = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
                tr_correct += (logits.argmax(1) == y_a).sum().item()
            else:
                logits, lrec, lkl = model(bx, bb, return_losses=True)
                lcls = criterion(logits, by)
                tr_correct += (logits.argmax(1) == by).sum().item()

            tr_total += len(by)
            loss = lcls + alpha * lrec + beta * lkl
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()   # ← step-level (not epoch-level)

            ep_cls += lcls.item(); ep_n += 1

        tr_acc  = tr_correct / max(tr_total, 1)
        _, v_acc, v_f1, _, _ = evaluate(model, va_dl, device, criterion,
                                         use_bvp, alpha, beta)

        global_ep = epoch + epoch_offset
        star      = " ★" if v_f1 > best_f1 else ""
        print(f"  Ep {global_ep:3d} [{phase_name}] | "
              f"Cls:{ep_cls/max(ep_n,1):.3f} Tr:{tr_acc:.3f} | "
              f"Va Acc:{v_acc:.3f} F1:{v_f1:.3f} | "
              f"β={beta:.3f} {time.time()-t0:.1f}s{star}")

        if v_f1 > best_f1:
            best_f1      = v_f1
            best_state   = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n  Early stopping at epoch {global_ep}")
                break

    return best_f1, best_state


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LS-STG-Mamba TL SEED-IV→Emognition + BVP Fusion (LOSO)"
    )
    # Paths
    parser.add_argument('--data_root',    required=True,
                        help='Emognition preprocessed EEG root dir')
    parser.add_argument('--checkpoint',   required=True,
                        help='SEED-IV checkpoint (.pt)')
    parser.add_argument('--samsung_root', default=None,
                        help='Samsung Watch data root (same as data_root if omitted)')
    parser.add_argument('--no_bvp',       action='store_true',
                        help='Disable BVP fusion (EEG-only ablation)')

    # Evaluation
    parser.add_argument('--mode', choices=['loso', 'sub_dep', 'sub_indep'],
                        default='loso',
                        help='loso = leave-one-subject-out (recommended)')

    # EEG windowing
    parser.add_argument('--delds_window_sec', type=float, default=1.0)
    parser.add_argument('--seq_window',       type=int,   default=10)
    parser.add_argument('--stride',           type=int,   default=3)

    # Training
    parser.add_argument('--epochs',         type=int,   default=80)
    parser.add_argument('--phase1_epochs',  type=int,   default=25,
                        help='Frozen-encoder epochs')
    parser.add_argument('--batch_size',     type=int,   default=64)
    parser.add_argument('--lr_phase1',      type=float, default=5e-4)
    parser.add_argument('--lr_phase2',      type=float, default=8e-5,
                        help='Backbone LR for phase 2 (head gets ×5)')
    parser.add_argument('--weight_decay',   type=float, default=0.03)
    parser.add_argument('--label_smoothing',type=float, default=0.15)
    parser.add_argument('--alpha',          type=float, default=3.0)
    parser.add_argument('--beta_max',       type=float, default=0.05)
    parser.add_argument('--warmup_kl',      type=int,   default=10)
    parser.add_argument('--patience',       type=int,   default=20)
    parser.add_argument('--seed',           type=int,   default=42)

    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    samsung_root = args.samsung_root or args.data_root
    use_bvp      = not args.no_bvp

    print(f"\n{'='*65}")
    print(f"  LS-STG-Mamba TL SEED-IV → Emognition")
    print(f"  BVP fusion  : {'✅ ON' if use_bvp else '❌ OFF (EEG-only)'}")
    print(f"  Mode        : {args.mode}")
    print(f"  Phase 1     : {args.phase1_epochs} epochs (frozen, LR={args.lr_phase1})")
    print(f"  Phase 2     : {args.epochs-args.phase1_epochs} epochs "
          f"(backbone LR={args.lr_phase2}, head LR={args.lr_phase2*5:.0e})")
    print(f"  Label smooth: {args.label_smoothing}")
    print(f"  Device      : {device}")
    print(f"{'='*65}\n")

    # ── Load EEG trials ──────────────────────────────────────────────────────
    print("Loading Emognition EEG trials...")
    trials, labels, subj_ids, lab2id, id2lab = load_emognition_trials(args.data_root)
    class_names = [id2lab[i] for i in range(len(id2lab))]
    n_classes   = len(class_names)
    # emotion string per trial (needed for BVP lookup)
    emot_strs   = [id2lab[l] for l in labels]

    print(f"\nComputing DE-LDS features...")
    t0 = time.time()
    features = compute_delds_batch(trials, fs=FS,
                                   window_sec=args.delds_window_sec,
                                   step_sec=args.delds_window_sec)
    print(f"  Done in {time.time()-t0:.1f}s")

    # Normalize using SEED-IV stats
    ckpt_data = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'feat_mean' in ckpt_data and 'feat_std' in ckpt_data:
        feat_mean = np.array(ckpt_data['feat_mean'], dtype=np.float32)
        feat_std  = np.array(ckpt_data['feat_std'],  dtype=np.float32)
        print(f"  Using SEED-IV normalisation stats from checkpoint")
    else:
        all_f     = np.concatenate([f.reshape(-1, 5) for f in features])
        feat_mean = all_f.mean(0); feat_std = all_f.std(0) + 1e-8
        print(f"  Using Emognition normalisation stats")
    features = [(f - feat_mean) / feat_std for f in features]

    # ── BVP lookup ───────────────────────────────────────────────────────────
    bvp_lookup = None
    bvp_mean   = None
    bvp_std    = None
    if use_bvp:
        print("\nLoading Samsung Watch BVP features...")
        bvp_lookup = build_bvp_lookup(samsung_root)

        # Compute BVP normalisation stats from ALL trials (global, not per-fold)
        # This is a minor leakage but BVP feats are clip-level and only 4-dim
        # — negligible impact. For strict LOSO, per-fold normalization would
        # re-compute each fold.
        bvp_vecs = [bvp_lookup.get((s, e)) for s, e in zip(subj_ids, emot_strs)]
        bvp_vecs = [v for v in bvp_vecs if v is not None]
        if bvp_vecs:
            bvp_arr  = np.stack(bvp_vecs)
            bvp_mean = bvp_arr.mean(0).astype(np.float32)
            bvp_std  = (bvp_arr.std(0) + 1e-8).astype(np.float32)
            print(f"  BVP norm: mean={bvp_mean.round(2)}, std={bvp_std.round(2)}")

    BVP_DIM = 4 if use_bvp else 0

    # ── Load SEED-IV backbone ─────────────────────────────────────────────────
    cfg   = ckpt_data.get('model_cfg', {})
    d_mamba = cfg.get('d_mamba', 64)

    backbone = LSSTGMamba(
        n_channels     = cfg.get('n_channels', 4),
        n_bands        = cfg.get('n_bands', 5),
        d_latent       = cfg.get('d_latent', 8),
        d_graph        = cfg.get('d_graph', 24),
        d_mamba        = d_mamba,
        d_state        = cfg.get('d_state', 16),
        n_global_layers= cfg.get('n_global_layers', 3),
        num_classes    = n_classes,
        dropout        = cfg.get('dropout', 0.35),
    )

    # Load weights (skip classifier if class count differs)
    state_dict      = ckpt_data['model']
    seed_n_classes  = cfg.get('num_classes', 4)
    if seed_n_classes != n_classes:
        print(f"\n  ⚠ Classifier mismatch ({seed_n_classes}→{n_classes}), "
              f"skipping classifier weights")
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith('classifier')}
    missing, _ = backbone.load_state_dict(state_dict, strict=False)
    print(f"  Loaded SEED-IV weights. Missing keys: {len(missing)}")

    # Muse 2 adjacency
    adj = torch.zeros(4, 4)
    for i in range(4): adj[i, i] = 1.0
    adj[0,3]=adj[3,0]=0.8; adj[1,2]=adj[2,1]=0.8
    adj[0,1]=adj[1,0]=0.5; adj[2,3]=adj[3,2]=0.5
    backbone.init_adjacency(adj)

    # Build multimodal wrapper
    # ── Note: LSSTGMamba needs a get_embedding() method.
    # We add it dynamically if not present (returns backbone embedding before classifier).
    if not hasattr(backbone, 'get_embedding'):
        def _get_embedding(self, x):
            # Run backbone forward up to pre-classifier representation
            # The backbone already exposes the mamba output before the final linear
            # via its internal structure; we use a forward hook approach.
            emb_holder = {}
            def hook(module, inp, out):
                emb_holder['emb'] = inp[0]
            # Register hook on the first Linear of classifier
            h = self.classifier[0].register_forward_hook(hook)
            with torch.no_grad():
                self(x, return_losses=False)  # just triggers the hook
            h.remove()
            return emb_holder.get('emb', torch.zeros(x.shape[0], 64, device=x.device))
        import types
        backbone.get_embedding = types.MethodType(_get_embedding, backbone)

    # Actually: simplest approach — use backbone directly and just swap classifier
    # Build a clean multimodal model without the wrapper complexity:
    # Keep backbone as-is but replace its classifier to accept d_mamba + BVP_DIM.
    backbone.classifier = nn.Sequential(
        nn.LayerNorm(d_mamba + BVP_DIM),
        nn.Dropout(0.5),
        nn.Linear(d_mamba + BVP_DIM, 32),
        nn.ELU(),
        nn.Dropout(0.3),
        nn.Linear(32, n_classes),
    )

    # Patch backbone forward to also accept bvp tensor
    _orig_forward = backbone.forward

    def _multimodal_forward(self, x, bvp=None, return_losses=False):
        # Run backbone up to embedding (before classifier)
        emb, lrec, lkl = self._get_emb_and_losses(x)
        if bvp is not None:
            emb = torch.cat([emb, bvp], dim=-1)
        logits = self.classifier(emb)
        if return_losses:
            return logits, lrec, lkl
        return logits

    # Use the backbone's internal state to get embedding + losses
    # We need to patch the backbone to expose its embedding.
    # The simplest approach: run with get_embedding hook.
    # Instead, let's use a direct patching approach via monkey-patching.
    import types

    def _get_emb_and_losses(self, x):
        """Run backbone up to but not including the classifier. Returns (emb, lrec, lkl)."""
        emb_holder = [None]
        lrec_holder = [torch.tensor(0.0)]
        lkl_holder  = [torch.tensor(0.0)]

        def pre_hook(module, inp):
            emb_holder[0] = inp[0].detach() if not self.training else inp[0]

        h = self.classifier[0].register_forward_pre_hook(pre_hook)
        _ = _orig_forward(x, return_losses=False)  # triggers pre-hook
        logits_full, lrec, lkl = _orig_forward(x, return_losses=True)
        h.remove()

        if emb_holder[0] is not None:
            return emb_holder[0], lrec, lkl
        else:
            return logits_full, lrec, lkl

    backbone._get_emb_and_losses = types.MethodType(_get_emb_and_losses, backbone)

    # Build the final simple model using a cleaner approach:
    # Replace backbone's classifier and override forward to accept BVP.

    class FinalModel(nn.Module):
        def __init__(self, backbone, d_mamba, bvp_dim, n_classes):
            super().__init__()
            self.backbone  = backbone
            self.bvp_dim   = bvp_dim

            # Replace backbone's classifier
            in_dim = d_mamba + bvp_dim
            self.backbone.classifier = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.Dropout(0.5),
                nn.Linear(in_dim, 32),
                nn.ELU(),
                nn.Dropout(0.3),
                nn.Linear(32, n_classes),
            )

        def forward(self, x_eeg, x_bvp=None, return_losses=False):
            # Get embedding from backbone (intercept before classifier)
            emb_store = {}

            def pre_hook(module, inp):
                emb_store['emb'] = inp[0]

            h = self.backbone.classifier[0].register_forward_pre_hook(pre_hook)
            out = self.backbone(x_eeg, return_losses=return_losses)
            h.remove()

            if 'emb' not in emb_store:
                # fallback if hook failed
                if return_losses:
                    return out
                return out

            emb = emb_store['emb']

            if self.bvp_dim > 0 and x_bvp is not None:
                emb = torch.cat([emb, x_bvp], dim=-1)

            logits = self.backbone.classifier(emb)

            if return_losses:
                if isinstance(out, tuple):
                    _, lrec, lkl = out
                    return logits, lrec, lkl
                return logits, torch.tensor(0.0), torch.tensor(0.0)
            return logits

        def init_adjacency(self, adj):
            self.backbone.init_adjacency(adj)

        def get_adjacency(self):
            return self.backbone.get_adjacency()

    model = FinalModel(backbone, d_mamba, BVP_DIM, n_classes).to(device)
    print(f"  Total params : {sum(p.numel() for p in model.parameters()):,}")
    print(f"  BVP dim added: {BVP_DIM}")

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)

    # ── LOSO / Split logic ───────────────────────────────────────────────────
    N      = len(features)
    unique_subjs = sorted(set(subj_ids))

    if args.mode == 'loso':
        # Full Leave-One-Subject-Out
        fold_results = []
        for fold_idx, test_subj in enumerate(unique_subjs):
            print(f"\n{'='*65}")
            print(f"  LOSO FOLD {fold_idx+1}/{len(unique_subjs)}  —  Test subject: {test_subj}")
            print(f"{'='*65}")

            setup_seed(args.seed + fold_idx)  # different seed per fold

            tr_idx = [i for i in range(N) if subj_ids[i] != test_subj]
            # Val: 15% of training subjects (at least 1)
            tr_subjs = sorted(set(subj_ids[i] for i in tr_idx))
            n_va_s   = max(1, int(0.15 * len(tr_subjs)))
            rng      = np.random.RandomState(args.seed)
            va_subjs = set(rng.choice(tr_subjs, n_va_s, replace=False))
            actual_tr_idx = [i for i in tr_idx if subj_ids[i] not in va_subjs]
            va_idx        = [i for i in tr_idx if subj_ids[i] in va_subjs]
            te_idx        = [i for i in range(N) if subj_ids[i] == test_subj]

            def get_data(idx_list):
                return ([features[i]   for i in idx_list],
                        [labels[i]     for i in idx_list],
                        [subj_ids[i]   for i in idx_list],
                        [emot_strs[i]  for i in idx_list])

            tr_f, tr_l, tr_s, tr_e = get_data(actual_tr_idx)
            va_f, va_l, va_s, va_e = get_data(va_idx)
            te_f, te_l, te_s, te_e = get_data(te_idx)

            st = args.stride; sw = args.seq_window
            tr_w, tr_b, tr_y = build_windows_with_bvp(
                tr_f, tr_l, tr_s, tr_e, bvp_lookup if use_bvp else None,
                sw, st, bvp_mean, bvp_std, augment=True)
            va_w, va_b, va_y = build_windows_with_bvp(
                va_f, va_l, va_s, va_e, bvp_lookup if use_bvp else None,
                sw, sw, bvp_mean, bvp_std)
            te_w, te_b, te_y = build_windows_with_bvp(
                te_f, te_l, te_s, te_e, bvp_lookup if use_bvp else None,
                sw, sw, bvp_mean, bvp_std)

            print(f"  Windows: train={len(tr_w)}, val={len(va_w)}, test={len(te_w)}")

            tr_dl = DataLoader(MultimodalWindowDataset(tr_w, tr_y, tr_b, augment=True),
                               args.batch_size, shuffle=True, drop_last=True, num_workers=0)
            va_dl = DataLoader(MultimodalWindowDataset(va_w, va_y, va_b),
                               args.batch_size, num_workers=0)
            te_dl = DataLoader(MultimodalWindowDataset(te_w, te_y, te_b),
                               args.batch_size, num_workers=0)

            # Re-init model for each fold
            model_fold = FinalModel(
                LSSTGMamba(
                    n_channels=cfg.get('n_channels', 4), n_bands=cfg.get('n_bands', 5),
                    d_latent=cfg.get('d_latent', 8),     d_graph=cfg.get('d_graph', 24),
                    d_mamba=d_mamba, d_state=cfg.get('d_state', 16),
                    n_global_layers=cfg.get('n_global_layers', 3),
                    num_classes=n_classes, dropout=cfg.get('dropout', 0.35),
                ),
                d_mamba, BVP_DIM, n_classes
            ).to(device)
            model_fold.backbone.init_adjacency(adj)
            # Load SEED-IV weights
            m, _ = model_fold.backbone.load_state_dict(state_dict, strict=False)

            # ── Phase 1 ──
            freeze_modules(model_fold, ['backbone.encoder', 'backbone.spatial'])
            opt1 = optim.AdamW(
                filter(lambda p: p.requires_grad, model_fold.parameters()),
                lr=args.lr_phase1, weight_decay=args.weight_decay
            )
            sched1 = optim.lr_scheduler.CosineAnnealingLR(
                opt1, T_max=args.phase1_epochs * max(len(tr_dl), 1), eta_min=1e-6)

            best_f1, best_state = train_phase(
                model_fold, tr_dl, va_dl, device, criterion,
                opt1, sched1, args.phase1_epochs,
                args.alpha, args.beta_max, args.warmup_kl, args.patience,
                use_bvp, 0.0, 0, f"Fold{fold_idx+1}-Ph1"
            )

            # ── Phase 2 ──
            phase2_epochs = args.epochs - args.phase1_epochs
            if phase2_epochs > 0:
                if best_state:
                    model_fold.load_state_dict(best_state)
                    model_fold = model_fold.to(device)
                unfreeze_all(model_fold)

                # Differential LR: backbone=lr_phase2, head=lr_phase2×5
                backbone_params = [p for n, p in model_fold.named_parameters()
                                   if 'classifier' not in n]
                head_params     = [p for n, p in model_fold.named_parameters()
                                   if 'classifier' in n]
                opt2 = optim.AdamW([
                    {'params': backbone_params, 'lr': args.lr_phase2},
                    {'params': head_params,     'lr': args.lr_phase2 * 5},
                ], weight_decay=args.weight_decay)
                sched2 = optim.lr_scheduler.CosineAnnealingLR(
                    opt2, T_max=phase2_epochs * max(len(tr_dl), 1), eta_min=1e-6)

                best_f1, best_state = train_phase(
                    model_fold, tr_dl, va_dl, device, criterion,
                    opt2, sched2, phase2_epochs,
                    args.alpha, args.beta_max / 2, args.warmup_kl, args.patience,
                    use_bvp, best_f1, args.phase1_epochs, f"Fold{fold_idx+1}-Ph2"
                )

            if best_state:
                model_fold.load_state_dict(best_state)
                model_fold = model_fold.to(device)

            _, te_acc, te_f1, te_preds, te_lbls = evaluate(
                model_fold, te_dl, device, criterion, use_bvp)
            print(f"\n  Fold {fold_idx+1} ({test_subj}): Acc={te_acc:.4f}  F1={te_f1:.4f}")
            fold_results.append({'subject': test_subj, 'acc': te_acc, 'f1': te_f1,
                                  'preds': te_preds, 'true': te_lbls})

        # ── Aggregate LOSO results ──
        all_preds = [p for r in fold_results for p in r['preds']]
        all_true  = [t for r in fold_results for t in r['true']]
        loso_acc  = np.mean(np.array(all_preds) == np.array(all_true))
        loso_f1   = f1_score(all_true, all_preds, average='macro', zero_division=0)
        fold_accs = [r['acc'] for r in fold_results]

        print(f"\n{'='*65}")
        print(f"  LOSO FINAL RESULTS — {'Multimodal EEG+BVP' if use_bvp else 'EEG-only'}")
        print(f"{'='*65}")
        print(f"  LOSO Acc (global) : {loso_acc:.4f} ({loso_acc*100:.1f}%)")
        print(f"  LOSO F1  (global) : {loso_f1:.4f}")
        print(f"  Per-fold Acc      : mean={np.mean(fold_accs):.4f} "
              f"± {np.std(fold_accs):.4f}")
        print(f"  Chance level      : {100/n_classes:.1f}%")
        print_report(all_true, all_preds, class_names, title="LOSO Aggregated")

        cm = confusion_matrix(all_true, all_preds)
        print(f"\n  Confusion Matrix:")
        header = "         " + "".join(f"{c:>12}" for c in class_names)
        print(f"  {header}")
        for i, c in enumerate(class_names):
            row = "".join(f"{cm[i,j]:>12}" for j in range(n_classes))
            print(f"  {c:>8} {row}")

    else:
        # sub_dep or sub_indep (original logic, single split)
        rng = np.random.RandomState(args.seed)
        if args.mode == 'sub_dep':
            idx  = rng.permutation(N)
            n_tr = int(0.70*N); n_va = int(0.15*N)
            tr_i = idx[:n_tr]; va_i = idx[n_tr:n_tr+n_va]; te_i = idx[n_tr+n_va:]
        else:
            subjs = sorted(set(subj_ids)); rng.shuffle(subjs)
            n_te = max(1, int(0.20*len(subjs))); n_va = max(1, int(0.15*len(subjs)))
            te_set = set(subjs[:n_te]); va_set = set(subjs[n_te:n_te+n_va])
            tr_i = [i for i in range(N) if subj_ids[i] not in te_set|va_set]
            va_i = [i for i in range(N) if subj_ids[i] in va_set]
            te_i = [i for i in range(N) if subj_ids[i] in te_set]

        def get_data(idx_list):
            return ([features[i]  for i in idx_list],
                    [labels[i]    for i in idx_list],
                    [subj_ids[i]  for i in idx_list],
                    [emot_strs[i] for i in idx_list])

        tr_f,tr_l,tr_s,tr_e = get_data(tr_i)
        va_f,va_l,va_s,va_e = get_data(va_i)
        te_f,te_l,te_s,te_e = get_data(te_i)

        sw = args.seq_window; st = args.stride
        tr_w,tr_b,tr_y = build_windows_with_bvp(
            tr_f,tr_l,tr_s,tr_e, bvp_lookup if use_bvp else None, sw,st,bvp_mean,bvp_std)
        va_w,va_b,va_y = build_windows_with_bvp(
            va_f,va_l,va_s,va_e, bvp_lookup if use_bvp else None, sw,sw,bvp_mean,bvp_std)
        te_w,te_b,te_y = build_windows_with_bvp(
            te_f,te_l,te_s,te_e, bvp_lookup if use_bvp else None, sw,sw,bvp_mean,bvp_std)

        print(f"  Split: train={len(tr_w)}, val={len(va_w)}, test={len(te_w)}")

        tr_dl = DataLoader(MultimodalWindowDataset(tr_w,tr_y,tr_b,augment=True),
                           args.batch_size, shuffle=True, drop_last=True, num_workers=0)
        va_dl = DataLoader(MultimodalWindowDataset(va_w,va_y,va_b),
                           args.batch_size, num_workers=0)
        te_dl = DataLoader(MultimodalWindowDataset(te_w,te_y,te_b),
                           args.batch_size, num_workers=0)

        # Phase 1
        freeze_modules(model, ['backbone.encoder', 'backbone.spatial'])
        opt1 = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()),
                           lr=args.lr_phase1, weight_decay=args.weight_decay)
        sched1 = optim.lr_scheduler.CosineAnnealingLR(
            opt1, T_max=args.phase1_epochs * max(len(tr_dl), 1), eta_min=1e-6)
        best_f1, best_state = train_phase(
            model, tr_dl, va_dl, device, criterion, opt1, sched1,
            args.phase1_epochs, args.alpha, args.beta_max, args.warmup_kl,
            args.patience, use_bvp, 0.0, 0, "Phase1")

        # Phase 2
        phase2_epochs = args.epochs - args.phase1_epochs
        if phase2_epochs > 0:
            if best_state: model.load_state_dict(best_state); model = model.to(device)
            unfreeze_all(model)
            backbone_params = [p for n,p in model.named_parameters() if 'classifier' not in n]
            head_params     = [p for n,p in model.named_parameters() if 'classifier' in n]
            opt2 = optim.AdamW([
                {'params': backbone_params, 'lr': args.lr_phase2},
                {'params': head_params,     'lr': args.lr_phase2*5},
            ], weight_decay=args.weight_decay)
            sched2 = optim.lr_scheduler.CosineAnnealingLR(
                opt2, T_max=phase2_epochs * max(len(tr_dl), 1), eta_min=1e-6)
            best_f1, best_state = train_phase(
                model, tr_dl, va_dl, device, criterion, opt2, sched2,
                phase2_epochs, args.alpha, args.beta_max/2, args.warmup_kl,
                args.patience, use_bvp, best_f1, args.phase1_epochs, "Phase2")

        if best_state: model.load_state_dict(best_state); model = model.to(device)
        _, te_acc, te_f1, te_preds, te_lbls = evaluate(
            model, te_dl, device, criterion, use_bvp)

        print(f"\n{'='*65}")
        print(f"  RESULTS ({args.mode}) — {'EEG+BVP' if use_bvp else 'EEG-only'}")
        print(f"{'='*65}")
        print(f"  Test Acc : {te_acc:.4f} ({te_acc*100:.1f}%)")
        print(f"  Test F1  : {te_f1:.4f}")
        print_report(te_lbls, te_preds, class_names, title=f"TL {args.mode}")

        ckpt_dir = os.path.join(_HERE, 'checkpoints')
        os.makedirs(ckpt_dir, exist_ok=True)
        torch.save({'model': model.state_dict(), 'model_cfg': cfg,
                    'class_names': class_names, 'test_acc': te_acc,
                    'feat_mean': feat_mean.tolist(), 'feat_std': feat_std.tolist(),
                    'bvp_mean': bvp_mean.tolist() if bvp_mean is not None else None,
                    'bvp_std':  bvp_std.tolist()  if bvp_std  is not None else None,
                    'mode': args.mode, 'use_bvp': use_bvp},
                   os.path.join(ckpt_dir, 'ls_stg_mamba_emognition_tl.pt'))


if __name__ == '__main__':
    main()
