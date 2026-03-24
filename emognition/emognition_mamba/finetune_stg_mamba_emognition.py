"""
LS-STG-Mamba Transfer Learning: SEED-IV → Emognition.

Loads a pre-trained LS-STG-Mamba checkpoint from SEED-IV training and
fine-tunes it on the Emognition dataset (Muse 2, 4 channels).

Why this works:
  - Same architecture (4ch, 5 DE-LDS bands, identical model)
  - Same feature type (DE-LDS), just computed from raw Muse 2 EEG
  - SEED-IV encoder learns rich emotion-related spectral representations
  - Fine-tuning adapts to Emognition's subjects and recording setup

Fine-tuning strategy (two phases):
  Phase 1 (epochs 1-N): FREEZE encoder + spatial graph → only train Mamba + classifier
           → Avoids forgetting SEED-IV representations early on
  Phase 2 (epochs N+1-end): UNFREEZE all → end-to-end fine-tuning with small LR
           → Allows full adaptation to Emognition domain

Compare test accuracy with train_stg_mamba_emognition.py to measure TL benefit.

Usage on Kaggle:
    python finetune_stg_mamba_emognition.py \
        --data_root /kaggle/input/datasets/uvindukodikara/emognition \
        --checkpoint /kaggle/working/LibEER/62cSEED/checkpoints/ls_stg_mamba_4ch_v3.pt \
        --epochs 100 --phase1_epochs 30
"""

import os, sys, time, random, argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

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


# ─────────────────────────────────────────────────
# Evaluate (same as direct train but standalone)
# ─────────────────────────────────────────────────

def evaluate(model, loader, device, criterion, alpha=0.0, beta=0.0):
    model.eval()
    preds, targets = [], []
    tloss = tcls = n = 0.0
    with torch.no_grad():
        for bx, by in loader:
            bx = bx.to(device); by = by.long().to(device)
            logits, lrec, lkl = model(bx, return_losses=True)
            lcls = criterion(logits, by)
            loss = lcls + alpha * lrec + beta * lkl
            tloss += loss.item(); tcls += lcls.item(); n += 1
            preds.extend(logits.argmax(1).cpu().numpy())
            targets.extend(by.cpu().numpy())
    nb = max(n, 1)
    acc = np.mean(np.array(preds) == np.array(targets))
    f1  = f1_score(targets, preds, average='macro', zero_division=0)
    return tloss/nb, tcls/nb, acc, f1, preds, targets


def freeze_modules(model, module_names):
    for name, param in model.named_parameters():
        for mn in module_names:
            if name.startswith(mn):
                param.requires_grad = False


def unfreeze_all(model):
    for param in model.parameters():
        param.requires_grad = True


# ─────────────────────────────────────────────────
# Train one phase
# ─────────────────────────────────────────────────

def train_phase(model, tr_dl, va_dl, device, criterion, optimizer, scheduler,
                n_epochs, alpha, beta_max, warmup_kl, patience,
                best_f1_init=0.0, epoch_offset=0, phase_name=""):
    best_f1 = best_f1_init
    best_state = None
    patience_ctr = 0
    epoch_times = []

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\n  [{phase_name}] Trainable params: {trainable:,}")

    for epoch in range(1, n_epochs + 1):
        model.train()
        t0 = time.time()
        beta = beta_max * min(1.0, epoch / max(warmup_kl, 1))
        ep_cls = ep_rec = ep_kl = 0.0
        tr_correct = tr_total = 0

        for bx, by in tr_dl:
            bx = bx.to(device); by = by.long().to(device)
            if random.random() < 0.5:
                bx, y_a, y_b, lam = mixup_data(bx, by)
                logits, l_rec, l_kl = model(bx, return_losses=True)
                l_cls = lam * criterion(logits, y_a) + (1-lam) * criterion(logits, y_b)
                tr_correct += (logits.argmax(1) == y_a).sum().item()
            else:
                logits, l_rec, l_kl = model(bx, return_losses=True)
                l_cls = criterion(logits, by)
                tr_correct += (logits.argmax(1) == by).sum().item()
            tr_total += len(by)

            loss = l_cls + alpha * l_rec + beta * l_kl
            optimizer.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            ep_cls += l_cls.item(); ep_rec += l_rec.item(); ep_kl += l_kl.item()

        if scheduler: scheduler.step()
        n_b = len(tr_dl)
        tr_acc = tr_correct / max(tr_total, 1)
        ep_time = time.time() - t0
        epoch_times.append(ep_time)

        _, _, v_acc, v_f1, _, _ = evaluate(model, va_dl, device, criterion, alpha, beta)

        global_ep = epoch + epoch_offset
        star = " ★" if v_f1 > best_f1 else ""
        print(f"  Ep {global_ep:3d} [{phase_name}] | "
              f"Tr Cls:{ep_cls/n_b:.3f} Rec:{ep_rec/n_b:.3f} Acc:{tr_acc:.3f} | "
              f"Va Acc:{v_acc:.3f} F1:{v_f1:.3f} | "
              f"β={beta:.3f} {ep_time:.1f}s{star}")

        if v_f1 > best_f1:
            best_f1 = v_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1
            if patience_ctr >= patience:
                print(f"\n  Early stopping at epoch {global_ep}")
                break

    return best_f1, best_state, epoch_times


# ─────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="LS-STG-Mamba TL: SEED-IV → Emognition"
    )
    # Data
    parser.add_argument('--data_root', required=True)
    parser.add_argument('--checkpoint', required=True,
                        help='Path to SEED-IV trained checkpoint (.pt)')
    parser.add_argument('--mode', choices=['sub_dep', 'sub_indep'], default='sub_dep')
    parser.add_argument('--delds_window_sec', type=float, default=1.0)
    parser.add_argument('--seq_window', type=int, default=10)
    parser.add_argument('--stride', type=int, default=3)

    # Fine-tuning
    parser.add_argument('--epochs', type=int, default=100,
                        help='Total fine-tuning epochs')
    parser.add_argument('--phase1_epochs', type=int, default=30,
                        help='Epochs with frozen encoder (Phase 1)')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr_phase1', type=float, default=5e-4,
                        help='LR for Phase 1 (frozen encoder)')
    parser.add_argument('--lr_phase2', type=float, default=1e-4,
                        help='LR for Phase 2 (full fine-tune, smaller)')
    parser.add_argument('--weight_decay', type=float, default=0.02)
    parser.add_argument('--alpha', type=float, default=3.0)
    parser.add_argument('--beta_max', type=float, default=0.05)
    parser.add_argument('--warmup_kl', type=int, default=10)
    parser.add_argument('--patience', type=int, default=25)
    parser.add_argument('--seed', type=int, default=42)

    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*60}")
    print(f"LS-STG-Mamba Transfer Learning: SEED-IV → Emognition")
    print(f"  Checkpoint : {args.checkpoint}")
    print(f"  Phase 1    : {args.phase1_epochs} epochs (frozen encoder, LR={args.lr_phase1})")
    print(f"  Phase 2    : {args.epochs - args.phase1_epochs} epochs (full, LR={args.lr_phase2})")
    print(f"  Mode       : {args.mode} | Device: {device}")
    print(f"{'='*60}\n")

    # ── Load trials & compute DE-LDS ──
    print("Loading Emognition trials...")
    trials, labels, subj_ids, lab2id, id2lab = load_emognition_trials(args.data_root)
    class_names = [id2lab[i] for i in range(len(id2lab))]
    n_classes = len(class_names)

    print(f"\nComputing DE-LDS features...")
    t0 = time.time()
    features = compute_delds_batch(trials, fs=FS,
                                   window_sec=args.delds_window_sec,
                                   step_sec=args.delds_window_sec)
    print(f"  Done in {time.time()-t0:.1f}s")

    # ── Normalize using SEED-IV stats from checkpoint ──
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    if 'feat_mean' in ckpt and 'feat_std' in ckpt:
        feat_mean = np.array(ckpt['feat_mean'], dtype=np.float32)
        feat_std  = np.array(ckpt['feat_std'],  dtype=np.float32)
        print(f"  Using SEED-IV normalization stats from checkpoint")
    else:
        all_f = np.concatenate([f.reshape(-1, 5) for f in features])
        feat_mean = all_f.mean(0); feat_std = all_f.std(0) + 1e-8
        print(f"  Using Emognition-computed normalization stats")
    features = [(f - feat_mean) / feat_std for f in features]

    # ── Split ──
    N = len(features)
    rng = np.random.RandomState(args.seed)
    if args.mode == 'sub_dep':
        idx = rng.permutation(N)
        n_tr = int(0.70*N); n_va = int(0.15*N)
        tr_i = idx[:n_tr]; va_i = idx[n_tr:n_tr+n_va]; te_i = idx[n_tr+n_va:]
    else:
        subjs = sorted(set(subj_ids)); rng.shuffle(subjs)
        n_te = max(1, int(0.20*len(subjs))); n_va = max(1, int(0.15*len(subjs)))
        te_set = set(subjs[:n_te]); va_set = set(subjs[n_te:n_te+n_va])
        tr_set = set(subjs[n_te+n_va:])
        tr_i = [i for i in range(N) if subj_ids[i] in tr_set]
        va_i = [i for i in range(N) if subj_ids[i] in va_set]
        te_i = [i for i in range(N) if subj_ids[i] in te_set]

    tr_w, tr_y = create_windows([features[i] for i in tr_i],
                                 [labels[i] for i in tr_i], args.seq_window, args.stride)
    va_w, va_y = create_windows([features[i] for i in va_i],
                                 [labels[i] for i in va_i], args.seq_window, args.seq_window)
    te_w, te_y = create_windows([features[i] for i in te_i],
                                 [labels[i] for i in te_i], args.seq_window, args.seq_window)

    print(f"\n  Split: train={len(tr_w)}, val={len(va_w)}, test={len(te_w)}")

    tr_dl = DataLoader(DeLDSWindowDataset(tr_w, tr_y, augment=True),
                       args.batch_size, shuffle=True, drop_last=True, num_workers=0)
    va_dl = DataLoader(DeLDSWindowDataset(va_w, va_y), args.batch_size, num_workers=0)
    te_dl = DataLoader(DeLDSWindowDataset(te_w, te_y), args.batch_size, num_workers=0)

    # ── Load SEED-IV model ──
    cfg = ckpt.get('model_cfg', {})
    model = LSSTGMamba(
        n_channels=cfg.get('n_channels', 4),
        n_bands=cfg.get('n_bands', 5),
        d_latent=cfg.get('d_latent', 8),
        d_graph=cfg.get('d_graph', 24),
        d_mamba=cfg.get('d_mamba', 64),
        d_state=cfg.get('d_state', 16),
        n_global_layers=cfg.get('n_global_layers', 3),
        num_classes=n_classes,           # Emognition classes (may differ from SEED-IV)
        dropout=cfg.get('dropout', 0.35),
    )

    # Load weights — skip classifier if output size differs
    state_dict = ckpt['model']
    seed_n_classes = cfg.get('num_classes', 4)
    if seed_n_classes != n_classes:
        print(f"\n  ⚠ Classifier mismatch: SEED-IV={seed_n_classes} → Emognition={n_classes}")
        print(f"    Skipping classifier weights — will train from scratch")
        state_dict = {k: v for k, v in state_dict.items()
                      if not k.startswith('classifier')}
    
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print(f"\n  Loaded SEED-IV weights:")
    print(f"    Missing keys  : {len(missing)} {missing[:3] if missing else ''}")
    print(f"    Unexpected    : {len(unexpected)}")

    # Re-initialize adjacency for Muse 2 electrode topology
    adj = torch.zeros(4, 4)
    for i in range(4): adj[i, i] = 1.0
    adj[0, 3] = adj[3, 0] = 0.8   # TP9-TP10
    adj[1, 2] = adj[2, 1] = 0.8   # AF7-AF8
    adj[0, 1] = adj[1, 0] = 0.5   # left hemisphere
    adj[2, 3] = adj[3, 2] = 0.5   # right hemisphere
    model.init_adjacency(adj)
    model = model.to(device)

    print(f"  Total params : {count_parameters(model):,}")

    criterion = nn.CrossEntropyLoss()
    all_epoch_times = []

    # ── Phase 1: Freeze encoder + spatial graph, train Mamba + classifier ──
    print(f"\n{'='*60}")
    print(f"PHASE 1: Frozen encoder + graph ({args.phase1_epochs} epochs)")
    print(f"{'='*60}")
    freeze_modules(model, ['encoder', 'spatial'])
    
    opt1 = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr_phase1, weight_decay=args.weight_decay
    )
    sched1 = optim.lr_scheduler.CosineAnnealingLR(opt1, T_max=args.phase1_epochs)

    best_f1, best_state, times1 = train_phase(
        model, tr_dl, va_dl, device, criterion, opt1, sched1,
        args.phase1_epochs, args.alpha, args.beta_max, args.warmup_kl,
        args.patience, best_f1_init=0.0, epoch_offset=0, phase_name="Phase1-Frozen"
    )
    all_epoch_times.extend(times1)

    # Load best phase-1 state before phase 2
    if best_state: model.load_state_dict(best_state)
    model = model.to(device)

    # ── Phase 2: Unfreeze all, fine-tune end-to-end with smaller LR ──
    phase2_epochs = args.epochs - args.phase1_epochs
    if phase2_epochs > 0:
        print(f"\n{'='*60}")
        print(f"PHASE 2: Full fine-tuning ({phase2_epochs} epochs, LR={args.lr_phase2})")
        print(f"{'='*60}")
        unfreeze_all(model)

        opt2 = optim.AdamW(model.parameters(),
                           lr=args.lr_phase2, weight_decay=args.weight_decay)
        sched2 = optim.lr_scheduler.CosineAnnealingLR(opt2, T_max=phase2_epochs)

        best_f1, best_state, times2 = train_phase(
            model, tr_dl, va_dl, device, criterion, opt2, sched2,
            phase2_epochs, args.alpha, args.beta_max // 2, args.warmup_kl,
            args.patience, best_f1_init=best_f1,
            epoch_offset=args.phase1_epochs, phase_name="Phase2-Full"
        )
        all_epoch_times.extend(times2)

    # ── Test ──
    if best_state: model.load_state_dict(best_state)
    model = model.to(device)
    _, _, te_acc, te_f1, te_preds, te_lbls = evaluate(model, te_dl, device, criterion)

    print(f"\n{'='*60}")
    print(f"RESULTS — LS-STG-Mamba TL SEED-IV→Emognition ({args.mode})")
    print(f"{'='*60}")
    print(f"  Best Val F1 : {best_f1:.4f}")
    print(f"  Test Acc    : {te_acc:.4f}")
    print(f"  Test F1     : {te_f1:.4f}")
    print(f"  Avg ep time : {np.mean(all_epoch_times):.1f}s")
    print(f"  Total time  : {sum(all_epoch_times)/60:.1f} min")
    print_report(te_lbls, te_preds, class_names, title="TL SEED-IV→Emognition")

    adj_np = model.get_adjacency()
    print(f"\n  Learned Adjacency (Muse 2):")
    print(f"  {'':>8}" + "".join(f"{c:>8}" for c in CHANNELS))
    for i, c in enumerate(CHANNELS):
        print(f"  {c:>8}" + "".join(f"{adj_np[i,j]:>8.3f}" for j in range(4)))

    # ── Save ──
    ckpt_dir = os.path.join(_HERE, 'checkpoints')
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_out = os.path.join(ckpt_dir, 'ls_stg_mamba_emognition_tl.pt')
    torch.save({
        'model': model.state_dict(),
        'model_cfg': cfg,
        'channels': CHANNELS, 'class_names': class_names,
        'val_f1': best_f1, 'test_acc': te_acc, 'test_f1': te_f1,
        'feat_mean': feat_mean.tolist(), 'feat_std': feat_std.tolist(),
        'adjacency': adj_np.tolist(), 'mode': args.mode,
        'source': 'SEED-IV TL',
    }, ckpt_out)
    print(f"\n  Checkpoint: {ckpt_out}\n")


if __name__ == '__main__':
    main()
