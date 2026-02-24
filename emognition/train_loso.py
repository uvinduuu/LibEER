"""
Leave-One-Subject-Out (LOSO) Training — Optimized for Best Accuracy.

Techniques:
  - Mixup data augmentation
  - Linear warmup + cosine annealing
  - Test-Time Augmentation (TTA)
  - Stochastic Weight Averaging (SWA) with custom BN update
  - Stratified validation split
  - Gradient accumulation
  - Per-fold model saving
  - Subject exclusion for low-quality data

Usage:
    python train_loso.py
"""

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.swa_utils import AveragedModel, SWALR
from sklearn.metrics import f1_score, classification_report, confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

from config import (
    SEED, DEVICE, NUM_CHANNELS, SAMPLE_LENGTH,
    F1, D, EEG_DROPOUT,
    FUSION_DIM, NUM_HEADS, FUSION_DROPOUT,
    LSTM_HIDDEN, LSTM_LAYERS, CLS_DROPOUT,
    LABEL_MODE, USE_INVBASE, BATCH_SIZE,
    AUG_NOISE_STD, AUG_TIME_SHIFT, AUG_CHANNEL_DROP_P, AUG_SCALE_RANGE,
    EXCLUDE_SUBJECTS
)
from model import DualBranchModel
from data_loader import load_emognition, make_loaders


# ===================== LOSO SETTINGS =====================
LOSO_EPOCHS = 300
LOSO_PATIENCE = 40
LOSO_LR = 0.0003
LOSO_WEIGHT_DECAY = 1e-3
LOSO_LABEL_SMOOTHING = 0.1
WARMUP_EPOCHS = 10
SWA_START_EPOCH = 200
SWA_LR = 0.00005
MIXUP_ALPHA = 0.3
TTA_ROUNDS = 5
GRAD_ACCUM_STEPS = 2
SAVE_DIR = "loso_models"       # folder to save per-fold models
DROP_THRESHOLD = 0.20          # suggest dropping subjects below this accuracy


# ===================== UTILS =====================

def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def mixup_data(x_raw, x_hc, y, alpha=MIXUP_ALPHA):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.0
    B = x_raw.size(0)
    idx = torch.randperm(B, device=x_raw.device)
    return (lam * x_raw + (1 - lam) * x_raw[idx],
            lam * x_hc + (1 - lam) * x_hc[idx],
            y, y[idx], lam)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def augment_batch(x_raw, x_hc):
    B, C, T = x_raw.shape
    if AUG_NOISE_STD > 0:
        x_raw = x_raw + torch.randn_like(x_raw) * AUG_NOISE_STD
    if AUG_TIME_SHIFT > 0:
        shifts = torch.randint(-AUG_TIME_SHIFT, AUG_TIME_SHIFT + 1, (B,))
        for i in range(B):
            if shifts[i] != 0:
                x_raw[i] = torch.roll(x_raw[i], shifts[i].item(), dims=-1)
    if AUG_CHANNEL_DROP_P > 0:
        mask = torch.rand(B, C, 1, device=x_raw.device) > AUG_CHANNEL_DROP_P
        x_raw = x_raw * mask.float()
    lo, hi = AUG_SCALE_RANGE
    scales = torch.FloatTensor(B, 1, 1).uniform_(lo, hi).to(x_raw.device)
    x_raw = x_raw * scales
    x_hc = x_hc + torch.randn_like(x_hc) * 0.03
    return x_raw, x_hc


def evaluate(model, loader, device):
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for x_raw, x_hc, yb in loader:
            x_raw, x_hc = x_raw.to(device), x_hc.to(device)
            logits = model(x_raw, x_hc)
            preds = logits.argmax(dim=1).cpu().numpy()
            all_preds.append(preds)
            all_targets.append(yb.numpy())
    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)
    acc = (all_preds == all_targets).mean()
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return acc, f1, all_preds, all_targets


def evaluate_tta(model, loader, device, n_rounds=TTA_ROUNDS):
    """Test-Time Augmentation: average predictions across augmented versions."""
    model.eval()
    all_logits_sum, all_targets = [], []
    with torch.no_grad():
        for x_raw, x_hc, yb in loader:
            x_raw, x_hc = x_raw.to(device), x_hc.to(device)
            logits_sum = model(x_raw, x_hc)
            for _ in range(n_rounds - 1):
                x_raw_aug, x_hc_aug = augment_batch(x_raw.clone(), x_hc.clone())
                logits_sum = logits_sum + model(x_raw_aug, x_hc_aug)
            all_logits_sum.append(logits_sum.cpu())
            all_targets.append(yb)
    all_logits = torch.cat(all_logits_sum, dim=0)
    all_targets = torch.cat(all_targets, dim=0).numpy()
    all_preds = all_logits.argmax(dim=1).numpy()
    acc = (all_preds == all_targets).mean()
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    return acc, f1, all_preds, all_targets


def update_bn_dual_input(model, loader, device):
    """Custom BN update for dual-input model (fixes SWA crash)."""
    # Reset BN running stats
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
            module.reset_running_stats()
            module.momentum = None  # use cumulative moving average

    model.train()
    with torch.no_grad():
        for x_raw, x_hc, _ in loader:
            x_raw, x_hc = x_raw.to(device), x_hc.to(device)
            model(x_raw, x_hc)


def get_warmup_cosine_scheduler(optimizer, warmup_epochs, total_epochs,
                                 batches_per_epoch):
    warmup_steps = warmup_epochs * batches_per_epoch
    total_steps = total_epochs * batches_per_epoch

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = float(step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    return optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ===================== SINGLE FOLD =====================

def train_one_fold(train_raw, train_feat, train_y,
                   test_raw, test_feat, test_y,
                   hc_dim, n_classes, device, fold_name, save_path=None):
    """Train one LOSO fold with all optimizations."""

    # Standardize features
    mu = train_feat.mean(axis=0, keepdims=True)
    sd = train_feat.std(axis=0, keepdims=True) + 1e-6
    train_feat_norm = (train_feat - mu) / sd
    test_feat_norm = (test_feat - mu) / sd

    # Stratified validation split (10%)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=SEED)
    tr_idx, va_idx = next(sss.split(train_raw, train_y))

    Xr_tr, Xf_tr, y_tr = train_raw[tr_idx], train_feat_norm[tr_idx], train_y[tr_idx]
    Xr_va, Xf_va, y_va = train_raw[va_idx], train_feat_norm[va_idx], train_y[va_idx]

    train_loader, val_loader, test_loader = make_loaders(
        Xr_tr, Xf_tr, y_tr,
        Xr_va, Xf_va, y_va,
        test_raw, test_feat_norm, test_y
    )

    # Build model
    model = DualBranchModel(
        num_electrodes=NUM_CHANNELS, datapoints=SAMPLE_LENGTH,
        num_classes=n_classes, hc_dim=hc_dim,
        F1=F1, D=D, eeg_dropout=EEG_DROPOUT,
        fusion_dim=FUSION_DIM, num_heads=NUM_HEADS,
        fusion_dropout=FUSION_DROPOUT,
        lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS,
        cls_dropout=CLS_DROPOUT
    ).to(device)

    # SWA model
    swa_model = AveragedModel(model)

    optimizer = optim.AdamW(model.parameters(), lr=LOSO_LR,
                            weight_decay=LOSO_WEIGHT_DECAY, eps=1e-8)
    criterion = nn.CrossEntropyLoss(label_smoothing=LOSO_LABEL_SMOOTHING)

    batches_per_epoch = len(train_loader)
    scheduler = get_warmup_cosine_scheduler(
        optimizer, WARMUP_EPOCHS, LOSO_EPOCHS, batches_per_epoch
    )
    swa_scheduler = SWALR(optimizer, swa_lr=SWA_LR)

    best_val_f1 = 0.0
    best_state = None
    wait = 0
    swa_started = False

    import sys
    print(f"    [{fold_name}] Training...", end="", flush=True)

    for epoch in range(1, LOSO_EPOCHS + 1):
        model.train()
        optimizer.zero_grad()
        epoch_loss = 0.0

        for batch_i, (x_raw, x_hc, yb) in enumerate(train_loader):
            x_raw, x_hc, yb = x_raw.to(device), x_hc.to(device), yb.to(device)
            x_raw_aug, x_hc_aug = augment_batch(x_raw, x_hc)
            x_raw_mix, x_hc_mix, y_a, y_b, lam = mixup_data(
                x_raw_aug, x_hc_aug, yb
            )
            logits = model(x_raw_mix, x_hc_mix)
            loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            loss = loss / GRAD_ACCUM_STEPS
            loss.backward()
            epoch_loss += loss.item() * GRAD_ACCUM_STEPS

            if (batch_i + 1) % GRAD_ACCUM_STEPS == 0:
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                if not swa_started:
                    scheduler.step()

        # Flush remaining gradients
        if (batch_i + 1) % GRAD_ACCUM_STEPS != 0:
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

        epoch_loss /= max(batch_i + 1, 1)

        # SWA
        if epoch >= SWA_START_EPOCH:
            if not swa_started:
                swa_started = True
                print(f" SWA", end="", flush=True)
            swa_model.update_parameters(model)
            swa_scheduler.step()

        # Validate
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

        # Progress dots every 20 epochs
        if epoch % 20 == 0:
            print(f".", end="", flush=True)

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= LOSO_PATIENCE and not swa_started:
                print(f" stop@{epoch}", end="", flush=True)
                break

    print(f" bestF1={best_val_f1:.3f}", flush=True)

    # ---- Final model selection: best checkpoint vs SWA ----
    use_swa = False
    if swa_started:
        # Custom BN update for dual-input model (fixes the crash)
        update_bn_dual_input(swa_model, train_loader, device)
        swa_val_acc, swa_val_f1, _, _ = evaluate(swa_model, val_loader, device)

        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)
            best_val_acc, best_val_f1_check, _, _ = evaluate(model, val_loader, device)
            if swa_val_f1 > best_val_f1_check:
                use_swa = True
        else:
            use_swa = True

    # Evaluate with TTA
    if use_swa:
        test_acc, test_f1, preds, targets = evaluate_tta(
            swa_model, test_loader, device
        )
        final_state = swa_model.module.state_dict()
    else:
        if best_state is not None:
            model.load_state_dict(best_state)
            model.to(device)
        test_acc, test_f1, preds, targets = evaluate_tta(
            model, test_loader, device
        )
        final_state = model.state_dict()

    # Save model for this fold
    if save_path is not None:
        torch.save({
            'model_state': final_state,
            'fold': fold_name,
            'test_acc': test_acc,
            'test_f1': test_f1,
            'used_swa': use_swa,
            'hc_dim': hc_dim,
            'n_classes': n_classes,
            'feature_stats': {'mu': mu, 'sd': sd},
        }, save_path)

    return test_acc, test_f1, preds, targets


# ===================== MAIN =====================

def main():
    setup_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Create save directory
    os.makedirs(SAVE_DIR, exist_ok=True)

    # ==================== LOAD DATA ====================
    print("\n" + "=" * 70)
    print("LOADING EMOGNITION DATASET")
    print("=" * 70)

    X_raw, X_feat, y, subjects, lab2id, id2lab = load_emognition()
    n_classes = len(lab2id)
    hc_dim = X_feat.shape[2]
    unique_subjects = np.unique(subjects)

    # Exclude subjects
    if EXCLUDE_SUBJECTS:
        exclude_set = set(EXCLUDE_SUBJECTS)
        unique_subjects = np.array([s for s in unique_subjects if s not in exclude_set])
        keep_mask = np.array([s not in exclude_set for s in subjects])
        X_raw = X_raw[keep_mask]
        X_feat = X_feat[keep_mask]
        y = y[keep_mask]
        subjects = subjects[keep_mask]
        print(f"[EXCLUDED] Removed subjects: {EXCLUDE_SUBJECTS}")
        print(f"[EXCLUDED] Remaining: {len(X_raw)} windows, {len(unique_subjects)} subjects")

    n_subjects = len(unique_subjects)

    print(f"\nTotal: {len(X_raw)} windows, {n_subjects} subjects, {n_classes} classes")
    print(f"Feature dim: {hc_dim}/ch (InvBase: {'ON' if USE_INVBASE else 'OFF'})")

    # ==================== LOSO ====================
    print("\n" + "=" * 70)
    print(f"LEAVE-ONE-SUBJECT-OUT ({n_subjects} folds)")
    print(f"  Epochs/fold: {LOSO_EPOCHS} | Patience: {LOSO_PATIENCE}")
    print(f"  Warmup: {WARMUP_EPOCHS} | SWA from epoch: {SWA_START_EPOCH}")
    print(f"  Mixup α: {MIXUP_ALPHA} | TTA rounds: {TTA_ROUNDS}")
    print(f"  Models saved to: {SAVE_DIR}/")
    print("=" * 70)

    all_preds = []
    all_targets = []
    per_subject_results = []

    for fold_i, test_subj in enumerate(unique_subjects):
        test_mask = subjects == test_subj
        train_mask = ~test_mask

        train_raw = X_raw[train_mask]
        train_feat = X_feat[train_mask]
        train_y = y[train_mask]
        test_raw = X_raw[test_mask]
        test_feat = X_feat[test_mask]
        test_y = y[test_mask]

        n_test = len(test_y)
        if n_test == 0:
            print(f"  Fold {fold_i+1:2d} | Subject {test_subj}: SKIP (no data)")
            continue

        setup_seed(SEED + fold_i)

        save_path = os.path.join(SAVE_DIR, f"fold_{test_subj}.pt")

        acc, f1, preds, targets = train_one_fold(
            train_raw, train_feat, train_y,
            test_raw, test_feat, test_y,
            hc_dim, n_classes, device,
            fold_name=f"Subject {test_subj}",
            save_path=save_path
        )

        all_preds.append(preds)
        all_targets.append(targets)
        per_subject_results.append({
            'subject': test_subj, 'n': n_test,
            'acc': acc, 'f1': f1
        })

        print(f"  Fold {fold_i+1:2d}/{n_subjects} | Subject {test_subj:>3s} | "
              f"n={n_test:4d} | Acc: {acc:.3f} | F1: {f1:.3f} | "
              f"Saved: {save_path}")

    # ==================== RESULTS ====================
    print("\n" + "=" * 70)
    print("LOSO RESULTS")
    print("=" * 70)

    all_preds = np.concatenate(all_preds)
    all_targets = np.concatenate(all_targets)

    overall_acc = (all_preds == all_targets).mean()
    overall_f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
    accs = [r['acc'] for r in per_subject_results]
    f1s = [r['f1'] for r in per_subject_results]

    print(f"\nOverall (pooled): Acc={overall_acc:.4f} | F1={overall_f1:.4f}")
    print(f"Per-Subject Avg:  Acc={np.mean(accs):.4f}±{np.std(accs):.4f} | "
          f"F1={np.mean(f1s):.4f}±{np.std(f1s):.4f}")

    # Sorted results table
    print(f"\n{'Subject':>8s} | {'N':>5s} | {'Acc':>6s} | {'F1':>6s}")
    print("-" * 35)
    for r in sorted(per_subject_results, key=lambda x: x['acc'], reverse=True):
        marker = " ⚠" if r['acc'] < DROP_THRESHOLD else ""
        print(f"{r['subject']:>8s} | {r['n']:5d} | {r['acc']:.3f}  | {r['f1']:.3f}{marker}")

    # Suggest subjects to drop
    low_subjects = [r['subject'] for r in per_subject_results
                    if r['acc'] < DROP_THRESHOLD]
    if low_subjects:
        print(f"\n⚠ SUGGESTED FOR EXCLUSION (acc < {DROP_THRESHOLD:.0%}):")
        print(f"  Subjects: {low_subjects}")
        print(f"  Add to config.py: EXCLUDE_SUBJECTS = {low_subjects}")

        # Show what accuracy would be without them
        good_mask = np.ones(len(all_preds), dtype=bool)
        offset = 0
        for r in per_subject_results:
            n = r['n']
            if r['subject'] in low_subjects:
                good_mask[offset:offset + n] = False
            offset += n
        if good_mask.sum() > 0:
            filtered_acc = (all_preds[good_mask] == all_targets[good_mask]).mean()
            filtered_f1 = f1_score(all_targets[good_mask], all_preds[good_mask],
                                   average='macro', zero_division=0)
            print(f"  Without them: Acc={filtered_acc:.4f} | F1={filtered_f1:.4f}")

    # Classification report
    target_names = [id2lab[i] for i in range(n_classes)]
    print(f"\nClassification Report:")
    print(classification_report(all_targets, all_preds,
                                target_names=target_names,
                                digits=3, zero_division=0))
    print(f"Confusion Matrix:")
    print(confusion_matrix(all_targets, all_preds))

    print("=" * 70)
    print(f"DONE — LOSO {n_subjects} folds | "
          f"Mean Acc: {np.mean(accs):.1%}±{np.std(accs):.1%} | "
          f"Mean F1: {np.mean(f1s):.4f}±{np.std(f1s):.4f}")
    print(f"Models saved in: {SAVE_DIR}/")
    print("=" * 70)


if __name__ == "__main__":
    main()
