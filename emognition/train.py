"""
Dual-Branch EEGNet-BiLSTM Training Script with Data Augmentation.

Architecture: EEGNet (features) + Handcrafted (features) → Cross-Attention → BiLSTM

Usage:
    python train.py
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import f1_score, classification_report, confusion_matrix

from config import (
    SEED, DEVICE, EPOCHS, LR, WEIGHT_DECAY, PATIENCE, LABEL_SMOOTHING,
    NUM_CHANNELS, SAMPLE_LENGTH,
    F1, D, EEG_DROPOUT,
    FUSION_DIM, NUM_HEADS, FUSION_DROPOUT,
    LSTM_HIDDEN, LSTM_LAYERS, CLS_DROPOUT,
    LABEL_MODE, USE_INVBASE,
    AUG_NOISE_STD, AUG_TIME_SHIFT, AUG_CHANNEL_DROP_P, AUG_SCALE_RANGE
)
from model import DualBranchModel
from data_loader import load_emognition, split_data, make_loaders


def setup_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def augment_batch(x_raw, x_hc):
    """
    Apply data augmentation to a training batch.

    Augmentations on raw EEG:
      - Gaussian noise injection
      - Random time shift
      - Random channel dropout
      - Random amplitude scaling

    Augmentations on handcrafted features:
      - Light Gaussian noise (to regularize)
    """
    B, C, T = x_raw.shape

    # 1) Gaussian noise
    if AUG_NOISE_STD > 0:
        noise = torch.randn_like(x_raw) * AUG_NOISE_STD
        x_raw = x_raw + noise

    # 2) Random time shift (circular)
    if AUG_TIME_SHIFT > 0:
        shifts = torch.randint(-AUG_TIME_SHIFT, AUG_TIME_SHIFT + 1, (B,))
        for i in range(B):
            if shifts[i] != 0:
                x_raw[i] = torch.roll(x_raw[i], shifts[i].item(), dims=-1)

    # 3) Random channel dropout
    if AUG_CHANNEL_DROP_P > 0:
        mask = torch.rand(B, C, 1, device=x_raw.device) > AUG_CHANNEL_DROP_P
        x_raw = x_raw * mask.float()

    # 4) Random amplitude scaling
    lo, hi = AUG_SCALE_RANGE
    scales = torch.FloatTensor(B, 1, 1).uniform_(lo, hi).to(x_raw.device)
    x_raw = x_raw * scales

    # 5) Light noise on handcrafted features
    x_hc = x_hc + torch.randn_like(x_hc) * 0.05

    return x_raw, x_hc


def evaluate(model, loader, device):
    """Evaluate model. Returns accuracy, macro-F1, preds, targets."""
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


def main():
    setup_seed(SEED)
    device = torch.device(DEVICE if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # ==================== LOAD DATA ====================
    print("\n" + "=" * 70)
    print("LOADING EMOGNITION DATASET (raw EEG + handcrafted features)")
    print("=" * 70)

    X_raw, X_feat, y, subjects, lab2id, id2lab = load_emognition()
    n_classes = len(lab2id)

    print(f"\nRaw EEG: {X_raw.shape} | HC Features: {X_feat.shape}")
    print(f"Label mode: {LABEL_MODE} ({n_classes} classes)")

    # ==================== SPLIT ====================
    print("\n" + "=" * 70)
    print("SPLITTING DATA")
    print("=" * 70)

    (Xr_tr, Xf_tr, y_tr,
     Xr_va, Xf_va, y_va,
     Xr_te, Xf_te, y_te) = split_data(X_raw, X_feat, y, subjects)

    print(f"Train: {Xr_tr.shape} + {Xf_tr.shape}")
    print(f"Val:   {Xr_va.shape} + {Xf_va.shape}")
    print(f"Test:  {Xr_te.shape} + {Xf_te.shape}")

    # ==================== DATALOADERS ====================
    train_loader, val_loader, test_loader = make_loaders(
        Xr_tr, Xf_tr, y_tr,
        Xr_va, Xf_va, y_va,
        Xr_te, Xf_te, y_te
    )

    # ==================== MODEL ====================
    print("\n" + "=" * 70)
    print("BUILDING DUAL-BRANCH EEGNet-BiLSTM MODEL")
    print("=" * 70)

    # Auto-detect handcrafted feature dim (26 base, or 36 with InvBase)
    hc_dim = Xf_tr.shape[2]
    print(f"Handcrafted feature dim: {hc_dim} per channel")

    model = DualBranchModel(
        num_electrodes=NUM_CHANNELS,
        datapoints=SAMPLE_LENGTH,
        num_classes=n_classes,
        hc_dim=hc_dim,
        F1=F1, D=D, eeg_dropout=EEG_DROPOUT,
        fusion_dim=FUSION_DIM, num_heads=NUM_HEADS,
        fusion_dropout=FUSION_DROPOUT,
        lstm_hidden=LSTM_HIDDEN, lstm_layers=LSTM_LAYERS,
        cls_dropout=CLS_DROPOUT
    ).to(device)

    eeg_p = sum(p.numel() for p in model.eegnet.parameters())
    fus_p = sum(p.numel() for p in model.fusion.parameters())
    cls_p = sum(p.numel() for p in model.classifier.parameters())
    total_p = sum(p.numel() for p in model.parameters())
    print(f"EEGNet (feature extractor):  {eeg_p:>8,} params")
    print(f"Cross-Attention (fusion):    {fus_p:>8,} params")
    print(f"BiLSTM (classifier):         {cls_p:>8,} params")
    print(f"{'Total:':>29s}  {total_p:>8,} params")

    # ==================== TRAINING SETUP ====================
    optimizer = optim.AdamW(model.parameters(), lr=LR,
                            weight_decay=WEIGHT_DECAY, eps=1e-8)
    criterion = nn.CrossEntropyLoss(label_smoothing=LABEL_SMOOTHING)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=20, T_mult=2
    )

    print("\n" + "=" * 70)
    print(f"TRAINING ({EPOCHS} epochs, patience={PATIENCE})")
    print(f"  Label smoothing: {LABEL_SMOOTHING}")
    print(f"  Data augmentation: noise={AUG_NOISE_STD}, "
          f"shift={AUG_TIME_SHIFT}, ch_drop={AUG_CHANNEL_DROP_P}")
    print("=" * 70)

    best_val_f1 = 0.0
    best_state = None
    wait = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0

        for x_raw, x_hc, yb in train_loader:
            x_raw, x_hc, yb = x_raw.to(device), x_hc.to(device), yb.to(device)

            # Apply augmentation
            x_raw_aug, x_hc_aug = augment_batch(x_raw, x_hc)

            optimizer.zero_grad()
            logits = model(x_raw_aug, x_hc_aug)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step(epoch + n_batches / len(train_loader))

            train_loss += loss.item()
            n_batches += 1

        train_loss /= max(n_batches, 1)
        val_acc, val_f1, _, _ = evaluate(model, val_loader, device)

        if epoch % 5 == 0 or epoch <= 5:
            lr_now = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch:03d} | Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_acc:.3f} | Val F1: {val_f1:.3f} | "
                  f"LR: {lr_now:.6f}")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
            if epoch % 5 == 0 or val_f1 > 0.30:
                print(f"  -> New best Val F1: {val_f1:.4f}")
        else:
            wait += 1
            if wait >= PATIENCE:
                print(f"\nEarly stopping at epoch {epoch} "
                      f"(best Val F1: {best_val_f1:.4f})")
                break

    # ==================== TEST ====================
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)

    print("\n" + "=" * 70)
    print("TEST RESULTS")
    print("=" * 70)

    test_acc, test_f1, test_preds, test_targets = evaluate(
        model, test_loader, device
    )

    print(f"\nTest Accuracy:  {test_acc:.4f}  ({test_acc * 100:.1f}%)")
    print(f"Test Macro-F1:  {test_f1:.4f}")

    target_names = [id2lab[i] for i in range(n_classes)]
    print(f"\nClassification Report:")
    print(classification_report(test_targets, test_preds,
                                target_names=target_names,
                                digits=3, zero_division=0))

    print(f"Confusion Matrix:")
    print(confusion_matrix(test_targets, test_preds))

    # ==================== SAVE ====================
    torch.save(best_state or model.state_dict(),
               "dual_branch_eegnet_bilstm.pt")
    print(f"\nModel saved to dual_branch_eegnet_bilstm.pt")
    print("=" * 70)
    print(f"DONE — Best Val F1: {best_val_f1:.4f} | "
          f"Test Acc: {test_acc:.1%} | Test F1: {test_f1:.4f}")
    print("=" * 70)


if __name__ == "__main__":
    main()
