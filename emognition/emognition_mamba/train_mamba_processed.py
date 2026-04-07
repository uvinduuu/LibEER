"""
Mamba EEG Training for Preprocessed Emognition Dataset.

Experiment modes (--mode):
  sub_dep     : Subject-dependent — pool all subjects, random trial-level split 70/15/15.
                Fast sanity check; data leaks across subjects.
  sub_split   : Subject-independent split — split SUBJECTS into train/val/test sets
                (default: 70% / 15% / 15%). Single run, no looping. ← RECOMMENDED
  sub_indep   : Leave-One-Subject-Out (LOSO) — train on N-1 subjects each fold.
                Most rigorous, but slow.

Input modes:
  (default)  : Windowed — each trial is sliced into fixed windows (--window_sec).
               Use --overlap 0.5 to get 2× more windows (50%% sliding step).
  --full_clip : Feed the ENTIRE trial as one sample. Mamba's SSM handles the full
               sequence via its conv stem (16× downsample) — but gives fewer samples.

Recommended settings for small datasets (< 200 trials):
  --window_sec 4 --overlap 0.5   → ~15× windows per trial, good data volume
  --window_sec 6 --overlap 0.5   → ~10× windows, better temporal context
  --window_sec 10                → ~6× windows (original default)

Usage:
  # Subject-split (70/15/15 subjects), 6s windows, 50%% overlap
  python train_mamba_processed.py --data_root /path --mode sub_split \\
      --window_sec 6 --overlap 0.5 --epochs 100

  # LOSO, 4s windows, 50%% overlap
  python train_mamba_processed.py --data_root /path --mode sub_indep \\
      --window_sec 4 --overlap 0.5 --epochs 60

  # Full clip, subject-split
  python train_mamba_processed.py --data_root /path --mode sub_split --full_clip --epochs 80
"""

import os
import sys
import argparse
import random
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, classification_report, confusion_matrix

# ── Local imports ──────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from emognition_processed_loader import load_emognition_processed, FS

# Import shared utilities from mamba/ folder
_mamba_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'mamba')
sys.path.insert(0, _mamba_dir)
from mamba_model import MambaEEGClassifier
from windowed_dataset import (
    WindowedEEGDataset, split_trial_into_windows,
    bandpass_filter, normalize_trial
)


# ══════════════════════════════════════════════════════════════════════════════
#  Dataset classes
# ══════════════════════════════════════════════════════════════════════════════

class FullClipEEGDataset(Dataset):
    """
    Dataset that feeds an entire EEG trial (variable length) as one sample.

    Within a batch, trials are zero-padded to the longest trial.
    Actual lengths are returned so the model can apply masked pooling.
    """

    def __init__(self, trials, labels, augment=False, sample_rate=FS):
        """
        Args:
            trials: list of (4, T_i) float32 numpy arrays (preprocessed)
            labels: list of int
            augment: whether to apply light augmentation
            sample_rate: used only for time-masking augmentation bounds
        """
        assert len(trials) == len(labels)
        self.trials = trials
        self.labels = labels
        self.augment = augment
        self.sample_rate = sample_rate

    def __len__(self):
        return len(self.trials)

    def __getitem__(self, idx):
        x = self.trials[idx].copy()   # (4, T)
        y = self.labels[idx]

        if self.augment:
            # Gaussian noise
            if random.random() < 0.5:
                x = x + np.random.normal(0, 0.02 * x.std(), x.shape).astype(np.float32)
            # Amplitude scaling
            if random.random() < 0.5:
                scale = np.random.uniform(0.85, 1.15)
                x = x * scale

        return torch.from_numpy(x).float(), y

    @staticmethod
    def collate_fn(batch):
        """
        Pad variable-length trials to the longest in the batch.
        Returns:
            x:       (B, 4, T_max)  — zero-padded
            lengths: (B,)           — original (unpadded) lengths
            y:       (B,)           — labels
        """
        xs, ys = zip(*batch)
        lengths = torch.tensor([x.shape[-1] for x in xs], dtype=torch.long)
        T_max = lengths.max().item()
        B, C = len(xs), xs[0].shape[0]
        x_padded = torch.zeros(B, C, T_max)
        for i, x in enumerate(xs):
            x_padded[i, :, :x.shape[-1]] = x
        return x_padded, lengths, torch.tensor(ys, dtype=torch.long)


# ══════════════════════════════════════════════════════════════════════════════
#  Utility functions
# ══════════════════════════════════════════════════════════════════════════════

def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def preprocess_trial(trial, sample_rate=FS):
    """Bandpass filter + z-score per channel."""
    trial = trial.astype(np.float32)
    trial = bandpass_filter(trial, lowcut=1.0, highcut=50.0, fs=sample_rate)
    trial = normalize_trial(trial)
    return trial


def evaluate_windowed(model, dataloader, device, criterion):
    """Evaluate windowed dataset. Returns loss, acc, macro-f1, preds, labels."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch_x, batch_y in dataloader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.long().to(device)
            outputs = model(batch_x)
            loss = criterion(outputs, batch_y)
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / max(n_batches, 1), acc, f1, all_preds, all_labels


def evaluate_fullclip(model, dataloader, device, criterion):
    """Evaluate full-clip dataset (receives padded batch + lengths)."""
    model.eval()
    all_preds, all_labels = [], []
    total_loss, n_batches = 0.0, 0

    with torch.no_grad():
        for batch_x, lengths, batch_y in dataloader:
            batch_x = batch_x.to(device)
            lengths = lengths.to(device)
            batch_y = batch_y.long().to(device)
            outputs = model(batch_x, lengths=lengths)
            loss = criterion(outputs, batch_y)
            all_preds.extend(torch.argmax(outputs, 1).cpu().numpy())
            all_labels.extend(batch_y.cpu().numpy())
            total_loss += loss.item()
            n_batches += 1

    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    return total_loss / max(n_batches, 1), acc, f1, all_preds, all_labels


def print_report(y_true, y_pred, class_names, title=""):
    n = len(class_names)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(n)))
    hdr = f"  Confusion Matrix{' (' + title + ')' if title else ''}:"
    print(f"\n{hdr}")
    print(f"  {'':>12}", end="")
    for nm in class_names:
        print(f"{nm:>12}", end="")
    print()
    for i, nm in enumerate(class_names):
        print(f"  {nm:>12}", end="")
        for j in range(n):
            print(f"{cm[i][j]:>12}", end="")
        print()
    print("\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names, digits=4))


# ══════════════════════════════════════════════════════════════════════════════
#  Core train + eval loop
# ══════════════════════════════════════════════════════════════════════════════

def train_one_split(
    train_ds, val_ds, test_ds,
    class_names, args, device,
    split_name="", verbose=True, full_clip=False,
):
    """
    Train and evaluate for a single data split.

    Returns dict with test acc, macro-f1, preds, labels.
    """
    # ── DataLoaders ──
    if full_clip:
        # Full-clip: use custom collate_fn to handle variable lengths
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True,
            collate_fn=FullClipEEGDataset.collate_fn, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=FullClipEEGDataset.collate_fn, num_workers=0,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False,
            collate_fn=FullClipEEGDataset.collate_fn, num_workers=0,
        )
        evaluate_fn = evaluate_fullclip
    else:
        train_loader = DataLoader(
            train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0,
        )
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        test_loader = DataLoader(
            test_ds, batch_size=args.batch_size, shuffle=False, num_workers=0,
        )
        evaluate_fn = evaluate_windowed

    # ── Model ──
    model = MambaEEGClassifier(
        in_channels=4, num_classes=len(class_names),
        d_model=args.d_model, n_layers=args.n_layers,
        d_state=args.d_state, dropout=args.dropout,
    ).to(device)

    if verbose:
        n_params = sum(p.numel() for p in model.parameters())
        mode_str = "full-clip" if full_clip else f"windowed {args.window_sec}s"
        print(f"  Model params: {n_params:,}  |  Mode: {mode_str}")
        print(f"  Train: {len(train_ds)} samples, Val: {len(val_ds)}, Test: {len(test_ds)}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay, eps=1e-8)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )

    best_val_f1 = 0.0
    best_state = None
    patience_ctr = 0

    for epoch in range(1, args.epochs + 1):
        model.train()
        ep_loss, ep_correct, ep_total = 0.0, 0, 0
        t0 = time.time()

        if full_clip:
            for bx, lengths, by in train_loader:
                bx = bx.to(device)
                lengths = lengths.to(device)
                by = by.long().to(device)
                optimizer.zero_grad()
                out = model(bx, lengths=lengths)
                loss = criterion(out, by)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                ep_loss += loss.item()
                ep_correct += (torch.argmax(out, 1) == by).sum().item()
                ep_total += len(by)
        else:
            for bx, by in train_loader:
                bx = bx.to(device)
                by = by.long().to(device)
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
        train_loss = ep_loss / max(len(train_loader), 1)
        train_acc = ep_correct / max(ep_total, 1)

        val_loss, val_acc, val_f1, _, _ = evaluate_fn(model, val_loader, device, criterion)
        elapsed = time.time() - t0

        if verbose and (epoch % 5 == 0 or epoch == 1):
            print(f"    Ep {epoch:3d}/{args.epochs} "
                  f"| Train {train_loss:.4f}/{train_acc:.4f} "
                  f"| Val {val_loss:.4f}/{val_acc:.4f}/F1:{val_f1:.4f} "
                  f"| {elapsed:.1f}s")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_ctr = 0
        else:
            patience_ctr += 1

        if args.patience > 0 and patience_ctr >= args.patience:
            if verbose:
                print(f"    Early stop at epoch {epoch}")
            break

    # ── Evaluate on test ──
    if best_state is not None:
        model.load_state_dict(best_state)
    model = model.to(device)

    _, test_acc, test_f1, test_preds, test_labels = evaluate_fn(
        model, test_loader, device, criterion
    )

    if verbose:
        print(f"\n  Best Val F1 : {best_val_f1:.4f}")
        print(f"  Test Acc    : {test_acc:.4f}")
        print(f"  Test Macro-F1: {test_f1:.4f}")
        print_report(test_labels, test_preds, class_names, title=split_name)

    # ── Save checkpoint ──
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', split_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'val_f1': best_val_f1,
        'test_acc': test_acc,
        'test_f1': test_f1,
        'class_names': class_names,
        'config': vars(args),
    }, os.path.join(ckpt_dir, 'best_model.pt'))

    return {
        'acc': test_acc,
        'macro-f1': test_f1,
        'val_f1': best_val_f1,
        'test_preds': test_preds,
        'test_labels': test_labels,
    }


# ══════════════════════════════════════════════════════════════════════════════
#  Experiment modes
# ══════════════════════════════════════════════════════════════════════════════

def split_with_overlap(trial, window_size, overlap=0.0):
    """
    Split trial (C, T) into possibly-overlapping windows.

    Args:
        overlap: fraction of window to overlap, e.g. 0.5 = 50% overlap.
                 0.0 = non-overlapping (original behaviour).
    Returns:
        List of (C, window_size) arrays.
    """
    if overlap == 0.0:
        return split_trial_into_windows(trial, window_size)

    C, T = trial.shape
    step = max(1, int(window_size * (1.0 - overlap)))
    windows = []
    start = 0
    while start + window_size <= T:
        windows.append(trial[:, start:start + window_size])
        start += step
    # Keep a final overlapping window if there's enough leftover
    if not windows and T >= window_size:
        windows.append(trial[:, T - window_size:T])
    return windows


def make_windowed_datasets(trial_indices, trials, labels, subject_ids,
                           window_size, augment, overlap=0.0):
    """Preprocess and window a set of trials (optionally with overlap)."""
    windows, win_labels = [], []
    for idx in trial_indices:
        trial = preprocess_trial(np.array(trials[idx]), sample_rate=FS)
        wins = split_with_overlap(trial, window_size, overlap=overlap)
        for w in wins:
            windows.append(w)
            win_labels.append(int(labels[idx]))
    return WindowedEEGDataset(windows, win_labels, augment=augment, sample_rate=FS)


def make_fullclip_datasets(trial_indices, trials, labels, augment):
    """Preprocess and return full-clip dataset."""
    processed = []
    clip_labels = []
    for idx in trial_indices:
        trial = preprocess_trial(np.array(trials[idx]), sample_rate=FS)
        processed.append(trial)
        clip_labels.append(int(labels[idx]))
    return FullClipEEGDataset(processed, clip_labels, augment=augment)


def run_subject_dependent(trials, labels, subject_ids, class_names, args, device):
    """Pool all subjects, split at trial level (no subject leakage)."""
    n = len(trials)
    rng = np.random.RandomState(args.seed)
    idx = rng.permutation(n)
    n_test = int(n * 0.15)
    n_val = int(n * 0.15)
    test_idx  = idx[:n_test]
    val_idx   = idx[n_test:n_test + n_val]
    train_idx = idx[n_test + n_val:]

    print(f"\n  Subject-Dependent split:")
    print(f"    Train: {len(train_idx)} trials | Val: {len(val_idx)} | Test: {len(test_idx)}")

    if args.full_clip:
        train_ds = make_fullclip_datasets(train_idx, trials, labels, augment=True)
        val_ds   = make_fullclip_datasets(val_idx,   trials, labels, augment=False)
        test_ds  = make_fullclip_datasets(test_idx,  trials, labels, augment=False)
    else:
        window_size = int(args.window_sec * FS)
        train_ds = make_windowed_datasets(train_idx, trials, labels, subject_ids,
                                          window_size, augment=True, overlap=args.overlap)
        val_ds   = make_windowed_datasets(val_idx,   trials, labels, subject_ids,
                                          window_size, augment=False, overlap=0.0)
        test_ds  = make_windowed_datasets(test_idx,  trials, labels, subject_ids,
                                          window_size, augment=False, overlap=0.0)
        print(f"    Windows — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    return train_one_split(
        train_ds, val_ds, test_ds, class_names, args, device,
        split_name="sub_dep", verbose=True, full_clip=args.full_clip,
    )


def run_subject_split(trials, labels, subject_ids, class_names, args, device):
    """
    Subject-independent split: assign SUBJECTS (not trials) to train/val/test.

    Default: 70% train subjects / 15% val subjects / 15% test subjects.
    No subject appears in more than one split → truly subject-independent.
    Single run (not LOSO) → much faster than LOSO while still being fair.
    """
    unique_subjects = sorted(set(subject_ids))
    n_subj = len(unique_subjects)

    rng = np.random.RandomState(args.seed)
    shuffled = list(unique_subjects)
    rng.shuffle(shuffled)

    n_test = max(1, int(n_subj * args.test_ratio))
    n_val  = max(1, int(n_subj * args.val_ratio))
    n_train = n_subj - n_test - n_val

    if n_train < 1:
        print("[ERROR] Not enough subjects for a 70/15/15 split. "
              "Consider reducing --test_ratio / --val_ratio.")
        sys.exit(1)

    test_subjs  = set(shuffled[:n_test])
    val_subjs   = set(shuffled[n_test:n_test + n_val])
    train_subjs = set(shuffled[n_test + n_val:])

    print(f"\n  Subject-Split ({int((1-args.test_ratio-args.val_ratio)*100)}/"
          f"{int(args.val_ratio*100)}/{int(args.test_ratio*100)}):")
    print(f"    Train ({len(train_subjs)} subj): {sorted(train_subjs)}")
    print(f"    Val   ({len(val_subjs)} subj) : {sorted(val_subjs)}")
    print(f"    Test  ({len(test_subjs)} subj) : {sorted(test_subjs)}")

    train_idx = [i for i, s in enumerate(subject_ids) if s in train_subjs]
    val_idx   = [i for i, s in enumerate(subject_ids) if s in val_subjs]
    test_idx  = [i for i, s in enumerate(subject_ids) if s in test_subjs]

    # Show label distribution per split
    for name, idxs in [("Train", train_idx), ("Val", val_idx), ("Test", test_idx)]:
        dist = Counter(labels[i] for i in idxs)
        dist_str = {class_names[k]: v for k, v in sorted(dist.items())}
        print(f"    {name:5s} label dist: {dist_str}")

    if args.full_clip:
        train_ds = make_fullclip_datasets(train_idx, trials, labels, augment=True)
        val_ds   = make_fullclip_datasets(val_idx,   trials, labels, augment=False)
        test_ds  = make_fullclip_datasets(test_idx,  trials, labels, augment=False)
    else:
        window_size = int(args.window_sec * FS)
        print(f"\n  Windowing: {args.window_sec}s ({window_size} samples), "
              f"overlap={args.overlap:.0%}")
        # Overlap only on TRAINING set — val/test use non-overlapping for clean eval
        train_ds = make_windowed_datasets(train_idx, trials, labels, subject_ids,
                                          window_size, augment=True, overlap=args.overlap)
        val_ds   = make_windowed_datasets(val_idx,   trials, labels, subject_ids,
                                          window_size, augment=False, overlap=0.0)
        test_ds  = make_windowed_datasets(test_idx,  trials, labels, subject_ids,
                                          window_size, augment=False, overlap=0.0)
        print(f"  Windows — Train: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

    return train_one_split(
        train_ds, val_ds, test_ds, class_names, args, device,
        split_name="sub_split", verbose=True, full_clip=args.full_clip,
    )


def run_loso(trials, labels, subject_ids, class_names, args, device):
    """Leave-One-Subject-Out cross-validation."""
    unique_subjects = sorted(set(subject_ids))
    print(f"\n  LOSO: {len(unique_subjects)} subjects")

    all_results = []
    all_preds, all_true = [], []

    for test_subj in unique_subjects:
        print(f"\n{'─'*50}")
        print(f"  Test subject: {test_subj}")

        test_trial_idx  = [i for i, s in enumerate(subject_ids) if s == test_subj]
        train_pool_idx  = [i for i, s in enumerate(subject_ids) if s != test_subj]

        if not test_trial_idx:
            print("  Skipping (no test trials)")
            continue

        # Carve out ~15% of remaining subjects as val (subject-level)
        rng = np.random.RandomState(args.seed)
        remaining_subjs = sorted(set(subject_ids[i] for i in train_pool_idx))
        rng.shuffle(remaining_subjs)
        n_val_subj = max(1, int(len(remaining_subjs) * 0.15))
        val_subjects = set(remaining_subjs[:n_val_subj])

        val_idx   = [i for i in train_pool_idx if subject_ids[i] in val_subjects]
        train_idx = [i for i in train_pool_idx if subject_ids[i] not in val_subjects]

        if args.full_clip:
            train_ds = make_fullclip_datasets(train_idx, trials, labels, augment=True)
            val_ds   = make_fullclip_datasets(val_idx,   trials, labels, augment=False)
            test_ds  = make_fullclip_datasets(test_trial_idx, trials, labels, augment=False)
        else:
            window_size = int(args.window_sec * FS)
            train_ds = make_windowed_datasets(train_idx, trials, labels, subject_ids,
                                              window_size, augment=True, overlap=args.overlap)
            val_ds   = make_windowed_datasets(val_idx,   trials, labels, subject_ids,
                                              window_size, augment=False, overlap=0.0)
            test_ds  = make_windowed_datasets(test_trial_idx, trials, labels, subject_ids,
                                              window_size, augment=False, overlap=0.0)
            print(f"    Train windows: {len(train_ds)} | Val: {len(val_ds)} | Test: {len(test_ds)}")

        result = train_one_split(
            train_ds, val_ds, test_ds, class_names, args, device,
            split_name=f"loso_{test_subj}", verbose=False, full_clip=args.full_clip,
        )
        print(f"  {test_subj} → Acc={result['acc']:.4f}  Macro-F1={result['macro-f1']:.4f}")

        all_results.append(result)
        all_preds.extend(result['test_preds'])
        all_true.extend(result['test_labels'])

    # ── Aggregate LOSO results ──
    accs = [r['acc'] for r in all_results]
    f1s  = [r['macro-f1'] for r in all_results]

    print(f"\n{'='*60}")
    print(f"OVERALL LOSO RESULTS ({len(all_results)} subjects)")
    print(f"{'='*60}")
    print(f"  Accuracy  : {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Macro-F1  : {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")
    print_report(all_true, all_preds, class_names, title="Overall LOSO")
    return all_results


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="Mamba EEG — Preprocessed Emognition Dataset"
    )

    # Dataset
    parser.add_argument('--data_root', type=str, required=True,
                        help='Root of Emognition Processed dataset')
    parser.add_argument('--mode', type=str, default='sub_split',
                        choices=['sub_dep', 'sub_split', 'sub_indep'],
                        help=('sub_dep    = subject-dependent (trial-level split, fast sanity check)\n'
                              'sub_split  = subject-independent split 70/15/15 (recommended)\n'
                              'sub_indep  = LOSO full cross-validation (most rigorous, slow)'))

    # Input mode
    parser.add_argument('--full_clip', action='store_true',
                        help='Feed ENTIRE trial as one sample (no windowing). '
                             'Mamba handles the full ~60s EEG clip.')
    parser.add_argument('--window_sec', type=float, default=6.0,
                        help='Window size in seconds (default: 6.0). Ignored with --full_clip.')
    parser.add_argument('--overlap', type=float, default=0.5,
                        help='Fraction of window overlap for TRAINING set (default: 0.5 = 50%%). '
                             'Val/Test always use non-overlapping windows. '
                             'Set to 0.0 to disable. Only used in windowed mode.')
    parser.add_argument('--test_ratio',  type=float, default=0.15,
                        help='Fraction of subjects for test set in sub_split mode (default: 0.15)')
    parser.add_argument('--val_ratio',   type=float, default=0.15,
                        help='Fraction of subjects for val set in sub_split mode (default: 0.15)')

    # Model hyper-params
    parser.add_argument('--d_model',  type=int,   default=128)
    parser.add_argument('--n_layers', type=int,   default=2)
    parser.add_argument('--d_state',  type=int,   default=16)
    parser.add_argument('--dropout',  type=float, default=0.3)

    # Training hyper-params
    parser.add_argument('--batch_size',    type=int,   default=16,
                        help='Batch size. Use 8-16 for full-clip mode to fit GPU memory.')
    parser.add_argument('--epochs',        type=int,   default=50)
    parser.add_argument('--lr',            type=float, default=5e-4)
    parser.add_argument('--weight_decay',  type=float, default=0.01)
    parser.add_argument('--patience',      type=int,   default=15)
    parser.add_argument('--seed',          type=int,   default=2024)
    parser.add_argument('--device',        type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu')

    args = parser.parse_args()
    setup_seed(args.seed)
    device = torch.device(args.device)

    # ── Header ──────────────────────────────────────────────────────────────
    if args.full_clip:
        input_mode = "FULL CLIP (entire trial ~60s)"
    else:
        overlap_str = f", {args.overlap:.0%} overlap on train" if args.overlap > 0 else ""
        input_mode = (f"WINDOWED {args.window_sec}s "
                      f"({int(args.window_sec * FS)} samples{overlap_str})")

    print(f"\n{'='*60}")
    print(f"MAMBA EEG — EMOGNITION PROCESSED")
    print(f"  Dataset : {args.data_root}")
    print(f"  Mode    : {args.mode.upper()}")
    print(f"  Input   : {input_mode}")
    print(f"  Device  : {device}")
    print(f"  Model   : d_model={args.d_model}, layers={args.n_layers}, "
          f"d_state={args.d_state}, dropout={args.dropout}")
    print(f"  Train   : batch={args.batch_size}, lr={args.lr}, "
          f"epochs={args.epochs}, patience={args.patience}")
    print(f"{'='*60}")

    # ── Load trials ─────────────────────────────────────────────────────────
    print(f"\nLoading Emognition Processed trials...")
    t0 = time.time()
    trials, labels, subject_ids, lab2id, id2lab = load_emognition_processed(
        args.data_root, verbose=True,
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    if not trials:
        print("\n[ERROR] No trials loaded. Check --data_root path and dataset structure.")
        sys.exit(1)

    class_names = [id2lab[i] for i in range(len(id2lab))]
    print(f"\n  Classes ({len(class_names)}): {class_names}")

    # ── Run experiment ───────────────────────────────────────────────────────
    if args.mode == 'sub_dep':
        results = run_subject_dependent(
            trials, labels, subject_ids, class_names, args, device
        )
    elif args.mode == 'sub_split':
        results = run_subject_split(
            trials, labels, subject_ids, class_names, args, device
        )
    else:  # sub_indep (LOSO)
        results = run_loso(
            trials, labels, subject_ids, class_names, args, device
        )

    ckpt_base = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'checkpoints')
    print(f"\n  Checkpoints saved under: {ckpt_base}")
    print(f"  Done!\n")


if __name__ == '__main__':
    main()
