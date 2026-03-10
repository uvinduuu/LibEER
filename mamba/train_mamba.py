"""
Mamba-based Full-Clip EEG Training Script for SEED-IV (4 Channels).

Trains a Mamba (SSM) classifier on entire raw EEG trial clips (~60s each)
instead of windowed DE-LDS features.

Usage:
    Subject-dependent:
        python train_mamba.py --dataset_path /path/to/SEED_IV --mode sub_dep --epochs 100

    Subject-independent (LOSO):
        python train_mamba.py --dataset_path /path/to/SEED_IV --mode sub_indep --epochs 100

    Overfit test (verify model can learn):
        python train_mamba.py --dataset_path /path/to/SEED_IV --mode sub_dep --overfit_test --epochs 50
"""

import os
import sys
import argparse
import random
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, Subset
from sklearn.metrics import confusion_matrix, classification_report, f1_score

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from mamba_model import MambaEEGClassifier
from dataset import SeedIVClipDataset, load_seediv_clips
from augmentations import get_train_augmentations, get_eval_augmentations


CLASS_NAMES = ['neutral', 'sad', 'fear', 'happy']


def setup_seed(seed):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_confusion_matrix(y_true, y_pred, num_classes=4, title=""):
    """Print formatted confusion matrix and classification report."""
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
    print(f"\n  Confusion Matrix{' (' + title + ')' if title else ''}:")
    print(f"  {'':>10}", end="")
    for name in CLASS_NAMES[:num_classes]:
        print(f"{name:>10}", end="")
    print()
    for i, name in enumerate(CLASS_NAMES[:num_classes]):
        print(f"  {name:>10}", end="")
        for j in range(num_classes):
            print(f"{cm[i][j]:>10}", end="")
        print()
    print(f"\n  Classification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES[:num_classes], digits=4))


def evaluate(model, dataloader, device, criterion):
    """Evaluate model on a dataloader. Returns loss, accuracy, macro-F1, predictions, labels."""
    model.eval()
    all_preds = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch_x, batch_y, batch_lengths in dataloader:
            batch_x = batch_x.to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.long).to(device) if not isinstance(batch_y, torch.Tensor) else batch_y.long().to(device)
            batch_lengths = batch_lengths.to(device)

            outputs = model(batch_x, lengths=batch_lengths)
            loss = criterion(outputs, batch_y)

            preds = torch.argmax(outputs, dim=1).cpu().numpy()
            labels = batch_y.cpu().numpy()

            all_preds.extend(preds)
            all_labels.extend(labels)
            total_loss += loss.item()
            n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    acc = np.mean(np.array(all_preds) == np.array(all_labels))
    macro_f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    return avg_loss, acc, macro_f1, all_preds, all_labels


def train_one_split(
    model, train_dataset, val_dataset, test_dataset,
    args, device, split_name="",
):
    """Train model on one train/val/test split. Returns best test metrics."""

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=args.lr,
        weight_decay=args.weight_decay, eps=1e-8
    )
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)

    best_val_f1 = 0.0
    best_model_state = None
    patience_counter = 0

    print(f"\n{'='*60}")
    print(f"Training: {split_name}")
    print(f"  Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    print(f"  LR: {args.lr}, Epochs: {args.epochs}, Batch: {args.batch_size}")
    print(f"{'='*60}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        epoch_correct = 0
        epoch_total = 0

        for batch_x, batch_y, batch_lengths in train_loader:
            batch_x = batch_x.to(device)
            batch_y = torch.tensor(batch_y, dtype=torch.long).to(device) if not isinstance(batch_y, torch.Tensor) else batch_y.long().to(device)
            batch_lengths = batch_lengths.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x, lengths=batch_lengths)
            loss = criterion(outputs, batch_y)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()

            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            epoch_correct += (preds == batch_y).sum().item()
            epoch_total += len(batch_y)

        scheduler.step()

        train_acc = epoch_correct / max(epoch_total, 1)
        train_loss = epoch_loss / max(len(train_loader), 1)

        # Validate
        val_loss, val_acc, val_f1, _, _ = evaluate(model, val_loader, device, criterion)

        if epoch % 10 == 0 or epoch == 1:
            print(f"  Epoch {epoch:3d}/{args.epochs} | "
                  f"Train Loss: {train_loss:.4f}, Acc: {train_acc:.4f} | "
                  f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f}, F1: {val_f1:.4f} | "
                  f"LR: {scheduler.get_last_lr()[0]:.6f}")

        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        # Early stopping
        if args.patience > 0 and patience_counter >= args.patience:
            print(f"  Early stopping at epoch {epoch} (patience={args.patience})")
            break

    # Load best model and evaluate on test set
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    model = model.to(device)

    test_loss, test_acc, test_f1, test_preds, test_labels = evaluate(
        model, test_loader, device, criterion
    )

    print(f"\n  Best Val F1: {best_val_f1:.4f}")
    print(f"  Test Acc: {test_acc:.4f}, Test F1: {test_f1:.4f}")
    print_confusion_matrix(test_labels, test_preds, title=split_name)

    # Save checkpoint
    ckpt_dir = os.path.join(os.path.dirname(__file__), 'checkpoints', split_name)
    os.makedirs(ckpt_dir, exist_ok=True)
    torch.save({
        'model': model.state_dict(),
        'val_f1': best_val_f1,
        'test_acc': test_acc,
        'test_f1': test_f1,
    }, os.path.join(ckpt_dir, 'best_model.pt'))

    return {
        'acc': test_acc,
        'macro-f1': test_f1,
        'val_f1': best_val_f1,
        'test_preds': test_preds,
        'test_labels': test_labels,
    }


def make_datasets(trials, labels, train_idx, val_idx, test_idx, fixed_length):
    """Create train/val/test SeedIVClipDatasets from index arrays."""
    train_trials = [trials[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]
    val_trials = [trials[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]
    test_trials = [trials[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    print(f"\n  Creating datasets (fixed_length={fixed_length})...")
    train_ds = SeedIVClipDataset(train_trials, train_labels, fixed_length=fixed_length,
                                  augment=True, filter_eeg=True, normalize=True)
    val_ds = SeedIVClipDataset(val_trials, val_labels, fixed_length=fixed_length,
                                augment=False, filter_eeg=True, normalize=True)
    test_ds = SeedIVClipDataset(test_trials, test_labels, fixed_length=fixed_length,
                                 augment=False, filter_eeg=True, normalize=True)

    return train_ds, val_ds, test_ds


def run_subject_dependent(trials, labels, subject_ids, args, device):
    """
    Subject-dependent: pool all trials, random split at trial level.
    Same subject CAN appear in both train and test, but same trial CANNOT.
    """
    print(f"\n{'#'*60}")
    print(f"# SUBJECT-DEPENDENT MODE")
    print(f"# {len(trials)} total trials, random 60/20/20 split")
    print(f"{'#'*60}")

    n = len(trials)
    indices = np.arange(n)
    np.random.shuffle(indices)

    n_test = int(n * args.test_size)
    n_val = int(n * args.val_size)

    test_idx = indices[:n_test]
    val_idx = indices[n_test:n_test + n_val]
    train_idx = indices[n_test + n_val:]

    print(f"  Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # Determine fixed length from all trials (95th percentile to avoid outlier padding)
    lengths = [t.shape[1] for t in trials]
    fixed_length = int(np.percentile(lengths, 95))

    train_ds, val_ds, test_ds = make_datasets(
        trials, labels, train_idx, val_idx, test_idx, fixed_length
    )

    model = MambaEEGClassifier(
        in_channels=4, num_classes=4,
        d_model=args.d_model, n_layers=args.n_layers,
        d_state=args.d_state, dropout=args.dropout
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  Model parameters: {n_params:,}")

    result = train_one_split(
        model, train_ds, val_ds, test_ds,
        args, device, split_name="sub_dep"
    )

    return [result]


def run_subject_independent(trials, labels, subject_ids, args, device):
    """
    Subject-independent (LOSO): train on 14 subjects, test on 1.
    Repeat for each subject.
    """
    unique_subjects = sorted(set(subject_ids))
    print(f"\n{'#'*60}")
    print(f"# SUBJECT-INDEPENDENT MODE (LOSO)")
    print(f"# {len(unique_subjects)} subjects, {len(trials)} total trials")
    print(f"{'#'*60}")

    # Determine fixed length from all trials (95th percentile to avoid outlier padding)
    lengths = [t.shape[1] for t in trials]
    fixed_length = int(np.percentile(lengths, 95))

    all_results = []
    all_preds = []
    all_true = []

    for test_subj in unique_subjects:
        print(f"\n--- LOSO: Test Subject {test_subj} ---")

        # Split by subject
        test_idx = [i for i, s in enumerate(subject_ids) if s == test_subj]
        train_pool = [i for i, s in enumerate(subject_ids) if s != test_subj]

        # Split train pool into train + val
        np.random.shuffle(train_pool)
        n_val = int(len(train_pool) * args.val_size)
        val_idx = train_pool[:n_val]
        train_idx = train_pool[n_val:]

        train_ds, val_ds, test_ds = make_datasets(
            trials, labels, train_idx, val_idx, test_idx, fixed_length
        )

        model = MambaEEGClassifier(
            in_channels=4, num_classes=4,
            d_model=args.d_model, n_layers=args.n_layers,
            d_state=args.d_state, dropout=args.dropout
        )

        result = train_one_split(
            model, train_ds, val_ds, test_ds,
            args, device, split_name=f"loso_subj{test_subj}"
        )

        all_results.append(result)
        all_preds.extend(result['test_preds'])
        all_true.extend(result['test_labels'])

    # Overall results
    print(f"\n{'='*60}")
    print("OVERALL LOSO RESULTS")
    print(f"{'='*60}")

    accs = [r['acc'] for r in all_results]
    f1s = [r['macro-f1'] for r in all_results]
    print(f"  Accuracy:  {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"  Macro-F1:  {np.mean(f1s):.4f} ± {np.std(f1s):.4f}")

    print_confusion_matrix(all_true, all_preds, title="Overall LOSO")

    return all_results


def run_overfit_test(trials, labels, args, device):
    """Quick overfitting test: train on a few trials to verify model can learn."""
    print(f"\n{'#'*60}")
    print(f"# OVERFIT TEST")
    print(f"# Training on 8 trials — loss should approach 0")
    print(f"{'#'*60}")

    # Take 8 trials (2 per class if possible)
    selected = []
    for cls in range(4):
        cls_idx = [i for i, l in enumerate(labels) if l == cls]
        selected.extend(cls_idx[:2])

    if len(selected) < 4:
        selected = list(range(min(8, len(trials))))

    sub_trials = [trials[i] for i in selected]
    sub_labels = [labels[i] for i in selected]
    fixed_length = max(t.shape[1] for t in sub_trials)

    # Use SAME data for train/val/test (it's an overfit test)
    ds = SeedIVClipDataset(sub_trials, sub_labels, fixed_length=fixed_length,
                            augment=False, filter_eeg=True, normalize=True)

    model = MambaEEGClassifier(
        in_channels=4, num_classes=4,
        d_model=args.d_model, n_layers=args.n_layers,
        d_state=args.d_state, dropout=0.0  # No dropout for overfit test
    )

    result = train_one_split(
        model, ds, ds, ds,
        args, device, split_name="overfit_test"
    )

    if result['acc'] > 0.9:
        print("\n  ✓ OVERFIT TEST PASSED — model can memorize data")
    else:
        print(f"\n  ✗ OVERFIT TEST FAILED — acc={result['acc']:.4f} (expected > 0.9)")

    return result


def main(args):
    setup_seed(args.seed)
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"MAMBA EEG CLASSIFIER — SEED-IV (4 Channels)")
    print(f"  Dataset: {args.dataset_path}")
    print(f"  Mode: {args.mode}")
    print(f"  Device: {device}")
    print(f"  d_model: {args.d_model}, layers: {args.n_layers}, d_state: {args.d_state}")
    print(f"{'='*60}")

    # Load all trials
    print("\nLoading SEED-IV raw data...")
    t0 = time.time()
    trials, labels, subject_ids, session_ids = load_seediv_clips(
        args.dataset_path, sessions=args.sessions
    )
    print(f"  Loaded in {time.time() - t0:.1f}s")

    # Run experiment
    if args.overfit_test:
        run_overfit_test(trials, labels, args, device)
    elif args.mode == 'sub_dep':
        results = run_subject_dependent(trials, labels, subject_ids, args, device)
        print(f"\n  Final Acc:  {results[0]['acc']:.4f}")
        print(f"  Final F1:   {results[0]['macro-f1']:.4f}")
    elif args.mode == 'sub_indep':
        run_subject_independent(trials, labels, subject_ids, args, device)

    print(f"\nCheckpoints saved to: {os.path.join(os.path.dirname(__file__), 'checkpoints')}")


def parse_args():
    parser = argparse.ArgumentParser(description="Mamba EEG Classifier for SEED-IV")

    # Dataset
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to SEED-IV root (containing eeg_raw_data/)')
    parser.add_argument('--mode', type=str, default='sub_dep',
                        choices=['sub_dep', 'sub_indep'],
                        help='Experiment mode (default: sub_dep)')
    parser.add_argument('--sessions', nargs='+', type=int, default=None,
                        help='Which sessions to use (1-3). Default: all')

    # Model architecture
    parser.add_argument('--d_model', type=int, default=128,
                        help='Model dimension (default: 128)')
    parser.add_argument('--n_layers', type=int, default=2,
                        help='Number of Mamba blocks (default: 2)')
    parser.add_argument('--d_state', type=int, default=16,
                        help='SSM state dimension (default: 16)')
    parser.add_argument('--dropout', type=float, default=0.3,
                        help='Dropout rate (default: 0.3)')

    # Training
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (default: 16)')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=5e-4,
                        help='Learning rate (default: 5e-4)')
    parser.add_argument('--weight_decay', type=float, default=0.01,
                        help='Weight decay (default: 0.01)')
    parser.add_argument('--patience', type=int, default=20,
                        help='Early stopping patience (0=disabled, default: 20)')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed (default: 2024)')
    parser.add_argument('--device', type=str,
                        default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (default: cuda if available)')

    # Split
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test split ratio (default: 0.2)')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')

    # Testing
    parser.add_argument('--overfit_test', action='store_true',
                        help='Run overfit test (train on 8 trials, verify model can learn)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
