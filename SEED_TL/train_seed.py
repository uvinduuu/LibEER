"""
SEED-IV 4-Channel Training Script
Trains DGCNN on SEED-IV using only 4 channels (TP7, F7, F8, TP8).

Uses LibEER's own preprocessing pipeline (DE-LDS features) and training loop.

Usage:
    Subject-dependent:
        python train_seed.py --dataset_path /path/to/SEED_IV --mode sub_dep --batch_size 32 --lr 0.0015 --epochs 150

    Subject-independent:
        python train_seed.py --dataset_path /path/to/SEED_IV --mode sub_indep --batch_size 32 --lr 0.0015 --epochs 150
"""

import sys
import os
import argparse
import random

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn

# --- Import LibEER modules directly via importlib to avoid __init__.py ---
# (models/__init__.py and Trainer/__init__.py import torch_geometric-dependent modules)
import importlib.util

_libeer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'LibEER')

def _import_from_file(module_name, file_path):
    """Import a module from a specific file path, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    # Ensure parent packages are in sys.modules so relative imports work
    parts = module_name.rsplit('.', 1)
    if len(parts) > 1:
        import types
        parent_name = parts[0]
        if parent_name not in sys.modules:
            parent_mod = types.ModuleType(parent_name)
            parent_mod.__path__ = [os.path.dirname(file_path)]
            sys.modules[parent_name] = parent_mod
    sys.modules[module_name] = mod
    spec.loader.exec_module(mod)
    return mod

# Add LibEER to path for nested imports within LibEER modules
sys.path.insert(0, _libeer_dir)

# Import DGCNN directly
_dgcnn_mod = _import_from_file('models.DGCNN', os.path.join(_libeer_dir, 'models', 'DGCNN.py'))
DGCNN = _dgcnn_mod.DGCNN

# Import preprocess and split (these are safe via sys.path since data_utils/__init__.py is clean)
from data_utils.preprocess import preprocess, label_process
from data_utils.split import merge_to_part, index_to_data, get_split_index

# Import Trainer.training directly
_trainer_mod = _import_from_file('Trainer.training', os.path.join(_libeer_dir, 'Trainer', 'training.py'))
train = _trainer_mod.train

from utils.utils import setup_seed

# Local imports
sys.path.insert(0, os.path.dirname(__file__))
from seed_loader import read_seedIV_raw_4ch, NUM_CHANNELS


class SeedIVSetting:
    """Mimics LibEER's Setting object for SEED-IV 4-channel experiments."""

    def __init__(self, args):
        self.dataset = "seediv_raw"
        self.dataset_path = args.dataset_path
        self.pass_band = [-1, -1]  # No bandpass (matching repo default for seediv_raw)
        self.extract_bands = None  # Default: [[0.5,4],[4,8],[8,14],[14,30],[30,50]]
        self.time_window = args.time_window
        self.overlap = args.overlap
        self.sample_length = args.sample_length
        self.stride = args.stride
        self.only_seg = False  # We want full feature extraction (DE-LDS)
        self.feature_type = args.feature_type
        self.eog_clean = False  # Skip EOG cleaning for 4 channels (no EOG channels)
        self.bounds = None
        self.onehot = args.onehot
        self.label_used = None  # SEED-IV uses discrete labels
        self.seed = args.seed

        # Experiment mode
        if args.mode == "sub_dep":
            self.experiment_mode = "subject-dependent"
            self.cross_trail = 'true'  # Split across trials (prevents clip leakage)
        elif args.mode == "sub_indep":
            self.experiment_mode = "subject-independent"
            self.cross_trail = 'true'
        else:
            raise ValueError(f"Unknown mode: {args.mode}")

        # Split settings
        self.split_type = "train-val-test"
        self.test_size = args.test_size
        self.val_size = args.val_size
        self.fold_num = 5
        self.fold_shuffle = 'false'

        # Session selection
        self.sessions = args.sessions
        self.normalize = False

        # Rounds
        self.pr = None  # Primary rounds (subjects)
        self.sr = None  # Secondary rounds


def get_data_4ch(setting):
    """
    Load SEED-IV data with 4 channels and apply LibEER preprocessing.
    Returns: (data, label, channels, feature_dim, num_classes)
    """
    print(f"\n{'='*60}")
    print(f"Loading SEED-IV with 4 channels")
    print(f"  Path: {setting.dataset_path}")
    print(f"  Feature type: {setting.feature_type}")
    print(f"  Time window: {setting.time_window}s")
    print(f"  Mode: {setting.experiment_mode}")
    print(f"{'='*60}\n")

    # Load raw data with 4 channels
    data, baseline, label, sample_rate, channels = read_seedIV_raw_4ch(setting.dataset_path)

    # Apply LibEER's preprocessing pipeline (bandpass → DE extraction → LDS → segment)
    all_data, feature_dim = preprocess(
        data=data, baseline=baseline, sample_rate=sample_rate,
        pass_band=setting.pass_band, extract_bands=setting.extract_bands,
        sample_length=setting.sample_length, stride=setting.stride,
        time_window=setting.time_window, overlap=setting.overlap,
        only_seg=setting.only_seg,
        feature_type=setting.feature_type,
        eog_clean=setting.eog_clean
    )

    # Process labels (expand to per-sample)
    all_data, all_label, num_classes = label_process(
        data=all_data, label=label, bounds=setting.bounds,
        onehot=setting.onehot, label_used=setting.label_used
    )

    print(f"  Channels: {channels}")
    print(f"  Feature dim: {feature_dim}")
    print(f"  Num classes: {num_classes}")

    return all_data, all_label, channels, feature_dim, num_classes


def main(args):
    # Setup
    setup_seed(args.seed)
    setting = SeedIVSetting(args)

    # Load and preprocess data
    data, label, channels, feature_dim, num_classes = get_data_4ch(setting)

    # Restructure by experiment mode
    data, label = merge_to_part(data, label, setting)

    print(f"\nAfter merge_to_part: {len(data)} groups")
    for i, d in enumerate(data):
        if isinstance(d, list):
            print(f"  Group {i}: {len(d)} items")

    device = torch.device(args.device)
    best_metrics = []
    subjects_metrics = [[] for _ in range(len(data))]

    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)

        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(
                zip(tts['train'], tts['test'], tts['val']), 1):
            setup_seed(args.seed)

            if val_indexes[0] == -1:
                print(f"\n--- Round {rridx}.{ridx}: train:{len(train_indexes)}, test:{len(test_indexes)} ---")
            else:
                print(f"\n--- Round {rridx}.{ridx}: train:{len(train_indexes)}, "
                      f"val:{len(val_indexes)}, test:{len(test_indexes)} ---")

            test_sub_label = None

            # For subject-independent: track which subject each test sample belongs to
            if setting.experiment_mode == "subject-independent":
                train_data, train_label, val_data, val_label, test_data, test_label = \
                    index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, True)
                test_sub_num = len(test_data)
                test_sub_label = []
                for i in range(test_sub_num):
                    test_sub_count = len(test_data[i])
                    test_sub_label.extend([i + 1 for _ in range(test_sub_count)])
                test_sub_label = np.array(test_sub_label)

            # Split train/val/test data
            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes, False)

            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label

            print(f"  Train: {train_data.shape}, Val: {val_data.shape}, Test: {test_data.shape}")

            # Create DGCNN model for 4 channels
            # Pass layers explicitly (DGCNN only has defaults for 62/32 electrodes)
            model = DGCNN(channels, feature_dim, num_classes, k=2, relu_is=1, layers=[64], dropout_rate=0.5)
            print(f"  Model: DGCNN(electrodes={channels}, features={feature_dim}, classes={num_classes}, layers=[64])")

            # Create datasets
            dataset_train = torch.utils.data.TensorDataset(
                torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(
                torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(
                torch.Tensor(test_data), torch.Tensor(test_label))

            # Optimizer and criterion
            optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4, eps=1e-4)
            criterion = nn.CrossEntropyLoss()

            # Output directory for checkpoints
            output_dir = os.path.join(
                os.path.dirname(__file__), 'checkpoints',
                f'{args.mode}_s{args.seed}_r{rridx}_{ridx}'
            )
            os.makedirs(output_dir, exist_ok=True)

            # Train using LibEER's training loop
            round_metric = train(
                model=model,
                dataset_train=dataset_train,
                dataset_val=dataset_val,
                dataset_test=dataset_test,
                device=device,
                output_dir=output_dir,
                metrics=args.metrics,
                metric_choose=args.metric_choose,
                optimizer=optimizer,
                batch_size=args.batch_size,
                epochs=args.epochs,
                criterion=criterion
            )

            best_metrics.append(round_metric)
            if setting.experiment_mode == "subject-dependent":
                subjects_metrics[rridx - 1].append(round_metric)

    # Log results
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    if setting.experiment_mode == "subject-dependent":
        # Print per-subject results
        for i, sub_metrics in enumerate(subjects_metrics):
            if sub_metrics:
                for m in args.metrics:
                    values = [sm[m] for sm in sub_metrics]
                    print(f"  Subject {i + 1}: {m} = {np.mean(values):.4f} ± {np.std(values):.4f}")

    # Print overall results
    for m in args.metrics:
        values = [bm[m] for bm in best_metrics]
        print(f"\n  Overall {m}: {np.mean(values):.4f} ± {np.std(values):.4f}")

    print(f"\nCheckpoints saved to: {os.path.join(os.path.dirname(__file__), 'checkpoints')}")


def parse_args():
    parser = argparse.ArgumentParser(description="SEED-IV 4-Channel DGCNN Training")

    # Dataset
    parser.add_argument('--dataset_path', type=str, required=True,
                        help='Path to SEED-IV root (containing eeg_raw_data/)')
    parser.add_argument('--mode', type=str, default='sub_dep',
                        choices=['sub_dep', 'sub_indep'],
                        help='Experiment mode: sub_dep or sub_indep')
    parser.add_argument('--sessions', nargs='+', type=int, default=None,
                        help='Which sessions to use (1-3). Default: all')

    # Features
    parser.add_argument('--feature_type', type=str, default='de_lds',
                        choices=['de', 'de_lds', 'psd', 'psd_lds'],
                        help='Feature type (default: de_lds)')
    parser.add_argument('--time_window', type=int, default=1,
                        help='Time window in seconds for DE extraction (default: 1)')
    parser.add_argument('--overlap', type=float, default=0,
                        help='Overlap for DE windows (default: 0)')
    parser.add_argument('--sample_length', type=int, default=1,
                        help='Number of consecutive DE windows per sample (default: 1)')
    parser.add_argument('--stride', type=int, default=1,
                        help='Stride for segment_data (default: 1)')

    # Training
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--epochs', type=int, default=150,
                        help='Number of epochs (default: 150)')
    parser.add_argument('--lr', type=float, default=0.0015,
                        help='Learning rate (default: 0.0015)')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed (default: 2024)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device (default: cuda if available)')
    parser.add_argument('--onehot', action='store_true', default=True,
                        help='Use one-hot encoding for labels (default: True)')
    parser.add_argument('--no_onehot', action='store_false', dest='onehot',
                        help='Disable one-hot encoding')

    # Metrics
    parser.add_argument('--metrics', nargs='+', default=['acc', 'macro-f1'],
                        help='Metrics to track (default: acc macro-f1)')
    parser.add_argument('--metric_choose', type=str, default='macro-f1',
                        help='Metric for model selection (default: macro-f1)')

    # Split
    parser.add_argument('--test_size', type=float, default=0.2,
                        help='Test split ratio (default: 0.2)')
    parser.add_argument('--val_size', type=float, default=0.2,
                        help='Validation split ratio (default: 0.2)')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    main(args)
