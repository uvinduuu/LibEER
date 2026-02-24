"""
Fine-tune SEED-IV pretrained DGCNN on Emognition dataset.

Pipeline:
  1. Load pretrained DGCNN checkpoint (from train_seed.py)
  2. Load Emognition raw EEG → resample 256→200Hz → extract DE-LDS features
  3. Map Emognition labels → SEED-IV label ordering
  4. Two-stage fine-tuning:
     Stage 1: Freeze graph conv layers, train FC head only (warmup)
     Stage 2: Unfreeze all, train with low LR (full fine-tune)
  5. LOSO or subject-dependent evaluation

Usage:
    python finetune_emognition.py --checkpoint ./checkpoints/sub_dep_s2024_r1_1 \
        --emognition_path /path/to/emognition --mode sub_dep
"""

import sys
import os
import argparse
import json
import glob

import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn
from scipy.signal import resample

# --- Import LibEER modules directly via importlib to avoid __init__.py ---
import importlib.util

_libeer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'LibEER')

def _import_from_file(module_name, file_path):
    """Import a module from a specific file path, bypassing __init__.py."""
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
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

sys.path.insert(0, _libeer_dir)

# Import DGCNN directly (bypass models/__init__.py)
_dgcnn_mod = _import_from_file('models.DGCNN', os.path.join(_libeer_dir, 'models', 'DGCNN.py'))
DGCNN = _dgcnn_mod.DGCNN

from data_utils.preprocess import de_extraction, lds

# Import Trainer.training directly (bypass Trainer/__init__.py)
_trainer_mod = _import_from_file('Trainer.training', os.path.join(_libeer_dir, 'Trainer', 'training.py'))
libeer_train = _trainer_mod.train

from utils.utils import setup_seed

# Add emognition to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'emognition'))


# ============= EMOGNITION LABEL MAPPING =============
# Emognition labels (alphabetically sorted): ENTHUSIASM=0, FEAR=1, NEUTRAL=2, SADNESS=3
# SEED-IV labels: neutral=0, sad=1, fear=2, happy=3
# Mapping: Emognition → SEED-IV ordering
EMOGNITION_TO_SEEDIV = {
    0: 3,  # ENTHUSIASM → happy
    1: 2,  # FEAR → fear
    2: 0,  # NEUTRAL → neutral
    3: 1,  # SADNESS → sad
}
SEEDIV_TO_EMOGNITION = {v: k for k, v in EMOGNITION_TO_SEEDIV.items()}


def load_emognition_data(emognition_path, target_fs=200, time_window=1,
                          feature_type='de_lds', overlap=0):
    """
    Load Emognition dataset, resample to SEED-IV rate, extract DE-LDS features.
    
    Returns:
        subjects_data: dict mapping subject_id → {
            'features': np.array (samples, channels=4, bands=5),
            'labels': np.array (samples,) in SEED-IV label space
        }
    """
    from config import CHANNELS, FS, LABEL_MODE, DATA_DIR

    # Find all subject JSON files
    json_files = sorted(glob.glob(os.path.join(emognition_path, DATA_DIR, '*.json')))
    if not json_files:
        # Try direct path
        json_files = sorted(glob.glob(os.path.join(emognition_path, '*.json')))

    if not json_files:
        raise FileNotFoundError(f"No JSON files found in {emognition_path}")

    print(f"Found {len(json_files)} subject files")

    extract_bands = [[0.5, 4], [4, 8], [8, 14], [14, 30], [30, 50]]
    isLds = feature_type.endswith('_lds')
    source_fs = FS  # Emognition sampling rate (256 Hz)

    subjects_data = {}
    label_map = {'ENTHUSIASM': 0, 'FEAR': 1, 'NEUTRAL': 2, 'SADNESS': 3}

    for json_file in json_files:
        subject_id = os.path.splitext(os.path.basename(json_file))[0]
        print(f"  Processing {subject_id}...")

        with open(json_file, 'r') as f:
            subject_json = json.load(f)

        all_features = []
        all_labels = []

        for entry in subject_json:
            # Get label
            emotion = entry.get('emotion', entry.get('label', None))
            if emotion not in label_map:
                continue

            emognition_label = label_map[emotion]
            seediv_label = EMOGNITION_TO_SEEDIV[emognition_label]

            # Get raw EEG (4 channels)
            raw_eeg = np.array(entry['eeg'])  # (channels, samples)
            if raw_eeg.shape[0] != 4:
                # Try transpose
                if raw_eeg.shape[1] == 4:
                    raw_eeg = raw_eeg.T
                else:
                    print(f"    Skipping entry: unexpected shape {raw_eeg.shape}")
                    continue

            # Resample 256 → 200 Hz
            num_samples_new = int(raw_eeg.shape[1] * target_fs / source_fs)
            resampled = np.zeros((4, num_samples_new))
            for ch in range(4):
                resampled[ch] = resample(raw_eeg[ch], num_samples_new)

            # Extract DE features: (time_windows, channels, bands)
            de_features = de_extraction(resampled, target_fs, extract_bands, time_window, overlap)

            if isLds and de_features.shape[0] > 1:
                de_features = lds(de_features)

            # Each time window becomes a sample
            num_windows = de_features.shape[0]
            all_features.append(de_features)
            all_labels.extend([seediv_label] * num_windows)

        if all_features:
            subjects_data[subject_id] = {
                'features': np.concatenate(all_features, axis=0),  # (total_windows, 4, 5)
                'labels': np.array(all_labels, dtype=np.int64)
            }
            print(f"    → {subjects_data[subject_id]['features'].shape[0]} samples, "
                  f"labels: {np.unique(all_labels, return_counts=True)}")

    return subjects_data


def freeze_graph_layers(model):
    """Freeze graph convolution layers (adjacency + graph conv weights)."""
    frozen = 0
    for name, param in model.named_parameters():
        if 'adj' in name or 'gc' in name or 'conv' in name.lower():
            param.requires_grad = False
            frozen += 1
    print(f"  Frozen {frozen} parameters (graph conv layers)")


def unfreeze_all(model):
    """Unfreeze all parameters."""
    for param in model.parameters():
        param.requires_grad = True
    print("  Unfrozen all parameters")


def finetune(args):
    setup_seed(args.seed)
    device = torch.device(args.device)

    print(f"\n{'='*60}")
    print(f"FINE-TUNING: SEED-IV → Emognition")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Emognition path: {args.emognition_path}")
    print(f"  Mode: {args.mode}")
    print(f"{'='*60}\n")

    # ===== Load Emognition data =====
    print("Loading Emognition data...")
    subjects_data = load_emognition_data(
        args.emognition_path,
        target_fs=200,
        time_window=args.time_window,
        feature_type=args.feature_type
    )

    if not subjects_data:
        print("ERROR: No data loaded from Emognition!")
        return

    subject_ids = sorted(subjects_data.keys())
    print(f"\nLoaded {len(subject_ids)} subjects: {subject_ids}")

    # ===== Prepare metrics =====
    all_results = []

    if args.mode == 'sub_dep':
        # Subject-dependent: train/val/test split per subject
        for sub_id in subject_ids:
            print(f"\n--- Subject: {sub_id} ---")
            features = subjects_data[sub_id]['features']
            labels = subjects_data[sub_id]['labels']

            # Shuffle and split
            indices = np.arange(len(features))
            np.random.shuffle(indices)
            n_test = int(len(indices) * args.test_size)
            n_val = int(len(indices) * args.val_size)

            test_idx = indices[:n_test]
            val_idx = indices[n_test:n_test + n_val]
            train_idx = indices[n_test + n_val:]

            result = _finetune_one_split(
                args, device, features, labels,
                train_idx, val_idx, test_idx, sub_id
            )
            all_results.append(result)

    elif args.mode == 'sub_indep':
        # LOSO: leave one subject out
        for test_sub in subject_ids:
            print(f"\n--- LOSO: Test subject = {test_sub} ---")
            train_subs = [s for s in subject_ids if s != test_sub]

            # Merge train subjects
            train_features = np.concatenate([subjects_data[s]['features'] for s in train_subs])
            train_labels = np.concatenate([subjects_data[s]['labels'] for s in train_subs])

            test_features = subjects_data[test_sub]['features']
            test_labels = subjects_data[test_sub]['labels']

            # Split some training data for validation
            indices = np.arange(len(train_features))
            np.random.shuffle(indices)
            n_val = int(len(indices) * args.val_size)
            val_idx = indices[:n_val]
            train_idx = indices[n_val:]

            all_features = np.concatenate([train_features, test_features])
            all_labels = np.concatenate([train_labels, test_labels])

            # Build index arrays
            final_train_idx = train_idx
            final_val_idx = val_idx
            final_test_idx = np.arange(len(train_features), len(all_features))

            result = _finetune_one_split(
                args, device, all_features, all_labels,
                final_train_idx, final_val_idx, final_test_idx, test_sub
            )
            all_results.append(result)

    # ===== Print final results =====
    print(f"\n{'='*60}")
    print("FINAL FINE-TUNING RESULTS")
    print(f"{'='*60}")
    for m in args.metrics:
        values = [r[m] for r in all_results if m in r]
        if values:
            print(f"  {m}: {np.mean(values):.4f} ± {np.std(values):.4f}")


def _finetune_one_split(args, device, features, labels, train_idx, val_idx, test_idx, split_name):
    """Fine-tune one train/val/test split."""

    # Convert labels to one-hot if needed
    num_classes = 4
    if args.onehot:
        oh_labels = np.zeros((len(labels), num_classes), dtype=np.float32)
        for i, l in enumerate(labels):
            oh_labels[i, l] = 1.0
        labels_tensor = torch.Tensor(oh_labels)
    else:
        labels_tensor = torch.LongTensor(labels)

    features_tensor = torch.Tensor(features)

    # Build datasets
    train_data = features_tensor[train_idx]
    train_labels = labels_tensor[train_idx]
    val_data = features_tensor[val_idx]
    val_labels = labels_tensor[val_idx]
    test_data = features_tensor[test_idx]
    test_labels = labels_tensor[test_idx]

    # Load pretrained model
    channels = 4
    feature_dim = features.shape[-1]  # Should be 5 (DE bands)
    model = DGCNN(channels, feature_dim, num_classes, k=2, relu_is=1, layers=[64], dropout_rate=0.5)

    checkpoint_path = os.path.join(args.checkpoint, f'checkpoint-best{args.metric_choose}')
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'], strict=False)
        print(f"  Loaded pretrained weights from {checkpoint_path}")
    else:
        print(f"  WARNING: No checkpoint found at {checkpoint_path}, training from scratch!")

    # ===== Stage 1: Warmup - freeze graph layers =====
    if args.stage1_epochs > 0:
        print(f"\n  Stage 1: Warmup ({args.stage1_epochs} epochs, LR={args.stage1_lr})")
        freeze_graph_layers(model)

        dataset_train = torch.utils.data.TensorDataset(train_data, train_labels)
        dataset_val = torch.utils.data.TensorDataset(val_data, val_labels)
        dataset_test = torch.utils.data.TensorDataset(test_data, test_labels)

        optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
                               lr=args.stage1_lr, weight_decay=1e-4)
        criterion = nn.CrossEntropyLoss()

        output_dir = os.path.join(os.path.dirname(__file__), 'finetune_checkpoints',
                                   f'{split_name}_stage1')
        os.makedirs(output_dir, exist_ok=True)

        libeer_train(
            model=model, dataset_train=dataset_train,
            dataset_val=dataset_val, dataset_test=dataset_test,
            device=device, output_dir=output_dir,
            metrics=args.metrics, metric_choose=args.metric_choose,
            optimizer=optimizer, batch_size=args.batch_size,
            epochs=args.stage1_epochs, criterion=criterion
        )

    # ===== Stage 2: Full fine-tuning =====
    print(f"\n  Stage 2: Full fine-tune ({args.stage2_epochs} epochs, LR={args.stage2_lr})")
    unfreeze_all(model)

    dataset_train = torch.utils.data.TensorDataset(train_data, train_labels)
    dataset_val = torch.utils.data.TensorDataset(val_data, val_labels)
    dataset_test = torch.utils.data.TensorDataset(test_data, test_labels)

    optimizer = optim.Adam(model.parameters(), lr=args.stage2_lr, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss()

    output_dir = os.path.join(os.path.dirname(__file__), 'finetune_checkpoints',
                               f'{split_name}_stage2')
    os.makedirs(output_dir, exist_ok=True)

    result = libeer_train(
        model=model, dataset_train=dataset_train,
        dataset_val=dataset_val, dataset_test=dataset_test,
        device=device, output_dir=output_dir,
        metrics=args.metrics, metric_choose=args.metric_choose,
        optimizer=optimizer, batch_size=args.batch_size,
        epochs=args.stage2_epochs, criterion=criterion
    )

    return result


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune SEED-IV pretrained DGCNN on Emognition")

    # Paths
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to SEED-IV pretrained checkpoint directory')
    parser.add_argument('--emognition_path', type=str, required=True,
                        help='Path to Emognition dataset')
    parser.add_argument('--mode', type=str, default='sub_dep',
                        choices=['sub_dep', 'sub_indep'],
                        help='Experiment mode (default: sub_dep)')

    # Features
    parser.add_argument('--feature_type', type=str, default='de_lds',
                        help='Feature type (default: de_lds)')
    parser.add_argument('--time_window', type=int, default=1,
                        help='DE time window in seconds (default: 1)')

    # Fine-tuning
    parser.add_argument('--stage1_epochs', type=int, default=20,
                        help='Stage 1 warmup epochs (default: 20)')
    parser.add_argument('--stage1_lr', type=float, default=0.001,
                        help='Stage 1 learning rate (default: 0.001)')
    parser.add_argument('--stage2_epochs', type=int, default=100,
                        help='Stage 2 full fine-tune epochs (default: 100)')
    parser.add_argument('--stage2_lr', type=float, default=0.0001,
                        help='Stage 2 learning rate (default: 0.0001)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--seed', type=int, default=2024,
                        help='Random seed (default: 2024)')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--onehot', action='store_true', default=True)
    parser.add_argument('--no_onehot', action='store_false', dest='onehot')

    # Split
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--val_size', type=float, default=0.2)

    # Metrics
    parser.add_argument('--metrics', nargs='+', default=['acc', 'macro-f1'])
    parser.add_argument('--metric_choose', type=str, default='macro-f1')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    finetune(args)
