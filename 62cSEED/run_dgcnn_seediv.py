"""
Run LibEER's DGCNN on SEED-IV with DE-LDS features.
This uses LibEER's own pipeline end-to-end.

Usage on Kaggle:
    cd /kaggle/working/LibEER/LibEER
    python ../62cSEED/run_dgcnn_seediv.py \
        --dataset_path /kaggle/input/datasets/phhasian0710/seed-iv/seed_iv \
        --epochs 150 --batch_size 32 --lr 0.0015

This is *equivalent* to the LibEER benchmark command:
    python DGCNN_train.py -metrics acc macro-f1 -metric_choose macro-f1 \
        -setting seediv_sub_dependent_train_val_test_setting \
        -dataset seediv_raw -batch_size 32 -epochs 150 \
        -time_window 1 -feature_type de_lds -seed 2024 -onehot
"""

import sys
import os

# Must run from LibEER/ directory
libeer_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'LibEER')
os.chdir(libeer_dir)
sys.path.insert(0, libeer_dir)

import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from models.Models import Model
from config.setting import seediv_sub_dependent_train_val_test_setting
from data_utils.load_data import get_data
from data_utils.split import merge_to_part, index_to_data, get_split_index
from utils.utils import setup_seed, result_log, sub_result_log
from utils.store import make_output_dir
from Trainer.training import train
from models.DGCNN import NewSparseL2Regularization


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', required=True)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--seed', type=int, default=2024)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parsed = parser.parse_args()

    # Build args namespace matching LibEER's expected format
    class Args:
        pass
    args = Args()
    args.dataset = 'seediv_raw'
    args.dataset_path = parsed.dataset_path
    args.low_pass = 0.3
    args.high_pass = 50
    args.time_window = 1
    args.overlap = 0
    args.sample_length = 1
    args.stride = 1
    args.feature_type = 'de_lds'
    args.only_seg = False
    args.normalize = True
    args.seed = parsed.seed
    args.onehot = True
    args.sessions = None
    args.pr = None
    args.sr = None
    args.label_used = None
    args.bounds = None
    args.setting = 'seediv_sub_dependent_train_val_test_setting'
    args.metrics = ['acc', 'macro-f1']
    args.metric_choose = 'macro-f1'
    args.batch_size = parsed.batch_size
    args.epochs = parsed.epochs
    args.lr = parsed.lr
    args.device = parsed.device
    args.keep_dim = False
    args.log_dir = './log/'
    args.output_dir = './result/'
    args.model = 'DGCNN'
    args.eog_clean = False
    args.cross_trail = 'true'
    args.split_type = 'train-val-test'
    args.fold_num = 5
    args.fold_shuffle = True
    args.front = 9
    import time as _time
    args.time = _time.localtime()

    print(f"\n{'='*60}")
    print(f"LibEER DGCNN on SEED-IV (DE-LDS features)")
    print(f"  Dataset   : seediv_raw → DE-LDS extraction")
    print(f"  Channels  : 62")
    print(f"  Features  : DE-LDS (5 bands × 62ch)")
    print(f"  Epochs    : {args.epochs}")
    print(f"  Batch     : {args.batch_size}")
    print(f"  LR        : {args.lr}")
    print(f"  Mode      : subject-dependent (train-val-test)")
    print(f"{'='*60}\n")

    # Use LibEER's preset setting
    setting = seediv_sub_dependent_train_val_test_setting(args)
    setup_seed(args.seed)

    print("Loading and preprocessing SEED-IV...")
    data, label, channels, feature_dim, num_classes = get_data(setting)
    print(f"  Channels: {channels}, Feature dim: {feature_dim}, Classes: {num_classes}")

    data, label = merge_to_part(data, label, setting)
    device = torch.device(args.device)

    best_metrics = []
    dependent_metrics = [[] for _ in range(len(data))]

    for rridx, (data_i, label_i) in enumerate(zip(data, label), 1):
        tts = get_split_index(data_i, label_i, setting)
        for ridx, (train_indexes, test_indexes, val_indexes) in enumerate(
            zip(tts['train'], tts['test'], tts['val']), 1
        ):
            setup_seed(args.seed)
            print(f"\n--- Subject {rridx}, Split {ridx} ---")

            train_data, train_label, val_data, val_label, test_data, test_label = \
                index_to_data(data_i, label_i, train_indexes, test_indexes, val_indexes)

            if len(val_data) == 0:
                val_data = test_data
                val_label = test_label

            model = Model['DGCNN'](channels, feature_dim, num_classes)

            dataset_train = torch.utils.data.TensorDataset(
                torch.Tensor(train_data), torch.Tensor(train_label))
            dataset_val = torch.utils.data.TensorDataset(
                torch.Tensor(val_data), torch.Tensor(val_label))
            dataset_test = torch.utils.data.TensorDataset(
                torch.Tensor(test_data), torch.Tensor(test_label))

            optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                                     weight_decay=1e-4, eps=1e-4)
            criterion = nn.CrossEntropyLoss()
            loss_func = NewSparseL2Regularization(0.01).to(device)

            output_dir = make_output_dir(args, "DGCNN")
            round_metric = train(
                model=model, dataset_train=dataset_train,
                dataset_val=dataset_val, dataset_test=dataset_test,
                device=device, output_dir=output_dir,
                metrics=args.metrics, metric_choose=args.metric_choose,
                optimizer=optimizer, batch_size=args.batch_size,
                epochs=args.epochs, criterion=criterion,
                loss_func=loss_func, loss_param=model
            )
            best_metrics.append(round_metric)
            dependent_metrics[rridx - 1].append(round_metric)

    print(f"\n{'='*60}")
    print(f"FINAL RESULTS — DGCNN + DE-LDS on SEED-IV (62ch)")
    print(f"{'='*60}")
    sub_result_log(args, dependent_metrics)


if __name__ == '__main__':
    main()
