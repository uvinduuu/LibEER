#!/usr/bin/env python3
"""
inspect_checkpoint.py  —  Inspect a .pt checkpoint for NaN weights.

Usage:
    python emognition/bimamba_ssl/inspect_checkpoint.py \
        --ckpt /path/to/pretrained_supcon_full.pt
"""

import argparse
import os
import torch
import numpy as np


def inspect(ckpt_path: str):
    size_kb = os.path.getsize(ckpt_path) / 1024
    print(f"\nFile : {ckpt_path}")
    print(f"Size : {size_kb:.1f} KB  (expected ~460 KB for 115K float32 params)")
    print()

    ckpt = torch.load(ckpt_path, map_location='cpu')

    # ── 1. Show top-level structure ──────────────────────────────────────────
    if isinstance(ckpt, dict):
        print(f"Top-level keys: {list(ckpt.keys())}")
    else:
        print(f"Type: {type(ckpt)}  (plain state_dict)")
        ckpt = {'<root>': ckpt}

    # ── 2. For each state_dict found, count NaN params ──────────────────────
    print()
    for section, sd in ckpt.items():
        if not isinstance(sd, dict):
            print(f"  [{section}] = {type(sd).__name__}: {sd}")
            continue

        total_params = 0
        nan_params   = 0
        inf_params   = 0
        nan_layers   = []

        for k, v in sd.items():
            if not isinstance(v, torch.Tensor):
                continue
            n = v.numel()
            n_nan = int(torch.isnan(v).sum())
            n_inf = int(torch.isinf(v).sum())
            total_params += n
            nan_params   += n_nan
            inf_params   += n_inf
            if n_nan > 0 or n_inf > 0:
                nan_layers.append(f"    {k}: {n_nan}/{n} NaN, {n_inf}/{n} Inf")

        pct = 100 * nan_params / max(total_params, 1)
        status = "✓ CLEAN" if nan_params == 0 and inf_params == 0 else "✗ CORRUPT"
        print(f"  [{section}]  {total_params:,} params  →  "
              f"{nan_params:,} NaN ({pct:.1f}%)  {inf_params:,} Inf  [{status}]")

        if nan_layers:
            print(f"    Corrupt layers (first 10):")
            for l in nan_layers[:10]:
                print(l)
            if len(nan_layers) > 10:
                print(f"    ... and {len(nan_layers)-10} more layers")
        print()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', required=True,
                        help='Path to the .pt checkpoint file to inspect')
    args = parser.parse_args()
    inspect(args.ckpt)
