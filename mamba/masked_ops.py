"""
Masking utilities for variable-length EEG sequences.

Provides masked global average pooling so the model ignores
zero-padded positions when computing trial-level representations.

Usage:
    from masked_ops import masked_global_avg_pool

    # x: (B, T, D) — sequence features
    # lengths: (B,) — actual (unpadded) length of each sequence
    pooled = masked_global_avg_pool(x, lengths)  # (B, D)
"""

import torch


def create_length_mask(lengths, max_len, device=None):
    """
    Create a boolean mask from sequence lengths.

    Args:
        lengths: (B,) tensor of actual sequence lengths
        max_len: int, the padded sequence length
        device: torch device

    Returns:
        mask: (B, max_len) bool tensor — True for real positions, False for padding
    """
    if device is None:
        device = lengths.device
    # (1, max_len) vs (B, 1)
    arange = torch.arange(max_len, device=device).unsqueeze(0)  # (1, T)
    lengths = lengths.unsqueeze(1)  # (B, 1)
    mask = arange < lengths  # (B, T)
    return mask


def masked_global_avg_pool(x, lengths):
    """
    Global average pooling that only averages over non-padded positions.

    Args:
        x: (B, T, D) — feature sequence (after Mamba blocks)
        lengths: (B,) — actual length of each sequence (in the T dimension)

    Returns:
        pooled: (B, D) — average pooled features (ignoring padding)
    """
    B, T, D = x.shape

    if lengths is None:
        # Fallback: plain average (no masking)
        return x.mean(dim=1)

    # Create mask: (B, T)
    mask = create_length_mask(lengths, T, device=x.device)

    # Expand mask for broadcasting: (B, T, 1)
    mask_expanded = mask.unsqueeze(-1).float()

    # Zero out padded positions and sum
    x_masked = x * mask_expanded  # (B, T, D)
    x_sum = x_masked.sum(dim=1)  # (B, D)

    # Divide by actual lengths (not padded length)
    lengths_clamped = lengths.float().clamp(min=1).unsqueeze(-1)  # (B, 1)
    pooled = x_sum / lengths_clamped  # (B, D)

    return pooled
