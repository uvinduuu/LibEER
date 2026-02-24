"""
Cross-Attention Fusion module.

Fuses EEGNet-learned features with handcrafted features
via bidirectional cross-attention, letting the model learn
which feature type is more useful.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    """
    Multi-head cross-attention: Q from one branch, K/V from the other.

    Args:
        d_model:   feature dimension (both branches projected to this)
        num_heads: number of attention heads
        dropout:   attention dropout
    """
    def __init__(self, d_model=128, num_heads=4, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, query, key_value):
        """
        Args:
            query:     (B, T_q, d_model) — the branch asking questions
            key_value: (B, T_kv, d_model) — the branch providing answers
        Returns:
            output:    (B, T_q, d_model) — attended features
        """
        B, T_q, _ = query.shape

        # Project Q, K, V
        Q = self.W_q(query)      # (B, T_q, d_model)
        K = self.W_k(key_value)  # (B, T_kv, d_model)
        V = self.W_v(key_value)  # (B, T_kv, d_model)

        # Reshape for multi-head: (B, heads, T, d_k)
        Q = Q.view(B, T_q, self.num_heads, self.d_k).transpose(1, 2)
        K = K.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = V.view(B, -1, self.num_heads, self.d_k).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)

        # Weighted sum
        context = torch.matmul(attn, V)  # (B, heads, T_q, d_k)
        context = context.transpose(1, 2).contiguous().view(B, T_q, self.d_model)

        # Output projection + residual + norm
        output = self.W_o(context)
        output = self.norm(query + self.dropout(output))

        return output


class CrossAttentionFusion(nn.Module):
    """
    Bidirectional cross-attention fusion between two feature branches.

    Branch A (learned): (B, T_a, dim_a) — from EEGNet
    Branch B (handcrafted): (B, T_b, dim_b) — from feature extraction

    Both are projected to d_model, then:
      1) A attends to B: "which handcrafted features help interpret learned features?"
      2) B attends to A: "which learned features are relevant to handcrafted features?"
      3) Both outputs are concatenated → fused sequence

    Args:
        dim_a:      feature dim of branch A (EEGNet output, e.g. 128)
        dim_b:      feature dim of branch B (handcrafted, e.g. 26)
        d_model:    shared projection dim for cross-attention
        num_heads:  attention heads
        dropout:    dropout rate
    """
    def __init__(self, dim_a=128, dim_b=26, d_model=128,
                 num_heads=4, dropout=0.1):
        super().__init__()

        # Project both branches to same dimension
        self.proj_a = nn.Sequential(
            nn.Linear(dim_a, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        ) if dim_a != d_model else nn.Identity()

        self.proj_b = nn.Sequential(
            nn.Linear(dim_b, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU()
        )

        # Bidirectional cross-attention
        self.cross_attn_a_to_b = CrossAttention(d_model, num_heads, dropout)
        self.cross_attn_b_to_a = CrossAttention(d_model, num_heads, dropout)

        # Fusion gate: learn how to balance the two branches
        self.gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )

        self.output_norm = nn.LayerNorm(d_model)
        self.d_model = d_model

    def forward(self, feat_a, feat_b):
        """
        Args:
            feat_a: (B, T_a, dim_a) learned features from EEGNet
            feat_b: (B, T_b, dim_b) handcrafted features
        Returns:
            fused: (B, T_a + T_b, d_model) fused feature sequence
        """
        # Project to shared dimension
        a = self.proj_a(feat_a)  # (B, T_a, d_model)
        b = self.proj_b(feat_b)  # (B, T_b, d_model)

        # Cross-attention: A queries B, B queries A
        a_attended = self.cross_attn_a_to_b(a, b)  # (B, T_a, d_model)
        b_attended = self.cross_attn_b_to_a(b, a)  # (B, T_b, d_model)

        # Gated fusion for branch A: blend original + attended
        gate_input_a = torch.cat([a, a_attended], dim=-1)  # (B, T_a, 2*d_model)
        g_a = self.gate(gate_input_a)                       # (B, T_a, d_model)
        a_fused = g_a * a + (1 - g_a) * a_attended

        # Gated fusion for branch B
        gate_input_b = torch.cat([b, b_attended], dim=-1)
        g_b = self.gate(gate_input_b)
        b_fused = g_b * b + (1 - g_b) * b_attended

        # Concatenate both branches as a unified sequence
        fused = torch.cat([a_fused, b_fused], dim=1)  # (B, T_a+T_b, d_model)
        fused = self.output_norm(fused)

        return fused
