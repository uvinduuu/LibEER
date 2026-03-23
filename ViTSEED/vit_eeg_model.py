"""
Vision Transformer (ViT) for 4-Channel EEG Emotion Recognition.

Patch-based 1D approach — no CWT needed:
    Input: (B, 4, T)  — 4-channel EEG window
    ↓ Split time axis into N patches of P samples each
      Patch: (4 channels × P samples) = 4P values
    ↓ Linear patch embedding: 4P → dim
    ↓ + learnable positional encoding
    ↓ Prepend [CLS] token  
    ↓ Transformer Encoder (depth layers, multi-head attention)
    ↓ CLS token output → classifier head

Why ViT outperforms Mamba on 4-channel EEG:
    - Self-attention captures F7↔F8 frontal asymmetry in one step
    - Every patch attends to every other patch (non-causal by design)
    - No sequential bottleneck — full GPU parallelism
    - 50 patches → 50² = 2500 attention ops (tiny, very fast)

Architecture parameters for 4ch / 4s window:
    Patch size P = 16 samples, window = 800 → N = 50 patches
    Patch dim  = 4 × 16 = 64 → projected to dim = 256
    Depth = 6 transformer layers
    Heads = 8 attention heads
    MLP   = 512 hidden dim
    ~3.5M parameters
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Patch Embedding ───────────────────────────────────────────

class PatchEmbedding(nn.Module):
    """
    Split EEG window into time patches and project to embedding dim.

    Input:  (B, n_channels, T)
    Output: (B, n_patches, dim)
    """
    def __init__(self, n_channels, patch_size, dim, dropout=0.1):
        super().__init__()
        self.patch_size  = patch_size
        self.patch_dim   = n_channels * patch_size   # raw values per patch
        self.projection  = nn.Sequential(
            nn.Linear(self.patch_dim, dim),
            nn.LayerNorm(dim),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """x: (B, C, T) → (B, N, dim)"""
        B, C, T = x.shape
        assert T % self.patch_size == 0, \
            f"T={T} not divisible by patch_size={self.patch_size}"
        n_patches = T // self.patch_size
        # Reshape: (B, C, N, P) → (B, N, C*P)
        x = x.reshape(B, C, n_patches, self.patch_size)
        x = x.permute(0, 2, 1, 3).reshape(B, n_patches, -1)   # (B, N, C*P)
        x = self.projection(x)                                  # (B, N, dim)
        return self.dropout(x)


# ─── Multi-Head Self-Attention ─────────────────────────────────

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, n_heads, dropout=0.1):
        super().__init__()
        assert dim % n_heads == 0
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale    = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out  = nn.Linear(dim, dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

    def forward(self, x):
        B, N, D = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)               # each: (B, H, N, head_dim)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        return self.proj_drop(self.out(x))


# ─── Transformer Block ─────────────────────────────────────────

class TransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn  = MultiHeadAttention(dim, n_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.ff    = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ff(self.norm2(x))
        return x


# ─── EEG ViT ───────────────────────────────────────────────────

class EEGViT(nn.Module):
    """
    Vision Transformer for EEG emotion recognition.

    Args:
        n_channels:  EEG channels (default: 4)
        n_samples:   Time samples per window (e.g., 800 for 4s@200Hz)
        patch_size:  Samples per patch (default: 16 → 50 patches for 4s)
        num_classes: Output classes (default: 4)
        dim:         Embedding dimension (default: 256)
        depth:       Transformer layers (default: 6)
        n_heads:     Attention heads (default: 8)
        mlp_dim:     FFN hidden dim (default: 512)
        dropout:     Dropout rate (default: 0.1)
        emb_dropout: Embedding dropout (default: 0.1)
    """

    def __init__(
        self,
        n_channels  = 4,
        n_samples   = 800,       # 4s × 200Hz
        patch_size  = 16,
        num_classes = 4,
        dim         = 256,
        depth       = 6,
        n_heads     = 8,
        mlp_dim     = 512,
        dropout     = 0.1,
        emb_dropout = 0.1,
    ):
        super().__init__()
        assert n_samples % patch_size == 0, \
            f"n_samples={n_samples} must be divisible by patch_size={patch_size}"

        self.n_patches  = n_samples // patch_size

        # Patch embedding
        self.patch_embed = PatchEmbedding(n_channels, patch_size, dim, emb_dropout)

        # CLS token + positional encoding
        self.cls_token   = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed   = nn.Parameter(
            torch.zeros(1, self.n_patches + 1, dim)   # +1 for CLS
        )
        self.pos_dropout = nn.Dropout(emb_dropout)

        # Transformer
        self.transformer = nn.Sequential(*[
            TransformerBlock(dim, n_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        # Head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, mlp_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim // 2, num_classes),
        )

        # Weight initialisation
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            nn.init.zeros_(m.bias)
            nn.init.ones_(m.weight)

    def forward(self, x):
        """
        Args:
            x: (B, n_channels, T)
        Returns:
            logits: (B, num_classes)
        """
        B = x.shape[0]

        # Patch + positional embedding
        x   = self.patch_embed(x)                                # (B, N, dim)
        cls = self.cls_token.expand(B, -1, -1)                   # (B, 1, dim)
        x   = torch.cat([cls, x], dim=1)                         # (B, N+1, dim)
        x   = self.pos_dropout(x + self.pos_embed)               # (B, N+1, dim)

        # Transformer
        x = self.transformer(x)                                  # (B, N+1, dim)

        # CLS token → head
        cls_out = self.norm(x[:, 0])                             # (B, dim)
        return self.head(cls_out)                                # (B, num_classes)


if __name__ == '__main__':
    # Quick test
    model  = EEGViT(n_channels=4, n_samples=800, patch_size=16,
                    num_classes=4, dim=256, depth=6, n_heads=8, mlp_dim=512)
    x      = torch.randn(4, 4, 800)
    n_par  = sum(p.numel() for p in model.parameters())
    y      = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {n_par:,}")
    print(f"Patches: {model.n_patches}")
