"""
Channel-Independent Mamba (CiMamba) for EEG Emotion Recognition.

Each EEG channel is processed independently by a SHARED encoder
(ConvStem + MambaBlocks), then channel features are aggregated.

This makes the model channel-position agnostic — enabling transfer
learning across datasets with different channel configurations
(e.g., SEED-IV → Emognition).

Architecture:
    Input: (B, C, T)  — C channels, T time samples
    For each channel i:
        (B, 1, T) → SharedConvStem → SharedMambaBlocks → pool → (B, D)
    Aggregate: mean over C channels → (B, D)
    Classify: Linear → (B, num_classes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────
# Reuse SSM components (same as mamba_model.py)
# ─────────────────────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    """Selective State Space Model (S6) — input-dependent SSM."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand)

        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner,
                                   kernel_size=d_conv, padding=d_conv - 1,
                                   groups=self.d_inner, bias=True)
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log    = nn.Parameter(torch.log(A))
        self.dt_proj  = nn.Linear(1, self.d_inner, bias=True)

        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            self.dt_proj.bias.copy_(dt_init + torch.log(-torch.expm1(-dt_init)))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        B, L, _ = x.shape

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_branch = self.conv1d(x_branch.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_branch = F.silu(x_branch)

        x_ssm = self.x_proj(x_branch)
        B_p = x_ssm[:, :, :self.d_state]
        C_p = x_ssm[:, :, self.d_state:2 * self.d_state]
        dt  = F.softplus(self.dt_proj(x_ssm[:, :, -1:]))
        A   = -torch.exp(self.A_log)

        y = self._selective_scan(x_branch, dt, A, B_p, C_p)
        y = y * F.silu(z)
        return self.out_proj(y) + residual

    def _selective_scan(self, u, delta, A, B, C):
        batch, seq_len, d_inner = u.shape
        delta_A   = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        delta_B_u = (delta * u).unsqueeze(-1) * B.unsqueeze(2)
        h  = torch.zeros(batch, d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys = []
        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B_u[:, t]
            ys.append((h * C[:, t].unsqueeze(1)).sum(dim=-1))
        return torch.stack(ys, dim=1)


class MambaBlock(nn.Module):
    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.ff  = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.ssm(x)
        return x + self.ff(x)


# ─────────────────────────────────────────────────────────────
# Channel-Independent Mamba
# ─────────────────────────────────────────────────────────────

class CiMambaEncoder(nn.Module):
    """
    Shared single-channel encoder: ConvStem + MambaBlocks.
    Applied identically to each input channel.
    """

    def __init__(self, d_model=64, n_layers=2, d_state=16, dropout=0.3):
        super().__init__()

        # ConvStem: (B, 1, T) → (B, d_model, T/16)
        # Input channels = 1 (single channel)
        self.conv_stem = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=25, stride=4, padding=12),
            nn.BatchNorm1d(32),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(32, d_model, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, dropout=dropout)
            for _ in range(n_layers)
        ])

        self.d_model = d_model

    def forward(self, x_single):
        """
        Args:
            x_single: (B, 1, T) — single channel
        Returns:
            feature: (B, d_model)
        """
        x = self.conv_stem(x_single)          # (B, d_model, T')
        x = x.transpose(1, 2)                 # (B, T', d_model)
        for layer in self.mamba_layers:
            x = layer(x)
        return x.mean(dim=1)                  # (B, d_model)  — global avg pool


class CiMambaClassifier(nn.Module):
    """
    Channel-Independent Mamba Classifier.

    Processes each EEG channel with a SHARED encoder, then aggregates
    channel-level features and classifies.

    Args:
        n_channels: Number of input EEG channels (default: 4)
        num_classes: Number of output classes (default: 4)
        d_model: Hidden dimension (default: 64)
        n_layers: Number of Mamba blocks (default: 2)
        d_state: SSM state dimension (default: 16)
        dropout: Dropout rate (default: 0.3)
        aggregation: How to combine channels — 'mean' or 'attention'
    """

    def __init__(
        self,
        n_channels=4,
        num_classes=4,
        d_model=64,
        n_layers=2,
        d_state=16,
        dropout=0.3,
        aggregation='mean',
    ):
        super().__init__()
        self.n_channels   = n_channels
        self.aggregation  = aggregation

        # Shared encoder — same weights applied to every channel
        self.encoder = CiMambaEncoder(d_model, n_layers, d_state, dropout)

        # Optional channel attention for aggregation
        if aggregation == 'attention':
            self.ch_attn = nn.Sequential(
                nn.Linear(d_model, 16),
                nn.Tanh(),
                nn.Linear(16, 1),
            )

        # Classifier head
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (B, n_channels, T)
        Returns:
            logits: (B, num_classes)
        """
        B, C, T = x.shape
        assert C == self.n_channels, f"Expected {self.n_channels} channels, got {C}"

        # Process each channel independently with shared encoder
        ch_features = []
        for i in range(C):
            xi = x[:, i:i+1, :]               # (B, 1, T)
            fi = self.encoder(xi)              # (B, d_model)
            ch_features.append(fi)

        ch_features = torch.stack(ch_features, dim=1)  # (B, C, d_model)

        # Aggregate channels
        if self.aggregation == 'attention':
            attn_weights = self.ch_attn(ch_features)   # (B, C, 1)
            attn_weights = F.softmax(attn_weights, dim=1)
            agg = (ch_features * attn_weights).sum(dim=1)  # (B, d_model)
        else:
            agg = ch_features.mean(dim=1)              # (B, d_model)

        return self.head(agg)


if __name__ == '__main__':
    model = CiMambaClassifier(n_channels=4, num_classes=4, d_model=64)
    x = torch.randn(2, 4, 2000)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Input:  {x.shape}")
    print(f"Params: {n_params:,}")
    y = model(x)
    print(f"Output: {y.shape}")
