"""
Mamba (State Space Model) based EEG Emotion Classifier.

Architecture:
    Raw EEG [B, 4, T]  (T ≈ 12000 at 200Hz)
        → Conv Stem (downsample + expand channels)
        → 2x MambaBlock (selective SSM in pure PyTorch)
        → Global Average Pool
        → FC classifier → 4 classes

Pure PyTorch implementation — no mamba-ssm CUDA kernels required.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SelectiveSSM(nn.Module):
    """
    Selective State Space Model (S6) — the core of Mamba.
    
    Implements input-dependent (selective) discretization of a continuous SSM:
        x'(t) = A x(t) + B u(t)
        y(t)  = C x(t)
    
    Where A, B, C, Δ are all computed from the input, making the model
    input-dependent (selective) unlike classic S4 which uses fixed parameters.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(d_model * expand)

        # Input projection: project to 2x inner dim (for x and z gate)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # 1D depthwise conv on the x branch
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )

        # SSM parameter projections (input-dependent)
        self.x_proj = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)  # B, C, dt

        # Learnable log(A) — initialized to HiPPO-like values
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # dt (delta) projection bias
        self.dt_proj = nn.Linear(1, self.d_inner, bias=True)

        # Initialization for dt bias: uniform in [0.001, 0.1] in log space
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(self.d_inner) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            # Inverse of softplus for initialization
            inv_softplus = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_softplus)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Layer norm
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)
        """
        residual = x
        x = self.norm(x)
        B, L, D = x.shape

        # Project input to 2x inner dim
        xz = self.in_proj(x)  # (B, L, 2 * d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # 1D conv on x branch
        x_branch = x_branch.transpose(1, 2)  # (B, d_inner, L)
        x_branch = self.conv1d(x_branch)[:, :, :L]  # causal: trim to L
        x_branch = x_branch.transpose(1, 2)  # (B, L, d_inner)
        x_branch = F.silu(x_branch)

        # Compute input-dependent SSM parameters
        x_ssm = self.x_proj(x_branch)  # (B, L, d_state*2 + 1)
        B_param = x_ssm[:, :, :self.d_state]  # (B, L, d_state)
        C_param = x_ssm[:, :, self.d_state:2 * self.d_state]  # (B, L, d_state)
        dt = x_ssm[:, :, -1:]  # (B, L, 1)

        # Compute delta (discretization step) — input dependent
        dt = self.dt_proj(dt)  # (B, L, d_inner)
        dt = F.softplus(dt)  # ensure positive

        # Get A (negative for stability)
        A = -torch.exp(self.A_log)  # (d_inner, d_state)

        # Selective scan (sequential — pure PyTorch, no CUDA kernel)
        y = self._selective_scan(x_branch, dt, A, B_param, C_param)

        # Gate with z
        y = y * F.silu(z)

        # Output projection
        out = self.out_proj(y)

        return out + residual

    def _selective_scan(self, u, delta, A, B, C):
        """
        Selective scan (S6) — sequential implementation.
        
        Args:
            u: (B, L, d_inner) — input
            delta: (B, L, d_inner) — discretization step
            A: (d_inner, d_state) — state matrix (negative)
            B: (B, L, d_state) — input-dependent B
            C: (B, L, d_state) — input-dependent C
        
        Returns:
            y: (B, L, d_inner)
        """
        batch, seq_len, d_inner = u.shape
        d_state = A.shape[1]

        # Discretize: A_bar = exp(delta * A), B_bar = delta * B
        # delta: (B, L, d_inner) -> (B, L, d_inner, 1)
        delta_A = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))  # (B, L, d_inner, d_state)

        # B: (B, L, d_state) -> need to combine with delta and u
        delta_B_u = (delta * u).unsqueeze(-1) * B.unsqueeze(2)  # (B, L, d_inner, d_state)

        # Sequential scan
        h = torch.zeros(batch, d_inner, d_state, device=u.device, dtype=u.dtype)
        ys = []

        for t in range(seq_len):
            h = delta_A[:, t] * h + delta_B_u[:, t]  # (B, d_inner, d_state)
            y_t = (h * C[:, t].unsqueeze(1)).sum(dim=-1)  # (B, d_inner)
            ys.append(y_t)

        y = torch.stack(ys, dim=1)  # (B, L, d_inner)
        return y


class MambaBlock(nn.Module):
    """Single Mamba block: SelectiveSSM + feed-forward."""

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.ssm = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        x = self.ssm(x)
        x = x + self.ff(x)
        return x


class MambaEEGClassifier(nn.Module):
    """
    Full-clip EEG emotion classifier using Mamba (SSM).
    
    Input:  (B, 4, T)  — 4-channel raw EEG, T ≈ 12000 samples at 200Hz
    Output: (B, num_classes) — emotion logits
    """

    def __init__(
        self,
        in_channels=4,
        num_classes=4,
        d_model=128,
        n_layers=2,
        d_state=16,
        d_conv=4,
        expand=2,
        dropout=0.3,
    ):
        super().__init__()

        # --- Conv Stem: downsample + expand channels ---
        # Stage 1: (B, 4, T) → (B, 64, T/4)
        self.conv_stem = nn.Sequential(
            nn.Conv1d(in_channels, 64, kernel_size=25, stride=4, padding=12),
            nn.BatchNorm1d(64),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            # Stage 2: (B, 64, T/4) → (B, d_model, T/16)
            nn.Conv1d(64, d_model, kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

        # --- Mamba blocks ---
        self.mamba_layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand, dropout)
            for _ in range(n_layers)
        ])

        # --- Classifier head ---
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (B, C, T) where C=4 channels, T=raw time samples
        Returns:
            logits: (B, num_classes)
        """
        # Conv stem: (B, 4, T) → (B, d_model, T')
        x = self.conv_stem(x)

        # Transpose for Mamba: (B, d_model, T') → (B, T', d_model)
        x = x.transpose(1, 2)

        # Mamba blocks
        for layer in self.mamba_layers:
            x = layer(x)

        # Global average pool: (B, T', d_model) → (B, d_model)
        x = x.mean(dim=1)

        # Classify
        logits = self.head(x)
        return logits


if __name__ == '__main__':
    # Quick shape verification
    model = MambaEEGClassifier(in_channels=4, num_classes=4)
    x = torch.randn(2, 4, 12000)  # 2 samples, 4 channels, 60s at 200Hz

    print(f"Input shape:  {x.shape}")

    # Count parameters
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters:   {n_params:,}")

    y = model(x)
    print(f"Output shape: {y.shape}")
    print(f"Output:\n{y}")
