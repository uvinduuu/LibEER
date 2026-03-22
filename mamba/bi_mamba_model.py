"""
Bidirectional Mamba (BiMamba) EEG Classifier.

Key change over standard Mamba:
  - Forward SSM:  h_0 → h_1 → ... → h_T  (left-to-right)
  - Backward SSM: h_T → h_{T-1} → ... → h_0  (right-to-left)
  - Concatenate [forward, backward] → project back to d_model

This is appropriate for EEG emotion recognition because:
  1. EEG emotion labels are assigned to entire trials — non-causal task
  2. Future context (e.g., sustained alpha suppression) helps classify past states
  3. BiLSTMs consistently outperform LSTMs on similar EEG tasks

Architecture:
    Input:  (B, 4, T)  — 4-channel raw EEG
    ConvStem (4→32→128, total stride 16):  (B, 128, T/16)
    N × BiMambaBlock:
        Forward  SSM on (B, T', 128)  → (B, T', 128)
        Backward SSM on reversed input → (B, T', 128)
        Concat + project:              → (B, T', 128)
    Global average pool: (B, 128)
    Classifier: (B, num_classes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─── Selective SSM (S6) ────────────────────────────────────────

class SelectiveSSM(nn.Module):
    """Input-dependent State Space Model (S6 from Mamba paper)."""

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

        A = torch.arange(1, d_state + 1, dtype=torch.float32
                         ).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log    = nn.Parameter(torch.log(A))
        self.dt_proj  = nn.Linear(1, self.d_inner, bias=True)

        with torch.no_grad():
            dt = torch.exp(
                torch.rand(self.d_inner) *
                (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            self.dt_proj.bias.copy_(dt + torch.log(-torch.expm1(-dt)))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = self.norm(x)
        B, L, _ = x.shape

        xz = self.in_proj(x)
        x_b, z = xz.chunk(2, dim=-1)

        x_b = self.conv1d(x_b.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_b = F.silu(x_b)

        params = self.x_proj(x_b)
        B_p = params[:, :, :self.d_state]
        C_p = params[:, :, self.d_state:2 * self.d_state]
        dt  = F.softplus(self.dt_proj(params[:, :, -1:]))
        A   = -torch.exp(self.A_log)

        y = self._scan(x_b, dt, A, B_p, C_p)
        y = y * F.silu(z)
        return self.out_proj(y) + residual

    def _scan(self, u, delta, A, B, C):
        """Sequential selective scan (pure PyTorch — compatible everywhere)."""
        batch, seq_len, d_inner = u.shape
        dA    = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dBu   = (delta * u).unsqueeze(-1) * B.unsqueeze(2)
        h     = torch.zeros(batch, d_inner, self.d_state, device=u.device, dtype=u.dtype)
        ys    = []
        for t in range(seq_len):
            h = dA[:, t] * h + dBu[:, t]
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
        return torch.stack(ys, dim=1)


# ─── BiMamba Block ─────────────────────────────────────────────

class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block.

    Runs two INDEPENDENT SSMs — one forward, one backward —
    then fuses their outputs via a learned projection.

    The forward and backward SSMs do NOT share weights so each
    can specialise in its respective temporal direction.
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2, dropout=0.1):
        super().__init__()
        self.ssm_fwd = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.ssm_bwd = SelectiveSSM(d_model, d_state, d_conv, expand)

        # Fuse [forward, backward] → d_model
        self.fuse_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # Feed-forward
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        """x: (B, T, d_model)"""
        # Forward pass
        fwd = self.ssm_fwd(x)                         # (B, T, d_model)

        # Backward pass: reverse sequence, run SSM, reverse back
        x_rev = torch.flip(x, dims=[1])
        bwd   = self.ssm_bwd(x_rev)
        bwd   = torch.flip(bwd, dims=[1])              # (B, T, d_model)

        # Fuse
        fused = self.fuse_proj(torch.cat([fwd, bwd], dim=-1))  # (B, T, d_model)

        # Residual + feed-forward
        return fused + self.ff(fused)                  # (B, T, d_model)


# ─── Conv Stem ─────────────────────────────────────────────────

class ConvStem(nn.Module):
    """
    1D convolutional stem: (B, in_ch, T) → (B, d_model, T/16).
    Two stages of stride-4 convolution = total stride 16.
    """
    def __init__(self, in_channels, d_model, dropout=0.2):
        super().__init__()
        mid = max(32, d_model // 2)
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, mid,     kernel_size=25, stride=4, padding=12),
            nn.BatchNorm1d(mid),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Conv1d(mid,        d_model,  kernel_size=15, stride=4, padding=7),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
        )

    def forward(self, x):
        return self.net(x)


# ─── BiMamba EEG Classifier ────────────────────────────────────

class BiMambaEEGClassifier(nn.Module):
    """
    Bidirectional Mamba classifier for EEG emotion recognition.

    Improvement over standard (unidirectional) MambaEEGClassifier:
      - Two SSMs per block (independent forward/backward weights)
      - Fused representation captures both past and future context
      - ~2× parameters per block, but same inference speed on GPU

    Args:
        in_channels:  Number of EEG channels (default: 4)
        num_classes:  Output classes (default: 4 for SEED-IV)
        d_model:      Hidden dimension (default: 128)
        n_layers:     Number of BiMamba blocks (default: 3)
        d_state:      SSM state dimension (default: 16)
        dropout:      Dropout rate (default: 0.4)
    """

    def __init__(
        self,
        in_channels=4,
        num_classes=4,
        d_model=128,
        n_layers=3,
        d_state=16,
        dropout=0.4,
    ):
        super().__init__()
        self.conv_stem = ConvStem(in_channels, d_model, dropout)
        self.bi_layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state, dropout=dropout)
            for _ in range(n_layers)
        ])
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def forward(self, x):
        """
        Args:
            x: (B, in_channels, T)
        Returns:
            logits: (B, num_classes)
        """
        x = self.conv_stem(x)           # (B, d_model, T/16)
        x = x.transpose(1, 2)           # (B, T/16, d_model)
        for layer in self.bi_layers:
            x = layer(x)
        x = x.mean(dim=1)              # (B, d_model)  global average pool
        return self.head(x)


if __name__ == '__main__':
    model = BiMambaEEGClassifier(in_channels=4, num_classes=4,
                                  d_model=128, n_layers=3)
    x      = torch.randn(2, 4, 2000)
    n_par  = sum(p.numel() for p in model.parameters())
    y      = model(x)
    print(f"Input : {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {n_par:,}")
    print(f"Seq len after stem: {x.shape[-1] // 16}")
