"""
Multi-Band InvBase BiMamba (MB-InvBase-BiMamba) EEG Emotion Classifier.

Architecture overview:
    InvBase-normalised EEG (4, T)
        → 5-band Butterworth filter → stack → (20, T)
        → ChannelAttentionGate(20)           reweight band-channel streams
        → ConvStem(20 → 64 → d_model, ×16)  temporal downsampling
        → N × BiMambaBlock                   bidirectional SSM
        → Masked Global Average Pooling
        → LayerNorm → Dropout → Linear → num_classes

Self-contained: all components (SelectiveSSM, BiMambaBlock, ConvStem,
ChannelAttentionGate, masked_avg_pool) are defined here so the file can
be imported from any working directory without path issues.

Input expected by MBInvBaseBiMamba.forward():
    x       : (B, 20, T)  — 5 bands × 4 channels, already InvBase-normalised
                            and band-filtered.  Variable T is handled by
                            'fixed-window' training (same T in a batch) or
                            by the optional 'lengths' argument.
    lengths : (B,) long   — actual (pre-padding) time steps; pass None when
                            all samples in the batch have the same length.
Output:
    logits  : (B, num_classes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ── shared constant ─────────────────────────────────────────────────────────

NUM_EEG_CHANNELS = 4    # Muse 2
NUM_BANDS        = 5    # delta, theta, alpha, beta, gamma
IN_CHANNELS      = NUM_BANDS * NUM_EEG_CHANNELS   # 20


# ── helpers ──────────────────────────────────────────────────────────────────

def masked_avg_pool(x: torch.Tensor, lengths=None) -> torch.Tensor:
    """
    Global average pool over the time dimension with optional length mask.

    Args:
        x:       (B, T, D) — sequence features
        lengths: (B,) long — real (unpadded) lengths, or None for unmasked pool
    Returns:
        (B, D)
    """
    if lengths is None:
        return x.mean(dim=1)
    B, T, D = x.shape
    mask   = torch.arange(T, device=x.device).unsqueeze(0) < lengths.unsqueeze(1)
    mask   = mask.float().unsqueeze(-1)            # (B, T, 1)
    pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    return pooled


# ── Selective SSM (S6) ───────────────────────────────────────────────────────

class SelectiveSSM(nn.Module):
    """
    Input-dependent State Space Model (S6) — the core Mamba operator.

    Implements the selective (input-dependent) discretisation:
        h_t = A_bar_t * h_{t-1} + B_bar_t * u_t
        y_t = C_t * h_t

    where A, B, C and Δ (step size) are all functions of the input.

    Args:
        d_model: model dimension
        d_state: SSM state dimension (default: 16)
        d_conv:  depthwise-conv kernel size (default: 4)
        expand:  inner-dimension expansion factor (default: 2)
    """

    def __init__(self, d_model: int, d_state: int = 16,
                 d_conv: int = 4, expand: int = 2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_inner = int(d_model * expand)

        # Input projection: d_model → 2 * d_inner  (x branch + z gate)
        self.in_proj  = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Depthwise causal conv on the x branch
        self.conv1d   = nn.Conv1d(self.d_inner, self.d_inner,
                                  kernel_size=d_conv, padding=d_conv - 1,
                                  groups=self.d_inner, bias=True)

        # Project x to (B, C, d_state), (C, d_state), dt
        self.x_proj   = nn.Linear(self.d_inner, d_state * 2 + 1, bias=False)

        # Learnable A (log-parametrised, HiPPO-like init)
        A = torch.arange(1, d_state + 1,
                         dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log    = nn.Parameter(torch.log(A))

        # Δ (delta) projection: 1 → d_inner
        self.dt_proj  = nn.Linear(1, self.d_inner, bias=True)
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(self.d_inner) *
                (math.log(0.1) - math.log(0.001)) + math.log(0.001)
            )
            self.dt_proj.bias.copy_(dt_init + torch.log(-torch.expm1(-dt_init)))

        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)
        self.norm     = nn.LayerNorm(d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            (B, L, d_model)  with residual connection
        """
        residual = x
        x        = self.norm(x)
        B, L, _  = x.shape

        xz       = self.in_proj(x)
        x_b, z   = xz.chunk(2, dim=-1)                   # each (B, L, d_inner)

        # Causal depthwise conv
        x_b = self.conv1d(x_b.transpose(1, 2))[:, :, :L].transpose(1, 2)
        x_b = F.silu(x_b)

        # Input-dependent SSM parameters
        params = self.x_proj(x_b)                         # (B, L, 2*d_state+1)
        B_p    = params[:, :, :self.d_state]              # (B, L, d_state)
        C_p    = params[:, :, self.d_state:2*self.d_state]
        dt     = F.softplus(self.dt_proj(params[:, :, -1:]))  # (B, L, d_inner)
        A      = -torch.exp(self.A_log)                   # (d_inner, d_state)

        y = self._selective_scan(x_b, dt, A, B_p, C_p)
        y = y * F.silu(z)
        return self.out_proj(y) + residual

    def _selective_scan(self, u, delta, A, B, C):
        """Sequential selective scan — pure PyTorch, compatible everywhere."""
        batch, seq_len, d_inner = u.shape
        dA  = torch.exp(delta.unsqueeze(-1) * A.unsqueeze(0).unsqueeze(0))
        dBu = (delta * u).unsqueeze(-1) * B.unsqueeze(2)
        h   = torch.zeros(batch, d_inner, self.d_state,
                          device=u.device, dtype=u.dtype)
        ys  = []
        for t in range(seq_len):
            h = dA[:, t] * h + dBu[:, t]
            ys.append((h * C[:, t].unsqueeze(1)).sum(-1))
        return torch.stack(ys, dim=1)


# ── BiMamba Block ────────────────────────────────────────────────────────────

class BiMambaBlock(nn.Module):
    """
    Bidirectional Mamba block.

    Two independent SelectiveSSMs (forward & backward) process the sequence
    in opposite temporal directions.  Their outputs are concatenated then
    projected back to d_model before a residual feed-forward sub-layer.

    Using two independent SSMs (rather than shared weights) lets each
    specialise: the forward SSM accumulates history; the backward SSM
    accumulates future context.

    Args:
        d_model:  model dimension
        d_state:  SSM state dimension (default: 16)
        d_conv:   depthwise-conv kernel size (default: 4)
        expand:   inner-dimension expansion factor inside SSM (default: 2)
        dropout:  dropout applied in the feed-forward sub-layer (default: 0.1)
    """

    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 4,
                 expand: int = 2, dropout: float = 0.1):
        super().__init__()
        self.ssm_fwd   = SelectiveSSM(d_model, d_state, d_conv, expand)
        self.ssm_bwd   = SelectiveSSM(d_model, d_state, d_conv, expand)

        # Fuse [forward ‖ backward] → d_model
        self.fuse_proj = nn.Sequential(
            nn.LayerNorm(d_model * 2),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # Feed-forward sub-layer (with residual baked-in below)
        self.ff = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 2, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, T, d_model)  →  (B, T, d_model)"""
        fwd   = self.ssm_fwd(x)                                      # (B, T, d)
        bwd   = torch.flip(self.ssm_bwd(torch.flip(x, dims=[1])),
                           dims=[1])                                  # (B, T, d)
        fused = self.fuse_proj(torch.cat([fwd, bwd], dim=-1))        # (B, T, d)
        return fused + self.ff(fused)


# ── Channel Attention Gate ───────────────────────────────────────────────────

class ChannelAttentionGate(nn.Module):
    """
    Squeeze-and-Excitation channel attention over the 20 band-channel streams.

    Learns which (band, electrode) combinations are most discriminative for
    emotion.  Operates on (B, C, T) — squeezes over T, produces per-channel
    sigmoid gates, then re-scales the input.

    Because emotion expression varies across frequency bands:
      - alpha (TP9/TP10) is most relevant for fear/relaxation
      - gamma (AF7/AF8) is most relevant for enthusiasm
    Letting the model learn these weights avoids hard-coding assumptions.

    Args:
        n_channels: number of input channels (default: 20 = 5 bands × 4 ch)
        reduction:  bottleneck ratio for the FC layers (default: 4)
    """

    def __init__(self, n_channels: int = IN_CHANNELS, reduction: int = 4):
        super().__init__()
        bottleneck = max(n_channels // reduction, 4)
        self.gate  = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),          # (B, C, T) → (B, C, 1)
            nn.Flatten(),                      # (B, C)
            nn.Linear(n_channels, bottleneck),
            nn.ReLU(inplace=True),
            nn.Linear(bottleneck, n_channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C, T)
        Returns:
            (B, C, T) — channel-wise rescaled
        """
        weights = self.gate(x).unsqueeze(-1)   # (B, C, 1)
        return x * weights


# ── Conv Stem ────────────────────────────────────────────────────────────────

class ConvStem(nn.Module):
    """
    Two-stage 1-D convolutional stem.

    Downsamples the temporal dimension by stride 4 × stride 4 = ×16 total,
    while expanding the channel dimension to d_model.  The large kernels
    (25 and 15 samples at 256 Hz) capture patterns at ~100 ms and ~60 ms
    scales respectively, which are relevant for EEG band oscillations.

    Input:  (B, in_channels, T)
    Output: (B, d_model, T // 16)  (approximately)
    """

    def __init__(self, in_channels: int, d_model: int, dropout: float = 0.2):
        super().__init__()
        mid = max(64, d_model)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ── Full Model ───────────────────────────────────────────────────────────────

class MBInvBaseBiMamba(nn.Module):
    """
    Multi-Band InvBase Bidirectional Mamba EEG Emotion Classifier.

    Designed to receive EEG trials that have been pre-processed by:
      1. Per-subject InvBase normalization  (emognition/invbase.py)
      2. 5-band Butterworth bandpass filtering
      3. Stacking the 5 band signals → 20-channel input

    The model then:
      a. Applies channel attention to learn band-electrode importance
      b. Projects to d_model via a conv stem (downsamples by ×16)
      c. Runs N bidirectional Mamba blocks for temporal modelling
      d. Masked global average pools → d_model vector
      e. Linear classifier → num_classes logits

    Args:
        in_channels:    number of input channels: NUM_BANDS × 4 = 20
        num_classes:    output emotion classes (default: 4)
        d_model:        hidden dimension (default: 64)
        n_layers:       number of BiMambaBlocks (default: 3)
        d_state:        SSM state dimension (default: 16)
        dropout:        dropout rate applied in BiMamba FF and head (default: 0.4)
        attn_reduction: ChannelAttentionGate reduction factor (default: 4)
    """

    def __init__(
        self,
        in_channels:    int   = IN_CHANNELS,
        num_classes:    int   = 4,
        d_model:        int   = 64,
        n_layers:       int   = 3,
        d_state:        int   = 16,
        dropout:        float = 0.4,
        attn_reduction: int   = 4,
    ):
        super().__init__()

        # (a) Channel attention over 20 band-channel streams
        self.channel_attn = ChannelAttentionGate(in_channels, reduction=attn_reduction)

        # (b) Convolutional stem: (B, 20, T) → (B, d_model, T//16)
        self.conv_stem = ConvStem(in_channels, d_model, dropout)

        # (c) Bidirectional Mamba blocks
        self.bi_layers = nn.ModuleList([
            BiMambaBlock(d_model, d_state, dropout=dropout)
            for _ in range(n_layers)
        ])

        # (e) Classifier head
        self.d_model = d_model          # expose for MultimodalMBModel
        self.head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Dropout(dropout),
            nn.Linear(d_model, num_classes),
        )

    def _encode(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """Run all layers except the classifier. Returns (B, d_model) embedding."""
        x = self.channel_attn(x)               # (B, 20, T)
        x = self.conv_stem(x)                  # (B, d_model, T//16)
        if lengths is not None:
            lengths_ds = ((lengths - 1) // 16 + 1).long().clamp(max=x.shape[2])
        else:
            lengths_ds = None
        x = x.transpose(1, 2)                  # (B, T//16, d_model)
        for layer in self.bi_layers:
            x = layer(x)
        return masked_avg_pool(x, lengths_ds)   # (B, d_model)

    def get_embedding(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """Return d_model-dim embedding (before classifier). Used by BVP fusion."""
        return self._encode(x, lengths)

    def forward(self, x: torch.Tensor, lengths=None) -> torch.Tensor:
        """
        Args:
            x:       (B, in_channels, T) — stacked band-channel signals
            lengths: (B,) long           — unpadded time steps, or None

        Returns:
            logits: (B, num_classes)
        """
        return self.head(self._encode(x, lengths))    # (B, num_classes)


# ── BVP Encoder ──────────────────────────────────────────────────────────────

IBI_MAX_BEATS = 64   # IBI series padded/truncated to this length (~60 s at 70 BPM)


class BVPEncoder(nn.Module):
    """
    Hybrid BVP encoder: two parallel branches fused into a single d_bvp embedding.

    Branch A — 1D CNN on the padded IBI series
        Learns temporal dynamics of the heartbeat sequence that scalar statistics
        cannot capture: e.g., HR deceleration shape at stimulus onset, oscillatory
        RSA patterns, beat-to-beat acceleration during fear.
        Input: (B, 1, IBI_MAX_BEATS)  Output: (B, d_bvp)

    Branch B — Linear projection of handcrafted HRV features
        Clinically validated indices (RMSSD, pNN50, LF/HF ratio …) that are
        robust and interpretable even with N=41 subjects.
        Input: (B, hand_dim)  Output: (B, d_bvp)

    Fusion: concat → Linear(2*d_bvp → d_bvp) → LayerNorm → GELU → Dropout

    Args:
        hand_dim : number of handcrafted HRV features  (default: 11)
        ibi_len  : padded IBI sequence length           (default: IBI_MAX_BEATS=64)
        d_bvp    : output embedding dimension           (default: 32)
        dropout  : dropout after fusion                 (default: 0.2)
    """

    def __init__(self, hand_dim: int = 11, ibi_len: int = IBI_MAX_BEATS,
                 d_bvp: int = 32, dropout: float = 0.2):
        super().__init__()

        # ── Branch A: CNN on IBI series ───────────────────────────────────────
        self.cnn = nn.Sequential(
            nn.Conv1d(1, 16, kernel_size=3, padding=1),
            nn.BatchNorm1d(16),
            nn.GELU(),
            nn.Conv1d(16, d_bvp, kernel_size=3, padding=1),
            nn.BatchNorm1d(d_bvp),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),   # (B, d_bvp, 1) → squeeze
        )

        # ── Branch B: handcrafted projection ─────────────────────────────────
        self.hand_proj = nn.Sequential(
            nn.Linear(hand_dim, d_bvp),
            nn.GELU(),
        )

        # ── Fusion ────────────────────────────────────────────────────────────
        self.fusion = nn.Sequential(
            nn.Linear(d_bvp * 2, d_bvp),
            nn.LayerNorm(d_bvp),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, hand: torch.Tensor, ibi: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hand : (B, hand_dim) — z-scored handcrafted HRV features
            ibi  : (B, ibi_len)  — per-clip z-scored padded IBI series
        Returns:
            (B, d_bvp) — hybrid BVP embedding
        """
        cnn_emb  = self.cnn(ibi.unsqueeze(1)).squeeze(-1)           # (B, d_bvp)
        hand_emb = self.hand_proj(hand)                              # (B, d_bvp)
        return self.fusion(torch.cat([cnn_emb, hand_emb], dim=-1))  # (B, d_bvp)


# ── FiLM Conditioning ─────────────────────────────────────────────────────────

class FiLMLayer(nn.Module):
    """
    Feature-wise Linear Modulation (Perez et al., AAAI 2018).

    Conditions the EEG embedding on the BVP physiological state:

        output = LayerNorm( γ(bvp) ⊙ eeg_emb + β(bvp) )

    γ and β are per-dimension vectors produced by small linear layers from
    the BVP embedding.  Each dimension of the EEG embedding gets its own
    independent scale and shift — so the BVP state can amplify fear-related
    beta/gamma dims and suppress relaxation-related alpha dims simultaneously.

    Physiological motivation
    ────────────────────────
    The ANS (BVP/HRV) and CNS (EEG) are co-driven by the same affective state
    through different pathways.  The same EEG feature pattern has different
    emotional meaning depending on the concurrent autonomic state:
      high HR + low HRV  → amplify fear/stress EEG dimensions
      low HR  + high HRV → amplify relaxation/enthusiasm dimensions
    FiLM encodes this interaction explicitly rather than leaving it to the
    classifier to discover from a flat concatenated vector.

    Initialisation (critical for training stability with N=41 subjects)
    ───────────────────────────────────────────────────────────────────
    γ bias ← 1.0, γ weight ← N(0, 0.01)  → γ ≈ 1 at start (identity scale)
    β weight ← 0,  β bias ← 0             → β ≈ 0 at start (no shift)
    Training starts as pure EEG classification.  FiLM modulation grows only
    as sufficient gradient evidence from BVP accumulates.

    Args:
        d_feat : EEG embedding dimension to modulate (= backbone d_model)
        d_cond : BVP conditioning vector dimension   (= BVPEncoder d_bvp)
    """

    def __init__(self, d_feat: int, d_cond: int):
        super().__init__()
        self.gamma_proj = nn.Linear(d_cond, d_feat)
        self.beta_proj  = nn.Linear(d_cond, d_feat)
        self.norm       = nn.LayerNorm(d_feat)

        # Stable init: γ ≈ 1 (identity scale), β ≈ 0 (no shift)
        nn.init.normal_(self.gamma_proj.weight, mean=0.0, std=0.01)
        nn.init.ones_(self.gamma_proj.bias)
        nn.init.zeros_(self.beta_proj.weight)
        nn.init.zeros_(self.beta_proj.bias)

    def forward(self, eeg_emb: torch.Tensor, bvp_emb: torch.Tensor) -> torch.Tensor:
        """
        Args:
            eeg_emb : (B, d_feat) — EEG backbone embedding
            bvp_emb : (B, d_cond) — BVP conditioning vector
        Returns:
            (B, d_feat) — FiLM-modulated, LayerNorm-stabilised EEG embedding
        """
        gamma = self.gamma_proj(bvp_emb)            # (B, d_feat)
        beta  = self.beta_proj(bvp_emb)             # (B, d_feat)
        return self.norm(gamma * eeg_emb + beta)    # (B, d_feat)


# ── quick sanity check ───────────────────────────────────────────────────────

if __name__ == "__main__":
    model    = MBInvBaseBiMamba(in_channels=20, num_classes=4,
                                d_model=64, n_layers=3, dropout=0.4)
    # Simulate a batch of 2 × 10-second windows at 256 Hz → 2560 samples
    x        = torch.randn(2, 20, 2560)
    n_params = sum(p.numel() for p in model.parameters())
    logits   = model(x)

    print(f"Input  shape : {x.shape}")
    print(f"Output shape : {logits.shape}")
    print(f"Parameters   : {n_params:,}")
    print(f"Mamba seq len: {2560 // 16}  (after conv stem stride ×16)")
