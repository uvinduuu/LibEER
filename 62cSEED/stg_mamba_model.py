"""
STG-Mamba: Spatio-Temporal Graph Mamba for EEG Emotion Recognition.

A novel architecture that jointly captures:
  1. SPATIAL  — Learnable graph convolution over electrode topology
  2. TEMPORAL — State-space model (Mamba) for sequential dynamics
  3. SPECTRAL — DE-LDS frequency-band features as input

Architecture:
  Input: (batch, T, n_channels, n_bands)
    │
    ├── Per-Timestep Graph Convolution (spatial)
    │     Learnable adjacency matrix over electrodes
    │     Chebyshev graph conv at each time step
    │     → (batch, T, n_channels, d_graph)
    │
    ├── Channel-Wise Mamba (temporal per electrode)
    │     Independent SSM per electrode channel
    │     → (batch, T, n_channels, d_mamba)
    │
    ├── Spatial-Temporal Fusion
    │     Concatenate graph + mamba features
    │     → (batch, T, d_fused)
    │
    ├── Global Mamba Block (temporal aggregation)
    │     SSM over fused sequence
    │     → (batch, d_fused)
    │
    └── Classification Head
          → (batch, n_classes)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────
# Graph Convolution Components (inspired by DGCNN)
# ─────────────────────────────────────────────────

class LearnableGraphConv(nn.Module):
    """Chebyshev graph convolution with learnable adjacency.
    
    Operates on (batch, n_nodes, in_features) per time step.
    The adjacency matrix is learned end-to-end.
    """
    def __init__(self, n_nodes, in_features, out_features, k=2):
        super().__init__()
        self.n_nodes = n_nodes
        self.k = k
        
        # Learnable adjacency
        self.adj = nn.Parameter(torch.randn(n_nodes, n_nodes) * 0.1)
        self.adj_bias = nn.Parameter(torch.zeros(1))
        
        # Chebyshev filter weights
        self.weight = nn.Parameter(torch.empty(k * in_features, out_features))
        nn.init.xavier_uniform_(self.weight)
        
        # Bias-ReLU (from DGCNN)
        self.bias = nn.Parameter(torch.zeros(1, 1, out_features))
        
    def _laplacian(self, adj):
        """Compute normalized Laplacian from adjacency."""
        d = adj.sum(dim=1)
        d_inv_sqrt = 1.0 / torch.sqrt(d + 1e-5)
        D = torch.diag_embed(d_inv_sqrt)
        I = torch.eye(adj.size(0), device=adj.device)
        return I - D @ adj @ D
    
    def forward(self, x):
        """x: (batch, n_nodes, in_features) → (batch, n_nodes, out_features)"""
        adj = F.relu(self.adj + self.adj_bias)
        lap = self._laplacian(adj)
        
        # Chebyshev polynomials
        T0 = torch.ones_like(x)
        cheb = [T0]
        if self.k >= 2:
            T1 = torch.matmul(lap, x)
            cheb.append(T1)
        for i in range(2, self.k):
            Tk = 2 * torch.matmul(lap, cheb[-1]) - cheb[-2]
            cheb.append(Tk)
        
        # Stack and filter: (batch, n_nodes, k * in_features)
        cheb_cat = torch.cat(cheb, dim=-1)
        out = torch.matmul(cheb_cat, self.weight)
        return F.relu(out + self.bias)


class SpatialGraphBlock(nn.Module):
    """Apply graph convolution at each time step independently."""
    def __init__(self, n_channels, in_bands, d_graph, k=2, dropout=0.2):
        super().__init__()
        self.gcn = LearnableGraphConv(n_channels, in_bands, d_graph, k=k)
        self.norm = nn.LayerNorm(d_graph)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """x: (batch, T, n_channels, n_bands) → (batch, T, n_channels, d_graph)"""
        B, T, C, F = x.shape
        # Reshape to apply GCN per timestep
        x_flat = x.reshape(B * T, C, F)
        out = self.gcn(x_flat)          # (B*T, C, d_graph)
        out = self.norm(out)
        out = self.dropout(out)
        return out.reshape(B, T, C, -1)


# ─────────────────────────────────────────────────
# Mamba Block (State-Space Model)
# ─────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """Selective state-space block with gated convolution."""
    def __init__(self, d_model, d_state=16, d_conv=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.norm = nn.LayerNorm(d_model)
        
        # Depthwise conv for local context
        self.conv1d = nn.Conv1d(
            d_model, d_model, kernel_size=d_conv, 
            padding=d_conv - 1, groups=d_model
        )
        
        # Gating mechanism
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()))
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.dt_proj = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.Softplus()
        )
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """x: (batch, T, d_model)"""
        residual = x
        x = self.norm(x)
        
        # Input projection + gating
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        
        # Depthwise conv (causal)
        x_conv = self.conv1d(x_main.transpose(1, 2))[:, :, :x.size(1)]
        x_conv = x_conv.transpose(1, 2)
        x_main = F.silu(x_conv)
        
        # SSM
        A = -torch.exp(self.A_log)  # (d_state,)
        dt = self.dt_proj(x_main)   # (B, T, D)
        B = self.B_proj(x_main)     # (B, T, N)
        C = self.C_proj(x_main)     # (B, T, N)
        
        # Discretize and scan
        dA = torch.exp(dt.unsqueeze(-1) * A)  # (B, T, D, N)
        dB_x = dt.unsqueeze(-1) * B.unsqueeze(2) * x_main.unsqueeze(-1)  # (B, T, D, N)
        
        # Sequential scan (more accurate than cumsum)
        B_dim, T_dim, D_dim, N_dim = dA.shape
        h = torch.zeros(B_dim, D_dim, N_dim, device=x.device)
        ys = []
        for t in range(T_dim):
            h = dA[:, t] * h + dB_x[:, t]
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)  # (B, D)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)  # (B, T, D)
        
        # Gate and project
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        
        return y + residual


# ─────────────────────────────────────────────────
# Channel-Wise Mamba
# ─────────────────────────────────────────────────

class ChannelWiseMamba(nn.Module):
    """Apply Mamba independently per electrode channel, then fuse.
    
    Each electrode's temporal evolution is modeled by its own Mamba,
    preserving spatial structure.
    """
    def __init__(self, d_channel, d_state=16, n_channels=4, dropout=0.2):
        super().__init__()
        self.n_channels = n_channels
        # Shared Mamba block across channels (weight sharing for efficiency)
        self.mamba = MambaBlock(d_channel, d_state=d_state, dropout=dropout)
        self.norm = nn.LayerNorm(d_channel)
        
    def forward(self, x):
        """x: (batch, T, n_channels, d_channel) → (batch, T, n_channels, d_channel)"""
        B, T, C, D = x.shape
        # Reshape: treat each channel as a separate sequence
        x_flat = x.permute(0, 2, 1, 3).reshape(B * C, T, D)  # (B*C, T, D)
        out = self.mamba(x_flat)  # (B*C, T, D)
        out = self.norm(out)
        out = out.reshape(B, C, T, D).permute(0, 2, 1, 3)  # (B, T, C, D)
        return out


# ─────────────────────────────────────────────────
# STG-Mamba: Full Architecture
# ─────────────────────────────────────────────────

class STGMamba(nn.Module):
    """
    Spatio-Temporal Graph Mamba for EEG Emotion Recognition.
    
    Combines:
      1. Spatial graph convolution (electrode topology)
      2. Channel-wise Mamba (per-electrode temporal dynamics)
      3. Global Mamba (cross-channel temporal aggregation)
    
    Args:
        n_channels: Number of EEG electrodes (e.g., 4)
        n_bands: Number of frequency bands in DE-LDS features (5)
        d_graph: Hidden dim for graph convolution
        d_mamba: Hidden dim for Mamba blocks
        d_state: SSM state dimension
        n_global_layers: Number of global Mamba layers
        num_classes: Number of emotion classes
        dropout: Dropout rate
        gcn_k: Chebyshev polynomial order
    """
    def __init__(
        self,
        n_channels=4,
        n_bands=5,
        d_graph=32,
        d_mamba=64,
        d_state=16,
        n_global_layers=3,
        num_classes=4,
        dropout=0.3,
        gcn_k=2,
    ):
        super().__init__()
        self.n_channels = n_channels
        
        # Stage 1: Spatial Graph Convolution
        self.spatial_block = SpatialGraphBlock(
            n_channels, n_bands, d_graph, k=gcn_k, dropout=dropout
        )
        
        # Stage 2: Channel-Wise Mamba (temporal per electrode)
        self.channel_mamba = ChannelWiseMamba(
            d_channel=d_graph, d_state=d_state,
            n_channels=n_channels, dropout=dropout
        )
        
        # Stage 3: Fusion — flatten channels into single feature vector per timestep
        d_fused = n_channels * d_graph
        self.fusion = nn.Sequential(
            nn.Linear(d_fused, d_mamba),
            nn.LayerNorm(d_mamba),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Stage 4: Global Mamba Blocks (cross-channel temporal)
        self.global_blocks = nn.ModuleList([
            MambaBlock(d_mamba, d_state=d_state, dropout=dropout)
            for _ in range(n_global_layers)
        ])
        
        # Stage 5: Classification
        self.final_norm = nn.LayerNorm(d_mamba)
        self.classifier = nn.Sequential(
            nn.Linear(d_mamba, d_mamba // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_mamba // 2, num_classes)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x):
        """
        x: (batch, T, n_channels, n_bands)
           e.g., (64, 50, 4, 5)
        
        Returns: (batch, num_classes)
        """
        # Stage 1: Spatial graph conv at each timestep
        # (B, T, C, bands) → (B, T, C, d_graph)
        x_spatial = self.spatial_block(x)
        
        # Stage 2: Channel-wise temporal Mamba
        # (B, T, C, d_graph) → (B, T, C, d_graph)
        x_temporal = self.channel_mamba(x_spatial)
        
        # Residual connection (spatial + temporal)
        x_combined = x_spatial + x_temporal
        
        # Stage 3: Flatten channels → fuse
        B, T, C, D = x_combined.shape
        x_flat = x_combined.reshape(B, T, C * D)  # (B, T, C*d_graph)
        x_fused = self.fusion(x_flat)              # (B, T, d_mamba)
        
        # Stage 4: Global temporal Mamba
        for block in self.global_blocks:
            x_fused = block(x_fused)
        
        # Global average pooling over time
        x_fused = self.final_norm(x_fused)
        x_pool = x_fused.mean(dim=1)  # (B, d_mamba)
        
        # Stage 5: Classify
        return self.classifier(x_pool)
    
    def get_attention_weights(self):
        """Return the learned adjacency matrix for visualization."""
        adj = F.relu(
            self.spatial_block.gcn.adj + self.spatial_block.gcn.adj_bias
        )
        return adj.detach().cpu().numpy()


def count_parameters(model):
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    # Quick test
    model = STGMamba(
        n_channels=4, n_bands=5, d_graph=32, d_mamba=64,
        d_state=16, n_global_layers=3, num_classes=4, dropout=0.3
    )
    print(f"STG-Mamba parameters: {count_parameters(model):,}")
    
    # Test forward pass
    x = torch.randn(8, 50, 4, 5)  # (batch=8, T=50, 4 channels, 5 bands)
    out = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {out.shape}")
    
    # Show learned adjacency
    adj = model.get_attention_weights()
    print(f"\nLearned adjacency (4×4):\n{adj}")
