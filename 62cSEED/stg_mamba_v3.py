"""
LS-STG-Mamba v3: Latent-Space Spatio-Temporal Graph Mamba.

Novel architecture for EEG Emotion Recognition that combines:
  1. VARIATIONAL LATENT ENCODER — Compresses DE-LDS features into a
     regularized latent space (bottleneck = anti-overfitting)
  2. SPATIAL GRAPH CONV — Learnable adjacency over electrode topology
  3. TEMPORAL MAMBA — State-space model for sequential dynamics
  4. AUXILIARY DECODER — Reconstructs DE-LDS for regularization

Multi-task loss: L = L_cls + α·L_recon + β·L_kl

Key anti-overfitting mechanisms:
  - Variational bottleneck (KL regularization)
  - Reconstruction constraint (multi-task)
  - Scheduled dropout (increases during training)
  - Compact latent representation

Citation-ready description:
  "We propose LS-STG-Mamba, a variational latent-space approach for
   EEG emotion recognition that compresses spectral features through
   a regularized bottleneck before graph-guided state-space modeling.
   The variational constraint prevents overfitting while an auxiliary
   reconstruction objective preserves discriminative information."
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────
# Variational Latent Encoder
# ─────────────────────────────────────────────────

class VariationalEncoder(nn.Module):
    """Projects per-channel DE-LDS features into a regularized latent space.
    
    Input:  (batch, T, n_channels, n_bands)  e.g., (B, T, 4, 5)
    Output: (batch, T, n_channels, d_latent), mu, logvar
    
    The bottleneck (n_bands → d_latent) forces compression.
    KL divergence keeps the latent space smooth.
    """
    def __init__(self, n_bands=5, d_latent=8, d_hidden=32):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(n_bands, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
        )
        self.mu_proj = nn.Linear(d_hidden, d_latent)
        self.logvar_proj = nn.Linear(d_hidden, d_latent)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        return mu  # deterministic at inference
    
    def forward(self, x):
        """x: (B, T, C, bands) → z: (B, T, C, d_latent), mu, logvar"""
        h = self.encoder(x)
        mu = self.mu_proj(h)
        logvar = self.logvar_proj(h)
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar


class Decoder(nn.Module):
    """Reconstructs DE-LDS features from latent space (training only).
    
    This auxiliary task regularizes the latent representation.
    """
    def __init__(self, d_latent=8, d_hidden=32, n_bands=5):
        super().__init__()
        self.decoder = nn.Sequential(
            nn.Linear(d_latent, d_hidden),
            nn.GELU(),
            nn.LayerNorm(d_hidden),
            nn.Linear(d_hidden, d_hidden),
            nn.GELU(),
            nn.Linear(d_hidden, n_bands),
        )
    
    def forward(self, z):
        """z: (B, T, C, d_latent) → x_recon: (B, T, C, n_bands)"""
        return self.decoder(z)


# ─────────────────────────────────────────────────
# Graph Convolution (operates in latent space)
# ─────────────────────────────────────────────────

class LatentGraphConv(nn.Module):
    """Chebyshev graph convolution with learnable adjacency in latent space."""
    def __init__(self, n_nodes, d_latent, d_out, k=2):
        super().__init__()
        self.n_nodes = n_nodes
        self.k = k
        
        # Learnable adjacency — initialized with electrode prior
        self.adj = nn.Parameter(torch.zeros(n_nodes, n_nodes))
        self.adj_bias = nn.Parameter(torch.zeros(1))
        
        # Chebyshev filter
        self.weight = nn.Parameter(torch.empty(k * d_latent, d_out))
        nn.init.xavier_uniform_(self.weight)
        self.bias = nn.Parameter(torch.zeros(1, 1, d_out))
        
    def forward(self, x):
        """x: (B, n_nodes, d_latent) → (B, n_nodes, d_out)"""
        adj = F.relu(self.adj + self.adj_bias)
        
        # Normalized Laplacian
        d = adj.sum(1)
        d_inv = 1.0 / torch.sqrt(d + 1e-5)
        D = torch.diag_embed(d_inv)
        I = torch.eye(self.n_nodes, device=x.device)
        lap = I - D @ adj @ D
        
        # Chebyshev polynomials
        T0 = torch.ones_like(x)
        cheb = [T0]
        if self.k >= 2:
            T1 = torch.matmul(lap, x)
            cheb.append(T1)
        for i in range(2, self.k):
            Tk = 2 * torch.matmul(lap, cheb[-1]) - cheb[-2]
            cheb.append(Tk)
        
        out = torch.matmul(torch.cat(cheb, dim=-1), self.weight)
        return F.relu(out + self.bias)


class SpatialGraphBlock(nn.Module):
    """Graph conv at each timestep in latent space."""
    def __init__(self, n_channels, d_latent, d_graph, k=2, dropout=0.2):
        super().__init__()
        self.gcn = LatentGraphConv(n_channels, d_latent, d_graph, k=k)
        self.norm = nn.LayerNorm(d_graph)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """x: (B, T, C, d_latent) → (B, T, C, d_graph)"""
        B, T, C, D = x.shape
        x_flat = x.reshape(B * T, C, D)
        out = self.gcn(x_flat)
        out = self.norm(out)
        out = self.dropout(out)
        return out.reshape(B, T, C, -1)


# ─────────────────────────────────────────────────
# Mamba Block (same as v2, optimized)
# ─────────────────────────────────────────────────

class MambaBlock(nn.Module):
    """Selective state-space block."""
    def __init__(self, d_model, d_state=16, d_conv=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        
        self.norm = nn.LayerNorm(d_model)
        self.conv1d = nn.Conv1d(
            d_model, d_model, kernel_size=d_conv,
            padding=d_conv - 1, groups=d_model
        )
        self.in_proj = nn.Linear(d_model, d_model * 2)
        self.out_proj = nn.Linear(d_model, d_model)
        
        # SSM parameters
        self.A_log = nn.Parameter(torch.log(torch.arange(1, d_state + 1).float()))
        self.B_proj = nn.Linear(d_model, d_state)
        self.C_proj = nn.Linear(d_model, d_state)
        self.dt_proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.Softplus()
        )
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        x = self.norm(x)
        
        xz = self.in_proj(x)
        x_main, z = xz.chunk(2, dim=-1)
        
        # Causal conv
        x_conv = self.conv1d(x_main.transpose(1, 2))[:, :, :x.size(1)]
        x_main = F.silu(x_conv.transpose(1, 2))
        
        # SSM
        A = -torch.exp(self.A_log)
        dt = self.dt_proj(x_main)
        B = self.B_proj(x_main)
        C = self.C_proj(x_main)
        
        dA = torch.exp(dt.unsqueeze(-1) * A)
        dB_x = dt.unsqueeze(-1) * B.unsqueeze(2) * x_main.unsqueeze(-1)
        
        # Sequential scan
        B_dim, T_dim, D_dim, N_dim = dA.shape
        h = torch.zeros(B_dim, D_dim, N_dim, device=x.device)
        ys = []
        for t in range(T_dim):
            h = dA[:, t] * h + dB_x[:, t]
            y_t = (h * C[:, t].unsqueeze(1)).sum(-1)
            ys.append(y_t)
        y = torch.stack(ys, dim=1)
        
        y = y * F.silu(z)
        y = self.out_proj(y)
        return self.dropout(y) + residual


class ChannelWiseMamba(nn.Module):
    """Mamba per electrode channel, shared weights."""
    def __init__(self, d_input, d_state=16, n_channels=4, dropout=0.2):
        super().__init__()
        self.n_channels = n_channels
        self.mamba = MambaBlock(d_input, d_state=d_state, dropout=dropout)
        self.norm = nn.LayerNorm(d_input)
        
    def forward(self, x):
        """x: (B, T, C, D) → (B, T, C, D)"""
        B, T, C, D = x.shape
        x_flat = x.permute(0, 2, 1, 3).reshape(B * C, T, D)
        out = self.mamba(x_flat)
        out = self.norm(out)
        return out.reshape(B, C, T, D).permute(0, 2, 1, 3)


# ─────────────────────────────────────────────────
# LS-STG-Mamba: Full Architecture
# ─────────────────────────────────────────────────

class LSSTGMamba(nn.Module):
    """
    Latent-Space Spatio-Temporal Graph Mamba.
    
    Pipeline:
      DE-LDS → VariationalEncoder → SpatialGraphConv → ChannelMamba
             → GlobalMamba → Classifier
             ↘ Decoder (auxiliary, training only)
    
    Args:
        n_channels: Number of EEG electrodes (e.g., 4)
        n_bands: DE-LDS frequency bands (5)
        d_latent: Latent space dimension per channel
        d_graph: Graph conv output dim
        d_mamba: Global Mamba hidden dim
        d_state: SSM state dimension
        n_global_layers: Number of global Mamba layers
        num_classes: Emotion classes
        dropout: Base dropout rate
    """
    def __init__(
        self,
        n_channels=4,
        n_bands=5,
        d_latent=8,
        d_graph=24,
        d_mamba=64,
        d_state=16,
        n_global_layers=3,
        num_classes=4,
        dropout=0.35,
    ):
        super().__init__()
        self.n_channels = n_channels
        self.n_bands = n_bands
        
        # Stage 1: Variational Latent Encoder
        self.encoder = VariationalEncoder(n_bands, d_latent, d_hidden=32)
        
        # Stage 2: Auxiliary Decoder (regularization)
        self.decoder = Decoder(d_latent, d_hidden=32, n_bands=n_bands)
        
        # Stage 3: Spatial Graph Conv (in latent space)
        self.spatial = SpatialGraphBlock(
            n_channels, d_latent, d_graph, k=2, dropout=dropout
        )
        
        # Stage 4: Channel-Wise Mamba
        self.channel_mamba = ChannelWiseMamba(
            d_graph, d_state=d_state, n_channels=n_channels, dropout=dropout
        )
        
        # Stage 5: Fusion + Global Mamba
        d_fused = n_channels * d_graph
        self.fusion = nn.Sequential(
            nn.Linear(d_fused, d_mamba),
            nn.LayerNorm(d_mamba),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.global_blocks = nn.ModuleList([
            MambaBlock(d_mamba, d_state=d_state, dropout=dropout)
            for _ in range(n_global_layers)
        ])
        
        # Stage 6: Classifier
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
    
    def init_adjacency(self, adj_matrix):
        """Initialize adjacency with electrode distance prior."""
        with torch.no_grad():
            self.spatial.gcn.adj.data.copy_(adj_matrix)
            self.spatial.gcn.adj_bias.data.fill_(0.0)
    
    def forward(self, x, return_losses=False):
        """
        x: (batch, T, n_channels, n_bands)
        
        Returns:
            logits: (batch, num_classes)
            recon_loss: scalar (if return_losses=True)
            kl_loss: scalar (if return_losses=True)
        """
        # Stage 1: Encode to latent space
        z, mu, logvar = self.encoder(x)  # (B, T, C, d_latent)
        
        # Stage 2: Reconstruct (auxiliary — only compute loss during training)
        recon_loss = torch.tensor(0.0, device=x.device)
        kl_loss = torch.tensor(0.0, device=x.device)
        if return_losses:
            x_recon = self.decoder(z)
            recon_loss = F.mse_loss(x_recon, x)
            kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Stage 3: Spatial graph conv (in latent space)
        x_spatial = self.spatial(z)  # (B, T, C, d_graph)
        
        # Stage 4: Channel-wise Mamba
        x_temporal = self.channel_mamba(x_spatial)
        x_combined = x_spatial + x_temporal  # residual
        
        # Stage 5: Fuse + Global Mamba
        B, T, C, D = x_combined.shape
        x_flat = x_combined.reshape(B, T, C * D)
        x_fused = self.fusion(x_flat)
        
        for block in self.global_blocks:
            x_fused = block(x_fused)
        
        # Pool + classify
        x_fused = self.final_norm(x_fused)
        x_pool = x_fused.mean(dim=1)
        logits = self.classifier(x_pool)
        
        if return_losses:
            return logits, recon_loss, kl_loss
        return logits
    
    def get_adjacency(self):
        """Return learned adjacency for visualization."""
        adj = F.relu(self.spatial.gcn.adj + self.spatial.gcn.adj_bias)
        return adj.detach().cpu().numpy()
    
    def get_latent_stats(self):
        """Get encoder stats for monitoring."""
        return {
            'encoder_weight_norm': sum(
                p.norm().item() for p in self.encoder.parameters()
            ),
        }


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == '__main__':
    model = LSSTGMamba(
        n_channels=4, n_bands=5, d_latent=8, d_graph=24,
        d_mamba=64, d_state=16, n_global_layers=3, num_classes=4
    )
    print(f"LS-STG-Mamba parameters: {count_parameters(model):,}")
    
    x = torch.randn(8, 10, 4, 5)
    logits = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {logits.shape}")
    
    logits, recon_loss, kl_loss = model(x, return_losses=True)
    print(f"Recon loss: {recon_loss:.4f}")
    print(f"KL loss:    {kl_loss:.4f}")
    
    adj = model.get_adjacency()
    print(f"Adjacency:\n{adj}")
