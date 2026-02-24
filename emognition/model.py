"""
Dual-Branch EEGNet-BiLSTM with Cross-Attention Fusion.

Architecture:
    Raw EEG → EEGNet (learned features)  ─┐
                                           ├→ Cross-Attention → BiLSTM → Classes
    Raw EEG → Handcrafted Features (26/ch) ─┘
"""

import torch
import torch.nn as nn

from eegnet import EEGNetFeatureExtractor
from fusion import CrossAttentionFusion
from classifier import BiLSTMClassifier


class DualBranchModel(nn.Module):
    """
    Two-branch model with cross-attention fusion.

    Branch 1: EEGNet conv blocks → learned temporal-spatial features
    Branch 2: Precomputed handcrafted features (DE, PSD, Hjorth, etc.)
    Fusion:   Bidirectional cross-attention
    Classifier: BiLSTM with attention

    Args:
        num_electrodes: EEG channels (4)
        datapoints:     time samples per window (1024)
        num_classes:    output classes (4)
        hc_dim:         handcrafted feature dim per channel (26)
        F1, D:          EEGNet temporal filters and depth multiplier
        eeg_dropout:    dropout for EEGNet
        fusion_dim:     cross-attention projection dimension
        num_heads:      attention heads
        fusion_dropout: dropout in cross-attention
        lstm_hidden:    BiLSTM hidden size
        lstm_layers:    BiLSTM layers
        cls_dropout:    classifier dropout
    """
    def __init__(self, num_electrodes=4, datapoints=1024, num_classes=4,
                 hc_dim=26,
                 F1=32, D=4, eeg_dropout=0.3,
                 fusion_dim=128, num_heads=4, fusion_dropout=0.1,
                 lstm_hidden=128, lstm_layers=2, cls_dropout=0.4):
        super().__init__()

        # Branch 1: EEGNet feature extractor
        self.eegnet = EEGNetFeatureExtractor(
            num_electrodes=num_electrodes,
            datapoints=datapoints,
            F1=F1, D=D, dropout=eeg_dropout
        )
        eeg_feat_dim = self.eegnet.feature_dim  # F1 * D

        # Branch 2: projection for handcrafted features (already precomputed)
        # No learnable extraction — just passes through to fusion

        # Cross-Attention Fusion
        self.fusion = CrossAttentionFusion(
            dim_a=eeg_feat_dim,
            dim_b=hc_dim,
            d_model=fusion_dim,
            num_heads=num_heads,
            dropout=fusion_dropout
        )

        # Classifier: BiLSTM on fused sequence
        self.classifier = BiLSTMClassifier(
            feature_dim=fusion_dim,
            hidden_dim=lstm_hidden,
            num_layers=lstm_layers,
            num_classes=num_classes,
            dropout=cls_dropout
        )

    def forward(self, x_raw, x_hc):
        """
        Args:
            x_raw: (B, 4, 1024) raw EEG signal
            x_hc:  (B, 4, 26)   precomputed handcrafted features
        Returns:
            logits: (B, num_classes)
        """
        # Branch 1: EEGNet → learned features
        feat_learned = self.eegnet(x_raw)       # (B, T_eeg, eeg_dim)

        # Branch 2: handcrafted features (already computed)
        feat_hc = x_hc                           # (B, 4, 26)

        # Cross-attention fusion
        fused = self.fusion(feat_learned, feat_hc)  # (B, T_eeg+4, fusion_dim)

        # Classify
        logits = self.classifier(fused)          # (B, num_classes)
        return logits
