"""
EEGNet Feature Extractor — conv blocks only, no classification head.

Learns temporal filters and spatial filters from raw EEG,
outputs a feature sequence for downstream classifiers.

Paper: https://pubmed.ncbi.nlm.nih.gov/29932424/
"""

import torch
import torch.nn as nn


class Conv2dWithConstraint(nn.Module):
    """Conv2d with max-norm weight clamping."""
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, max_value=1.0, bias=False, groups=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride, padding, groups=groups, bias=bias)
        self.max_value = max_value

    def forward(self, x):
        self.conv.weight.data = torch.clamp(self.conv.weight.data,
                                            -self.max_value, self.max_value)
        return self.conv(x)


class EEGNetFeatureExtractor(nn.Module):
    """
    EEGNet convolutional blocks as a feature extractor.

    Input:  (batch, channels, timepoints)  e.g. (B, 4, 1024)
    Output: (batch, seq_len, feature_dim)  e.g. (B, 32, 128)
            where seq_len = timepoints // 32, feature_dim = F1 * D

    Args:
        num_electrodes: EEG channels (4 for Muse 2)
        datapoints:     time samples per window
        F1:             temporal filters
        D:              depth multiplier
        dropout:        dropout rate
    """
    def __init__(self, num_electrodes=4, datapoints=1024,
                 F1=32, D=4, dropout=0.4):
        super().__init__()
        self.F1 = F1
        self.D = D
        F2 = F1 * D
        self.feature_dim = F2
        self.seq_len = datapoints // 32

        # Block 1: Temporal convolution
        self.conv1 = nn.Conv2d(1, F1, (1, datapoints // 2),
                               padding='same', bias=False)
        self.bn1 = nn.BatchNorm2d(F1)

        # Block 1: Depthwise spatial convolution
        self.depth_conv = Conv2dWithConstraint(
            F1, F2, (num_electrodes, 1), bias=False, groups=F1
        )
        self.bn2 = nn.BatchNorm2d(F2)
        self.act1 = nn.ELU(inplace=True)
        self.pool1 = nn.AvgPool2d((1, 4), stride=4)
        self.drop1 = nn.Dropout(dropout)

        # Block 2: Separable convolution
        self.sep_conv_depth = nn.Conv2d(
            F2, F2, (1, 16), padding='same', bias=False, groups=F2
        )
        self.sep_conv_point = nn.Conv2d(F2, F2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(F2)
        self.act2 = nn.ELU(inplace=True)
        self.pool2 = nn.AvgPool2d((1, 8), stride=8)
        self.drop2 = nn.Dropout(dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_normal_(self.conv1.weight)
        nn.init.kaiming_normal_(self.depth_conv.conv.weight)
        nn.init.kaiming_normal_(self.sep_conv_depth.weight)
        nn.init.kaiming_normal_(self.sep_conv_point.weight)

    def forward(self, x):
        """
        Args:
            x: (batch, channels, timepoints) raw EEG
        Returns:
            features: (batch, seq_len, feature_dim) feature sequence
        """
        # (B, C, T) → (B, 1, C, T)
        x = x.unsqueeze(1)

        # Block 1: temporal + spatial
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.depth_conv(x)
        x = self.bn2(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.drop1(x)

        # Block 2: separable conv
        x = self.sep_conv_depth(x)
        x = self.sep_conv_point(x)
        x = self.bn3(x)
        x = self.act2(x)
        x = self.pool2(x)
        x = self.drop2(x)

        # (B, F2, 1, T') → (B, F2, T') → (B, T', F2)
        x = x.squeeze(2)
        x = x.permute(0, 2, 1)

        return x  # (B, seq_len, feature_dim)
