"""
BiLSTM Classifier with Attention for EEG emotion recognition.

Takes a feature sequence from EEGNet and classifies it.
Uses bidirectional LSTM + attention pooling for temporal aggregation.
"""

import torch
import torch.nn as nn


class Attention(nn.Module):
    """Attention mechanism to pool over temporal dimension."""
    def __init__(self, input_dim):
        super().__init__()
        self.attn = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, dim)
        Returns:
            pooled: (batch, dim)
        """
        scores = self.attn(x)                 # (B, T, 1)
        weights = torch.softmax(scores, dim=1)  # (B, T, 1)
        pooled = (weights * x).sum(dim=1)     # (B, dim)
        return pooled


class BiLSTMClassifier(nn.Module):
    """
    Bidirectional LSTM classifier with attention.

    Input:  (batch, seq_len, feature_dim)  — from EEGNet feature extractor
    Output: (batch, num_classes)

    Args:
        feature_dim:  input feature dimension from EEGNet (F1 * D)
        hidden_dim:   LSTM hidden size
        num_layers:   number of LSTM layers
        num_classes:  number of output classes
        dropout:      dropout rate
    """
    def __init__(self, feature_dim=128, hidden_dim=128,
                 num_layers=2, num_classes=4, dropout=0.4):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0
        )

        lstm_out_dim = hidden_dim * 2  # bidirectional

        self.layer_norm = nn.LayerNorm(lstm_out_dim)
        self.dropout = nn.Dropout(dropout)
        self.attention = Attention(lstm_out_dim)

        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )

        self._init_weights()

    def _init_weights(self):
        for name, param in self.lstm.named_parameters():
            if 'weight_ih' in name:
                nn.init.xavier_uniform_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
            elif 'bias' in name:
                nn.init.zeros_(param)
        for m in self.classifier.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Args:
            x: (batch, seq_len, feature_dim)
        Returns:
            logits: (batch, num_classes)
        """
        # BiLSTM
        h, _ = self.lstm(x)             # (B, T, 2*hidden)
        h = self.layer_norm(h)          # (B, T, 2*hidden)
        h = self.dropout(h)

        # Attention pooling over time
        pooled = self.attention(h)      # (B, 2*hidden)

        # Classification
        logits = self.classifier(pooled)  # (B, num_classes)
        return logits
