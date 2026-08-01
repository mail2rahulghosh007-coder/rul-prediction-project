# src/phase9_transformer_model.py
# A lightweight Transformer encoder for RUL regression, kept intentionally
# small (few layers/heads) since CMAPSS FD001 is a small dataset and
# large Transformers tend to overfit on it.

import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    """
    Standard sinusoidal positional encoding (as in 'Attention is All You Need').
    Transformers have no built-in notion of sequence order (unlike an LSTM),
    so we inject position information into the input embeddings directly.
    """
    def __init__(self, d_model, max_len=100):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))  # shape (1, max_len, d_model)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]


class TransformerRULModel(nn.Module):
    def __init__(self, num_features, window_size, d_model=64, nhead=4,
                 num_layers=2, dim_feedforward=128, dropout=0.3):
        super().__init__()

        # Project raw sensor features (17-dim) into the Transformer's model
        # dimension (d_model) -- same idea as a word embedding, but for
        # continuous sensor values instead of tokens.
        self.input_projection = nn.Linear(num_features, d_model)

        self.pos_encoder = PositionalEncoding(d_model, max_len=window_size)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.dropout = nn.Dropout(dropout)

        # Regression head: pool over the time dimension, then predict RUL
        self.fc = nn.Linear(d_model, 1)

    def forward(self, x):
        # x: (batch, window_size, num_features)
        x = self.input_projection(x)          # (batch, window_size, d_model)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x)        # (batch, window_size, d_model)

        # Mean-pool across the time dimension (uses info from ALL cycles in
        # the window, not just the last one -- unlike our CNN-LSTM which
        # only used the LSTM's final timestep)
        pooled = x.mean(dim=1)                 # (batch, d_model)
        pooled = self.dropout(pooled)

        rul_prediction = self.fc(pooled)       # (batch, 1)
        return rul_prediction


if __name__ == "__main__":
    num_features = 17
    window_size = 30
    batch_size = 4

    model = TransformerRULModel(num_features, window_size)
    dummy_input = torch.randn(batch_size, window_size, num_features)
    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Total trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))