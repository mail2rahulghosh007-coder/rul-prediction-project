# src/phase5_model.py
# UPDATED: added dropout regularization to reduce overfitting on this small dataset

import torch
import torch.nn as nn

class CNN_LSTM_Model(nn.Module):
    def __init__(self, num_features, window_size, dropout=0.3):
        super(CNN_LSTM_Model, self).__init__()

        # 1D-CNN layer: slides along the time axis, looking at 3 cycles at a time
        self.conv1 = nn.Conv1d(
            in_channels=num_features,
            out_channels=32,
            kernel_size=3,
            stride=1,
            padding=1
        )
        self.relu = nn.ReLU()

        # Dropout after CNN feature extraction -- randomly zeroes some channels
        # during training so the model can't over-rely on any single sensor pattern
        self.conv_dropout = nn.Dropout(dropout)

        # LSTM layer: reads the CNN's output sequence and learns the trend over time
        self.lstm = nn.LSTM(
            input_size=32,
            hidden_size=64,
            num_layers=1,
            batch_first=True
        )

        # Dropout on LSTM's final representation before the regression head
        self.lstm_dropout = nn.Dropout(dropout)

        # Final layer: turns LSTM's output into a single RUL number
        self.fc = nn.Linear(64, 1)

    def forward(self, x):
        # x comes in as (batch, 30, 17) -> Conv1d wants (batch, 17, 30)
        x = x.permute(0, 2, 1)

        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv_dropout(x)

        # back to (batch, 30, 32) for the LSTM
        x = x.permute(0, 2, 1)

        lstm_out, (hidden, cell) = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        last_output = self.lstm_dropout(last_output)

        rul_prediction = self.fc(last_output)
        return rul_prediction


# Quick sanity check
if __name__ == "__main__":
    num_features = 17
    window_size = 30
    batch_size = 4

    model = CNN_LSTM_Model(num_features, window_size)
    dummy_input = torch.randn(batch_size, window_size, num_features)
    output = model(dummy_input)

    print("Input shape:", dummy_input.shape)
    print("Output shape:", output.shape)
    print("Total trainable params:", sum(p.numel() for p in model.parameters() if p.requires_grad))