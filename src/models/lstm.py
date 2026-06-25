"""LSTM architecture — used in Phase 4 inference."""
import torch
import torch.nn as nn


class ThreatLSTM(nn.Module):
    def __init__(self, input_dim: int, hidden_size: int = 64,
                 num_layers: int = 2, dropout: float = 0.3):
        super().__init__()
        self.seq_len = 10
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 32), nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        return self.classifier(lstm_out[:, -1, :]).squeeze(-1)

    def predict_proba(self, x: torch.Tensor) -> torch.Tensor:
        """Return attack probability for a batch of sequences."""
        with torch.no_grad():
            return self(x)