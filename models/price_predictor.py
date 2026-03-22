"""
RSR — Neural Network Architecture
LSTM-based binary classifier: predicts probability that tomorrow's price is UP.
Input:  (batch, lookback_days, num_features)
Output: (batch,) — probability in [0, 1]
"""
import torch
import torch.nn as nn


class RSRPredictor(nn.Module):
    """
    Two-layer LSTM with a fully-connected classification head.
    """

    def __init__(
        self,
        num_features: int,
        hidden_size:  int   = 128,
        num_layers:   int   = 2,
        dropout:      float = 0.3,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers  = num_layers

        # LSTM stack
        self.lstm = nn.LSTM(
            input_size  = num_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
            dropout     = dropout if num_layers > 1 else 0.0,
        )

        # Attention over time steps (optional lightweight attention)
        self.attention = nn.Linear(hidden_size, 1)

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout / 2),
            nn.Linear(32, 1),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, num_features)
        returns: (batch,) probabilities
        """
        lstm_out, _ = self.lstm(x)          # (batch, seq_len, hidden)

        # Soft attention: weight each timestep
        attn_weights = torch.softmax(self.attention(lstm_out), dim=1)  # (batch, seq, 1)
        context = (lstm_out * attn_weights).sum(dim=1)                 # (batch, hidden)

        return self.classifier(context).squeeze(-1)                    # (batch,)


class RSRPredictorSimple(nn.Module):
    """
    Simpler version without attention — faster training, slightly lower accuracy.
    Use this on low-resource machines.
    """

    def __init__(self, num_features: int, hidden_size: int = 64, num_layers: int = 1, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size  = num_features,
            hidden_size = hidden_size,
            num_layers  = num_layers,
            batch_first = True,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)
