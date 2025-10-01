"""
LSTM-based models for time series prediction.

Includes:
- Basic LSTM
- Stacked LSTM
- Bidirectional LSTM
- LSTM with Attention
"""

import torch
import torch.nn as nn
from src.models.base_model import BaseModel


class BasicLSTM(BaseModel):
    """Basic LSTM model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 1,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, device)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # Take output from last time step
        last_output = lstm_out[:, -1, :]
        # Predict
        output = self.fc(last_output)
        return output


class StackedLSTM(BaseModel):
    """Stacked LSTM with multiple layers."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, device)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output


class BidirectionalLSTM(BaseModel):
    """Bidirectional LSTM model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, device)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True,
            bidirectional=True
        )

        # Bidirectional doubles the hidden size
        self.fc = nn.Linear(hidden_size * 2, 1)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output


class AttentionLSTM(BaseModel):
    """LSTM with attention mechanism."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, device)

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Attention layer
        self.attention = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        lstm_out, _ = self.lstm(x)
        # lstm_out shape: (batch_size, sequence_length, hidden_size)

        # Compute attention weights
        attention_weights = self.attention(lstm_out)  # (batch_size, sequence_length, 1)
        attention_weights = torch.softmax(attention_weights, dim=1)

        # Apply attention weights
        context_vector = torch.sum(attention_weights * lstm_out, dim=1)  # (batch_size, hidden_size)

        # Predict
        output = self.fc(context_vector)
        return output


# Factory function
def create_lstm_model(
    model_type: str,
    input_size: int,
    **kwargs
) -> BaseModel:
    """
    Create an LSTM model by type.

    Args:
        model_type: One of ['basic', 'stacked', 'bidirectional', 'attention']
        input_size: Number of input features
        **kwargs: Additional arguments for the model

    Returns:
        Instantiated model
    """
    models = {
        'basic': BasicLSTM,
        'stacked': StackedLSTM,
        'bidirectional': BidirectionalLSTM,
        'attention': AttentionLSTM
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from {list(models.keys())}"
        )

    return models[model_type](input_size=input_size, **kwargs)