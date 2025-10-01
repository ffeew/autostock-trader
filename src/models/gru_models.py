"""
GRU-based models for time series prediction.

GRU is more efficient than LSTM with similar performance.
Includes residual connections and batch normalization.
"""

import torch
import torch.nn as nn
from src.models.base_model import BaseModel


class BasicGRU(BaseModel):
    """Basic GRU model."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, device)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        self.fc = nn.Linear(hidden_size, 1)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        output = self.fc(last_output)
        return output


class ResidualGRU(BaseModel):
    """GRU with residual connections."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 3,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, device)

        # Project input to hidden_size for residual connection
        self.input_projection = nn.Linear(input_size, hidden_size)

        self.gru_layers = nn.ModuleList([
            nn.GRU(
                input_size=hidden_size if i > 0 else hidden_size,
                hidden_size=hidden_size,
                num_layers=1,
                batch_first=True
            )
            for i in range(num_layers)
        ])

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project input
        x = self.input_projection(x)  # (batch, seq, hidden_size)

        # Apply GRU layers with residual connections
        for gru_layer in self.gru_layers:
            residual = x
            out, _ = gru_layer(x)
            out = self.dropout(out)
            x = self.layer_norm(out + residual)  # Residual connection

        # Take last time step
        last_output = x[:, -1, :]
        output = self.fc(last_output)
        return output


class BatchNormGRU(BaseModel):
    """GRU with batch normalization."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, device)

        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )

        # Batch normalization after GRU
        self.batch_norm = nn.BatchNorm1d(hidden_size)
        self.dropout = nn.Dropout(dropout)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size // 2),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]

        # Apply batch norm
        last_output = self.batch_norm(last_output)
        last_output = self.dropout(last_output)

        output = self.fc(last_output)
        return output


# Factory function
def create_gru_model(
    model_type: str,
    input_size: int,
    **kwargs
) -> BaseModel:
    """
    Create a GRU model by type.

    Args:
        model_type: One of ['basic', 'residual', 'batchnorm']
        input_size: Number of input features
        **kwargs: Additional arguments for the model

    Returns:
        Instantiated model
    """
    models = {
        'basic': BasicGRU,
        'residual': ResidualGRU,
        'batchnorm': BatchNormGRU
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from {list(models.keys())}"
        )

    return models[model_type](input_size=input_size, **kwargs)