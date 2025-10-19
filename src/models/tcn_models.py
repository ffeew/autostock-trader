"""
Temporal Convolutional Network (TCN) models.

TCN uses dilated causal convolutions for parallel processing of sequences.
Excellent for minute-level data due to efficient computation.
"""

import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from src.models.base_model import BaseModel


class Chomp1d(nn.Module):
    """Removes trailing padding from conv output to maintain causality."""

    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    """
    Temporal block for TCN.

    Consists of two dilated causal convolutions with residual connection.
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        padding: int,
        dropout: float = 0.2
    ):
        super().__init__()

        self.conv1 = weight_norm(nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        ))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(
            self.conv1, self.chomp1, self.relu1, self.dropout1,
            self.conv2, self.chomp2, self.relu2, self.dropout2
        )

        # Residual connection
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class BasicTCN(BaseModel):
    """Basic Temporal Convolutional Network."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        prediction_steps: int = 30,
        target_feature_idx: int = 3
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, device, prediction_steps, target_feature_idx)

        layers = []
        num_levels = num_layers
        num_channels = [hidden_size] * num_levels

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]

            layers += [TemporalBlock(
                in_channels, out_channels, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            )]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)
        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        # TCN expects: (batch_size, input_size, sequence_length)
        x = x.transpose(1, 2)

        # Apply TCN
        x = self.network(x)

        # Take last time step
        x = x[:, :, -1]

        # Predict
        output = self.fc(x)
        return output


class MultiScaleTCN(BaseModel):
    """Multi-scale TCN with parallel receptive fields."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        kernel_sizes: list = [3, 5, 7],
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        prediction_steps: int = 30,
        target_feature_idx: int = 3
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, device, prediction_steps, target_feature_idx)

        self.scales = nn.ModuleList()

        for kernel_size in kernel_sizes:
            layers = []
            num_levels = num_layers
            num_channels = [hidden_size] * num_levels

            for i in range(num_levels):
                dilation_size = 2 ** i
                in_channels = input_size if i == 0 else num_channels[i-1]
                out_channels = num_channels[i]

                layers += [TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation_size,
                    padding=(kernel_size-1) * dilation_size,
                    dropout=dropout
                )]

            self.scales.append(nn.Sequential(*layers))

        # Combine multi-scale features
        self.fc = nn.Sequential(
            nn.Linear(hidden_size * len(kernel_sizes), hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, sequence_length)

        # Apply each scale
        outputs = []
        for scale in self.scales:
            out = scale(x)
            out = out[:, :, -1]  # Take last time step
            outputs.append(out)

        # Concatenate multi-scale features
        combined = torch.cat(outputs, dim=1)

        # Predict
        output = self.fc(combined)
        return output


class ResidualTCN(BaseModel):
    """TCN with additional residual connections between blocks."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 6,
        kernel_size: int = 3,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        prediction_steps: int = 30,
        target_feature_idx: int = 3
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, device, prediction_steps, target_feature_idx)

        # Initial projection
        self.input_projection = nn.Conv1d(input_size, hidden_size, 1)

        # Temporal blocks
        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            dilation_size = 2 ** i
            self.blocks.append(TemporalBlock(
                hidden_size, hidden_size, kernel_size,
                stride=1, dilation=dilation_size,
                padding=(kernel_size-1) * dilation_size,
                dropout=dropout
            ))

        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        self.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)
        x = x.transpose(1, 2)  # (batch_size, input_size, sequence_length)

        # Initial projection
        x = self.input_projection(x)

        # Apply blocks with skip connections
        for block in self.blocks:
            residual = x
            x = block(x)
            # Skip connection every 2 blocks
            if len(self.blocks) % 2 == 0:
                x = x + residual

        # Take last time step
        x = x[:, :, -1]

        # Layer norm
        x = self.layer_norm(x)

        # Predict
        output = self.fc(x)
        return output


# Factory function
def create_tcn_model(
    model_type: str,
    input_size: int,
    **kwargs
) -> BaseModel:
    """
    Create a TCN model by type.

    Args:
        model_type: One of ['basic', 'multiscale', 'residual']
        input_size: Number of input features
        **kwargs: Additional arguments for the model

    Returns:
        Instantiated model
    """
    models = {
        'basic': BasicTCN,
        'multiscale': MultiScaleTCN,
        'residual': ResidualTCN
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from {list(models.keys())}"
        )

    return models[model_type](input_size=input_size, **kwargs)