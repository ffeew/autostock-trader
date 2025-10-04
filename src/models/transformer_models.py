"""
Transformer-based models for time series prediction.

Transformers use self-attention to capture long-range dependencies.
Includes positional encoding and causal masking for time series.
"""

import math
import torch
import torch.nn as nn
from src.models.base_model import BaseModel


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Create positional encoding
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))

        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Tensor of shape (seq_len, batch_size, d_model)
        """
        x = x + self.pe[:x.size(0)]
        return self.dropout(x)


class BasicTransformer(BaseModel):
    """Basic Transformer model for time series."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        prediction_steps: int = 30
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, device, prediction_steps)

        self.num_heads = num_heads

        # Input projection
        self.input_projection = nn.Linear(input_size, hidden_size)

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)

        # Transformer encoder
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=False  # (seq_len, batch, features)
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)

        # Initialize weights
        self._init_weights()

        self.to(device)

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)

        # Project input
        x = self.input_projection(x)  # (batch, seq, hidden)

        # Transpose for transformer: (seq, batch, hidden)
        x = x.transpose(0, 1)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create causal mask (prevent looking at future)
        seq_len = x.size(0)
        mask = self._generate_square_subsequent_mask(seq_len).to(self.device)

        # Apply transformer
        x = self.transformer_encoder(x, mask=mask)

        # Take last time step: (batch, hidden)
        x = x[-1, :, :]

        # Predict
        output = self.fc(x)
        return output

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask to prevent attention to future positions."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class MultiHeadTransformer(BaseModel):
    """Transformer with multiple attention heads and layer normalization."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 6,
        num_heads: int = 8,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        prediction_steps: int = 30
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, device, prediction_steps)

        self.num_heads = num_heads

        # Input projection with layer norm
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)

        # Transformer encoder with layer norm
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=False,
            norm_first=True  # Pre-LN transformer
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )

        # Output layers with residual
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        # Initialize weights
        self._init_weights()

        self.to(device)

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)

        # Project input
        x = self.input_projection(x)  # (batch, seq, hidden)

        # Transpose for transformer
        x = x.transpose(0, 1)  # (seq, batch, hidden)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create causal mask
        seq_len = x.size(0)
        mask = self._generate_square_subsequent_mask(seq_len).to(self.device)

        # Apply transformer
        x = self.transformer_encoder(x, mask=mask)

        # Take last time step
        x = x[-1, :, :]  # (batch, hidden)

        # Predict
        output = self.fc(x)
        return output

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


class InformerTransformer(BaseModel):
    """
    Informer-inspired transformer for long sequences.

    Uses ProbSparse self-attention for efficiency.
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = 128,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        prediction_steps: int = 30
    ):
        super().__init__(input_size, hidden_size, num_layers, dropout, device, prediction_steps)

        self.num_heads = num_heads

        # Input projection with layer norm
        self.input_projection = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size)
        )

        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, dropout=dropout)

        # Transformer encoder (simplified - no distilling for now)
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=False,
            norm_first=True  # Pre-LN for stability
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layers,
            num_layers=num_layers
        )

        # Output layer with attention pooling
        self.attention_pool = nn.Sequential(
            nn.Linear(hidden_size, 1),
            nn.Softmax(dim=0)
        )

        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1)
        )

        # Initialize weights
        self._init_weights()

        self.to(device)

    def _init_weights(self):
        """Initialize weights using Xavier uniform initialization."""
        for name, param in self.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                nn.init.xavier_uniform_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: (batch_size, sequence_length, input_size)

        # Project input
        x = self.input_projection(x)  # (batch, seq, hidden)

        # Transpose for transformer
        x = x.transpose(0, 1)  # (seq, batch, hidden)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Create causal mask
        seq_len = x.size(0)
        mask = self._generate_square_subsequent_mask(seq_len).to(self.device)

        # Apply transformer
        x = self.transformer_encoder(x, mask=mask)

        # Use last time step (or could use attention pooling)
        x = x[-1, :, :]  # (batch, hidden)

        # Predict
        output = self.fc(x)
        return output

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate causal mask."""
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask


# Factory function
def create_transformer_model(
    model_type: str,
    input_size: int,
    **kwargs
) -> BaseModel:
    """
    Create a Transformer model by type.

    Args:
        model_type: One of ['basic', 'multihead', 'informer']
        input_size: Number of input features
        **kwargs: Additional arguments for the model

    Returns:
        Instantiated model
    """
    models = {
        'basic': BasicTransformer,
        'multihead': MultiHeadTransformer,
        'informer': InformerTransformer
    }

    if model_type not in models:
        raise ValueError(
            f"Unknown model type: {model_type}. "
            f"Choose from {list(models.keys())}"
        )

    return models[model_type](input_size=input_size, **kwargs)
