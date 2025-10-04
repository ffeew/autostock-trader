"""
Data loader for time series prediction.

This module handles:
- Loading and preprocessing data
- Creating sequences for time series models
- Train/validation/test splitting (time-based)
- Data normalization and scaling
- Efficient batch generation
"""

import os
import logging
import numpy as np
import pandas as pd
from typing import Tuple, Optional
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset, DataLoader
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeSeriesDataset(Dataset):
    """PyTorch Dataset for time series data."""

    def __init__(
        self,
        data: np.ndarray,
        targets: np.ndarray,
        sequence_length: int,
        prediction_steps: int = 30
    ):
        """
        Initialize dataset.

        Args:
            data: Feature data (samples, features)
            targets: Target values (samples,)
            sequence_length: Length of input sequences
            prediction_steps: Number of steps to predict ahead (default: 30)
        """
        self.data = data
        self.targets = targets
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps

    def __len__(self) -> int:
        # Need room for: sequence_length input + prediction_steps output
        return len(self.data) - self.sequence_length - self.prediction_steps + 1

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Get input sequence
        x = self.data[idx:idx + self.sequence_length]

        # Get target: SEQUENCE of next N values (auto-regressive)
        # targets[idx + sequence_length] = t+1
        # targets[idx + sequence_length + prediction_steps - 1] = t+N
        start_idx = idx + self.sequence_length
        end_idx = start_idx + self.prediction_steps
        y = self.targets[start_idx:end_idx]

        return (
            torch.FloatTensor(x),
            torch.FloatTensor(y)  # Shape: (prediction_steps,) e.g., (30,)
        )


class StockDataLoader:
    """Loads and prepares stock data for model training."""

    def __init__(
        self,
        data_path: str = 'data/SPY_features.parquet',
        target_column: str = 'close',
        sequence_length: int = 60,
        prediction_steps: int = 30,
        train_ratio: float = 0.7,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
        batch_size: int = 32
    ):
        """
        Initialize data loader.

        Args:
            data_path: Path to feature dataset
            target_column: Column to predict
            sequence_length: Length of input sequences (e.g., 60 minutes)
            prediction_steps: Steps ahead to predict
            train_ratio: Proportion of data for training
            val_ratio: Proportion of data for validation
            test_ratio: Proportion of data for testing
            batch_size: Batch size for training
        """
        self.data_path = data_path
        self.target_column = target_column
        self.sequence_length = sequence_length
        self.prediction_steps = prediction_steps
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio
        self.batch_size = batch_size

        # Load prediction steps from .env if available
        load_dotenv()
        env_pred_steps = os.getenv('PREDICTION_STEPS')
        if env_pred_steps:
            self.prediction_steps = int(env_pred_steps)
            logger.info(f"Using PREDICTION_STEPS from .env: {self.prediction_steps}")

        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()  # Add scaler for target variable
        self.feature_columns = None
        self.n_features = None

    def load_data(self) -> pd.DataFrame:
        """Load dataset from parquet file."""
        logger.info(f"Loading data from {self.data_path}")

        if not os.path.exists(self.data_path):
            raise FileNotFoundError(
                f"Dataset not found at {self.data_path}. "
                "Run generate_dataset.py first."
            )

        df = pd.read_parquet(self.data_path)
        logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
        logger.info(f"Date range: {df.index[0]} to {df.index[-1]}")

        return df

    def prepare_data(
        self
    ) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, StandardScaler]:
        """
        Prepare data for training.

        Returns:
            train_loader: Training data loader
            val_loader: Validation data loader
            test_loader: Test data loader
            scaler: Fitted scaler for features inverse transform
            target_scaler: Fitted scaler for target inverse transform
        """
        # Load data
        df = self.load_data()

        # Get feature columns (all except target)
        self.feature_columns = [col for col in df.columns if col != self.target_column]
        self.n_features = len(self.feature_columns)

        logger.info(f"Using {self.n_features} features for prediction")

        # Extract features and targets
        features = df[self.feature_columns].values
        targets = df[self.target_column].values

        # Shift targets by prediction_steps
        # Target at time t is the price at time t + prediction_steps
        targets = np.roll(targets, -self.prediction_steps)
        # Remove last prediction_steps rows (no valid target)
        features = features[:-self.prediction_steps]
        targets = targets[:-self.prediction_steps]

        logger.info(f"After shifting targets by {self.prediction_steps} steps:")
        logger.info(f"  Features shape: {features.shape}")
        logger.info(f"  Targets shape: {targets.shape}")

        # Time-based split (NOT random)
        n_samples = len(features)
        train_end = int(n_samples * self.train_ratio)
        val_end = int(n_samples * (self.train_ratio + self.val_ratio))

        train_features = features[:train_end]
        train_targets = targets[:train_end]

        val_features = features[train_end:val_end]
        val_targets = targets[train_end:val_end]

        test_features = features[val_end:]
        test_targets = targets[val_end:]

        logger.info("Data split:")
        logger.info(f"  Train: {len(train_features)} samples")
        logger.info(f"  Val:   {len(val_features)} samples")
        logger.info(f"  Test:  {len(test_features)} samples")

        # Fit scaler on training data only
        self.scaler.fit(train_features)

        # Fit target scaler on training targets only
        self.target_scaler.fit(train_targets.reshape(-1, 1))

        # Transform all datasets
        train_features = self.scaler.transform(train_features)
        val_features = self.scaler.transform(val_features)
        test_features = self.scaler.transform(test_features)

        # Transform targets (CRITICAL: normalize target variable)
        train_targets = self.target_scaler.transform(train_targets.reshape(-1, 1)).flatten()
        val_targets = self.target_scaler.transform(val_targets.reshape(-1, 1)).flatten()
        test_targets = self.target_scaler.transform(test_targets.reshape(-1, 1)).flatten()

        logger.info("Data normalized using StandardScaler")
        logger.info("Target variable normalized (mean=0, std=1)")

        # Create PyTorch datasets
        train_dataset = TimeSeriesDataset(
            train_features, train_targets, self.sequence_length, self.prediction_steps
        )
        val_dataset = TimeSeriesDataset(
            val_features, val_targets, self.sequence_length, self.prediction_steps
        )
        test_dataset = TimeSeriesDataset(
            test_features, test_targets, self.sequence_length, self.prediction_steps
        )

        # Create data loaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # NO shuffling for time series - preserves temporal order
            num_workers=0,  # Use 0 for compatibility
            pin_memory=True
        )

        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True
        )

        logger.info(f"Created data loaders with batch size: {self.batch_size}")
        logger.info(f"Train batches: {len(train_loader)}")
        logger.info(f"Val batches:   {len(val_loader)}")
        logger.info(f"Test batches:  {len(test_loader)}")

        return train_loader, val_loader, test_loader, self.scaler, self.target_scaler

    def get_sample_batch(self) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get a single batch for testing model architecture."""
        train_loader, _, _, _, _ = self.prepare_data()
        x, y = next(iter(train_loader))
        logger.info(f"Sample batch shapes: X={x.shape}, Y={y.shape}")
        return x, y


def test_data_loader():
    """Test the data loader."""
    logger.info("Testing StockDataLoader")

    # Create loader
    loader = StockDataLoader(
        data_path='data/SPY_features.parquet',
        sequence_length=60,
        prediction_steps=30,
        batch_size=32
    )

    # Prepare data
    train_loader, val_loader, test_loader, scaler, target_scaler = loader.prepare_data()

    # Get sample batch
    x, y = next(iter(train_loader))
    logger.info(f"Sample batch:")
    logger.info(f"  Input shape: {x.shape}")  # (batch_size, sequence_length, n_features)
    logger.info(f"  Target shape: {y.shape}")  # (batch_size, 1)
    logger.info(f"  Input range: [{x.min():.3f}, {x.max():.3f}]")
    logger.info(f"  Target range: [{y.min():.3f}, {y.max():.3f}]")

    logger.info("\n✓ Data loader test completed successfully!")


if __name__ == '__main__':
    test_data_loader()