"""
Ensemble methods for combining multiple models.

Combines predictions from multiple models using:
- Simple averaging
- Weighted averaging
- Stacking (meta-learner)
"""

import os
import logging
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.models.base_model import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SimpleEnsemble:
    """
    Simple ensemble using average predictions.

    Takes predictions from multiple models and averages them.
    """

    def __init__(self, models: List[BaseModel]):
        """
        Initialize ensemble.

        Args:
            models: List of trained models
        """
        self.models = models
        logger.info(f"Created SimpleEnsemble with {len(models)} models")

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using ensemble.

        Args:
            data_loader: Data loader

        Returns:
            predictions: Averaged predictions
            actuals: Actual values
        """
        all_model_predictions = []

        # Get predictions from each model
        for i, model in enumerate(self.models):
            logger.info(f"Getting predictions from model {i+1}/{len(self.models)}")
            predictions, actuals = model.predict(data_loader)
            all_model_predictions.append(predictions)

        # Average predictions
        ensemble_predictions = np.mean(all_model_predictions, axis=0)

        return ensemble_predictions, actuals

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate ensemble on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of evaluation metrics
        """
        predictions, actuals = self.predict(test_loader)

        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))

        # Directional accuracy
        pred_returns = np.diff(predictions.flatten())
        actual_returns = np.diff(actuals.flatten())
        directional_accuracy = np.mean(
            np.sign(pred_returns) == np.sign(actual_returns)
        )

        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'directional_accuracy': float(directional_accuracy)
        }

        logger.info("Ensemble evaluation metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.6f}")

        return metrics


class WeightedEnsemble:
    """
    Weighted ensemble with learned weights.

    Optimizes weights for each model based on validation performance.
    """

    def __init__(
        self,
        models: List[BaseModel],
        val_loader: DataLoader,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize weighted ensemble.

        Args:
            models: List of trained models
            val_loader: Validation data loader for optimizing weights
            device: Device to run on
        """
        self.models = models
        self.device = device
        self.weights = None

        logger.info(f"Created WeightedEnsemble with {len(models)} models")

        # Optimize weights on validation set
        self._optimize_weights(val_loader)

    def _optimize_weights(self, val_loader: DataLoader):
        """Optimize ensemble weights on validation set."""
        logger.info("Optimizing ensemble weights...")

        # Get predictions from all models
        all_predictions = []
        actuals = None

        for i, model in enumerate(self.models):
            predictions, actuals = model.predict(val_loader)
            all_predictions.append(predictions.flatten())

        actuals = actuals.flatten()
        all_predictions = np.array(all_predictions).T  # (n_samples, n_models)

        # Initialize weights uniformly
        weights = torch.nn.Parameter(torch.ones(len(self.models)) / len(self.models))
        weights = weights.to(self.device)

        # Convert to tensors
        predictions_tensor = torch.FloatTensor(all_predictions).to(self.device)
        actuals_tensor = torch.FloatTensor(actuals).to(self.device)

        # Optimize weights using gradient descent
        optimizer = torch.optim.Adam([weights], lr=0.01)
        criterion = nn.MSELoss()

        best_loss = float('inf')
        best_weights = None

        for epoch in range(100):
            optimizer.zero_grad()

            # Ensure weights are positive and sum to 1
            normalized_weights = torch.softmax(weights, dim=0)

            # Weighted predictions
            weighted_pred = torch.matmul(predictions_tensor, normalized_weights)

            # Loss
            loss = criterion(weighted_pred, actuals_tensor)

            loss.backward()
            optimizer.step()

            if loss.item() < best_loss:
                best_loss = loss.item()
                best_weights = normalized_weights.detach().cpu().numpy()

        self.weights = best_weights

        logger.info("Optimized ensemble weights:")
        for i, weight in enumerate(self.weights):
            logger.info(f"  Model {i+1}: {weight:.4f}")
        logger.info(f"  Validation MSE: {best_loss:.6f}")

    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions using weighted ensemble.

        Args:
            data_loader: Data loader

        Returns:
            predictions: Weighted predictions
            actuals: Actual values
        """
        all_model_predictions = []

        # Get predictions from each model
        for i, model in enumerate(self.models):
            predictions, actuals = model.predict(data_loader)
            all_model_predictions.append(predictions.flatten())

        # Weighted average
        all_predictions = np.array(all_model_predictions).T  # (n_samples, n_models)
        ensemble_predictions = np.dot(all_predictions, self.weights)
        ensemble_predictions = ensemble_predictions.reshape(-1, 1)

        return ensemble_predictions, actuals

    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """
        Evaluate ensemble on test set.

        Args:
            test_loader: Test data loader

        Returns:
            Dictionary of evaluation metrics
        """
        predictions, actuals = self.predict(test_loader)

        # Calculate metrics
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        mae = np.mean(np.abs(predictions - actuals))

        # Directional accuracy
        pred_returns = np.diff(predictions.flatten())
        actual_returns = np.diff(actuals.flatten())
        directional_accuracy = np.mean(
            np.sign(pred_returns) == np.sign(actual_returns)
        )

        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'directional_accuracy': float(directional_accuracy)
        }

        logger.info("Weighted ensemble evaluation metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.6f}")

        return metrics


class StackingEnsemble(BaseModel):
    """
    Stacking ensemble with meta-learner.

    Uses predictions from base models as features for a meta-learner.
    """

    def __init__(
        self,
        base_models: List[BaseModel],
        meta_hidden_size: int = 64,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
        """
        Initialize stacking ensemble.

        Args:
            base_models: List of trained base models
            meta_hidden_size: Hidden size for meta-learner
            dropout: Dropout rate
            device: Device to run on
        """
        # Initialize BaseModel with dummy values
        super().__init__(
            input_size=len(base_models),
            hidden_size=meta_hidden_size,
            num_layers=1,
            dropout=dropout,
            device=device
        )

        self.base_models = base_models
        self.n_base_models = len(base_models)

        # Meta-learner: simple MLP
        self.meta_learner = nn.Sequential(
            nn.Linear(self.n_base_models, meta_hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(meta_hidden_size, meta_hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(meta_hidden_size // 2, 1)
        )

        self.to(device)

        logger.info(f"Created StackingEnsemble with {self.n_base_models} base models")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through stacking ensemble.

        Args:
            x: Input tensor (batch_size, sequence_length, input_size)

        Returns:
            Meta predictions
        """
        # Get predictions from all base models
        base_predictions = []

        for model in self.base_models:
            with torch.no_grad():
                pred = model(x)
                base_predictions.append(pred)

        # Stack predictions: (batch_size, n_base_models)
        base_predictions = torch.cat(base_predictions, dim=1)

        # Meta-learner prediction
        meta_prediction = self.meta_learner(base_predictions)

        return meta_prediction

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 20,
        learning_rate: float = 0.001,
        early_stopping_patience: int = 5,
        save_path: Optional[str] = None
    ) -> Dict[str, List[float]]:
        """
        Train the meta-learner.

        Base models should already be trained.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
            save_path: Path to save best model

        Returns:
            Dictionary with training history
        """
        logger.info("Training meta-learner (base models frozen)")

        # Freeze base models
        for model in self.base_models:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        # Train meta-learner using parent class method
        return super().fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            learning_rate=learning_rate,
            early_stopping_patience=early_stopping_patience,
            save_path=save_path
        )


def create_ensemble(
    ensemble_type: str,
    models: List[BaseModel],
    val_loader: Optional[DataLoader] = None,
    **kwargs
):
    """
    Create an ensemble by type.

    Args:
        ensemble_type: One of ['simple', 'weighted', 'stacking']
        models: List of trained base models
        val_loader: Validation data loader (required for weighted and stacking)
        **kwargs: Additional arguments for ensemble

    Returns:
        Instantiated ensemble
    """
    if ensemble_type == 'simple':
        return SimpleEnsemble(models)

    elif ensemble_type == 'weighted':
        if val_loader is None:
            raise ValueError("val_loader required for weighted ensemble")
        return WeightedEnsemble(models, val_loader, **kwargs)

    elif ensemble_type == 'stacking':
        if val_loader is None:
            raise ValueError("val_loader required for stacking ensemble")
        ensemble = StackingEnsemble(models, **kwargs)
        # Note: User must call ensemble.fit() to train meta-learner
        return ensemble

    else:
        raise ValueError(
            f"Unknown ensemble type: {ensemble_type}. "
            f"Choose from ['simple', 'weighted', 'stacking']"
        )
