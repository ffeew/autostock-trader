"""
Base model class for all prediction models.

Provides common functionality for training, evaluation, and saving/loading models.
"""

import os
import logging
from abc import ABC, abstractmethod
from typing import Dict, List, Tuple, Optional
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class BaseModel(ABC, nn.Module):
    """Abstract base class for all time series prediction models."""

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        dropout: float = 0.2,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        prediction_steps: int = 30,
        target_feature_idx: int = 3
    ):
        """
        Initialize base model.

        Args:
            input_size: Number of input features
            hidden_size: Size of hidden layers
            num_layers: Number of layers
            dropout: Dropout rate
            device: Device to run model on
            prediction_steps: Number of timesteps to predict (autoregressive)
            target_feature_idx: Index of target feature for auto-regressive feedback
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.prediction_steps = prediction_steps
        self.target_feature_idx = target_feature_idx

        # Training history
        self.train_losses = []
        self.train_losses_ema = []  # Exponential moving average
        self.val_losses = []
        self.best_val_loss = float('inf')
        self.epochs_trained = 0
        self.ema_alpha = 0.1  # Smoothing factor for EMA

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)

        Returns:
            Output tensor of shape (batch_size, 1)
        """
        pass

    def forward_autoregressive(
        self,
        x: torch.Tensor,
        num_steps: int,
        target_feature_idx: int = 3,
        feature_scaler: 'StandardScaler' = None,
        ground_truth: torch.Tensor = None,
        teacher_forcing_ratio: float = 0.0
    ) -> torch.Tensor:
        """
        Auto-regressive forward pass (like LLM generation) with optional teacher forcing.

        Predicts num_steps into the future by feeding predictions back as input.
        With teacher forcing, uses ground truth values instead of predictions during training.

        Args:
            x: Input tensor of shape (batch_size, sequence_length, input_size)
            num_steps: Number of steps to predict ahead
            target_feature_idx: Index of the target feature (e.g., 'close' price) in the feature array
            feature_scaler: Optional scaler to denormalize/renormalize predictions
            ground_truth: Ground truth targets of shape (batch_size, num_steps) for teacher forcing
            teacher_forcing_ratio: Probability [0,1] of using ground truth vs prediction (0=no TF, 1=full TF)

        Returns:
            Predictions of shape (batch_size, num_steps, 1)
        """
        batch_size, seq_len, n_features = x.shape
        predictions = []

        # Current input window
        current_x = x.clone()

        # Determine whether to use teacher forcing for this batch
        use_teacher_forcing = (self.training and
                               ground_truth is not None and
                               teacher_forcing_ratio > 0.0)

        for step in range(num_steps):
            # Predict next value
            with torch.set_grad_enabled(self.training):
                pred = self.forward(current_x)  # (batch, 1)

            predictions.append(pred)

            # Prepare next input: slide window and append prediction or ground truth
            # New feature vector: copy last timestep features, update target column
            next_features = current_x[:, -1, :].clone()  # (batch, n_features)

            # Decide whether to use teacher forcing for this step
            if use_teacher_forcing:
                # Randomly decide to use ground truth based on teacher_forcing_ratio
                use_gt = torch.rand(1).item() < teacher_forcing_ratio

                if use_gt and step < ground_truth.shape[1]:
                    # Use ground truth value (already normalized)
                    next_value = ground_truth[:, step].detach()
                else:
                    # Use model prediction
                    next_value = pred.squeeze(-1).detach()
            else:
                # No teacher forcing - use model prediction
                # CRITICAL: Detach prediction when feeding back to prevent gradient issues
                next_value = pred.squeeze(-1).detach()

            # Update the target feature (e.g., 'close' price) with the chosen value
            next_features[:, target_feature_idx] = next_value

            # Slide window: remove oldest timestep, add new one
            next_features = next_features.unsqueeze(1)  # (batch, 1, n_features)
            current_x = torch.cat([current_x[:, 1:, :], next_features], dim=1)

        # Stack predictions: (batch, num_steps, 1)
        predictions = torch.stack(predictions, dim=1)

        return predictions

    def train_epoch(
        self,
        train_loader: DataLoader,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        writer: Optional[SummaryWriter] = None,
        epoch: int = 0,
        scaler: Optional[torch.amp.GradScaler] = None,
        use_amp: bool = False,
        teacher_forcing_ratio: float = 0.0
    ) -> Tuple[float, List[float]]:
        """
        Train for one epoch.

        Args:
            train_loader: Training data loader
            criterion: Loss function
            optimizer: Optimizer
            writer: Optional TensorBoard writer for batch-level logging
            epoch: Current epoch number
            scaler: GradScaler for mixed precision training
            use_amp: Whether to use automatic mixed precision
            teacher_forcing_ratio: Probability of using ground truth vs prediction (0.0-1.0)

        Returns:
            Tuple of (average training loss, list of batch losses)
        """
        self.train()
        total_loss = 0
        n_batches = 0
        batch_losses = []

        for batch_idx, (batch_x, batch_y) in enumerate(tqdm(train_loader, desc="Training", leave=False)):
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Forward pass with mixed precision
            optimizer.zero_grad()

            # Use autocast for mixed precision
            with torch.amp.autocast('cuda',enabled=use_amp):
                # Auto-regressive training: predict multiple steps with teacher forcing
                predictions = self.forward_autoregressive(
                    batch_x,
                    self.prediction_steps,
                    self.target_feature_idx,
                    ground_truth=batch_y,
                    teacher_forcing_ratio=teacher_forcing_ratio
                )
                # predictions: (batch, num_steps, 1)
                # batch_y: (batch, num_steps)

                # Compute loss on ALL predictions (not just final)
                predictions_flat = predictions.squeeze(-1)  # (batch, num_steps)
                loss = criterion(predictions_flat, batch_y)

            # Backward pass with gradient scaling
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)
                optimizer.step()

            batch_loss = loss.item()
            total_loss += batch_loss
            batch_losses.append(batch_loss)
            n_batches += 1

            # Log batch-level metrics to TensorBoard
            if writer is not None and batch_idx % 10 == 0:
                global_step = epoch * len(train_loader) + batch_idx
                writer.add_scalar('Loss/train_batch', batch_loss, global_step)

        avg_loss = total_loss / n_batches
        return avg_loss, batch_losses

    def validate(
        self,
        val_loader: DataLoader,
        criterion: nn.Module,
        use_amp: bool = False
    ) -> float:
        """
        Validate the model.

        Args:
            val_loader: Validation data loader
            criterion: Loss function
            use_amp: Use automatic mixed precision

        Returns:
            Average validation loss
        """
        self.eval()
        total_loss = 0
        n_batches = 0

        with torch.no_grad():
            for batch_x, batch_y in tqdm(val_loader, desc="Validating", leave=False):
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Use autocast for mixed precision
                with torch.amp.autocast('cuda',enabled=use_amp):
                    # Auto-regressive validation: predict multiple steps
                    predictions = self.forward_autoregressive(batch_x, self.prediction_steps, self.target_feature_idx)
                    # predictions: (batch, num_steps, 1)
                    # batch_y: (batch, num_steps)

                    # Compute loss on ALL predictions (not just final)
                    predictions_flat = predictions.squeeze(-1)  # (batch, num_steps)
                    loss = criterion(predictions_flat, batch_y)

                total_loss += loss.item()
                n_batches += 1

        avg_loss = total_loss / n_batches
        return avg_loss

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 50,
        learning_rate: float = 0.0005,
        early_stopping_patience: int = 10,
        save_path: Optional[str] = None,
        tensorboard_log_dir: Optional[str] = None,
        val_frequency: int = 1,
        use_amp: bool = True,
        teacher_forcing_ratio: float = 0.5,
        teacher_forcing_decay: float = 0.02,
        weight_decay: float = 1e-5
    ) -> Dict[str, List[float]]:
        """
        Train the model.

        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs to train
            learning_rate: Learning rate
            early_stopping_patience: Patience for early stopping
            save_path: Path to save best model
            tensorboard_log_dir: Directory for TensorBoard logs (optional)
            val_frequency: Validate every N epochs (default: 1)
            use_amp: Use Automatic Mixed Precision for 2x speedup (default: True)
            teacher_forcing_ratio: Initial probability of using ground truth (0.0-1.0, default: 0.5)
            teacher_forcing_decay: Amount to reduce TF ratio each epoch (default: 0.02)
            weight_decay: L2 regularization weight decay (default: 1e-5)

        Returns:
            Dictionary with training history
        """
        logger.info(f"Training {self.__class__.__name__} on {self.device}")
        logger.info(f"Epochs: {epochs}, LR: {learning_rate}, Weight Decay: {weight_decay}")
        logger.info(f"Auto-regressive prediction: {self.prediction_steps} steps")
        logger.info(f"Teacher Forcing: Initial={teacher_forcing_ratio:.2f}, Decay={teacher_forcing_decay:.3f}")

        # Check if AMP is available and enabled
        use_amp = use_amp and torch.cuda.is_available() and self.device == 'cuda'
        if use_amp:
            logger.info(f"Mixed Precision Training: ENABLED (AMP with float16)")
        else:
            logger.info(f"Mixed Precision Training: DISABLED")

        # Initialize TensorBoard writer if log directory provided
        writer = None
        if tensorboard_log_dir:
            writer = SummaryWriter(log_dir=tensorboard_log_dir)
            logger.info(f"TensorBoard logging enabled: {tensorboard_log_dir}")
            logger.info("View with: tensorboard --logdir=runs")

        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate, weight_decay=weight_decay)

        # Initialize GradScaler for mixed precision training
        scaler = torch.amp.GradScaler('cuda', enabled=use_amp)

        # Add learning rate scheduler
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
            min_lr=1e-6
        )

        patience_counter = 0
        current_tf_ratio = teacher_forcing_ratio  # Track current teacher forcing ratio

        for epoch in range(epochs):
            # Train with current teacher forcing ratio
            train_loss, batch_losses = self.train_epoch(
                train_loader, criterion, optimizer, writer, epoch, scaler, use_amp,
                teacher_forcing_ratio=current_tf_ratio
            )
            self.train_losses.append(train_loss)

            # Compute EMA of training loss
            if len(self.train_losses_ema) == 0:
                train_loss_ema = train_loss
            else:
                train_loss_ema = (self.ema_alpha * train_loss +
                                 (1 - self.ema_alpha) * self.train_losses_ema[-1])
            self.train_losses_ema.append(train_loss_ema)

            # Validate (only every val_frequency epochs)
            should_validate = (epoch % val_frequency == 0) or (epoch == epochs - 1)

            if should_validate:
                val_loss = self.validate(val_loader, criterion, use_amp)
                self.val_losses.append(val_loss)
            else:
                # Use last validation loss for scheduler
                val_loss = self.val_losses[-1] if self.val_losses else float('inf')

            self.epochs_trained += 1

            # Step the learning rate scheduler
            scheduler.step(val_loss)

            # Get current learning rate
            current_lr = optimizer.param_groups[0]['lr']

            # TensorBoard logging
            if writer:
                writer.add_scalar('Loss/train', train_loss, epoch)
                writer.add_scalar('Loss/train_ema', train_loss_ema, epoch)
                if should_validate:
                    writer.add_scalar('Loss/validation', val_loss, epoch)
                writer.add_scalar('Learning_Rate', current_lr, epoch)
                writer.add_scalar('Teacher_Forcing_Ratio', current_tf_ratio, epoch)

                # Log training loss variance
                batch_loss_variance = np.var(batch_losses)
                batch_loss_std = np.std(batch_losses)
                writer.add_scalar('Loss/train_batch_variance', batch_loss_variance, epoch)
                writer.add_scalar('Loss/train_batch_std', batch_loss_std, epoch)

                # Log gradient norms
                total_norm = 0
                for p in self.parameters():
                    if p.grad is not None:
                        param_norm = p.grad.data.norm(2)
                        total_norm += param_norm.item() ** 2
                total_norm = total_norm ** 0.5
                writer.add_scalar('Gradient/norm', total_norm, epoch)

                # Log model weights histograms every 10 epochs
                if epoch % 10 == 0:
                    for name, param in self.named_parameters():
                        writer.add_histogram(f'Weights/{name}', param, epoch)
                        if param.grad is not None:
                            writer.add_histogram(f'Gradients/{name}', param.grad, epoch)

            # Log training progress
            if should_validate:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.6f} (EMA: {train_loss_ema:.6f}), "
                    f"Val Loss: {val_loss:.6f}, TF Ratio: {current_tf_ratio:.3f}"
                )
            else:
                logger.info(
                    f"Epoch {epoch+1}/{epochs} - "
                    f"Train Loss: {train_loss:.6f} (EMA: {train_loss_ema:.6f}), TF Ratio: {current_tf_ratio:.3f}"
                )

            # Early stopping (only check on validation epochs)
            if should_validate and val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0

                # Log best model to TensorBoard
                if writer:
                    writer.add_scalar('Best_Val_Loss', val_loss, epoch)

                # Save best model
                if save_path:
                    self.save(save_path)
                    logger.info(f"✓ Saved best model to {save_path}")
            else:
                patience_counter += 1

            if patience_counter >= early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break

            # Decay teacher forcing ratio (scheduled sampling)
            current_tf_ratio = max(0.0, current_tf_ratio - teacher_forcing_decay)

        logger.info(f"Training completed. Best val loss: {self.best_val_loss:.6f}")
        logger.info(f"Final teacher forcing ratio: {current_tf_ratio:.3f}")

        # Close TensorBoard writer
        if writer:
            writer.close()

        return {
            'train_losses': self.train_losses,
            'train_losses_ema': self.train_losses_ema,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss
        }

    def predict(
        self,
        data_loader: DataLoader
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on a dataset.

        Args:
            data_loader: Data loader

        Returns:
            predictions: Predicted values (batch, num_steps)
            actuals: Actual values (batch, num_steps)
        """
        self.eval()
        all_predictions = []
        all_actuals = []

        with torch.no_grad():
            for batch_x, batch_y in tqdm(data_loader, desc="Predicting", leave=False):
                batch_x = batch_x.to(self.device)

                # Auto-regressive: predict full sequence
                predictions = self.forward_autoregressive(batch_x, self.prediction_steps, self.target_feature_idx)
                predictions = predictions.squeeze(-1)  # (batch, num_steps)

                all_predictions.append(predictions.cpu().numpy())
                all_actuals.append(batch_y.numpy())

        predictions = np.concatenate(all_predictions, axis=0)
        actuals = np.concatenate(all_actuals, axis=0)

        return predictions, actuals

    def evaluate(
        self,
        test_loader: DataLoader
    ) -> Dict[str, float]:
        """
        Evaluate model on test set.

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

        # Directional accuracy (did we predict the direction correctly?)
        # For multi-step predictions, compute on final timestep
        pred_returns = np.diff(predictions[:, -1])
        actual_returns = np.diff(actuals[:, -1])

        directional_accuracy = np.mean(
            np.sign(pred_returns) == np.sign(actual_returns)
        )

        metrics = {
            'mse': float(mse),
            'rmse': float(rmse),
            'mae': float(mae),
            'directional_accuracy': float(directional_accuracy)
        }

        logger.info("Evaluation metrics:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value:.6f}")

        return metrics

    def save(self, path: str):
        """Save model checkpoint."""
        os.makedirs(os.path.dirname(path), exist_ok=True)

        checkpoint = {
            'model_state_dict': self.state_dict(),
            'input_size': self.input_size,
            'hidden_size': self.hidden_size,
            'num_layers': self.num_layers,
            'dropout': self.dropout,
            'prediction_steps': self.prediction_steps,
            'target_feature_idx': self.target_feature_idx,
            'train_losses': self.train_losses,
            'train_losses_ema': self.train_losses_ema,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'epochs_trained': self.epochs_trained
        }

        torch.save(checkpoint, path)

    @classmethod
    def load(cls, path: str, device: str = 'cpu'):
        """Load model checkpoint."""
        checkpoint = torch.load(path, map_location=device)

        model = cls(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers'],
            dropout=checkpoint['dropout'],
            device=device,
            prediction_steps=checkpoint.get('prediction_steps', 30),
            target_feature_idx=checkpoint.get('target_feature_idx', 3)
        )

        model.load_state_dict(checkpoint['model_state_dict'])
        model.train_losses = checkpoint.get('train_losses', [])
        model.train_losses_ema = checkpoint.get('train_losses_ema', [])
        model.val_losses = checkpoint.get('val_losses', [])
        model.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        model.epochs_trained = checkpoint.get('epochs_trained', 0)

        model.to(device)
        return model

    def count_parameters(self) -> int:
        """Count trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def summary(self):
        """Print model summary."""
        logger.info(f"\n{self.__class__.__name__} Summary:")
        logger.info(f"  Input size: {self.input_size}")
        logger.info(f"  Hidden size: {self.hidden_size}")
        logger.info(f"  Num layers: {self.num_layers}")
        logger.info(f"  Dropout: {self.dropout}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Trainable parameters: {self.count_parameters():,}")
        logger.info(f"  Epochs trained: {self.epochs_trained}")