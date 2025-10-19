"""
Main training script for all models.

Trains multiple model architectures and saves the best checkpoints.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List
from datetime import datetime
import torch
from dotenv import load_dotenv

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.data_loader import StockDataLoader
from src.models.lstm_models import create_lstm_model
from src.models.gru_models import create_gru_model
from src.models.tcn_models import create_tcn_model
from src.models.transformer_models import create_transformer_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def train_single_model(
    model_family: str,
    model_type: str,
    input_size: int,
    train_loader,
    val_loader,
    epochs: int = 50,
    learning_rate: float = 0.001,
    early_stopping_patience: int = 10,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
    tensorboard_log_dir: str = None,
    val_frequency: int = 1,
    use_amp: bool = True,
    teacher_forcing_ratio: float = 0.5,
    teacher_forcing_decay: float = 0.02,
    weight_decay: float = 1e-5,
    **model_kwargs
) -> Dict:
    """
    Train a single model.

    Args:
        model_family: One of ['lstm', 'gru', 'tcn', 'transformer']
        model_type: Specific model type (e.g., 'basic', 'attention')
        input_size: Number of input features
        train_loader: Training data loader
        val_loader: Validation data loader
        epochs: Number of epochs
        learning_rate: Learning rate
        early_stopping_patience: Early stopping patience
        device: Device to run on
        teacher_forcing_ratio: Initial teacher forcing ratio (0.0-1.0)
        teacher_forcing_decay: Teacher forcing decay per epoch
        weight_decay: Weight decay for L2 regularization
        **model_kwargs: Additional model arguments

    Returns:
        Training history dictionary
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"Training {model_family.upper()} - {model_type}")
    logger.info(f"{'='*60}")

    # Create model
    if model_family == 'lstm':
        model = create_lstm_model(model_type, input_size, device=device, **model_kwargs)
    elif model_family == 'gru':
        model = create_gru_model(model_type, input_size, device=device, **model_kwargs)
    elif model_family == 'tcn':
        model = create_tcn_model(model_type, input_size, device=device, **model_kwargs)
    elif model_family == 'transformer':
        model = create_transformer_model(model_type, input_size, device=device, **model_kwargs)
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    # Print model summary
    model.summary()

    # Save path
    save_dir = f'models/checkpoints/{model_family}'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{model_type}_best.pt')

    # Train
    history = model.fit(
        train_loader=train_loader,
        val_loader=val_loader,
        epochs=epochs,
        learning_rate=learning_rate,
        early_stopping_patience=early_stopping_patience,
        save_path=save_path,
        tensorboard_log_dir=tensorboard_log_dir,
        val_frequency=val_frequency,
        use_amp=use_amp,
        teacher_forcing_ratio=teacher_forcing_ratio,
        teacher_forcing_decay=teacher_forcing_decay,
        weight_decay=weight_decay
    )

    logger.info(f"✓ Training completed for {model_family}-{model_type}")
    logger.info(f"  Best val loss: {history['best_val_loss']:.6f}")
    logger.info(f"  Saved to: {save_path}\n")

    return history


def main():
    parser = argparse.ArgumentParser(description='Train stock prediction models')
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/SPY_features.parquet',
        help='Path to feature dataset'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=60,
        help='Length of input sequences (minutes)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=256,
        help='Batch size'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=50,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--learning-rate',
        type=float,
        default=0.0005,
        help='Learning rate'
    )
    parser.add_argument(
        '--hidden-size',
        type=int,
        default=128,
        help='Hidden size for models'
    )
    parser.add_argument(
        '--num-layers',
        type=int,
        default=2,
        help='Number of layers'
    )
    parser.add_argument(
        '--dropout',
        type=float,
        default=0.3,
        help='Dropout rate (increased from 0.2 to 0.3 for better regularization)'
    )
    parser.add_argument(
        '--early-stopping',
        type=int,
        default=10,
        help='Early stopping patience'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        help='Models to train (e.g., lstm:basic gru:residual) or "all"'
    )
    parser.add_argument(
        '--tensorboard',
        action='store_true',
        help='Enable TensorBoard logging for training visualization'
    )
    parser.add_argument(
        '--val-frequency',
        type=int,
        default=1,
        help='Validate every N epochs (default: 1, use 2-3 for faster training)'
    )
    parser.add_argument(
        '--no-amp',
        action='store_true',
        help='Disable automatic mixed precision training'
    )
    parser.add_argument(
        '--teacher-forcing',
        type=float,
        default=0.5,
        help='Initial teacher forcing ratio (0.0-1.0, default: 0.5)'
    )
    parser.add_argument(
        '--tf-decay',
        type=float,
        default=0.02,
        help='Teacher forcing decay per epoch (default: 0.02)'
    )
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=1e-5,
        help='Weight decay for L2 regularization (default: 1e-5)'
    )

    args = parser.parse_args()

    # Setup logging to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'training_{timestamp}.log')

    # Configure logging to both file and console
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))

    # Get root logger and add handlers
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # Clear existing handlers
    root_logger.addHandler(file_handler)
    root_logger.addHandler(console_handler)

    logger.info(f"Logging to: {log_file}")

    # Load environment
    load_dotenv()

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("\n" + "="*60)
    logger.info("Loading and preparing data")
    logger.info("="*60)

    data_loader = StockDataLoader(
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size
    )

    train_loader, val_loader, test_loader, scaler, target_scaler, target_feature_idx = data_loader.prepare_data()
    input_size = data_loader.n_features

    logger.info(f"✓ Data loaded successfully")
    logger.info(f"  Input features: {input_size}")
    logger.info(f"  Target feature index: {target_feature_idx}")
    logger.info(f"  Sequence length: {args.sequence_length}")
    logger.info(f"  Batch size: {args.batch_size}")
    logger.info(f"  Target normalization: ENABLED (mean=0, std=1)\n")

    # Model configurations
    model_configs = {
        'lstm': ['basic', 'stacked', 'bidirectional', 'attention'],
        'gru': ['basic', 'residual', 'batchnorm'],
        'tcn': ['basic', 'multiscale', 'residual'],
        'transformer': ['basic', 'multihead', 'informer']
    }

    # Determine which models to train
    if 'all' in args.models:
        models_to_train = []
        for family, types in model_configs.items():
            for model_type in types:
                models_to_train.append((family, model_type))
    else:
        models_to_train = []
        for model_spec in args.models:
            if ':' in model_spec:
                family, model_type = model_spec.split(':')
                models_to_train.append((family, model_type))
            else:
                logger.warning(f"Skipping invalid model spec: {model_spec}")

    logger.info(f"Training {len(models_to_train)} model(s)")

    # TensorBoard setup
    if args.tensorboard:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        tensorboard_base_dir = f'runs/{timestamp}'
        logger.info(f"\n✓ TensorBoard logging enabled")
        logger.info(f"  Log directory: {tensorboard_base_dir}")
        logger.info(f"  View with: tensorboard --logdir=runs")
        logger.info(f"  Then open: http://localhost:6006\n")
    else:
        tensorboard_base_dir = None

    # Common model kwargs
    model_kwargs = {
        'hidden_size': args.hidden_size,
        'num_layers': args.num_layers,
        'dropout': args.dropout,
        'prediction_steps': data_loader.prediction_steps,
        'target_feature_idx': target_feature_idx
    }

    # Log auto-regressive mode and training improvements
    logger.info(f"\n🔄 AUTO-REGRESSIVE TRAINING WITH TEACHER FORCING")
    logger.info(f"  Prediction steps: {data_loader.prediction_steps}")
    logger.info(f"  Training: Model will predict {data_loader.prediction_steps} steps iteratively")
    logger.info(f"  Each prediction feeds back as input for next prediction")
    logger.info(f"\n📊 TRAINING IMPROVEMENTS:")
    logger.info(f"  ✓ Teacher Forcing: {args.teacher_forcing:.2f} (decays by {args.tf_decay:.3f}/epoch)")
    logger.info(f"  ✓ Weight Decay (L2): {args.weight_decay:.0e}")
    logger.info(f"  ✓ Dropout: {args.dropout:.2f}")
    logger.info(f"  ✓ Scheduled Sampling: Enabled (prevents overfitting)\n")

    # Train each model
    results = {}
    for family, model_type in models_to_train:
        try:
            # Create TensorBoard log directory for this model
            tb_log_dir = None
            if tensorboard_base_dir:
                tb_log_dir = os.path.join(tensorboard_base_dir, f'{family}_{model_type}')
                os.makedirs(tb_log_dir, exist_ok=True)

            history = train_single_model(
                model_family=family,
                model_type=model_type,
                input_size=input_size,
                train_loader=train_loader,
                val_loader=val_loader,
                epochs=args.epochs,
                learning_rate=args.learning_rate,
                early_stopping_patience=args.early_stopping,
                device=device,
                tensorboard_log_dir=tb_log_dir,
                val_frequency=args.val_frequency,
                use_amp=not args.no_amp,
                teacher_forcing_ratio=args.teacher_forcing,
                teacher_forcing_decay=args.tf_decay,
                weight_decay=args.weight_decay,
                **model_kwargs
            )

            results[f"{family}_{model_type}"] = history

        except Exception as e:
            logger.error(f"Failed to train {family}-{model_type}: {str(e)}")
            continue

    # Summary
    logger.info("\n" + "="*60)
    logger.info("Training Summary")
    logger.info("="*60)

    if results:
        # Sort by best validation loss
        sorted_results = sorted(
            results.items(),
            key=lambda x: x[1]['best_val_loss']
        )

        logger.info("\nModel Performance (sorted by validation loss):")
        for i, (model_name, history) in enumerate(sorted_results, 1):
            logger.info(
                f"{i}. {model_name:30s} - Val Loss: {history['best_val_loss']:.6f}"
            )

        best_model = sorted_results[0][0]
        best_loss = sorted_results[0][1]['best_val_loss']
        logger.info(f"\n✓ Best model: {best_model} (Val Loss: {best_loss:.6f})")
    else:
        logger.warning("No models were successfully trained")

    logger.info("\n" + "="*60)
    logger.info("Training completed!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
