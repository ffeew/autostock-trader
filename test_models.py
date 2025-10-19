"""
Quick test script to verify all models work correctly.

Tests models with a small sample of data to ensure:
- Models can be instantiated
- Forward pass works
- Training loop runs without errors
"""

import os
import sys
import logging
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.data_loader import StockDataLoader
from src.models.lstm_models import create_lstm_model
from src.models.gru_models import create_gru_model
from src.models.tcn_models import create_tcn_model
from src.models.transformer_models import create_transformer_model

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def test_model(
    model_family: str,
    model_type: str,
    input_size: int,
    train_loader,
    val_loader,
    device: str = 'cpu'
) -> bool:
    """
    Test a single model.

    Args:
        model_family: Model family
        model_type: Model type
        input_size: Number of input features
        train_loader: Training data loader
        val_loader: Validation data loader
        device: Device to use

    Returns:
        True if test passed
    """
    try:
        logger.info(f"\nTesting {model_family}-{model_type}...")

        # Create model
        if model_family == 'lstm':
            model = create_lstm_model(
                model_type, input_size,
                hidden_size=64, num_layers=2, device=device
            )
        elif model_family == 'gru':
            model = create_gru_model(
                model_type, input_size,
                hidden_size=64, num_layers=2, device=device
            )
        elif model_family == 'tcn':
            model = create_tcn_model(
                model_type, input_size,
                hidden_size=64, num_layers=2, device=device
            )
        elif model_family == 'transformer':
            model = create_transformer_model(
                model_type, input_size,
                hidden_size=64, num_layers=2, num_heads=4, device=device
            )
        else:
            raise ValueError(f"Unknown model family: {model_family}")

        logger.info(f"  ✓ Model instantiated ({model.count_parameters():,} parameters)")

        # Test forward pass
        x, y = next(iter(train_loader))
        x = x.to(device)
        output = model(x)
        logger.info(f"  ✓ Forward pass successful (output shape: {output.shape})")

        # Test short training
        history = model.fit(
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=2,
            learning_rate=0.001,
            early_stopping_patience=5
        )
        logger.info(f"  ✓ Training completed (train loss: {history['train_losses'][-1]:.4f})")

        # Test prediction
        predictions, actuals = model.predict(val_loader)
        logger.info(f"  ✓ Prediction successful ({len(predictions)} samples)")

        logger.info(f"✓ {model_family}-{model_type} PASSED")
        return True

    except Exception as e:
        logger.error(f"✗ {model_family}-{model_type} FAILED: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def main():
    logger.info("="*60)
    logger.info("Model Testing Script")
    logger.info("="*60)

    # Setup device (use CPU for testing to avoid memory issues)
    device = 'cpu'
    logger.info(f"Using device: {device}")

    # Load small sample of data
    logger.info("\nLoading data...")
    data_loader = StockDataLoader(
        data_path='data/SPY_features.parquet',
        sequence_length=30,  # Shorter for faster testing
        batch_size=16  # Smaller batch for testing
    )

    train_loader, val_loader, test_loader, scaler, target_scaler, target_idx = data_loader.prepare_data()
    input_size = data_loader.n_features

    logger.info(f"✓ Data loaded (input_size={input_size}, target_idx={target_idx})")

    # Model configurations
    model_configs = {
        'lstm': ['basic', 'stacked', 'bidirectional', 'attention'],
        'gru': ['basic', 'residual', 'batchnorm'],
        'tcn': ['basic', 'multiscale', 'residual'],
        'transformer': ['basic', 'multihead', 'informer']
    }

    # Test all models
    results = {}
    passed = 0
    failed = 0

    logger.info("\n" + "="*60)
    logger.info("Running Tests")
    logger.info("="*60)

    for family, types in model_configs.items():
        for model_type in types:
            success = test_model(
                family, model_type,
                input_size, train_loader, val_loader,
                device=device
            )

            results[f"{family}_{model_type}"] = success
            if success:
                passed += 1
            else:
                failed += 1

    # Summary
    logger.info("\n" + "="*60)
    logger.info("Test Summary")
    logger.info("="*60)

    total = passed + failed
    logger.info(f"\nTotal: {total} models")
    logger.info(f"Passed: {passed} ({100*passed/total:.1f}%)")
    logger.info(f"Failed: {failed} ({100*failed/total:.1f}%)")

    if failed > 0:
        logger.info("\nFailed models:")
        for model_name, success in results.items():
            if not success:
                logger.info(f"  - {model_name}")

    logger.info("\n" + "="*60)
    if failed == 0:
        logger.info("✓ ALL TESTS PASSED!")
    else:
        logger.info("✗ SOME TESTS FAILED")
    logger.info("="*60)

    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
