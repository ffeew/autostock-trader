"""
Evaluation script for trained models.

Evaluates models on test set and generates visualizations.
"""

import os
import sys
import argparse
import logging
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from dotenv import load_dotenv
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from src.models.data_loader import StockDataLoader
from src.models.lstm_models import create_lstm_model
from src.models.gru_models import create_gru_model
from src.models.tcn_models import create_tcn_model
from src.models.transformer_models import create_transformer_model
from src.models.ensemble import create_ensemble

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)


def load_model(
    model_family: str,
    model_type: str,
    checkpoint_path: str,
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
):
    """
    Load a trained model from checkpoint.

    Args:
        model_family: One of ['lstm', 'gru', 'tcn', 'transformer']
        model_type: Specific model type
        checkpoint_path: Path to checkpoint
        device: Device to load on

    Returns:
        Loaded model
    """
    logger.info(f"Loading {model_family}-{model_type} from {checkpoint_path}")

    # Get model class
    if model_family == 'lstm':
        from src.models.lstm_models import BasicLSTM, StackedLSTM, BidirectionalLSTM, AttentionLSTM
        model_classes = {
            'basic': BasicLSTM,
            'stacked': StackedLSTM,
            'bidirectional': BidirectionalLSTM,
            'attention': AttentionLSTM
        }
    elif model_family == 'gru':
        from src.models.gru_models import BasicGRU, ResidualGRU, BatchNormGRU
        model_classes = {
            'basic': BasicGRU,
            'residual': ResidualGRU,
            'batchnorm': BatchNormGRU
        }
    elif model_family == 'tcn':
        from src.models.tcn_models import BasicTCN, MultiScaleTCN, ResidualTCN
        model_classes = {
            'basic': BasicTCN,
            'multiscale': MultiScaleTCN,
            'residual': ResidualTCN
        }
    elif model_family == 'transformer':
        from src.models.transformer_models import BasicTransformer, MultiHeadTransformer, InformerTransformer
        model_classes = {
            'basic': BasicTransformer,
            'multihead': MultiHeadTransformer,
            'informer': InformerTransformer
        }
    else:
        raise ValueError(f"Unknown model family: {model_family}")

    model_class = model_classes[model_type]
    model = model_class.load(checkpoint_path, device=device)

    logger.info("✓ Model loaded successfully")
    return model


def plot_predictions(
    predictions: np.ndarray,
    actuals: np.ndarray,
    model_name: str,
    save_path: str,
    n_samples: int = 500
):
    """
    Plot predictions vs actuals.

    Args:
        predictions: Model predictions
        actuals: Actual values
        model_name: Name of model
        save_path: Path to save plot
        n_samples: Number of samples to plot
    """
    # Take last n_samples for clarity
    pred_subset = predictions[-n_samples:].flatten()
    actual_subset = actuals[-n_samples:].flatten()

    fig, axes = plt.subplots(2, 1, figsize=(14, 10))

    # Plot 1: Time series comparison
    axes[0].plot(actual_subset, label='Actual', alpha=0.7, linewidth=1.5)
    axes[0].plot(pred_subset, label='Predicted', alpha=0.7, linewidth=1.5)
    axes[0].set_title(f'{model_name} - Predictions vs Actuals', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Time Step')
    axes[0].set_ylabel('Price')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Scatter plot
    axes[1].scatter(actual_subset, pred_subset, alpha=0.5, s=10)

    # Add diagonal line (perfect prediction)
    min_val = min(actual_subset.min(), pred_subset.min())
    max_val = max(actual_subset.max(), pred_subset.max())
    axes[1].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction', linewidth=2)

    axes[1].set_title('Prediction Scatter Plot', fontsize=14, fontweight='bold')
    axes[1].set_xlabel('Actual Price')
    axes[1].set_ylabel('Predicted Price')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved plot to {save_path}")


def plot_residuals(
    predictions: np.ndarray,
    actuals: np.ndarray,
    model_name: str,
    save_path: str
):
    """
    Plot residual analysis.

    Args:
        predictions: Model predictions
        actuals: Actual values
        model_name: Name of model
        save_path: Path to save plot
    """
    residuals = actuals.flatten() - predictions.flatten()

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Residuals over time
    axes[0, 0].plot(residuals, alpha=0.5, linewidth=0.5)
    axes[0, 0].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[0, 0].set_title('Residuals Over Time', fontsize=12, fontweight='bold')
    axes[0, 0].set_xlabel('Time Step')
    axes[0, 0].set_ylabel('Residual')
    axes[0, 0].grid(True, alpha=0.3)

    # Plot 2: Residual histogram
    axes[0, 1].hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    axes[0, 1].set_title('Residual Distribution', fontsize=12, fontweight='bold')
    axes[0, 1].set_xlabel('Residual')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].grid(True, alpha=0.3)

    # Plot 3: Q-Q plot
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot', fontsize=12, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)

    # Plot 4: Residuals vs predictions
    axes[1, 1].scatter(predictions.flatten(), residuals, alpha=0.5, s=10)
    axes[1, 1].axhline(y=0, color='r', linestyle='--', linewidth=2)
    axes[1, 1].set_title('Residuals vs Predictions', fontsize=12, fontweight='bold')
    axes[1, 1].set_xlabel('Predicted Value')
    axes[1, 1].set_ylabel('Residual')
    axes[1, 1].grid(True, alpha=0.3)

    fig.suptitle(f'{model_name} - Residual Analysis', fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"  Saved residual plot to {save_path}")


def evaluate_single_model(
    model,
    test_loader,
    model_name: str,
    plots_dir: str
) -> Dict[str, float]:
    """
    Evaluate a single model.

    Args:
        model: Trained model
        test_loader: Test data loader
        model_name: Name of model
        plots_dir: Directory to save plots

    Returns:
        Dictionary of metrics
    """
    logger.info(f"\nEvaluating {model_name}...")

    # Get predictions
    predictions, actuals = model.predict(test_loader)

    # Calculate metrics
    metrics = model.evaluate(test_loader)

    # Generate plots
    os.makedirs(plots_dir, exist_ok=True)

    plot_predictions(
        predictions, actuals, model_name,
        os.path.join(plots_dir, f'{model_name}_predictions.png')
    )

    plot_residuals(
        predictions, actuals, model_name,
        os.path.join(plots_dir, f'{model_name}_residuals.png')
    )

    return metrics


def compare_models(results: Dict[str, Dict[str, float]], save_path: str):
    """
    Create comparison plots for all models.

    Args:
        results: Dictionary mapping model names to metrics
        save_path: Path to save comparison plot
    """
    metrics_to_plot = ['rmse', 'mae', 'directional_accuracy']

    fig, axes = plt.subplots(1, len(metrics_to_plot), figsize=(18, 6))

    for idx, metric in enumerate(metrics_to_plot):
        model_names = list(results.keys())
        values = [results[name][metric] for name in model_names]

        axes[idx].bar(range(len(model_names)), values, alpha=0.7)
        axes[idx].set_xticks(range(len(model_names)))
        axes[idx].set_xticklabels(model_names, rotation=45, ha='right')
        axes[idx].set_title(metric.upper().replace('_', ' '), fontsize=12, fontweight='bold')
        axes[idx].set_ylabel('Value')
        axes[idx].grid(True, alpha=0.3, axis='y')

        # Add value labels on bars
        for i, v in enumerate(values):
            axes[idx].text(i, v, f'{v:.4f}', ha='center', va='bottom', fontsize=8)

    plt.suptitle('Model Comparison', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"Saved comparison plot to {save_path}")


def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/SPY_features.parquet',
        help='Path to feature dataset'
    )
    parser.add_argument(
        '--checkpoints-dir',
        type=str,
        default='models/checkpoints',
        help='Directory containing model checkpoints'
    )
    parser.add_argument(
        '--plots-dir',
        type=str,
        default='models/plots',
        help='Directory to save plots'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['all'],
        help='Models to evaluate (e.g., lstm:basic gru:residual) or "all"'
    )
    parser.add_argument(
        '--sequence-length',
        type=int,
        default=60,
        help='Sequence length (must match training)'
    )
    parser.add_argument(
        '--batch-size',
        type=int,
        default=32,
        help='Batch size'
    )
    parser.add_argument(
        '--evaluate-ensemble',
        action='store_true',
        help='Also evaluate ensemble of all models'
    )

    args = parser.parse_args()

    # Load environment
    load_dotenv()

    # Setup device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info(f"Using device: {device}")

    # Load data
    logger.info("\n" + "="*60)
    logger.info("Loading data")
    logger.info("="*60)

    data_loader = StockDataLoader(
        data_path=args.data_path,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size
    )

    train_loader, val_loader, test_loader, scaler = data_loader.prepare_data()

    logger.info("✓ Data loaded\n")

    # Find available checkpoints
    available_models = []
    for family in ['lstm', 'gru', 'tcn', 'transformer']:
        checkpoint_dir = os.path.join(args.checkpoints_dir, family)
        if os.path.exists(checkpoint_dir):
            for file in os.listdir(checkpoint_dir):
                if file.endswith('_best.pt'):
                    model_type = file.replace('_best.pt', '')
                    available_models.append((family, model_type))

    logger.info(f"Found {len(available_models)} trained models")

    # Determine which models to evaluate
    if 'all' in args.models:
        models_to_eval = available_models
    else:
        models_to_eval = []
        for model_spec in args.models:
            if ':' in model_spec:
                family, model_type = model_spec.split(':')
                if (family, model_type) in available_models:
                    models_to_eval.append((family, model_type))
                else:
                    logger.warning(f"Checkpoint not found for {family}:{model_type}")

    if not models_to_eval:
        logger.error("No models to evaluate!")
        return

    logger.info(f"Evaluating {len(models_to_eval)} model(s)\n")

    # Evaluate each model
    results = {}
    loaded_models = {}

    for family, model_type in models_to_eval:
        try:
            checkpoint_path = os.path.join(
                args.checkpoints_dir, family, f'{model_type}_best.pt'
            )

            model = load_model(family, model_type, checkpoint_path, device)
            model_name = f"{family}_{model_type}"

            metrics = evaluate_single_model(
                model, test_loader, model_name, args.plots_dir
            )

            results[model_name] = metrics
            loaded_models[model_name] = model

        except Exception as e:
            logger.error(f"Failed to evaluate {family}-{model_type}: {str(e)}")
            continue

    # Generate comparison plots
    if len(results) > 1:
        logger.info("\nGenerating comparison plots...")
        compare_models(
            results,
            os.path.join(args.plots_dir, 'model_comparison.png')
        )

    # Evaluate ensemble
    if args.evaluate_ensemble and len(loaded_models) > 1:
        logger.info("\n" + "="*60)
        logger.info("Evaluating Ensemble Models")
        logger.info("="*60)

        models_list = list(loaded_models.values())

        # Simple ensemble
        logger.info("\nSimple Ensemble (averaging)...")
        simple_ensemble = create_ensemble('simple', models_list)
        ensemble_metrics = simple_ensemble.evaluate(test_loader)
        results['ensemble_simple'] = ensemble_metrics

        # Weighted ensemble
        logger.info("\nWeighted Ensemble (optimized weights)...")
        weighted_ensemble = create_ensemble('weighted', models_list, val_loader=val_loader, device=device)
        ensemble_metrics = weighted_ensemble.evaluate(test_loader)
        results['ensemble_weighted'] = ensemble_metrics

    # Summary
    logger.info("\n" + "="*60)
    logger.info("Evaluation Summary")
    logger.info("="*60)

    # Create summary DataFrame
    df = pd.DataFrame(results).T
    df = df.sort_values('rmse')

    logger.info("\nResults (sorted by RMSE):")
    logger.info(df.to_string())

    # Save results to CSV
    results_path = os.path.join(args.plots_dir, 'evaluation_results.csv')
    df.to_csv(results_path)
    logger.info(f"\n✓ Results saved to {results_path}")

    # Best model
    best_model = df.index[0]
    logger.info(f"\n✓ Best model: {best_model}")
    logger.info(f"  RMSE: {df.loc[best_model, 'rmse']:.6f}")
    logger.info(f"  MAE: {df.loc[best_model, 'mae']:.6f}")
    logger.info(f"  Directional Accuracy: {df.loc[best_model, 'directional_accuracy']:.4f}")

    logger.info("\n" + "="*60)
    logger.info("Evaluation completed!")
    logger.info("="*60)


if __name__ == '__main__':
    main()
