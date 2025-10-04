# Claude Development Notes

This document contains important guidelines and notes for AI assistants (like Claude) working on this codebase.

## Project Overview

**AutoStock Trader** is a sophisticated automated stock trading system using dual-model architecture:
- **Price Prediction Models**: LSTM, RNN/GRU, TCN, Transformers for forecasting
- **Action Decision Models**: Reinforcement learning (DQN/PPO) for trade execution
- **Data Pipeline**: Minute-level market data with 78 technical indicators

## Architecture & Design Principles

### 1. Data Pipeline Philosophy

- **Timeframe-First Design**: The system defaults to **minute-level data** for intraday predictions
- **Modular Components**: Data fetching (`fetcher.py`) and feature engineering (`features.py`) are independent
- **Multiple Formats**: Always save both CSV (human-readable) and Parquet (efficient storage)
- **Comprehensive Logging**: All operations logged to `logs/` directory with timestamps

### 2. Environment Configuration

**Critical `.env` Variables:**
```bash
ALPACA_API_KEY=<required>
ALPACA_SECRET_KEY=<required>
SYMBOLS=SPY,QQQ              # Primary and market symbols
START_DATE=2024-10-01        # For minute data: 1-6 months recommended
END_DATE=2024-12-31
TIMEFRAME=Minute             # Minute, Hour, Day, Week, Month
PREDICTION_STEPS=30          # How many steps ahead to predict
```

**Important Constraints:**
- Minute data: ~7,500 rows per trading day per symbol
- 3 months ≈ 45K rows | 6 months ≈ 90K rows | 1 year ≈ 180K rows
- Always test with small date ranges first before full dataset generation

### 3. Feature Engineering

**78 Features Generated (7 Categories):**
1. **OHLCV (7)**: Raw price/volume data from Alpaca
2. **Basic (8)**: Returns, gaps, intraday ranges
3. **Moving Averages (18)**: SMA/EMA for 5,10,20,50,200 periods
4. **Momentum (7)**: RSI, Stochastic, ROC, Williams %R
5. **Volatility (10)**: Bollinger Bands, ATR, Keltner Channels
6. **Trend (10)**: MACD, ADX, Aroon, CCI
7. **Volume (7)**: OBV, ADI, CMF, volume ratios
8. **Market Sentiment (7)**: QQQ correlation features

**Key Points:**
- Uses `ta` library for technical indicators
- Automatically drops NaN rows (first ~200 rows due to 200-period MA)
- Market sentiment requires both primary symbol (SPY) and market index (QQQ)
- All features are time-aligned on the same index

## Code Organization

```
autostock-trader/
├── src/
│   ├── data/
│   │   ├── fetcher.py          # Alpaca API integration
│   │   └── features.py         # Technical indicator calculation
│   ├── models/
│   │   ├── base_model.py       # Abstract base class for all models
│   │   ├── data_loader.py      # Time series data preparation
│   │   ├── lstm_models.py      # LSTM variants (4 models)
│   │   ├── gru_models.py       # GRU variants (3 models)
│   │   ├── tcn_models.py       # TCN variants (3 models)
│   │   ├── transformer_models.py # Transformer variants (3 models)
│   │   └── ensemble.py         # Ensemble methods (3 types)
│   └── __init__.py
├── models/
│   ├── checkpoints/             # Saved model weights
│   ├── plots/                   # Evaluation visualizations
│   └── logs/                    # Training logs
├── data/                        # Generated datasets (gitignored)
├── logs/                        # Application logs
├── venv/                        # Python virtual environment
├── generate_dataset.py          # Data pipeline orchestrator
├── train_models.py              # Model training script
├── evaluate_models.py           # Model evaluation script
├── test_models.py               # Model testing script
├── verify_setup.py              # Setup validation script
├── requirements.txt             # Python dependencies
├── .env                         # API credentials (gitignored)
└── .env.example                 # Template for configuration
```

## Important Development Guidelines

### When Adding New Features

1. **Always update both modules:**
   - Add calculation in `src/data/features.py`
   - Update feature count in documentation

2. **Test incrementally:**
   - Use small date ranges first (1-5 days)
   - Verify no NaN values introduced
   - Check memory usage for large datasets

3. **Maintain backwards compatibility:**
   - Don't change existing feature names
   - Don't reorder columns without reason
   - Keep metadata.json structure consistent

### When Modifying the Pipeline

1. **Preserve logging:**
   - Log all major operations
   - Use appropriate log levels (INFO for progress, ERROR for failures)
   - Include row counts and date ranges in logs

2. **Handle errors gracefully:**
   - Catch API failures (rate limits, network issues)
   - Validate date ranges before API calls
   - Provide clear error messages with solutions

3. **Memory considerations:**
   - Minute data can be memory-intensive (45K rows × 78 features)
   - Use parquet format for efficient storage
   - Consider chunking for very large date ranges

### When Working with Alpaca API

**Key Facts:**
- Free tier provides minute data for several years back
- Data includes pre/post market hours (timestamps in UTC)
- API returns multi-index DataFrames (symbol, timestamp)
- Rate limits exist - be mindful of request frequency

**Best Practices:**
- Always test API connection with `verify_setup.py` first
- Use small date ranges for initial testing
- Cache raw data before feature engineering
- Handle timezone conversions explicitly

## Testing Strategy

### Before Committing Code

1. **Run verification script:**
   ```bash
   python verify_setup.py
   ```

2. **Test with minimal data:**
   ```bash
   # Test with 1 week of data
   python generate_dataset.py --start-date 2024-12-23 --end-date 2024-12-31 --timeframe Minute
   ```

3. **Verify output files:**
   - Check `data/SPY_metadata.json` for correct row counts
   - Inspect `data/SPY_summary.csv` for feature statistics
   - Ensure no NaN values in final dataset

### Common Issues & Solutions

**Problem**: "No data after feature engineering (0 rows)"
- **Cause**: Date range too short for 200-day moving average
- **Solution**: Use at least 1 year of data OR reduce MA window sizes

**Problem**: "Memory error with large datasets"
- **Cause**: Loading entire dataset into memory
- **Solution**: Reduce date range OR implement chunked processing

**Problem**: "API rate limit exceeded"
- **Cause**: Too many rapid API requests
- **Solution**: Add delays between requests OR reduce symbol count

## Model Development Notes

### Prediction Models (Phase 2 - COMPLETE)

**Implemented Model Architectures:**

1. **LSTM Models (4 variants)** - `src/models/lstm_models.py`
   - BasicLSTM: Standard LSTM for sequence modeling
   - StackedLSTM: Deep multi-layer LSTM with dropout
   - BidirectionalLSTM: Processes sequences in both directions
   - AttentionLSTM: LSTM with attention mechanism for focusing on important time steps

2. **GRU Models (3 variants)** - `src/models/gru_models.py`
   - BasicGRU: Efficient alternative to LSTM
   - ResidualGRU: GRU with residual connections and layer normalization
   - BatchNormGRU: GRU with batch normalization for stable training

3. **TCN Models (3 variants)** - `src/models/tcn_models.py`
   - BasicTCN: Temporal Convolutional Network with dilated causal convolutions
   - MultiScaleTCN: Parallel TCNs with different kernel sizes (3, 5, 7)
   - ResidualTCN: TCN with skip connections between blocks

4. **Transformer Models (3 variants)** - `src/models/transformer_models.py`
   - BasicTransformer: Standard transformer with causal masking
   - MultiHeadTransformer: Pre-LN transformer with multiple attention heads
   - InformerTransformer: Informer-inspired with distilling operation

5. **Ensemble Methods (3 types)** - `src/models/ensemble.py`
   - SimpleEnsemble: Averages predictions from multiple models
   - WeightedEnsemble: Optimizes weights based on validation performance
   - StackingEnsemble: Meta-learner trained on base model predictions

**Key Implementation Details:**
- Base class: `src/models/base_model.py` (abstract class with common methods)
- Data loader: `src/models/data_loader.py` (time series sequence generation)
- Input shape: (batch_size, sequence_length=60, input_features=77)
- **Training mode**: Auto-regressive sequential prediction (all models)
  - Predicts 30 timesteps sequentially (like LLM token generation)
  - Each prediction feeds back as input for next step
  - Loss computed on ALL 30 predictions, not just final
- Target: Sequence of 30 future prices (configurable via PREDICTION_STEPS)
- Validation: Time-based splits (70/15/15 train/val/test), NOT random splits
- Early stopping with configurable patience
- Model checkpointing (saves best model during training)
- Metrics: MSE, RMSE, MAE, Directional Accuracy

**Training & Evaluation Scripts:**
- `train_models.py`: Train all or specific models with customizable hyperparameters
- `evaluate_models.py`: Comprehensive evaluation with visualizations
- `test_models.py`: Quick verification of all model architectures

**Usage Examples:**
```bash
# Train all models (auto-regressive by default)
python train_models.py

# Train with TensorBoard visualization (recommended)
python train_models.py --tensorboard

# Train specific models
python train_models.py --models lstm:attention gru:residual --tensorboard

# Customize training (all models predict 30 steps auto-regressively)
python train_models.py --epochs 100 --hidden-size 256 --tensorboard

# Evaluate with ensemble
python evaluate_models.py --evaluate-ensemble

# Quick test (2 epochs)
python test_models.py
```

**TensorBoard Integration:**

Training visualization is now fully integrated via TensorBoard:

- **Enable with**: `--tensorboard` flag during training
- **View with**: `python view_tensorboard.py` or `tensorboard --logdir=runs`
- **Logs tracked**: Training/validation loss, learning rate, gradient norms, weight histograms
- **Benefits**: Real-time monitoring, early detection of overfitting, model comparison
- **Directory structure**: `runs/{timestamp}/{model_family}_{model_type}/`

The base model automatically logs:
- Scalar metrics: Loss curves, learning rate, gradient norms
- Histograms: Model weights and gradients (every 10 epochs)
- Best validation loss tracking

All models inherit this functionality, no additional code needed.

### Action Decision Models (To Be Implemented)

**RL Agents:**
- DQN: Discrete actions (buy/sell/hold)
- PPO: Continuous actions (position sizing)

**Important:**
- Reward function must include transaction costs
- State space: Predicted prices + current portfolio + market sentiment
- Action space: Trade decisions with position sizing

## File Naming Conventions

**Data Files:**
- Raw data: `{SYMBOL}_raw.{csv|parquet}`
- Processed features: `{SYMBOL}_features.{csv|parquet}`
- Metadata: `{SYMBOL}_metadata.json`
- Statistics: `{SYMBOL}_summary.csv`

**Code Files:**
- Lowercase with underscores: `data_fetcher.py`
- Module names match directory: `src/data/`
- Test files prefix: `test_*.py`
- Script files: `*_dataset.py`, `verify_*.py`

## Dependencies Management

**Core Libraries:**
```
# Data Pipeline
alpaca-py>=0.42.2      # Alpaca API SDK
pandas>=2.3.3           # Data manipulation
numpy>=2.3.3            # Numerical computing
ta>=0.11.0              # Technical analysis indicators
python-dotenv>=1.1.1    # Environment management
pyarrow>=21.0.0         # Parquet file support

# Machine Learning (Phase 2)
torch>=2.0.0            # PyTorch deep learning framework
tensorboard>=2.18.0     # Training visualization dashboard
scikit-learn>=1.3.0     # ML utilities (scaling, metrics)
optuna>=3.0.0           # Hyperparameter optimization
matplotlib>=3.8.0       # Plotting and visualization
seaborn>=0.13.0         # Statistical visualizations
tqdm>=4.66.0            # Progress bars
scipy>=1.11.0           # Scientific computing
```

**When Adding Dependencies:**
1. Install in venv: `pip install <package>`
2. Update requirements: `pip freeze > requirements.txt`
3. Document why the dependency is needed
4. Check for conflicts with existing packages

## Security & API Keys

**Critical Rules:**
1. **NEVER commit `.env` file** (already in `.gitignore`)
2. **NEVER hardcode API keys** in source code
3. **Always use `python-dotenv`** to load credentials
4. **Rotate keys** if accidentally exposed
5. **Use paper trading** (`ALPACA_PAPER=true`) for development

## Performance Optimization Tips

### Data Fetching
- Fetch both symbols (SPY, QQQ) in single API call
- Use parquet format (3x smaller than CSV)
- Cache raw data before feature engineering

### Feature Engineering
- Vectorize calculations (avoid loops)
- Use pandas built-in functions when possible
- Drop NaN rows once at the end, not per feature

### Memory Management
- For datasets > 100K rows, consider:
  - Processing in monthly chunks
  - Using dask for out-of-core computation
  - Downsampling to hourly data

## Common Patterns

### Safe API Call Pattern
```python
try:
    fetcher = AlpacaDataFetcher()
    data = fetcher.fetch_stock_bars(
        symbols=['SPY', 'QQQ'],
        start_date='2024-12-01',
        end_date='2024-12-31',
        timeframe=TimeFrame.Minute
    )
except Exception as e:
    logger.error(f"API call failed: {str(e)}")
    raise
```

### Feature Addition Pattern
```python
def add_new_feature(self, df: pd.DataFrame) -> pd.DataFrame:
    """
    Add custom feature to dataframe.

    Args:
        df: DataFrame with OHLCV data

    Returns:
        DataFrame with added feature
    """
    logger.info("Adding new feature...")

    # Calculate feature
    df['new_feature'] = df['close'].rolling(window=10).mean()

    # Track it
    self.features_added.append('new_feature')

    return df
```

### Validation Pattern
```python
# Always validate outputs
assert len(df) > 0, "Dataset is empty"
assert df.isnull().sum().sum() == 0, "Dataset contains NaN values"
assert 'close' in df.columns, "Missing required column"
```

## Git Workflow

**Recommended Branches:**
- `main`: Stable, production-ready code
- `develop`: Integration branch for features
- `feature/*`: Individual feature development
- `hotfix/*`: Critical bug fixes

**Commit Messages:**
```
feat: Add minute-level data support
fix: Handle API rate limiting
docs: Update README with minute data info
refactor: Extract feature calculation logic
test: Add unit tests for fetcher
```

## Questions to Ask Before Making Changes

1. **Does this change affect existing datasets?**
   - Will users need to regenerate data?
   - Are feature names/order preserved?

2. **Is this change backwards compatible?**
   - Can old code read new data format?
   - Are new parameters optional?

3. **How will this scale?**
   - Works with 45K rows - what about 500K?
   - Memory usage acceptable?

4. **Is it properly tested?**
   - Manual test with small dataset done?
   - Edge cases considered?

5. **Is documentation updated?**
   - README reflects new features?
   - Docstrings are clear and accurate?

## Useful Commands

```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Phase 1: Data Pipeline
# ----------------------
# Verify setup
python verify_setup.py

# Generate dataset with different timeframes
python generate_dataset.py --timeframe Minute
python generate_dataset.py --timeframe Hour
python generate_dataset.py --timeframe Day

# Custom date range
python generate_dataset.py --start-date 2024-12-01 --end-date 2024-12-31

# Check dataset info
python -c "import json; print(json.dumps(json.load(open('data/SPY_metadata.json')), indent=2))"

# View data sample
python -c "import pandas as pd; df = pd.read_parquet('data/SPY_features.parquet'); print(df.head())"

# Check memory usage
python -c "import pandas as pd; df = pd.read_parquet('data/SPY_features.parquet'); print(f'{df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB')"

# Phase 2: Model Training & Evaluation
# -------------------------------------
# Quick test all models (2 epochs)
python test_models.py

# Train all models
python train_models.py

# Train with TensorBoard visualization (recommended!)
python train_models.py --tensorboard

# Train specific models
python train_models.py --models lstm:basic gru:residual --tensorboard

# Train with custom hyperparameters (all auto-regressive)
python train_models.py --epochs 100 --batch-size 64 --learning-rate 0.001 --hidden-size 256 --tensorboard

# View TensorBoard (in separate terminal while training)
python view_tensorboard.py
# Or manually: tensorboard --logdir=runs
# Open browser: http://localhost:6006

# Evaluate all trained models
python evaluate_models.py

# Evaluate with ensemble
python evaluate_models.py --evaluate-ensemble

# Evaluate specific models
python evaluate_models.py --models lstm:attention gru:residual

# Check model parameters
python -c "from src.models.lstm_models import BasicLSTM; model = BasicLSTM(77, 128); print(f'{model.count_parameters():,} parameters')"

# View training results
ls -lh models/checkpoints/*/
ls -lh models/plots/
ls -lh runs/  # TensorBoard logs
```

## Future Development Roadmap

### Phase 1: Data Pipeline ✅ (COMPLETE)
- [x] Alpaca API integration
- [x] Minute-level data fetching
- [x] 78 technical indicators
- [x] Market sentiment features

### Phase 2: Prediction Models ✅ (COMPLETE)
- [x] LSTM implementation (4 variants: Basic, Stacked, Bidirectional, Attention)
- [x] GRU variants (3 variants: Basic, Residual, BatchNorm)
- [x] TCN implementation (3 variants: Basic, MultiScale, Residual)
- [x] Transformer models (3 variants: Basic, MultiHead, Informer)
- [x] Ensemble methods (Simple, Weighted, Stacking)
- [x] Training & evaluation framework
- [ ] Hyperparameter optimization (Optuna)

### Phase 3: Action Models
- [ ] DQN agent
- [ ] PPO agent
- [ ] Policy networks
- [ ] Reward function design
- [ ] Risk management integration

### Phase 4: Backtesting
- [ ] Walk-forward validation
- [ ] Transaction cost modeling
- [ ] Performance metrics (Sharpe, Drawdown)
- [ ] Monte Carlo simulations

### Phase 5: Production
- [ ] Real-time data streaming
- [ ] Live trading execution
- [ ] Portfolio management
- [ ] Monitoring & alerting
- [ ] MLflow integration

## Contact & Support

- **Documentation**: See README.md for user-facing docs
- **Issues**: This is a local development project
- **Alpaca API Docs**: https://docs.alpaca.markets/

---

**Last Updated**: 2025-01-10
**Dataset Version**: v1.0 (Minute-level, 78 features, 1.67M rows)
**Models Version**: v1.0 (10 architectures + 3 ensembles)
**Python Version**: 3.12+
**PyTorch Version**: 2.0+
**Alpaca API Version**: alpaca-py 0.42.2