# 📈 AutoStock Trader

> A sophisticated automated stock trading system leveraging deep learning for minute-level price predictions and reinforcement learning for intelligent trade execution.

[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Data Pipeline](https://img.shields.io/badge/Phase%201-Complete-brightgreen.svg)](https://github.com)
[![Prediction Models](https://img.shields.io/badge/Phase%202-Complete-brightgreen.svg)](https://github.com)

## 🎯 Overview

AutoStock Trader is an AI-powered trading system that combines:

- **⏱️ Minute-Level Predictions**: Intraday forecasting using 1-minute bar data
- **🤖 Dual-Model Architecture**: Separate models for price prediction and action decision-making
- **📊 Rich Feature Engineering**: 78 technical indicators per time step
- **🎲 Ensemble Methods**: Multiple model architectures working together
- **📉 QQQ Market Windvane**: High-beta QQQ features (0.8053 correlation) as market sentiment indicators

**Current Status**: Phase 1 (Data Pipeline) & Phase 2 (Prediction Models) are complete with **1.67M rows** of minute-level data from 2016-2025 and **10 model architectures** ready for training.

---

## ✨ Key Features

### Data Pipeline (✅ **COMPLETE**)

- ✅ Alpaca API integration for real-time and historical data
- ✅ Minute-level data fetching (2016-2025: 1,674,536 bars)
- ✅ 78 technical indicators automatically calculated
- ✅ **QQQ windvane features** (7 features: returns, RSI, volatility, correlation)
- ✅ Strong QQQ-SPY correlation (0.8053) validates market sentiment hypothesis
- ✅ Multiple timeframe support (Minute, Hour, Day, Week, Month)
- ✅ Efficient storage (Parquet format: ~600 MB for 1.67M rows)

### Prediction Models (✅ **COMPLETE**)

- ✅ LSTM variants (Basic, Stacked, Bidirectional, Attention)
- ✅ GRU variants (Basic, Residual, BatchNorm)
- ✅ Temporal Convolutional Networks (Basic, MultiScale, Residual TCN)
- ✅ Transformer models (Basic, MultiHead, Informer-inspired)
- ✅ Ensemble methods (Simple averaging, Weighted, Stacking)
- ✅ Training & evaluation framework with early stopping
- 🔜 Hyperparameter optimization (Optuna)

### Action Decision Models (🔜 **PLANNED**)

- 🔜 Deep Q-Network (DQN) for discrete actions
- 🔜 Proximal Policy Optimization (PPO) for continuous control
- 🔜 Risk-adjusted reward functions
- 🔜 Transaction cost modeling
- 🔜 Position sizing strategies

### Backtesting & Risk Management (🔜 **PLANNED**)

- 🔜 Walk-forward validation
- 🔜 Sharpe ratio, max drawdown, win rate
- 🔜 Monte Carlo simulations
- 🔜 Portfolio management with correlation limits

---

## 🚀 Quick Start

### Prerequisites

- Python 3.12 or higher
- Alpaca Markets account ([sign up free](https://alpaca.markets/))
- 2GB+ RAM for minute-level data processing

### Installation

```bash
# Clone the repository
git clone https://github.com/ffeew/autostock-trader
cd autostock-trader

# Create virtual environment
python -m venv venv

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Configuration

1. **Get Alpaca API Keys**:

   - Sign up at [Alpaca Markets](https://alpaca.markets/)
   - Navigate to your dashboard and generate API keys
   - Use paper trading keys for testing

2. **Configure Environment**:

   ```bash
   # Copy example configuration
   cp .env.example .env

   # Edit .env with your API credentials
   nano .env  # or use your preferred editor
   ```

   Your `.env` should look like:

   ```bash
   ALPACA_API_KEY=your_api_key_here
   ALPACA_SECRET_KEY=your_secret_key_here
   ALPACA_PAPER=true

   SYMBOLS=SPY,QQQ
   START_DATE=2024-10-01
   END_DATE=2024-12-31
   TIMEFRAME=Minute
   PREDICTION_STEPS=30
   ```

3. **Verify Setup**:
   ```bash
   python verify_setup.py
   ```

### Generate Your First Dataset

```bash
# Generate 3 months of minute-level data (default)
python generate_dataset.py

# Or specify custom parameters
python generate_dataset.py \
    --symbols SPY,QQQ \
    --start-date 2024-10-01 \
    --end-date 2024-12-31 \
    --timeframe Minute

# Use different timeframes
python generate_dataset.py --timeframe Day    # Daily bars
python generate_dataset.py --timeframe Hour   # Hourly bars
```

**Expected output**:

```
✓ Fetching data from Alpaca API...
✓ Adding 78 technical indicators...
✓ Saving to data/SPY_features.parquet...
✓ Dataset generation complete!

Dataset: 45,062 rows × 78 features
Size: ~28 MB (parquet)
Date range: 2024-10-01 to 2024-12-31
```

---

# <<<<<<< Updated upstream

## 🤖 Training Models (Phase 2)

### Train All Models

**All models use auto-regressive sequential prediction** (like LLM token generation):
- Predicts 30 timesteps sequentially
- Each prediction feeds back as input for the next step
- Loss computed on ALL 30 predictions

```bash
# Train all model architectures (10 models)
python train_models.py

# Train with TensorBoard visualization (recommended!)
python train_models.py --tensorboard

# Train specific models only
python train_models.py --models lstm:basic gru:residual tcn:multiscale

# Customize training parameters
python train_models.py \
    --epochs 100 \
    --batch-size 64 \
    --learning-rate 0.001 \
    --hidden-size 256 \
    --early-stopping 15 \
    --tensorboard
```

**Available models**:

- LSTM: `basic`, `stacked`, `bidirectional`, `attention`
- GRU: `basic`, `residual`, `batchnorm`
- TCN: `basic`, `multiscale`, `residual`
- Transformer: `basic`, `multihead`, `informer`

### Evaluate Models

```bash
# Evaluate all trained models
python evaluate_models.py

# Evaluate specific models
python evaluate_models.py --models lstm:attention gru:residual

# Include ensemble evaluation
python evaluate_models.py --evaluate-ensemble
```

**Output includes**:

- Test set metrics (RMSE, MAE, Directional Accuracy)
- Prediction vs actual plots
- Residual analysis
- Model comparison charts
- Results saved to `models/plots/`

### Quick Test

```bash
# Test all models with 2 epochs (fast verification)
python test_models.py
```

### TensorBoard Visualization

**Monitor training in real-time:**

```bash
# 1. Start training with TensorBoard logging
python train_models.py --tensorboard

# 2. In a separate terminal, launch TensorBoard
python view_tensorboard.py

# Or manually:
tensorboard --logdir=runs
```

Then open your browser to **http://localhost:6006**

**Features:**

- 📈 Real-time loss curves (train vs validation)
- 📊 Gradient flow analysis
- 🔍 Weight/bias distributions
- 🔄 Compare multiple models side-by-side
- 💾 Export visualizations

---

> > > > > > > Stashed changes

## 📊 Current Dataset

**Generated Dataset Statistics** (as of latest generation):

| Metric           | Value                                 |
| ---------------- | ------------------------------------- |
| **Date Range**   | 2016-01-04 to 2025-08-29              |
| **Total Rows**   | 1,674,536 minutes                     |
| **Features**     | 78 per time step                      |
| **Symbols**      | SPY (primary), QQQ (market sentiment) |
| **File Size**    | ~600 MB (parquet), ~1.2 GB (CSV)      |
| **Trading Days** | ~2,400 days                           |
| **Avg Bars/Day** | ~700 minutes                          |

---

## 🎨 Architecture Overview

### Dual-Model Design

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT DATA                           │
│          (Minute-level OHLCV + 78 features)             │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│            PRICE PREDICTION MODELS (Type 1)             │
│  ┌─────────┐  ┌─────────┐  ┌─────────┐  ┌──────────┐    │
│  │  LSTM   │  │   GRU   │  │   TCN   │  │Transform.│    │
│  └────┬────┘  └────┬────┘  └────┬────┘  └────┬─────┘    │
│       └────────────┴────────────┴────────────┘          │
│                 Ensemble Predictions                    │
└────────────────┬────────────────────────────────────────┘
                 │
                 ▼
┌─────────────────────────────────────────────────────────┐
│          ACTION DECISION MODELS (Type 2)                │
│  ┌─────────────────────┐  ┌──────────────────────────┐  │
│  │  DQN Agent          │  │  PPO Agent               │  │
│  │  (Discrete Actions) │  │  (Continuous Control)    │  │
│  └─────────┬───────────┘  └───────────┬──────────────┘  │
└────────────┼──────────────────────────┼─────────────────┘
             │                          │
             ▼                          ▼
      ┌───────────┐              ┌──────────┐
      │ BUY/SELL  │              │ Position │
      │   HOLD    │              │  Sizing  │
      └───────────┘              └──────────┘
```

### Model Components

#### 1. Price Prediction Models

**LSTM (Long Short-Term Memory)**:

- Captures long-term dependencies in price movements
- Variants: Basic, Stacked, Bidirectional
- Attention mechanisms for focus on important time steps
- **Auto-regressive**: Predicts 30-step sequences iteratively

**GRU (Gated Recurrent Unit)**:

- More efficient than LSTM with similar performance
- Enhanced with residual connections
- Batch normalization for stable training
- **Auto-regressive**: Sequential 30-step prediction

**TCN (Temporal Convolutional Network)**:

- Parallel processing of sequences
- Multi-scale receptive fields
- Excellent for minute-level data
- **Auto-regressive**: Iterative multi-step forecasting

**Transformer**:

- Self-attention mechanisms
- Positional encoding for temporal information
- Causal attention for realistic predictions
- **Auto-regressive**: Like GPT, predicts sequences step-by-step

**Ensemble Methods**:

- Weighted averaging based on recent performance
- Regime-aware ensembles (bull/bear/sideways)
- Stacking with meta-learner
- **All use auto-regressive base models**

#### 2. Action Decision Models

**DQN (Deep Q-Network)**:

- Discrete action space: Buy, Sell, Hold
- Experience replay for stable learning
- Target network for convergence

**PPO (Proximal Policy Optimization)**:

- Continuous action space: Position sizing
- Clipped objective for stable updates
- Actor-critic architecture

**Reward Function**:

```python
reward = profit - transaction_costs - risk_penalty
risk_penalty = max_drawdown_weight * drawdown + volatility_weight * volatility
```

---

## 📈 Data Pipeline Details

### Feature Engineering (78 Features)

#### OHLCV Data (7 features)

- `open`, `high`, `low`, `close`: Price data
- `volume`: Trading volume
- `trade_count`: Number of trades
- `vwap`: Volume-weighted average price

#### Basic Features (8 features)

- `returns`: Percentage returns
- `log_returns`: Log returns
- `high_low_range`: High-low price range
- `high_low_pct`: Range as percentage of close
- `gap`, `gap_pct`: Opening gaps
- `intraday_change`, `intraday_change_pct`: Close - Open

#### Moving Averages (18 features)

- SMA: 5, 10, 20, 50, 200 periods
- EMA: 5, 10, 20, 50, 200 periods
- Price distance to each MA
- Golden cross (50/200) and other crossovers

#### Momentum Indicators (7 features)

- `rsi_14`, `rsi_7`: Relative Strength Index
- `stoch_k`, `stoch_d`: Stochastic Oscillator
- `roc_10`, `roc_20`: Rate of Change
- `williams_r`: Williams %R

#### Volatility Indicators (10 features)

- Bollinger Bands: `bb_high`, `bb_mid`, `bb_low`, `bb_width`, `bb_position`
- `atr_14`: Average True Range
- Keltner Channels: `keltner_high`, `keltner_low`
- Historical volatility: `volatility_10`, `volatility_20`

#### Trend Indicators (10 features)

- MACD: `macd`, `macd_signal`, `macd_diff`
- ADX: `adx`, `adx_pos`, `adx_neg`
- Aroon: `aroon_up`, `aroon_down`, `aroon_indicator`
- `cci`: Commodity Channel Index

#### Volume Indicators (7 features)

- `volume_change`, `volume_ma_20`, `volume_ratio`
- `obv`: On-Balance Volume
- `adi`: Accumulation/Distribution Index
- `cmf`: Chaikin Money Flow
- `close_to_vwap`: Price distance to VWAP

#### Market Sentiment (7 features)

- `market_close`: QQQ price
- `market_returns`: QQQ returns
- `market_volume`: QQQ volume
- `market_relative_returns`: SPY vs QQQ performance
- `market_rsi`: QQQ momentum
- `market_volatility`: QQQ volatility
- `market_correlation_20`: 20-period correlation

### Timeframe Options

| Timeframe  | Bars/Day | Best For              | Data Size (1 year) |
| ---------- | -------- | --------------------- | ------------------ |
| **Minute** | ~390     | Intraday trading, HFT | ~100K rows         |
| **Hour**   | ~6.5     | Swing trading         | ~1.6K rows         |
| **Day**    | 1        | Position trading      | ~252 rows          |
| **Week**   | 0.2      | Long-term holds       | ~52 rows           |

**Recommendations**:

- **Minute**: 3-12 months for practical training
- **Hour**: 2-5 years for swing strategies
- **Day**: 10-20 years for position trading

### Data Quality

- ✅ No missing values (NaN rows dropped)
- ✅ Timezone-aware timestamps (UTC)
- ✅ Includes pre/post market data
- ✅ Validated against known market events
- ✅ Consistent feature scaling

---

## 🗂️ Project Structure

```
autostock-trader/
│
├── src/                          # Source code
│   ├── __init__.py
│   └── data/                     # Data pipeline modules
│       ├── __init__.py
│       ├── fetcher.py           # Alpaca API integration
│       └── features.py          # Technical indicator calculation
│
├── data/                         # Generated datasets (gitignored)
│   ├── SPY_raw.parquet          # Raw OHLCV data
│   ├── SPY_features.parquet     # Processed features
│   ├── SPY_metadata.json        # Dataset metadata
│   ├── SPY_summary.csv          # Statistical summary
│   ├── QQQ_raw.parquet          # Market index data
│   └── QQQ_features.parquet
│
├── logs/                         # Application logs
│   └── dataset_generation.log
│
├── venv/                         # Virtual environment (gitignored)
│
├── generate_dataset.py           # Main dataset generation script
├── verify_setup.py              # Setup validation script
├── requirements.txt             # Python dependencies
├── .env                         # API credentials (gitignored)
├── .env.example                 # Configuration template
├── claude.md                    # Development notes for AI assistants
└── README.md                    # This file
```

### Key Files

**`generate_dataset.py`**: Main orchestrator

- Fetches data from Alpaca API
- Calculates all 78 features
- Saves to multiple formats
- Generates metadata and statistics

**`src/data/fetcher.py`**: API integration

- `AlpacaDataFetcher`: Main client class
- `fetch_stock_bars()`: Retrieve historical data
- `save_data()`, `load_data()`: Persistence

**`src/data/features.py`**: Feature engineering

- `FeatureEngineer`: Calculator class
- Individual indicator methods
- `create_features()`: Convenience function

**`verify_setup.py`**: Validation

- Checks dependencies
- Validates API credentials
- Tests data fetching
- Verifies generated files

---

## 🛠️ Development Roadmap

### ✅ Phase 1: Data Pipeline (COMPLETE)

- [x] Alpaca API integration
- [x] Minute-level data fetching
- [x] 78 technical indicators
- [x] Market sentiment features (QQQ)
- [x] Multiple timeframe support
- [x] Data validation and cleaning
- [x] Efficient storage (Parquet)
- [x] Comprehensive logging

**Deliverable**: 1.67M rows of high-quality training data

### 🔜 Phase 2: Prediction Models (NEXT)

- [ ] LSTM implementation
  - [ ] Basic LSTM
  - [ ] Stacked LSTM
  - [ ] Bidirectional LSTM
  - [ ] LSTM with attention
- [ ] GRU implementation
  - [ ] Basic GRU
  - [ ] GRU with residual connections
- [ ] TCN implementation
  - [ ] Multi-scale TCN
  - [ ] Residual TCN
- [ ] Transformer implementation
  - [ ] Positional encoding
  - [ ] Causal self-attention
- [ ] Ensemble methods
  - [ ] Weighted averaging
  - [ ] Adaptive/regime-aware
  - [ ] Stacking ensemble
- [ ] Hyperparameter optimization (Optuna)
- [ ] Model evaluation framework

**Target**: Accurate 30-minute ahead predictions

### 🔜 Phase 3: Action Decision Models

- [ ] Environment setup
  - [ ] State space definition
  - [ ] Action space design
  - [ ] Reward function implementation
- [ ] DQN agent
  - [ ] Experience replay
  - [ ] Target network
  - [ ] Double DQN variant
- [ ] PPO agent
  - [ ] Actor-critic architecture
  - [ ] Clipped objective
  - [ ] Advantage estimation
- [ ] Risk management
  - [ ] Position sizing algorithms
  - [ ] Stop-loss mechanisms
  - [ ] Correlation limits

**Target**: Profitable trading strategies

### 🔜 Phase 4: Backtesting

- [ ] Backtesting framework
  - [ ] Walk-forward validation
  - [ ] Time-based splits
- [ ] Transaction cost modeling
  - [ ] Slippage estimation
  - [ ] Commission structure
  - [ ] Bid-ask spread
- [ ] Performance metrics
  - [ ] Sharpe ratio
  - [ ] Maximum drawdown
  - [ ] Win rate, profit factor
  - [ ] Calmar ratio, alpha/beta
- [ ] Monte Carlo simulations
- [ ] Stress testing

**Target**: Validated, robust strategies

### 🔜 Phase 5: Production Deployment

- [ ] Real-time data streaming
- [ ] Live trading execution
- [ ] Portfolio management
- [ ] Monitoring and alerting
- [ ] MLflow integration
- [ ] API endpoints
- [ ] Web dashboard

**Target**: Fully automated trading system

---

## 🐛 Troubleshooting

### Common Issues

#### "API key not found"

**Problem**: Missing or invalid Alpaca API credentials

**Solution**:

```bash
# Check .env file exists
ls -la .env

# Verify credentials are set (without showing values)
grep "ALPACA_API_KEY" .env
grep "ALPACA_SECRET_KEY" .env

# Re-generate keys from Alpaca dashboard if needed
```

#### "No data after feature engineering (0 rows)"

**Problem**: Date range too short for 200-period moving average

**Solution**:

- Use at least 1 year of data for daily timeframe
- Or use smaller MA windows (edit `src/data/features.py`)
- Or use minute/hour timeframes (less data needed)

#### "Memory error with large datasets"

**Problem**: Loading 1M+ rows into memory

**Solution**:

```python
# Option 1: Reduce date range
python generate_dataset.py --start-date 2024-01-01 --end-date 2024-12-31

# Option 2: Use daily timeframe instead of minute
python generate_dataset.py --timeframe Day

# Option 3: Process in chunks (modify generate_dataset.py)
```

#### "API rate limit exceeded"

**Problem**: Too many rapid API requests

**Solution**:

- Alpaca free tier has rate limits
- Add delays between requests
- Reduce symbol count
- Upgrade Alpaca account for higher limits

#### "Import error: No module named 'alpaca'"

**Problem**: Dependencies not installed

**Solution**:

```bash
# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
pip install -r requirements.txt

# Verify installation
python -c "import alpaca; print('Success!')"
```

---

## 📖 Usage Examples

### Generate Different Timeframes

```bash
# Minute-level (default)
python generate_dataset.py --timeframe Minute

# Hourly data for swing trading
python generate_dataset.py \
    --start-date 2020-01-01 \
    --end-date 2024-12-31 \
    --timeframe Hour

# Daily data for long-term analysis
python generate_dataset.py \
    --start-date 2010-01-01 \
    --end-date 2024-12-31 \
    --timeframe Day
```

### Inspect Generated Data

```bash
# View metadata
cat data/SPY_metadata.json | python -m json.tool

# Load and inspect with Python
python -c "
import pandas as pd
df = pd.read_parquet('data/SPY_features.parquet')
print(f'Shape: {df.shape}')
print(f'\\nColumns:\\n{df.columns.tolist()}')
print(f'\\nFirst row:\\n{df.head(1)}')
print(f'\\nMemory: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB')
"

# View summary statistics
head -20 data/SPY_summary.csv
```

### Programmatic Access

```python
from src.data.fetcher import AlpacaDataFetcher
from src.data.features import create_features
from alpaca.data.timeframe import TimeFrame

# Fetch data
fetcher = AlpacaDataFetcher()
raw_data = fetcher.fetch_stock_bars(
    symbols=['SPY', 'QQQ'],
    start_date='2024-10-01',
    end_date='2024-12-31',
    timeframe=TimeFrame.Minute
)

# Generate features
dataset = create_features(
    raw_data,
    primary_symbol='SPY',
    market_symbol='QQQ'
)

print(f"Dataset shape: {dataset.shape}")
print(f"Features: {dataset.columns.tolist()}")
```

---

## 📚 References & Resources

### Alpaca API

- [Alpaca Documentation](https://docs.alpaca.markets/)
- [Python SDK](https://github.com/alpacahq/alpaca-py)
- [Market Data API](https://docs.alpaca.markets/docs/market-data)

### Technical Analysis

- [`ta` Library Documentation](https://technical-analysis-library-in-python.readthedocs.io/)
- Moving Averages, RSI, MACD implementations

### Machine Learning

- LSTM: [Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/)
- TCN: [Temporal Convolutional Networks](https://arxiv.org/abs/1803.01271)
- Transformers: [Attention Is All You Need](https://arxiv.org/abs/1706.03762)

### Reinforcement Learning

- DQN: [Playing Atari with Deep RL](https://arxiv.org/abs/1312.5602)
- PPO: [Proximal Policy Optimization](https://arxiv.org/abs/1707.06347)

---

## 🤝 Contributing

This is currently a personal project. Contributions, suggestions, and feedback are welcome!

### Development Setup

```bash
# Clone repository
git clone https://github.com/ffeew/autostock-trader
cd autostock-trader

# Create development environment
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Run verification
python verify_setup.py

# Generate test dataset
python generate_dataset.py --start-date 2024-12-01 --end-date 2024-12-31
```

### Code Style

- Follow PEP 8 guidelines
- Use type hints where appropriate
- Add docstrings to all functions
- Include logging for major operations

---

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice
- Past performance does not guarantee future results
- Trading involves substantial risk of loss
- Use paper trading for testing
- Consult with a financial advisor before live trading

**The author is not responsible for any financial losses incurred through the use of this software.**

---

## 📄 License

MIT License - see LICENSE file for details

---

## 📧 Contact & Support

- **Issues**: Open a GitHub issue for bugs or feature requests
- **Documentation**: See `claude.md` for detailed development notes
- **API Support**: [Alpaca Support](https://alpaca.markets/support)

---

## 🙏 Acknowledgments

- [Alpaca Markets](https://alpaca.markets/) for providing excellent free market data API
- [TA-Lib](https://technical-analysis-library-in-python.readthedocs.io/) for technical analysis indicators
- Open source Python ecosystem (pandas, numpy, scikit-learn)

---

**Last Updated**: 2024-10-01
**Version**: 1.0.0 (Phase 1 Complete)
**Dataset Version**: 1.67M rows (2016-2025, minute-level)
