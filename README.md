# AutoStock Trader

A sophisticated automated stock trading system that leverages a dual-model architecture: multiple time series prediction models for price forecasting and action decision models for trade execution. The system integrates high-beta index signals for enhanced market sentiment analysis and uses ensemble learning for robust predictions.

## 🚀 Features

- **Dual-Model Architecture**: Price prediction models coupled with action decision models for optimal trade execution
- **Multiple Model Architectures**: LSTM, RNN/GRU, TCN (Temporal Convolutional Networks), and Transformers
- **Ensemble Learning**: Weighted, regime-aware, and stacking ensembles with dynamic rebalancing
- **High-Beta Integration**: Uses high-beta indices (QQQ) as market sentiment signals to improve predictions
- **Hyperparameter Optimization**: Optuna integration for automated tuning of model parameters
- **Data Pipeline**: Fetches and processes data from Alpaca API with technical indicators
- **Market Regime Detection**: Bull/bear/sideways market classification for adaptive strategies
- **Risk Management**: Position sizing, stop-loss, and correlation limits
- **Experiment Tracking**: MLflow integration for model versioning and comparison
- **Production Ready**: Comprehensive logging, error handling, and security measures

## 🎯 Dual-Model Architecture

### Price Prediction Models (Type 1)

#### Individual Models

- **LSTM**: Basic, stacked, and bidirectional variants with attention mechanisms
- **RNN/GRU**: Enhanced GRU with residual connections and batch normalization
- **TCN**: Temporal Convolutional Networks with multi-scale and residual architectures
- **Transformers**: Time series transformers with positional encoding and causal attention

#### Ensemble Methods

- **Weighted Ensemble**: Dynamic weight adjustment based on recent performance
- **Adaptive Ensemble**: Regime-aware ensembles that adapt to market conditions
- **Stacking Ensemble**: Meta-learner combines predictions from base models

### Action Decision Models (Type 2)

- **Reinforcement Learning Agent**: Deep Q-Network (DQN) or Proximal Policy Optimization (PPO) for action selection
- **Policy Network**: Maps predicted prices + market state → trading actions (buy/sell/hold)
- **Risk-Adjusted Reward Function**: Incorporates Sharpe ratio, maximum drawdown, and transaction costs
- **Action Space**: Discrete (buy/sell/hold) or continuous (position sizing)

## 📈 Data Pipeline

- **Data Sources**: Alpaca API for stock prices and high-beta indices
- **Technical Indicators**: Moving averages, RSI, MACD, Bollinger Bands, etc
- **Data Preprocessing**: Normalization, missing value handling, and feature engineering
- **Train-Test Split**: Time-based splitting with walk-forward validation
- **Batching**: Efficient data loaders for training

## 💰 Trading Execution Framework

### Transaction Cost Modeling

- **Slippage Estimation**: Market impact models for realistic execution
- **Commission Structure**: Per-trade and percentage-based fees
- **Bid-Ask Spread**: Consideration for entry/exit costs

### Portfolio Management

- **Position Sizing**: Kelly Criterion or risk-parity approaches
- **Multi-Asset Allocation**: Dynamic rebalancing across multiple stocks
- **Leverage Control**: Maximum exposure limits
- **Correlation-Based Diversification**: Minimize portfolio correlation risk

## 📊 Backtesting & Performance Metrics

### Backtesting Framework

- **Walk-Forward Analysis**: Rolling window validation
- **Out-of-Sample Testing**: Holdout period evaluation
- **Monte Carlo Simulations**: Stress testing with synthetic scenarios

### Trading-Specific Metrics

- **Sharpe Ratio**: Risk-adjusted returns
- **Maximum Drawdown**: Worst peak-to-trough decline
- **Win Rate & Profit Factor**: Trade success metrics
- **Calmar Ratio**: Return vs. max drawdown
- **Alpha & Beta**: Market-relative performance

## 🔧 Advanced Features

### Market Microstructure

- **Order Book Imbalance**: Buy/sell pressure indicators
- **Volume Profile**: Support/resistance levels
- **Tick Data**: High-frequency patterns

### Alternative Data Integration

- **Sentiment Analysis**: News and social media sentiment
- **Options Flow**: Put/call ratios and unusual activity
- **Sector Rotation**: Cross-sector momentum signals
