"""
Feature engineering module for stock data.

This module adds technical indicators and derived features to raw stock data,
preparing it for use in machine learning models.
"""

import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Optional
from ta import trend, momentum, volatility, volume


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Adds technical indicators and features to stock data."""

    def __init__(self):
        """Initialize the feature engineer."""
        self.features_added = []

    def add_basic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add basic price-based features.

        Args:
            df: DataFrame with OHLCV data

        Returns:
            DataFrame with added features
        """
        logger.info("Adding basic features...")

        # Price changes
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))

        # Price ranges
        df['high_low_range'] = df['high'] - df['low']
        df['high_low_pct'] = (df['high'] - df['low']) / df['close']

        # Gaps
        df['gap'] = df['open'] - df['close'].shift(1)
        df['gap_pct'] = df['gap'] / df['close'].shift(1)

        # Intraday movement
        df['intraday_change'] = df['close'] - df['open']
        df['intraday_change_pct'] = df['intraday_change'] / df['open']

        self.features_added.extend([
            'returns', 'log_returns', 'high_low_range', 'high_low_pct',
            'gap', 'gap_pct', 'intraday_change', 'intraday_change_pct'
        ])

        return df

    def add_moving_averages(
        self,
        df: pd.DataFrame,
        windows: List[int] = [5, 10, 20, 50, 200]
    ) -> pd.DataFrame:
        """
        Add moving average indicators.

        Args:
            df: DataFrame with price data
            windows: List of window sizes for moving averages

        Returns:
            DataFrame with added features
        """
        logger.info(f"Adding moving averages: {windows}...")

        for window in windows:
            # Simple Moving Average
            df[f'sma_{window}'] = df['close'].rolling(window=window).mean()

            # Exponential Moving Average
            df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()

            # Price relative to MA
            df[f'close_to_sma_{window}'] = df['close'] / df[f'sma_{window}'] - 1
            df[f'close_to_ema_{window}'] = df['close'] / df[f'ema_{window}'] - 1

            self.features_added.extend([
                f'sma_{window}', f'ema_{window}',
                f'close_to_sma_{window}', f'close_to_ema_{window}'
            ])

        # Moving average crossovers
        if 20 in windows and 50 in windows:
            df['sma_20_50_cross'] = df['sma_20'] / df['sma_50'] - 1
            self.features_added.append('sma_20_50_cross')

        if 50 in windows and 200 in windows:
            df['sma_50_200_cross'] = df['sma_50'] / df['sma_200'] - 1
            self.features_added.append('sma_50_200_cross')

        return df

    def add_momentum_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add momentum-based technical indicators.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with added features
        """
        logger.info("Adding momentum indicators...")

        # RSI (Relative Strength Index)
        df['rsi_14'] = momentum.RSIIndicator(df['close'], window=14).rsi()
        df['rsi_7'] = momentum.RSIIndicator(df['close'], window=7).rsi()

        # Stochastic Oscillator
        stoch = momentum.StochasticOscillator(
            df['high'], df['low'], df['close'],
            window=14, smooth_window=3
        )
        df['stoch_k'] = stoch.stoch()
        df['stoch_d'] = stoch.stoch_signal()

        # Rate of Change (ROC)
        df['roc_10'] = momentum.ROCIndicator(df['close'], window=10).roc()
        df['roc_20'] = momentum.ROCIndicator(df['close'], window=20).roc()

        # Williams %R
        df['williams_r'] = momentum.WilliamsRIndicator(
            df['high'], df['low'], df['close'], lbp=14
        ).williams_r()

        self.features_added.extend([
            'rsi_14', 'rsi_7', 'stoch_k', 'stoch_d',
            'roc_10', 'roc_20', 'williams_r'
        ])

        return df

    def add_volatility_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volatility-based technical indicators.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with added features
        """
        logger.info("Adding volatility indicators...")

        # Bollinger Bands
        bollinger = volatility.BollingerBands(
            df['close'], window=20, window_dev=2
        )
        df['bb_high'] = bollinger.bollinger_hband()
        df['bb_mid'] = bollinger.bollinger_mavg()
        df['bb_low'] = bollinger.bollinger_lband()
        df['bb_width'] = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_position'] = (df['close'] - df['bb_low']) / (df['bb_high'] - df['bb_low'])

        # Average True Range (ATR)
        df['atr_14'] = volatility.AverageTrueRange(
            df['high'], df['low'], df['close'], window=14
        ).average_true_range()

        # Keltner Channel
        keltner = volatility.KeltnerChannel(
            df['high'], df['low'], df['close'], window=20
        )
        df['keltner_high'] = keltner.keltner_channel_hband()
        df['keltner_low'] = keltner.keltner_channel_lband()

        # Historical volatility (standard deviation of returns)
        df['volatility_10'] = df['returns'].rolling(window=10).std()
        df['volatility_20'] = df['returns'].rolling(window=20).std()

        self.features_added.extend([
            'bb_high', 'bb_mid', 'bb_low', 'bb_width', 'bb_position',
            'atr_14', 'keltner_high', 'keltner_low',
            'volatility_10', 'volatility_20'
        ])

        return df

    def add_trend_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add trend-based technical indicators.

        Args:
            df: DataFrame with price data

        Returns:
            DataFrame with added features
        """
        logger.info("Adding trend indicators...")

        # MACD (Moving Average Convergence Divergence)
        macd = trend.MACD(df['close'], window_slow=26, window_fast=12, window_sign=9)
        df['macd'] = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff'] = macd.macd_diff()

        # ADX (Average Directional Index)
        adx = trend.ADXIndicator(df['high'], df['low'], df['close'], window=14)
        df['adx'] = adx.adx()
        df['adx_pos'] = adx.adx_pos()
        df['adx_neg'] = adx.adx_neg()

        # Aroon Indicator
        aroon = trend.AroonIndicator(df['high'], df['low'], window=25)
        df['aroon_up'] = aroon.aroon_up()
        df['aroon_down'] = aroon.aroon_down()
        df['aroon_indicator'] = aroon.aroon_indicator()

        # CCI (Commodity Channel Index)
        df['cci'] = trend.CCIIndicator(
            df['high'], df['low'], df['close'], window=20
        ).cci()

        self.features_added.extend([
            'macd', 'macd_signal', 'macd_diff',
            'adx', 'adx_pos', 'adx_neg',
            'aroon_up', 'aroon_down', 'aroon_indicator',
            'cci'
        ])

        return df

    def add_volume_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Add volume-based technical indicators.

        Args:
            df: DataFrame with price and volume data

        Returns:
            DataFrame with added features
        """
        logger.info("Adding volume indicators...")

        # Volume changes
        df['volume_change'] = df['volume'].pct_change()
        df['volume_ma_20'] = df['volume'].rolling(window=20).mean()
        df['volume_ratio'] = df['volume'] / df['volume_ma_20']

        # On-Balance Volume (OBV)
        df['obv'] = volume.OnBalanceVolumeIndicator(
            df['close'], df['volume']
        ).on_balance_volume()

        # Accumulation/Distribution Index
        df['adi'] = volume.AccDistIndexIndicator(
            df['high'], df['low'], df['close'], df['volume']
        ).acc_dist_index()

        # Chaikin Money Flow
        df['cmf'] = volume.ChaikinMoneyFlowIndicator(
            df['high'], df['low'], df['close'], df['volume'], window=20
        ).chaikin_money_flow()

        # Volume Weighted Average Price ratio
        df['close_to_vwap'] = df['close'] / df['vwap'] - 1

        self.features_added.extend([
            'volume_change', 'volume_ma_20', 'volume_ratio',
            'obv', 'adi', 'cmf', 'close_to_vwap'
        ])

        return df

    def add_all_features(
        self,
        df: pd.DataFrame,
        ma_windows: List[int] = [5, 10, 20, 50, 200]
    ) -> pd.DataFrame:
        """
        Add all technical indicators and features.

        Args:
            df: DataFrame with OHLCV data
            ma_windows: List of window sizes for moving averages

        Returns:
            DataFrame with all features added
        """
        logger.info("Adding all features to dataframe...")

        df = self.add_basic_features(df)
        df = self.add_moving_averages(df, windows=ma_windows)
        df = self.add_momentum_indicators(df)
        df = self.add_volatility_indicators(df)
        df = self.add_trend_indicators(df)
        df = self.add_volume_indicators(df)

        logger.info(f"Added {len(self.features_added)} features")

        return df

    def add_high_beta_features(
        self,
        target_df: pd.DataFrame,
        market_df: pd.DataFrame,
        prefix: str = 'market'
    ) -> pd.DataFrame:
        """
        Add high-beta index features as market sentiment signals.

        Args:
            target_df: DataFrame for target stock (e.g., SPY)
            market_df: DataFrame for market index (e.g., QQQ)
            prefix: Prefix for market feature columns

        Returns:
            DataFrame with market features added
        """
        logger.info(f"Adding high-beta features with prefix '{prefix}'...")

        # Align indices
        market_df = market_df.reindex(target_df.index)

        # Market price features
        target_df[f'{prefix}_close'] = market_df['close']
        target_df[f'{prefix}_returns'] = market_df['close'].pct_change()
        target_df[f'{prefix}_volume'] = market_df['volume']

        # Relative performance
        target_df[f'{prefix}_relative_returns'] = target_df['returns'] - market_df['close'].pct_change()

        # Market momentum
        target_df[f'{prefix}_rsi'] = momentum.RSIIndicator(
            market_df['close'], window=14
        ).rsi()

        # Market volatility
        target_df[f'{prefix}_volatility'] = market_df['close'].pct_change().rolling(window=20).std()

        # Correlation (rolling)
        target_df[f'{prefix}_correlation_20'] = target_df['returns'].rolling(
            window=20
        ).corr(market_df['close'].pct_change())

        logger.info(f"Added {prefix} features for market sentiment analysis")

        return target_df


def create_features(
    data: Dict[str, pd.DataFrame],
    primary_symbol: str = 'SPY',
    market_symbol: str = 'QQQ'
) -> pd.DataFrame:
    """
    Create features for the dataset.

    Args:
        data: Dictionary mapping symbol to DataFrame
        primary_symbol: Primary symbol to generate features for
        market_symbol: Market/high-beta symbol for sentiment features

    Returns:
        DataFrame with all features for the primary symbol
    """
    engineer = FeatureEngineer()

    # Get primary data
    df = data[primary_symbol].copy()

    # Add all technical features
    df = engineer.add_all_features(df)

    # Add market sentiment features if available
    if market_symbol in data and market_symbol != primary_symbol:
        df = engineer.add_high_beta_features(df, data[market_symbol], prefix='market')

    # Drop rows with NaN values (due to indicator calculations)
    initial_rows = len(df)
    df = df.dropna()
    dropped_rows = initial_rows - len(df)

    if dropped_rows > 0:
        logger.info(f"Dropped {dropped_rows} rows with NaN values (from indicator calculations)")

    logger.info(f"Final dataset shape: {df.shape}")

    return df


if __name__ == '__main__':
    # Test the feature engineer
    from src.data.fetcher import AlpacaDataFetcher

    logger.info("Testing feature engineering...")

    # Fetch sample data
    fetcher = AlpacaDataFetcher()
    data = fetcher.fetch_stock_bars(
        symbols=['SPY', 'QQQ'],
        start_date='2024-01-01',
        end_date='2024-08-31'
    )

    # Create features
    df_features = create_features(data, primary_symbol='SPY', market_symbol='QQQ')

    # Print summary
    print(f"\nFeature Engineering Summary:")
    print(f"  Total rows: {len(df_features)}")
    print(f"  Total features: {len(df_features.columns)}")
    print(f"  Date range: {df_features.index[0]} to {df_features.index[-1]}")
    print(f"\nColumns: {df_features.columns.tolist()}")
    print(f"\nFirst few rows:")
    print(df_features.head())
    print(f"\nDescriptive statistics:")
    print(df_features.describe())