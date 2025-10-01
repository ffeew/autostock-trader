"""
Data fetching module for Alpaca API.

This module handles fetching historical stock data from the Alpaca API,
including stock prices and high-beta indices for market sentiment analysis.
"""

import os
import logging
from datetime import datetime
from typing import List, Dict, Optional
import pandas as pd
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest
from alpaca.data.timeframe import TimeFrame
from dotenv import load_dotenv


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/data_fetcher.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class AlpacaDataFetcher:
    """Fetches historical stock data from Alpaca API."""

    def __init__(self, api_key: Optional[str] = None, secret_key: Optional[str] = None):
        """
        Initialize the Alpaca data fetcher.

        Args:
            api_key: Alpaca API key (if None, loads from .env)
            secret_key: Alpaca secret key (if None, loads from .env)
        """
        # Load environment variables
        load_dotenv()

        # Get API credentials
        self.api_key = api_key or os.getenv('ALPACA_API_KEY')
        self.secret_key = secret_key or os.getenv('ALPACA_SECRET_KEY')

        if not self.api_key or not self.secret_key:
            raise ValueError("Alpaca API credentials not found. Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file.")

        # Initialize the client
        self.client = StockHistoricalDataClient(self.api_key, self.secret_key)
        logger.info("Alpaca data fetcher initialized successfully")

    def fetch_stock_bars(
        self,
        symbols: List[str],
        start_date: str,
        end_date: str,
        timeframe: TimeFrame = TimeFrame.Minute
    ) -> Dict[str, pd.DataFrame]:
        """
        Fetch historical stock bars for given symbols.

        Args:
            symbols: List of stock symbols (e.g., ['SPY', 'QQQ'])
            start_date: Start date in format 'YYYY-MM-DD'
            end_date: End date in format 'YYYY-MM-DD'
            timeframe: Data timeframe (default: Day)

        Returns:
            Dictionary mapping symbol to DataFrame with OHLCV data
        """
        logger.info(f"Fetching data for symbols: {symbols}")
        logger.info(f"Date range: {start_date} to {end_date}")

        try:
            # Convert dates to datetime objects
            start_dt = datetime.strptime(start_date, '%Y-%m-%d')
            end_dt = datetime.strptime(end_date, '%Y-%m-%d')

            # Create request
            request_params = StockBarsRequest(
                symbol_or_symbols=symbols,
                timeframe=timeframe,
                start=start_dt,
                end=end_dt
            )

            # Fetch bars
            bars = self.client.get_stock_bars(request_params)

            # Convert to DataFrame
            df = bars.df

            logger.info(f"Successfully fetched {len(df)} rows of data")

            # Split into separate DataFrames per symbol
            result = {}
            if 'symbol' in df.index.names:
                # Multi-index with symbol
                for symbol in symbols:
                    if symbol in df.index.get_level_values('symbol'):
                        symbol_df = df.xs(symbol, level='symbol')
                        result[symbol] = symbol_df
                        logger.info(f"{symbol}: {len(symbol_df)} bars fetched")
            else:
                # Single symbol
                result[symbols[0]] = df

            return result

        except Exception as e:
            logger.error(f"Error fetching data: {str(e)}")
            raise

    def save_data(
        self,
        data: Dict[str, pd.DataFrame],
        output_dir: str = 'data',
        format: str = 'csv'
    ) -> None:
        """
        Save fetched data to disk.

        Args:
            data: Dictionary mapping symbol to DataFrame
            output_dir: Output directory
            format: Output format ('csv' or 'parquet')
        """
        os.makedirs(output_dir, exist_ok=True)

        for symbol, df in data.items():
            if format == 'csv':
                filepath = os.path.join(output_dir, f'{symbol}_raw.csv')
                df.to_csv(filepath)
            elif format == 'parquet':
                filepath = os.path.join(output_dir, f'{symbol}_raw.parquet')
                df.to_parquet(filepath)
            else:
                raise ValueError(f"Unsupported format: {format}")

            logger.info(f"Saved {symbol} data to {filepath}")

    def load_data(
        self,
        symbol: str,
        data_dir: str = 'data',
        format: str = 'csv'
    ) -> pd.DataFrame:
        """
        Load previously saved data from disk.

        Args:
            symbol: Stock symbol
            data_dir: Data directory
            format: Data format ('csv' or 'parquet')

        Returns:
            DataFrame with historical data
        """
        if format == 'csv':
            filepath = os.path.join(data_dir, f'{symbol}_raw.csv')
            df = pd.read_csv(filepath, index_col=0, parse_dates=True)
        elif format == 'parquet':
            filepath = os.path.join(data_dir, f'{symbol}_raw.parquet')
            df = pd.read_parquet(filepath)
        else:
            raise ValueError(f"Unsupported format: {format}")

        logger.info(f"Loaded {symbol} data from {filepath}")
        return df


def fetch_and_save_data(
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None
) -> Dict[str, pd.DataFrame]:
    """
    Convenience function to fetch and save data using environment variables.

    Args:
        symbols: List of symbols (if None, uses SYMBOLS from .env)
        start_date: Start date (if None, uses START_DATE from .env)
        end_date: End date (if None, uses END_DATE from .env)

    Returns:
        Dictionary mapping symbol to DataFrame
    """
    load_dotenv()

    # Get configuration from environment
    if symbols is None:
        symbols_str = os.getenv('SYMBOLS', 'SPY,QQQ')
        symbols = [s.strip() for s in symbols_str.split(',')]

    if start_date is None:
        start_date = os.getenv('START_DATE', '2010-01-01')

    if end_date is None:
        end_date = os.getenv('END_DATE', '2024-12-31')

    # Fetch data
    fetcher = AlpacaDataFetcher()
    data = fetcher.fetch_stock_bars(symbols, start_date, end_date)

    # Save data
    fetcher.save_data(data)

    return data


if __name__ == '__main__':
    # Test the fetcher
    logger.info("Starting data fetch test")
    data = fetch_and_save_data()
    logger.info("Data fetch test completed")

    # Print summary
    for symbol, df in data.items():
        print(f"\n{symbol} Summary:")
        print(f"  Rows: {len(df)}")
        print(f"  Date range: {df.index[0]} to {df.index[-1]}")
        print(f"  Columns: {df.columns.tolist()}")
        print(f"\nFirst few rows:")
        print(df.head())