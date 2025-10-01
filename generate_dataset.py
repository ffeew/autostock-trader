"""
Main script to generate the complete training dataset.

This script orchestrates the entire data pipeline:
1. Fetches historical stock data from Alpaca API
2. Adds technical indicators and features
3. Saves processed dataset for model training
"""

import os
import sys
import logging
import argparse
from datetime import datetime
from typing import List, Optional
import pandas as pd
from dotenv import load_dotenv

from src.data.fetcher import AlpacaDataFetcher
from src.data.features import create_features
from alpaca.data.timeframe import TimeFrame


# Set up logging
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/dataset_generation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


def generate_dataset(
    symbols: Optional[List[str]] = None,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    timeframe: Optional[str] = None,
    output_dir: str = 'data',
    primary_symbol: str = 'SPY',
    market_symbol: str = 'QQQ'
) -> pd.DataFrame:
    """
    Generate the complete training dataset.

    Args:
        symbols: List of stock symbols to fetch
        start_date: Start date for data (YYYY-MM-DD)
        end_date: End date for data (YYYY-MM-DD)
        output_dir: Directory to save output files
        primary_symbol: Primary symbol for feature generation
        market_symbol: Market/high-beta symbol for sentiment features

    Returns:
        DataFrame with complete dataset
    """
    logger.info("="*60)
    logger.info("Starting dataset generation pipeline")
    logger.info("="*60)

    # Load environment variables
    load_dotenv()

    # Use defaults from .env if not provided
    if symbols is None:
        symbols_str = os.getenv('SYMBOLS', 'SPY,QQQ')
        symbols = [s.strip() for s in symbols_str.split(',')]

    if start_date is None:
        start_date = os.getenv('START_DATE', '2001-01-01')

    if end_date is None:
        end_date = os.getenv('END_DATE', '2024-12-31')

    if timeframe is None:
        timeframe = os.getenv('TIMEFRAME', 'Minute')

    # Convert timeframe string to TimeFrame object
    timeframe_map = {
        'Minute': TimeFrame.Minute,
        'Hour': TimeFrame.Hour,
        'Day': TimeFrame.Day,
        'Week': TimeFrame.Week,
        'Month': TimeFrame.Month
    }
    timeframe_obj = timeframe_map.get(timeframe, TimeFrame.Minute)

    logger.info(f"Configuration:")
    logger.info(f"  Symbols: {symbols}")
    logger.info(f"  Date range: {start_date} to {end_date}")
    logger.info(f"  Timeframe: {timeframe}")
    logger.info(f"  Primary symbol: {primary_symbol}")
    logger.info(f"  Market symbol: {market_symbol}")
    logger.info(f"  Output directory: {output_dir}")

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Step 1: Fetch raw data
    logger.info("\n" + "="*60)
    logger.info("STEP 1: Fetching raw data from Alpaca API")
    logger.info("="*60)

    try:
        fetcher = AlpacaDataFetcher()
        raw_data = fetcher.fetch_stock_bars(
            symbols=symbols,
            start_date=start_date,
            end_date=end_date,
            timeframe=timeframe_obj
        )

        # Save raw data
        logger.info("Saving raw data...")
        fetcher.save_data(raw_data, output_dir=output_dir, format='csv')

        # Also save as parquet for efficient storage
        fetcher.save_data(raw_data, output_dir=output_dir, format='parquet')

    except Exception as e:
        logger.error(f"Failed to fetch data: {str(e)}")
        raise

    # Step 2: Feature engineering
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Feature engineering")
    logger.info("="*60)

    try:
        dataset = create_features(
            raw_data,
            primary_symbol=primary_symbol,
            market_symbol=market_symbol
        )

    except Exception as e:
        logger.error(f"Failed to create features: {str(e)}")
        raise

    # Step 3: Save processed dataset
    logger.info("\n" + "="*60)
    logger.info("STEP 3: Saving processed dataset")
    logger.info("="*60)

    try:
        # Save as CSV
        csv_path = os.path.join(output_dir, f'{primary_symbol}_features.csv')
        dataset.to_csv(csv_path)
        logger.info(f"Saved CSV to: {csv_path}")

        # Save as Parquet (more efficient)
        parquet_path = os.path.join(output_dir, f'{primary_symbol}_features.parquet')
        dataset.to_parquet(parquet_path)
        logger.info(f"Saved Parquet to: {parquet_path}")

        # Save metadata
        metadata = {
            'symbols': symbols,
            'primary_symbol': primary_symbol,
            'market_symbol': market_symbol,
            'start_date': start_date,
            'end_date': end_date,
            'actual_start': str(dataset.index[0]),
            'actual_end': str(dataset.index[-1]),
            'num_rows': len(dataset),
            'num_features': len(dataset.columns),
            'features': dataset.columns.tolist(),
            'generated_at': str(datetime.now())
        }

        import json
        metadata_path = os.path.join(output_dir, f'{primary_symbol}_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Saved metadata to: {metadata_path}")

    except Exception as e:
        logger.error(f"Failed to save dataset: {str(e)}")
        raise

    # Step 4: Generate summary report
    logger.info("\n" + "="*60)
    logger.info("STEP 4: Dataset summary")
    logger.info("="*60)

    logger.info(f"Dataset shape: {dataset.shape}")
    logger.info(f"Date range: {dataset.index[0]} to {dataset.index[-1]}")
    logger.info(f"Total features: {len(dataset.columns)}")
    logger.info(f"Missing values: {dataset.isnull().sum().sum()}")

    # Memory usage
    memory_mb = dataset.memory_usage(deep=True).sum() / 1024 / 1024
    logger.info(f"Memory usage: {memory_mb:.2f} MB")

    # Save summary statistics
    summary_path = os.path.join(output_dir, f'{primary_symbol}_summary.csv')
    dataset.describe().to_csv(summary_path)
    logger.info(f"Saved summary statistics to: {summary_path}")

    logger.info("\n" + "="*60)
    logger.info("Dataset generation completed successfully!")
    logger.info("="*60)

    return dataset


def main():
    """Main entry point for the script."""
    parser = argparse.ArgumentParser(
        description='Generate training dataset from Alpaca API'
    )
    parser.add_argument(
        '--symbols',
        type=str,
        help='Comma-separated list of symbols (e.g., SPY,QQQ)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        help='Start date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        help='End date in YYYY-MM-DD format'
    )
    parser.add_argument(
        '--timeframe',
        type=str,
        choices=['Minute', 'Hour', 'Day', 'Week', 'Month'],
        help='Data timeframe (default: from .env or Minute)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='data',
        help='Output directory for dataset files'
    )
    parser.add_argument(
        '--primary-symbol',
        type=str,
        default='SPY',
        help='Primary symbol for feature generation'
    )
    parser.add_argument(
        '--market-symbol',
        type=str,
        default='QQQ',
        help='Market/high-beta symbol for sentiment features'
    )

    args = parser.parse_args()

    # Parse symbols
    symbols = None
    if args.symbols:
        symbols = [s.strip() for s in args.symbols.split(',')]

    try:
        dataset = generate_dataset(
            symbols=symbols,
            start_date=args.start_date,
            end_date=args.end_date,
            timeframe=args.timeframe,
            output_dir=args.output_dir,
            primary_symbol=args.primary_symbol,
            market_symbol=args.market_symbol
        )

        # Print sample
        print("\n" + "="*60)
        print("Sample of generated dataset (first 5 rows):")
        print("="*60)
        print(dataset.head())

        print("\n" + "="*60)
        print("Feature columns:")
        print("="*60)
        for i, col in enumerate(dataset.columns, 1):
            print(f"  {i:3d}. {col}")

        return 0

    except Exception as e:
        logger.error(f"Dataset generation failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())