"""
Verification script to test the dataset generation setup.

This script checks that:
1. All dependencies are installed
2. API credentials are configured
3. The data pipeline works end-to-end
"""

import os
import sys
import json
from datetime import datetime, timedelta


def check_dependencies():
    """Check if all required packages are installed."""
    print("\n" + "="*60)
    print("Checking dependencies...")
    print("="*60)

    required_packages = [
        'pandas',
        'numpy',
        'alpaca',
        'ta',
        'dotenv',
        'pyarrow'
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} (missing)")
            missing.append(package)

    if missing:
        print(f"\nMissing packages: {', '.join(missing)}")
        print("Install with: pip install -r requirements.txt")
        return False

    return True


def check_env_file():
    """Check if .env file exists and has required variables."""
    print("\n" + "="*60)
    print("Checking .env configuration...")
    print("="*60)

    if not os.path.exists('.env'):
        print("  ✗ .env file not found")
        print("  Copy .env.example to .env and add your API keys")
        return False

    from dotenv import load_dotenv
    load_dotenv()

    required_vars = [
        'ALPACA_API_KEY',
        'ALPACA_SECRET_KEY',
        'SYMBOLS',
        'START_DATE',
        'END_DATE'
    ]

    missing = []
    for var in required_vars:
        value = os.getenv(var)
        if value and value != f'your_{var.lower()}_here':
            print(f"  ✓ {var}")
        else:
            print(f"  ✗ {var} (not set)")
            missing.append(var)

    if missing:
        print(f"\nMissing variables: {', '.join(missing)}")
        print("Update your .env file with proper values")
        return False

    return True


def test_api_connection():
    """Test connection to Alpaca API."""
    print("\n" + "="*60)
    print("Testing Alpaca API connection...")
    print("="*60)

    try:
        from src.data.fetcher import AlpacaDataFetcher

        fetcher = AlpacaDataFetcher()

        # Fetch just 1 day of data as a test
        end_date = datetime.now()
        start_date = end_date - timedelta(days=5)

        data = fetcher.fetch_stock_bars(
            symbols=['SPY'],
            start_date=start_date.strftime('%Y-%m-%d'),
            end_date=end_date.strftime('%Y-%m-%d')
        )

        if 'SPY' in data and len(data['SPY']) > 0:
            print(f"  ✓ Successfully fetched {len(data['SPY'])} bars")
            return True
        else:
            print("  ✗ No data returned")
            return False

    except Exception as e:
        print(f"  ✗ Connection failed: {str(e)}")
        return False


def check_data_directory():
    """Check if data has been generated."""
    print("\n" + "="*60)
    print("Checking generated data...")
    print("="*60)

    if not os.path.exists('data'):
        print("  ✗ data/ directory not found")
        print("  Run: python generate_dataset.py")
        return False

    required_files = [
        'SPY_features.parquet',
        'SPY_metadata.json'
    ]

    missing = []
    for filename in required_files:
        filepath = os.path.join('data', filename)
        if os.path.exists(filepath):
            size = os.path.getsize(filepath) / 1024  # KB
            print(f"  ✓ {filename} ({size:.1f} KB)")
        else:
            print(f"  ✗ {filename} (not found)")
            missing.append(filename)

    if missing:
        print(f"\nMissing files: {', '.join(missing)}")
        print("Run: python generate_dataset.py")
        return False

    # Check metadata
    try:
        with open('data/SPY_metadata.json') as f:
            metadata = json.load(f)
            print(f"\n  Dataset info:")
            print(f"    Rows: {metadata['num_rows']:,}")
            print(f"    Features: {metadata['num_features']}")
            print(f"    Date range: {metadata['actual_start']} to {metadata['actual_end']}")
    except Exception as e:
        print(f"  ⚠ Could not read metadata: {str(e)}")

    return True


def main():
    """Run all verification checks."""
    print("="*60)
    print("AutoStock Trader - Setup Verification")
    print("="*60)

    checks = [
        ("Dependencies", check_dependencies),
        ("Environment Config", check_env_file),
        ("API Connection", test_api_connection),
        ("Generated Data", check_data_directory)
    ]

    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n  ✗ Error in {name}: {str(e)}")
            results.append((name, False))

    # Summary
    print("\n" + "="*60)
    print("Verification Summary")
    print("="*60)

    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"  {status}: {name}")

    all_passed = all(result for _, result in results)

    print("\n" + "="*60)
    if all_passed:
        print("✓ All checks passed! Your setup is ready.")
        print("\nNext steps:")
        print("  1. Generate full dataset: python generate_dataset.py")
        print("  2. Start building models!")
    else:
        print("✗ Some checks failed. Please fix the issues above.")
    print("="*60)

    return 0 if all_passed else 1


if __name__ == '__main__':
    sys.exit(main())