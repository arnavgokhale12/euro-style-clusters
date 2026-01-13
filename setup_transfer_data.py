#!/usr/bin/env python3
"""
Setup script to download TransferMarkt data.

Usage:
    python setup_transfer_data.py

This downloads transfer market data from the salimt/football-datasets repository
and prepares it for use with the Euro Soccer Clusters app.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.transfer_data import download_transfer_data, load_transfer_data, TRANSFER_DATA_DIR


def main():
    print("=" * 60)
    print("TransferMarkt Data Setup")
    print("=" * 60)
    print()

    print(f"Data will be saved to: {TRANSFER_DATA_DIR.absolute()}")
    print()

    # Download data
    print("Downloading transfer market data...")
    success = download_transfer_data(force=False)

    if not success:
        print()
        print("ERROR: Failed to download some files.")
        print("Please check your internet connection and try again.")
        sys.exit(1)

    print()
    print("Verifying data...")

    try:
        data = load_transfer_data()

        print()
        print("Data loaded successfully!")
        print("-" * 40)
        for name, df in data.items():
            print(f"  {name}: {len(df):,} rows, {len(df.columns)} columns")

        print()
        print("=" * 60)
        print("Setup complete! You can now run the app:")
        print("  streamlit run app.py")
        print("=" * 60)

    except Exception as e:
        print(f"ERROR loading data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
