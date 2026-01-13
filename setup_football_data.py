#!/usr/bin/env python3
"""
Setup script to download football match data from football-data.co.uk.

This downloads match statistics for the top 5 European leagues from 2016/17 onwards.
Data includes: shots, corners, fouls, goals, cards (but NOT possession or crosses).

Usage:
    python setup_football_data.py
"""

from src.football_data_loader import (
    download_all_data,
    load_all_football_data,
    AVAILABLE_SEASONS,
    LEAGUE_CODES,
    FOOTBALL_DATA_DIR
)


def main():
    print("=" * 60)
    print("Football-Data.co.uk Data Setup")
    print("=" * 60)
    print()
    print(f"Downloading data for {len(AVAILABLE_SEASONS)} seasons:")
    for s in AVAILABLE_SEASONS:
        start = 2000 + int(s[:2])
        end = 2000 + int(s[2:])
        print(f"  - {start}/{end}")
    print()
    print(f"Leagues: {', '.join(LEAGUE_CODES.values())}")
    print(f"Data will be saved to: {FOOTBALL_DATA_DIR.absolute()}")
    print()
    print("-" * 60)

    # Download all data
    results = download_all_data()

    print()
    print("-" * 60)
    print(f"Downloaded {len(results)} season/league files")

    # Test loading
    print()
    print("Testing data loading...")
    df = load_all_football_data()

    if len(df) > 0:
        print(f"Successfully loaded {len(df)} team-seasons")
        print()
        print("Sample data:")
        print(df[["team_long_name", "league_name", "season", "avg_shots", "avg_corners", "avg_fouls", "win_rate"]].head(10))
        print()
        print("Data setup complete!")
    else:
        print("Warning: No data was loaded. Check the downloaded files.")


if __name__ == "__main__":
    main()
