"""
Football-Data.co.uk Data Loader

Downloads and processes match data from football-data.co.uk for seasons 2016/17 onwards.
This data source provides shots, corners, and fouls but NOT possession or crosses.

Data is available for the top 5 European leagues:
- E0: English Premier League
- SP1: Spanish La Liga
- I1: Italian Serie A
- D1: German Bundesliga
- F1: French Ligue 1
"""

import pandas as pd
import requests
from pathlib import Path
from typing import Optional, List, Dict
import time

# Data directory
FOOTBALL_DATA_DIR = Path("data/football_data")

# League codes for football-data.co.uk
LEAGUE_CODES = {
    "E0": "Premier League",
    "SP1": "La Liga",
    "I1": "Serie A",
    "D1": "Bundesliga",
    "F1": "Ligue 1",
}

# Season format: "1617" means 2016/17
AVAILABLE_SEASONS = [
    "1617", "1718", "1819", "1920", "2021", "2122", "2223", "2324", "2425"
]

# Column mappings from football-data.co.uk format
COLUMN_MAPPING = {
    "Date": "date",
    "HomeTeam": "home_team",
    "AwayTeam": "away_team",
    "FTHG": "home_goals",
    "FTAG": "away_goals",
    "FTR": "result",
    "HS": "home_shots",
    "AS": "away_shots",
    "HST": "home_shots_on_target",
    "AST": "away_shots_on_target",
    "HC": "home_corners",
    "AC": "away_corners",
    "HF": "home_fouls",
    "AF": "away_fouls",
    "HY": "home_yellow",
    "AY": "away_yellow",
    "HR": "home_red",
    "AR": "away_red",
}


def get_season_display(season_code: str) -> str:
    """Convert season code to display format (e.g., '1617' -> '2016/2017')."""
    start_year = 2000 + int(season_code[:2])
    end_year = 2000 + int(season_code[2:])
    return f"{start_year}/{end_year}"


def download_season_data(
    season: str,
    league: str,
    force: bool = False
) -> Optional[pd.DataFrame]:
    """
    Download a single season/league CSV from football-data.co.uk.

    Args:
        season: Season code (e.g., "2324" for 2023/24)
        league: League code (e.g., "E0" for Premier League)
        force: If True, re-download even if cached

    Returns:
        DataFrame with match data or None if download fails
    """
    FOOTBALL_DATA_DIR.mkdir(parents=True, exist_ok=True)

    cache_file = FOOTBALL_DATA_DIR / f"{league}_{season}.csv"

    if cache_file.exists() and not force:
        try:
            return pd.read_csv(cache_file)
        except Exception:
            pass  # Re-download if cache is corrupted

    url = f"https://www.football-data.co.uk/mmz4281/{season}/{league}.csv"

    try:
        response = requests.get(url, timeout=30)
        response.raise_for_status()

        # Save to cache
        cache_file.write_bytes(response.content)

        # Parse CSV
        df = pd.read_csv(cache_file)
        return df

    except requests.RequestException as e:
        print(f"Failed to download {league} {season}: {e}")
        return None


def download_all_data(
    seasons: List[str] = None,
    leagues: List[str] = None,
    force: bool = False
) -> Dict[str, pd.DataFrame]:
    """
    Download all specified seasons and leagues.

    Args:
        seasons: List of season codes (default: all available)
        leagues: List of league codes (default: top 5)
        force: If True, re-download even if cached

    Returns:
        Dict mapping "{league}_{season}" to DataFrames
    """
    if seasons is None:
        seasons = AVAILABLE_SEASONS
    if leagues is None:
        leagues = list(LEAGUE_CODES.keys())

    results = {}
    total = len(seasons) * len(leagues)
    count = 0

    for season in seasons:
        for league in leagues:
            count += 1
            key = f"{league}_{season}"
            print(f"[{count}/{total}] Downloading {LEAGUE_CODES.get(league, league)} {get_season_display(season)}...")

            df = download_season_data(season, league, force)
            if df is not None:
                results[key] = df

            # Be nice to the server
            time.sleep(0.5)

    return results


def transform_to_team_centric(df: pd.DataFrame, league: str, season: str) -> pd.DataFrame:
    """
    Transform match data from home/away format to team-centric format.

    Each match becomes two rows: one for home team, one for away team.
    """
    # Rename columns
    df = df.rename(columns={k: v for k, v in COLUMN_MAPPING.items() if k in df.columns})

    # Required columns
    required = ["home_team", "away_team", "home_goals", "away_goals"]
    if not all(c in df.columns for c in required):
        return pd.DataFrame()

    # Home team rows
    home_rows = []
    for _, row in df.iterrows():
        home_data = {
            "team": row.get("home_team"),
            "opponent": row.get("away_team"),
            "is_home": True,
            "goals_scored": row.get("home_goals", 0),
            "goals_conceded": row.get("away_goals", 0),
            "shots": row.get("home_shots", 0),
            "shots_on_target": row.get("home_shots_on_target", 0),
            "corners": row.get("home_corners", 0),
            "fouls": row.get("home_fouls", 0),
            "yellow_cards": row.get("home_yellow", 0),
            "red_cards": row.get("home_red", 0),
            "league_code": league,
            "league_name": LEAGUE_CODES.get(league, league),
            "season": get_season_display(season),
        }

        # Determine result
        if home_data["goals_scored"] > home_data["goals_conceded"]:
            home_data["result"] = "W"
            home_data["points"] = 3
        elif home_data["goals_scored"] < home_data["goals_conceded"]:
            home_data["result"] = "L"
            home_data["points"] = 0
        else:
            home_data["result"] = "D"
            home_data["points"] = 1

        home_rows.append(home_data)

    # Away team rows
    away_rows = []
    for _, row in df.iterrows():
        away_data = {
            "team": row.get("away_team"),
            "opponent": row.get("home_team"),
            "is_home": False,
            "goals_scored": row.get("away_goals", 0),
            "goals_conceded": row.get("home_goals", 0),
            "shots": row.get("away_shots", 0),
            "shots_on_target": row.get("away_shots_on_target", 0),
            "corners": row.get("away_corners", 0),
            "fouls": row.get("away_fouls", 0),
            "yellow_cards": row.get("away_yellow", 0),
            "red_cards": row.get("away_red", 0),
            "league_code": league,
            "league_name": LEAGUE_CODES.get(league, league),
            "season": get_season_display(season),
        }

        # Determine result
        if away_data["goals_scored"] > away_data["goals_conceded"]:
            away_data["result"] = "W"
            away_data["points"] = 3
        elif away_data["goals_scored"] < away_data["goals_conceded"]:
            away_data["result"] = "L"
            away_data["points"] = 0
        else:
            away_data["result"] = "D"
            away_data["points"] = 1

        away_rows.append(away_data)

    return pd.DataFrame(home_rows + away_rows)


def aggregate_team_season(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate match-level data to team-season level.

    Returns DataFrame with one row per team-season with aggregated stats.
    """
    if matches_df.empty:
        return pd.DataFrame()

    # Group by team, league, season
    grouped = matches_df.groupby(["team", "league_name", "season"])

    # Aggregate stats
    agg_funcs = {
        "goals_scored": "sum",
        "goals_conceded": "sum",
        "shots": "sum",
        "corners": "sum",
        "fouls": "sum",
        "points": "sum",
        "result": lambda x: (x == "W").sum(),  # wins
    }

    # Filter to columns that exist
    agg_funcs = {k: v for k, v in agg_funcs.items() if k in matches_df.columns}

    result = grouped.agg(agg_funcs).reset_index()
    result = result.rename(columns={"result": "wins"})

    # Calculate matches played
    match_counts = grouped.size().reset_index(name="matches_played")
    result = result.merge(match_counts, on=["team", "league_name", "season"])

    # Calculate per-game averages
    result["avg_goals_scored"] = result["goals_scored"] / result["matches_played"]
    result["avg_goals_conceded"] = result["goals_conceded"] / result["matches_played"]
    result["avg_shots"] = result["shots"] / result["matches_played"]
    result["avg_corners"] = result["corners"] / result["matches_played"]
    result["avg_fouls"] = result["fouls"] / result["matches_played"]

    # Calculate derived metrics
    result["win_rate"] = result["wins"] / result["matches_played"]
    result["goal_difference"] = result["goals_scored"] - result["goals_conceded"]
    result["points_per_game"] = result["points"] / result["matches_played"]

    # Rename team column for consistency
    result = result.rename(columns={"team": "team_long_name"})

    return result


def load_all_football_data(
    start_season: str = "1617",
    end_season: str = "2425"
) -> pd.DataFrame:
    """
    Load all available football-data.co.uk data, processed and aggregated.

    Args:
        start_season: First season to include (e.g., "1617")
        end_season: Last season to include (e.g., "2425")

    Returns:
        DataFrame with team-season level aggregated stats
    """
    # Filter seasons
    seasons = [s for s in AVAILABLE_SEASONS if s >= start_season and s <= end_season]

    all_matches = []

    for season in seasons:
        for league in LEAGUE_CODES.keys():
            cache_file = FOOTBALL_DATA_DIR / f"{league}_{season}.csv"

            if cache_file.exists():
                try:
                    df = pd.read_csv(cache_file)
                    matches = transform_to_team_centric(df, league, season)
                    if not matches.empty:
                        all_matches.append(matches)
                except Exception as e:
                    print(f"Error processing {league}_{season}: {e}")

    if not all_matches:
        return pd.DataFrame()

    # Combine all matches
    combined = pd.concat(all_matches, ignore_index=True)

    # Aggregate to team-season level
    return aggregate_team_season(combined)


def check_data_exists() -> bool:
    """Check if football-data.co.uk data has been downloaded."""
    if not FOOTBALL_DATA_DIR.exists():
        return False

    # Check for at least one file
    csv_files = list(FOOTBALL_DATA_DIR.glob("*.csv"))
    return len(csv_files) > 0
