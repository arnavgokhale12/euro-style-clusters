"""
Transfer Market Data Module

Loads and processes transfer data from TransferMarkt datasets.
Provides financial metrics for teams including:
- Squad market value
- Transfer spending (what was paid for players)
- Net spend (spent - received)
- Value efficiency (market value / cost ratio)
- ROI metrics
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional, Dict, Tuple

# Data source configuration
TRANSFER_DATA_DIR = Path("data/transfermarkt")
KAGGLE_DATASET = "davidcariboo/player-scores"

# Top 5 league team name mappings (TransferMarkt naming conventions)
TOP_5_LEAGUE_COUNTRIES = ["England", "Spain", "Italy", "Germany", "France"]

# Files we need from the dataset (Kaggle davidcariboo/player-scores)
REQUIRED_FILES = {
    "players": "players.csv",
    "transfers": "transfers.csv",
    "valuations": "player_valuations.csv",
    "clubs": "clubs.csv",
}


def download_transfer_data(force: bool = False) -> bool:
    """
    Download transfer market data from Kaggle.

    Args:
        force: If True, re-download even if files exist

    Returns:
        True if successful, False otherwise
    """
    TRANSFER_DATA_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already downloaded
    all_exist = all((TRANSFER_DATA_DIR / f"{name}.csv").exists() for name in REQUIRED_FILES.keys())
    if all_exist and not force:
        print("Transfer data already exists. Use force=True to re-download.")
        return True

    try:
        import kagglehub
        print(f"Downloading {KAGGLE_DATASET} from Kaggle...")
        dataset_path = kagglehub.dataset_download(KAGGLE_DATASET)
        dataset_path = Path(dataset_path)
        print(f"  Downloaded to: {dataset_path}")

        # Copy required files to our data directory
        for name, filename in REQUIRED_FILES.items():
            src = dataset_path / filename
            dst = TRANSFER_DATA_DIR / f"{name}.csv"

            if src.exists():
                import shutil
                shutil.copy(src, dst)
                print(f"  Copied {filename} -> {dst}")
            else:
                print(f"  Warning: {filename} not found in dataset")

        return True

    except ImportError:
        print("ERROR: kagglehub not installed. Run: pip install kagglehub")
        return False
    except Exception as e:
        print(f"ERROR downloading data: {e}")
        print("\nTo fix this:")
        print("1. Go to https://www.kaggle.com/settings -> API -> Create New Token")
        print("2. Save kaggle.json to ~/.kaggle/kaggle.json")
        print("3. Run: chmod 600 ~/.kaggle/kaggle.json")
        return False


def load_transfer_data() -> Dict[str, pd.DataFrame]:
    """
    Load all transfer market data files.

    Returns:
        Dictionary with DataFrames for players, transfers, valuations, clubs
    """
    data = {}

    for name in REQUIRED_FILES.keys():
        local_path = TRANSFER_DATA_DIR / f"{name}.csv"

        if not local_path.exists():
            raise FileNotFoundError(
                f"Transfer data not found at {local_path}. "
                "Run download_transfer_data() first or use setup_transfer_data.py"
            )

        data[name] = pd.read_csv(local_path, low_memory=False)

    return data


def get_top5_clubs(clubs_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Filter to only include clubs from top 5 European leagues.
    Uses player nationality and club associations to infer league.
    """
    # Get unique club IDs from players who played in top 5 countries
    # This is a heuristic - we filter based on domestic competition
    top5_clubs = clubs_df.copy()

    # Filter by domestic competition if available
    if 'domestic_competition_id' in top5_clubs.columns:
        # Top 5 league competition IDs (TransferMarkt)
        top5_comp_ids = ['GB1', 'ES1', 'IT1', 'L1', 'FR1']  # Premier League, La Liga, Serie A, Bundesliga, Ligue 1
        top5_clubs = top5_clubs[top5_clubs['domestic_competition_id'].isin(top5_comp_ids)]

    return top5_clubs


def parse_transfer_fee(fee_str) -> Optional[float]:
    """
    Parse transfer fee string to numeric value in euros.

    Handles formats like:
    - "25.00m" -> 25,000,000
    - "500k" -> 500,000
    - "Free transfer" -> 0
    - "Loan" -> None (excluded from calculations)
    - "-" or "?" -> None
    """
    if pd.isna(fee_str):
        return None

    fee_str = str(fee_str).lower().strip()

    # Handle special cases
    if fee_str in ['-', '?', 'nan', '', 'none']:
        return None
    if 'free' in fee_str:
        return 0.0
    if 'loan' in fee_str:
        return None  # Loans don't count as transfer spending
    if 'end of loan' in fee_str:
        return None

    # Remove currency symbols
    fee_str = fee_str.replace('€', '').replace('$', '').replace('£', '').strip()

    try:
        # Parse millions
        if 'm' in fee_str:
            return float(fee_str.replace('m', '').strip()) * 1_000_000
        # Parse thousands
        elif 'k' in fee_str:
            return float(fee_str.replace('k', '').strip()) * 1_000
        # Parse billions (rare)
        elif 'bn' in fee_str or 'b' in fee_str:
            return float(fee_str.replace('bn', '').replace('b', '').strip()) * 1_000_000_000
        else:
            # Try direct numeric parsing
            return float(fee_str)
    except (ValueError, AttributeError):
        return None


def parse_market_value(value_str) -> Optional[float]:
    """Parse market value string to numeric (same format as transfer fees)."""
    return parse_transfer_fee(value_str)


def calculate_team_financials(
    transfers_df: pd.DataFrame,
    valuations_df: pd.DataFrame,
    players_df: pd.DataFrame,
    clubs_df: pd.DataFrame,
    season_start: int = 2020,
    season_end: int = 2026
) -> pd.DataFrame:
    """
    Calculate financial metrics for each team.

    Returns DataFrame with:
    - club_id, club_name
    - squad_value: Current total market value of squad
    - total_spent: Sum of transfer fees paid (incoming)
    - total_received: Sum of transfer fees received (outgoing)
    - net_spend: total_spent - total_received
    - value_efficiency: squad_value / total_spent ratio
    - avg_player_age: Average age of squad
    """
    results = []

    # Parse transfer fees - handle different column names
    transfers_df = transfers_df.copy()
    fee_col = None
    for col in ['transfer_fee', 'fee', 'transfer_fee_in_eur', 'fee_in_eur']:
        if col in transfers_df.columns:
            fee_col = col
            break

    if fee_col:
        transfers_df['fee_euros'] = transfers_df[fee_col].apply(parse_transfer_fee)
    else:
        # If no fee column, try to use market value as proxy
        transfers_df['fee_euros'] = 0

    # Filter by date range - handle different column names
    date_col = None
    for col in ['transfer_date', 'date', 'transfer_period']:
        if col in transfers_df.columns:
            date_col = col
            break

    if date_col:
        transfers_df[date_col] = pd.to_datetime(transfers_df[date_col], errors='coerce')
        transfers_df['transfer_year'] = transfers_df[date_col].dt.year
        transfers_df = transfers_df[
            (transfers_df['transfer_year'] >= season_start) &
            (transfers_df['transfer_year'] <= season_end)
        ]
    elif 'season' in transfers_df.columns:
        def parse_season_year(s):
            if pd.isna(s):
                return None
            s = str(s)
            if '/' in s:
                return int(s.split('/')[0])
            try:
                return int(s)
            except:
                return None
        transfers_df['transfer_year'] = transfers_df['season'].apply(parse_season_year)
        transfers_df = transfers_df[
            (transfers_df['transfer_year'] >= season_start) &
            (transfers_df['transfer_year'] <= season_end)
        ]

    # Get latest valuations per player - handle different column names
    valuations_df = valuations_df.copy()
    value_col = None
    for col in ['market_value', 'market_value_in_eur', 'value', 'market_value_in_gbp']:
        if col in valuations_df.columns:
            value_col = col
            break

    if value_col:
        valuations_df['value_euros'] = valuations_df[value_col].apply(parse_market_value)
    else:
        valuations_df['value_euros'] = 0

    val_date_col = None
    for col in ['date', 'datetime', 'valuation_date']:
        if col in valuations_df.columns:
            val_date_col = col
            break

    if val_date_col:
        valuations_df[val_date_col] = pd.to_datetime(valuations_df[val_date_col], errors='coerce')
        latest_vals = valuations_df.sort_values(val_date_col).groupby('player_id').last().reset_index()
    else:
        latest_vals = valuations_df.groupby('player_id').last().reset_index()

    # Get player ages
    players_df = players_df.copy()
    if 'date_of_birth' in players_df.columns:
        players_df['date_of_birth'] = pd.to_datetime(players_df['date_of_birth'], errors='coerce')
        players_df['age'] = (pd.Timestamp.now() - players_df['date_of_birth']).dt.days / 365.25

    # Get current club assignments
    club_id_col = None
    for col in ['current_club_id', 'club_id', 'current_club']:
        if col in players_df.columns:
            club_id_col = col
            break

    if club_id_col and 'age' in players_df.columns:
        player_clubs = players_df[['player_id', club_id_col, 'age']].copy()
        player_clubs = player_clubs.rename(columns={club_id_col: 'club_id'})
    elif club_id_col:
        player_clubs = players_df[['player_id', club_id_col]].copy()
        player_clubs = player_clubs.rename(columns={club_id_col: 'club_id'})
    else:
        player_clubs = pd.DataFrame()

    # Merge player valuations with club assignments
    if not player_clubs.empty and 'value_euros' in latest_vals.columns:
        player_values = player_clubs.merge(
            latest_vals[['player_id', 'value_euros']],
            on='player_id',
            how='left'
        )
    else:
        player_values = pd.DataFrame()

    # Determine transfer club column names
    to_club_col = None
    from_club_col = None
    for col in ['to_club_id', 'club_id', 'new_club_id']:
        if col in transfers_df.columns:
            to_club_col = col
            break
    for col in ['from_club_id', 'old_club_id', 'previous_club_id']:
        if col in transfers_df.columns:
            from_club_col = col
            break

    # Determine club ID and name columns
    club_id_field = None
    club_name_field = None
    for col in ['club_id', 'id']:
        if col in clubs_df.columns:
            club_id_field = col
            break
    for col in ['club_name', 'name', 'club']:
        if col in clubs_df.columns:
            club_name_field = col
            break

    if not club_id_field:
        return pd.DataFrame()

    # Process each club
    for _, club in clubs_df.iterrows():
        club_id = club.get(club_id_field)
        club_name = club.get(club_name_field, f'Club {club_id}') if club_name_field else f'Club {club_id}'

        # Squad value (sum of current player valuations)
        if not player_values.empty:
            club_players = player_values[player_values['club_id'] == club_id]
            squad_value = club_players['value_euros'].sum()
            avg_age = club_players['age'].mean() if 'age' in club_players.columns else None
            squad_size = len(club_players)
        else:
            squad_value = 0
            avg_age = None
            squad_size = 0

        # Transfers in (spending)
        if to_club_col:
            incoming = transfers_df[transfers_df[to_club_col] == club_id]
            total_spent = incoming['fee_euros'].sum() if 'fee_euros' in incoming.columns else 0
            transfers_in = len(incoming[incoming['fee_euros'].notna()]) if 'fee_euros' in incoming.columns else 0
        else:
            total_spent = 0
            transfers_in = 0

        # Transfers out (revenue)
        if from_club_col:
            outgoing = transfers_df[transfers_df[from_club_col] == club_id]
            total_received = outgoing['fee_euros'].sum() if 'fee_euros' in outgoing.columns else 0
            transfers_out = len(outgoing[outgoing['fee_euros'].notna()]) if 'fee_euros' in outgoing.columns else 0
        else:
            total_received = 0
            transfers_out = 0

        # Net spend
        net_spend = total_spent - total_received

        # Value efficiency (value generated per euro spent)
        if total_spent > 0:
            value_efficiency = squad_value / total_spent
        else:
            value_efficiency = None

        # Value vs cost difference (positive = overperforming)
        value_vs_cost = squad_value - total_spent

        results.append({
            'club_id': club_id,
            'club_name': club_name,
            'squad_value': squad_value,
            'total_spent': total_spent,
            'total_received': total_received,
            'net_spend': net_spend,
            'value_efficiency': value_efficiency,
            'value_vs_cost': value_vs_cost,
            'avg_player_age': avg_age,
            'squad_size': squad_size,
            'transfers_in': transfers_in,
            'transfers_out': transfers_out,
        })

    df = pd.DataFrame(results)

    # Remove clubs with no data
    if len(df) > 0:
        df = df[
            (df['squad_value'] > 0) |
            (df['total_spent'] > 0) |
            (df['total_received'] > 0)
        ]

    return df


def get_team_efficiency_ranking(financials_df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank teams by value efficiency.

    Returns teams sorted by how much value they're getting relative to spending.
    """
    df = financials_df.copy()

    # Filter to teams with meaningful spending
    df = df[df['total_spent'] > 1_000_000]  # At least 1M spent

    # Calculate efficiency percentile
    df['efficiency_rank'] = df['value_efficiency'].rank(pct=True) * 100

    # Categorize
    def categorize_efficiency(eff):
        if pd.isna(eff):
            return "Unknown"
        if eff >= 2.0:
            return "Excellent (2x+ value)"
        elif eff >= 1.0:
            return "Good (value >= cost)"
        elif eff >= 0.5:
            return "Fair (50%+ value)"
        else:
            return "Poor (<50% value)"

    df['efficiency_category'] = df['value_efficiency'].apply(categorize_efficiency)

    return df.sort_values('value_efficiency', ascending=False)


def get_smart_spenders(financials_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Get teams that are getting the most value for their money.

    These are teams with high squad value relative to their spending -
    either through smart recruitment, good player development, or both.
    """
    df = financials_df.copy()

    # Need some spending to be considered
    df = df[df['total_spent'] > 5_000_000]  # At least 5M spent

    # Sort by value vs cost (how much more value than what was paid)
    df = df.nlargest(top_n, 'value_vs_cost')

    return df


def get_big_spenders(financials_df: pd.DataFrame, top_n: int = 20) -> pd.DataFrame:
    """
    Get teams that have spent the most in transfers.
    """
    df = financials_df.copy()
    return df.nlargest(top_n, 'total_spent')


def get_transfer_summary_stats(financials_df: pd.DataFrame) -> Dict:
    """
    Get summary statistics across all teams.
    """
    return {
        'total_market_value': financials_df['squad_value'].sum(),
        'total_spending': financials_df['total_spent'].sum(),
        'avg_squad_value': financials_df['squad_value'].mean(),
        'avg_net_spend': financials_df['net_spend'].mean(),
        'most_valuable_club': financials_df.loc[financials_df['squad_value'].idxmax(), 'club_name'] if len(financials_df) > 0 else None,
        'biggest_spender': financials_df.loc[financials_df['total_spent'].idxmax(), 'club_name'] if len(financials_df) > 0 else None,
        'most_efficient': financials_df.loc[financials_df['value_efficiency'].idxmax(), 'club_name'] if len(financials_df) > 0 and financials_df['value_efficiency'].notna().any() else None,
    }


def format_currency(value: float, compact: bool = True) -> str:
    """Format a currency value for display."""
    if pd.isna(value):
        return "-"

    if compact:
        if abs(value) >= 1_000_000_000:
            return f"€{value/1_000_000_000:.1f}B"
        elif abs(value) >= 1_000_000:
            return f"€{value/1_000_000:.1f}M"
        elif abs(value) >= 1_000:
            return f"€{value/1_000:.0f}K"
        else:
            return f"€{value:.0f}"
    else:
        return f"€{value:,.0f}"
