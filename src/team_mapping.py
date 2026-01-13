"""
Team Name Mapping Module

Provides fuzzy matching between Kaggle European Soccer Database team names
and TransferMarkt club names to enable merging of style and financial data.
"""

import re
import unicodedata
from difflib import SequenceMatcher
from typing import Dict, Optional, Tuple
import pandas as pd


# League ID mappings between datasets
KAGGLE_TO_TM_LEAGUE = {
    1729: "GB1",   # England Premier League
    21518: "ES1",  # Spain La Liga
    10257: "IT1",  # Italy Serie A
    7809: "L1",    # Germany Bundesliga
    4769: "FR1",   # France Ligue 1
}

TM_TO_KAGGLE_LEAGUE = {v: k for k, v in KAGGLE_TO_TM_LEAGUE.items()}


def normalize_team_name(name: str) -> str:
    """
    Normalize team name for matching.

    - Removes common suffixes (FC, Football Club, etc.)
    - Converts to lowercase
    - Removes accents/diacritics
    - Strips extra whitespace and punctuation
    """
    if pd.isna(name):
        return ""

    name = str(name)

    # Convert to lowercase
    name = name.lower()

    # Remove accents/diacritics
    name = unicodedata.normalize('NFKD', name)
    name = ''.join(c for c in name if not unicodedata.combining(c))

    # Remove common suffixes and prefixes
    patterns_to_remove = [
        r'\bfootball club\b',
        r'\bfutbol club\b',
        r'\bclub de futbol\b',
        r'\bs\.?a\.?d\.?\b',  # S.A.D. (Spanish legal suffix)
        r'\bs\.?p\.?a\.?\b',  # S.p.A. (Italian legal suffix)
        r'\bs\.?r\.?l\.?\b',  # S.r.l. (Italian)
        r'\b1\.\s*',          # German numbering (1. FC Köln)
        r'\bfc\b',
        r'\bafc\b',
        r'\bcf\b',
        r'\bac\b',
        r'\bas\b',
        r'\bsc\b',
        r'\bssc\b',
        r'\bog\b',
        r'\bvfb\b',
        r'\btsv\b',
        r'\bvfl\b',
        r'\bsv\b',
        r'\bverein fur bewegungsspiele\b',
        r'\bverein fur leibesubungen\b',
        r'\bassociazione sportiva\b',
        r'\bunione sportiva\b',
        r'\bsocieta sportiva\b',
        r'\breal\b',  # Keep for matching but remove for normalization
    ]

    for pattern in patterns_to_remove:
        name = re.sub(pattern, ' ', name, flags=re.IGNORECASE)

    # Remove punctuation except spaces
    name = re.sub(r'[^\w\s]', ' ', name)

    # Normalize whitespace
    name = ' '.join(name.split())

    return name.strip()


def calculate_similarity(name1: str, name2: str) -> float:
    """Calculate similarity ratio between two normalized names."""
    norm1 = normalize_team_name(name1)
    norm2 = normalize_team_name(name2)

    if not norm1 or not norm2:
        return 0.0

    # Direct match
    if norm1 == norm2:
        return 1.0

    # Check if one contains the other
    if norm1 in norm2 or norm2 in norm1:
        return 0.9

    # Sequence matcher
    return SequenceMatcher(None, norm1, norm2).ratio()


def create_team_mapping(
    kaggle_teams: pd.DataFrame,
    transfermarkt_clubs: pd.DataFrame,
    similarity_threshold: float = 0.75
) -> Dict[int, int]:
    """
    Create mapping between Kaggle team_api_id and TransferMarkt club_id.

    Args:
        kaggle_teams: DataFrame with 'team_api_id', 'team_long_name', optionally 'league_id'
        transfermarkt_clubs: DataFrame with 'club_id', 'club_name', optionally 'domestic_competition_id'
        similarity_threshold: Minimum similarity score to consider a match

    Returns:
        Dict mapping kaggle team_api_id -> transfermarkt club_id
    """
    mapping = {}

    # Get unique teams from each dataset
    kaggle_unique = kaggle_teams.drop_duplicates(subset=['team_api_id'])
    tm_unique = transfermarkt_clubs.drop_duplicates(subset=['club_id'])

    for _, kg_row in kaggle_unique.iterrows():
        kg_id = kg_row['team_api_id']
        kg_name = kg_row.get('team_long_name', '')
        kg_league = kg_row.get('league_id')

        best_match = None
        best_score = 0.0

        # If we know the league, filter TM clubs to same league first
        tm_candidates = tm_unique
        if kg_league and kg_league in KAGGLE_TO_TM_LEAGUE:
            tm_league = KAGGLE_TO_TM_LEAGUE[kg_league]
            if 'domestic_competition_id' in tm_unique.columns:
                league_filtered = tm_unique[tm_unique['domestic_competition_id'] == tm_league]
                if len(league_filtered) > 0:
                    tm_candidates = league_filtered

        for _, tm_row in tm_candidates.iterrows():
            tm_id = tm_row['club_id']
            tm_name = tm_row.get('club_name', tm_row.get('name', ''))

            score = calculate_similarity(kg_name, tm_name)

            if score > best_score and score >= similarity_threshold:
                best_score = score
                best_match = tm_id

        if best_match is not None:
            mapping[kg_id] = best_match

    return mapping


def get_team_financials(
    team_api_id: int,
    mapping: Dict[int, int],
    financials_df: pd.DataFrame
) -> Optional[pd.Series]:
    """
    Get financial data for a Kaggle team using the mapping.

    Args:
        team_api_id: Kaggle team API ID
        mapping: Dict from create_team_mapping()
        financials_df: TransferMarkt financials DataFrame

    Returns:
        Series with financial data or None if not found
    """
    if team_api_id not in mapping:
        return None

    club_id = mapping[team_api_id]

    matches = financials_df[financials_df['club_id'] == club_id]
    if len(matches) == 0:
        return None

    return matches.iloc[0]


def merge_financial_data(
    style_df: pd.DataFrame,
    financials_df: pd.DataFrame,
    mapping: Dict[int, int]
) -> pd.DataFrame:
    """
    Merge financial data into style clustering DataFrame.

    Args:
        style_df: DataFrame with style clustering data (must have 'team_api_id')
        financials_df: TransferMarkt financials DataFrame
        mapping: Dict from create_team_mapping()

    Returns:
        style_df with added financial columns (squad_value, total_spent, value_efficiency, etc.)
    """
    result = style_df.copy()

    # Add mapped club_id column
    result['tm_club_id'] = result['team_api_id'].map(mapping)

    # Select financial columns to merge
    fin_cols = ['club_id', 'squad_value', 'total_spent', 'total_received',
                'net_spend', 'value_efficiency', 'value_vs_cost']
    available_cols = [c for c in fin_cols if c in financials_df.columns]

    if 'club_id' not in available_cols:
        return result

    # Merge on mapped club_id
    result = result.merge(
        financials_df[available_cols],
        left_on='tm_club_id',
        right_on='club_id',
        how='left',
        suffixes=('', '_fin')
    )

    # Clean up duplicate club_id column if created
    if 'club_id_fin' in result.columns:
        result = result.drop(columns=['club_id_fin'])
    if 'club_id' in result.columns and 'tm_club_id' in result.columns:
        result = result.drop(columns=['club_id'])

    return result
