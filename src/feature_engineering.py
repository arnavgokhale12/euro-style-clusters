"""
Feature engineering module for team-season level aggregation.

Separates STYLE features (how a team plays) from QUALITY features (how well they play).
Style features are used for clustering; quality features are analyzed as outcomes.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
import xml.etree.ElementTree as ET


# =============================================================================
# FEATURE DEFINITIONS
# =============================================================================

# STYLE FEATURES: Describe "how" a team plays (use for clustering)
STYLE_FEATURES = [
    "avg_possession",      # Ball control preference
    "avg_shots",           # Shot frequency / attacking intent
    "avg_crosses",         # Width of play / crossing tendency
    "avg_corners",         # Set piece focus
    "avg_fouls",           # Aggression / physicality
]

# REDUCED STYLE FEATURES: For post-2016 data where possession/crosses unavailable
STYLE_FEATURES_REDUCED = [
    "avg_shots",           # Shot frequency / attacking intent
    "avg_corners",         # Set piece focus
    "avg_fouls",           # Aggression / physicality
]

# QUALITY FEATURES: Describe "how well" a team plays (analyze after clustering)
QUALITY_FEATURES = [
    "win_rate",            # Primary success metric
    "avg_goals_scored",    # Attacking output
    "avg_goals_conceded",  # Defensive solidity
    "goal_difference",     # Net performance
    "shot_conversion_rate", # Finishing efficiency
    "points_per_game",     # League performance
]

# METADATA COLUMNS: IDs and descriptors (not for clustering)
METADATA_COLS = [
    "team_api_id",
    "season",
    "league_id",
    "matches_played",
    "wins",
    "draws",
    "losses",
    "points",
]


def parse_xml_column(xml_string: str) -> List[Dict]:
    """
    Parse XML string from match event columns.

    Args:
        xml_string: XML formatted string from database

    Returns:
        List of dictionaries containing parsed event data
    """
    if pd.isna(xml_string) or not xml_string:
        return []
    try:
        root = ET.fromstring(f"<root>{xml_string}</root>")
        events = []
        for value in root.findall(".//value"):
            event = {}
            for child in value:
                event[child.tag] = child.text
            if event:
                events.append(event)
        return events
    except ET.ParseError:
        return []


def count_events_by_team(events: List[Dict], team_id: int, team_field: str = "team") -> int:
    """
    Count events belonging to a specific team.

    Args:
        events: List of event dictionaries
        team_id: Team API ID to filter for
        team_field: Field name containing team ID in events

    Returns:
        Count of events for the team
    """
    count = 0
    for event in events:
        event_team = event.get(team_field)
        if event_team is not None:
            try:
                if int(event_team) == team_id:
                    count += 1
            except (ValueError, TypeError):
                pass
    return count


def transform_to_team_centric(matches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transform match data from home/away format to team-centric format.

    Each match produces two rows: one for home team perspective, one for away.

    Args:
        matches_df: DataFrame with match data in home/away format

    Returns:
        DataFrame with one row per team per match
    """
    home_rows = []
    away_rows = []

    # Columns to keep from original match data
    match_cols = ["id", "league_id", "season", "date"]

    for _, match in matches_df.iterrows():
        base_data = {col: match.get(col) for col in match_cols if col in match.index}

        # Home team perspective
        home_data = base_data.copy()
        home_data["team_api_id"] = match.get("home_team_api_id")
        home_data["opponent_api_id"] = match.get("away_team_api_id")
        home_data["is_home"] = True
        home_data["goals_scored"] = match.get("home_team_goal", 0)
        home_data["goals_conceded"] = match.get("away_team_goal", 0)

        # Parse XML event columns for home team stats
        home_team_id = match.get("home_team_api_id")
        away_team_id = match.get("away_team_api_id")

        # Shots on target
        shoton_events = parse_xml_column(match.get("shoton", ""))
        home_data["shots_on_target"] = count_events_by_team(shoton_events, home_team_id)

        # Shots off target
        shotoff_events = parse_xml_column(match.get("shotoff", ""))
        home_data["shots_off_target"] = count_events_by_team(shotoff_events, home_team_id)

        # Total shots
        home_data["shots"] = home_data["shots_on_target"] + home_data["shots_off_target"]

        # Crosses
        cross_events = parse_xml_column(match.get("cross", ""))
        home_data["crosses"] = count_events_by_team(cross_events, home_team_id)

        # Corners
        corner_events = parse_xml_column(match.get("corner", ""))
        home_data["corners"] = count_events_by_team(corner_events, home_team_id)

        # Fouls committed
        foul_events = parse_xml_column(match.get("foulcommit", ""))
        home_data["fouls_committed"] = count_events_by_team(foul_events, home_team_id)

        # Cards received
        card_events = parse_xml_column(match.get("card", ""))
        home_data["cards"] = count_events_by_team(card_events, home_team_id)

        # Possession (if available as numeric)
        possession = match.get("possession")
        if pd.notna(possession):
            poss_events = parse_xml_column(str(possession))
            # Possession is typically home/away split
            home_data["possession"] = 50  # Default
            for event in poss_events:
                if event.get("homepos"):
                    try:
                        home_data["possession"] = float(event["homepos"])
                    except (ValueError, TypeError):
                        pass

        # Goal difference and result
        home_data["goal_diff"] = home_data["goals_scored"] - home_data["goals_conceded"]
        home_data["win"] = 1 if home_data["goal_diff"] > 0 else 0
        home_data["draw"] = 1 if home_data["goal_diff"] == 0 else 0
        home_data["loss"] = 1 if home_data["goal_diff"] < 0 else 0

        home_rows.append(home_data)

        # Away team perspective
        away_data = base_data.copy()
        away_data["team_api_id"] = match.get("away_team_api_id")
        away_data["opponent_api_id"] = match.get("home_team_api_id")
        away_data["is_home"] = False
        away_data["goals_scored"] = match.get("away_team_goal", 0)
        away_data["goals_conceded"] = match.get("home_team_goal", 0)

        # Parse XML event columns for away team stats
        away_data["shots_on_target"] = count_events_by_team(shoton_events, away_team_id)
        away_data["shots_off_target"] = count_events_by_team(shotoff_events, away_team_id)
        away_data["shots"] = away_data["shots_on_target"] + away_data["shots_off_target"]
        away_data["crosses"] = count_events_by_team(cross_events, away_team_id)
        away_data["corners"] = count_events_by_team(corner_events, away_team_id)
        away_data["fouls_committed"] = count_events_by_team(foul_events, away_team_id)
        away_data["cards"] = count_events_by_team(card_events, away_team_id)

        # Away possession
        away_data["possession"] = 50
        if pd.notna(possession):
            for event in poss_events:
                if event.get("awaypos"):
                    try:
                        away_data["possession"] = float(event["awaypos"])
                    except (ValueError, TypeError):
                        pass

        away_data["goal_diff"] = away_data["goals_scored"] - away_data["goals_conceded"]
        away_data["win"] = 1 if away_data["goal_diff"] > 0 else 0
        away_data["draw"] = 1 if away_data["goal_diff"] == 0 else 0
        away_data["loss"] = 1 if away_data["goal_diff"] < 0 else 0

        away_rows.append(away_data)

    df = pd.concat([pd.DataFrame(home_rows), pd.DataFrame(away_rows)], ignore_index=True)
    return df


def aggregate_team_season_features(matches_df: pd.DataFrame,
                                   team_id_col: str = "team_api_id",
                                   season_col: str = "season") -> pd.DataFrame:
    """
    Aggregate match-level statistics to team-season level features.

    For each team-season combination, compute aggregated statistics for:
    - Attack: goals scored, shots, shot accuracy
    - Possession: possession percentage, crosses, corners
    - Defense: goals conceded, clean sheets, fouls
    - Game state: win rate, home/away performance

    Args:
        matches_df: DataFrame containing match-level statistics (team-centric format)
        team_id_col: Name of column containing team IDs
        season_col: Name of column containing season information

    Returns:
        DataFrame with one row per team-season and aggregated features
    """
    # If match data is still in home/away format, transform it
    if "home_team_api_id" in matches_df.columns and team_id_col not in matches_df.columns:
        matches_df = transform_to_team_centric(matches_df)

    # Ensure numeric columns
    numeric_cols = ["goals_scored", "goals_conceded", "shots", "shots_on_target",
                    "shots_off_target", "crosses", "corners", "fouls_committed",
                    "cards", "possession", "win", "draw", "loss"]

    for col in numeric_cols:
        if col in matches_df.columns:
            matches_df[col] = pd.to_numeric(matches_df[col], errors="coerce").fillna(0)

    # Group by team and season
    grouped = matches_df.groupby([team_id_col, season_col])

    # Aggregate features
    agg_dict = {
        # Match counts
        "id": "count",

        # Attack features
        "goals_scored": ["sum", "mean"],
        "shots": ["sum", "mean"],
        "shots_on_target": ["sum", "mean"],

        # Possession features
        "possession": "mean",
        "crosses": ["sum", "mean"],
        "corners": ["sum", "mean"],

        # Defense features
        "goals_conceded": ["sum", "mean"],
        "fouls_committed": ["sum", "mean"],
        "cards": ["sum", "mean"],

        # Results
        "win": "sum",
        "draw": "sum",
        "loss": "sum",
    }

    # Filter to only existing columns
    agg_dict = {k: v for k, v in agg_dict.items() if k in matches_df.columns}

    aggregated = grouped.agg(agg_dict)

    # Flatten column names
    aggregated.columns = ["_".join(col).strip("_") for col in aggregated.columns]

    # Rename for clarity
    rename_map = {
        "id_count": "matches_played",
        "goals_scored_sum": "total_goals_scored",
        "goals_scored_mean": "avg_goals_scored",
        "goals_conceded_sum": "total_goals_conceded",
        "goals_conceded_mean": "avg_goals_conceded",
        "shots_sum": "total_shots",
        "shots_mean": "avg_shots",
        "shots_on_target_sum": "total_shots_on_target",
        "shots_on_target_mean": "avg_shots_on_target",
        "possession_mean": "avg_possession",
        "crosses_sum": "total_crosses",
        "crosses_mean": "avg_crosses",
        "corners_sum": "total_corners",
        "corners_mean": "avg_corners",
        "fouls_committed_sum": "total_fouls",
        "fouls_committed_mean": "avg_fouls",
        "cards_sum": "total_cards",
        "cards_mean": "avg_cards",
        "win_sum": "wins",
        "draw_sum": "draws",
        "loss_sum": "losses",
    }
    aggregated = aggregated.rename(columns=rename_map)

    # Reset index
    aggregated = aggregated.reset_index()

    # Compute derived features
    aggregated = compute_derived_features(aggregated)

    return aggregated


def compute_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute derived features from aggregated statistics.

    Args:
        df: DataFrame with aggregated team-season statistics

    Returns:
        DataFrame with additional derived features
    """
    # Avoid division by zero
    eps = 1e-10

    # Attack efficiency
    if "total_goals_scored" in df.columns and "total_shots" in df.columns:
        df["shot_conversion_rate"] = df["total_goals_scored"] / (df["total_shots"] + eps)

    if "total_shots_on_target" in df.columns and "total_shots" in df.columns:
        df["shot_accuracy"] = df["total_shots_on_target"] / (df["total_shots"] + eps)

    # Goal difference
    if "total_goals_scored" in df.columns and "total_goals_conceded" in df.columns:
        df["goal_difference"] = df["total_goals_scored"] - df["total_goals_conceded"]
        df["goals_per_conceded"] = df["total_goals_scored"] / (df["total_goals_conceded"] + eps)

    # Win rate
    if "wins" in df.columns and "matches_played" in df.columns:
        df["win_rate"] = df["wins"] / (df["matches_played"] + eps)
        df["points"] = df["wins"] * 3 + df.get("draws", 0)
        df["points_per_game"] = df["points"] / (df["matches_played"] + eps)

    # Clean sheets (games with 0 goals conceded) - approximated
    if "avg_goals_conceded" in df.columns:
        # Lower average goals conceded suggests more clean sheets
        df["defensive_strength"] = 1 / (df["avg_goals_conceded"] + eps)

    # Discipline
    if "total_cards" in df.columns and "matches_played" in df.columns:
        df["cards_per_game"] = df["total_cards"] / (df["matches_played"] + eps)

    if "total_fouls" in df.columns and "matches_played" in df.columns:
        df["fouls_per_game"] = df["total_fouls"] / (df["matches_played"] + eps)

    # Set piece threat (corners + crosses as proxy)
    if "total_corners" in df.columns and "total_crosses" in df.columns:
        df["set_piece_threat"] = df["total_corners"] + df["total_crosses"]

    return df


def compute_attack_features(matches_df: pd.DataFrame,
                            is_home: bool = True) -> pd.DataFrame:
    """
    Compute attack-related features from match data.

    Args:
        matches_df: DataFrame containing match statistics
        is_home: If True, compute for home team; if False, for away team

    Returns:
        DataFrame with attack features
    """
    prefix = "home" if is_home else "away"
    team_col = f"{prefix}_team_api_id"
    goal_col = f"{prefix}_team_goal"

    features = pd.DataFrame()
    features["team_api_id"] = matches_df[team_col]
    features["season"] = matches_df["season"]
    features["goals_scored"] = matches_df[goal_col]

    # Parse shots from XML if available
    if "shoton" in matches_df.columns:
        features["shots_on_target"] = matches_df.apply(
            lambda row: count_events_by_team(
                parse_xml_column(row.get("shoton", "")),
                row[team_col]
            ), axis=1
        )

    if "shotoff" in matches_df.columns:
        features["shots_off_target"] = matches_df.apply(
            lambda row: count_events_by_team(
                parse_xml_column(row.get("shotoff", "")),
                row[team_col]
            ), axis=1
        )

    if "shots_on_target" in features.columns and "shots_off_target" in features.columns:
        features["shots"] = features["shots_on_target"] + features["shots_off_target"]

    return features


def compute_possession_features(matches_df: pd.DataFrame,
                                is_home: bool = True) -> pd.DataFrame:
    """
    Compute possession-related features from match data.

    Args:
        matches_df: DataFrame containing match statistics
        is_home: If True, compute for home team; if False, for away team

    Returns:
        DataFrame with possession features
    """
    prefix = "home" if is_home else "away"
    team_col = f"{prefix}_team_api_id"
    pos_field = "homepos" if is_home else "awaypos"

    features = pd.DataFrame()
    features["team_api_id"] = matches_df[team_col]
    features["season"] = matches_df["season"]

    # Parse possession from XML
    def get_possession(row):
        poss_events = parse_xml_column(str(row.get("possession", "")))
        for event in poss_events:
            if event.get(pos_field):
                try:
                    return float(event[pos_field])
                except (ValueError, TypeError):
                    pass
        return 50.0  # Default

    features["possession"] = matches_df.apply(get_possession, axis=1)

    # Crosses and corners as passing/build-up proxies
    if "cross" in matches_df.columns:
        features["crosses"] = matches_df.apply(
            lambda row: count_events_by_team(
                parse_xml_column(row.get("cross", "")),
                row[team_col]
            ), axis=1
        )

    if "corner" in matches_df.columns:
        features["corners"] = matches_df.apply(
            lambda row: count_events_by_team(
                parse_xml_column(row.get("corner", "")),
                row[team_col]
            ), axis=1
        )

    return features


def compute_defense_features(matches_df: pd.DataFrame,
                             is_home: bool = True) -> pd.DataFrame:
    """
    Compute defense-related features from match data.

    Args:
        matches_df: DataFrame containing match statistics
        is_home: If True, compute for home team; if False, for away team

    Returns:
        DataFrame with defense features
    """
    prefix = "home" if is_home else "away"
    opp_prefix = "away" if is_home else "home"
    team_col = f"{prefix}_team_api_id"
    opp_goal_col = f"{opp_prefix}_team_goal"

    features = pd.DataFrame()
    features["team_api_id"] = matches_df[team_col]
    features["season"] = matches_df["season"]
    features["goals_conceded"] = matches_df[opp_goal_col]

    # Fouls committed (discipline/aggression)
    if "foulcommit" in matches_df.columns:
        features["fouls_committed"] = matches_df.apply(
            lambda row: count_events_by_team(
                parse_xml_column(row.get("foulcommit", "")),
                row[team_col]
            ), axis=1
        )

    # Cards received
    if "card" in matches_df.columns:
        features["cards"] = matches_df.apply(
            lambda row: count_events_by_team(
                parse_xml_column(row.get("card", "")),
                row[team_col]
            ), axis=1
        )

    return features


def compute_game_state_features(matches_df: pd.DataFrame,
                                is_home: bool = True) -> pd.DataFrame:
    """
    Compute game state features (leading/trailing performance, set pieces).

    Args:
        matches_df: DataFrame containing match statistics
        is_home: If True, compute for home team; if False, for away team

    Returns:
        DataFrame with game state features
    """
    prefix = "home" if is_home else "away"
    opp_prefix = "away" if is_home else "home"
    team_col = f"{prefix}_team_api_id"
    goal_col = f"{prefix}_team_goal"
    opp_goal_col = f"{opp_prefix}_team_goal"

    features = pd.DataFrame()
    features["team_api_id"] = matches_df[team_col]
    features["season"] = matches_df["season"]

    # Match result indicators
    goal_diff = matches_df[goal_col] - matches_df[opp_goal_col]
    features["win"] = (goal_diff > 0).astype(int)
    features["draw"] = (goal_diff == 0).astype(int)
    features["loss"] = (goal_diff < 0).astype(int)
    features["goal_difference"] = goal_diff

    # Set pieces (corners + crosses as proxy)
    if "corner" in matches_df.columns:
        features["corners"] = matches_df.apply(
            lambda row: count_events_by_team(
                parse_xml_column(row.get("corner", "")),
                row[team_col]
            ), axis=1
        )

    if "cross" in matches_df.columns:
        features["crosses"] = matches_df.apply(
            lambda row: count_events_by_team(
                parse_xml_column(row.get("cross", "")),
                row[team_col]
            ), axis=1
        )

    return features


def normalize_features(features_df: pd.DataFrame,
                       method: str = "standard",
                       exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize features for clustering algorithms.

    Args:
        features_df: DataFrame with features to normalize
        method: Normalization method ('standard', 'minmax', 'robust')
        exclude_cols: Columns to exclude from normalization (e.g., IDs, season)

    Returns:
        DataFrame with normalized features
    """
    if exclude_cols is None:
        exclude_cols = ["team_api_id", "season", "league_id"]

    # Identify columns to normalize
    numeric_cols = features_df.select_dtypes(include=[np.number]).columns.tolist()
    cols_to_normalize = [col for col in numeric_cols if col not in exclude_cols]

    # Select scaler
    if method == "standard":
        scaler = StandardScaler()
    elif method == "minmax":
        scaler = MinMaxScaler()
    elif method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")

    # Create output DataFrame
    result = features_df.copy()

    if cols_to_normalize:
        # Handle any remaining NaN values
        result[cols_to_normalize] = result[cols_to_normalize].fillna(0)

        # Apply normalization
        result[cols_to_normalize] = scaler.fit_transform(result[cols_to_normalize])

    return result


def get_feature_matrix(features_df: pd.DataFrame,
                       feature_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Extract feature matrix for clustering, using only STYLE features by default.

    Args:
        features_df: DataFrame with all features
        feature_cols: Specific columns to include. If None, uses STYLE_FEATURES.

    Returns:
        DataFrame with only clustering features
    """
    if feature_cols is None:
        # Default to style features only for clustering
        feature_cols = [col for col in STYLE_FEATURES if col in features_df.columns]

    return features_df[feature_cols].copy()


def get_style_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only style features (for clustering).

    Style features describe HOW a team plays, not how well.
    """
    cols = [col for col in STYLE_FEATURES if col in features_df.columns]
    return features_df[cols].copy()


def get_quality_features(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract only quality features (for outcome analysis).

    Quality features describe HOW WELL a team plays.
    """
    cols = [col for col in QUALITY_FEATURES if col in features_df.columns]
    return features_df[cols].copy()


def build_feature_pipeline(matches_df: pd.DataFrame,
                           normalize: bool = True,
                           normalize_method: str = "standard",
                           normalize_style_only: bool = True) -> pd.DataFrame:
    """
    Complete feature engineering pipeline from raw matches to clustering-ready features.

    Args:
        matches_df: Raw match DataFrame from data_loader
        normalize: Whether to normalize features
        normalize_method: Normalization method if normalize=True
        normalize_style_only: If True, only normalize STYLE features (recommended
                              for clustering so quality metrics remain interpretable)

    Returns:
        DataFrame with team-season rows containing both style and quality features.
        Style features are normalized (if requested); quality features kept raw.
    """
    # Transform to team-centric format
    team_matches = transform_to_team_centric(matches_df)

    # Aggregate to team-season level
    features = aggregate_team_season_features(team_matches)

    # Normalize if requested
    if normalize:
        if normalize_style_only:
            # Only normalize style features, keep quality features interpretable
            style_cols = [col for col in STYLE_FEATURES if col in features.columns]
            features = normalize_features(
                features,
                method=normalize_method,
                exclude_cols=METADATA_COLS + QUALITY_FEATURES
            )
        else:
            # Normalize all numeric features
            features = normalize_features(features, method=normalize_method)

    return features


def build_features_separated(matches_df: pd.DataFrame,
                             normalize_method: str = "standard") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Build features and return style and quality features separately.

    This is the recommended pipeline for clustering analysis.

    Args:
        matches_df: Raw match DataFrame from data_loader
        normalize_method: Normalization method for style features

    Returns:
        Tuple of (metadata_df, style_df_normalized, quality_df_raw):
        - metadata_df: Team-season identifiers
        - style_df_normalized: Normalized style features for clustering
        - quality_df_raw: Raw quality features for outcome analysis
    """
    # Transform and aggregate
    team_matches = transform_to_team_centric(matches_df)
    features = aggregate_team_season_features(team_matches)

    # Extract metadata
    meta_cols = [col for col in METADATA_COLS if col in features.columns]
    metadata_df = features[meta_cols].copy()

    # Extract and normalize style features
    style_cols = [col for col in STYLE_FEATURES if col in features.columns]
    style_df = features[style_cols].copy()

    # Normalize style features
    if normalize_method == "standard":
        scaler = StandardScaler()
    elif normalize_method == "minmax":
        scaler = MinMaxScaler()
    elif normalize_method == "robust":
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {normalize_method}")

    style_df_normalized = pd.DataFrame(
        scaler.fit_transform(style_df.fillna(0)),
        columns=style_cols,
        index=features.index
    )

    # Extract quality features (keep raw for interpretability)
    quality_cols = [col for col in QUALITY_FEATURES if col in features.columns]
    quality_df = features[quality_cols].copy()

    return metadata_df, style_df_normalized, quality_df
