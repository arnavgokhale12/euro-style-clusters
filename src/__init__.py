"""
European Soccer Style Clustering Package

A package for clustering European club teams into distinct playing styles
using match and season data from the Kaggle European Soccer Database.
"""

__version__ = "0.1.0"

from .data_loader import (
    connect_to_database,
    get_table_names,
    load_matches,
    load_teams,
    load_team_attributes,
    load_league_info,
    load_top5_matches,
)

from .feature_engineering import (
    transform_to_team_centric,
    aggregate_team_season_features,
    normalize_features,
    build_feature_pipeline,
    get_feature_matrix,
)
