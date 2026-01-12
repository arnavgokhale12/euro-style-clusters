"""
Configuration file for European Soccer Style Clustering project.

Contains constants, file paths, and configuration parameters used throughout
the pipeline.
"""

# Database configuration
DATABASE_PATH = "data/database.sqlite"

# League IDs for top 5 European leagues (Kaggle database)
# These may need to be verified during data exploration
TOP_5_LEAGUES = {
    "Premier League": 1729,  # England
    "La Liga": 21518,        # Spain
    "Serie A": 10257,        # Italy
    "Bundesliga": 7809,      # Germany
    "Ligue 1": 4769          # France
}

# Feature groups
FEATURE_GROUPS = {
    "attack": ["goals_scored", "shots", "assists", "key_passes"],
    "possession": ["possession", "pass_completion", "passes", "touches"],
    "defense": ["goals_conceded", "tackles", "interceptions", "clearances"],
    "game_state": ["goals_when_leading", "goals_when_trailing", "set_pieces"]
}

# Clustering parameters
K_MEANS_RANGE = range(2, 11)  # Range of k values to test
RANDOM_STATE = 42
N_COMPONENTS_PCA = 2  # For dimensionality reduction visualization

# Output paths
REPORTS_DIR = "reports"
FIGURES_DIR = "reports/figures"
