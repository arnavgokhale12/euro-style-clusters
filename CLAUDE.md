# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project provides two main analyses of European soccer teams from the top 5 leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1):

1. **Style Clustering (2008-2016)**: Clusters teams into playing styles using the Kaggle European Soccer Database with k-means/hierarchical clustering on match statistics.

2. **MoneyBall Analysis (2015-2026)**: Analyzes transfer market efficiency using TransferMarkt data - showing which teams get the best value for their spending vs those just throwing cash around.

## Setup

```bash
pip install -r requirements.txt
```

**For Style Clustering:**
The SQLite database must be placed at `data/database.sqlite` (download from Kaggle: https://www.kaggle.com/datasets/hugomathien/soccer).

**For MoneyBall Analysis:**
```bash
python setup_transfer_data.py
```
This downloads transfer market data from [salimt/football-datasets](https://github.com/salimt/football-datasets) to `data/transfermarkt/`.

## Running the Project

**Interactive Dashboard (recommended):**
```bash
streamlit run app.py
```
Features:
- Style clustering with adjustable k (3-8 clusters)
- League/season filters
- Team search and timeline views
- **MoneyBall tab**: Squad value vs cost analysis, efficiency rankings, net spend, ROI metrics

**CLI Analysis:**
```bash
python main.py
```
Outputs to `reports/`: cluster_assignments.csv, cluster_summary.csv, figures/

**Data Exploration:**
```bash
jupyter notebook notebooks/01_explore_data.ipynb
```

## Architecture

The codebase follows a modular pipeline architecture:

### Style Clustering Pipeline

1. **`src/data_loader.py`** - SQLite database connection and data extraction. Filters matches by league IDs defined in config.

2. **`src/feature_engineering.py`** - Aggregates match-level stats to team-season level across four dimensions:
   - Attack: goals, shots, assists, key passes
   - Possession: possession %, pass completion, total passes
   - Defense: goals conceded, tackles, interceptions, clearances
   - Game state: leading/trailing performance, set pieces

3. **`src/clustering.py`** - Implements k-means (with optimal k selection via silhouette score) and hierarchical clustering (ward linkage).

4. **`src/visualization.py`** - PCA/t-SNE projections, cluster center plots, and method comparisons.

5. **`src/config.py`** - Central configuration including:
   - `TOP_5_LEAGUES`: League ID mapping for filtering
   - `FEATURE_GROUPS`: Feature category definitions
   - `K_MEANS_RANGE`: k values to test (2-10)
   - `RANDOM_STATE`: 42 for reproducibility

### MoneyBall Pipeline

6. **`src/transfer_data.py`** - Transfer market data processing:
   - Downloads data from GitHub (salimt/football-datasets)
   - Parses transfer fees and market valuations
   - Calculates team financials: squad value, total spent, net spend
   - Computes efficiency metrics: value_efficiency = squad_value / total_spent
   - Key functions: `calculate_team_financials()`, `get_smart_spenders()`, `get_big_spenders()`

7. **`setup_transfer_data.py`** - One-time setup script to download TransferMarkt data

## Key Implementation Notes

- `data_loader.py` and `feature_engineering.py` are fully implemented
- `clustering.py`, `visualization.py`, `utils.py` remain stubs
- `main.py` provides end-to-end MVP analysis using sklearn directly
- Match data is unpivoted from home/away format to team-centric via `transform_to_team_centric()`
- XML event columns (shoton, shotoff, cross, corner, etc.) are parsed for detailed stats
- Features are normalized before clustering (standard/minmax/robust scaling available)
- The database uses `league_id` to filter matches and `team_api_id` to identify teams
