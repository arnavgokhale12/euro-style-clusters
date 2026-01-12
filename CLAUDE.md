# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This project clusters European soccer teams from the top 5 leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1) into distinct playing styles using the Kaggle European Soccer Database (2008-2016). It uses unsupervised learning (k-means and hierarchical clustering) on team-season aggregated features.

## Setup

```bash
pip install -r requirements.txt
```

The SQLite database must be placed at `data/database.sqlite` (download from Kaggle: https://www.kaggle.com/datasets/hugomathien/soccer).

## Running the Project

**Interactive Dashboard (recommended):**
```bash
streamlit run app.py
```
Features: cluster slider, league/season filters, team search, interactive plots

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

## Key Implementation Notes

- `data_loader.py` and `feature_engineering.py` are fully implemented
- `clustering.py`, `visualization.py`, `utils.py` remain stubs
- `main.py` provides end-to-end MVP analysis using sklearn directly
- Match data is unpivoted from home/away format to team-centric via `transform_to_team_centric()`
- XML event columns (shoton, shotoff, cross, corner, etc.) are parsed for detailed stats
- Features are normalized before clustering (standard/minmax/robust scaling available)
- The database uses `league_id` to filter matches and `team_api_id` to identify teams
