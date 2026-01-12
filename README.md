# European Soccer Style Clustering

## Project Description

This project clusters European club teams from the top 5 leagues (Premier League, La Liga, Serie A, Bundesliga, Ligue 1) into distinct styles of play using match and season data from the Kaggle European Soccer Database.

The analysis aggregates team-season level features across multiple dimensions:
- **Attack**: Goals scored, shots, assists, attacking style
- **Possession**: Ball possession, pass completion, build-up play
- **Defense**: Goals conceded, tackles, interceptions, defensive shape
- **Game State**: Performance when leading/trailing, set pieces, transitions

Using unsupervised learning (k-means and hierarchical clustering), we identify distinct playing styles and visualize the clusters through dimensionality reduction and comparative analyses.

## Dataset

**Source**: [Kaggle European Soccer Database](https://www.kaggle.com/datasets/hugomathien/soccer)

**Format**: SQLite database containing:
- Match results and statistics
- Team and player information
- League and country data
- Attributes and ratings

**Coverage**: Top 5 European leagues (2008-2016)
- English Premier League
- Spanish La Liga
- Italian Serie A
- German Bundesliga
- French Ligue 1

**Expected File Location**: `data/database.sqlite`

## Pipeline Overview

1. **Data Loading** (`src/data_loader.py`)
   - Connect to SQLite database
   - Extract match and team data
   - Filter top 5 leagues

2. **Feature Engineering** (`src/feature_engineering.py`)
   - Aggregate match-level stats to team-season level
   - Compute attack, possession, defense, and game state features
   - Normalize features for clustering

3. **Clustering** (`src/clustering.py`)
   - K-means clustering with optimal k selection
   - Hierarchical clustering (ward linkage)
   - Cluster evaluation metrics

4. **Visualization** (`src/visualization.py`)
   - PCA/t-SNE for dimensionality reduction
   - Cluster plots and comparisons
   - Feature importance analysis

5. **Reporting** (`notebooks/`)
   - Data exploration notebook
   - Analysis and results notebooks
   - Final report generation

## Repository Structure

```
euro-style-clusters/
├── data/              # SQLite database and processed data
├── src/               # Python source code
│   ├── __init__.py
│   ├── config.py      # Configuration and constants
│   ├── data_loader.py # Data loading functions
│   ├── feature_engineering.py # Feature aggregation
│   ├── clustering.py  # Clustering algorithms
│   ├── visualization.py # Plotting functions
│   └── utils.py       # Utility functions
├── notebooks/         # Jupyter notebooks for analysis
│   └── 01_explore_data.ipynb
└── reports/           # Generated reports and figures
```

## Setup

1. Place the Kaggle SQLite database in `data/database.sqlite`
2. Install dependencies: `pip install -r requirements.txt`
3. Run notebooks in order: `01_explore_data.ipynb`, etc.

## Outputs

- Cluster assignments for each team-season
- Visualizations of playing styles
- Short report summarizing findings (in `reports/`)
