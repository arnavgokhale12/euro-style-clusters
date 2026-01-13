"""
Streamlit Dashboard - European Soccer Style Clustering
Run with: streamlit run app.py
"""

import sqlite3
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.feature_engineering import (
    build_feature_pipeline, get_feature_matrix,
    STYLE_FEATURES, QUALITY_FEATURES
)
from src.config import RANDOM_STATE, TOP_5_LEAGUES
from src.transfer_data import (
    download_transfer_data, load_transfer_data, calculate_team_financials,
    get_team_efficiency_ranking, get_smart_spenders, get_big_spenders,
    format_currency, TRANSFER_DATA_DIR
)

st.set_page_config(page_title="Euro Soccer Clusters", layout="wide")

KAGGLE_DATASET = "hugomathien/soccer"
LOCAL_DB_PATH = "data/database.sqlite"
LEAGUE_ID_TO_NAME = {v: k for k, v in TOP_5_LEAGUES.items()}

# League flag emojis
LEAGUE_FLAGS = {
    "Premier League": "🏴󠁧󠁢󠁥󠁮󠁧󠁿",
    "La Liga": "🇪🇸",
    "Serie A": "🇮🇹",
    "Bundesliga": "🇩🇪",
    "Ligue 1": "🇫🇷"
}


def get_database_path():
    if Path(LOCAL_DB_PATH).exists():
        return LOCAL_DB_PATH
    try:
        import kagglehub
        path = kagglehub.dataset_download(KAGGLE_DATASET)
        db_file = Path(path) / "database.sqlite"
        if db_file.exists():
            return str(db_file)
        raise FileNotFoundError(f"database.sqlite not found in {path}")
    except Exception as e:
        raise e


def check_kaggle_credentials():
    return (Path.home() / ".kaggle" / "kaggle.json").exists()


@st.cache_data
def load_data_cached(_db_path: str):
    conn = sqlite3.connect(_db_path)
    league_ids = list(TOP_5_LEAGUES.values())
    placeholders = ",".join("?" * len(league_ids))
    matches = pd.read_sql_query(
        f"SELECT * FROM Match WHERE league_id IN ({placeholders})",
        conn, params=league_ids
    )
    teams = pd.read_sql_query("SELECT * FROM Team", conn)
    player_attrs = pd.read_sql_query(
        "SELECT player_api_id, date, overall_rating FROM Player_Attributes WHERE overall_rating IS NOT NULL",
        conn
    )
    conn.close()
    return matches, teams, player_attrs


@st.cache_data
def calculate_squad_strength(matches_df: pd.DataFrame, player_attrs: pd.DataFrame) -> pd.DataFrame:
    """Calculate average squad rating per team-season."""
    player_cols = [f"home_player_{i}" for i in range(1, 12)] + [f"away_player_{i}" for i in range(1, 12)]

    # Get latest rating per player (simplified)
    latest_ratings = player_attrs.sort_values("date").groupby("player_api_id")["overall_rating"].last()

    results = []
    for _, match in matches_df.iterrows():
        season = match["season"]

        # Home team
        home_players = [match.get(f"home_player_{i}") for i in range(1, 12)]
        home_players = [int(p) for p in home_players if pd.notna(p)]
        home_ratings = [latest_ratings.get(p, np.nan) for p in home_players]
        home_avg = np.nanmean(home_ratings) if home_ratings else np.nan

        # Away team
        away_players = [match.get(f"away_player_{i}") for i in range(1, 12)]
        away_players = [int(p) for p in away_players if pd.notna(p)]
        away_ratings = [latest_ratings.get(p, np.nan) for p in away_players]
        away_avg = np.nanmean(away_ratings) if away_ratings else np.nan

        results.append({
            "team_api_id": match["home_team_api_id"],
            "season": season,
            "squad_rating": home_avg
        })
        results.append({
            "team_api_id": match["away_team_api_id"],
            "season": season,
            "squad_rating": away_avg
        })

    df = pd.DataFrame(results)
    return df.groupby(["team_api_id", "season"])["squad_rating"].mean().reset_index()


@st.cache_data
def calculate_league_standings(matches_df: pd.DataFrame) -> pd.DataFrame:
    """Calculate final league position per team-season from match results."""
    standings = []

    for (league_id, season), group in matches_df.groupby(["league_id", "season"]):
        team_points = {}
        team_gd = {}

        for _, match in group.iterrows():
            home_id = match["home_team_api_id"]
            away_id = match["away_team_api_id"]
            home_goals = match.get("home_team_goal", 0) or 0
            away_goals = match.get("away_team_goal", 0) or 0

            # Initialize
            if home_id not in team_points:
                team_points[home_id] = 0
                team_gd[home_id] = 0
            if away_id not in team_points:
                team_points[away_id] = 0
                team_gd[away_id] = 0

            # Calculate points
            if home_goals > away_goals:
                team_points[home_id] += 3
            elif away_goals > home_goals:
                team_points[away_id] += 3
            else:
                team_points[home_id] += 1
                team_points[away_id] += 1

            # Goal difference
            team_gd[home_id] += home_goals - away_goals
            team_gd[away_id] += away_goals - home_goals

        # Rank teams
        ranking = sorted(team_points.keys(),
                        key=lambda t: (team_points[t], team_gd[t]),
                        reverse=True)

        for position, team_id in enumerate(ranking, 1):
            standings.append({
                "team_api_id": team_id,
                "season": season,
                "league_id": league_id,
                "final_position": position,
                "points": team_points[team_id],
                "is_champion": position == 1
            })

    return pd.DataFrame(standings)


@st.cache_data
def build_features_cached(matches_json: str):
    matches = pd.read_json(matches_json)
    return build_feature_pipeline(matches, normalize=True, normalize_method="standard")


@st.cache_data
def build_features_raw_cached(matches_json: str):
    matches = pd.read_json(matches_json)
    return build_feature_pipeline(matches, normalize=False)


def run_clustering(X, k):
    km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    sil = silhouette_score(X, labels)
    return labels, sil, km


def generate_cluster_labels(cluster_stats: pd.DataFrame) -> dict:
    """
    Generate meaningful labels for each cluster based on STYLE features.

    Labels describe HOW a team plays, not how well they perform.
    Quality features (win_rate, goals, etc.) are shown as outcomes, not used for labeling.
    """
    labels = {}

    # Use only style features for label generation
    style_cols = [c for c in STYLE_FEATURES if c in cluster_stats.columns]
    style_stats = cluster_stats[style_cols] if style_cols else cluster_stats

    # Normalize to percentiles for comparison
    stats_pct = style_stats.rank(pct=True)

    for cluster in cluster_stats.index:
        pct = stats_pct.loc[cluster]
        raw = style_stats.loc[cluster]

        traits = []

        # Possession style
        if pct.get('avg_possession', 0.5) >= 0.75:
            traits.append(("possession_high", "Possession-Based"))
        elif pct.get('avg_possession', 0.5) <= 0.25:
            traits.append(("possession_low", "Direct"))

        # Attack intent (shots per game)
        if pct.get('avg_shots', 0.5) >= 0.75:
            traits.append(("shots_high", "High-Volume Shooting"))
        elif pct.get('avg_shots', 0.5) <= 0.25:
            traits.append(("shots_low", "Selective Shooting"))

        # Width of play (crosses)
        if pct.get('avg_crosses', 0.5) >= 0.75:
            traits.append(("crosses_high", "Wide Play"))

        # Set pieces (corners)
        if pct.get('avg_corners', 0.5) >= 0.75:
            traits.append(("corners_high", "Set Piece Focus"))

        # Physicality (fouls)
        if pct.get('avg_fouls', 0.5) >= 0.75:
            traits.append(("fouls_high", "Physical"))
        elif pct.get('avg_fouls', 0.5) <= 0.25:
            traits.append(("fouls_low", "Clean"))

        # Generate label based on style traits
        trait_keys = [t[0] for t in traits]

        if "possession_high" in trait_keys and "shots_high" in trait_keys:
            label = "Possession Attackers"
            emoji = "🎮"
        elif "possession_high" in trait_keys:
            label = "Ball Controllers"
            emoji = "⚽"
        elif "possession_low" in trait_keys and "crosses_high" in trait_keys:
            label = "Counter & Cross"
            emoji = "↗️"
        elif "possession_low" in trait_keys:
            label = "Direct Play"
            emoji = "⚡"
        elif "shots_high" in trait_keys and "crosses_high" in trait_keys:
            label = "Wide Attackers"
            emoji = "🎯"
        elif "shots_high" in trait_keys:
            label = "Shot Volume"
            emoji = "🔥"
        elif "fouls_high" in trait_keys and "corners_high" in trait_keys:
            label = "Physical Set Piece"
            emoji = "💪"
        elif "fouls_high" in trait_keys:
            label = "Physical Style"
            emoji = "🛡️"
        elif "fouls_low" in trait_keys:
            label = "Technical Play"
            emoji = "✨"
        elif "crosses_high" in trait_keys:
            label = "Wing Focus"
            emoji = "↔️"
        elif len(traits) == 0:
            label = "Balanced"
            emoji = "⚖️"
        else:
            label = "Mixed Style"
            emoji = "🔄"

        labels[cluster] = {"name": label, "emoji": emoji}

    return labels


def get_top_teams_per_cluster(df: pd.DataFrame, n: int = 5) -> dict:
    """Get top performing teams for each cluster."""
    top_teams = {}

    for cluster in df["cluster"].unique():
        cluster_data = df[df["cluster"] == cluster].copy()

        # Aggregate by team (across seasons)
        team_agg = cluster_data.groupby("team_long_name").agg({
            "win_rate": "mean",
            "avg_goals_scored": "mean",
            "avg_possession": "mean",
            "season": "count",
            "league_name": "first"
        }).rename(columns={"season": "seasons_in_cluster"})

        # Score = win_rate * 0.6 + goals * 0.2 + consistency * 0.2
        team_agg["score"] = (
            team_agg["win_rate"] * 0.6 +
            (team_agg["avg_goals_scored"] / team_agg["avg_goals_scored"].max()) * 0.2 +
            (team_agg["seasons_in_cluster"] / team_agg["seasons_in_cluster"].max()) * 0.2
        )

        top = team_agg.nlargest(n, "score").reset_index()
        top_teams[cluster] = top

    return top_teams


def show_setup_instructions():
    st.error("Database not found!")
    st.markdown("""
    ### Setup Instructions
    1. **Set up Kaggle API:** Go to [kaggle.com/settings](https://www.kaggle.com/settings) → API → Create New Token
    2. **Configure credentials:**
       ```bash
       mkdir -p ~/.kaggle && mv ~/Downloads/kaggle.json ~/.kaggle/ && chmod 600 ~/.kaggle/kaggle.json
       ```
    3. **Refresh this page**
    """)


@st.cache_data
def load_transfer_data_cached():
    """Load and process transfer market data."""
    try:
        if not TRANSFER_DATA_DIR.exists():
            download_transfer_data()
        data = load_transfer_data()
        return data
    except Exception as e:
        return None


@st.cache_data
def calculate_financials_cached(season_start: int, season_end: int):
    """Calculate team financials with caching."""
    try:
        data = load_transfer_data_cached()
        if data is None:
            return None

        financials = calculate_team_financials(
            transfers_df=data['transfers'],
            valuations_df=data['valuations'],
            players_df=data['players'],
            clubs_df=data['clubs'],
            season_start=season_start,
            season_end=season_end
        )
        return financials
    except Exception as e:
        st.error(f"Error calculating financials: {e}")
        return None


def main():
    st.title("⚽ European Soccer Analytics")
    st.markdown("*Style clustering (2008-2016) + MoneyBall analysis (2020-2026)*")

    # Sidebar
    st.sidebar.header("Data")
    if st.sidebar.button("🔄 Refresh Data"):
        st.cache_data.clear()
        st.rerun()

    # Load data
    try:
        if Path(LOCAL_DB_PATH).exists():
            db_path = LOCAL_DB_PATH
        else:
            if not check_kaggle_credentials():
                show_setup_instructions()
                return
            with st.spinner("Downloading database from Kaggle..."):
                db_path = get_database_path()

        with st.spinner("Loading data..."):
            matches, teams, player_attrs = load_data_cached(db_path)
            matches_json = matches.to_json()
            features_norm = build_features_cached(matches_json)
            features_raw = build_features_raw_cached(matches_json)

        with st.spinner("Calculating squad ratings & standings..."):
            squad_ratings = calculate_squad_strength(matches, player_attrs)
            standings = calculate_league_standings(matches)
    except Exception as e:
        show_setup_instructions()
        return

    # Merge team names
    features_norm = features_norm.merge(
        teams[["team_api_id", "team_long_name"]],
        on="team_api_id", how="left"
    )
    features_raw = features_raw.merge(
        teams[["team_api_id", "team_long_name"]],
        on="team_api_id", how="left"
    )

    # Add league names from match data
    match_leagues = matches[["home_team_api_id", "league_id"]].drop_duplicates()
    match_leagues = match_leagues.rename(columns={"home_team_api_id": "team_api_id"})

    features_norm = features_norm.merge(match_leagues, on="team_api_id", how="left")
    features_raw = features_raw.merge(match_leagues, on="team_api_id", how="left")

    features_norm["league_name"] = features_norm["league_id"].map(LEAGUE_ID_TO_NAME)
    features_raw["league_name"] = features_raw["league_id"].map(LEAGUE_ID_TO_NAME)

    # Merge squad ratings and standings
    features_raw = features_raw.merge(squad_ratings, on=["team_api_id", "season"], how="left")
    features_raw = features_raw.merge(
        standings[["team_api_id", "season", "final_position", "is_champion"]],
        on=["team_api_id", "season"], how="left"
    )

    # Sidebar controls
    st.sidebar.header("Clustering")
    k = st.sidebar.slider("Number of Styles", min_value=3, max_value=8, value=5)

    st.sidebar.header("Filters")
    league_names = list(TOP_5_LEAGUES.keys())
    selected_leagues = st.sidebar.multiselect("Leagues", league_names, default=league_names)
    all_seasons = sorted(features_norm["season"].unique())
    selected_seasons = st.sidebar.multiselect("Seasons", all_seasons, default=all_seasons)

    # Apply filters
    mask = (features_norm["season"].isin(selected_seasons)) & (features_norm["league_name"].isin(selected_leagues))
    filtered_norm = features_norm[mask].copy()
    filtered_raw = features_raw[mask].copy()

    if len(filtered_norm) < k:
        st.warning(f"Not enough data ({len(filtered_norm)} points) for {k} clusters.")
        return

    # Run clustering on STYLE features only
    X_df = get_feature_matrix(filtered_norm)  # Now defaults to STYLE_FEATURES
    X = X_df.values
    labels, sil_score, km_model = run_clustering(X, k)
    filtered_norm["cluster"] = labels
    filtered_raw["cluster"] = labels

    # Get style and quality columns that exist in our data
    style_cols = [c for c in STYLE_FEATURES if c in filtered_raw.columns]
    quality_cols = [c for c in QUALITY_FEATURES if c in filtered_raw.columns]

    # Extended stats including squad rating and position
    all_analysis_cols = style_cols + quality_cols
    extended_cols = all_analysis_cols + [c for c in ["squad_rating", "final_position"] if c in filtered_raw.columns]

    cluster_stats = filtered_raw.groupby("cluster")[extended_cols].mean()

    # Generate labels based on STYLE features only
    cluster_labels = generate_cluster_labels(cluster_stats)
    top_teams = get_top_teams_per_cluster(filtered_raw, n=5)

    # Create label mapping
    def get_label(c):
        return f"{cluster_labels[c]['emoji']} {cluster_labels[c]['name']}"

    filtered_raw["style"] = filtered_raw["cluster"].map(get_label)
    filtered_norm["style"] = filtered_norm["cluster"].map(get_label)

    # Stats
    st.sidebar.markdown("---")
    st.sidebar.caption(f"**{len(matches):,}** matches • **{len(filtered_raw):,}** team-seasons")

    # === CLUSTER OVERVIEW ===
    st.subheader(f"🎨 {k} Playing Styles Identified")
    st.caption(f"Clustered on style features (possession, shots, crosses, corners, fouls) • Silhouette: {sil_score:.3f}")

    # Info box explaining the separation
    st.info("""
    **Style vs Quality:** Clusters are formed based on *how* teams play (style features), not *how well* they play.
    Quality metrics (win rate, goals) are shown as *outcomes* for each style.
    """)

    # Build comparison table with separated style and quality columns
    comparison_data = []
    for cluster in sorted(cluster_labels.keys()):
        label = cluster_labels[cluster]
        stats = cluster_stats.loc[cluster]
        top = top_teams[cluster]
        cluster_data = filtered_raw[filtered_raw['cluster'] == cluster]
        top_3 = ", ".join([f"{row['team_long_name']}" for _, row in top.head(3).iterrows()])
        championships = cluster_data['is_champion'].sum() if 'is_champion' in cluster_data.columns else 0

        row_data = {
            "Style": f"{label['emoji']} {label['name']}",
            "Teams": len(cluster_data),
            # Style features (used for clustering)
            "Poss.": f"{stats.get('avg_possession', 0):.0f}%",
            "Shots": f"{stats.get('avg_shots', 0):.1f}",
            "Crosses": f"{stats.get('avg_crosses', 0):.1f}",
            "Corners": f"{stats.get('avg_corners', 0):.1f}",
            "Fouls": f"{stats.get('avg_fouls', 0):.1f}",
            # Quality features (outcomes)
            "Win %": f"{stats.get('win_rate', 0)*100:.0f}%",
            "Goals": f"{stats.get('avg_goals_scored', 0):.2f}",
            "Conc.": f"{stats.get('avg_goals_conceded', 0):.2f}",
            "🏆": int(championships),
        }
        comparison_data.append(row_data)

    comparison_df = pd.DataFrame(comparison_data)

    # Display as styled table with style/quality grouping
    st.markdown("##### Style Features (clustering inputs) | Quality Outcomes")
    st.dataframe(
        comparison_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Style": st.column_config.TextColumn("Style", width="medium"),
            "Teams": st.column_config.NumberColumn("N", width="small", help="Number of team-seasons"),
            # Style features
            "Poss.": st.column_config.TextColumn("Poss", width="small", help="Avg possession %"),
            "Shots": st.column_config.TextColumn("Shots", width="small", help="Shots per game"),
            "Crosses": st.column_config.TextColumn("Cross", width="small", help="Crosses per game"),
            "Corners": st.column_config.TextColumn("Corn", width="small", help="Corners per game"),
            "Fouls": st.column_config.TextColumn("Fouls", width="small", help="Fouls per game"),
            # Quality outcomes
            "Win %": st.column_config.TextColumn("Win%", width="small", help="Win rate (outcome)"),
            "Goals": st.column_config.TextColumn("GF", width="small", help="Goals scored per game"),
            "Conc.": st.column_config.TextColumn("GA", width="small", help="Goals conceded per game"),
            "🏆": st.column_config.NumberColumn("🏆", width="small", help="Championships won"),
        }
    )

    # Add insight callouts about style-outcome relationships
    if comparison_df is not None and len(comparison_df) > 0:
        most_titles_idx = comparison_df["🏆"].idxmax()
        best_win_idx = comparison_df["Win %"].apply(lambda x: float(x.replace("%", ""))).idxmax()

        insights = []
        if comparison_df.loc[most_titles_idx, "🏆"] > 0:
            insights.append(f"🏆 **{comparison_df.loc[most_titles_idx, 'Style']}** won {comparison_df.loc[most_titles_idx, '🏆']} championships")
        insights.append(f"📈 **{comparison_df.loc[best_win_idx, 'Style']}** has highest win rate ({comparison_df.loc[best_win_idx, 'Win %']})")

        if insights:
            st.success(" • ".join(insights))

    # Expandable details for each cluster
    st.markdown("---")
    st.markdown("##### 📋 Cluster Details")

    for cluster in sorted(cluster_labels.keys()):
        label = cluster_labels[cluster]
        stats = cluster_stats.loc[cluster]
        top = top_teams[cluster]
        cluster_data = filtered_raw[filtered_raw['cluster'] == cluster]
        count = len(cluster_data)
        championships = cluster_data['is_champion'].sum() if 'is_champion' in cluster_data.columns else 0

        with st.expander(f"{label['emoji']} **{label['name']}** — {count} team-seasons • {int(championships)} 🏆"):
            col1, col2, col3 = st.columns([1, 1, 1.5])

            with col1:
                st.markdown("**🎨 Style Profile:**")
                style_metrics = [
                    ("Possession", f"{stats.get('avg_possession', 0):.0f}%"),
                    ("Shots/Game", f"{stats.get('avg_shots', 0):.1f}"),
                    ("Crosses/Game", f"{stats.get('avg_crosses', 0):.1f}"),
                    ("Corners/Game", f"{stats.get('avg_corners', 0):.1f}"),
                    ("Fouls/Game", f"{stats.get('avg_fouls', 0):.1f}"),
                ]
                style_table = pd.DataFrame(style_metrics, columns=["Feature", "Value"])
                st.dataframe(style_table, hide_index=True, use_container_width=True)

            with col2:
                st.markdown("**📊 Quality Outcomes:**")
                quality_metrics = [
                    ("Win Rate", f"{stats.get('win_rate', 0)*100:.0f}%"),
                    ("Goals/Game", f"{stats.get('avg_goals_scored', 0):.2f}"),
                    ("Conceded/Game", f"{stats.get('avg_goals_conceded', 0):.2f}"),
                    ("Goal Diff", f"{stats.get('goal_difference', 0):.1f}"),
                    ("Championships", f"{int(championships)}"),
                ]
                quality_table = pd.DataFrame(quality_metrics, columns=["Metric", "Value"])
                st.dataframe(quality_table, hide_index=True, use_container_width=True)

                # Style effectiveness insight
                win_rate = stats.get('win_rate', 0)
                if win_rate >= 0.6:
                    st.success("⭐ Highly effective style")
                elif win_rate >= 0.4:
                    st.info("📊 Average effectiveness")
                else:
                    st.warning("📉 Lower success rate")

            with col3:
                st.markdown("**⭐ Top Teams in this Style:**")
                top_display = top.head(5).copy()
                top_display["League"] = top_display["league_name"].map(lambda x: f"{LEAGUE_FLAGS.get(x, '')} {x}")
                top_display["Win Rate"] = (top_display["win_rate"] * 100).round(0).astype(int).astype(str) + "%"
                top_display["Seasons"] = top_display["seasons_in_cluster"].astype(int)
                top_display = top_display.rename(columns={"team_long_name": "Team"})
                st.dataframe(
                    top_display[["Team", "League", "Win Rate", "Seasons"]],
                    hide_index=True,
                    use_container_width=True
                )

                # Show champions from this cluster
                if 'is_champion' in cluster_data.columns:
                    champions = cluster_data[cluster_data['is_champion'] == True][['team_long_name', 'season', 'league_name']].copy()
                    if len(champions) > 0:
                        st.markdown("**🏆 League Champions:**")
                        champions["League"] = champions["league_name"].map(lambda x: LEAGUE_FLAGS.get(x, '') + " " + str(x))
                        champions = champions.rename(columns={"team_long_name": "Team", "season": "Season"})
                        st.dataframe(champions[["Team", "Season", "League"]].head(5), hide_index=True, use_container_width=True)

    # === TABS ===
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📊 Overview", "🔍 Team Search", "📈 Timeline", "🌍 Leagues", "💰 MoneyBall", "💾 Export"
    ])

    # === TAB 1: OVERVIEW ===
    with tab1:
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Team Map by Playing Style")
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X)

            plot_df = pd.DataFrame({
                "PC1": X_pca[:, 0], "PC2": X_pca[:, 1],
                "style": filtered_raw["style"].values,
                "team": filtered_raw["team_long_name"].values,
                "season": filtered_raw["season"].values,
                "goals": filtered_raw["avg_goals_scored"].values,
                "possession": filtered_raw["avg_possession"].values
            })

            fig = px.scatter(plot_df, x="PC1", y="PC2", color="style",
                           hover_data=["team", "season", "goals", "possession"],
                           color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=450, legend=dict(orientation="h", yanchor="bottom", y=-0.3))
            fig.update_traces(marker=dict(size=8))
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            st.subheader("Style Feature Radar")
            styles_to_compare = st.multiselect(
                "Compare styles:",
                options=[get_label(c) for c in sorted(cluster_labels.keys())],
                default=[get_label(c) for c in sorted(cluster_labels.keys())[:3]]
            )

            if styles_to_compare:
                # Use STYLE features for radar chart
                summary = filtered_raw.groupby("style")[style_cols].mean()
                summary_norm = (summary - summary.min()) / (summary.max() - summary.min() + 1e-10)

                # Better labels for radar
                radar_labels = {
                    "avg_possession": "Possession",
                    "avg_shots": "Shots",
                    "avg_crosses": "Crosses",
                    "avg_corners": "Corners",
                    "avg_fouls": "Fouls"
                }
                display_cols = [radar_labels.get(c, c) for c in style_cols]

                fig2 = go.Figure()
                colors = px.colors.qualitative.Set2
                for i, style in enumerate(styles_to_compare):
                    if style in summary_norm.index:
                        vals = summary_norm.loc[style].values.tolist()
                        vals.append(vals[0])
                        fig2.add_trace(go.Scatterpolar(
                            r=vals, theta=display_cols + [display_cols[0]],
                            name=style, fill="toself", opacity=0.6,
                            line=dict(color=colors[i % len(colors)])
                        ))
                fig2.update_layout(
                    polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                    height=400,
                    title="Style Features (how teams play)"
                )
                st.plotly_chart(fig2, use_container_width=True)

        # Separate style and quality summaries
        col1, col2 = st.columns(2)

        with col1:
            st.subheader("🎨 Style Features by Cluster")
            style_summary = filtered_raw.groupby("style")[style_cols].mean().round(2)
            style_summary.columns = [c.replace("avg_", "").title() for c in style_summary.columns]
            st.dataframe(style_summary, use_container_width=True)

        with col2:
            st.subheader("📊 Quality Outcomes by Cluster")
            quality_summary = filtered_raw.groupby("style")[quality_cols].mean().round(3)
            quality_summary.columns = [c.replace("avg_", "").replace("_", " ").title() for c in quality_summary.columns]
            quality_summary = quality_summary.sort_values("Win Rate", ascending=False)
            st.dataframe(quality_summary, use_container_width=True)

    # === TAB 2: TEAM SEARCH ===
    with tab2:
        st.subheader("🔍 Find Your Team's Playing Style")

        all_teams = sorted(filtered_raw["team_long_name"].dropna().unique())
        selected_team = st.selectbox("Select a team:", [""] + list(all_teams))

        if selected_team:
            team_data = filtered_raw[filtered_raw["team_long_name"] == selected_team]

            if not team_data.empty:
                # Team header
                primary_style = team_data["style"].mode().values[0]
                primary_league = team_data["league_name"].mode().values[0]
                flag = LEAGUE_FLAGS.get(primary_league, '')

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 20px; border-radius: 15px; margin: 10px 0;">
                    <h2 style="color: white; margin: 0;">{flag} {selected_team}</h2>
                    <p style="color: #ddd; font-size: 1.2em;">Primary Style: <strong>{primary_style}</strong></p>
                    <p style="color: #ccc;">{len(team_data)} seasons in dataset</p>
                </div>
                """, unsafe_allow_html=True)

                col1, col2 = st.columns([1, 2])

                with col1:
                    st.markdown("**Style by Season:**")
                    for _, row in team_data.sort_values("season").iterrows():
                        st.markdown(f"- {row['season']}: {row['style']}")

                with col2:
                    # Highlight on PCA
                    fig = px.scatter(plot_df, x="PC1", y="PC2", color="style",
                                   opacity=0.3, color_discrete_sequence=px.colors.qualitative.Set2)
                    team_pca = plot_df[plot_df["team"] == selected_team]
                    fig.add_trace(go.Scatter(
                        x=team_pca["PC1"], y=team_pca["PC2"],
                        mode="markers+text",
                        marker=dict(size=18, color="red", symbol="star", line=dict(width=2, color="white")),
                        text=team_pca["season"], textposition="top center",
                        name=selected_team, textfont=dict(size=10)
                    ))
                    fig.update_layout(height=400, title=f"{selected_team} Across Seasons", showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)

                # Similar teams
                st.markdown("**🤝 Similar Teams (same primary style):**")
                same_style = filtered_raw[filtered_raw["style"] == primary_style]
                similar = same_style.groupby("team_long_name").agg({
                    "win_rate": "mean", "league_name": "first"
                }).nlargest(6, "win_rate")
                similar = similar[similar.index != selected_team].head(5)

                similar_cols = st.columns(5)
                for i, (team, row) in enumerate(similar.iterrows()):
                    with similar_cols[i]:
                        flag = LEAGUE_FLAGS.get(row['league_name'], '')
                        st.markdown(f"**{flag} {team}**")
                        st.caption(f"Win: {row['win_rate']*100:.0f}%")

    # === TAB 3: TIMELINE ===
    with tab3:
        st.subheader("📈 Team Style Evolution")

        timeline_team = st.selectbox("Select team:", [""] + list(all_teams), key="timeline")

        if timeline_team:
            team_history = filtered_raw[filtered_raw["team_long_name"] == timeline_team].sort_values("season")

            if len(team_history) > 1:
                fig = make_subplots(rows=2, cols=2,
                                   subplot_titles=["Goals Scored", "Possession %", "Win Rate", "Goals Conceded"])

                metrics = ["avg_goals_scored", "avg_possession", "win_rate", "avg_goals_conceded"]
                colors = ["#2ecc71", "#3498db", "#9b59b6", "#e74c3c"]

                for i, (col, color) in enumerate(zip(metrics, colors)):
                    if col in team_history.columns:
                        row = i // 2 + 1
                        col_num = i % 2 + 1
                        fig.add_trace(
                            go.Scatter(x=team_history["season"], y=team_history[col],
                                     mode="lines+markers", name=col,
                                     line=dict(color=color, width=3),
                                     marker=dict(size=10)),
                            row=row, col=col_num
                        )

                fig.update_layout(height=500, showlegend=False,
                                title=f"{timeline_team} - Performance Over Time")
                st.plotly_chart(fig, use_container_width=True)

                # Style changes
                st.markdown("**Style Journey:**")
                prev_style = None
                for _, row in team_history.iterrows():
                    if row["style"] != prev_style:
                        st.markdown(f"🔄 **{row['season']}**: Changed to {row['style']}")
                    else:
                        st.markdown(f"➡️ {row['season']}: {row['style']}")
                    prev_style = row["style"]

    # === TAB 4: LEAGUE DISTRIBUTION ===
    with tab4:
        st.subheader("🌍 Playing Styles by League")

        col1, col2 = st.columns(2)

        with col1:
            league_style = filtered_raw.groupby(["style", "league_name"]).size().reset_index(name="count")
            fig = px.bar(league_style, x="style", y="count", color="league_name",
                        title="Style Distribution by League",
                        color_discrete_map={
                            "Premier League": "#3d195b",
                            "La Liga": "#ee8707",
                            "Serie A": "#008fd7",
                            "Bundesliga": "#d20515",
                            "Ligue 1": "#dae025"
                        })
            fig.update_layout(height=400, xaxis_tickangle=-45)
            st.plotly_chart(fig, use_container_width=True)

        with col2:
            selected_style = st.selectbox("Select style:", [get_label(c) for c in sorted(cluster_labels.keys())])
            style_data = filtered_raw[filtered_raw["style"] == selected_style]

            league_counts = style_data["league_name"].value_counts()
            fig = px.pie(values=league_counts.values, names=league_counts.index,
                        title=f"{selected_style} - League Breakdown",
                        color=league_counts.index,
                        color_discrete_map={
                            "Premier League": "#3d195b",
                            "La Liga": "#ee8707",
                            "Serie A": "#008fd7",
                            "Bundesliga": "#d20515",
                            "Ligue 1": "#dae025"
                        })
            fig.update_layout(height=350)
            st.plotly_chart(fig, use_container_width=True)

        # Heatmap
        st.subheader("Style Preference by League")
        pivot = filtered_raw.pivot_table(index="league_name", columns="style", aggfunc="size", fill_value=0)
        pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100

        fig = px.imshow(pivot_pct, text_auto=".0f", aspect="auto",
                       labels=dict(x="Playing Style", y="League", color="% of Teams"),
                       color_continuous_scale="Viridis")
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

    # === TAB 5: MONEYBALL ===
    with tab5:
        st.subheader("💰 MoneyBall Analysis")
        st.markdown("*Which teams are smart with money vs just throwing cash around?*")

        # Year range filter for transfer data
        st.markdown("---")
        col1, col2 = st.columns(2)
        with col1:
            mb_year_start = st.selectbox("From Year", options=list(range(2015, 2027)), index=5, key="mb_start")
        with col2:
            mb_year_end = st.selectbox("To Year", options=list(range(2015, 2027)), index=11, key="mb_end")

        if mb_year_start > mb_year_end:
            st.warning("Start year must be before end year")
        else:
            # Load transfer data
            with st.spinner("Loading transfer market data..."):
                financials = calculate_financials_cached(mb_year_start, mb_year_end)

            if financials is None or len(financials) == 0:
                st.warning("Transfer data not available. Run `python setup_transfer_data.py` to download it.")
                st.code("python setup_transfer_data.py", language="bash")
            else:
                # Summary metrics
                st.markdown("### Key Metrics")
                m1, m2, m3, m4 = st.columns(4)

                total_value = financials['squad_value'].sum()
                total_spent = financials['total_spent'].sum()
                avg_efficiency = financials[financials['value_efficiency'].notna()]['value_efficiency'].mean()

                with m1:
                    st.metric("Total Squad Values", format_currency(total_value))
                with m2:
                    st.metric("Total Spending", format_currency(total_spent))
                with m3:
                    st.metric("Avg Efficiency", f"{avg_efficiency:.2f}x" if not pd.isna(avg_efficiency) else "-")
                with m4:
                    st.metric("Teams Analyzed", len(financials))

                st.markdown("---")

                # Two column layout for main analysis
                col1, col2 = st.columns(2)

                with col1:
                    st.markdown("### 🎯 Smart Spenders (Best Value/Cost)")
                    st.caption("Teams getting the most value for their money")

                    smart = get_smart_spenders(financials, top_n=15)
                    if len(smart) > 0:
                        display_smart = smart[['club_name', 'squad_value', 'total_spent', 'value_vs_cost', 'value_efficiency']].copy()
                        display_smart['squad_value'] = display_smart['squad_value'].apply(lambda x: format_currency(x))
                        display_smart['total_spent'] = display_smart['total_spent'].apply(lambda x: format_currency(x))
                        display_smart['value_vs_cost'] = display_smart['value_vs_cost'].apply(lambda x: format_currency(x))
                        display_smart['value_efficiency'] = display_smart['value_efficiency'].apply(lambda x: f"{x:.2f}x" if pd.notna(x) else "-")
                        display_smart.columns = ['Club', 'Squad Value', 'Spent', 'Value - Cost', 'Efficiency']
                        st.dataframe(display_smart, hide_index=True, use_container_width=True)

                        # Value vs Cost scatter for smart spenders
                        fig = px.scatter(
                            smart,
                            x='total_spent',
                            y='squad_value',
                            text='club_name',
                            title='Squad Value vs Transfer Spending',
                            labels={'total_spent': 'Total Spent (€)', 'squad_value': 'Squad Value (€)'}
                        )
                        fig.add_trace(go.Scatter(
                            x=[0, smart['total_spent'].max()],
                            y=[0, smart['total_spent'].max()],
                            mode='lines',
                            name='Break Even (1:1)',
                            line=dict(dash='dash', color='gray')
                        ))
                        fig.update_traces(textposition='top center', marker=dict(size=12, color='#2ecc71'))
                        fig.update_layout(height=400, showlegend=True)
                        st.plotly_chart(fig, use_container_width=True)

                with col2:
                    st.markdown("### 💸 Biggest Spenders")
                    st.caption("Teams that have spent the most on transfers")

                    big = get_big_spenders(financials, top_n=15)
                    if len(big) > 0:
                        display_big = big[['club_name', 'total_spent', 'total_received', 'net_spend', 'squad_value']].copy()
                        display_big['total_spent'] = display_big['total_spent'].apply(lambda x: format_currency(x))
                        display_big['total_received'] = display_big['total_received'].apply(lambda x: format_currency(x))
                        display_big['net_spend'] = display_big['net_spend'].apply(lambda x: format_currency(x))
                        display_big['squad_value'] = display_big['squad_value'].apply(lambda x: format_currency(x))
                        display_big.columns = ['Club', 'Spent', 'Received', 'Net Spend', 'Squad Value']
                        st.dataframe(display_big, hide_index=True, use_container_width=True)

                        # Net spend bar chart
                        fig2 = px.bar(
                            big.head(10),
                            x='club_name',
                            y='net_spend',
                            title='Net Transfer Spend (Top 10)',
                            color='net_spend',
                            color_continuous_scale=['#27ae60', '#f1c40f', '#e74c3c'],
                            labels={'club_name': 'Club', 'net_spend': 'Net Spend (€)'}
                        )
                        fig2.update_layout(height=400, xaxis_tickangle=-45)
                        st.plotly_chart(fig2, use_container_width=True)

                st.markdown("---")

                # Efficiency ranking
                st.markdown("### 📊 Value Efficiency Rankings")
                st.caption("Efficiency = Squad Value / Total Spent. Higher = better ROI on transfers.")

                efficient = get_team_efficiency_ranking(financials)
                if len(efficient) > 0:
                    # Top 10 vs Bottom 10
                    col1, col2 = st.columns(2)

                    with col1:
                        st.markdown("**🏆 Most Efficient (Top 10)**")
                        top10 = efficient.head(10)[['club_name', 'value_efficiency', 'efficiency_category']].copy()
                        top10['value_efficiency'] = top10['value_efficiency'].apply(lambda x: f"{x:.2f}x")
                        top10.columns = ['Club', 'Efficiency', 'Category']
                        st.dataframe(top10, hide_index=True, use_container_width=True)

                    with col2:
                        st.markdown("**📉 Least Efficient (Bottom 10)**")
                        bottom10 = efficient.tail(10)[['club_name', 'value_efficiency', 'efficiency_category']].copy()
                        bottom10 = bottom10.sort_values('value_efficiency', ascending=True)
                        bottom10['value_efficiency'] = bottom10['value_efficiency'].apply(lambda x: f"{x:.2f}x")
                        bottom10.columns = ['Club', 'Efficiency', 'Category']
                        st.dataframe(bottom10, hide_index=True, use_container_width=True)

                    # Efficiency distribution
                    fig3 = px.histogram(
                        efficient,
                        x='value_efficiency',
                        nbins=30,
                        title='Distribution of Value Efficiency Across Clubs',
                        labels={'value_efficiency': 'Value Efficiency (Squad Value / Spent)'}
                    )
                    fig3.add_vline(x=1.0, line_dash="dash", line_color="red",
                                  annotation_text="Break Even (1.0)")
                    fig3.update_layout(height=350)
                    st.plotly_chart(fig3, use_container_width=True)

                # Average age analysis if available
                if 'avg_player_age' in financials.columns and financials['avg_player_age'].notna().any():
                    st.markdown("---")
                    st.markdown("### 👶 Squad Age Analysis")

                    age_data = financials[financials['avg_player_age'].notna()].copy()
                    if len(age_data) > 0:
                        col1, col2 = st.columns(2)

                        with col1:
                            st.markdown("**Youngest Squads**")
                            youngest = age_data.nsmallest(10, 'avg_player_age')[['club_name', 'avg_player_age', 'squad_value']].copy()
                            youngest['avg_player_age'] = youngest['avg_player_age'].apply(lambda x: f"{x:.1f} years")
                            youngest['squad_value'] = youngest['squad_value'].apply(lambda x: format_currency(x))
                            youngest.columns = ['Club', 'Avg Age', 'Squad Value']
                            st.dataframe(youngest, hide_index=True, use_container_width=True)

                        with col2:
                            st.markdown("**Oldest Squads**")
                            oldest = age_data.nlargest(10, 'avg_player_age')[['club_name', 'avg_player_age', 'squad_value']].copy()
                            oldest['avg_player_age'] = oldest['avg_player_age'].apply(lambda x: f"{x:.1f} years")
                            oldest['squad_value'] = oldest['squad_value'].apply(lambda x: format_currency(x))
                            oldest.columns = ['Club', 'Avg Age', 'Squad Value']
                            st.dataframe(oldest, hide_index=True, use_container_width=True)

                # Export financial data
                st.markdown("---")
                st.download_button(
                    "📥 Download Financial Data (CSV)",
                    financials.to_csv(index=False),
                    "team_financials.csv",
                    "text/csv"
                )

    # === TAB 6: EXPORT ===
    with tab6:
        st.subheader("💾 Export Data")

        # Include both style and quality features in export
        export_cols = ["team_long_name", "season", "style", "league_name"] + style_cols + quality_cols
        export_cols = [c for c in export_cols if c in filtered_raw.columns]
        export_df = filtered_raw[export_cols].copy()
        export_df = export_df.rename(columns={
            "team_long_name": "Team",
            "season": "Season",
            "style": "Playing Style",
            "league_name": "League"
        })

        st.markdown(f"**{len(export_df)} rows ready for export**")
        st.info("Export includes style features (clustering inputs) and quality features (outcomes)")
        st.dataframe(export_df.head(20), use_container_width=True)

        col1, col2, col3 = st.columns(3)
        with col1:
            st.download_button("📥 Full Data (CSV)",
                             export_df.to_csv(index=False),
                             "playing_styles.csv", "text/csv")
        with col2:
            # Style summary
            style_export = filtered_raw.groupby("style")[style_cols].mean().round(3)
            style_export["team_count"] = filtered_raw.groupby("style").size()
            st.download_button("📥 Style Summary (CSV)",
                             style_export.to_csv(),
                             "style_features.csv", "text/csv")
        with col3:
            # Quality summary
            quality_export = filtered_raw.groupby("style")[quality_cols].mean().round(3)
            quality_export["team_count"] = filtered_raw.groupby("style").size()
            st.download_button("📥 Quality Summary (CSV)",
                             quality_export.to_csv(),
                             "quality_outcomes.csv", "text/csv")


if __name__ == "__main__":
    main()
