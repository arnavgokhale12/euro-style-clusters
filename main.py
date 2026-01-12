"""
MVP Clustering Analysis - European Soccer Playing Styles
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.data_loader import connect_to_database, load_top5_matches, load_teams
from src.feature_engineering import build_feature_pipeline, get_feature_matrix
from src.config import RANDOM_STATE, REPORTS_DIR, FIGURES_DIR


def find_optimal_k(X, k_range):
    """Test different k values and return scores."""
    results = {}
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = km.fit_predict(X)
        sil = silhouette_score(X, labels)
        results[k] = {"inertia": km.inertia_, "silhouette": sil, "labels": labels}
        print(f"  k={k}: silhouette={sil:.3f}")
    return results


def plot_elbow_curve(results, output_path):
    """Plot elbow curve for k selection."""
    ks = sorted(results.keys())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    ax1.plot(ks, [results[k]["inertia"] for k in ks], "bo-")
    ax1.set_xlabel("k"); ax1.set_ylabel("Inertia"); ax1.set_title("Elbow Curve")
    ax2.plot(ks, [results[k]["silhouette"] for k in ks], "go-")
    ax2.set_xlabel("k"); ax2.set_ylabel("Silhouette"); ax2.set_title("Silhouette Scores")
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    print(f"  Saved: {output_path}")


def plot_pca_clusters(X, labels, output_path):
    """Plot PCA projection colored by cluster."""
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels, cmap="viridis", alpha=0.6)
    plt.colorbar(scatter, label="Cluster")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
    plt.title("Team-Seasons by Playing Style (PCA)")
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    print(f"  Saved: {output_path}")


def plot_feature_comparison(df, labels, output_path):
    """Plot key feature comparisons across clusters."""
    df = df.copy()
    df["cluster"] = labels
    cols = [c for c in ["avg_goals_scored", "avg_possession", "win_rate", "defensive_strength"] if c in df.columns]
    if len(cols) < 2:
        return
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for i, col in enumerate(cols[:4]):
        df.boxplot(column=col, by="cluster", ax=axes.flatten()[i])
        axes.flatten()[i].set_title(col.replace("_", " ").title())
    plt.suptitle("Feature Distribution by Cluster", fontsize=12)
    plt.tight_layout(); plt.savefig(output_path, dpi=150); plt.close()
    print(f"  Saved: {output_path}")


def main():
    print("=" * 60)
    print("European Soccer Style Clustering - MVP Analysis")
    print("=" * 60)

    os.makedirs(REPORTS_DIR, exist_ok=True)
    os.makedirs(FIGURES_DIR, exist_ok=True)

    # 1. Load data
    print("\n[1/5] Loading data...")
    try:
        conn = connect_to_database()
        matches = load_top5_matches(conn)
        teams = load_teams(conn)
        conn.close()
        print(f"  Loaded {len(matches)} matches from top 5 leagues")
    except FileNotFoundError as e:
        print(f"  ERROR: {e}")
        print("  Please download database from Kaggle -> data/database.sqlite")
        return

    # 2. Build features
    print("\n[2/5] Building features...")
    features = build_feature_pipeline(matches, normalize=True, normalize_method="standard")
    print(f"  Created {len(features)} team-season observations")

    # 3. Prepare feature matrix
    print("\n[3/5] Preparing feature matrix...")
    X_df = get_feature_matrix(features)
    X = X_df.values
    print(f"  Feature matrix shape: {X.shape}")

    # 4. Clustering
    print("\n[4/5] Running K-means clustering (k=4-6)...")
    k_results = find_optimal_k(X, range(4, 7))
    best_k = max(k_results, key=lambda k: k_results[k]["silhouette"])
    best_labels = k_results[best_k]["labels"]
    print(f"\n  Best k={best_k} (silhouette={k_results[best_k]['silhouette']:.3f})")

    # 5. Generate outputs
    print("\n[5/5] Generating outputs...")

    # Results with team names
    results = features.copy()
    results["cluster"] = best_labels
    results = results.merge(teams[["team_api_id", "team_long_name"]], on="team_api_id", how="left")
    results.to_csv(os.path.join(REPORTS_DIR, "cluster_assignments.csv"), index=False)

    # Cluster summary
    summary_cols = [c for c in ["avg_goals_scored", "avg_goals_conceded", "avg_possession",
                                "avg_shots", "shot_conversion_rate", "win_rate", "defensive_strength"] if c in features.columns]
    summary = features.copy()
    summary["cluster"] = best_labels
    summary = summary.groupby("cluster")[summary_cols].mean().round(3)
    summary["team_count"] = pd.Series(best_labels).value_counts().sort_index()
    summary.to_csv(os.path.join(REPORTS_DIR, "cluster_summary.csv"))

    print(f"\n{'=' * 60}")
    print(f"CLUSTER SUMMARY (k={best_k})")
    print("=" * 60)
    print(summary.to_string())

    # Visualizations
    plot_elbow_curve(k_results, os.path.join(FIGURES_DIR, "elbow_curve.png"))
    plot_pca_clusters(X, best_labels, os.path.join(FIGURES_DIR, "pca_clusters.png"))
    plot_feature_comparison(features, best_labels, os.path.join(FIGURES_DIR, "feature_comparison.png"))

    print(f"\n{'=' * 60}")
    print(f"Analysis complete! Results in: {REPORTS_DIR}/")
    print("=" * 60)


if __name__ == "__main__":
    main()
