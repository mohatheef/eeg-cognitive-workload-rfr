import os
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score


def main():
    parser = argparse.ArgumentParser(
        description="Generate Paper-Equivalent Visualizations for EEG Cognitive Workload"
    )
    parser.add_argument(
        "--feature_file",
        type=str,
        default="data/features/epoch_features.csv",
        help="Path to the generated epoch features CSV"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="data/results_plots",
        help="Directory to save the generated figures"
    )
    args = parser.parse_args()

    FEATURE_FILE = args.feature_file
    OUTPUT_DIR = args.output_dir

    if not os.path.exists(FEATURE_FILE):
        print(f"Error: Feature file '{FEATURE_FILE}' does not exist.")
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load data
    df = pd.read_csv(FEATURE_FILE)
    df = df.dropna().reset_index(drop=True)

    # Validate required columns
    required_cols = {"cortical_area", "mean"}
    if not required_cols.issubset(df.columns):
        raise ValueError(
            f"Feature file must contain columns: {required_cols}"
        )

    # Extract feature list
    drop_cols = ["cortical_area", "channel", "epoch", "source_file", "mean"]
    feature_cols = [col for col in df.columns if col not in drop_cols]

    sns.set_theme(style="whitegrid")

    # ==========================================
    # Fig. 9: Feature Importance Across Regions
    # ==========================================
    print("Generating Figure 9: Feature Importances...")
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()

    importance_results = {}

    for i, area in enumerate(df["cortical_area"].unique()):
        area_df = df[df["cortical_area"] == area]
        X = area_df[feature_cols]
        y = area_df["mean"]

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X, y)

        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        # Store for reference
        importance_results[area] = pd.DataFrame({
            "Feature": [feature_cols[idx] for idx in indices],
            "Importance": importances[indices]
        })

        y_labels = [feature_cols[idx] for idx in indices]
        sns.barplot(
            ax=axes[i],
            x=importances[indices],
            y=y_labels,
            hue=y_labels,
            legend=False,
            palette="viridis"
        )
        axes[i].set_title(f"Feature Importance for {area} Lobe", fontsize=14, fontweight="bold")
        axes[i].set_xlabel("Importance Score", fontsize=12)
        axes[i].set_ylabel("Features", fontsize=12)

    plt.tight_layout()
    fig9_path = os.path.join(OUTPUT_DIR, "figure_9_feature_importance.png")
    plt.savefig(fig9_path, dpi=300)
    plt.close()
    print(f"Saved Figure 9 to: {fig9_path}")

    # ==========================================
    # Fig. 10: Correlation Matrix Heatmap
    # ==========================================
    print("Generating Figure 10: Correlation Matrix...")
    # Compute correlation across the numerical features (excluding non-features)
    corr_df = df[feature_cols + ["mean"]].rename(columns={"mean": "mean_activation"})
    corr_matrix = corr_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        square=True,
        cbar_kws={"shrink": .8},
        linewidths=0.5
    )
    plt.title("Correlation Matrix of EEG Features", fontsize=16, fontweight="bold", pad=20)
    plt.tight_layout()
    fig10_path = os.path.join(OUTPUT_DIR, "figure_10_correlation_matrix.png")
    plt.savefig(fig10_path, dpi=300)
    plt.close()
    print(f"Saved Figure 10 to: {fig10_path}")

    # ==========================================
    # Fig. 11: Regression Performance (Predicted vs. Actual)
    # ==========================================
    print("Generating Figure 11: Predicted vs. Actual workload...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    for i, area in enumerate(df["cortical_area"].unique()):
        area_df = df[df["cortical_area"] == area]
        X = area_df[feature_cols]
        y = area_df["mean"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Plot scatter
        axes[i].scatter(y_test, y_pred, alpha=0.6, edgecolors="k", color="mediumseagreen", label="Predictions")
        
        # Ideal fit line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        axes[i].plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal Fit")

        axes[i].set_title(f"{area} Lobe Performance (R² = {r2_score(y_test, y_pred):.4f})", fontsize=14, fontweight="bold")
        axes[i].set_xlabel("True Workload Proxy Value", fontsize=12)
        axes[i].set_ylabel("Predicted Value", fontsize=12)
        axes[i].legend(loc="upper left")

    plt.tight_layout()
    fig11_path = os.path.join(OUTPUT_DIR, "figure_11_predicted_vs_actual.png")
    plt.savefig(fig11_path, dpi=300)
    plt.close()
    print(f"Saved Figure 11 to: {fig11_path}")

    # ==========================================
    # Fig. 12: Cortical Area Performance Comparison
    # ==========================================
    print("Generating Figure 12: Cortical Area Performance Comparison...")
    # Gather performance metrics for RFR across all regions
    perf_data = []

    for area in df["cortical_area"].unique():
        area_df = df[df["cortical_area"] == area]
        X = area_df[feature_cols]
        y = area_df["mean"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        mae = np.mean(np.abs(y_test - y_pred))
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))

        perf_data.append({
            "Cortical Area": area,
            "R²": r2,
            "MAE": mae,
            "RMSE": rmse
        })

    perf_df = pd.DataFrame(perf_data)

    # Plot R², RMSE, MAE comparison
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    metrics = ["R²", "MAE", "RMSE"]
    colors = ["cornflowerblue", "sandybrown", "lightcoral"]

    for idx, metric in enumerate(metrics):
        sns.barplot(
            ax=axes[idx],
            x="Cortical Area",
            y=metric,
            data=perf_df,
            hue="Cortical Area",
            legend=False,
            palette=[colors[idx]] * len(perf_df)
        )
        axes[idx].set_title(f"{metric} across Cortical Areas", fontsize=14, fontweight="bold")
        axes[idx].set_xlabel("Cortical Area", fontsize=12)
        axes[idx].set_ylabel(metric, fontsize=12)
        
        # Add labels on top of bars
        for p in axes[idx].patches:
            axes[idx].annotate(
                f"{p.get_height():.4f}",
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='center',
                xytext=(0, 5),
                textcoords='offset points',
                fontsize=10, fontweight="bold"
            )

    plt.tight_layout()
    fig12_path = os.path.join(OUTPUT_DIR, "figure_12_performance_comparison.png")
    plt.savefig(fig12_path, dpi=300)
    plt.close()
    print(f"Saved Figure 12 to: {fig12_path}")


if __name__ == "__main__":
    main()
