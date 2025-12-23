"""
Random Forest Regression with Train–Test Split
(Cortical Region-wise Analysis)

This script implements Random Forest Regression for EEG-based cognitive workload
estimation using a stratified cortical-region-wise analysis, as described in the study:

"Estimating Cognitive Workload in Robot-Assisted Surgery Using Time and Frequency
Features from EEG Epochs with Random Forest Regression".

Evaluation metrics:
- R²
- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)

Note:
- Feature CSV files are generated using epoch_feature_extraction.py
- EEG data are not included in this repository.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import matplotlib.pyplot as plt
import seaborn as sns


# =========================
# USER-DEFINED PATH
# =========================

FEATURE_FILE = "PATH_TO_FEATURES/epoch_features.csv"


# =========================
# LOAD AND PREPROCESS DATA
# =========================

df = pd.read_csv(FEATURE_FILE)

# Remove missing values
df = df.dropna().reset_index(drop=True)

# Validate required column
if "cortical_area" not in df.columns:
    raise ValueError("Column 'cortical_area' not found in feature file.")

if "mean" not in df.columns:
    raise ValueError("Target variable 'mean' not found in feature file.")


# =========================
# TRAIN–TEST MODELING
# =========================

results = {}

for area in df["cortical_area"].unique():

    area_df = df[df["cortical_area"] == area]

    X = area_df.drop(columns=["cortical_area"])
    y = area_df["mean"]   # Regression target (as defined in the manuscript)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42
    )

    model.fit(X_train, y_train)

    # Predictions
    y_train_pred = model.predict(X_train)
    y_test_pred = model.predict(X_test)

    # Metrics
    results[area] = {
        "Number of Observations": len(area_df),
        "R² (Train)": r2_score(y_train, y_train_pred),
        "MAE (Train)": mean_absolute_error(y_train, y_train_pred),
        "RMSE (Train)": np.sqrt(mean_squared_error(y_train, y_train_pred)),
        "R² (Test)": r2_score(y_test, y_test_pred),
        "MAE (Test)": mean_absolute_error(y_test, y_test_pred),
        "RMSE (Test)": np.sqrt(mean_squared_error(y_test, y_test_pred))
    }


# =========================
# RESULTS SUMMARY
# =========================

results_df = pd.DataFrame.from_dict(results, orient="index")
print("\nTrain–Test Performance by Cortical Region:\n")
print(results_df)


# =========================
# OPTIONAL VISUALIZATION
# =========================

metrics = ["R²", "MAE", "RMSE"]

for metric in metrics:
    plt.figure(figsize=(10, 5))
    sns.barplot(
        data=results_df.reset_index(),
        x="index",
        y=f"{metric} (Train)",
        color="skyblue",
        label="Train"
    )
    sns.barplot(
        data=results_df.reset_index(),
        x="index",
        y=f"{metric} (Test)",
        color="orange",
        label="Test"
    )

    plt.title(f"{metric} Comparison by Cortical Area")
    plt.xlabel("Cortical Area")
    plt.ylabel(metric)
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.show()

