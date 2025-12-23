"""
Random Forest Regression with 10-Fold Cross-Validation
and Statistical Significance Testing

This script evaluates EEG-based cognitive workload estimation using
Random Forest Regression with 10-fold cross-validation, as described in:

"Estimating Cognitive Workload in Robot-Assisted Surgery Using Time and Frequency
Features from EEG Epochs with Random Forest Regression".

Model performance is compared against a baseline predictor (mean of training data)
using a paired t-test on absolute errors.

Metrics reported:
- R² (10-fold CV)
- MAE (10-fold CV)
- RMSE (10-fold CV)
- Paired t-test p-value
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_rel


# =========================
# USER-DEFINED PATH
# =========================

FEATURE_FILE = "PATH_TO_FEATURES/epoch_features.csv"


# =========================
# LOAD AND PREPROCESS DATA
# =========================

df = pd.read_csv(FEATURE_FILE)
df = df.dropna().reset_index(drop=True)

# Validate required columns
required_cols = {"cortical_area", "mean"}
if not required_cols.issubset(df.columns):
    raise ValueError(
        f"Feature file must contain columns: {required_cols}"
    )


# =========================
# CROSS-VALIDATION SETUP
# =========================

kf = KFold(
    n_splits=10,
    shuffle=True,
    random_state=42
)

results = {}


# =========================
# REGION-WISE MODELING
# =========================

for area in df["cortical_area"].unique():

    area_df = df[df["cortical_area"] == area]

    X = area_df.drop(columns=["cortical_area", "mean"])
    y = area_df["mean"]

    r2_scores, maes, rmses = [], [], []
    model_errors, baseline_errors = [], []

    for train_idx, test_idx in kf.split(X):

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        # Train Random Forest Regressor
        model = RandomForestRegressor(
            n_estimators=100,
            random_state=42
        )
        model.fit(X_train, y_train)

        # Predictions
        y_pred = model.predict(X_test)

        # Baseline predictor (mean of training labels)
        baseline_pred = np.full_like(
            y_test.values,
            fill_value=y_train.mean(),
            dtype=float
        )

        # Absolute errors for paired statistical test
        model_errors.extend(
            np.abs(y_test.values - y_pred)
        )
        baseline_errors.extend(
            np.abs(y_test.values - baseline_pred)
        )

        # Performance metrics
        r2_scores.append(r2_score(y_test, y_pred))
        maes.append(mean_absolute_error(y_test, y_pred))
        rmses.append(
            np.sqrt(mean_squared_error(y_test, y_pred))
        )

    # =========================
    # STATISTICAL SIGNIFICANCE
    # =========================
    t_stat, p_value = ttest_rel(
        baseline_errors,
        model_errors
    )

    results[area] = {
        "Number of Observations": len(area_df),
        "R² (10-fold CV)": np.mean(r2_scores),
        "MAE (10-fold CV)": np.mean(maes),
        "RMSE (10-fold CV)": np.mean(rmses),
        "Paired t-test p-value": p_value,
        "Significant (p < 0.05)": "Yes" if p_value < 0.05 else "No"
    }


# =========================
# RESULTS SUMMARY
# =========================

results_df = pd.DataFrame.from_dict(results, orient="index")
print("\n10-Fold Cross-Validation Results with Statistical Testing:\n")
print(results_df)
