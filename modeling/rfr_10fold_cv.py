import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import ttest_rel


def main():
    parser = argparse.ArgumentParser(
        description="EEG Cognitive Workload Regression - 10-Fold Cross Validation"
    )
    parser.add_argument(
        "--feature_file",
        type=str,
        default="data/features/epoch_features.csv",
        help="Path to the generated epoch features CSV file"
    )
    args = parser.parse_args()

    FEATURE_FILE = args.feature_file

    if not os.path.exists(FEATURE_FILE):
        print(f"Error: Feature file '{FEATURE_FILE}' does not exist.")
        return

    # Load data
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

    results = []

    # =========================
    # REGION-WISE MODELING
    # =========================

    for area in df["cortical_area"].unique():
        area_df = df[df["cortical_area"] == area]

        drop_cols = ["cortical_area", "channel", "epoch", "source_file", "mean"]
        X = area_df.drop(columns=[col for col in drop_cols if col in area_df.columns])
        y = area_df["mean"]

        # Models to evaluate
        models = {
            "Random Forest Regression": RandomForestRegressor(
                n_estimators=100,
                random_state=42
            ),
            "SVR": SVR(
                kernel="rbf",
                C=1.0,
                epsilon=0.1,
                gamma="scale"
            ),
            "Linear Regression": LinearRegression(
                fit_intercept=True
            ),
            "XGBoost": XGBRegressor(
                n_estimators=100,
                max_depth=3,
                learning_rate=0.1,
                subsample=0.8,
                random_state=42
            )
        }

        for model_name, model in models.items():
            r2_scores, maes, rmses = [], [], []
            model_errors, baseline_errors = [], []

            for train_idx, test_idx in kf.split(X):
                X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
                y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

                # Train model
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
            # STATISTICAL SIGNIFICANCE (Paired t-test against training mean baseline)
            # =========================
            t_stat, p_value = ttest_rel(
                baseline_errors,
                model_errors
            )

            results.append({
                "Cortical Area": area,
                "Model": model_name,
                "Observations": len(area_df),
                "R² (10-fold CV)": np.mean(r2_scores),
                "MAE (10-fold CV)": np.mean(maes),
                "RMSE (10-fold CV)": np.mean(rmses),
                "p-value (vs Baseline)": p_value,
                "Significant (p < 0.05)": "Yes" if p_value < 0.05 else "No"
            })

    # =========================
    # RESULTS SUMMARY
    # =========================

    results_df = pd.DataFrame(results)
    print("\n10-Fold Cross-Validation Results with Statistical Testing:\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()
