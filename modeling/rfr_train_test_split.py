import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error


def main():
    parser = argparse.ArgumentParser(
        description="EEG Cognitive Workload Regression - Train-Test Split"
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
    # TRAIN–TEST MODELING
    # =========================

    results = []

    for area in df["cortical_area"].unique():
        area_df = df[df["cortical_area"] == area]

        # Extract features and targets
        # Note: Drop non-feature metadata and target column to prevent leak
        drop_cols = ["cortical_area", "channel", "epoch", "source_file", "mean"]
        X = area_df.drop(columns=[col for col in drop_cols if col in area_df.columns])
        y = area_df["mean"]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=0.2,
            random_state=42
        )

        # Define models to benchmark
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
            model.fit(X_train, y_train)

            # Predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Metrics
            results.append({
                "Cortical Area": area,
                "Model": model_name,
                "Observations": len(area_df),
                "R² (Train)": r2_score(y_train, y_train_pred),
                "R² (Test)": r2_score(y_test, y_test_pred),
                "MAE (Train)": mean_absolute_error(y_train, y_train_pred),
                "MAE (Test)": mean_absolute_error(y_test, y_test_pred),
                "RMSE (Train)": np.sqrt(mean_squared_error(y_train, y_train_pred)),
                "RMSE (Test)": np.sqrt(mean_squared_error(y_test, y_test_pred))
            })

    # =========================
    # RESULTS SUMMARY
    # =========================

    results_df = pd.DataFrame(results)
    print("\nTrain–Test Benchmarking Performance by Cortical Region:\n")
    print(results_df.to_string(index=False))


if __name__ == "__main__":
    main()

