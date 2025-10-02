"""
Train model from config/model_config.json for reproducible CI/CD builds.
"""
import json
import pandas as pd
import joblib
import os
import sys
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
import numpy as np


def load_config(config_path: str = "config/model_config.json") -> dict:
    """Load model configuration."""
    with open(config_path, "r") as f:
        return json.load(f)


def build_preprocessor(numeric_cols, categorical_cols):
    """Build preprocessing pipeline from config."""
    return ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="median")),
                        ("scaler", StandardScaler()),
                    ]
                ),
                numeric_cols,
            ),
            (
                "cat",
                Pipeline(
                    steps=[
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        ("oh", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
                    ]
                ),
                categorical_cols,
            ),
        ]
    )


def build_model(config: dict, preprocessor):
    """Build model pipeline from config."""
    model_type = config["model_type"]
    hyperparams = config["hyperparameters"]
    
    if model_type == "LinearRegression":
        from sklearn.linear_model import LinearRegression
        estimator = LinearRegression(**hyperparams)
    elif model_type == "RandomForestRegressor":
        from sklearn.ensemble import RandomForestRegressor
        estimator = RandomForestRegressor(**hyperparams)
    elif model_type == "XGBRegressor":
        from xgboost import XGBRegressor
        estimator = XGBRegressor(**hyperparams)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return Pipeline(steps=[("prep", preprocessor), ("model", estimator)])


def train_and_validate(config_path: str = "config/model_config.json", 
                       data_path: str = "prompt/final_processed_zudio_data.csv",
                       output_path: str = "models/model.joblib",
                       min_r2: float = 0.5):
    """Train model from config and validate performance."""
    
    print(f"Loading config from {config_path}...")
    config = load_config(config_path)
    
    print(f"Loading data from {data_path}...")
    df = pd.read_csv(data_path)
    
    target_col = config["target_column"]
    feature_cols = config["feature_columns"]
    numeric_cols = config["numeric_columns"]
    categorical_cols = config["categorical_columns"]
    
    print(f"Target: {target_col}")
    print(f"Features: {len(feature_cols)} columns")
    print(f"Model: {config['model_type']}")
    print(f"Hyperparameters: {config['hyperparameters']}")
    
    # Prepare data
    X = df[feature_cols]
    y = df[target_col]
    
    # Split
    split_config = config.get("train_test_split", {})
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=split_config.get("test_size", 0.2),
        random_state=split_config.get("random_state", 42)
    )
    
    # Build pipeline
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)
    model = build_model(config, preprocessor)
    
    # Train
    print("Training model...")
    model.fit(X_train, y_train)
    
    # Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    
    print(f"\nMetrics on test set:")
    print(f"  R²:   {r2:.4f}")
    print(f"  MAE:  {mae:.4f}")
    print(f"  RMSE: {rmse:.4f}")
    
    # Compare with expected metrics from config
    expected_metrics = config.get("metrics", {})
    if expected_metrics:
        r2_diff = abs(r2 - expected_metrics.get("R2", r2))
        print(f"\nMetric drift from config:")
        print(f"  ΔR²: {r2_diff:.4f}")
        if r2_diff > 0.05:
            print(f"  ⚠️  Warning: R² drift exceeds 0.05")
    
    # Validation threshold
    if r2 < min_r2:
        print(f"\n❌ FAILED: R² ({r2:.4f}) below threshold ({min_r2})")
        sys.exit(1)
    
    # Save model
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    bundle = {
        "model": model,
        "config": config,
        "columns": feature_cols,
        "target": target_col,
        "metrics": {"R2": r2, "MAE": mae, "RMSE": rmse},
        "trained_at": pd.Timestamp.now().isoformat()
    }
    joblib.dump(bundle, output_path)
    print(f"\n✅ Model saved to {output_path}")
    
    return model, r2, mae, rmse


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train model from config")
    parser.add_argument("--config", default="config/model_config.json", help="Path to config file")
    parser.add_argument("--data", default="prompt/final_processed_zudio_data.csv", help="Path to data")
    parser.add_argument("--output", default="models/model.joblib", help="Output model path")
    parser.add_argument("--min-r2", type=float, default=0.5, help="Minimum R² threshold")
    args = parser.parse_args()
    
    train_and_validate(args.config, args.data, args.output, args.min_r2)