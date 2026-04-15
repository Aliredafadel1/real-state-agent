from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.load_data import load_train_data
from src.features.feature_engineering import engineer_features
from src.models.split_data import select_features, split_data
from src.models.preprocess import build_preprocessor
from pathlib import Path
import json


def evaluate_model(name, pipeline, X_train, y_train, X_val, y_val):
    pipeline.fit(X_train, y_train)
    y_val_pred = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, y_val_pred)
    rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
    r2 = r2_score(y_val, y_val_pred)

    return {
        "model_name": name,
        "pipeline": pipeline,
        "mae": mae,
        "rmse": rmse,
        "r2": r2
    }


def main():
    print("Starting training...")

    df = load_train_data()
    print(f"Data loaded: {df.shape}")

    df = engineer_features(df)
    print(f"After feature engineering: {df.shape}")

    X, y = select_features(df)
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)
    print(f"Train shape: {X_train.shape}")
    print(f"Validation shape: {X_val.shape}")
    print(f"Test shape: {X_test.shape}")

    preprocessor = build_preprocessor(use_scaler=False)

    models = {
        "LinearRegression": LinearRegression(),
        "RandomForestRegressor": RandomForestRegressor(
            n_estimators=200,
            random_state=42
        ),
        "GradientBoostingRegressor": GradientBoostingRegressor(
            random_state=42
        ),
    }

    results = []

    for model_name, model in models.items():
        print(f"\nTraining {model_name}...")

        pipeline = Pipeline([
            ("preprocessor", preprocessor),
            ("model", model)
        ])

        result = evaluate_model(
            name=model_name,
            pipeline=pipeline,
            X_train=X_train,
            y_train=y_train,
            X_val=X_val,
            y_val=y_val
        )

        results.append(result)

        print(f"{model_name} Validation Metrics:")
        print(f"MAE  : {result['mae']:.2f}")
        print(f"RMSE : {result['rmse']:.2f}")
        print(f"R2   : {result['r2']:.4f}")

    # Choose best model based on RMSE
    best_result = min(results, key=lambda x: x["rmse"])
    best_pipeline = best_result["pipeline"]

    print("\nBest Model Selected:")
    print(f"Model: {best_result['model_name']}")
    print(f"MAE  : {best_result['mae']:.2f}")
    print(f"RMSE : {best_result['rmse']:.2f}")
    print(f"R2   : {best_result['r2']:.4f}")

    # Persist per-model metrics for EDA / comparison plots
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = artifacts_dir / "models_metrics.json"
    serializable = [
        {
            "model_name": r["model_name"],
            "mae": r["mae"],
            "rmse": r["rmse"],
            "r2": r["r2"]
        }
        for r in results
    ]

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved per-model metrics to {metrics_path}")

    # Final evaluation on test set
    y_test_pred = best_pipeline.predict(X_test)

    test_mae = mean_absolute_error(y_test, y_test_pred)
    test_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5
    test_r2 = r2_score(y_test, y_test_pred)

    print("\nTest Metrics:")
    print(f"MAE  : {test_mae:.2f}")
    print(f"RMSE : {test_rmse:.2f}")
    print(f"R2   : {test_r2:.4f}")

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    model_path = artifacts_dir / "best_model.joblib"
    metrics_path = artifacts_dir / "metrics.json"

    joblib.dump(best_pipeline, model_path)

    metrics = {
        "best_model": best_result["model_name"],
        "validation_mae": best_result["mae"],
        "validation_rmse": best_result["rmse"],
        "validation_r2": best_result["r2"],
        "test_mae": test_mae,
        "test_rmse": test_rmse,
        "test_r2": test_r2
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nBest model saved at: {model_path}")
    print(f"Metrics saved at: {metrics_path}")
    print(f"File size: {model_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()