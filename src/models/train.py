from pathlib import Path
import json
import joblib
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.clean_data import prepare_training_dataframe
from src.data.load_data import load_train_data
from src.models.split_data import select_features, split_data
from src.models.preprocess import build_preprocessor


def evaluate_model(
    name,
    pipeline,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
):
    """
    Fit a pipeline and compute RMSE on train/validation/test splits.
    """
    pipeline.fit(X_train, y_train)

    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)

    train_rmse = mean_squared_error(y_train, y_train_pred) ** 0.5
    val_rmse = mean_squared_error(y_val, y_val_pred) ** 0.5
    test_rmse = mean_squared_error(y_test, y_test_pred) ** 0.5

    return {
        "model_name": name,
        "pipeline": pipeline,
        "train_rmse": train_rmse,
        "val_rmse": val_rmse,
        "test_rmse": test_rmse,
    }


def main():
    print("Starting training...")

    df_raw = load_train_data()
    print(f"Data loaded: {df_raw.shape}")

    df = prepare_training_dataframe(df_raw)
    print(f"After cleaning + feature engineering: {df.shape}")

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
            y_val=y_val,
            X_test=X_test,
            y_test=y_test,
        )

        results.append(result)

        print(f"{model_name} RMSE:")
        print(f"  Train: {result['train_rmse']:.2f}")
        print(f"  Val  : {result['val_rmse']:.2f}")
        print(f"  Test : {result['test_rmse']:.2f}")

    # Choose best model based on validation RMSE
    best_result = min(results, key=lambda x: x["val_rmse"])
    best_pipeline = best_result["pipeline"]

    print("\nBest Model Selected:")
    print(f"Model: {best_result['model_name']}")
    print(f"Train RMSE: {best_result['train_rmse']:.2f}")
    print(f"Val   RMSE: {best_result['val_rmse']:.2f}")
    print(f"Test  RMSE: {best_result['test_rmse']:.2f}")

    # Persist per-model metrics for EDA / comparison plots
    artifacts_dir = Path(__file__).parent.parent / "artifacts"
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = artifacts_dir / "models_metrics.json"
    serializable = [
        {
            "model_name": r["model_name"],
            "train_rmse": r["train_rmse"],
            "val_rmse": r["val_rmse"],
            "test_rmse": r["test_rmse"],
        }
        for r in results
    ]

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(serializable, f, indent=2)

    print(f"Saved per-model metrics to {metrics_path}")

    artifacts_dir = Path("artifacts")
    artifacts_dir.mkdir(exist_ok=True)

    model_path = artifacts_dir / "best_model.joblib"
    metrics_path = artifacts_dir / "metrics.json"

    joblib.dump(best_pipeline, model_path)

    # Persist compact metrics for the selected best model
    metrics = {
        "best_model": best_result["model_name"],
        "train_rmse": best_result["train_rmse"],
        "val_rmse": best_result["val_rmse"],
        "test_rmse": best_result["test_rmse"],
    }

    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4)

    print(f"\nBest model saved at: {model_path}")
    print(f"Metrics saved at: {metrics_path}")
    print(f"File size: {model_path.stat().st_size} bytes")


if __name__ == "__main__":
    main()