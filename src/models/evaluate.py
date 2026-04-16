from pathlib import Path
import joblib
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.data.clean_data import prepare_training_dataframe
from src.data.load_data import load_train_data
from src.models.split_data import split_data, select_features


def evaluate(y_true, y_pred, name="Dataset"):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)

    print(f"\n{name} Evaluation")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")
    print(f"R2   : {r2:.4f}")

    return {"mae": mae, "rmse": rmse, "r2": r2}


def main():
    model_path = Path("artifacts/model.pkl")

    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    pipeline = joblib.load(model_path)

    if not hasattr(pipeline, "predict"):
        raise TypeError(
            "Loaded object does not have predict(). "
            "Make sure train.py saves a real sklearn Pipeline."
        )

    # Load and prepare data
    df = prepare_training_dataframe(load_train_data())

    X, y = select_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    # Predictions
    y_train_pred = pipeline.predict(X_train)
    y_val_pred = pipeline.predict(X_val)
    y_test_pred = pipeline.predict(X_test)

    # Metrics
    train_metrics = evaluate(y_train, y_train_pred, "Train")
    val_metrics = evaluate(y_val, y_val_pred, "Validation")
    test_metrics = evaluate(y_test, y_test_pred, "Test")

    # Overfitting check
    print("\nOverfitting Check")
    gap = val_metrics["rmse"] - train_metrics["rmse"]

    if gap > 5000:
        print(f"⚠️ Possible overfitting detected. RMSE gap = {gap:.2f}")
    else:
        print(f"✅ Generalization looks reasonable. RMSE gap = {gap:.2f}")


if __name__ == "__main__":
    main()