from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.models.split_data import NUMERIC_FEATURES, CATEGORICAL_FEATURES


def build_preprocessor(use_scaler: bool = False):
    """
    Build a robust preprocessing pipeline.

    Parameters
    ----------
    use_scaler : bool
        If True, apply StandardScaler to numeric features.
        If False, only apply imputation to numeric features.

    Returns
    -------
    ColumnTransformer
        Preprocessing object for numeric and categorical columns.
    """

    numeric_steps = [
        ("imputer", SimpleImputer(strategy="median"))
    ]

    if use_scaler:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES)
    ])

    return preprocessor


if __name__ == "__main__":
    from src.data.load_data import load_train_data
    from src.features.feature_engineering import engineer_features
    from src.models.split_data import select_features, split_data

    df = load_train_data()
    df = engineer_features(df)

    X, y = select_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    preprocessor = build_preprocessor(use_scaler=False)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)
    X_test_processed = preprocessor.transform(X_test)

    print("Original X_train shape:", X_train.shape)
    print("Processed X_train shape:", X_train_processed.shape)
    print("Processed X_val shape:", X_val_processed.shape)
    print("Processed X_test shape:", X_test_processed.shape)