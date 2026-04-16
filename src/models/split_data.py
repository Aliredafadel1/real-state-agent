from sklearn.model_selection import train_test_split

import pandas as pd

from src.data.clean_data import prepare_training_dataframe
from src.data.load_data import load_train_data


NUMERIC_FEATURES = [
    "OverallQual",
    "GrLivArea",
    "GarageCars",
    "TotalBsmtSF",
    "FullBath",
    "YearBuilt",
    "TotalSF",
    "HouseAge",
]

CATEGORICAL_FEATURES = [
    "Neighborhood",
    "HouseStyle",
    "GarageType",
    "ExterQual",
]

TARGET_COLUMN = "SalePrice"


def select_features(df: pd.DataFrame):
    """
    Select input features X and target y from the dataframe.
    """
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X = df[feature_columns].copy()
    y = df[TARGET_COLUMN].copy()

    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, random_state: int = 42):
    """
    Split data into train, validation, and test sets.
    """
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=random_state
    )

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test


if __name__ == "__main__":
    df = load_train_data()
    df = prepare_training_dataframe(df)

    X, y = select_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    print("Train:", X_train.shape, y_train.shape)
    print("Validation:", X_val.shape, y_val.shape)
    print("Test:", X_test.shape, y_test.shape)
    print("\nTrain columns:")
    print(X_train.columns.tolist())