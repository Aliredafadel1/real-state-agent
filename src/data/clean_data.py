from pathlib import Path

import pandas as pd


def clean_data(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Clean the House Prices dataset and return features (X) and target (y).

    Steps:
    1. Copy dataset
    2. Separate target
    3. Drop useless columns
    4. Drop columns with too many missing values
    5. Fill columns where missing means absence
    6. Fill remaining numeric missing values with median
    7. Fill remaining categorical missing values with 'Missing'
    8. Fix data types for categorical-like numeric columns
    9. Remove duplicate rows
    """

    df = df.copy()

    if "SalePrice" not in df.columns:
        raise ValueError("Target column 'SalePrice' not found in dataframe.")

    # Separate target and features
    y = df["SalePrice"]
    X = df.drop(columns=["SalePrice"])

    # Drop useless identifier column
    X = X.drop(columns=["Id"], errors="ignore")

    # Drop columns with too many missing values
    high_missing_cols = ["PoolQC", "Fence", "MiscFeature"]
    X = X.drop(columns=high_missing_cols, errors="ignore")

    # Columns where missing means the feature does not exist
    none_fill_cols = [
        "Alley",
        "FireplaceQu",
        "GarageType",
        "GarageFinish",
        "GarageQual",
        "GarageCond",
        "BsmtQual",
        "BsmtCond",
        "BsmtExposure",
        "BsmtFinType1",
        "BsmtFinType2",
        "MasVnrType",
    ]

    for col in none_fill_cols:
        if col in X.columns:
            X[col] = X[col].fillna("None")

    # Special numeric columns where missing means absence
    zero_fill_cols = [
        "GarageYrBlt",
        "GarageArea",
        "GarageCars",
        "BsmtFinSF1",
        "BsmtFinSF2",
        "BsmtUnfSF",
        "TotalBsmtSF",
        "BsmtFullBath",
        "BsmtHalfBath",
        "MasVnrArea",
    ]

    for col in zero_fill_cols:
        if col in X.columns:
            X[col] = X[col].fillna(0)

    # Fill LotFrontage with neighborhood median if possible
    if "LotFrontage" in X.columns:
        if "Neighborhood" in X.columns:
            X["LotFrontage"] = X.groupby("Neighborhood")["LotFrontage"].transform(
                lambda s: s.fillna(s.median())
            )
        X["LotFrontage"] = X["LotFrontage"].fillna(X["LotFrontage"].median())

    # Fill remaining numeric missing values with median
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns
    for col in numeric_cols:
        X[col] = X[col].fillna(X[col].median())

    # Fill remaining categorical missing values with "Missing"
    categorical_cols = X.select_dtypes(include=["object"]).columns
    for col in categorical_cols:
        X[col] = X[col].fillna("Missing")

    # Convert categorical-like numeric column to string
    if "MSSubClass" in X.columns:
        X["MSSubClass"] = X["MSSubClass"].astype(str)

    # Remove duplicate rows
    X = X.drop_duplicates()

    # Keep target aligned with cleaned features
    y = y.loc[X.index]

    return X, y


def load_raw_data(file_path: str | Path) -> pd.DataFrame:
    """
    Load raw CSV dataset from a given path.
    """
    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    return pd.read_csv(file_path)


if __name__ == "__main__":
    data_path = Path(__file__).resolve().parents[2] / "data" / "raw" / "train.csv"

    df = load_raw_data(data_path)
    X, y = clean_data(df)

    print("Cleaning completed successfully.")
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Remaining missing values in X: {X.isnull().sum().sum()}")