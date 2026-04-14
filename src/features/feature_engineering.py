import pandas as pd


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create engineered features for the dataset.
    """
    df = df.copy()

    # -------------------------
    # Total square footage
    # -------------------------
    df["TotalSF"] = (
        df["TotalBsmtSF"].fillna(0)
        + df["1stFlrSF"].fillna(0)
        + df["2ndFlrSF"].fillna(0)
    )

    # -------------------------
    # House age
    # -------------------------
    df["HouseAge"] = df["YrSold"] - df["YearBuilt"]

    # -------------------------
    # Overall score
    # -------------------------
    df["OverallScore"] = df["OverallQual"] * df["OverallCond"]

    # -------------------------
    # Garage score
    # -------------------------
    df["GarageScore"] = (
        df["GarageCars"].fillna(0)
        * df["GarageArea"].fillna(0)
    )

    return df