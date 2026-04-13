# ================================
# 📦 IMPORTS
# ================================
from pathlib import Path
import pandas as pd


# ================================
# 📍 PATH HANDLING (Project structure)
# ================================

# Get the root directory of the project
def get_project_root() -> Path:
    return Path(__file__).resolve().parents[1]


# Build full path to data file inside data/raw/
def get_data_path(filename: str) -> Path:
    data_path = get_project_root() / "data" / "raw" / filename

    # Check if file exists
    if not data_path.exists():
        raise FileNotFoundError(
            f"File '{filename}' not found at: {data_path}"
        )

    return data_path


# ================================
# 🔍 DATA INSPECTION (Quick checks)
# ================================

# Function to print basic info about dataset
def inspect_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> None:
    print(f"\n{name} inspection")
    print("-" * 50)
    print(f"Shape: {df.shape}")
    print(f"\nColumns:\n{df.columns.tolist()}")
    print(f"\nData types:\n{df.dtypes}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    print(f"\nFirst 5 rows:\n{df.head()}")


# ================================
# 📥 DATA LOADING CORE FUNCTION
# ================================

# Generic loader for any CSV file
def load_csv(filename: str, inspect: bool = False) -> pd.DataFrame:
    path = get_data_path(filename)
    df = pd.read_csv(path)

    # Optional inspection
    if inspect:
        inspect_dataframe(df, name=filename)

    return df


# ================================
# 📊 SPECIFIC DATA LOADERS
# ================================

# Load training dataset
def load_train_data(inspect: bool = False) -> pd.DataFrame:
    return load_csv("train.csv", inspect=inspect)


# Load test dataset
def load_test_data(inspect: bool = False) -> pd.DataFrame:
    return load_csv("test.csv", inspect=inspect)


# Load sample submission file
def load_sample_submission(inspect: bool = False) -> pd.DataFrame:
    return load_csv("sample_submission.csv", inspect=inspect)


# ================================
# 🚀 LOCAL TESTING (Run this file directly)
# ================================

if __name__ == "__main__":
    train_df = load_train_data(inspect=True)