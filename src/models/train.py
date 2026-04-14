from src.data.load_data import load_train_data
from src.features.feature_engineering import engineer_features
from src.models.split_data import select_features, split_data


# 1. Load data
df = load_train_data()

# 2. Apply feature engineering
df = engineer_features(df)

# 3. Select features + target
X, y = select_features(df)

# 4. Split data
X_train, X_val, X_test, y_train, y_val, y_test = split_data(X, y)

# 5. Check shapes
print("Train:", X_train.shape)
print("Validation:", X_val.shape)
print("Test:", X_test.shape)