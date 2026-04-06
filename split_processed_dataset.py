import pandas as pd
from sklearn.model_selection import train_test_split


DATA_PATH = "processed_features.csv"
TARGET_CANDIDATES = ["churn", "churnlabel"]


df = pd.read_csv(DATA_PATH)

target_col = next((col for col in TARGET_CANDIDATES if col in df.columns), None)
if target_col is None:
    raise KeyError(
        "Target column not found. Expected: churn or churnlabel."
    )

if target_col != "churn":
    df = df.rename(columns={target_col: "churn"})

X = df.drop(columns=["churn"])
y = df["churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"Dataset shape: {df.shape}")
print("Class distribution:")
print(y.value_counts().to_string())
print(f"Train shape: X={X_train.shape}, y={y_train.shape}")
print(f"Test shape : X={X_test.shape}, y={y_test.shape}")
