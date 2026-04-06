import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = "processed_features.csv"
TARGET_COL = "churn"
BATCH_SIZE = 32

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError as exc:
    raise FileNotFoundError(
        f"Input dataset not found at '{DATA_PATH}'. Ensure preprocessing has run and the file exists."
    ) from exc

if TARGET_COL not in df.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found in {DATA_PATH}.")

X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

X_train_tensor = torch.tensor(X_train.values, dtype=torch.float32)
X_test_tensor = torch.tensor(X_test.values, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

print("\n" + "=" * 60)
print("PyTorch Churn Data Preparation")
print("=" * 60)
print(f"X_train shape : {tuple(X_train_tensor.shape)}")
print(f"y_train shape : {tuple(y_train_tensor.shape)}")
print(f"Train batches : {len(train_loader)}")
print(f"Test batches  : {len(test_loader)}")
