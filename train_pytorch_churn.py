import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = "processed_features.csv"
TARGET_COL = "churn"
BATCH_SIZE = 32
DROPOUT_RATE = 0.3


class TabularDLModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.block1 = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )
        self.block2 = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )
        self.block3 = nn.Sequential(
            nn.Linear(128, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )
        self.block4 = nn.Sequential(
            nn.Linear(64, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(DROPOUT_RATE),
        )
        self.output = nn.Linear(32, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        return self.output(x)

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
y_train_tensor = torch.tensor(y_train.values, dtype=torch.float32).unsqueeze(1)
y_test_tensor = torch.tensor(y_test.values, dtype=torch.float32).unsqueeze(1)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

model = TabularDLModel(input_dim=X_train_tensor.shape[1])
sample_logits = model(X_train_tensor[: min(BATCH_SIZE, X_train_tensor.shape[0])])

print("\n" + "=" * 60)
print("PyTorch Churn Data Preparation")
print("=" * 60)
print(f"X_train shape : {tuple(X_train_tensor.shape)}")
print(f"y_train shape : {tuple(y_train_tensor.shape)}")
print(f"Train batches : {len(train_loader)}")
print(f"Test batches  : {len(test_loader)}")
print(f"Sample logits shape : {tuple(sample_logits.shape)}")
