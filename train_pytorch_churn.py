import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset


DATA_PATH = "processed_features.csv"
TARGET_COL = "churn"
BATCH_SIZE = 32
DROPOUT_RATE = 0.3
LR = 0.001
EPOCHS = 25


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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = TabularDLModel(input_dim=X_train_tensor.shape[1]).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

print("\n" + "=" * 60)
print("PyTorch Churn Training")
print("=" * 60)
print(f"X_train shape : {tuple(X_train_tensor.shape)}")
print(f"y_train shape : {tuple(y_train_tensor.shape)}")
print(f"Train batches : {len(train_loader)}")
print(f"Test batches  : {len(test_loader)}")
print(f"Device        : {device}")

train_losses = []
train_accuracies = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    epoch_loss = 0.0
    correct = 0
    total = 0

    for batch_x, batch_y in train_loader:
        batch_x = batch_x.to(device)
        batch_y = batch_y.to(device)

        optimizer.zero_grad()
        logits = model(batch_x)
        loss = criterion(logits, batch_y)
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item() * batch_x.size(0)

        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        correct += (preds == batch_y).sum().item()
        total += batch_y.numel()

    avg_loss = epoch_loss / len(train_loader.dataset)
    epoch_acc = correct / total
    train_losses.append(avg_loss)
    train_accuracies.append(epoch_acc)
    print(f"Epoch {epoch:02d}/{EPOCHS} - Loss: {avg_loss:.4f} - Accuracy: {epoch_acc:.4f}")

model.eval()
all_probs = []
all_preds = []
all_targets = []

with torch.no_grad():
    for batch_x, batch_y in test_loader:
        batch_x = batch_x.to(device)
        logits = model(batch_x)
        probs = torch.sigmoid(logits).squeeze(1)
        preds = (probs >= 0.5).float()

        all_probs.extend(probs.cpu().numpy().tolist())
        all_preds.extend(preds.cpu().numpy().tolist())
        all_targets.extend(batch_y.squeeze(1).cpu().numpy().tolist())

accuracy = accuracy_score(all_targets, all_preds)
precision = precision_score(all_targets, all_preds, zero_division=0)
recall = recall_score(all_targets, all_preds, zero_division=0)
f1 = f1_score(all_targets, all_preds, zero_division=0)
roc_auc = roc_auc_score(all_targets, all_probs)

print("\n" + "=" * 60)
print("PyTorch Churn Evaluation (Test Set)")
print("=" * 60)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

cm = confusion_matrix(all_targets, all_preds)
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].imshow(cm, interpolation="nearest", cmap="Blues")
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted label")
axes[0].set_ylabel("True label")
axes[0].set_xticks([0, 1])
axes[0].set_yticks([0, 1])
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[0].text(j, i, str(cm[i, j]), ha="center", va="center", color="black")

fpr, tpr, _ = roc_curve(all_targets, all_probs)
axes[1].plot(fpr, tpr, label=f"ROC (AUC = {roc_auc:.4f})")
axes[1].plot([0, 1], [0, 1], linestyle="--")
axes[1].set_title("ROC Curve")
axes[1].set_xlabel("False Positive Rate")
axes[1].set_ylabel("True Positive Rate")
axes[1].legend(loc="lower right")

plt.tight_layout()
plt.show()

epochs = list(range(1, EPOCHS + 1))
plt.figure(figsize=(10, 4))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_losses, marker="o")
plt.title("Training Loss by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.grid(alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(epochs, train_accuracies, marker="o")
plt.title("Training Accuracy by Epoch")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.ylim(0, 1)
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig("pytorch_training_curves.png", dpi=150)
plt.show()
