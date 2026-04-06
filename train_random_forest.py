import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split


DATA_PATH = "processed_features.csv"
TARGET_COL = "churn"

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

model = RandomForestClassifier(
    n_estimators=200, class_weight="balanced", random_state=42
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, zero_division=0)
recall = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)

importances = pd.Series(model.feature_importances_, index=X.columns).sort_values(
    ascending=False
)

print("\n" + "=" * 60)
print("Random Forest Evaluation (Churn Prediction)")
print("=" * 60)
print(f"Dataset shape      : {df.shape}")
print("Class distribution :")
print(y.value_counts().to_string())
print("\nMetrics (test set):")
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")

print("\nFeature importances:")
for rank, (feature, score) in enumerate(importances.items(), start=1):
    print(f"{rank:>2}. {feature:<35} {score:.6f}")
