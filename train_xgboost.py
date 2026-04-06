import pandas as pd
from sklearn.metrics import f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


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

pos_count = int((y_train == 1).sum())
neg_count = int((y_train == 0).sum())
scale_pos_weight = (neg_count / pos_count) if pos_count > 0 else 1.0

model = XGBClassifier(
    n_estimators=200,
    random_state=42,
    eval_metric="logloss",
    scale_pos_weight=scale_pos_weight,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

f1 = f1_score(y_test, y_pred, zero_division=0)
roc_auc = roc_auc_score(y_test, y_proba)

print("\n" + "=" * 60)
print("XGBoost Evaluation (Churn Prediction)")
print("=" * 60)
print(f"Dataset shape      : {df.shape}")
print("Class distribution :")
print(y.value_counts().to_string())
print(f"\nscale_pos_weight   : {scale_pos_weight:.4f}")
print("\nMetrics (test set):")
print(f"F1-score  : {f1:.4f}")
print(f"ROC-AUC   : {roc_auc:.4f}")
