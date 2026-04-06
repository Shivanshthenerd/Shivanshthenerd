import pandas as pd

from prepare_for_modeling import X, y  # X: DataFrame, y: Series

# ── Validate: no missing values ───────────────────────────────────────────────
x_nulls = X.isnull().sum().sum()
y_nulls = y.isnull().sum()

if x_nulls > 0:
    raise ValueError(f"X contains {x_nulls} missing value(s). "
                     "Fix the pipeline before saving.")
if y_nulls > 0:
    raise ValueError(f"y contains {y_nulls} missing value(s). "
                     "Fix the pipeline before saving.")

target_values = set(y.unique().tolist())
if not target_values.issubset({0, 1}):
    raise ValueError(
        f"Target must be binary 0/1 before save. Found values: {sorted(target_values)}"
    )

class_counts = y.value_counts()
if len(class_counts) < 2:
    raise ValueError("Target must contain at least two classes before save.")
if class_counts.min() < 2:
    raise ValueError(
        f"Each class must have at least 2 samples before save. Counts: {class_counts.to_dict()}"
    )

# ── Combine X and y into a single DataFrame ───────────────────────────────────
processed = X.copy()
processed["churn"] = y.values  # append target as the last column

if processed.isnull().sum().sum() > 0:
    raise ValueError("Final processed dataset contains null values.")

# ── Save to CSV ───────────────────────────────────────────────────────────────
FEATURES_PATH = "processed_features.csv"
processed.to_csv(FEATURES_PATH, index=False)

# ── Print summary ─────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Processed dataset saved successfully.")
print("=" * 50)
print(f"\nFile            : {FEATURES_PATH}")
print(f"Final shape     : {processed.shape}  "
      f"({processed.shape[0]} rows × {processed.shape[1]} columns)")
print(f"Feature columns : {X.shape[1]}")
print(f"Target column   : churn")
print(f"\nMissing values  : {processed.isnull().sum().sum()} (verified clean)")
print(f"\nTarget distribution:\n{processed['churn'].value_counts().to_string()}")
print("\nColumn list:")
for col in processed.columns:
    print(f"  {col}")
