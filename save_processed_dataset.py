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

# ── Combine X and y into a single DataFrame ───────────────────────────────────
processed = X.copy()
processed["churn"] = y.values  # append target as the last column

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
