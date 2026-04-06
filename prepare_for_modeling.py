import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from feature_engineering import df

MISSING_CATEGORICAL_PLACEHOLDER = "unknown"
TARGET_CANDIDATES = ["churn", "churnlabel"]
RISK_FEATURES = {
    "premium",
    "smoker",
    "diabetes",
    "chronic_disease",
    "pre_existing_conditions",
}

df_model = df.copy()

TARGET_COL = next((c for c in TARGET_CANDIDATES if c in df_model.columns), None)
if TARGET_COL is None:
    raise KeyError(f"Target column not found. Expected one of: {TARGET_CANDIDATES}")

def _fill_numeric_series(series: pd.Series) -> pd.Series:
    median = series.median()
    fill_value = 0 if pd.isna(median) else median
    return series.fillna(fill_value)


def _fill_missing_values(df_in: pd.DataFrame) -> pd.DataFrame:
    out = df_in.copy()
    for col in out.columns:
        if out[col].isnull().any():
            if pd.api.types.is_numeric_dtype(out[col]):
                out[col] = _fill_numeric_series(out[col])
            else:
                non_null = out[col].dropna()
                mode = non_null.mode() if not non_null.empty else pd.Series(dtype=object)
                out[col] = out[col].fillna(
                    mode.iloc[0] if not mode.empty else MISSING_CATEGORICAL_PLACEHOLDER
                )
    return out


# Fill missing values before encoding/scaling
df_model = _fill_missing_values(df_model)

# Encode categorical columns
categorical_cols = [
    c for c in df_model.columns
    if c != TARGET_COL and (
        pd.api.types.is_object_dtype(df_model[c])
        or pd.api.types.is_bool_dtype(df_model[c])
        or str(df_model[c].dtype) == "category"
    )
]

label_encoders = {}
for col in categorical_cols:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

# Ensure non-target columns are numeric
feature_cols = [c for c in df_model.columns if c != TARGET_COL]
for col in feature_cols:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce")
    if df_model[col].isnull().any():
        df_model[col] = _fill_numeric_series(df_model[col])

# Data leakage prevention: ensure risk-label features are excluded from X
overlap = sorted(set(feature_cols).intersection(RISK_FEATURES))
if overlap:
    raise ValueError(
        "Data leakage prevention check failed: risk features present in X: "
        f"{overlap}"
    )

# Scale numeric features
scaler = StandardScaler()
df_model[feature_cols] = scaler.fit_transform(df_model[feature_cols])

# Split X and y for downstream scripts
y_numeric = pd.to_numeric(df_model[TARGET_COL], errors="coerce")
if y_numeric.isnull().any():
    bad_count = int(y_numeric.isnull().sum())
    raise ValueError(
        f"Target column '{TARGET_COL}' has {bad_count} non-numeric value(s). "
        "Fix target values before modeling."
    )
y = y_numeric.astype(int)
X = df_model[feature_cols]

print("\n" + "=" * 50)
print("Modeling preparation complete.")
print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"Target  : {TARGET_COL}")
print("Data leakage prevention: risk features removed from training X.")
print("Final training features:")
for col in X.columns:
    print(f"  {col}")
