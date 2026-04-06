import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from feature_engineering import df


df_model = df.copy()

TARGET_CANDIDATES = ["churn", "churnlabel"]
target_col = next((c for c in TARGET_CANDIDATES if c in df_model.columns), None)
if target_col is None:
    raise KeyError(f"Target column not found. Expected one of: {TARGET_CANDIDATES}")

# Fill missing values before encoding/scaling
for col in df_model.columns:
    if df_model[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df_model[col]):
            df_model[col] = df_model[col].fillna(df_model[col].median())
        else:
            mode = df_model[col].mode(dropna=True)
            df_model[col] = df_model[col].fillna(mode.iloc[0] if not mode.empty else "unknown")

# Encode categorical columns
categorical_cols = [
    c for c in df_model.columns
    if c != target_col and (
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
feature_cols = [c for c in df_model.columns if c != target_col]
for col in feature_cols:
    df_model[col] = pd.to_numeric(df_model[col], errors="coerce")
    if df_model[col].isnull().any():
        df_model[col] = df_model[col].fillna(df_model[col].median())

# Scale numeric features
scaler = StandardScaler()
df_model[feature_cols] = scaler.fit_transform(df_model[feature_cols])

# Split X and y for downstream scripts
y = pd.to_numeric(df_model[target_col], errors="coerce").fillna(0).astype(int)
X = df_model[feature_cols]

print("\n" + "=" * 50)
print("Modeling preparation complete.")
print(f"X shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"Target  : {target_col}")
