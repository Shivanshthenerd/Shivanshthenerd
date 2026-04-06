import pandas as pd


def clean_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Clean a dataframe by:
    1. Standardising column names (lowercase, spaces → underscores)
    2. Removing duplicate rows
    3. Filling missing numerical values with the column mean
    4. Filling missing categorical values with the column mode
    """
    print(f"\n{'=' * 50}")
    print(f"Cleaning: {name}")
    print(f"  Shape before cleaning : {df.shape}")
    print(f"  Null counts before    :\n{df.isnull().sum().to_string()}")

    # 1. Standardise column names
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    # 2. Remove duplicates
    before_dedup = len(df)
    df = df.drop_duplicates()
    dropped = before_dedup - len(df)
    if dropped:
        print(f"  Duplicates removed    : {dropped}")

    # 3. Handle missing values
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                fill_value = df[col].mean()
                df[col] = df[col].fillna(fill_value)
                print(f"  Filled '{col}' (numeric) with mean = {fill_value:.4f}")
            else:
                fill_value = df[col].mode()[0]
                df[col] = df[col].fillna(fill_value)
                print(f"  Filled '{col}' (categorical) with mode = '{fill_value}'")

    print(f"  Shape after cleaning  : {df.shape}")
    print(f"  Null counts after     :\n{df.isnull().sum().to_string()}")
    return df


# ── Load datasets ────────────────────────────────────────────────────────────
df_premium = pd.read_csv("data/medical_insurance_premium.csv")
df_claims = pd.read_csv("data/insurance_claims.csv")
df_policy = pd.read_csv("data/policy.csv")

# ── Clean datasets ───────────────────────────────────────────────────────────
df_premium = clean_df(df_premium, "Medical Insurance Premium")
df_claims = clean_df(df_claims, "Insurance Claims")
df_policy = clean_df(df_policy, "Policy")

# ── Summary ──────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Cleaned column names:")
print(f"  df_premium : {df_premium.columns.tolist()}")
print(f"  df_claims  : {df_claims.columns.tolist()}")
print(f"  df_policy  : {df_policy.columns.tolist()}")
