import pandas as pd

from clean_datasets import df_claims, df_policy, df_premium


def _require_columns(df: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(f"{name} missing required columns: {missing}")


_require_columns(df_policy, {"policyid"}, "df_policy")
_require_columns(df_premium, {"policyid"}, "df_premium")
_require_columns(df_claims, {"policyid", "claimid", "claimamount", "settlementamount"}, "df_claims")

# ── Aggregate claims per policy ───────────────────────────────────────────────
# df_claims has multiple rows per policyid; summarise to one row per policy.
df_claims_agg = (
    df_claims
    .groupby("policyid", as_index=False)
    .agg(
        total_claims=("claimid", "count"),
        avg_claim_amount=("claimamount", "mean"),
        total_settlement_amount=("settlementamount", "sum"),
    )
)

# ── Merge all datasets ────────────────────────────────────────────────────────
# Base: df_policy (one row per policy / customer)
# Step 1: add user-level premium info
df = pd.merge(df_policy, df_premium, on="policyid", how="left", suffixes=("", "_prem"))

# Drop duplicate customerid column introduced by the merge (if present)
dup_cols = [c for c in df.columns if c.endswith("_prem")]
df = df.drop(columns=dup_cols)

# Step 2: add aggregated claims info
df = pd.merge(df, df_claims_agg, on="policyid", how="left")

# ── Handle missing values after merge ─────────────────────────────────────────
# Policies with no claims → fill aggregated claim columns with 0
for col in ["total_claims", "avg_claim_amount", "total_settlement_amount"]:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Any remaining nulls: numeric → mean, categorical → mode
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Final merged dataframe (df):")
print(f"  Shape   : {df.shape}")
print(f"  Columns : {df.columns.tolist()}")
print(f"  Nulls   :\n{df.isnull().sum().to_string()}")
print("\nFirst few rows:")
print(df.head().to_string(index=False))
