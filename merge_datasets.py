import pandas as pd

from clean_datasets import load_and_clean_datasets

# ── Load & clean datasets ─────────────────────────────────────────────────────
df_premium, df_claims, df_policy = load_and_clean_datasets("data")

# Enforce one-policy grain for policy and premium tables.
df_policy = (
    df_policy.sort_values(["policyid", "policyenddate"]) if "policyenddate" in df_policy.columns else df_policy
)
df_policy = df_policy.drop_duplicates(subset=["policyid"], keep="last")

df_premium = df_premium.drop_duplicates(subset=["policyid"], keep="last")

# Guard invalid claim amounts and statuses.
for col in ["claimamount", "settlementamount"]:
    if col in df_claims.columns:
        df_claims[col] = pd.to_numeric(df_claims[col], errors="coerce").fillna(0).clip(lower=0)
if "claimstatus" in df_claims.columns:
    df_claims["claimstatus"] = df_claims["claimstatus"].astype(str).str.lower().fillna("pending")

# ── Aggregate claims per policy ───────────────────────────────────────────────
agg_spec = {
    "total_claims": ("claimid", "nunique") if "claimid" in df_claims.columns else ("policyid", "count"),
    "avg_claim_amount": ("claimamount", "mean"),
    "total_settlement_amount": ("settlementamount", "sum"),
}
if "claimstatus" in df_claims.columns:
    agg_spec["approved_claims"] = ("claimstatus", lambda s: (s == "approved").sum())

df_claims_agg = df_claims.groupby("policyid", as_index=False).agg(**agg_spec)

# ── Merge all datasets ────────────────────────────────────────────────────────
df = pd.merge(df_policy, df_premium, on="policyid", how="left", suffixes=("", "_prem"))

# Drop duplicate customerid and non-canonical overlap cols introduced by merge.
dup_cols = [c for c in df.columns if c.endswith("_prem")]
if "customerid_prem" in dup_cols:
    dup_cols.remove("customerid_prem")
if "customerid_prem" in df.columns and "customerid" in df.columns:
    df["customerid"] = df["customerid"].fillna(df["customerid_prem"])
    dup_cols.append("customerid_prem")
if dup_cols:
    df = df.drop(columns=dup_cols)

df = pd.merge(df, df_claims_agg, on="policyid", how="left")

# ── Handle missing values after merge ─────────────────────────────────────────
for col in ["total_claims", "avg_claim_amount", "total_settlement_amount", "approved_claims"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

for col in ["annualpremium", "suminsured"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median() if not df[col].dropna().empty else 0)
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Final merged dataframe (df):")
print(f"  Shape   : {df.shape}")
print(f"  Columns : {df.columns.tolist()}")
print(f"  Nulls   :\n{df.isnull().sum().to_string()}")
print("\nFirst few rows:")
print(df.head().to_string(index=False))
