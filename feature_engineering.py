import pandas as pd

from clean_datasets import clean_df

# ── Re-load & clean datasets (mirrors merge_datasets.py) ──────────────────────
df_premium = clean_df(pd.read_csv("data/medical_insurance_premium.csv"), "Medical Insurance Premium")
df_claims = clean_df(pd.read_csv("data/insurance_claims.csv"), "Insurance Claims")
df_policy = clean_df(pd.read_csv("data/policy.csv"), "Policy")

# ── Aggregate claims per policy ───────────────────────────────────────────────
df_claims_agg = (
    df_claims
    .groupby("policyid", as_index=False)
    .agg(
        total_claims=("claimid", "count"),
        avg_claim_amount=("claimamount", "mean"),
        total_settlement_amount=("settlementamount", "sum"),
        approved_claims=("claimstatus", lambda s: (s.str.lower() == "approved").sum()),
    )
)

# ── Merge all datasets ────────────────────────────────────────────────────────
df = pd.merge(df_policy, df_premium, on="policyid", how="left", suffixes=("", "_prem"))
df = df.drop(columns=[c for c in df.columns if c.endswith("_prem")])
df = pd.merge(df, df_claims_agg, on="policyid", how="left")

# Fill claim columns for policies with no claims
for col in ["total_claims", "avg_claim_amount", "total_settlement_amount", "approved_claims"]:
    if col in df.columns:
        df[col] = df[col].fillna(0)

# Any remaining nulls: numeric → mean, categorical → mode
for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].mean())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])

# ── Feature engineering ───────────────────────────────────────────────────────

# 1. policy_duration_days: number of days the policy was active
df["policystartdate"] = pd.to_datetime(df["policystartdate"])
df["policyenddate"] = pd.to_datetime(df["policyenddate"])
df["policy_duration_days"] = (df["policyenddate"] - df["policystartdate"]).dt.days

# 2. claim_frequency_ratio = total_claims / policy_duration_days
#    Guard against zero-duration policies (division by zero → 0)
df["claim_frequency_ratio"] = df.apply(
    lambda row: row["total_claims"] / row["policy_duration_days"]
    if row["policy_duration_days"] > 0
    else 0.0,
    axis=1,
)

# 3. avg_claim_amount is already present from the aggregation step above.
#    No recomputation needed; confirm it is in df.
assert "avg_claim_amount" in df.columns, "avg_claim_amount missing from df"

# 4. claim_approval_rate = approved_claims / total_claims
#    Guard against policies with zero claims (division by zero → 0)
df["claim_approval_rate"] = df.apply(
    lambda row: row["approved_claims"] / row["total_claims"]
    if row["total_claims"] > 0
    else 0.0,
    axis=1,
)

# ── Summary ───────────────────────────────────────────────────────────────────
new_features = [
    "policy_duration_days",
    "claim_frequency_ratio",
    "avg_claim_amount",
    "claim_approval_rate",
]

print("\n" + "=" * 50)
print("Feature engineering complete. New / confirmed features:")
for feat in new_features:
    print(f"  {feat}")

print(f"\ndf shape : {df.shape}")
print(f"Nulls    :\n{df[new_features].isnull().sum().to_string()}")
print("\nFeature preview:")
print(df[["policyid"] + new_features].to_string(index=False))
