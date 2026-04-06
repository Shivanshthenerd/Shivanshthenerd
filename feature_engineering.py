import pandas as pd

from clean_datasets import load_and_clean_datasets

# ── Re-load & clean datasets ──────────────────────────────────────────────────
df_premium, df_claims, df_policy = load_and_clean_datasets("data")

df_policy = (
    df_policy.sort_values(["policyid", "policyenddate"]) if "policyenddate" in df_policy.columns else df_policy
)
df_policy = df_policy.drop_duplicates(subset=["policyid"], keep="last")
df_premium = df_premium.drop_duplicates(subset=["policyid"], keep="last")

# Normalize amounts and claim status.
for col in ["claimamount", "settlementamount"]:
    if col in df_claims.columns:
        df_claims[col] = pd.to_numeric(df_claims[col], errors="coerce").fillna(0).clip(lower=0)
if "claimstatus" in df_claims.columns:
    df_claims["claimstatus"] = df_claims["claimstatus"].astype(str).str.lower().fillna("pending")

# ── Aggregate claims per policy ───────────────────────────────────────────────
df_claims_agg = (
    df_claims
    .groupby("policyid", as_index=False)
    .agg(
        total_claims=("claimid", "nunique") if "claimid" in df_claims.columns else ("policyid", "count"),
        avg_claim_amount=("claimamount", "mean"),
        total_settlement_amount=("settlementamount", "sum"),
        approved_claims=("claimstatus", lambda s: (s == "approved").sum()) if "claimstatus" in df_claims.columns else ("policyid", "count"),
    )
)

# ── Merge all datasets ────────────────────────────────────────────────────────
df = pd.merge(df_policy, df_premium, on="policyid", how="left", suffixes=("", "_prem"))
if "customerid_prem" in df.columns and "customerid" in df.columns:
    df["customerid"] = df["customerid"].fillna(df["customerid_prem"])
df = df.drop(columns=[c for c in df.columns if c.endswith("_prem")], errors="ignore")
df = pd.merge(df, df_claims_agg, on="policyid", how="left")

for col in ["total_claims", "avg_claim_amount", "total_settlement_amount", "approved_claims"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)
for col in ["annualpremium", "suminsured", "age", "bmi"]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).clip(lower=0)

for col in df.columns:
    if df[col].isnull().any():
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median() if not df[col].dropna().empty else 0)
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")

# ── Feature engineering ───────────────────────────────────────────────────────
df["policystartdate"] = pd.to_datetime(df["policystartdate"], errors="coerce")
df["policyenddate"] = pd.to_datetime(df["policyenddate"], errors="coerce")
df["policy_duration_days"] = (df["policyenddate"] - df["policystartdate"]).dt.days
df["policy_duration_days"] = pd.to_numeric(df["policy_duration_days"], errors="coerce").fillna(0).clip(lower=0)

df["claim_frequency_ratio"] = df.apply(
    lambda row: row["total_claims"] / row["policy_duration_days"]
    if row["policy_duration_days"] > 0
    else 0.0,
    axis=1,
)

df["claim_approval_rate"] = df.apply(
    lambda row: row["approved_claims"] / row["total_claims"] if row["total_claims"] > 0 else 0.0,
    axis=1,
)

TIER_1_REGIONS = {"delhi", "maharashtra", "karnataka", "tamil nadu"}
TIER_2_REGIONS = {"gujarat", "rajasthan", "telangana", "west bengal", "punjab"}
TIER_BASE_INCOME = {1: 800_000, 2: 500_000, 3: 300_000}


def _estimate_income(region: str, age: float) -> float:
    tier = (
        1 if str(region).strip().lower() in TIER_1_REGIONS
        else 2 if str(region).strip().lower() in TIER_2_REGIONS
        else 3
    )
    base = TIER_BASE_INCOME[tier]
    if age < 25:
        age_factor = 0.6
    elif age < 35:
        age_factor = 0.85
    elif age < 50:
        age_factor = 1.0
    else:
        age_factor = 0.9
    return base * age_factor


df["estimated_income"] = df.apply(lambda row: _estimate_income(row["region"], row["age"]), axis=1)
df["premium_to_income_proxy"] = df.apply(
    lambda row: row["annualpremium"] / row["estimated_income"] if row["estimated_income"] > 0 else 0.0,
    axis=1,
)
df = df.drop(columns=["estimated_income"])

MAX_WAITING_PERIOD_DAYS = 4 * 365
df["waiting_period_remaining"] = (MAX_WAITING_PERIOD_DAYS - df["policy_duration_days"]).clip(lower=0)

df["family_floater_flag"] = df["policytype"].astype(str).str.lower().str.contains("family").astype(int)


def _claim_experience_score(row) -> float:
    if row["total_claims"] == 0:
        return 100.0
    rejection_penalty = (1.0 - row["claim_approval_rate"]) * 40.0
    amount_intensity = min(row["avg_claim_amount"] / row["suminsured"], 1.0) * 40.0 if row["suminsured"] > 0 else 0.0
    freq_penalty = min(row["claim_frequency_ratio"] * 365.0, 1.0) * 20.0
    return max(0.0, 100.0 - rejection_penalty - amount_intensity - freq_penalty)


df["claim_experience_score"] = df.apply(_claim_experience_score, axis=1)

df["policy_duration_years"] = df["policy_duration_days"] / 365.0
df["no_claim_years"] = (df["policy_duration_years"] - df["total_claims"]).clip(lower=0)
df = df.drop(columns=["policy_duration_years"])

df["renewal_flag"] = (df["renewalstatus"].astype(str).str.lower() == "renewed").astype(float)
df["engagement_score"] = (
    df["total_claims"].clip(upper=5) / 5.0 * 0.6
    + df["renewal_flag"] * 0.4
)
df = df.drop(columns=["renewal_flag"])

df_claims_sorted = (
    df_claims
    .copy()
    .assign(claimdate=lambda d: pd.to_datetime(d["claimdate"], errors="coerce"))
    .sort_values(["policyid", "claimdate"])
)


def _claim_trend(group) -> int:
    amounts = group["claimamount"].tolist()
    if len(amounts) < 2:
        return 0
    return 1 if amounts[-1] > amounts[0] else (-1 if amounts[-1] < amounts[0] else 0)


claim_trend_map = (
    df_claims_sorted
    .groupby("policyid")
    .apply(_claim_trend)
    .rename("claim_trend")
    .reset_index()
)
df = df.merge(claim_trend_map, on="policyid", how="left")
df["claim_trend"] = df["claim_trend"].fillna(0).astype(int)

premium_median = df["annualpremium"].median()
df["churn_risk_flag"] = (
    ((df["total_claims"] == 0) & (df["annualpremium"] > premium_median))
    | (df["claim_experience_score"] < 60)
).astype(int)

new_features = [
    "policy_duration_days",
    "claim_frequency_ratio",
    "avg_claim_amount",
    "claim_approval_rate",
    "premium_to_income_proxy",
    "waiting_period_remaining",
    "family_floater_flag",
    "claim_experience_score",
    "no_claim_years",
    "engagement_score",
    "claim_trend",
    "churn_risk_flag",
]

print("\n" + "=" * 50)
print("Feature engineering complete. New / confirmed features:")
for feat in new_features:
    print(f"  {feat}")

print(f"\ndf shape : {df.shape}")
print(f"Nulls    :\n{df[new_features].isnull().sum().to_string()}")
print("\nFeature preview:")
print(df[["policyid"] + new_features].to_string(index=False))
