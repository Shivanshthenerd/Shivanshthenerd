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

# ── India-specific features ───────────────────────────────────────────────────

# 5. premium_to_income_proxy
#    Estimate a monthly income proxy from city tier (derived from region) and age.
#    City tier mapping based on Indian metro / tier-2 / tier-3 classification:
#      Tier 1 (metros)     – Delhi, Maharashtra, Karnataka, Tamil Nadu
#      Tier 2 (mid-cities) – Gujarat, Rajasthan, Telangana, West Bengal, Punjab
#      Tier 3 (others)     – everything else (smaller states / UTs)
TIER_1_REGIONS = {"delhi", "maharashtra", "karnataka", "tamil nadu"}
TIER_2_REGIONS = {"gujarat", "rajasthan", "telangana", "west bengal", "punjab"}

TIER_BASE_INCOME = {1: 800_000, 2: 500_000, 3: 300_000}   # annual ₹ estimate


def _estimate_income(region: str, age: float) -> float:
    """Return a simple estimated annual income based on city tier and age."""
    tier = (
        1 if str(region).strip().lower() in TIER_1_REGIONS
        else 2 if str(region).strip().lower() in TIER_2_REGIONS
        else 3
    )
    base = TIER_BASE_INCOME[tier]
    # Age factor: income ramps up through mid-career then plateaus
    if age < 25:
        age_factor = 0.6
    elif age < 35:
        age_factor = 0.85
    elif age < 50:
        age_factor = 1.0
    else:
        age_factor = 0.9
    return base * age_factor


df["estimated_income"] = df.apply(
    lambda row: _estimate_income(row["region"], row["age"]), axis=1
)
# Guard: estimated_income should always be > 0 given the mapping above,
# but add a safety net just in case.
df["premium_to_income_proxy"] = df.apply(
    lambda row: row["annualpremium"] / row["estimated_income"]
    if row["estimated_income"] > 0
    else 0.0,
    axis=1,
)
df = df.drop(columns=["estimated_income"])   # helper column; not a final feature

# 6. waiting_period_remaining
#    Standard IRDAI max waiting period for pre-existing diseases = 4 years (1 460 days).
#    Once the policy has been active for >= 4 years the waiting period is fully served.
MAX_WAITING_PERIOD_DAYS = 4 * 365   # 1 460 days

df["waiting_period_remaining"] = (MAX_WAITING_PERIOD_DAYS - df["policy_duration_days"]).clip(lower=0)

# 7. family_floater_flag
#    PolicyType values in this dataset: "Individual", "Family Floater"
df["family_floater_flag"] = df["policytype"].str.lower().str.contains("family").astype(int)

# 8. claim_experience_score  (0 – 100; higher = better claim experience)
#    Three penalty components (all bounded to [0, 1] before scaling):
#      a. Rejection penalty  = (1 – claim_approval_rate)          → weight 40
#      b. Amount intensity   = avg_claim_amount / suminsured       → weight 40
#      c. Frequency penalty  = claim_frequency_ratio * 365        → weight 20
#    For policies with no claims the score is 100 (neutral; no adverse experience).


def _claim_experience_score(row) -> float:
    if row["total_claims"] == 0:
        return 100.0
    rejection_penalty = (1.0 - row["claim_approval_rate"]) * 40.0
    amount_intensity = min(row["avg_claim_amount"] / row["suminsured"], 1.0) * 40.0 if row["suminsured"] > 0 else 0.0
    freq_penalty = min(row["claim_frequency_ratio"] * 365.0, 1.0) * 20.0
    return max(0.0, 100.0 - rejection_penalty - amount_intensity - freq_penalty)


df["claim_experience_score"] = df.apply(_claim_experience_score, axis=1)

# ── Summary ───────────────────────────────────────────────────────────────────
new_features = [
    "policy_duration_days",
    "claim_frequency_ratio",
    "avg_claim_amount",
    "claim_approval_rate",
    "premium_to_income_proxy",
    "waiting_period_remaining",
    "family_floater_flag",
    "claim_experience_score",
]

print("\n" + "=" * 50)
print("Feature engineering complete. New / confirmed features:")
for feat in new_features:
    print(f"  {feat}")

print(f"\ndf shape : {df.shape}")
print(f"Nulls    :\n{df[new_features].isnull().sum().to_string()}")
print("\nFeature preview:")
print(df[["policyid"] + new_features].to_string(index=False))
