import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

from feature_engineering import df  # noqa: F401 – brings in fully-engineered df

# ── Work on a modeling-ready copy ────────────────────────────────────────────
df_model = df.copy()

# Drop identifier and raw date columns that carry no predictive signal
DROP_COLS = ["policyid", "customerid", "policystartdate", "policyenddate",
             "agent", "channel"]
df_model = df_model.drop(columns=[c for c in DROP_COLS if c in df_model.columns])

# ── 1. Derive city_tier from region ──────────────────────────────────────────
#    Mirrors the tier logic in feature_engineering.py (IRDAI city-tier proxy):
#      1 – Tier-1 metros, 2 – Tier-2 cities, 3 – everything else
TIER_1_REGIONS = {"delhi", "maharashtra", "karnataka", "tamil nadu"}
TIER_2_REGIONS = {"gujarat", "rajasthan", "telangana", "west bengal", "punjab"}


def _city_tier(region: str) -> int:
    r = str(region).strip().lower()
    if r in TIER_1_REGIONS:
        return 1
    if r in TIER_2_REGIONS:
        return 2
    return 3


df_model["city_tier"] = df_model["region"].apply(_city_tier)

# ── 2. Encode categorical variables ──────────────────────────────────────────
#    Columns explicitly called out in the task, plus any remaining object columns
#    (gender, premiumtype, insuranceplan, bloodpressure, preexistingconditions,
#     renewalstatus) that must be numeric before fitting a model.
CATEGORICAL_COLS = [
    "region",
    "smoker",
    "city_tier",     # already int but ordinal – encode for uniformity
    "policytype",
    "gender",
    "premiumtype",
    "insuranceplan",
    "bloodpressure",
    "preexistingconditions",
    "renewalstatus",
]
# Keep only columns that are actually present in df_model
CATEGORICAL_COLS = [c for c in CATEGORICAL_COLS if c in df_model.columns]

label_encoders: dict[str, LabelEncoder] = {}
for col in CATEGORICAL_COLS:
    le = LabelEncoder()
    df_model[col] = le.fit_transform(df_model[col].astype(str))
    label_encoders[col] = le

# ── 3. Normalize numerical features ──────────────────────────────────────────
#    Core columns called out in the task + the engineered numeric features.
NUMERIC_COLS = [
    # Raw
    "age",
    "bmi",
    "annualpremium",
    "suminsured",
    # Claim aggregates
    "total_claims",
    "avg_claim_amount",
    "total_settlement_amount",
    "approved_claims",
    # Engineered features
    "policy_duration_days",
    "claim_frequency_ratio",
    "claim_approval_rate",
    "premium_to_income_proxy",
    "waiting_period_remaining",
    "claim_experience_score",
    "no_claim_years",
    "engagement_score",
]
# Keep only columns that are actually present in df_model
NUMERIC_COLS = [c for c in NUMERIC_COLS if c in df_model.columns]

scaler = StandardScaler()
df_model[NUMERIC_COLS] = scaler.fit_transform(df_model[NUMERIC_COLS])

# ── 4. Separate features (X) and target (y) ──────────────────────────────────
#    Target: churnlabel  (0 = not churned, 1 = churned)
#    Fallback: if churnlabel is absent, derive from renewalstatus encoded value.
TARGET_COL = "churnlabel"
if TARGET_COL not in df_model.columns:
    raise KeyError(f"Target column '{TARGET_COL}' not found in df_model. "
                   "Check that the policy dataset contains a ChurnLabel column.")

y = df_model[TARGET_COL].astype(int)
X = df_model.drop(columns=[TARGET_COL])

if y.isnull().any():
    raise ValueError("Target contains null values after preprocessing.")

target_values = set(y.unique().tolist())
if not target_values.issubset({0, 1}):
    raise ValueError(
        f"Target must be binary 0/1, found values: {sorted(target_values)}"
    )

if X.isnull().sum().sum() > 0:
    raise ValueError("Features contain null values after preprocessing.")

class_counts = y.value_counts()
if len(class_counts) < 2:
    raise ValueError("Target must contain at least two classes for stratified split.")
if class_counts.min() < 2:
    raise ValueError(
        f"Each class must have at least 2 samples for stratified split. Counts: {class_counts.to_dict()}"
    )

# ── Summary ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 50)
print("Modeling preparation complete.")
print(f"\nX shape : {X.shape}")
print(f"y shape : {y.shape}")
print(f"\nFeature columns ({len(X.columns)}):")
for col in X.columns:
    print(f"  {col}")
print(f"\nTarget distribution:\n{y.value_counts().to_string()}")
print("\nX preview (first 5 rows):")
print(X.head().to_string(index=False))
