from pathlib import Path

import pandas as pd


DATA_DIR = Path("data")
LEGACY_PREMIUM_PATH = DATA_DIR / "medical_insurance_premium.csv"
LEGACY_CLAIMS_PATH = DATA_DIR / "insurance_claims.csv"
LEGACY_POLICY_PATH = DATA_DIR / "policy.csv"
KAGGLE_RAW_DIR = DATA_DIR / "kaggle_raw"
KAGGLE_TELCO_CANDIDATES = [
    KAGGLE_RAW_DIR / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    KAGGLE_RAW_DIR / "Telco-Customer-Churn.csv",
]


def clean_df(df: pd.DataFrame, name: str) -> pd.DataFrame:
    print(f"\n{'=' * 50}")
    print(f"Cleaning: {name}")
    print(f"  Shape before cleaning : {df.shape}")
    print(f"  Null counts before    :\n{df.isnull().sum().to_string()}")

    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )

    before_dedup = len(df)
    df = df.drop_duplicates()
    dropped = before_dedup - len(df)
    if dropped:
        print(f"  Duplicates removed    : {dropped}")

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


def _kaggle_telco_path() -> Path | None:
    for p in KAGGLE_TELCO_CANDIDATES:
        if p.exists():
            return p
    return None


def _require_columns(df: pd.DataFrame, required: set[str], dataset_name: str) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise KeyError(
            f"{dataset_name} missing required columns: {missing}"
        )


def _map_payment_to_region(payment_method: str) -> str:
    mapping = {
        "electronic check": "karnataka",
        "mailed check": "gujarat",
        "bank transfer (automatic)": "maharashtra",
        "credit card (automatic)": "delhi",
    }
    return mapping.get(str(payment_method).strip().lower(), "rajasthan")


def _telco_to_legacy_tables(df_telco: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_telco = df_telco.copy()
    df_telco.columns = (
        df_telco.columns
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
    )
    _require_columns(
        df_telco,
        {
            "customerid",
            "gender",
            "seniorcitizen",
            "dependents",
            "tenure",
            "contract",
            "paperlessbilling",
            "paymentmethod",
            "monthlycharges",
            "totalcharges",
            "internetservice",
            "churn",
        },
        "Kaggle Telco dataset",
    )

    df_telco["totalcharges"] = pd.to_numeric(df_telco["totalcharges"], errors="coerce")
    df_telco["totalcharges"] = df_telco["totalcharges"].fillna(df_telco["monthlycharges"] * df_telco["tenure"])

    n = len(df_telco)
    policy_ids = [f"P{i:06d}" for i in range(1, n + 1)]
    reference_date = pd.Timestamp("2024-12-31")
    tenure_days = (df_telco["tenure"].astype(float) * 30).astype(int).clip(lower=30)
    policy_start = reference_date - pd.to_timedelta(tenure_days, unit="D")
    churn_flag = (df_telco["churn"].astype(str).str.strip().str.lower() == "yes").astype(int)
    renewal_status = churn_flag.map({1: "Not Renewed", 0: "Renewed"})
    policy_type = df_telco["contract"].astype(str).str.lower().map(
        {"month-to-month": "Individual", "one year": "Family Floater", "two year": "Family Floater"}
    ).fillna("Individual")
    premium_type = df_telco["contract"].astype(str).str.lower().map(
        {"month-to-month": "Monthly", "one year": "Annual", "two year": "Annual"}
    ).fillna("Monthly")
    insurance_plan = df_telco["internetservice"].astype(str).str.lower().map(
        {"fiber optic": "Gold", "dsl": "Silver", "no": "Bronze"}
    ).fillna("Silver")

    df_policy = pd.DataFrame(
        {
            "policyid": policy_ids,
            "customerid": df_telco["customerid"].astype(str),
            "age": 30 + df_telco["seniorcitizen"].astype(int) * 25 + (df_telco["tenure"] // 12).astype(int),
            "gender": df_telco["gender"].astype(str).replace({"Male": "Male", "Female": "Female"}),
            "bmi": (18.5 + (df_telco["monthlycharges"] / 10.0)).clip(upper=40),
            "smoker": df_telco["internetservice"].astype(str).str.lower().eq("fiber optic").map({True: "Yes", False: "No"}),
            "region": df_telco["paymentmethod"].apply(_map_payment_to_region),
            "policytype": policy_type,
            "suminsured": (df_telco["monthlycharges"] * 120).round(2).clip(lower=100000),
            "policystartdate": policy_start.dt.strftime("%Y-%m-%d"),
            "policyenddate": (reference_date + pd.to_timedelta(365, unit="D")).strftime("%Y-%m-%d"),
            "agent": df_telco["paymentmethod"].astype(str).str.contains("automatic", case=False).map({True: "DigitalAgent", False: "BranchAgent"}),
            "channel": df_telco["paperlessbilling"].astype(str).str.lower().map({"yes": "Online", "no": "Offline"}).fillna("Online"),
            "premiumtype": premium_type,
            "insuranceplan": insurance_plan,
            "bloodpressure": df_telco["seniorcitizen"].astype(int).map({1: "High", 0: "Normal"}),
            "preexistingconditions": df_telco["dependents"].astype(str).str.lower().map({"yes": "Yes", "no": "No"}).fillna("No"),
            "renewalstatus": renewal_status,
            "churnlabel": churn_flag,
        }
    )

    df_premium = pd.DataFrame(
        {
            "policyid": policy_ids,
            "customerid": df_telco["customerid"].astype(str),
            "annualpremium": (df_telco["monthlycharges"] * 12).round(2),
            "premiumtype": premium_type,
            "insuranceplan": insurance_plan,
        }
    )

    has_claim = (df_telco["tenure"] >= 12) | (df_telco["monthlycharges"] >= df_telco["monthlycharges"].median())
    claim_source = df_telco.loc[has_claim].copy()
    claim_policy_ids = [policy_ids[i] for i in claim_source.index]
    claim_amount = (claim_source["monthlycharges"] * 10).round(2)
    claim_status = (1 - churn_flag.loc[claim_source.index]).map({1: "Approved", 0: "Rejected"})
    settlement_multiplier = claim_status.str.lower().map({"approved": 0.85, "rejected": 0.0}).fillna(0.0)
    settlement_amount = (claim_amount * settlement_multiplier).round(2)
    claim_date = (reference_date - pd.to_timedelta((claim_source["tenure"].clip(lower=1) * 15).astype(int), unit="D")).dt.strftime("%Y-%m-%d")

    df_claims = pd.DataFrame(
        {
            "claimid": [f"C{i:06d}" for i in range(1, len(claim_source) + 1)],
            "policyid": claim_policy_ids,
            "claimamount": claim_amount.values,
            "settlementamount": settlement_amount.values,
            "claimstatus": claim_status.values,
            "claimdate": claim_date.values,
        }
    )
    if df_claims.empty:
        df_claims = pd.DataFrame(columns=["claimid", "policyid", "claimamount", "settlementamount", "claimstatus", "claimdate"])

    return df_premium, df_claims, df_policy


def load_raw_datasets() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    has_legacy = LEGACY_PREMIUM_PATH.exists() and LEGACY_CLAIMS_PATH.exists() and LEGACY_POLICY_PATH.exists()
    kaggle_path = _kaggle_telco_path()

    if kaggle_path is not None:
        print(f"\nUsing Kaggle dataset: {kaggle_path}")
        df_telco = pd.read_csv(kaggle_path)
        return _telco_to_legacy_tables(df_telco)

    if has_legacy:
        print("\nUsing legacy multi-file dataset from data/")
        return (
            pd.read_csv(LEGACY_PREMIUM_PATH),
            pd.read_csv(LEGACY_CLAIMS_PATH),
            pd.read_csv(LEGACY_POLICY_PATH),
        )

    raise FileNotFoundError(
        "No supported dataset source found.\n"
        "Expected either:\n"
        "  - data/kaggle_raw/WA_Fn-UseC_-Telco-Customer-Churn.csv (or Telco-Customer-Churn.csv)\n"
        "  - OR legacy files data/medical_insurance_premium.csv, data/insurance_claims.csv, data/policy.csv"
    )


df_premium_raw, df_claims_raw, df_policy_raw = load_raw_datasets()

df_premium = clean_df(df_premium_raw, "Medical Insurance Premium")
df_claims = clean_df(df_claims_raw, "Insurance Claims")
df_policy = clean_df(df_policy_raw, "Policy")

print("\n" + "=" * 50)
print("Cleaned column names:")
print(f"  df_premium : {df_premium.columns.tolist()}")
print(f"  df_claims  : {df_claims.columns.tolist()}")
print(f"  df_policy  : {df_policy.columns.tolist()}")
