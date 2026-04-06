import re
from typing import Literal

import pandas as pd

DatasetKind = Literal["premium", "claims", "policy"]

DATASET_CONTRACT = {
    "premium": {
        "required": [
            "policyid",
            "customerid",
            "age",
            "gender",
            "region",
            "bmi",
            "bloodpressure",
            "smoker",
            "preexistingconditions",
            "annualpremium",
            "premiumtype",
            "insuranceplan",
        ],
        "date_cols": [],
        "numeric_cols": ["age", "bmi", "annualpremium"],
    },
    "claims": {
        "required": [
            "claimid",
            "policyid",
            "customerid",
            "claimdate",
            "claimamount",
            "claimstatus",
            "settlementamount",
        ],
        "date_cols": ["claimdate"],
        "numeric_cols": ["claimamount", "settlementamount"],
    },
    "policy": {
        "required": [
            "policyid",
            "customerid",
            "policystartdate",
            "policyenddate",
            "policytype",
            "suminsured",
            "renewalstatus",
            "churnlabel",
        ],
        "date_cols": ["policystartdate", "policyenddate"],
        "numeric_cols": ["suminsured", "churnlabel"],
    },
}

COLUMN_ALIASES = {
    "policy_id": "policyid",
    "policy_no": "policyid",
    "policy_number": "policyid",
    "policynumber": "policyid",
    "customer_id": "customerid",
    "cust_id": "customerid",
    "customer_number": "customerid",
    "churn": "churnlabel",
    "churn_flag": "churnlabel",
    "churnstatus": "churnlabel",
    "claim_date": "claimdate",
    "claim_amount": "claimamount",
    "claim_status": "claimstatus",
    "settled_amount": "settlementamount",
    "settlement_amount": "settlementamount",
    "sum_insured": "suminsured",
    "sum_assured": "suminsured",
    "premium_amount": "annualpremium",
    "annual_premium": "annualpremium",
    "policy_start_date": "policystartdate",
    "policy_end_date": "policyenddate",
    "renewal_status": "renewalstatus",
    "pre_existing_conditions": "preexistingconditions",
    "blood_pressure": "bloodpressure",
    "insurance_plan": "insuranceplan",
    "premium_type": "premiumtype",
}

CLAIM_STATUS_MAP = {
    "approved": "approved",
    "accept": "approved",
    "accepted": "approved",
    "settled": "approved",
    "rejected": "rejected",
    "declined": "rejected",
    "denied": "rejected",
    "pending": "pending",
    "under process": "pending",
    "in review": "pending",
    "processing": "pending",
}

RENEWAL_STATUS_MAP = {
    "renewed": "renewed",
    "active": "renewed",
    "inforce": "renewed",
    "lapsed": "lapsed",
    "expired": "lapsed",
    "cancelled": "lapsed",
    "pending": "pending",
    "due": "pending",
}


NUMERIC_CLIP_NONNEGATIVE = {
    "age",
    "bmi",
    "annualpremium",
    "claimamount",
    "settlementamount",
    "suminsured",
}
DATETIME_FALLBACK = pd.Timestamp("2000-01-01")  # conservative sentinel for entirely invalid date columns


def _normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _standardize_column_name(col: str) -> str:
    normalized = (
        str(col)
        .strip()
        .lower()
        .replace("%", "percent")
    )
    normalized = re.sub(r"[^a-z0-9]+", "_", normalized)
    normalized = normalized.strip("_")
    return COLUMN_ALIASES.get(normalized, normalized)


def _looks_day_first(series: pd.Series) -> bool:
    non_null = series.dropna().astype(str).head(20)
    for val in non_null:
        m = re.match(r"^\s*(\d{1,2})[-/](\d{1,2})[-/]\d{2,4}\s*$", val)
        if not m:
            continue
        first = int(m.group(1))
        second = int(m.group(2))
        if first > 12 and second <= 12:
            return True
    return False


def _to_numeric_series(series: pd.Series) -> pd.Series:
    cleaned = (
        series.astype(str)
        .str.replace(r"[₹,]", "", regex=True)
        .str.replace(r"[^0-9.\-]", "", regex=True)
        .replace({"": pd.NA, "nan": pd.NA, "none": pd.NA})
    )
    return pd.to_numeric(cleaned, errors="coerce")


def _datetime_median(series: pd.Series):
    non_null = pd.to_datetime(series.dropna(), errors="coerce").dropna().sort_values()
    if non_null.empty:
        return pd.NaT
    return non_null.iloc[len(non_null) // 2]


def _normalize_status_columns(df: pd.DataFrame) -> None:
    if "claimstatus" in df.columns:
        df["claimstatus"] = (
            df["claimstatus"]
            .map(
                lambda x: (
                    "pending"
                    if pd.isna(x)
                    else CLAIM_STATUS_MAP.get(_normalize_text(x), _normalize_text(x))
                )
            )
            .fillna("pending")
        )
    if "renewalstatus" in df.columns:
        df["renewalstatus"] = (
            df["renewalstatus"]
            .map(
                lambda x: (
                    "pending"
                    if pd.isna(x)
                    else RENEWAL_STATUS_MAP.get(_normalize_text(x), _normalize_text(x))
                )
            )
            .fillna("pending")
        )


def _validate_required_columns(df: pd.DataFrame, dataset_kind: DatasetKind) -> None:
    required = DATASET_CONTRACT[dataset_kind]["required"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(
            f"Missing required columns for '{dataset_kind}' dataset: {missing}. "
            f"Found columns: {df.columns.tolist()}"
        )


def clean_df(df: pd.DataFrame, dataset_kind: DatasetKind) -> pd.DataFrame:
    """Normalize dataset schema and values for premium/claims/policy inputs."""
    if dataset_kind not in DATASET_CONTRACT:
        raise ValueError(f"Unsupported dataset_kind '{dataset_kind}'.")

    print(f"\n{'=' * 50}")
    print(f"Cleaning: {dataset_kind}")
    print(f"  Shape before cleaning : {df.shape}")

    df = df.copy()
    df.columns = [_standardize_column_name(c) for c in df.columns]

    # Remove duplicate columns introduced by aliases.
    df = df.loc[:, ~df.columns.duplicated()]

    _validate_required_columns(df, dataset_kind)

    # Drop rows with missing join keys.
    before = len(df)
    df = df.dropna(subset=["policyid", "customerid"])
    dropped = before - len(df)
    if dropped:
        print(f"  Dropped rows missing join keys : {dropped}")

    # Parse numeric/date columns per contract.
    for col in DATASET_CONTRACT[dataset_kind]["numeric_cols"]:
        if col in df.columns:
            df[col] = _to_numeric_series(df[col])
            if col in NUMERIC_CLIP_NONNEGATIVE:
                df[col] = df[col].clip(lower=0)

    for col in DATASET_CONTRACT[dataset_kind]["date_cols"]:
        if col in df.columns:
            dayfirst = _looks_day_first(df[col])
            df[col] = pd.to_datetime(df[col], errors="coerce", dayfirst=dayfirst)

    _normalize_status_columns(df)

    # Remove exact duplicates.
    before_dedup = len(df)
    df = df.drop_duplicates()
    if before_dedup - len(df):
        print(f"  Duplicates removed            : {before_dedup - len(df)}")

    # Fill nulls.
    for col in df.columns:
        if not df[col].isnull().any():
            continue
        if pd.api.types.is_numeric_dtype(df[col]):
            fill_value = df[col].median() if not df[col].dropna().empty else 0
            df[col] = df[col].fillna(fill_value)
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            fill_value = _datetime_median(df[col])
            if pd.notna(fill_value):
                df[col] = df[col].fillna(fill_value)
            else:
                df[col] = df[col].fillna(DATETIME_FALLBACK)
        else:
            mode = df[col].mode(dropna=True)
            df[col] = df[col].fillna(mode.iloc[0] if not mode.empty else "unknown")

    print(f"  Shape after cleaning  : {df.shape}")
    print(f"  Null counts after     :\n{df.isnull().sum().to_string()}")
    return df


def load_and_clean_datasets(base_path: str = "data") -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    df_premium = clean_df(pd.read_csv(f"{base_path}/medical_insurance_premium.csv"), "premium")
    df_claims = clean_df(pd.read_csv(f"{base_path}/insurance_claims.csv"), "claims")
    df_policy = clean_df(pd.read_csv(f"{base_path}/policy.csv"), "policy")
    return df_premium, df_claims, df_policy


if __name__ == "__main__":
    premium, claims, policy = load_and_clean_datasets("data")
    print("\n" + "=" * 50)
    print("Cleaned column names:")
    print(f"  premium : {premium.columns.tolist()}")
    print(f"  claims  : {claims.columns.tolist()}")
    print(f"  policy  : {policy.columns.tolist()}")
