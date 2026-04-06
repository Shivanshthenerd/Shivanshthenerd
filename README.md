# Insurance Churn Modeling Pipeline (India-focused)

This project builds churn prediction features and models from three input datasets:

- `data/medical_insurance_premium.csv`
- `data/insurance_claims.csv`
- `data/policy.csv`

## 1) Target dataset contract

### Premium dataset (`medical_insurance_premium.csv`)
Required columns:
- `PolicyID` (join key)
- `CustomerID` (join key)
- `Age`, `Gender`, `Region`, `BMI`, `BloodPressure`, `Smoker`, `PreExistingConditions`
- `AnnualPremium`, `PremiumType`, `InsurancePlan`

### Claims dataset (`insurance_claims.csv`)
Required columns:
- `ClaimID`
- `PolicyID` (join key)
- `CustomerID` (join key)
- `ClaimDate`, `ClaimAmount`, `ClaimStatus`, `SettlementAmount`

### Policy dataset (`policy.csv`)
Required columns:
- `PolicyID` (join key)
- `CustomerID` (join key)
- `PolicyStartDate`, `PolicyEndDate`, `PolicyType`, `SumInsured`, `RenewalStatus`
- `ChurnLabel` (target: 0/1)

## 2) Approved source shortlist for larger Indian real-life data

Use only datasets with Indian geography, policy+claims+premium signal, and enough scale/recentness.

- IRDAI and other Indian regulator/open-data portals (policy issuance, claims trends, renewals)
- Kaggle India health insurance policy/claim datasets (must include policy and claims link keys)
- Indian hospital claims datasets with policy references or joinable customer/policy IDs

Before onboarding a source, confirm:
- Indian states/cities/hospital context exists
- joinable policy/customer IDs are present or can be mapped
- row count and time coverage are materially larger than current sample

## 3) Source-to-schema mapping sheet

Maintain a mapping sheet (per source) to internal canonical names used by scripts:

- `policyid`, `customerid`, `claimid`, `claimdate`, `claimamount`, `claimstatus`, `settlementamount`
- `policystartdate`, `policyenddate`, `policytype`, `suminsured`, `renewalstatus`, `churnlabel`
- `age`, `gender`, `region`, `bmi`, `bloodpressure`, `smoker`, `preexistingconditions`, `annualpremium`, `premiumtype`, `insuranceplan`

Normalization rules to apply in mapping:
- currency normalization to numeric INR (strip `₹`, commas, text)
- date normalization to `YYYY-MM-DD`
- status normalization (`approved/rejected/pending`; `renewed/lapsed/pending`)
- missing join-key rows dropped (`policyid`, `customerid`)

## 4) Ingestion/cleaning rules implemented

`clean_datasets.py` now does contract-driven normalization:
- column alias normalization (e.g., `policy_no`→`policyid`, `sum_insured`→`suminsured`, `churn`→`churnlabel`)
- required-column validation by dataset type (`premium`, `claims`, `policy`)
- numeric cleanup for Indian currency formatting and invalid values
- date parsing with coercion and standard output format
- claim/renewal status canonicalization
- duplicate removal, null filling, and one-path loading helper (`load_and_clean_datasets`)

## 5) Merge + feature-engineering robustness

- one-policy grain enforced (`policyid` dedup keep latest)
- premium dedup by `policyid`
- robust claims aggregation for many-claims-per-policy
- invalid numeric values clipped/filled before feature creation
- date coercion safeguards for policy duration and trend features

## 6) Exact pipeline run order

Run in this order:

1. `python3 clean_datasets.py`
2. `python3 merge_datasets.py`
3. `python3 feature_engineering.py`
4. `python3 prepare_for_modeling.py`
5. `python3 save_processed_dataset.py`
6. `python3 split_processed_dataset.py`
7. `python3 train_logistic_regression.py`
8. `python3 train_random_forest.py`
9. `python3 train_xgboost.py`
10. `python3 train_pytorch_churn.py`

## 7) Validation checklist after data refresh

After loading bigger datasets:
- ensure `processed_features.csv` has zero nulls
- verify class distribution (`churn`) is reasonable and not collapsed
- compare key feature distributions and target balance vs baseline
- retrain all models and compare F1 / ROC-AUC against baseline metrics
