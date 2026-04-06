import pandas as pd

# Load datasets
df_premium = pd.read_csv("data/medical_insurance_premium.csv")
df_claims = pd.read_csv("data/insurance_claims.csv")
df_policy = pd.read_csv("data/policy.csv")

# --- Medical Insurance Premium Dataset ---
print("=== Medical Insurance Premium Dataset ===")
print(f"Shape: {df_premium.shape}")
print(f"Columns: {df_premium.columns.tolist()}")

# --- Insurance Claims Dataset ---
print("\n=== Insurance Claims Dataset ===")
print(f"Shape: {df_claims.shape}")
print(f"Columns: {df_claims.columns.tolist()}")

# --- Policy Dataset ---
print("\n=== Policy Dataset ===")
print(f"Shape: {df_policy.shape}")
print(f"Columns: {df_policy.columns.tolist()}")
