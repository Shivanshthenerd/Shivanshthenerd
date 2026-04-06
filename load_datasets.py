from clean_datasets import df_claims, df_policy, df_premium

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
