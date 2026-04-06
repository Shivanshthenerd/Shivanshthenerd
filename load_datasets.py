from clean_datasets import load_and_clean_datasets


df_premium, df_claims, df_policy = load_and_clean_datasets("data")

print("=== Medical Insurance Premium Dataset ===")
print(f"Shape: {df_premium.shape}")
print(f"Columns: {df_premium.columns.tolist()}")

print("\n=== Insurance Claims Dataset ===")
print(f"Shape: {df_claims.shape}")
print(f"Columns: {df_claims.columns.tolist()}")

print("\n=== Policy Dataset ===")
print(f"Shape: {df_policy.shape}")
print(f"Columns: {df_policy.columns.tolist()}")
