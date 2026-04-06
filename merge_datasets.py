from feature_engineering import df

print("\n" + "=" * 50)
print("Merged dataframe built from the 4 new datasets")
print(f"Shape   : {df.shape}")
print(f"Columns : {df.columns.tolist()}")
print(f"Nulls   :\n{df.isnull().sum().to_string()}")
print("\nFirst few rows:")
print(df.head().to_string(index=False))
