from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd


DATA_DIR = Path("data")
INDIAN_DATA_PATH = DATA_DIR / "Indian_Insurance_Data.csv"
MEDICAL_PREMIUM_PATH = DATA_DIR / "Medicalpremium.csv"
DATASET11_PATH = DATA_DIR / "dataset11.xlsx"
HANDBOOK_PATH = DATA_DIR / "Hand Book 2021-22_Part III_Health Insurance.xlsx"

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


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


def _xlsx_row_counts(path: Path) -> dict[str, int]:
    with ZipFile(path) as zf:
        sheets = sorted(name for name in zf.namelist() if name.startswith("xl/worksheets/sheet"))
        out = {}
        for sheet in sheets:
            root = ET.fromstring(zf.read(sheet))
            out[sheet] = len(root.findall(".//a:sheetData/a:row", NS))
        return out


# Load & clean CSV datasets from the new dataset set
indian_df = clean_df(pd.read_csv(INDIAN_DATA_PATH), "Indian Insurance Data")
medical_df = clean_df(pd.read_csv(MEDICAL_PREMIUM_PATH), "Medical Premium")

# Validate workbook availability from the new dataset set
dataset11_rows = _xlsx_row_counts(DATASET11_PATH)
handbook_rows = _xlsx_row_counts(HANDBOOK_PATH)

print("\n" + "=" * 50)
print("New dataset validation summary:")
print(f"  Indian Insurance Data      : {indian_df.shape}")
print(f"  Medical Premium            : {medical_df.shape}")
print(f"  dataset11.xlsx sheets      : {len(dataset11_rows)}")
print(f"  handbook.xlsx sheets       : {len(handbook_rows)}")
