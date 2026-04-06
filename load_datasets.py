from pathlib import Path
from zipfile import ZipFile
import xml.etree.ElementTree as ET

import pandas as pd


DATA_DIR = Path("data")
DATASETS = {
    "indian_insurance": DATA_DIR / "Indian_Insurance_Data.csv",
    "medicalpremium": DATA_DIR / "Medicalpremium.csv",
    "dataset11": DATA_DIR / "dataset11.xlsx",
    "handbook": DATA_DIR / "Hand Book 2021-22_Part III_Health Insurance.xlsx",
}

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def _xlsx_sheet_row_counts(path: Path) -> dict[str, int]:
    with ZipFile(path) as zf:
        sheets = sorted(name for name in zf.namelist() if name.startswith("xl/worksheets/sheet"))
        out = {}
        for sheet in sheets:
            root = ET.fromstring(zf.read(sheet))
            out[sheet] = len(root.findall(".//a:sheetData/a:row", NS))
        return out


for name, path in DATASETS.items():
    print(f"\n=== {name} ===")
    if not path.exists():
        print(f"Missing: {path}")
        continue

    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
        print(f"File: {path}")
        print(f"Shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()}")
    else:
        counts = _xlsx_sheet_row_counts(path)
        print(f"File: {path}")
        print(f"Sheets: {len(counts)}")
        for sheet, rows in list(counts.items())[:5]:
            print(f"  {sheet}: {rows} rows")
