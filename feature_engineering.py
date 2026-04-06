from pathlib import Path
from zipfile import ZipFile
import re
import xml.etree.ElementTree as ET

import pandas as pd


DATA_DIR = Path("data")
INDIAN_DATA_PATH = DATA_DIR / "Indian_Insurance_Data.csv"
MEDICAL_PREMIUM_PATH = DATA_DIR / "Medicalpremium.csv"
DATASET11_PATH = DATA_DIR / "dataset11.xlsx"
HANDBOOK_PATH = DATA_DIR / "Hand Book 2021-22_Part III_Health Insurance.xlsx"

NS = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
NUMERIC_CONVERSION_THRESHOLD = 0.8
PREMIUM_BURDEN_WEIGHT = 40.0
SMOKER_WEIGHT = 2.0
DIABETES_WEIGHT = 1.2
CHRONIC_DISEASE_WEIGHT = 1.2
PRE_EXISTING_WEIGHT = 1.0


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = (
        out.columns.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace(r"[^a-z0-9_]", "", regex=True)
    )
    return out


def _excel_col_to_idx(cell_ref: str) -> int:
    letters = "".join(ch for ch in cell_ref if ch.isalpha()).upper()
    idx = 0
    for ch in letters:
        idx = idx * 26 + (ord(ch) - ord("A") + 1)
    return max(idx - 1, 0)


def _load_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    with ZipFile(path) as zf:
        shared_strings = []
        if "xl/sharedStrings.xml" in zf.namelist():
            root = ET.fromstring(zf.read("xl/sharedStrings.xml"))
            for si in root.findall("a:si", NS):
                text = "".join((t.text or "") for t in si.findall(".//a:t", NS))
                shared_strings.append(text)

        sheet_paths = sorted(
            name for name in zf.namelist() if name.startswith("xl/worksheets/sheet")
        )
        if not sheet_paths:
            raise ValueError(f"No worksheet found in {path}")

        root = ET.fromstring(zf.read(sheet_paths[0]))
        rows = []
        for row in root.findall(".//a:sheetData/a:row", NS):
            row_vals: dict[int, str] = {}
            for cell in row.findall("a:c", NS):
                ref = cell.attrib.get("r", "")
                idx = _excel_col_to_idx(ref) if ref else 0
                cell_type = cell.attrib.get("t")
                value_node = cell.find("a:v", NS)
                inline_node = cell.find("a:is/a:t", NS)

                value = ""
                if cell_type == "s" and value_node is not None and value_node.text is not None:
                    si = int(value_node.text)
                    value = shared_strings[si] if si < len(shared_strings) else ""
                elif cell_type == "inlineStr" and inline_node is not None:
                    value = inline_node.text or ""
                elif value_node is not None and value_node.text is not None:
                    value = value_node.text

                row_vals[idx] = value

            if row_vals:
                max_idx = max(row_vals)
                rows.append([row_vals.get(i, "") for i in range(max_idx + 1)])

    if not rows:
        return pd.DataFrame()

    header = [str(x).strip() for x in rows[0]]
    body = rows[1:]
    width = len(header)
    normalized_body = [r + [""] * (width - len(r)) if len(r) < width else r[:width] for r in body]
    df = pd.DataFrame(normalized_body, columns=header)

    for col in df.columns:
        converted = pd.to_numeric(df[col], errors="coerce")
        # Convert a column to numeric only when most values parse as numeric.
        # This avoids accidental conversion of predominantly text columns.
        if converted.notna().mean() > NUMERIC_CONVERSION_THRESHOLD:
            df[col] = converted

    return _normalize_columns(df)


def _extract_xlsx_numeric_values(path: Path) -> list[float]:
    numeric_vals: list[float] = []
    pattern = re.compile(r"^-?\d+(?:\.\d+)?$")

    with ZipFile(path) as zf:
        for sheet in sorted(
            name for name in zf.namelist() if name.startswith("xl/worksheets/sheet")
        ):
            root = ET.fromstring(zf.read(sheet))
            for cell in root.findall(".//a:sheetData/a:row/a:c", NS):
                v = cell.find("a:v", NS)
                if v is None or v.text is None:
                    continue
                txt = v.text.strip()
                if pattern.match(txt):
                    numeric_vals.append(float(txt))

    return numeric_vals


def _safe_load_csv(path: Path) -> pd.DataFrame:
    try:
        return _normalize_columns(pd.read_csv(path))
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required dataset file not found: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to read dataset file: {path}") from exc


def _safe_load_xlsx_first_sheet(path: Path) -> pd.DataFrame:
    try:
        return _load_xlsx_first_sheet(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required dataset file not found: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to parse workbook first sheet: {path}") from exc


def _safe_extract_xlsx_numeric_values(path: Path) -> list[float]:
    try:
        return _extract_xlsx_numeric_values(path)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"Required dataset file not found: {path}") from exc
    except Exception as exc:
        raise RuntimeError(f"Failed to extract numeric values from workbook: {path}") from exc


def _safe_divide_premium_by_income(
    med_premium: pd.Series, ds11_premium_annualized: pd.Series, income: pd.Series
) -> pd.Series:
    return (med_premium + ds11_premium_annualized).div(income.replace(0, pd.NA)).fillna(0)


# 1) Load all 4 newly-added datasets
indian_df = _safe_load_csv(INDIAN_DATA_PATH)
medical_df = _safe_load_csv(MEDICAL_PREMIUM_PATH)
dataset11_df = _safe_load_xlsx_first_sheet(DATASET11_PATH)
handbook_numeric = _safe_extract_xlsx_numeric_values(HANDBOOK_PATH)

# 2) Align and combine datasets using deterministic row-wise repetition
# The 4 files do not share stable keys for a relational join, so rows are aligned
# deterministically by repeating shorter datasets over the base dataset index.
# This keeps every new dataset in use while producing a single modeling table.
base = indian_df.reset_index(drop=True).copy()
med_repeated = medical_df.reindex(base.index % len(medical_df)).reset_index(drop=True)
ds11_repeated = dataset11_df.reindex(base.index % len(dataset11_df)).reset_index(drop=True)

med_repeated = med_repeated.add_prefix("med_")
ds11_repeated = ds11_repeated.add_prefix("ds11_")

df = pd.concat([base, med_repeated, ds11_repeated], axis=1)

# 3) Add handbook-derived macro features
if handbook_numeric:
    handbook_series = pd.Series(handbook_numeric)
    df["handbook_numeric_mean"] = float(handbook_series.mean())
    df["handbook_numeric_std"] = float(handbook_series.std(ddof=0))
    df["handbook_numeric_p90"] = float(handbook_series.quantile(0.90))
else:
    df["handbook_numeric_mean"] = 0.0
    df["handbook_numeric_std"] = 0.0
    df["handbook_numeric_p90"] = 0.0

# 4) Build churn label from combined risk signals
health_signal_cols = ["smoker", "med_diabetes", "med_anychronicdiseases", "ds11_pre_existing_conditions"]
missing_health_cols = [col for col in health_signal_cols if col not in df.columns]
if missing_health_cols:
    print(f"Warning: Missing expected health signal column(s): {missing_health_cols}")

for col in health_signal_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().replace({"true": "1", "false": "0", "yes": "1", "no": "0"})
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

income = pd.to_numeric(df.get("income_lpa", 0), errors="coerce").fillna(0) * 100_000
med_premium = pd.to_numeric(df.get("med_premiumprice", 0), errors="coerce").fillna(0)
ds11_premium_annualized = pd.to_numeric(df.get("ds11_monthly_premium", 0), errors="coerce").fillna(0) * 12

premium_burden = _safe_divide_premium_by_income(med_premium, ds11_premium_annualized, income)

risk_score = (
    premium_burden * PREMIUM_BURDEN_WEIGHT
    + pd.to_numeric(df.get("smoker", 0), errors="coerce").fillna(0) * SMOKER_WEIGHT
    + pd.to_numeric(df.get("med_diabetes", 0), errors="coerce").fillna(0) * DIABETES_WEIGHT
    + pd.to_numeric(df.get("med_anychronicdiseases", 0), errors="coerce").fillna(0) * CHRONIC_DISEASE_WEIGHT
    + pd.to_numeric(df.get("ds11_pre_existing_conditions", 0), errors="coerce").fillna(0) * PRE_EXISTING_WEIGHT
)

# This is a proxy churn target derived from risk signals because the added
# datasets do not provide a direct churn label for every combined row.
df["churnlabel"] = (risk_score > risk_score.median()).astype(int)

print("\n" + "=" * 50)
print("Feature engineering complete using all 4 newly-added datasets")
print(f"Rows: {len(df)}")
print(f"Columns: {len(df.columns)}")
print("Target column: churnlabel")
