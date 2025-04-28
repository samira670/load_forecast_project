# ───────────────────────────────────────────────
#  src/etl_build_dataset.py   ← paste ALL of this
# ───────────────────────────────────────────────

from pathlib import Path
import pandas as pd

# ------------------------------------------------------------------
# 0.  Folders
# ------------------------------------------------------------------
RAW   = Path("data/raw")            # Excel files live here
CLEAN = Path("data/clean")          # parquet output
CLEAN.mkdir(parents=True, exist_ok=True)

# ------------------------------------------------------------------
# 1.  Helper: choose correct pandas engine
# ------------------------------------------------------------------
def choose_engine(p: Path):
    return "xlrd" if p.suffix == ".xls" else "openpyxl"

# ------------------------------------------------------------------
# 2.  Workbook readers
# ------------------------------------------------------------------
def read_2011(p: Path):
    return pd.read_excel(p, engine=choose_engine(p),
                         sheet_name="Load by AESO Planning Area",
                         skiprows=1)

def read_2017(p: Path):
    return pd.read_excel(p, engine=choose_engine(p),
                         sheet_name="Load by Area and Region")

def read_2020_or_2023(p: Path):
    return pd.read_excel(p, engine=choose_engine(p),
                         sheet_name="Sheet1")

# ------------------------------------------------------------------
# 3.  Filenames ONLY  (strings)  – NO paths here
# ------------------------------------------------------------------
files = [
    ("2011", read_2011, "Hourly-load-by-area-and-region-2011-to-2017.xlsx"),
    ("2017", read_2017, "Hourly-load-by-area-and-region-2017-2020.xlsx"),
    ("2020", read_2020_or_2023, "Hourly-load-by-area-and-region-May-2020-to-Oct-2023.xlsx"),
    ("2023", read_2020_or_2023, "Hourly-load-by-area-and-region-Nov-2023-to-Dec-2024.xlsx"),
]

# ------------------------------------------------------------------
# 4.  Read each workbook safely
# ------------------------------------------------------------------
dfs = []
for label, reader, fname in files:
    full_path = RAW / fname      # build full path exactly once
    print("DEBUG opening →", full_path)   # show the exact path

    try:
        df = reader(full_path)
        print(f"✅ {label} {fname}  →  {df.shape}")
        dfs.append(df)
    except Exception as e:
        print(f"⚠️ {label} {fname} skipped → {e}")

if not dfs:
    raise RuntimeError("Nothing could be read – check file names & location!")

# ------------------------------------------------------------------
# 5.  Merge & initial clean
# ------------------------------------------------------------------
load_raw = (pd.concat(dfs, ignore_index=True)
              .dropna(axis=1, how="all")
              .apply(pd.to_numeric, errors="coerce"))

load_raw.to_parquet(RAW / "load_raw.parquet", index=False)
print("✅ wrote data/raw/load_raw.parquet")

# ------------------------------------------------------------------
# 6.  Build timestamp
# ------------------------------------------------------------------
if "DT_MST" in load_raw.columns:
    load_raw["timestamp"] = pd.to_datetime(load_raw["DT_MST"])
else:
    load_raw["timestamp"] = (pd.to_datetime(load_raw["DATE"]) +
                              pd.to_timedelta(load_raw["HOUR ENDING"].astype(int)-1, "h"))

# Drop stray numeric column names
load_raw = load_raw.loc[:, ~load_raw.columns.astype(str).str.match(r"^\d+(\.\d+)?$")]

# ------------------------------------------------------------------
# 7.  Save parquet (wide + long)
# ------------------------------------------------------------------
load_raw.to_parquet(CLEAN / "load_wide_clean.parquet", index=False)

load_long = (load_raw
             .set_index("timestamp")
             .reset_index()
             .melt(id_vars="timestamp", var_name="region", value_name="load_MW"))
load_long.to_parquet(CLEAN / "load_long.parquet", index=False)

print("✅ wrote data/clean/load_long.parquet — ETL complete!")
