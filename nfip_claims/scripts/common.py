"""Shared paths/helpers for the NFIP claims pipeline."""
from pathlib import Path
import json, numpy as np, pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "raw"; OUT = ROOT / "outputs"; FIG = ROOT / "figures"
for p in (RAW, OUT, FIG): p.mkdir(exist_ok=True)

FULL_PARQUET = RAW / "NfipClaimsV3_full.parquet"
FILTERED = RAW / "nfip_claims_2020_2025_raw.parquet"
FIELDS_JSON = RAW / "fields.json"
BULK_URL = "https://www.fema.gov/about/reports-and-data/openfema/v3/NfipClaimsV3.parquet"
FIELDS_URL = ("https://www.fema.gov/api/open/v1/OpenFemaDataSetFields?$filter="
              "openFemaDataSet%20eq%20'NfipClaims'%20and%20datasetVersion%20eq%203&$top=1000")
START, END = "2020-01-01", "2025-12-31"
SEED = 42

def save_json(obj, name):
    (OUT / name).write_text(json.dumps(obj, indent=2, default=str))

def load_json(name):
    return json.loads((OUT / name).read_text())

def dec_to_float(s: pd.Series) -> pd.Series:
    """OpenFEMA decimal columns arrive as python Decimal objects in the parquet; convert to float64."""
    return pd.to_numeric(s.map(lambda x: float(x) if x is not None and x == x else np.nan), errors="coerce")

def load_field_dictionary():
    f = json.loads(FIELDS_JSON.read_text())
    key = [k for k in f if k != "metadata"][0]
    return {x["name"]: x for x in f[key]}
