"""Step 1 - download OpenFEMA NFIP Redacted Claims v3 (bulk Parquet), filter to 2020-01-01..2025-12-31
by dateOfLoss, record row counts, save the raw filtered file. Also cross-checks the count with the API."""
import requests, pandas as pd
from common import *

def download(url, dest):
    if dest.exists() and dest.stat().st_size > 1e6:
        print("using cached", dest); return
    with requests.get(url, stream=True, timeout=600) as r:
        r.raise_for_status()
        with open(dest, "wb") as fh:
            for chunk in r.iter_content(1 << 20): fh.write(chunk)

download(BULK_URL, FULL_PARQUET)
download(FIELDS_URL, FIELDS_JSON)

df = pd.read_parquet(FULL_PARQUET)
dol = pd.to_datetime(df["dateOfLoss"], errors="coerce")
mask = (dol >= START) & (dol <= END)
sub = df.loc[mask].copy()
counts = {
    "bulk_file": BULK_URL,
    "total_rows_full_dataset": int(len(df)),
    "total_columns": int(df.shape[1]),
    "dateOfLoss_min_full": str(dol.min().date()), "dateOfLoss_max_full": str(dol.max().date()),
    "rows_dateOfLoss_null": int(dol.isna().sum()),
    "rows_in_2020_2025_by_dateOfLoss": int(mask.sum()),
    "rows_in_2020_2025_by_yearOfLoss": int(df["yearOfLoss"].between(2020, 2025).sum()),
    "rows_where_yearOfLoss_ne_year_of_dateOfLoss": int((dol.dt.year != df["yearOfLoss"]).sum()),
    "per_yearOfLoss": {int(k): int(v) for k, v in sub["yearOfLoss"].value_counts().sort_index().items()},
    "asOfDate_values": [str(x) for x in sub["asOfDate"].unique()],
}
# API cross-check (count only)
try:
    u = ("https://www.fema.gov/api/open/v3/NfipClaims?$filter=dateOfLoss%20ge%20'2020-01-01T00:00:00.000Z'"
         "%20and%20dateOfLoss%20le%20'2025-12-31T23:59:59.000Z'&$top=1&$select=id&$count=true")
    counts["api_count_same_filter"] = requests.get(u, timeout=60).json()["metadata"]["count"]
except Exception as e:
    counts["api_count_same_filter"] = f"unavailable: {e}"
sub.to_parquet(FILTERED, index=False)
counts["raw_filtered_file"] = str(FILTERED.relative_to(ROOT))
save_json(counts, "row_counts.json")
print(json.dumps(counts, indent=2))
