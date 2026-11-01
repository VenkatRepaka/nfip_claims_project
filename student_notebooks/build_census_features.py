"""Download ACS 2020-2024 5-year block-group data (income, home value, renters) and save data/acs_blockgroups.parquet.

Source: US Census Bureau, American Community Survey 5-year estimates, table-based summary files
(https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/). No API key is needed.
Run once from the student_notebooks folder:  python build_census_features.py
"""
from pathlib import Path
import urllib.request

import numpy as np
import pandas as pd

YEAR = 2024
BASE = f'https://www2.census.gov/programs-surveys/acs/summary_file/{YEAR}/table-based-SF/data/5YRData/acsdt5y{YEAR}-{{}}.dat'
TABLES = {'b19013': {'B19013_E001': 'median_household_income'},
          'b25077': {'B25077_E001': 'median_home_value'},
          'b25003': {'B25003_E001': 'occupied_units', 'B25003_E003': 'renter_units'}}
CACHE = Path('data/acs')
CACHE.mkdir(parents=True, exist_ok=True)

frames = []
for table, columns in TABLES.items():
    path = CACHE / f'acsdt5y{YEAR}-{table}.dat'
    if not path.exists():
        print('downloading', table)
        urllib.request.urlretrieve(BASE.format(table), path)
    t = pd.read_csv(path, sep='|', dtype=str, usecols=['GEO_ID'] + list(columns))
    t = t[t['GEO_ID'].str.startswith('1500000US')]                     # summary level 150 = block group
    t['censusGeoid'] = t['GEO_ID'].str[-12:]
    t = t.drop(columns='GEO_ID').set_index('censusGeoid').rename(columns=columns)
    frames.append(t.apply(pd.to_numeric, errors='coerce'))

acs = pd.concat(frames, axis=1)
acs = acs.mask(acs < 0)                                                # Census codes like -666666666 mean "not available"
acs['renter_share'] = acs['renter_units'] / acs['occupied_units'].where(acs['occupied_units'] > 0)
acs = acs.reset_index()
acs.to_parquet('data/acs_blockgroups.parquet', index=False)
print(f'block groups: {len(acs):,}')
print(acs.describe().round(2).to_string())
