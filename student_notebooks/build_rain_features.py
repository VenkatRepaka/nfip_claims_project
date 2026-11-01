"""Build rainfall features for every claim from PRISM daily precipitation grids.

Run from student_notebooks/ after 01_EDA.ipynb:   python build_rain_features.py
Optional: python build_rain_features.py GRID_DIR   reuses grids already saved there as YYYYMMDD.npy
(PRISM allows each file to be downloaded only twice a day, so do not re-download days you already have.)
Needs: pandas, numpy, requests, tifffile, imagecodecs (pip install tifffile imagecodecs).
Takes about 70 minutes the first time (one PRISM download per day). It can be stopped and restarted:
finished days are kept in data/prism_days/.

Output: data/rain_features.parquet, one row per claim (id) with:
  rain_lossday_mm   rain in the 48 hours around the loss date
  rain_3d_mm        rain from the day before to the day after the loss date
  rain_5d_mm        rain from 3 days before to 1 day after the loss date
  rain_max1d_mm     wettest single day in that 5-day window
  rain_nb_max1d_mm  wettest single day in the wettest 4 km cell within about 6 km
  rain_nb_5d_mm     5-day rain, taking the wettest nearby cell each day
  rain_location     'block group' or 'rounded lat/lon' (where we read the grid)

Data: PRISM Climate Group, Oregon State University, https://prism.oregonstate.edu (daily 4 km precipitation).
A PRISM day is the 24 hours ending at 12:00 UTC (7-8 am Eastern), so the rain of the loss date is mostly in the
PRISM day after it. That is why the window runs from 3 days before to 1 day after.
PRISM only covers the lower 48 states: claims in HI, AK, PR and the territories get no rain value.
"""
import io
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests
import tifffile

RAW_FILE = '/Users/home/workspace/School/DataMining/project/NfipClaimsV3.parquet'   # same as notebook 01
CENTERS_URL = 'https://www2.census.gov/geo/docs/reference/cenpop2020/blkgrp/CenPop2020_Mean_BG.txt'
PRISM_URL = 'https://services.nacse.org/prism/data/get/us/4km/ppt/{:%Y%m%d}'
DATA = Path('data')
DAY_DIR = DATA / 'prism_days'
WINDOW = range(-3, 2)                                         # days relative to the loss date
X0, Y0, RES, N_ROWS, N_COLS = -125.0208333333335, 49.9375000000005, 0.041666666667, 621, 1405


def claim_cells():
    """Grid cell (row, col) of every claim: the block-group population centre, else the rounded lat/lon."""
    claims = pd.read_parquet(DATA / 'claims_clean.parquet', columns=['id', 'dateOfLoss', 'latitude', 'longitude'])
    geoid = pd.read_parquet(RAW_FILE, columns=['id', 'censusGeoid'])
    centers = pd.read_csv(CENTERS_URL, dtype=str, encoding='utf-8-sig')
    centers['censusGeoid'] = centers.STATEFP + centers.COUNTYFP + centers.TRACTCE + centers.BLKGRPCE
    centers[['bg_lat', 'bg_lon']] = centers[['LATITUDE', 'LONGITUDE']].astype(float)
    claims = claims.merge(geoid, on='id', how='left').merge(centers[['censusGeoid', 'bg_lat', 'bg_lon']], on='censusGeoid', how='left')
    claims['rain_location'] = np.where(claims.bg_lat.notna(), 'block group', 'rounded lat/lon')
    lat = claims.bg_lat.fillna(claims.latitude)
    lon = claims.bg_lon.fillna(claims.longitude)
    claims['row'] = np.floor((Y0 - lat) / RES).astype(int)
    claims['col'] = np.floor((lon - X0) / RES).astype(int)
    # keep one cell of margin for the 3x3 neighbourhood; outside the grid = not in the lower 48 states
    claims['on_grid'] = claims.row.between(1, N_ROWS - 2) & claims.col.between(1, N_COLS - 2)
    return claims


def download_grid(day, grid_dir=None):
    if grid_dir and (Path(grid_dir) / f'{day:%Y%m%d}.npy').exists():
        return np.load(Path(grid_dir) / f'{day:%Y%m%d}.npy')
    for attempt in range(4):
        try:
            r = requests.get(PRISM_URL.format(day), timeout=120)
            r.raise_for_status()
            if b'more than twice' in r.content[:300]:     # PRISM daily limit: retrying would risk an IP block
                raise SystemExit(f'PRISM: {day:%Y-%m-%d} was already downloaded twice today. Try again tomorrow.')
            z = zipfile.ZipFile(io.BytesIO(r.content))
            tif = next(n for n in z.namelist() if n.endswith('.tif'))
            a = tifffile.imread(io.BytesIO(z.read(tif))).astype(np.float32)
            a[a < 0] = np.nan                                 # -9999 = no data (sea, outside the US)
            return a
        except Exception as e:                               # network hiccup or server busy: wait and retry
            print(f'   {day:%Y-%m-%d} attempt {attempt + 1} failed: {e}', flush=True)
            time.sleep(30 * (attempt + 1))
    raise RuntimeError(f'could not download PRISM day {day:%Y-%m-%d}')


def sample_days(claims, grid_dir=None):
    """For every PRISM day we need, read the rain at the grid cells of the claims whose window contains that day."""
    DAY_DIR.mkdir(exist_ok=True)
    on = claims[claims.on_grid]
    cells_by_day = {}
    for k in WINDOW:
        shifted = on.assign(day=on.dateOfLoss + pd.Timedelta(days=k))[['day', 'row', 'col']]
        for day, cells in shifted.groupby('day'):
            cells_by_day.setdefault(day, []).append(cells[['row', 'col']])
    days = sorted(cells_by_day)
    todo = [d for d in days if not (DAY_DIR / f'{d:%Y%m%d}.parquet').exists()]
    print(f'{len(days):,} PRISM days needed, {len(todo):,} still to download', flush=True)
    t0 = time.time()
    for i, day in enumerate(todo):
        a = download_grid(day, grid_dir)
        cells = pd.concat(cells_by_day[day]).drop_duplicates()
        r, c = cells.row.values, cells.col.values
        block = np.stack([a[r + dr, c + dc] for dr in (-1, 0, 1) for dc in (-1, 0, 1)])
        ppt = np.where(np.isnan(a[r, c]), np.nanmean(block, axis=0), a[r, c])   # coastal cell on the sea: use the land around it
        nb = np.nanmax(block, axis=0)
        pd.DataFrame({'day': day, 'row': r, 'col': c, 'ppt': ppt, 'ppt_nb_max': nb}).to_parquet(DAY_DIR / f'{day:%Y%m%d}.parquet')
        time.sleep(1)                                        # be polite to the PRISM server
        if i % 50 == 0 or i == len(todo) - 1:
            left = (time.time() - t0) / (i + 1) * (len(todo) - i - 1) / 60
            print(f'{i + 1:,}/{len(todo):,}  {day:%Y-%m-%d}  about {left:.0f} min left', flush=True)
    return pd.concat(pd.read_parquet(DAY_DIR / f'{d:%Y%m%d}.parquet') for d in days)


def rain_features(claims, points):
    on = claims[claims.on_grid]
    per_day = []
    for k in WINDOW:
        t = on.assign(day=on.dateOfLoss + pd.Timedelta(days=k)).merge(points, on=['day', 'row', 'col'], how='left')
        per_day.append(t[['id', 'ppt', 'ppt_nb_max']].assign(k=k))
    w = pd.concat(per_day).pivot_table(index='id', columns='k', values=['ppt', 'ppt_nb_max'], dropna=False)
    ppt, nb = w['ppt'], w['ppt_nb_max']
    out = pd.DataFrame(index=w.index)
    out['rain_lossday_mm'] = ppt[0] + ppt[1]
    out['rain_3d_mm'] = ppt[[-1, 0, 1]].sum(axis=1, min_count=3)
    out['rain_5d_mm'] = ppt.sum(axis=1, min_count=5)
    out['rain_max1d_mm'] = ppt.max(axis=1)
    out['rain_nb_max1d_mm'] = nb.max(axis=1)
    out['rain_nb_5d_mm'] = nb.sum(axis=1, min_count=5)
    out = claims[['id', 'rain_location']].merge(out.reset_index(), on='id', how='left')
    out.loc[~claims.on_grid.values, 'rain_location'] = 'outside PRISM grid'
    return out


if __name__ == '__main__':
    claims = claim_cells()
    print(f'Claims: {len(claims):,}   on the PRISM grid: {claims.on_grid.mean():.2%}   '
          f'located by block group: {(claims.rain_location == "block group").mean():.1%}', flush=True)
    points = sample_days(claims, sys.argv[1] if len(sys.argv) > 1 else None)
    features = rain_features(claims, points)
    features.to_parquet(DATA / 'rain_features.parquet', index=False)
    print(f'Saved data/rain_features.parquet: {len(features):,} claims, '
          f'{features.rain_5d_mm.notna().mean():.2%} with a rain value')
    print(features.describe().round(1).T)
