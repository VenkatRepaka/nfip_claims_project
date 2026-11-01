"""Build hurricane track features for every claim from NOAA HURDAT2.

Run from student_notebooks/ after 01_EDA.ipynb:   python build_track_features.py
Needs: pandas, numpy, requests. Takes about 2 minutes (one small download).

Output: data/track_features.parquet, one row per claim (id) with:
  trk_storm          name of the closest Atlantic storm within 2 days of the loss date (not a model feature)
  trk_dist_km        distance from the claim to that storm's centre (5000 = no storm in the window)
  trk_wind_kt        the storm's maximum sustained wind at that moment (knots, 0 = no storm)
  trk_pres_mb        the storm's central pressure at that moment (lower = stronger)
  trk_r34_km         the storm's largest tropical-storm-force wind radius (how big the storm was)
  trk_maxwind_200km  strongest wind of the storm while its centre was within 200 km (0 = never that close)
  trk_right_side     1 if the claim was on the right of the storm's direction of motion (stronger winds and surge)
  trk_inside_r34     1 if the claim was inside the tropical-storm-force wind area

Data: NOAA National Hurricane Center, HURDAT2 Atlantic best track, https://www.nhc.noaa.gov/data/#hurdat
The best track has one point every 6 hours; we interpolate to every hour. The wind is the storm's maximum wind
near its centre, not the wind at the house. Only Atlantic storms are included.
"""
import io
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RAW_FILE = '/Users/home/workspace/School/DataMining/project/NfipClaimsV3.parquet'   # same as notebook 01
HURDAT_URL = 'https://www.nhc.noaa.gov/data/hurdat/hurdat2-1851-2025-092326.txt'
CENTERS_URL = 'https://www2.census.gov/geo/docs/reference/cenpop2020/blkgrp/CenPop2020_Mean_BG.txt'
DATA = Path('data')
NO_STORM_KM = 5000


def read_hurdat():
    """One row per 6-hourly best-track point since December 2019."""
    rows = []
    for line in requests.get(HURDAT_URL, timeout=120).text.splitlines():
        p = [x.strip() for x in line.split(',')]
        if p[0].startswith('AL'):                            # header line of a new storm
            storm_id, name = p[0], p[1].title()
            continue
        if p[0] < '20191201':
            continue
        lat = float(p[4][:-1]) * (1 if p[4][-1] == 'N' else -1)
        lon = float(p[5][:-1]) * (-1 if p[5][-1] == 'W' else 1)
        rows.append({'storm_id': storm_id, 'name': name, 'time': pd.to_datetime(p[0] + p[1], format='%Y%m%d%H%M'),
                     'lat': lat, 'lon': lon, 'wind_kt': int(p[6]), 'pres_mb': int(p[7]),
                     'r34_ne': int(p[8]), 'r34_se': int(p[9]), 'r34_sw': int(p[10]), 'r34_nw': int(p[11])})
    return pd.DataFrame(rows).replace(-999, np.nan)


def hourly_tracks(points):
    """Interpolate each storm to one point per hour and add its direction of motion."""
    out = []
    for storm_id, g in points.groupby('storm_id'):
        g = g.drop_duplicates('time').set_index('time').sort_index()
        hours = pd.date_range(g.index.min(), g.index.max(), freq='h')
        num = g[['lat', 'lon', 'wind_kt', 'pres_mb', 'r34_ne', 'r34_se', 'r34_sw', 'r34_nw']]
        h = num.reindex(hours.union(g.index)).interpolate('time').reindex(hours)
        h['name'] = g.name.iloc[0]
        h['dlat'], h['dlon'] = h.lat.diff().bfill(), h.lon.diff().bfill()
        out.append(h.reset_index(names='time'))
    tracks = pd.concat(out, ignore_index=True)
    tracks['r34_km'] = tracks[['r34_ne', 'r34_se', 'r34_sw', 'r34_nw']].max(axis=1) * 1.852   # nautical miles -> km
    return tracks


def claim_locations():
    """Block-group population centre of every claim, else its rounded lat/lon (same as build_rain_features.py)."""
    claims = pd.read_parquet(DATA / 'claims_clean.parquet', columns=['id', 'dateOfLoss', 'latitude', 'longitude'])
    geoid = pd.read_parquet(RAW_FILE, columns=['id', 'censusGeoid'])
    centers = pd.read_csv(CENTERS_URL, dtype=str, encoding='utf-8-sig')
    centers['censusGeoid'] = centers.STATEFP + centers.COUNTYFP + centers.TRACTCE + centers.BLKGRPCE
    centers[['bg_lat', 'bg_lon']] = centers[['LATITUDE', 'LONGITUDE']].astype(float)
    claims = claims.merge(geoid, on='id', how='left').merge(centers[['censusGeoid', 'bg_lat', 'bg_lon']], on='censusGeoid', how='left')
    claims['lat'] = claims.bg_lat.fillna(claims.latitude)
    claims['lon'] = claims.bg_lon.fillna(claims.longitude)
    return claims[['id', 'dateOfLoss', 'lat', 'lon']]


def km_between(lat1, lon1, lat2, lon2):
    la1, lo1, la2, lo2 = map(np.radians, (lat1, lon1, lat2, lon2))
    a = np.sin((la2 - la1) / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin((lo2 - lo1) / 2) ** 2
    return 2 * 6371 * np.arcsin(np.sqrt(a))


def track_features(claims, tracks):
    out = []
    for day, day_claims in claims.groupby('dateOfLoss'):
        window = tracks[(tracks.time >= day - pd.Timedelta(days=2)) & (tracks.time < day + pd.Timedelta(days=2))]
        if window.empty:
            out.append(day_claims[['id']])
            continue
        loc = day_claims[['lat', 'lon']].drop_duplicates()
        dist = km_between(loc.lat.values[:, None], loc.lon.values[:, None], window.lat.values[None, :], window.lon.values[None, :])
        near = window.iloc[dist.argmin(axis=1)].reset_index(drop=True)
        f = pd.DataFrame({'lat': loc.lat.values, 'lon': loc.lon.values, 'trk_storm': near.name.values,
                          'trk_dist_km': dist.min(axis=1), 'trk_wind_kt': near.wind_kt.values,
                          'trk_pres_mb': near.pres_mb.values, 'trk_r34_km': near.r34_km.values,
                          'trk_maxwind_200km': np.where(dist <= 200, window.wind_kt.values[None, :], 0).max(axis=1)})
        # cross product of the motion and the storm->claim direction: negative = claim is right of the track
        cross = near.dlon.values * (f.lat.values - near.lat.values) - near.dlat.values * (f.lon.values - near.lon.values)
        f['trk_right_side'] = (cross < 0).astype(float)
        out.append(day_claims[['id', 'lat', 'lon']].merge(f, on=['lat', 'lon'], how='left').drop(columns=['lat', 'lon']))
    feat = pd.concat(out, ignore_index=True)
    feat['trk_inside_r34'] = (feat.trk_dist_km <= feat.trk_r34_km.fillna(0)).astype(float)
    feat['trk_dist_km'] = feat.trk_dist_km.fillna(NO_STORM_KM)
    feat[['trk_wind_kt', 'trk_maxwind_200km']] = feat[['trk_wind_kt', 'trk_maxwind_200km']].fillna(0)
    return feat


if __name__ == '__main__':
    tracks = hourly_tracks(read_hurdat())
    print(f'HURDAT2: {tracks.name.nunique()} storms, {tracks.time.min():%Y-%m-%d} to {tracks.time.max():%Y-%m-%d}')
    claims = claim_locations()
    features = track_features(claims, tracks)
    features.to_parquet(DATA / 'track_features.parquet', index=False)
    print(f'Saved data/track_features.parquet: {len(features):,} claims')
    print(f'  a storm within 2 days of the loss date: {features.trk_storm.notna().mean():.1%}')
    print(f'  storm centre within 200 km:             {(features.trk_dist_km <= 200).mean():.1%}')
