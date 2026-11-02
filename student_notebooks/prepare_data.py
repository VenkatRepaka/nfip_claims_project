"""Prepare the claims table before building external features on a new machine."""
from project_utils import DATA, clean_claims

if __name__ == '__main__':
    claims = clean_claims()
    print(f'Saved {len(claims):,} claims to {DATA / "claims_clean.parquet"}')
    for name in ['rain_features.parquet', 'track_features.parquet', 'acs_blockgroups.parquet']:
        print(f'{name}: {"ready" if (DATA / name).exists() else "run the matching build script"}')
