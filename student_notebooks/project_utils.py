"""Shared data preparation and evaluation for the student notebook sequence.

No network requests are made here. Missing predictor values are filled inside
each fitted model, using its training data only.
"""
from pathlib import Path
import os
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.compose import TransformedTargetRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import (r2_score, mean_absolute_error, median_absolute_error,
                             precision_score, recall_score, f1_score, roc_auc_score,
                             average_precision_score)

ROOT = Path(__file__).resolve().parent
DATA = ROOT / 'data'
RESULTS = ROOT / 'results'
SEED = 42
OCCUPANCY = {'11': 'single family', '12': '2-4 units', '13': '5+ units',
             '14': 'mobile home', '15': 'condo building', '16': 'unit in building',
             '17': 'non-res. mobile', '18': 'non-residential', '19': 'non-res. unit'}
FOUNDATION = {'0': 'not recorded', '1': 'slab', '2': 'basement', '3': 'crawlspace',
              '4': 'elevated (4)', '5': 'elevated (5)', '6': 'elevated (6)'}
BANDS = ['minor (0-10%)', 'moderate (10-25%)', 'major (25-50%)', 'severe (over 50%)']
DEDUCTIBLES = {'0': 500, '1': 1000, '2': 2000, '3': 3000, '4': 4000,
              '5': 5000, '9': 750, 'A': 10000, 'B': 15000, 'C': 20000,
              'D': 25000, 'E': 50000, 'F': 1250, 'G': 1500, 'H': 200}


def raw_path():
    candidates = [Path(os.environ.get('NFIP_RAW_FILE', 'NfipClaimsV3.parquet')),
                  DATA / 'NfipClaimsV3.parquet',
                  ROOT.parent.parent / 'project' / 'NfipClaimsV3.parquet']
    for path in candidates:
        if path.exists():
            return path.resolve()
    raise FileNotFoundError('Set NFIP_RAW_FILE to the raw NfipClaimsV3.parquet file.')


def code(series):
    return series.astype('string').str.replace(r'\.0$', '', regex=True).fillna('unknown').astype(str)


def clean_claims():
    raw = pd.read_parquet(raw_path(), filters=[('yearOfLoss', '>=', 2020),
                                             ('yearOfLoss', '<=', 2025)])
    d = raw.loc[pd.to_numeric(raw.yearOfLoss).between(2020, 2025)].copy().reset_index(drop=True)
    if not d.id.is_unique:
        raise ValueError('Repeated claim IDs need review before analysis.')
    numeric = ['amountPaidOnBuildingClaim', 'buildingDamageAmount', 'buildingPropertyValue',
               'buildingReplacementCost', 'totalBuildingInsuranceCoverage', 'totalContentsInsuranceCoverage',
               'numberOfUnits', 'policyCount', 'crsClassCode', 'latitude', 'longitude', 'elevationDifference']
    for c in numeric:
        d[c] = pd.to_numeric(d[c], errors='coerce').astype(float)
    for c in ['dateOfLoss', 'openDate', 'mostRecentPaymentDate', 'originalConstructionDate', 'asOfDate']:
        d[c] = pd.to_datetime(d[c].astype(str), errors='coerce')
        d[c] = d[c].where(d[c].dt.year >= 1800)
    d['loss_year'] = d.dateOfLoss.dt.year
    d['loss_month'] = d.dateOfLoss.dt.month
    d['building_age'] = (d.dateOfLoss - d.originalConstructionDate).dt.days / 365.25
    d.loc[d.building_age < 0, 'building_age'] = np.nan
    lag = (d.openDate - d.dateOfLoss).dt.days
    d['report_lag_days'] = lag.where(lag >= 0).clip(upper=365)
    d['negative_payment_flag'] = d.amountPaidOnBuildingClaim < 0
    d.loc[d.negative_payment_flag, 'amountPaidOnBuildingClaim'] = np.nan
    d['named_storm'] = d.floodEvent.notna().astype(int)
    def zone(z):
        z = str(z).upper()
        if z in ['X', 'B', 'C']: return 'X'
        if z.startswith('V'): return 'V'
        if z in ['A', 'AE', 'AH', 'AO', 'A99']: return z
        if z.startswith('A'): return 'AE'
        return 'unknown'
    d['zone'] = d.ratedFloodZone.map(zone)
    d['occupancy'] = code(d.occupancyType.replace({1: 11, 2: 12, 3: 13, 4: 18, 6: 18}))
    d['occupancy_name'] = d.occupancy.map(OCCUPANCY).fillna('unknown')
    d['foundation'] = code(d.foundationType)
    d['foundation_name'] = d.foundation.map(FOUNDATION).fillna('unknown')
    d['numberOfFloorsInTheInsuredBuilding'] = code(d.numberOfFloorsInTheInsuredBuilding)
    d['state'] = d.state.fillna('unknown').astype(str)
    d['deductible_usd'] = code(d.buildingDeductibleCode).map(DEDUCTIBLES)
    d['rating_system'] = np.where(d.rateMethod == 'RE', 'Risk Rating 2.0', 'legacy / other')
    d['elevationDifference'] = d.elevationDifference.where(d.elevationDifference.abs() < 500)
    ratio = d.buildingDamageAmount / d.buildingPropertyValue.where(d.buildingPropertyValue > 0)
    d['ratio_above_one_flag'] = ratio > 1
    d['damage_ratio_raw'] = ratio
    d['damage_ratio'] = ratio.where(ratio.between(0, 1))
    d['severity_band'] = pd.cut(d.damage_ratio, [0, .1, .25, .5, 1], labels=BANDS, include_lowest=True)
    d['censusGeoid'] = code(d.censusGeoid)
    # Unknown binary values stay missing; a training pipeline handles them.
    for c in ['elevatedBuildingIndicator', 'postFIRMConstructionIndicator',
              'primaryResidenceIndicator', 'rentalPropertyIndicator']:
        d[c] = pd.to_numeric(d[c], errors='coerce').astype(float)
    compare = ['dateOfLoss', 'state', 'latitude', 'longitude', 'occupancy',
               'totalBuildingInsuranceCoverage', 'amountPaidOnBuildingClaim', 'buildingDamageAmount']
    d['similar_record_flag'] = d.duplicated(compare, keep=False)
    DATA.mkdir(exist_ok=True)
    d.to_parquet(DATA / 'claims_clean.parquet', index=False)
    return d


def load_joined():
    d = pd.read_parquet(DATA / 'claims_clean.parquet')
    if 'censusGeoid' not in d:
        raise ValueError('Run the revised notebook 01 first.')
    before = len(d)
    for filename, key, how in [('rain_features.parquet', 'id', 'one_to_one'),
                                ('track_features.parquet', 'id', 'one_to_one'),
                                ('acs_blockgroups.parquet', 'censusGeoid', 'many_to_one')]:
        path = DATA / filename
        if not path.exists():
            raise FileNotFoundError(f'{path.name} is missing. Run its build script; see README.')
        t = pd.read_parquet(path)
        if key == 'censusGeoid':
            t[key] = code(t[key])
        d = d.merge(t, on=key, how='left', validate=how, sort=False)
        assert len(d) == before, 'A join changed the number of claims.'
    # Older feature caches were built after missing coordinates were filled.
    # Keep valid block-group matches, but discard any fallback at an unknown location.
    unknown_location = (d.latitude.isna() | d.longitude.isna()) & d.rain_location.ne('block group')
    weather = [c for c in d if (c.startswith('rain_') and c != 'rain_location')
               or (c.startswith('trk_') and c != 'trk_storm')]
    d.loc[unknown_location, weather] = np.nan
    d.loc[unknown_location, 'rain_location'] = 'location unavailable'
    for c in ['rain_5d_mm', 'trk_dist_km', 'median_household_income']:
        d[c + '_missing'] = d[c].isna().astype(int)
    # A few storm parameters are not supplied for every historical track point.
    for c in ['rain_5d_mm', 'rain_max1d_mm', 'trk_dist_km', 'trk_wind_kt', 'trk_pres_mb',
              'buildingReplacementCost', 'buildingPropertyValue', 'totalBuildingInsuranceCoverage',
              'totalContentsInsuranceCoverage', 'policyCount', 'numberOfUnits', 'deductible_usd',
              'median_household_income', 'median_home_value']:
        d['log_' + c] = np.log1p(d[c].clip(lower=0))
    d['county_group'] = code(d.countyCode)
    return add_claim_context(d)


def add_claim_context(d):
    """Add simple counts of how many claims came from the same storm and place.

    These are counts of claim records, never payments or damage. They are a
    rough sign of how widely a storm flooded an area. They are known only after
    claims arrive, so models that use them are retrospective comparisons.
    """
    storm = d.floodEvent.fillna('unnamed-' + d.loss_year.astype(str) + '-' + d.loss_month.astype(str))
    block = d.censusGeoid.where(~d.censusGeoid.isin(['unknown', 'nan']), 'no block group')
    d['claims_block_storm'] = np.log1p(d.groupby([storm, block]).id.transform('size'))
    d['claims_county_storm'] = np.log1p(d.groupby([storm, d.county_group]).id.transform('size'))
    d['claims_state_storm'] = np.log1p(d.groupby([storm, d.state]).id.transform('size'))
    d['claims_storm_total'] = np.log1p(d.groupby(storm).id.transform('size'))
    d['county_share_of_storm'] = np.expm1(d.claims_county_storm) / np.expm1(d.claims_storm_total)
    # Simple building and storm ratios.
    d['is_condo'] = d.occupancy.eq('15').astype(int)
    d['coverage_per_unit'] = d.log_totalBuildingInsuranceCoverage - d.log_numberOfUnits
    d['storm_strength'] = d.trk_wind_kt / (1 + d.trk_dist_km.clip(upper=500) / 50)
    d['wind_on_right_side'] = d.trk_wind_kt * d.trk_right_side
    return d


POLICY = ['log_totalBuildingInsuranceCoverage', 'log_totalContentsInsuranceCoverage',
          'log_deductible_usd', 'log_policyCount', 'log_numberOfUnits']
BUILDING = ['building_age', 'zone', 'foundation', 'occupancy',
            'numberOfFloorsInTheInsuredBuilding', 'crsClassCode', 'elevatedBuildingIndicator',
            'postFIRMConstructionIndicator', 'primaryResidenceIndicator', 'rentalPropertyIndicator']
PLACE = ['state', 'loss_month', 'named_storm', 'report_lag_days']
CORE = POLICY + BUILDING + PLACE
RAIN = ['rain_lossday_mm', 'rain_3d_mm', 'rain_5d_mm', 'rain_max1d_mm',
        'rain_nb_max1d_mm', 'rain_nb_5d_mm', 'rain_5d_mm_missing']
TRACK = ['trk_dist_km', 'trk_wind_kt', 'trk_pres_mb', 'trk_r34_km',
         'trk_maxwind_200km', 'trk_right_side', 'trk_inside_r34', 'trk_dist_km_missing']
CENSUS = ['log_median_household_income', 'log_median_home_value', 'renter_share', 'median_household_income_missing']
LOCAL_COUNTS = ['claims_block_storm', 'claims_county_storm']
STORM_COUNTS = ['claims_state_storm', 'claims_storm_total', 'county_share_of_storm']
CLAIM_COUNTS = LOCAL_COUNTS + STORM_COUNTS
RATIOS = ['is_condo', 'coverage_per_unit', 'storm_strength', 'wind_on_right_side']
FEATURE_SETS = {
    'policy only': POLICY,
    'building + place': BUILDING + PLACE,
    'core': CORE,
    'core + rain': CORE + RAIN,
    'core + track': CORE + TRACK,
    'core + rain + track': CORE + RAIN + TRACK,
    'core + census': CORE + CENSUS,
    'all sources': CORE + RAIN + TRACK + CENSUS,
    'core + local counts': CORE + LOCAL_COUNTS,
    'core + claim counts': CORE + CLAIM_COUNTS,
    'core + ratios': CORE + RATIOS,
    'core + track + claim counts': CORE + TRACK + CLAIM_COUNTS,
    'core + track + ratios': CORE + TRACK + RATIOS,
    'best mix': CORE + TRACK + RATIOS + CLAIM_COUNTS,
    'every source + mix': CORE + RAIN + TRACK + CENSUS + RATIOS + CLAIM_COUNTS,
    'compact hazards': POLICY + ['building_age', 'foundation', 'occupancy', 'zone', 'state',
                                'report_lag_days', 'rain_5d_mm', 'rain_max1d_mm',
                                'trk_dist_km', 'trk_wind_kt', 'trk_right_side'],
}
CATEGORICAL = ['state', 'zone', 'foundation', 'occupancy', 'numberOfFloorsInTheInsuredBuilding',
               'county_group', 'storm', 'income_q', 'severity_band', 'itv_band', 'deductible_band']


def prepare(X, scale=False):
    cat = [c for c in X if c in CATEGORICAL or not pd.api.types.is_numeric_dtype(X[c])]
    num = [c for c in X if c not in cat]
    numeric = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()) if scale else SimpleImputer(strategy='median', add_indicator=True)
    return ColumnTransformer([
        ('number', numeric, num),
        ('category', make_pipeline(SimpleImputer(strategy='most_frequent'),
                                  OneHotEncoder(handle_unknown='ignore', sparse_output=False)), cat)],
        sparse_threshold=0)


def model_for(X, task, iterations=140, leaves=15, l2=10):
    cls = HistGradientBoostingRegressor if task == 'regression' else HistGradientBoostingClassifier
    kw = {'loss': 'poisson'} if task == 'regression' else {}
    model = cls(max_iter=iterations, max_leaf_nodes=leaves, min_samples_leaf=50,
                l2_regularization=l2, learning_rate=.06, early_stopping=False,
                random_state=SEED, **kw)
    return make_pipeline(prepare(X), model)


def candidate_model(X, task, family='boosting'):
    if family == 'dollar ridge':
        return make_pipeline(prepare(X, scale=True), Ridge(alpha=100))
    if family.startswith('tree depth '):
        depth = int(family.rsplit(' ', 1)[-1])
        return make_pipeline(prepare(X), DecisionTreeRegressor(
            max_depth=depth, min_samples_leaf=30, random_state=SEED))
    if family in ['PCA boosting', 'top 10 boosting']:
        reduction = PCA(n_components=.80, svd_solver='full') if family == 'PCA boosting' else SelectKBest(f_regression, k=10)
        learner = model_for(X, task).steps[-1][1]
        return make_pipeline(prepare(X, scale=True), reduction, learner)
    if family == 'coverage fraction':
        return CoverageFractionModel()
    if family == 'flexible boosting':
        return model_for(X, task, iterations=300, leaves=31, l2=0)
    if family == 'boosting':
        return model_for(X, task)
    if family == 'small boosting':
        return model_for(X, task, iterations=200, leaves=7, l2=100)
    if family == 'forest':
        cls = RandomForestRegressor if task == 'regression' else RandomForestClassifier
        extra = {} if task == 'regression' else {'class_weight': 'balanced'}
        return make_pipeline(prepare(X), cls(n_estimators=120, max_depth=18,
                             min_samples_leaf=15, max_samples=.6, n_jobs=-1,
                             random_state=SEED, **extra))
    if task == 'regression':
        model = TransformedTargetRegressor(regressor=Ridge(alpha=100),
                                            func=np.log1p, inverse_func=np.expm1)
    else:
        model = LogisticRegression(C=.1, class_weight='balanced', max_iter=1500)
    return make_pipeline(prepare(X, scale=True), model)


class CoverageFractionModel:
    """Learn the paid share of coverage, then convert it back to dollars.

    The cap is a modeling assumption, not a correction to the observed target.
    Claims exceeding their recorded coverage stay in evaluation.
    """
    def fit(self, X, y):
        coverage = np.expm1(X['log_totalBuildingInsuranceCoverage'])
        valid = coverage > 0
        fraction = (np.asarray(y)[valid] / coverage[valid]).clip(0, 1)
        self.model = model_for(X, 'regression', iterations=220, leaves=15, l2=10)
        self.model.fit(X.loc[valid], fraction)
        return self

    def predict(self, X):
        coverage = np.expm1(X['log_totalBuildingInsuranceCoverage']).clip(lower=0)
        return np.clip(self.model.predict(X), 0, 1) * coverage


def regression_scores(y, p):
    return {'R2': r2_score(y, p), 'MAE $': mean_absolute_error(y, p),
            'median error $': median_absolute_error(y, p)}


def classification_scores(y, p, threshold=.5):
    pred = np.asarray(p) >= threshold
    return {'precision': precision_score(y, pred, zero_division=0),
            'recall': recall_score(y, pred, zero_division=0), 'F1': f1_score(y, pred, zero_division=0),
            'ROC-AUC': roc_auc_score(y, p) if len(np.unique(y)) > 1 else np.nan,
            'PR-AUC': average_precision_score(y, p) if len(np.unique(y)) > 1 else np.nan}


def save_table(table, name):
    RESULTS.mkdir(exist_ok=True)
    table.to_csv(RESULTS / (name + '.csv'))


def neighborhood_groups(d):
    """Spatial groups nested within storms; missing events remain split by year."""
    event = d.floodEvent.fillna('unnamed-' + d.loss_year.astype(str))
    geo = d.censusGeoid.where(~d.censusGeoid.isin(['unknown', 'nan']), code(d.countyCode))
    return event.astype(str) + ':' + geo.astype(str)


def cluster_bootstrap_indices(groups, rng):
    codes, unique = pd.factorize(groups, sort=False)
    order = np.argsort(codes, kind='stable')
    blocks = np.split(order, np.flatnonzero(np.diff(codes[order])) + 1)
    # Draw entire neighborhoods, retaining claims that share a local event.
    return np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
