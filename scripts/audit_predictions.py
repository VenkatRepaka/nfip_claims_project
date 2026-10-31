"""Reproducible audit. Run from any directory: python scripts/audit_predictions.py.
Raw snapshot is required; override NFIP_RAW_PATH if it has moved.
Existing notebooks and their outputs are read only. No test-driven parameter search.
"""
from pathlib import Path
import os, json, time, platform, hashlib
import numpy as np
import pandas as pd
from sklearn import __version__ as sklearn_version
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import make_pipeline
from sklearn.ensemble import HistGradientBoostingRegressor, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import (r2_score, mean_absolute_error, mean_squared_error, roc_auc_score,
    average_precision_score, precision_score, recall_score, f1_score, brier_score_loss, log_loss)
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'reports/prediction_audit'
OUT.mkdir(parents=True, exist_ok=True)
RAW = Path(os.environ.get('NFIP_RAW_PATH', '/Users/home/workspace/School/DataMining/project/NfipClaimsV3.parquet'))
SEED = 42

def save(name, rows):
    table = pd.DataFrame(rows)
    table.to_csv(OUT / (name + '.csv'), index=False)
    return table

def load_data():
    d = pd.read_parquet(RAW, filters=[('yearOfLoss', '>=', 2020), ('yearOfLoss', '<=', 2025)])
    keys = pd.read_parquet(ROOT / 'notebooks/outputs/nfip_claims_simple_keys.parquet')
    assert d.id.is_unique and keys.id.is_unique
    d = d.set_index('id').loc[keys.id].reset_index()
    assert np.array_equal(d.id, keys.id)
    for c in ['dateOfLoss', 'originalConstructionDate', 'originalNBDate', 'openDate']:
        d[c] = pd.to_datetime(d[c].astype(str), errors='coerce', format='%Y-%m-%d')
        d[c] = d[c].where(d[c].dt.year.between(1800, 2025))
    for c in ['amountPaidOnBuildingClaim', 'netBuildingPaymentAmount', 'latitude', 'longitude']:
        d[c] = pd.to_numeric(d[c], errors='coerce').astype(float)
    d.loc[d.buildingDamageAmount.eq(1), 'buildingDamageAmount'] = np.nan
    # Leave unknown targets unknown. Clip negative recoveries only for the paid-only cohort.
    d['year'] = d.dateOfLoss.dt.year
    return d

CAT = ['ratedFloodZone', 'foundationType', 'occupancyType', 'basementEnclosureCrawlspaceType',
       'numberOfFloorsInTheInsuredBuilding', 'condominiumCoverageTypeCode', 'state']
NUM = ['totalBuildingInsuranceCoverage', 'totalContentsInsuranceCoverage', 'policyCount', 'numberOfUnits',
       'building_age', 'policy_tenure', 'report_lag', 'loss_month', 'bldg_deductible', 'contents_deductible',
       'elevatedBuildingIndicator', 'postFIRMConstructionIndicator', 'primaryResidenceIndicator',
       'rentalPropertyIndicator', 'smallBusinessIndicatorBuilding', 'crsClassCode']
DED = {'0': 500, '1': 1000, '2': 2000, '3': 3000, '4': 4000, '5': 5000, '9': 750,
       'A': 10000, 'B': 15000, 'C': 20000, 'D': 25000, 'E': 50000, 'F': 1250, 'G': 1500, 'H': 200}

def features(d, coords=False):
    X = d[CAT + [c for c in NUM if c in d]].copy()
    X['occupancyType'] = d.occupancyType.replace({1:11, 2:12, 3:13, 4:18, 6:18})
    age = (d.dateOfLoss - d.originalConstructionDate).dt.days / 365.25
    X['building_age'] = age.where(age.between(0, 250))
    tenure = (d.dateOfLoss - d.originalNBDate).dt.days / 365.25
    X['policy_tenure'] = tenure.where(tenure.between(0, 100))
    lag = (d.openDate - d.dateOfLoss).dt.days
    X['report_lag'] = lag.where(lag.ge(0)).clip(upper=365)
    X['loss_month'] = d.dateOfLoss.dt.month
    X['bldg_deductible'] = d.buildingDeductibleCode.map(DED)
    X['contents_deductible'] = d.contentsDeductibleCode.map(DED)
    X['crsClassCode'] = X.crsClassCode.where(X.crsClassCode.between(1, 10))
    for c in CAT:
        X[c] = X[c].astype('string').fillna('unknown').astype(str)
    if coords:
        X[['latitude', 'longitude']] = d[['latitude', 'longitude']]
    return X

def prep(X, linear=False):
    cats = [c for c in CAT if c in X]
    nums = [c for c in X if c not in cats]
    num_pipe = make_pipeline(SimpleImputer(strategy='median', add_indicator=True), StandardScaler()) if linear else SimpleImputer(strategy='median', add_indicator=True)
    return ColumnTransformer([('num', num_pipe, nums), ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False), cats)], sparse_threshold=0)

def reg_scores(y, p):
    return {'r2_dollars': r2_score(y,p), 'mae_dollars': mean_absolute_error(y,p),
            'median_error_dollars': float(np.median(np.abs(np.asarray(y)-p))),
            'rmse_dollars': float(np.sqrt(mean_squared_error(y,p)))}

def cls_scores(y, p, t):
    pred = p >= t
    return {'threshold': float(t), 'prevalence': float(np.mean(y)), 'roc_auc': roc_auc_score(y,p),
            'average_precision': average_precision_score(y,p), 'brier': brier_score_loss(y,p),
            'log_loss': log_loss(y,p), 'precision': precision_score(y,pred,zero_division=0),
            'recall': recall_score(y,pred,zero_division=0), 'f1': f1_score(y,pred,zero_division=0),
            'flagged_share': float(pred.mean())}

SPECS = {
    'direct_poisson': dict(loss='poisson', max_leaf_nodes=15, l2_regularization=10),
    'direct_squared': dict(loss='squared_error', max_leaf_nodes=15, l2_regularization=10),
    'coverage_ratio_squared': dict(loss='squared_error', max_leaf_nodes=15, l2_regularization=10),
    'coverage_ratio_poisson': dict(loss='poisson', max_leaf_nodes=15, l2_regularization=10),
    'log_squared': dict(loss='squared_error', max_leaf_nodes=15, l2_regularization=10),
    'direct_squared_coords': dict(loss='squared_error', max_leaf_nodes=15, l2_regularization=10),
}

def fit_reg(d, tr, te, name):
    X = features(d, coords=name.endswith('_coords'))
    pp = prep(X)
    a = pp.fit_transform(X.loc[tr]); b = pp.transform(X.loc[te])
    y = d.loc[tr,'amountPaidOnBuildingClaim'].to_numpy()
    cov = d.totalBuildingInsuranceCoverage.clip(lower=1).to_numpy()
    target = y / cov[tr] if name.startswith('coverage_ratio') else np.log1p(y) if name.startswith('log_') else y
    m = HistGradientBoostingRegressor(max_iter=200, learning_rate=.05, min_samples_leaf=100,
            early_stopping=False, random_state=SEED, **SPECS[name]).fit(a,target)
    p = m.predict(b)
    if name.startswith('coverage_ratio'):p = np.clip(p,0,1)*cov[te]
    elif name.startswith('log_'):p = np.expm1(p)
    return np.maximum(p,0)

def main():
    t0 = time.time()
    d = load_data()
    train = d.year.le(2023).to_numpy(); test = d.year.ge(2024).to_numpy()
    paid = d.amountPaidOnBuildingClaim.gt(0).to_numpy()
    known = d.buildingDamageAmount.notna().to_numpy()
    cutoff = float(d.loc[train & known,'buildingDamageAmount'].quantile(.8))
    old = pd.read_parquet(ROOT/'notebooks/outputs/nfip_claims_simple_clean.parquet')
    full_cutoff = float(d.loc[known,'buildingDamageAmount'].quantile(.8))
    meta = {'raw_path':str(RAW),'raw_bytes':RAW.stat().st_size,'rows':len(d),'python':platform.python_version(),
            'sklearn':sklearn_version,'numpy':np.__version__,'pandas':pd.__version__, 'seed':SEED,
            'target_cutoff_train_2020_2023':cutoff,'target_cutoff_all_years':full_cutoff,
            'old_label_disagreements_with_train_cutoff':int(((d.buildingDamageAmount>=cutoff).to_numpy()!=old.high_severity.eq(1).to_numpy())[known].sum()),
            'unknown_building_payments':int(d.amountPaidOnBuildingClaim.isna().sum()),
            'unknown_payment_positive_net':int((d.amountPaidOnBuildingClaim.isna() & d.netBuildingPaymentAmount.gt(0)).sum()),
            'paid_above_building_coverage':int((paid & d.amountPaidOnBuildingClaim.gt(d.totalBuildingInsuranceCoverage).to_numpy()).sum()),
            'selection_protocol':'two forward folds: <=2021 -> 2022 and <=2022 -> 2023; choose mean validation dollar R2; refit <=2023; test 2024-25',
            'limitations':['cohort uses existing deduplication keys; snapshot features are not point-in-time verified',
                           '2024-25 already inspected in historical notebooks; this is an audited retrospective comparison, not a fresh prospective holdout']}
    (OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    print('DATA',json.dumps(meta),flush=True)
    save('cohort_by_year',[{'year':int(yr),'rows':len(g),'paid':int(g.amountPaidOnBuildingClaim.gt(0).sum()),
        'damage_known':int(g.buildingDamageAmount.notna().sum()),'severe_rate_train_cutoff':float(g.loc[g.buildingDamageAmount.notna(),'buildingDamageAmount'].ge(cutoff).mean())}
        for yr,g in d.groupby('year')])
    # Reproduce original 04 national dollar results, not a new honest holdout.
    fl = json.loads((ROOT/'notebooks/outputs/simple_feature_lists.json').read_text())
    Xold = pd.get_dummies(old[fl['NOMINAL_FEATURES']+fl['NUMERIC_FEATURES']],columns=fl['NOMINAL_FEATURES'],dtype=int)[fl['RQ1']]
    tr, te = train & paid, test & paid
    ytr = d.loc[tr,'amountPaidOnBuildingClaim']; yte = d.loc[te,'amountPaidOnBuildingClaim']
    ref=[]
    for name,p in [('training_median',np.full(te.sum(),ytr.median())),('training_mean',np.full(te.sum(),ytr.mean()))]:
        ref.append({'model':name,**reg_scores(yte,p)})
    lin=Xold.copy()
    for c in fl['LOG_COLS']:
        if c in lin:lin[c]=np.log1p(lin[c])
    m=make_pipeline(StandardScaler(),Ridge(alpha=1)).fit(lin.loc[tr],np.log1p(ytr))
    ref.append({'model':'original_ridge_log',**reg_scores(yte,np.expm1(m.predict(lin.loc[te])) )})
    m=HistGradientBoostingRegressor(loss='poisson',max_iter=400,learning_rate=.05,random_state=SEED).fit(Xold.loc[tr],ytr)
    ref.append({'model':'original_poisson_15',**reg_scores(yte,m.predict(Xold.loc[te]))})
    print(save('original_regression_reproduction',ref).round(4).to_string(index=False),flush=True)
    # Select candidates with older data only. No early stopping on a random row subset.
    validation=[]
    for end in [2021,2022]:
        a=d.year.le(end).to_numpy() & paid; b=d.year.eq(end+1).to_numpy() & paid
        for name in SPECS:
            p=fit_reg(d,a,b,name)
            r={'fit_through':end,'validation_year':end+1,'model':name,**reg_scores(d.loc[b,'amountPaidOnBuildingClaim'],p)}
            validation.append(r);save('regression_forward_validation',validation)
            print('VALIDATION',r,flush=True)
    tab=pd.DataFrame(validation).groupby('model').r2_dollars.mean().sort_values(ascending=False)
    chosen=tab.index[0];meta['selected_regression']=chosen
    # Only the preselected winner is evaluated on the later years.
    p=fit_reg(d,tr,te,chosen)
    pd.DataFrame({'id':d.loc[te,'id'], 'year':d.loc[te,'year'], 'state':d.loc[te,'state'],
        'actual':yte.to_numpy(), 'predicted':p}).to_parquet(OUT/'regression_predictions.parquet',index=False)
    final=[{'scope':'National','model':chosen,'n':int(te.sum()),**reg_scores(yte,p)}]
    for scope,states in [('Florida',['FL']),('Gulf',['LA','TX','MS','AL']),('Northeast',['NJ','NY','CT','RI','MA','MD','DE','DC','NH','ME']),('2024',None),('2025',None)]:
        mask=d.loc[te,'state'].isin(states).to_numpy() if states else d.loc[te,'year'].eq(int(scope)).to_numpy()
        final.append({'scope':scope,'model':chosen,'n':int(mask.sum()),**reg_scores(yte.to_numpy()[mask],p[mask])})
    save('regression_selected_test',final)
    # One fixed comparative loss, declared in advance, shows whether optimizing MAE changes the answer.
    # Classification: cutoff fixed from <=2023, calibration/tuning use 2023 only.
    y=d.buildingDamageAmount.ge(cutoff).astype(int).to_numpy()
    a=d.year.le(2022).to_numpy() & known; v=d.year.eq(2023).to_numpy() & known; b=test & known
    X=features(d);pp=prep(X);Xa=pp.fit_transform(X.loc[a]);Xv=pp.transform(X.loc[v]);Xb=pp.transform(X.loc[b])
    candidates={
      'unweighted_hgb':HistGradientBoostingClassifier(max_iter=200,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=10,early_stopping=False,random_state=SEED),
      'balanced_hgb':HistGradientBoostingClassifier(class_weight='balanced',max_iter=200,learning_rate=.05,max_leaf_nodes=15,min_samples_leaf=100,l2_regularization=10,early_stopping=False,random_state=SEED),
    }
    cvrows=[]; fitted={}
    for name,m in candidates.items():
        m.fit(Xa,y[a]);pv=m.predict_proba(Xv)[:,1]
        fitted[name]=(m,pv)
        cvrows.append({'model':name,**cls_scores(y[v],pv,.5)})
    save('classification_validation',cvrows)
    # Fix model by validation average precision, calibrate sigmoid on 2023.
    selected=max(cvrows,key=lambda r:r['average_precision'])['model'];m,pv=fitted[selected]
    logits=lambda p:np.log(np.clip(p,1e-6,1-1e-6)/(1-np.clip(p,1e-6,1-1e-6))).reshape(-1,1)
    cal=LogisticRegression(C=1e6).fit(logits(pv),y[v]);pcv=cal.predict_proba(logits(pv))[:,1]
    pt=m.predict_proba(Xb)[:,1];pct=cal.predict_proba(logits(pt))[:,1]
    grid=np.linspace(.05,.95,181);f1s=[f1_score(y[v],pcv>=t) for t in grid];threshold=float(grid[int(np.argmax(f1s))])
    precision_candidates=[t for t in grid if (pcv>=t).sum()>=100 and precision_score(y[v],pcv>=t,zero_division=0)>=.8]
    high_t=float(min(precision_candidates)) if precision_candidates else None
    cls=[{'model':selected,'variant':'uncalibrated_0.5',**cls_scores(y[b],pt,.5)},
         {'model':selected,'variant':'calibrated_0.5',**cls_scores(y[b],pct,.5)},
         {'model':selected,'variant':'calibrated_validation_F1',**cls_scores(y[b],pct,threshold)}]
    if high_t is not None:cls.append({'model':selected,'variant':'validation_precision_80pct',**cls_scores(y[b],pct,high_t)})
    save('classification_test',cls)
    pd.DataFrame({'id':d.loc[b,'id'], 'year':d.loc[b,'year'], 'actual':y[b],
        'raw_probability':pt, 'calibrated_probability':pct}).to_parquet(OUT/'classification_predictions.parquet',index=False)
    save('classification_fixed_confidence',[
        {'probability_cut':q, 'positive_predictions':int((pct>=q).sum()),
         'positive_precision':float(y[b][pct>=q].mean()) if (pct>=q).any() else np.nan,
         'positive_recall':float(((pct>=q)&(y[b]==1)).sum()/y[b].sum()),
         'retained_share_with_abstention':float(((pct>=q)|(pct<=1-q)).mean()),
         'retained_accuracy':float(((pct>=.5)==y[b])[(pct>=q)|(pct<=1-q)].mean())}
         for q in [.7,.8,.9]])
    # Reliability tables and coverage of abstaining on uncertain predictions. No claim of guaranteed confidence.
    bins=pd.cut(pct,np.linspace(0,1,11),include_lowest=True)
    reliability=pd.DataFrame({'bin':bins,'probability':pct,'actual':y[b]}).groupby('bin',observed=True).agg(n=('actual','size'),mean_probability=('probability','mean'),observed_severe_rate=('actual','mean')).reset_index()
    reliability.to_csv(OUT/'classification_reliability.csv',index=False)
    save('classification_by_year',[{'year':yr,**cls_scores(y[b][d.loc[b,'year'].eq(yr).to_numpy()],pct[d.loc[b,'year'].eq(yr).to_numpy()],threshold)} for yr in [2024,2025]])
    meta.update({'selected_classification':selected,'classification_fit_through':2022,'classification_calibration_year':2023,
                 'classification_f1_threshold':threshold,'classification_80pct_precision_threshold':high_t,
                 'classification_calibration_warning':'calibration and threshold share 2023; all reported final metrics use 2024-25, not calibration-fit scores',
                 'runtime_seconds':time.time()-t0})
    (OUT/'run_metadata.json').write_text(json.dumps(meta,indent=2))
    print('FINAL REGRESSION',pd.DataFrame(final).to_string(index=False),flush=True)
    print('FINAL CLASSIFICATION',pd.DataFrame(cls).to_string(index=False),flush=True)
    print('DONE',round(time.time()-t0,1),'seconds',flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main()
