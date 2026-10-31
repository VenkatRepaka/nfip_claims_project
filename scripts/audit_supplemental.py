"""Validation-driven sensitivity checks; run after audit_predictions.py."""
from audit_predictions import *
import sys

def bounded_ratio(d,tr,te,loss):
    # Explicitly alternative target: cap payment/coverage at one and exclude zero coverage.
    # This is not a repair of the payment labels; scoring retains original, uncapped dollars.
    tr = tr & d.totalBuildingInsuranceCoverage.gt(0).to_numpy()
    X=features(d); pp=prep(X);a=pp.fit_transform(X.loc[tr]);b=pp.transform(X.loc[te])
    ratio=(d.loc[tr,'amountPaidOnBuildingClaim']/d.loc[tr,'totalBuildingInsuranceCoverage']).clip(0,1)
    m=HistGradientBoostingRegressor(loss=loss,max_iter=200,learning_rate=.05,max_leaf_nodes=15,
        min_samples_leaf=100,l2_regularization=10,early_stopping=False,random_state=SEED).fit(a,ratio)
    return np.clip(m.predict(b),0,1)*d.loc[te,'totalBuildingInsuranceCoverage'].to_numpy()

def main_extra():
    d=load_data();paid=d.amountPaidOnBuildingClaim.gt(0).to_numpy();v=[]
    reuse = '--reuse-validation' in sys.argv
    if reuse:
        v = pd.read_csv(OUT/'regression_bounded_ratio_validation.csv').to_dict('records')
    for end in ([] if reuse else [2021,2022]):
        tr=d.year.le(end).to_numpy() & paid; te=d.year.eq(end+1).to_numpy() & paid
        for loss in ['squared_error','poisson']:
            p=bounded_ratio(d,tr,te,loss)
            v.append({'fit_through':end,'validation_year':end+1,'model':'bounded_ratio_'+loss,**reg_scores(d.loc[te,'amountPaidOnBuildingClaim'],p)})
    save('regression_bounded_ratio_validation',v)
    # Compare all candidates on pre-2024 validation only, then score the winner.
    old=pd.read_csv(OUT/'regression_forward_validation.csv')
    assert len(old) == 12 and old.groupby('model').size().eq(2).all(), 'Wait for audit_predictions.py to finish first.'
    rank=pd.concat([old,pd.DataFrame(v)]).groupby('model').r2_dollars.mean().sort_values(ascending=False)
    print('VALIDATION RANKING',rank.to_string(),flush=True)
    best=rank.index[0]
    if best.startswith('bounded_ratio_'):
        tr=d.year.le(2023).to_numpy() & paid;te=d.year.ge(2024).to_numpy() & paid
        p=bounded_ratio(d,tr,te,best.removeprefix('bounded_ratio_'))
        save('regression_bounded_ratio_selected_test',[{'model':best,**reg_scores(d.loc[te,'amountPaidOnBuildingClaim'],p)}])
        print('BOUNDED RATIO FINAL',pd.read_csv(OUT/'regression_bounded_ratio_selected_test.csv').to_string(index=False),flush=True)
    else:
        (OUT/'regression_bounded_ratio_decision.json').write_text(json.dumps({'winner':best,'decision':'bounded ratio did not win forward validation; no additional future-year score computed'},indent=2))
    if reuse and (OUT/'clustering_reproduction.csv').exists():
        return
    # Re-run a small clustering check with the exact original sample and feature representation.
    old=pd.read_parquet(ROOT/'notebooks/outputs/nfip_claims_simple_clean.parquet')
    fl=json.loads((ROOT/'notebooks/outputs/simple_feature_lists.json').read_text())
    w=old[old.foundationType.ne('0')].copy();w['occupancyType']=w.occupancyType.replace({1:11,2:12,3:13,4:18,6:18})
    s=w.sample(20000,random_state=SEED)
    from sklearn.cluster import KMeans
    from sklearn.metrics import silhouette_score,adjusted_rand_score
    cat=[c for c in fl['RQ3'] if c in fl['NOMINAL_FEATURES']]
    X=pd.get_dummies(s[fl['RQ3']],columns=cat,dtype=int);X.totalBuildingInsuranceCoverage=np.log1p(X.totalBuildingInsuranceCoverage)
    Z=StandardScaler().fit_transform(X)
    rows=[]
    for k in [2,7]:
        m=KMeans(k,n_init=10,random_state=SEED).fit(Z)
        counts=pd.Series(m.labels_).value_counts()
        rows.append({'representation':'building profiles','k':k,'silhouette':silhouette_score(Z,m.labels_,sample_size=5000,random_state=SEED),'largest_share':counts.max()/len(s)})
    for name,frame,k in [('coordinates',s[['latitude','longitude']],4),('zone_foundation',pd.get_dummies(s[['flood_zone_family','foundationType']],dtype=int),8)]:
        z=StandardScaler().fit_transform(frame);m=KMeans(k,n_init=10,random_state=SEED).fit(z)
        rows.append({'representation':name,'k':k,'silhouette':silhouette_score(z,m.labels_,sample_size=5000,random_state=SEED),'largest_share':pd.Series(m.labels_).value_counts().max()/len(s)})
    print(save('clustering_reproduction',rows).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main_extra()
