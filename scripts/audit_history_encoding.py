"""Isolate self-inclusion in the original random-split history experiment.
This reproduces retrospective scoring only: it does not fix chronological availability.
"""
from audit_predictions import *
from sklearn.model_selection import KFold

def main_history():
    nb=json.loads((ROOT/'notebooks/06_rq1_national_and_regional.ipynb').read_text())
    env={'__name__':'audit_history_notebook_cells'}
    before=Path.cwd()
    try:
        os.chdir(ROOT/'notebooks')
        for i in [2,3,5]:exec(compile(''.join(nb['cells'][i]['source']),f'notebook06_cell_{i}','exec'),env)
    finally:os.chdir(before)
    tr,te=env['SPLITS']['A. known storms'];y=env['y'];paid=env['paid'];F=env['F'];history=env['history']
    rows=[]
    for variant in ['self_inclusive_original','out_of_fold_train_history','no_target_history']:
        a,b=F.loc[tr].copy(),F.loc[te].copy()
        if variant!='no_target_history':
            for name,key in [('county_history',paid.countyCode),('storm_history',env['storm'])]:
                if variant=='self_inclusive_original':a[name]=history(key,tr,tr)
                else:
                    a[name]=np.nan
                    for fit_i,val_i in KFold(5,shuffle=True,random_state=SEED).split(tr):
                        a.loc[tr[val_i],name]=history(key,tr[fit_i],tr[val_i])
                b[name]=history(key,tr,te)
        m=env['model']().fit(a,y.loc[tr]);p=m.predict(b)
        rows.append({'variant':variant,**reg_scores(y.loc[te],p)})
    print(save('history_encoding_diagnostic',rows).to_string(index=False),flush=True)

if __name__=='__main__':
    with threadpool_limits(limits=4):main_history()
