"""Step 5 - feature selection. Targets:
  T1 regression      : log1p(total_paid = building + contents) on PAID claims
  T2 classification  : is_paid (total_paid > 0)
  T3 classification  : high_severity (total_paid >= top-decile threshold of paid claims in training years)
Evidence: Spearman correlation matrix, mutual information, RandomForest impurity importance, XGBoost gain importance and
permutation importance on the held-out period (2024-25). Time-based split: train = 2020-2023, test = 2024-2025.
Feature sets: A = known at notice (policy/property/location/date); B = A + adjuster-assessed fields; LEAK = B + post-outcome columns (demonstration only)."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.feature_selection import mutual_info_regression, mutual_info_classif
from sklearn.inspection import permutation_importance
from sklearn.metrics import r2_score, mean_squared_error, roc_auc_score, average_precision_score
from sklearn.preprocessing import OrdinalEncoder
from xgboost import XGBRegressor, XGBClassifier
from common import *
sns.set_theme(style="whitegrid")

df = pd.read_parquet(ROOT / "cleaned_claims.parquet")
meta = load_json("clean_meta.json"); TRAIN_MAX = meta["train_max_year"]

CAT = ["occupancy_std", "flood_zone_family", "rating_group", "foundation_type", "basement_type", "contents_location", "building_use",
       "condo_type", "state_grp", "num_floors_code", "flood_event_grp"]
CAT_B = ["cause_group", "repl_cost_basis"]
NUM_A = ["log_bldg_coverage", "log_cont_coverage", "log_policy_count", "log_num_units", "building_age", "policy_tenure_years", "report_lag_days",
         "loss_month_sin", "loss_month_cos", "loss_dow", "latitude", "longitude", "bldg_deductible_usd", "cont_deductible_usd", "crs_class",
         "elevated_building", "post_firm", "primary_residence", "rental_property", "small_business", "is_rr2", "no_contents_cov", "no_building_cov",
         "crs_unknown", "named_event", "has_elevation_data", "has_elevation_cert", "construction_date_jan1", "bldg_age_invalid_or_missing",
         "tenure_invalid_or_missing", "bldg_deductible_missing", "cont_deductible_missing"]
NUM_B = ["log_bldg_damage", "log_bldg_value", "log_repl_cost", "log_cont_damage", "log_cont_value", "water_depth", "interior_depth", "exterior_depth",
         "damage_to_value_ratio", "coverage_to_repl_ratio", "adjuster_info_missing", "cont_damage_missing", "water_depth_missing",
         "interior_depth_missing", "exterior_depth_missing", "bldg_value_recoded_missing"]
# post-outcome columns, used ONLY to demonstrate leakage
df["leak_log_net_bldg_payment"] = np.log1p(df.net_building_payment.clip(lower=0))
df["leak_log_net_cont_payment"] = np.log1p(df.net_contents_payment.clip(lower=0))
df["leak_nonpay_reason_building_present"] = df.nonpay_reason_building.notna().astype("int8")
df["leak_nonpay_reason_contents_present"] = df.nonpay_reason_contents.notna().astype("int8")
LEAK = ["leak_log_net_bldg_payment", "leak_log_net_cont_payment", "leak_nonpay_reason_building_present", "leak_nonpay_reason_contents_present"]
SETS = {"A": NUM_A + CAT, "B": NUM_A + CAT + NUM_B + CAT_B, "LEAK": NUM_A + CAT + NUM_B + CAT_B + LEAK}
TIER = {**{c: "A" for c in NUM_A + CAT}, **{c: "B" for c in NUM_B + CAT_B}, **{c: "LEAK" for c in LEAK}}

allf = SETS["LEAK"]; X = df[allf].copy()
cat_all = [c for c in CAT + CAT_B]
X[cat_all] = OrdinalEncoder().fit_transform(X[cat_all].astype(str))
is_bin = {c: (X[c].nunique() <= 2) for c in allf}
train = (df.yearOfLoss <= TRAIN_MAX).values; test = ~train
paid = (df.is_paid == 1).values
TARGETS = {
    "T1_reg_log_total_paid (paid claims)": dict(y=df.log_total_paid.values, kind="reg", mask=paid),
    "T2_cls_is_paid": dict(y=df.is_paid.values, kind="cls", mask=np.ones(len(df), bool)),
    "T3_cls_high_severity": dict(y=df.high_severity.values, kind="cls", mask=np.ones(len(df), bool)),
}
summary = {"train_rows": int(train.sum()), "test_rows": int(test.sum()),
           "paid_rate_train": float(df.is_paid[train].mean()), "paid_rate_test": float(df.is_paid[test].mean()),
           "high_sev_rate_train": float(df.high_severity[train].mean()), "high_sev_rate_test": float(df.high_severity[test].mean()),
           "paid_claims_train": int((train & paid).sum()), "paid_claims_test": int((test & paid).sum())}

# ------------------------------------------------------------ correlation matrix
num_for_corr = [c for c in NUM_A + NUM_B if not c.endswith("_missing") and c not in ("loss_dow",)] + ["log_total_paid", "is_paid", "high_severity"]
samp = df.sample(100000, random_state=SEED)
corr = samp[num_for_corr].corr(method="spearman")
corr.to_csv(OUT / "corr_spearman_numeric.csv")
fig, ax = plt.subplots(figsize=(16, 13))
sns.heatmap(corr, cmap="RdBu_r", center=0, vmin=-1, vmax=1, ax=ax, annot=True, fmt=".1f", annot_kws={"size": 5}, cbar_kws={"shrink": .6})
ax.set_title("Spearman correlation (100k-row sample, cleaned/imputed numeric features + targets)"); ax.tick_params(labelsize=7)
plt.tight_layout(); plt.savefig(FIG / "09_correlation_matrix.png", dpi=120); plt.close()
fc = [c for c in num_for_corr if c not in ("log_total_paid", "is_paid", "high_severity")]
cc = corr.loc[fc, fc].where(np.triu(np.ones((len(fc), len(fc))), 1).astype(bool)).stack()
pairs = cc[cc.abs() >= 0.80].sort_values(key=abs, ascending=False).reset_index(); pairs.columns = ["feature_1", "feature_2", "spearman"]
pairs.to_csv(OUT / "corr_high_pairs.csv", index=False)
tcorr = corr.loc[fc, ["log_total_paid", "is_paid", "high_severity"]]
# (note: log_total_paid correlation here is over ALL claims incl. unpaid; paid-only version below)
paid_s = samp[samp.is_paid == 1]
tcorr["spearman_vs_log_total_paid_PAID_ONLY"] = paid_s[fc + ["log_total_paid"]].corr(method="spearman")["log_total_paid"].reindex(fc)
tcorr.to_csv(OUT / "corr_with_targets.csv")

# ------------------------------------------------------------ mutual information (train sample)
mi_rows = []
rs = np.random.RandomState(SEED)
for tname, T in TARGETS.items():
    idx = np.where(train & T["mask"])[0]; idx = rs.choice(idx, min(60000, len(idx)), replace=False)
    Xs = X.iloc[idx][allf].values; ys = T["y"][idx]
    disc = np.array([c in cat_all or is_bin[c] for c in allf])
    mi = (mutual_info_regression if T["kind"] == "reg" else mutual_info_classif)(Xs, ys, discrete_features=disc, random_state=SEED)
    mi_rows.append(pd.DataFrame({"target": tname, "feature": allf, "mi": mi}))
mi_df = pd.concat(mi_rows); mi_df.to_csv(OUT / "mutual_information.csv", index=False)

# ------------------------------------------------------------ models: performance by feature set + importances
def fit_xgb(kind, Xtr, ytr, w=None):
    if kind == "reg": m = XGBRegressor(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=8, random_state=SEED)
    else:
        spw = (ytr == 0).sum() / max((ytr == 1).sum(), 1)
        m = XGBClassifier(n_estimators=300, max_depth=6, learning_rate=0.08, subsample=0.8, colsample_bytree=0.8, tree_method="hist", n_jobs=8, random_state=SEED,
                          scale_pos_weight=min(spw, 10), eval_metric="logloss")
    return m.fit(Xtr, ytr)
def metrics(kind, m, Xte, yte):
    if kind == "reg":
        p = m.predict(Xte); return dict(R2=r2_score(yte, p), RMSE_log=float(np.sqrt(mean_squared_error(yte, p))))
    p = m.predict_proba(Xte)[:, 1]; return dict(AUC=roc_auc_score(yte, p), PR_AUC=average_precision_score(yte, p), base_rate=float(yte.mean()))

perf = []; imp_all = []
for tname, T in TARGETS.items():
    tr = train & T["mask"]; te = test & T["mask"]; y = T["y"]
    for sname, cols in SETS.items():
        m = fit_xgb(T["kind"], X.loc[tr, cols], y[tr])
        r = dict(target=tname, feature_set=sname, n_features=len(cols), model="XGBoost", n_train=int(tr.sum()), n_test=int(te.sum())); r.update(metrics(T["kind"], m, X.loc[te, cols], y[te])); perf.append(r)
        print(r)
        if sname in ("B", "A"):
            gain = pd.Series(m.get_booster().get_score(importance_type="total_gain")).reindex(cols).fillna(0); gain = gain / gain.sum()
            ti = np.random.RandomState(SEED).choice(np.where(te)[0], min(30000, te.sum()), replace=False)
            sc = "r2" if T["kind"] == "reg" else "roc_auc"
            pi = permutation_importance(m, X.loc[ti, cols], y[ti], scoring=sc, n_repeats=3, random_state=SEED, n_jobs=1)
            # random forest impurity importance (train subsample)
            tri = np.random.RandomState(SEED).choice(np.where(tr)[0], min(100000, tr.sum()), replace=False)
            rf = (RandomForestRegressor if T["kind"] == "reg" else RandomForestClassifier)(n_estimators=150, min_samples_leaf=25, max_features=0.4, n_jobs=8, random_state=SEED,
                                                                                         **({} if T["kind"] == "reg" else {"class_weight": "balanced_subsample"}))
            rf.fit(X.loc[tri, cols], y[tri]); rfm = metrics(T["kind"], rf, X.loc[te, cols], y[te])
            perf.append(dict(target=tname, feature_set=sname, n_features=len(cols), model="RandomForest", n_train=len(tri), n_test=int(te.sum()), **rfm))
            d = pd.DataFrame({"target": tname, "feature_set": sname, "feature": cols, "xgb_gain": gain.values, "xgb_perm_drop": pi.importances_mean, "xgb_perm_sd": pi.importances_std,
                              "rf_impurity": rf.feature_importances_})
            imp_all.append(d)
perf = pd.DataFrame(perf); perf.to_csv(OUT / "model_performance_by_feature_set.csv", index=False)
imp = pd.concat(imp_all).merge(mi_df, on=["target", "feature"], how="left")
imp["tier"] = imp.feature.map(TIER)
imp.to_csv(OUT / "feature_importances.csv", index=False)

# ------------------------------------------------------------ composite ranking + redundancy pruning -> recommended lists
def rank_norm(s): return s.rank(ascending=False, method="average") / len(s)
POLICY_EXCLUDE = {"flood_event_grp": "event identity: encodes which storm, cannot generalise to unseen events (use named_event instead)"}
rec_rows = []; rec_lists = {}
cor_full = samp[[c for c in allf if c in samp and pd.api.types.is_numeric_dtype(samp[c])]].corr(method="spearman").abs() if False else None
Xs100 = X.loc[samp.index]
rho = Xs100[allf].corr(method="spearman").abs()
for (tname, sname), g in imp.groupby(["target", "feature_set"]):
    g = g.copy()
    g["rank_mi"] = rank_norm(g.mi); g["rank_rf"] = rank_norm(g.rf_impurity); g["rank_xgb_gain"] = rank_norm(g.xgb_gain); g["rank_perm"] = rank_norm(g.xgb_perm_drop)
    g["composite_rank"] = g[["rank_mi", "rank_rf", "rank_xgb_gain", "rank_perm"]].mean(axis=1)
    noise = 2 * g.xgb_perm_sd                          # permutation importance must exceed 2 SD of its own repeats to count as signal
    g["passes_perm_signal"] = (g.xgb_perm_drop > np.maximum(noise, 1e-4))
    g = g.sort_values("composite_rank")
    keep = []
    for _, r in g.iterrows():
        if r.feature in POLICY_EXCLUDE: continue
        if not r.passes_perm_signal and r.composite_rank > 0.20: continue
        if any(rho.loc[r.feature, k] >= 0.90 for k in keep): continue    # redundant with an already-kept, higher-ranked feature
        keep.append(r.feature)
    g["recommended"] = g.feature.isin(keep)
    rec_rows.append(g); rec_lists[f"{tname}|{sname}"] = keep
rec = pd.concat(rec_rows); rec.to_csv(OUT / "feature_ranking_full.csv", index=False)

# evaluate recommended lists vs full sets (refit XGB)
chk = []
for key, keep in rec_lists.items():
    tname, sname = key.split("|"); T = TARGETS[tname]; tr = train & T["mask"]; te = test & T["mask"]
    full_cols = SETS[sname]
    m = fit_xgb(T["kind"], X.loc[tr, keep], T["y"][tr]); r = dict(target=tname, feature_set=sname, n_features_full=len(full_cols), n_features_recommended=len(keep)); r.update({f"rec_{k}": v for k, v in metrics(T["kind"], m, X.loc[te, keep], T["y"][te]).items()})
    base = perf[(perf.target == tname) & (perf.feature_set == sname) & (perf.model == "XGBoost")].iloc[0]
    for k in (["R2", "RMSE_log"] if T["kind"] == "reg" else ["AUC", "PR_AUC"]): r[f"full_{k}"] = base[k]
    chk.append(r)
chk = pd.DataFrame(chk); chk.to_csv(OUT / "recommended_vs_full.csv", index=False); print(chk.T)
save_json({"policy_excluded": POLICY_EXCLUDE, "recommended": rec_lists, "summary": summary, "tier": TIER, "sets": SETS}, "feature_selection_results.json")

# ------------------------------------------------------------ figures
short = {k: k.split(" ")[0] for k in TARGETS}
fig, axes = plt.subplots(1, 3, figsize=(21, 9))
for ax, tname in zip(axes, TARGETS):
    g = rec[(rec.target == tname) & (rec.feature_set == "B")].sort_values("composite_rank").head(25).iloc[::-1]
    col = g.tier.map({"A": "#2980b9", "B": "#e67e22"}); ax.barh(g.feature, g.xgb_gain, color=col)
    ax.set_title(f"{tname}\nXGBoost gain share, feature set B (blue = tier A, orange = adjuster tier B)", fontsize=9); ax.tick_params(labelsize=8)
plt.tight_layout(); plt.savefig(FIG / "10_xgb_importance_setB.png", dpi=115); plt.close()
fig, axes = plt.subplots(1, 3, figsize=(21, 9))
for ax, tname in zip(axes, TARGETS):
    g = rec[(rec.target == tname) & (rec.feature_set == "B")].sort_values("mi", ascending=False).head(25).iloc[::-1]
    ax.barh(g.feature, g.mi, color=g.tier.map({"A": "#2980b9", "B": "#e67e22"})); ax.set_title(f"{tname}\nMutual information (nats), train sample", fontsize=9); ax.tick_params(labelsize=8)
plt.tight_layout(); plt.savefig(FIG / "11_mutual_information.png", dpi=115); plt.close()
fig, axes = plt.subplots(1, 3, figsize=(21, 9))
for ax, tname in zip(axes, TARGETS):
    g = rec[(rec.target == tname) & (rec.feature_set == "B")].sort_values("xgb_perm_drop", ascending=False).head(25).iloc[::-1]
    ax.barh(g.feature, g.xgb_perm_drop, xerr=g.xgb_perm_sd, color=g.tier.map({"A": "#2980b9", "B": "#e67e22"}))
    ax.set_title(f"{tname}\nPermutation importance on 2024-25 test (drop in {'R2' if 'reg' in tname else 'AUC'})", fontsize=9); ax.tick_params(labelsize=8)
plt.tight_layout(); plt.savefig(FIG / "12_permutation_importance_setB.png", dpi=115); plt.close()
# performance A vs B vs LEAK
px = perf[perf.model == "XGBoost"].copy(); px["metric"] = np.where(px.target.str.contains("reg"), px.R2, px.AUC)
fig, ax = plt.subplots(figsize=(10, 4.5)); sns.barplot(data=px, x="target", y="metric", hue="feature_set", ax=ax)
for c in ax.containers: ax.bar_label(c, fmt="%.3f", fontsize=7)
ax.set_ylabel("R2 (T1) / ROC-AUC (T2,T3) on 2024-25"); ax.set_title("Leakage check: performance by feature set (XGBoost, time split)"); ax.set_xticklabels([short[t.get_text()] if t.get_text() in short else t.get_text() for t in ax.get_xticklabels()])
plt.tight_layout(); plt.savefig(FIG / "13_performance_by_feature_set.png", dpi=120); plt.close()
print(json.dumps(summary, indent=1))
print(px[["target", "feature_set", "metric"]])
