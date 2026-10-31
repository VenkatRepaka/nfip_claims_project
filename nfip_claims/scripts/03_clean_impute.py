"""Step 3+4 - cleaning, feature construction and imputation.
Imputers are FIT on the training period (yearOfLoss <= 2023) and APPLIED to all rows so that the 2024-25 test period
never influences any imputation parameter. Outputs: cleaned_claims.parquet/.csv, outputs/cleaning_log.csv,
outputs/imputation_*.csv, figures 06-09."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer, KNNImputer
from sklearn.linear_model import BayesianRidge
from sklearn.preprocessing import StandardScaler
from common import *
sns.set_theme(style="whitegrid")
rng = np.random.default_rng(SEED)
TRAIN_MAX_YEAR = 2023

raw = pd.read_parquet(FILTERED)
log = []   # (step, description, rows_affected)
def L(step, desc, n): log.append(dict(step=step, description=desc, rows_affected=int(n))); print(f"[{step}] {desc}: {int(n):,}")

# ---------------------------------------------------------------- 1. dedupe
dup_cols = [c for c in raw.columns if c != "id"]
dups = raw.duplicated(subset=dup_cols, keep="first")
L("dedupe", "exact duplicate rows (all columns except surrogate id) dropped, first copy kept", dups.sum())
df = raw.loc[~dups].copy().reset_index(drop=True)
L("dedupe", "rows remaining", len(df))

# ---------------------------------------------------------------- 2. types
for c in df.columns:
    if df[c].dtype == object and df[c].notna().any() and df[c].dropna().map(lambda x: type(x).__name__).eq("Decimal").all():
        df[c] = dec_to_float(df[c])
for c in ["dateOfLoss", "originalConstructionDate", "originalNBDate", "openDate"]:
    df[c] = pd.to_datetime(df[c], errors="coerce")
bool_cols = [c for c in df.columns if df[c].dtype == bool]
for c in bool_cols: df[c] = df[c].astype("int8")
L("types", "Decimal columns -> float64, date objects -> datetime64, bool -> int8", 0)
df["state"] = df["state"].str.strip().replace({"": np.nan})
L("types", "blank state set to missing", df.state.isna().sum())
df["reportedZipCode"] = df["reportedZipCode"].astype("string").str.strip()
L("types", "ZIP codes whose length != 5 (kept as-is, not used as a feature)", (df.reportedZipCode.str.len() != 5).sum())

out = pd.DataFrame({"id": df["id"], "dateOfLoss": df["dateOfLoss"], "yearOfLoss": df["yearOfLoss"].astype(int)})

# ---------------------------------------------------------------- 3. payments / targets
pb, pc, pi = df.amountPaidOnBuildingClaim, df.amountPaidOnContentsClaim, df.amountPaidOnIncreasedCostOfComplianceClaim
L("payments", "amountPaid* NaN (claim closed w/o payment; net payment==0 in all of them) -> 0", pb.isna().sum())
L("payments", "negative building payments (re-issued checks) clipped to 0 in target", (pb < 0).sum())
L("payments", "negative contents payments clipped to 0 in target", (pc < 0).sum())
out["paid_building"] = pb.fillna(0).clip(lower=0)
out["paid_contents"] = pc.fillna(0).clip(lower=0)
out["paid_icc"] = pi.fillna(0).clip(lower=0)
out["total_paid"] = out.paid_building + out.paid_contents
out["log_total_paid"] = np.log1p(out.total_paid)
out["is_paid"] = (out.total_paid > 0).astype("int8")
paid_idx = out.is_paid == 1
train_paid = paid_idx & (out.yearOfLoss <= TRAIN_MAX_YEAR)
thr = out.loc[train_paid, "total_paid"].quantile(0.90)
out["high_severity"] = ((out.total_paid >= thr) & paid_idx).astype("int8")   # top-decile of PAID claims (threshold from training years)
L("payments", f"high_severity threshold = 90th pct of paid claims in training years = ${thr:,.0f}; positives", out.high_severity.sum())
out["net_building_payment"] = df.netBuildingPaymentAmount
out["net_contents_payment"] = df.netContentsPaymentAmount
nb = df.nonPaymentReasonBuilding.astype("string"); nc = df.nonPaymentReasonContents.astype("string")
out["nonpay_reason_building"] = nb; out["nonpay_reason_contents"] = nc
L("payments", "building payment exceeds building coverage by more than $1 (flag only: kept, may include ICC/limit adjustments)", (out.paid_building > df.totalBuildingInsuranceCoverage + 1).sum())

# ---------------------------------------------------------------- 4. dates
dl = df.dateOfLoss
oc = df.originalConstructionDate.copy(); bad_oc = (oc.dt.year < 1850) | (oc > dl)
L("dates", "originalConstructionDate < 1850 or after loss -> NaT", bad_oc.sum()); oc[bad_oc] = pd.NaT
out["construction_date_jan1"] = ((oc.dt.month == 1) & (oc.dt.day == 1)).astype("int8")   # placeholder-day indicator
nbd = df.originalNBDate.copy(); bad_nb = (nbd > dl) | (nbd.dt.year < 1968)
L("dates", "originalNBDate after loss or before 1968 (NFIP start) -> NaT", bad_nb.sum()); nbd[bad_nb] = pd.NaT
out["building_age"] = (dl - oc).dt.days / 365.25
out["policy_tenure_years"] = (dl - nbd).dt.days / 365.25
lag = (df.openDate - dl).dt.days
L("dates", "report lag (openDate - dateOfLoss) negative -> NaN", (lag < 0).sum()); lag = lag.where(lag >= 0)
L("dates", "report lag > 365 days winsorised to 365", (lag > 365).sum()); out["report_lag_days"] = lag.clip(upper=365)
out["loss_year"] = df.yearOfLoss.astype(int)
out["loss_month"] = dl.dt.month; out["loss_month_sin"] = np.sin(2 * np.pi * dl.dt.month / 12); out["loss_month_cos"] = np.cos(2 * np.pi * dl.dt.month / 12)
out["loss_dow"] = dl.dt.dayofweek

# ---------------------------------------------------------------- 5. categoricals
occ_map = {1: 11, 2: 12, 3: 13, 4: 18, 6: 18}   # legacy codes -> Risk Rating 2.0 codes
occ = df.occupancyType.replace(occ_map)
L("categorical", "legacy 1-digit occupancyType codes harmonised to 2-digit (1->11,2->12,3->13,4/6->18)", df.occupancyType.isin(occ_map).sum())
occ_lbl = {11: "single_family", 12: "res_2_4_units", 13: "res_5plus_units", 14: "mobile_home", 15: "condo_assoc_bldg",
           16: "unit_in_multiunit", 17: "nonres_mobile", 18: "nonres_building", 19: "other_19"}
out["occupancy_std"] = occ.map(occ_lbl).fillna("unknown")

def zone_family(z):
    if z is None or z != z: return "unknown"
    z = str(z).strip().upper()
    if z in ("X", "B", "C"): return "X"
    if z.startswith("V"): return "V"
    if z in ("AH", "AHB"): return "AH"
    if z in ("AO", "AOB"): return "AO"
    if z == "A99": return "A99"
    if z == "A": return "A"
    if z == "AE" or (z.startswith("A") and z[1:].isdigit()): return "AE"   # A1-A30 == AE
    if z == "D": return "D"
    return "other"
out["flood_zone_family"] = df.ratedFloodZone.map(zone_family)
L("categorical", "ratedFloodZone 52 labels -> zone families", df.ratedFloodZone.notna().sum())
rg = {"RE": "RR2.0", "1": "manual", "7": "PRP", "P": "PRP", "Q": "PRP", "G": "group_GFIP", "B": "prefirm_elev", "W": "prefirm_elev", "2": "specific"}
out["rating_group"] = df.rateMethod.map(rg).fillna("other")
out["is_rr2"] = (df.rateMethod == "RE").astype("int8")
ded = {"0": 500, "1": 1000, "2": 2000, "3": 3000, "4": 4000, "5": 5000, "9": 750, "A": 10000, "B": 15000, "C": 20000, "D": 25000, "E": 50000, "F": 1250, "G": 1500, "H": 200}
cause = {"4": "rainfall_accum", "1": "tidal", "2": "stream_river", "0": "other", "3": "other", "7": "other", "8": "other", "9": "other", "A": "other", "B": "expedited", "C": "expedited", "D": "expedited"}
out["cause_group"] = df.causeOfDamage.map(cause).fillna("unknown")
fnd = {"1": "slab", "2": "basement", "3": "crawlspace", "4": "elevated_open", "5": "elevated_enclosed_pile", "6": "elevated_enclosed_other", "0": "unknown"}
out["foundation_type"] = df.foundationType.map(fnd).fillna("unknown")
out["basement_type"] = df.basementEnclosureCrawlspaceType.map({0: "none", 1: "finished", 2: "unfinished", 3: "crawlspace", 4: "subgrade_crawl"}).fillna("unknown")
out["contents_location"] = df.locationOfContents.map(lambda x: f"loc{int(x)}" if x == x else "unknown")
bd = df.buildingDescriptionCode.map({1: "main_house", 8: "other", 10: "apartment_bldg"})
out["building_use"] = bd.where(df.buildingDescriptionCode.notna(), "unknown").fillna("other_use")
out["crs_class"] = df.crsClassCode      # numeric, imputed below
out["crs_unknown"] = df.crsClassCode.isna().astype("int8")
for src, dst in [("elevatedBuildingIndicator", "elevated_building"), ("postFIRMConstructionIndicator", "post_firm"), ("primaryResidenceIndicator", "primary_residence"),
                 ("rentalPropertyIndicator", "rental_property"), ("smallBusinessIndicatorBuilding", "small_business")]:
    out[dst] = df[src]
out["condo_type"] = df.condominiumCoverageTypeCode.fillna(df.condominiumCoverageTypeCode.mode()[0])
L("impute-mode", "condominiumCoverageTypeCode missing -> mode", df.condominiumCoverageTypeCode.isna().sum())
out["repl_cost_basis"] = df.replacementCostBasis.fillna(df.replacementCostBasis.mode()[0])
L("impute-mode", "replacementCostBasis missing -> mode", df.replacementCostBasis.isna().sum())
mode_fl = df.numberOfFloorsInTheInsuredBuilding.mode()[0]
out["num_floors_code"] = df.numberOfFloorsInTheInsuredBuilding.fillna(mode_fl).astype(int).astype(str)
L("impute-mode", f"numberOfFloorsInTheInsuredBuilding missing -> mode ({int(mode_fl)})", df.numberOfFloorsInTheInsuredBuilding.isna().sum())
top_states = df.state.value_counts().head(15).index
out["state_grp"] = df.state.where(df.state.isin(top_states), "OTHER").fillna("OTHER")
out["state"] = df.state.fillna("UNK")
out["named_event"] = df.floodEvent.notna().astype("int8")
top_ev = df.floodEvent.value_counts().head(12).index
out["flood_event_grp"] = df.floodEvent.where(df.floodEvent.isin(top_ev), "OTHER_EVENT").fillna("NO_NAMED_EVENT")
out["flood_event"] = df.floodEvent
for c in ["reportedZipCode", "countyCode", "censusGeoid"]: out[c] = df[c]

# deductibles
bded = df.buildingDeductibleCode.map(ded)
out["bldg_deductible_missing"] = bded.isna().astype("int8")
# contents coverage structure
cc0 = df.totalContentsInsuranceCoverage.fillna(0) == 0
bc0 = df.totalBuildingInsuranceCoverage == 0
out["no_contents_cov"] = cc0.astype("int8"); out["no_building_cov"] = bc0.astype("int8")
cded = df.contentsDeductibleCode.map(ded)
L("zero-vs-missing", "contentsDeductibleCode NaN where contents coverage==0 -> 0 (no contents deductible applies)", (cded.isna() & cc0).sum())
cded = cded.where(~(cded.isna() & cc0), 0)
out["cont_deductible_usd"] = cded; out["cont_deductible_missing"] = cded.isna().astype("int8")
out["bldg_deductible_usd"] = bded

# ---------------------------------------------------------------- 6. impossible values / sentinels / zero-vs-missing
def money_log(s, name, hi_q=0.999):
    cap = s.dropna().quantile(hi_q); n_clip = int((s > cap).sum())
    L("winsorise", f"{name} clipped at p99.9 (${cap:,.0f}) before log1p", n_clip)
    return np.log1p(s.clip(lower=0, upper=cap))
out["log_bldg_coverage"] = money_log(df.totalBuildingInsuranceCoverage, "totalBuildingInsuranceCoverage")
out["log_cont_coverage"] = money_log(df.totalContentsInsuranceCoverage.fillna(0), "totalContentsInsuranceCoverage")
pc_clip = df.policyCount.clip(upper=df.policyCount.quantile(0.999)); L("winsorise", "policyCount clipped at p99.9", (df.policyCount > df.policyCount.quantile(0.999)).sum())
out["log_policy_count"] = np.log1p(pc_clip)
nu = df.numberOfUnits.clip(upper=df.numberOfUnits.quantile(0.999)); out["log_num_units"] = np.log1p(nu.fillna(nu.median()))
L("impute-median", "numberOfUnits missing -> median", df.numberOfUnits.isna().sum())

bdam, bval, brep = df.buildingDamageAmount.copy(), df.buildingPropertyValue.copy(), df.buildingReplacementCost.copy()
z_val = (bval == 0) & ~bc0; z_rep = (brep == 0) & ~bc0
L("zero-vs-missing", "buildingPropertyValue == 0 with building coverage > 0 (impossible) -> NaN", z_val.sum()); bval[z_val] = np.nan
L("zero-vs-missing", "buildingReplacementCost == 0 with building coverage > 0 (impossible) -> NaN", z_rep.sum()); brep[z_rep] = np.nan
L("zero-vs-missing", "buildingPropertyValue/ReplacementCost == 0 with NO building coverage kept as 0 (contents-only policies)", ((df.buildingPropertyValue == 0) & bc0).sum())
cdam, cval = df.contentsDamageAmount.copy(), df.contentsPropertyValue.copy()
L("zero-vs-missing", "contentsPropertyValue == -1 (invalid) -> NaN", (cval < 0).sum()); cval[cval < 0] = np.nan
s0d = cdam.isna() & cc0; s0v = cval.isna() & cc0
L("zero-vs-missing", "contentsDamageAmount NaN where contents coverage==0 -> 0 (structural)", s0d.sum()); cdam[s0d] = 0
L("zero-vs-missing", "contentsPropertyValue NaN where contents coverage==0 -> 0 (structural)", s0v.sum()); cval[s0v] = 0

wd = df.waterDepth.copy()
L("sentinel", "waterDepth == 99 (sentinel spike; claims at 99 behave like 'unknown', not like 98/100) -> NaN", (wd == 99).sum())
_paid = ((df.amountPaidOnBuildingClaim.fillna(0).clip(lower=0) + df.amountPaidOnContentsClaim.fillna(0).clip(lower=0)) > 0)
L("info", f"waterDepth < 0 RETAINED ({100 * _paid[wd < 0].mean():.0f}% of those claims are paid vs {100 * _paid[wd == 0].mean():.0f}% at depth 0 and {100 * _paid[wd.isna()].mean():.0f}% when depth is missing: carries signal)", (wd < 0).sum())
wd[wd == 99] = np.nan
idp, edp = df.interiorWaterDepth.copy(), df.exteriorWaterDepth.copy()
L("sentinel", "interiorWaterDepth >= 999 (sentinel) -> NaN", (idp >= 999).sum()); idp[idp >= 999] = np.nan
L("sentinel", "exteriorWaterDepth >= 999 (sentinel/implausible) -> NaN", (edp >= 999).sum()); edp[edp >= 999] = np.nan
L("info", "interior/exterior depth negatives retained (relative to lowest adjacent grade); all three depths winsorised at p1/p99.5", ((idp < 0).sum() + (edp < 0).sum()))
for nm, s in (("waterDepth", wd), ("interiorWaterDepth", idp), ("exteriorWaterDepth", edp)):
    pass
def wins(s, lo=0.01, hi=0.995):
    a, b = s.quantile(lo), s.quantile(hi); return s.clip(a, b)
wd, idp, edp = wins(wd), wins(idp), wins(edp)

# elevation columns are dropped (>60% missing) but clean the sentinels only to record availability
elev = df[["elevationDifference", "baseFloodElevation", "lowestFloorElevation", "lowestAdjacentGrade"]].copy()
elev_sent = ((elev.baseFloodElevation >= 9990) | (elev.lowestFloorElevation.abs() >= 9990) | (elev.lowestAdjacentGrade.abs() >= 9990) | (elev.elevationDifference.abs() >= 500))
L("sentinel", "elevation columns with sentinel/implausible values (dropped columns; flagged only)", elev_sent.sum())
out["has_elevation_data"] = (elev.notna().any(axis=1) & ~elev_sent).astype("int8")
out["has_elevation_cert"] = df.elevationCertificateIndicator.notna().astype("int8")

cap = lambda s: s.dropna().quantile(0.999)
# missing indicators (BEFORE imputation)
out["adjuster_info_missing"] = bdam.isna().astype("int8")         # damage/value/replacement are all missing for exactly the same raw rows (no adjuster record)
out["cont_damage_missing"] = cdam.isna().astype("int8")
out["water_depth_missing"] = wd.isna().astype("int8"); out["interior_depth_missing"] = idp.isna().astype("int8"); out["exterior_depth_missing"] = edp.isna().astype("int8")
L("flag", "adjuster_info_missing (building damage/value/replacement cost missing)", out.adjuster_info_missing.sum())
L("flag", "building value or replacement cost missing although damage is present (zero->NaN recode; value-valuation not recorded)", ((bval.isna() | brep.isna()) & bdam.notna()).sum())
out["bldg_value_recoded_missing"] = (z_val | z_rep).astype("int8")

# log transforms (money)
out["log_bldg_damage"] = np.log1p(bdam.clip(0, cap(bdam)))
out["log_bldg_value"] = np.log1p(bval.clip(0, cap(bval)))
out["log_repl_cost"] = np.log1p(brep.clip(0, cap(brep)))
out["log_cont_damage"] = np.log1p(cdam.clip(0, cap(cdam)))
out["log_cont_value"] = np.log1p(cval.clip(0, cap(cval)))
for _n, _s in (("buildingDamageAmount", bdam), ("buildingPropertyValue", bval), ("buildingReplacementCost", brep), ("contentsDamageAmount", cdam), ("contentsPropertyValue", cval)):
    L("winsorise", f"{_n} clipped at p99.9 (${cap(_s):,.0f}) then log1p", (_s > cap(_s)).sum())
out["water_depth"] = wd; out["interior_depth"] = idp; out["exterior_depth"] = edp
out["latitude"] = df.latitude; out["longitude"] = df.longitude

# skew before/after (observed values)
skew_rows = []
for raw_c, new_c in [("totalBuildingInsuranceCoverage", "log_bldg_coverage"), ("totalContentsInsuranceCoverage", "log_cont_coverage"),
                     ("buildingDamageAmount", "log_bldg_damage"), ("buildingPropertyValue", "log_bldg_value"), ("buildingReplacementCost", "log_repl_cost"),
                     ("contentsDamageAmount", "log_cont_damage"), ("contentsPropertyValue", "log_cont_value"), ("policyCount", "log_policy_count")]:
    a = df[raw_c].dropna(); b = out[new_c].dropna()
    skew_rows.append(dict(column=raw_c, skew_raw=stats.skew(a), transformed=new_c, skew_after=stats.skew(b), max_raw=a.max(), max_after_expm1=np.expm1(b.max())))
a = out.total_paid; skew_rows.append(dict(column="total_paid (target)", skew_raw=stats.skew(a), transformed="log_total_paid", skew_after=stats.skew(out.log_total_paid), max_raw=a.max(), max_after_expm1=a.max()))
ap = out.loc[paid_idx, "total_paid"]; skew_rows.append(dict(column="total_paid | paid>0", skew_raw=stats.skew(ap), transformed="log1p", skew_after=stats.skew(np.log1p(ap)), max_raw=ap.max(), max_after_expm1=ap.max()))
pd.DataFrame(skew_rows).to_csv(OUT / "cleaning_skew_before_after.csv", index=False)

# ---------------------------------------------------------------- 7. IMPUTATION
train = out.yearOfLoss <= TRAIN_MAX_YEAR
imp_log = []
def record(col, strat, why, n): imp_log.append(dict(column=col, n_imputed=int(n), pct_imputed=round(100 * n / len(out), 2), strategy=strat, justification=why))

# 7a. location: county median, then state median, then global (fit on train)
for c in ("latitude", "longitude"):
    n = out[c].isna().sum()
    cm = out[train].groupby("countyCode")[c].median(); sm = out[train].groupby("state")[c].median()
    out[c] = out[c].fillna(out.countyCode.map(cm)).fillna(out.state.map(sm)).fillna(out.loc[train, c].median())
    record(c, "group-wise median (county -> state -> global)", "coordinates are rounded to 0.1deg; location missingness is tiny and geography is the dominant structure", n)

# 7b. building age / tenure / crs: group median by occupancy x state-group (fit on train)
for c in ("building_age", "policy_tenure_years"):
    n = out[c].isna().sum(); g = ["occupancy_std", "state_grp"]
    med = out[train].groupby(g)[c].median().rename("m").reset_index()
    fill = out[g].merge(med, on=g, how="left")["m"].values
    out[c] = out[c].fillna(pd.Series(fill, index=out.index)).fillna(out.loc[train, c].median())
    record(c, "group-wise median (occupancy x state group), fallback global median", "dates invalid/missing in <3% of rows; age/tenure differ strongly by occupancy and region", n)
out["bldg_age_invalid_or_missing"] = log_flag = pd.Series(oc.isna().astype("int8").values, index=out.index)
out["tenure_invalid_or_missing"] = pd.Series(nbd.isna().astype("int8").values, index=out.index)
n = out.crs_class.isna().sum()
med = out[train].groupby("state_grp").crs_class.median(); out["crs_class"] = out.crs_class.fillna(out.state_grp.map(med)).fillna(out.loc[train, "crs_class"].median())
record("crs_class", "group-wise median by state + crs_unknown flag", "29.8% missing, but missingness itself is kept as a flag (crs_unknown); numeric value only for tree/linear convenience", n)
n = out.report_lag_days.isna().sum(); out["report_lag_days"] = out.report_lag_days.fillna(out.loc[train, "report_lag_days"].median())
record("report_lag_days", "global median", f"only {n} invalid rows", n)
for c in ("bldg_deductible_usd",):
    n = out[c].isna().sum(); m = out.loc[train, c].mode()[0]; out[c] = out[c].fillna(m)
    record(c, "mode + bldg_deductible_missing flag", "2.4% missing, categorical-like with a dominant level; flag preserves missingness", n)
record("cont_deductible_usd", "structural 0 where no contents coverage (done in cleaning); remaining NaN -> mode", "97% of NaNs are 'no contents coverage'", cded.isna().sum())
out["cont_deductible_usd"] = out.cont_deductible_usd.fillna(out.loc[train, "cont_deductible_usd"].mode()[0])
out["log_cont_coverage"] = out.log_cont_coverage.fillna(0)
record("contents_location / basement_type / building_use", "explicit 'unknown' category", "45% / 49% / 26% missing and not random (relates to rating system & occupancy); imputing a category would invent information", 0)

# 7c. numeric adjuster block: validation of imputer choice, then IterativeImputer
block = ["log_bldg_damage", "log_bldg_value", "log_repl_cost", "log_cont_damage", "log_cont_value",
         "water_depth", "interior_depth", "exterior_depth",
         "log_bldg_coverage", "log_cont_coverage", "building_age", "policy_tenure_years", "latitude", "longitude", "is_rr2", "elevated_building", "post_firm"]
tgt_cols = ["log_bldg_damage", "log_bldg_value", "log_repl_cost", "log_cont_damage", "log_cont_value", "water_depth", "interior_depth", "exterior_depth"]
Xb = out[block].copy()
lo = Xb.loc[train].min().values; hi = Xb.loc[train].max().values
def make_iter(): return IterativeImputer(estimator=BayesianRidge(), max_iter=8, random_state=SEED, min_value=lo, max_value=hi, initial_strategy="median", skip_complete=True)

# validation: two experiments on fully-observed training rows
#   (i) "single": mask 10% of ONE column at a time (easy: all other predictors observed)
#   (ii) "joint": mask the WHOLE adjuster block (all 8 columns) for 10% of rows - mimics reality, where damage/value/depth are
#        missing together because no adjuster record exists. Method per column is chosen from the JOINT experiment.
full = Xb.loc[train].dropna(); full = full.sample(min(20000, len(full)), random_state=SEED)
grp_cols = out.loc[full.index, ["occupancy_std", "flood_zone_family"]]
def run_methods(Xm, m, cols_eval):
    res = {}
    sc = StandardScaler().fit(Xm); Z = pd.DataFrame(sc.transform(Xm), columns=block)
    knn = KNNImputer(n_neighbors=10).fit_transform(Z)
    it = IterativeImputer(estimator=BayesianRidge(), max_iter=8, random_state=SEED, initial_strategy="median").fit_transform(Xm)
    for tc in cols_eval:
        j = block.index(tc); r = {}
        r["global_median"] = np.full(m.sum(), full[tc][~m].median())
        gm = pd.concat([grp_cols, full[tc]], axis=1)[~m].groupby(["occupancy_std", "flood_zone_family"])[tc].median()
        key = pd.MultiIndex.from_frame(grp_cols[m]); r["group_median(occupancy x zone)"] = pd.Series(gm.reindex(key).values).fillna(full[tc][~m].median()).values
        r["KNN(k=10)"] = knn[m, j] * sc.scale_[j] + sc.mean_[j]
        r["Iterative(BayesianRidge)"] = it[m, j]
        res[tc] = r
    return res
val_rows = []
for tc in tgt_cols:                                   # (i) single-column masking
    m = rng.random(len(full)) < 0.10; Xm = full.copy(); Xm.loc[m, tc] = np.nan
    for k, p in run_methods(Xm, m, [tc])[tc].items():
        tr = full[tc].values[m]; val_rows.append(dict(scenario="single", column=tc, method=k, RMSE=float(np.sqrt(np.mean((p - tr) ** 2))), MAE=float(np.mean(np.abs(p - tr))), sd_truth=float(tr.std())))
m = rng.random(len(full)) < 0.10; Xm = full.copy(); Xm.loc[m, tgt_cols] = np.nan   # (ii) joint block masking
for tc, r in run_methods(Xm, m, tgt_cols).items():
    tr = full[tc].values[m]
    for k, p in r.items():
        val_rows.append(dict(scenario="joint", column=tc, method=k, RMSE=float(np.sqrt(np.mean((p - tr) ** 2))), MAE=float(np.mean(np.abs(p - tr))), sd_truth=float(tr.std())))
val = pd.DataFrame(val_rows); val.to_csv(OUT / "imputation_validation.csv", index=False)
for sc_name in ("single", "joint"):
    print(sc_name); print(val[val.scenario == sc_name].pivot(index="column", columns="method", values="RMSE").round(3))
vj = val[val.scenario == "joint"]
best = vj[vj.method.isin(["KNN(k=10)", "Iterative(BayesianRidge)"])].sort_values("RMSE").groupby("column").first().method.to_dict()
print("chosen:", best)

# fit on a 40k training sample, apply to every row
fit_sample = Xb.loc[train].sample(40000, random_state=SEED)
imp = make_iter().fit(fit_sample)
before = Xb.copy()
filled = pd.DataFrame(imp.transform(Xb), columns=block, index=Xb.index)
knn_cols = [c for c, mth in best.items() if mth.startswith("KNN")]
if knn_cols:
    sc = StandardScaler().fit(fit_sample); knn = KNNImputer(n_neighbors=10).fit(pd.DataFrame(sc.transform(fit_sample), columns=block))
    need = before[knn_cols].isna().any(axis=1)
    Zn = pd.DataFrame(sc.transform(before[need]), columns=block, index=before.index[need])
    parts = [pd.DataFrame(knn.transform(Zn.iloc[i:i + 20000]), columns=block, index=Zn.index[i:i + 20000]) for i in range(0, len(Zn), 20000)]
    K = pd.concat(parts) * sc.scale_ + sc.mean_
    for c in knn_cols:
        miss = before[c].isna(); filled.loc[miss, c] = K.loc[miss[miss].index, c]
for c in tgt_cols:
    n = before[c].isna().sum(); out[c] = filled[c]
    mth = best[c]
    record(c, f"{mth} (chosen by joint-masking validation) + missing flag; fit on 2020-23 sample of 40k; values clipped to observed range",
           "adjuster-block values are missing together, so the method was chosen on the JOINT-masking experiment; flag retained because missingness is informative", n)
imp_df = pd.DataFrame(imp_log); imp_df.to_csv(OUT / "imputation_log.csv", index=False)

# before/after distribution comparison (observed vs imputed values + KS)
ks_rows = []
fig, axes = plt.subplots(2, 4, figsize=(18, 8))
for ax, c in zip(axes.ravel(), tgt_cols):
    miss = before[c].isna(); obs_mask = ~miss
    if c in ("log_cont_damage", "log_cont_value"):       # compare like with like: structural zeros (no contents cover) are not part of the imputed population
        obs_mask = obs_mask & (out["no_contents_cov"] == 0)
    obs = before.loc[obs_mask, c]; imp_v = filled.loc[miss, c]
    if miss.sum() == 0: continue
    ks = stats.ks_2samp(obs, imp_v)
    ks_rows.append(dict(column=c, n_observed=len(obs), n_imputed=int(miss.sum()), mean_observed=obs.mean(), mean_imputed=imp_v.mean(),
                        median_observed=obs.median(), median_imputed=imp_v.median(), sd_observed=obs.std(), sd_imputed=imp_v.std(),
                        mean_all_after=filled[c].mean(), ks_stat=ks.statistic, ks_p=ks.pvalue))
    sns.kdeplot(obs, ax=ax, label="observed", color="#2980b9", fill=True, alpha=.3, warn_singular=False)
    sns.kdeplot(imp_v, ax=ax, label="imputed", color="#c0392b", fill=True, alpha=.3, warn_singular=False)
    ax.set_title(f"{c}  (n_imputed={int(miss.sum()):,}, KS={ks.statistic:.2f}" + (", observed excl. no-contents-cover rows)" if c in ("log_cont_damage", "log_cont_value") else ")"), fontsize=9); ax.legend(fontsize=7)
plt.suptitle("Observed vs imputed value distributions (method per column chosen by validation)"); plt.tight_layout(); plt.savefig(FIG / "07_imputation_distributions.png", dpi=110); plt.close()
pd.DataFrame(ks_rows).to_csv(OUT / "imputation_before_after.csv", index=False)

# imputation RMSE figure
fig, axes = plt.subplots(1, 2, figsize=(18, 5), sharey=True)
for ax, scn in zip(axes, ("single", "joint")):
    v = val[val.scenario == scn]; piv = v.pivot(index="column", columns="method", values="RMSE").div(v.groupby("column").sd_truth.first(), axis=0)
    piv.plot.bar(ax=ax, rot=30, legend=(scn == "single")); ax.set_title(f"masking: {scn}"); ax.set_ylabel("RMSE / SD(truth) (1 = no skill)")
plt.suptitle("Imputer comparison on masked observed values (10% masked, 20k fully-observed 2020-23 rows)"); plt.tight_layout(); plt.savefig(FIG / "08_imputation_validation.png", dpi=120); plt.close()

# derived ratios (after imputation, B tier)
out["damage_to_value_ratio"] = (np.expm1(out.log_bldg_damage) / np.expm1(out.log_bldg_value).clip(lower=1)).clip(upper=2)
out["coverage_to_repl_ratio"] = (np.expm1(out.log_bldg_coverage) / np.expm1(out.log_repl_cost).clip(lower=1)).clip(upper=5)

# ---------------------------------------------------------------- 8. final checks & save
feat_num = [c for c in out.columns if pd.api.types.is_numeric_dtype(out[c]) and c not in ("net_building_payment", "net_contents_payment")]
remaining_na = out.isna().sum(); remaining_na = remaining_na[remaining_na > 0]
print("Remaining NaN columns:\n", remaining_na)
pd.DataFrame({"column": remaining_na.index, "n_missing": remaining_na.values}).to_csv(OUT / "cleaned_remaining_missing.csv", index=False)
for c in out.select_dtypes("object").columns:
    if c not in ("flood_event",): out[c] = out[c].astype("string")
out.to_parquet(ROOT / "cleaned_claims.parquet", index=False)
csv_path = ROOT / "cleaned_claims.csv"; out.to_csv(csv_path, index=False)
sz = csv_path.stat().st_size / 1e6
L("save", f"cleaned_claims.csv size {sz:.0f} MB; rows", len(out))
if sz > 200:
    csv_path.unlink(); out.sample(50000, random_state=SEED).to_csv(ROOT / "cleaned_claims_sample50k.csv", index=False)
    L("save", "CSV > 200MB: wrote 50k-row sample instead", 50000)
pd.DataFrame(log).to_csv(OUT / "cleaning_log.csv", index=False)
save_json({"train_max_year": TRAIN_MAX_YEAR, "high_severity_threshold_usd": float(thr), "n_rows_clean": len(out), "n_cols_clean": out.shape[1], "csv_mb": sz}, "clean_meta.json")

# figure: before/after skew
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
for ax, (rc, nc_) in zip(axes.ravel(), [("totalBuildingInsuranceCoverage", "log_bldg_coverage"), ("buildingDamageAmount", "log_bldg_damage"), ("buildingPropertyValue", "log_bldg_value"), ("buildingReplacementCost", "log_repl_cost")] ):
    pass
fig, axes = plt.subplots(2, 4, figsize=(18, 7))
pairs = [("totalBuildingInsuranceCoverage", "log_bldg_coverage"), ("buildingDamageAmount", "log_bldg_damage"), ("buildingPropertyValue", "log_bldg_value"), ("total_paid", "log_total_paid")]
for i, (rc, nc_) in enumerate(pairs):
    a = (df[rc] if rc in df else out[rc]).dropna(); b = out[nc_] if nc_ == "log_total_paid" else out[nc_][~pd.isna(df[rc])]
    axes[0, i].hist(a.clip(upper=a.quantile(.99)), bins=60, color="#7f8c8d"); axes[0, i].set_title(f"{rc}\nraw (clip p99) skew={stats.skew(a):.1f}", fontsize=9)
    axes[1, i].hist(b, bins=60, color="#16a085"); axes[1, i].set_title(f"{nc_}  skew={stats.skew(b):.2f}", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "06_log_transform_before_after.png", dpi=110); plt.close()
print("done", out.shape)
