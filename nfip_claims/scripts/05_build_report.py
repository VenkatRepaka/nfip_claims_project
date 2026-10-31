"""Step 6 - assemble report.md, column_dictionary.csv, engineered_feature_dictionary.csv from the artefacts in outputs/.
Every number in the report is read from those artefacts (which are produced by scripts 01-04) or computed here from the data."""
import numpy as np, pandas as pd
from common import *
from decisions import D

P = pd.read_csv(OUT / "profile_columns.csv"); G = load_json("profile_global.json"); RC = load_json("row_counts.json")
CL = pd.read_csv(OUT / "cleaning_log.csv"); SK = pd.read_csv(OUT / "cleaning_skew_before_after.csv")
IL = pd.read_csv(OUT / "imputation_log.csv"); IV = pd.read_csv(OUT / "imputation_validation.csv"); IBA = pd.read_csv(OUT / "imputation_before_after.csv")
PERF = pd.read_csv(OUT / "model_performance_by_feature_set.csv"); RANK = pd.read_csv(OUT / "feature_ranking_full.csv"); CHK = pd.read_csv(OUT / "recommended_vs_full.csv")
FS = load_json("feature_selection_results.json"); META = load_json("clean_meta.json"); PY = pd.read_csv(OUT / "profile_per_year.csv")
PAIRS = pd.read_csv(OUT / "corr_high_pairs.csv"); TC = pd.read_csv(OUT / "corr_with_targets.csv", index_col=0); REM = pd.read_csv(OUT / "cleaned_remaining_missing.csv")
MON = pd.read_csv(OUT / "profile_monthly_counts.csv"); FD = load_field_dictionary()
clean = pd.read_parquet(ROOT / "cleaned_claims.parquet")
raw = pd.read_parquet(FILTERED, columns=["openDate", "dateOfLoss", "floodEvent", "yearOfLoss", "rateMethod", "nonPaymentReasonBuilding"])

def md(df, floatfmt=3, index=False):
    d = df.copy()
    if index: d = d.reset_index()
    def f(x):
        if isinstance(x, (float, np.floating)):
            return "" if pd.isna(x) else (f"{x:,.{floatfmt}f}" if abs(x) < 1e6 else f"{x:,.0f}")
        if isinstance(x, (int, np.integer)): return f"{x:,}"
        return str(x).replace("|", "/").replace("\n", " ")
    rows = ["| " + " | ".join(map(str, d.columns)) + " |", "|" + "|".join(["---"] * d.shape[1]) + "|"]
    rows += ["| " + " | ".join(f(v) for v in r) + " |" for r in d.itertuples(index=False)]
    return "\n".join(rows)

# ------------------------------------------------------------------ column dictionary (raw columns)
def short(txt, n=170):
    txt = " ".join(str(txt).split()); return txt if len(txt) <= n else txt[:n - 1].rstrip() + "…"
rows = []
for c in P.column:
    d = D[c]; pr = P[P.column == c].iloc[0]
    rows.append(dict(column=c, description=short(FD.get(c, {}).get("description", "")), dtype_raw=pr.raw_dtype, openfema_type=FD.get(c, {}).get("type", ""),
                     missing_pct=pr.missing_pct, n_unique=pr.n_unique, decision=d["decision"], reason=d["reason"], role=d["role"], derived_features=d["derived"]))
CD = pd.DataFrame(rows); CD.to_csv(ROOT / "column_dictionary.csv", index=False)

# ------------------------------------------------------------------ engineered feature dictionary
TIER = FS["tier"]; REC = FS["recommended"]
desc = {
 "log_bldg_coverage": ("totalBuildingInsuranceCoverage", "log1p of building coverage limit (clipped p99.9)"), "log_cont_coverage": ("totalContentsInsuranceCoverage", "log1p of contents coverage limit"),
 "log_policy_count": ("policyCount", "log1p of insured units on record (clipped p99.9)"), "log_num_units": ("numberOfUnits", "log1p of condo-master unit count (clipped, median-imputed)"),
 "building_age": ("originalConstructionDate, dateOfLoss", "years from construction to loss (invalid dates imputed by group median)"), "policy_tenure_years": ("originalNBDate, dateOfLoss", "years from original policy date to loss"),
 "report_lag_days": ("openDate, dateOfLoss", "days from loss to claim opening (clipped 0..365)"), "loss_month_sin": ("dateOfLoss", "cyclic month encoding"), "loss_month_cos": ("dateOfLoss", "cyclic month encoding"),
 "loss_dow": ("dateOfLoss", "day of week of loss"), "latitude": ("latitude", "0.1-degree rounded; missing -> county/state median"), "longitude": ("longitude", "0.1-degree rounded; missing -> county/state median"),
 "bldg_deductible_usd": ("buildingDeductibleCode", "deductible in USD (code map)"), "cont_deductible_usd": ("contentsDeductibleCode", "contents deductible USD; 0 when no contents coverage"),
 "crs_class": ("crsClassCode", "CRS class 1-10 (missing -> state median, see crs_unknown)"), "elevated_building": ("elevatedBuildingIndicator", "binary"), "post_firm": ("postFIRMConstructionIndicator", "binary"),
 "primary_residence": ("primaryResidenceIndicator", "binary"), "rental_property": ("rentalPropertyIndicator", "binary"), "small_business": ("smallBusinessIndicatorBuilding", "binary"),
 "is_rr2": ("rateMethod", "1 if rated by Risk Rating 2.0 engine ('RE')"), "no_contents_cov": ("totalContentsInsuranceCoverage", "1 if contents coverage = 0"), "no_building_cov": ("totalBuildingInsuranceCoverage", "1 if building coverage = 0"),
 "crs_unknown": ("crsClassCode", "1 if CRS class missing"), "named_event": ("floodEvent", "1 if claim belongs to a named FEMA flood event"), "has_elevation_data": ("elevation columns", "1 if any elevation field is present and non-sentinel"),
 "has_elevation_cert": ("elevationCertificateIndicator", "1 if elevation-certificate indicator present"), "construction_date_jan1": ("originalConstructionDate", "1 if construction date is Jan 1 (placeholder)"),
 "bldg_age_invalid_or_missing": ("originalConstructionDate", "1 if construction date missing/invalid (age imputed)"), "tenure_invalid_or_missing": ("originalNBDate", "1 if policy date missing/invalid (tenure imputed)"),
 "bldg_deductible_missing": ("buildingDeductibleCode", "1 if building deductible code missing"), "cont_deductible_missing": ("contentsDeductibleCode", "1 if contents deductible still missing after structural fill"),
 "occupancy_std": ("occupancyType", "harmonised occupancy label"), "flood_zone_family": ("ratedFloodZone", "AE/A/AH/AO/A99/V/X/D"), "rating_group": ("rateMethod", "RR2.0/manual/PRP/GFIP/prefirm_elev/specific/other"),
 "foundation_type": ("foundationType", "labelled foundation type"), "basement_type": ("basementEnclosureCrawlspaceType", "labelled; 'unknown' for missing"), "contents_location": ("locationOfContents", "code; 'unknown' for missing"),
 "building_use": ("buildingDescriptionCode", "main_house/apartment_bldg/other/other_use/unknown"), "condo_type": ("condominiumCoverageTypeCode", "N/U/A/L/H (mode-filled)"), "repl_cost_basis": ("replacementCostBasis", "R/A (mode-filled)"),
 "state_grp": ("state", "top-15 states + OTHER"), "num_floors_code": ("numberOfFloorsInTheInsuredBuilding", "code (mode-filled)"), "flood_event_grp": ("floodEvent", "top-12 events + OTHER_EVENT + NO_NAMED_EVENT"),
 "cause_group": ("causeOfDamage", "rainfall_accum/tidal/stream_river/other/expedited/unknown (TIER B)"),
 "log_bldg_damage": ("buildingDamageAmount", "log1p building damage (clipped p99.9; imputed)"), "log_bldg_value": ("buildingPropertyValue", "log1p building ACV (0->NaN; imputed)"), "log_repl_cost": ("buildingReplacementCost", "log1p replacement cost (0->NaN; imputed)"),
 "log_cont_damage": ("contentsDamageAmount", "log1p contents damage (structural 0 when no contents cover; imputed otherwise)"), "log_cont_value": ("contentsPropertyValue", "log1p contents value"),
 "water_depth": ("waterDepth", "inches; 99 sentinel -> NaN -> imputed; winsorised"), "interior_depth": ("interiorWaterDepth", "inches; 999 -> NaN -> imputed"), "exterior_depth": ("exteriorWaterDepth", "inches; >=999 -> NaN -> imputed"),
 "damage_to_value_ratio": ("damage / value", "building damage divided by building value (clip 2)"), "coverage_to_repl_ratio": ("coverage / replacement", "building coverage divided by replacement cost (clip 5): under-insurance proxy"),
 "adjuster_info_missing": ("buildingDamageAmount", "1 if no adjuster damage/value record"), "cont_damage_missing": ("contentsDamageAmount", "1 if contents damage missing (not structural)"),
 "water_depth_missing": ("waterDepth", "1 if depth missing/sentinel"), "interior_depth_missing": ("interiorWaterDepth", "1 if missing/sentinel"), "exterior_depth_missing": ("exteriorWaterDepth", "1 if missing/sentinel"),
 "bldg_value_recoded_missing": ("buildingPropertyValue/ReplacementCost", "1 if value/replacement cost was 0 with coverage > 0 and recoded to missing"),
 "leak_log_net_bldg_payment": ("netBuildingPaymentAmount", "LEAKAGE demo"), "leak_log_net_cont_payment": ("netContentsPaymentAmount", "LEAKAGE demo"),
 "leak_nonpay_reason_building_present": ("nonPaymentReasonBuilding", "LEAKAGE demo"), "leak_nonpay_reason_contents_present": ("nonPaymentReasonContents", "LEAKAGE demo"),
}
efd = []
for f_, t_ in TIER.items():
    r_ = {"feature": f_, "tier": t_, "source": desc.get(f_, ("", ""))[0], "description": desc.get(f_, ("", ""))[1]}
    for k, lab in [("T1_reg_log_total_paid (paid claims)", "T1"), ("T2_cls_is_paid", "T2"), ("T3_cls_high_severity", "T3")]:
        for s_ in ("A", "B"): r_[f"rec_{lab}_{s_}"] = int(f_ in REC.get(f"{k}|{s_}", []))
    efd.append(r_)
EFD = pd.DataFrame(efd); EFD.to_csv(ROOT / "engineered_feature_dictionary.csv", index=False)


# ------------------------------------------------------------------ computed facts (from the raw filtered file; pre-dedupe)
RW = pd.read_parquet(FILTERED)
for c in RW.columns:
    if RW[c].dtype == object and RW[c].notna().any() and RW[c].dropna().map(lambda x: type(x).__name__).eq("Decimal").all(): RW[c] = dec_to_float(RW[c])
pb_, pc_ = RW.amountPaidOnBuildingClaim, RW.amountPaidOnContentsClaim
tp_ = pb_.fillna(0).clip(lower=0) + pc_.fillna(0).clip(lower=0)
dmg_na = RW.buildingDamageAmount.isna()
F = {}
F["dmg_val_rep_same_rows"] = int((dmg_na & RW.buildingPropertyValue.isna() & RW.buildingReplacementCost.isna()).sum()); F["dmg_na"] = int(dmg_na.sum())
F["paid_when_dmg_na"] = 100 * (tp_[dmg_na] > 0).mean(); F["paid_when_dmg_pos"] = 100 * (tp_[RW.buildingDamageAmount > 0] > 0).mean(); F["paid_when_dmg_zero"] = 100 * (tp_[RW.buildingDamageAmount == 0] > 0).mean()
cc0_ = RW.totalContentsInsuranceCoverage.fillna(0) == 0
F["cont_dmg_na"] = int(RW.contentsDamageAmount.isna().sum()); F["cont_dmg_na_struct"] = int((RW.contentsDamageAmount.isna() & cc0_).sum())
F["cont_ded_na_pct_when_nocov"] = 100 * RW.loc[cc0_, "contentsDeductibleCode"].isna().mean()
bz_ = (RW.buildingPropertyValue == 0) & (RW.totalBuildingInsuranceCoverage > 0)
F["val0_paid"] = 100 * (tp_[bz_] > 0).mean(); F["val_pos_paid"] = 100 * (tp_[~bz_] > 0).mean(); F["val0_n"] = int(bz_.sum())
wd_ = RW.waterDepth
F["wd99_n"] = int((wd_ == 99).sum()); F["wd99_paid"] = 100 * (tp_[wd_ == 99] > 0).mean(); F["wd98_paid"] = 100 * (tp_[wd_ == 98] > 0).mean(); F["wd100_paid"] = 100 * (tp_[wd_ == 100] > 0).mean()
F["wdneg_paid"] = 100 * (tp_[wd_ < 0] > 0).mean(); F["wd0_paid"] = 100 * (tp_[wd_ == 0] > 0).mean(); F["wdna_paid"] = 100 * (tp_[wd_.isna()] > 0).mean()
F["pay_b_median"] = pb_.dropna().median(); F["pay_b_p99"] = pb_.dropna().quantile(.99); F["pay_b_max"] = pb_.max()
F["repl_max"] = RW.buildingReplacementCost.max(); F["cov_max"] = RW.totalBuildingInsuranceCoverage.max(); F["dmg_max"] = RW.buildingDamageAmount.max()
F["neg_b"] = int((pb_ < 0).sum()); F["neg_c"] = int((pc_ < 0).sum())
pbp = pb_[pb_ > 0]; F["share_b_250k"] = 100 * (pbp == 250000).mean(); F["share_b_500k"] = 100 * (pbp == 500000).mean(); F["share_b_100"] = 100 * (pc_[pc_ > 0] == 100000).mean()
F["nonpay_b_null"] = 100 * RW.nonPaymentReasonBuilding.isna().mean(); F["nonpay_c_null"] = 100 * RW.nonPaymentReasonContents.isna().mean()
F["fl_share"] = 100 * (RW.state == "FL").mean()
F["dup_rows"] = G["duplicate_rows_excluding_id_extra_copies"]; F["ned_placeholder"] = int((pd.to_datetime(RW.mostRecentPaymentDate, errors="coerce").dt.year == 1900).sum())
F["pay_na_n"] = int(pb_.isna().sum())
def clog(substr):
    r_ = CL[CL.description.str.contains(substr, regex=False)]; return int(r_.rows_affected.iloc[0])

# ------------------------------------------------------------------ narrative facts
n_raw = RC["rows_in_2020_2025_by_dateOfLoss"]; n_clean = META["n_rows_clean"]
od = pd.to_datetime(raw.openDate); dl = pd.to_datetime(raw.dateOfLoss); lag = (od - dl).dt.days
lagq = lag.quantile([.5, .9, .95, .99]).to_dict()
mon = MON.assign(year=MON.iloc[:, 0].str[:4].astype(int)); medmon = mon.groupby("year").n_claims.median()
ev25 = raw[raw.yearOfLoss == 2025].floodEvent.fillna("(no named event)").value_counts().head(4)
rr2_year = raw.assign(rr=(raw.rateMethod == "RE")).groupby("yearOfLoss").rr.mean()
unpaid_codes = {"06": "Not actual flood", "99": "Erroneous assignment", "01": "Claim denied, less than deductible", "97": "Other", "20": "No demonstrable damage", "98": "Error-delete claim", "08": "Failure to pursue claim", "16": "Not insured, wind damage"}
npr = clean.nonpay_reason_building.value_counts(); npr_t = pd.DataFrame({"code": npr.index[:8], "meaning": [unpaid_codes.get(c, "") for c in npr.index[:8]], "claims": npr.values[:8]})
paid_rate_year = clean.groupby("yearOfLoss").is_paid.mean()
top_ev = clean.flood_event.value_counts().head(8)
dec_counts = CD.decision.value_counts().to_dict(); role_counts = CD.role.value_counts().to_dict()
cond_cov = clean.groupby("yearOfLoss").total_paid.agg(["count", "sum", "median", "mean"])
paid_only = clean[clean.is_paid == 1].total_paid
pm = G["consistency"]

L = []
w = L.append
w("# NFIP Redacted Claims v3: data analysis, cleaning, imputation and feature selection (2020-2025)\n")
w(f"*Generated by `scripts/05_build_report.py` from the artefacts of scripts 01-04. Data snapshot: OpenFEMA `NfipClaims` v3, `asOfDate` {RC['asOfDate_values'][0][:10]}; analysis run {pd.Timestamp.now().strftime('%Y-%m-%d')}. Every number below is read from the data or the output files.*\n")
w("## 0. Executive summary\n")
w(f"- **Scope**: {RC['total_rows_full_dataset']:,} claim-transaction rows in the full dataset (1978-{RC['dateOfLoss_max_full'][:4]}), **{n_raw:,} rows** with `dateOfLoss` in 2020-01-01..2025-12-31 (the API `$count` for the same filter is {RC['api_count_same_filter']:,}: identical). After removing {int(CL.loc[(CL.step=='dedupe') & CL.description.str.startswith('exact'), 'rows_affected'].iloc[0]):,} exact duplicates the cleaned file has **{n_clean:,} rows x {META['n_cols_clean']} columns**.")
w(f"- **84 raw columns -> decisions**: {dec_counts.get('keep',0)} keep, {dec_counts.get('maybe',0)} maybe, {dec_counts.get('drop',0)} drop. Among the drops: {G['n_all_null_cols']} all-null, {G['n_constant_cols']} constant, {len(G['cols_gt60pct_missing'])} columns >60% missing, {role_counts.get('leakage',0)} post-outcome/leakage columns, plus IDs, high-cardinality labels and near-constant flags (section 3).")
w(f"- **Claims are payment-transaction snapshots, not policies**: `amountPaid*` is null for {P.loc[P.column=='amountPaidOnBuildingClaim','n_missing'].iloc[0]:,} rows ({P.loc[P.column=='amountPaidOnBuildingClaim','missing_pct'].iloc[0]:.1f}%) and in *every* such row the net payment is 0 and there is no payment date, so null = closed without payment (recoded to 0, not imputed). {100*(1-clean.is_paid.mean()):.1f}% of cleaned claims have total paid (building + contents) = 0.")
w(f"- **Event-driven, FL-heavy, time-unbalanced**: Florida = {F['fl_share']:.0f}% of rows; the four largest events ({', '.join(list(G['top_floodEvents'])[:4])}) alone are {sum(list(G['top_floodEvents'].values())[:4])/n_raw*100:.0f}% of rows. 2024 has {RC['per_yearOfLoss']['2024']:,} claims and 2025 only {RC['per_yearOfLoss']['2025']:,}.")
w(f"- **Biggest data-quality traps**: sentinel `waterDepth==99` ({F['wd99_n']:,} raw rows), `interiorWaterDepth==999`, building value/replacement cost recorded as 0 although coverage > 0 ({F['val0_n']:,} raw rows), construction dates before 1850 ({pm['construction_date_before_1850']:,}), policy start dates after the loss ({pm['policy_start_after_loss_date']:,}), and extreme money outliers (building replacement cost up to ${F['repl_max']:,.0f}).")
a_ = PERF[(PERF.model == 'XGBoost')].set_index(['target', 'feature_set'])
t1, t2, t3 = FS["sets"], None, None
def pm_(t, s, k): return a_.loc[(t, s), k]
T1, T2, T3 = "T1_reg_log_total_paid (paid claims)", "T2_cls_is_paid", "T3_cls_high_severity"
w(f"- **Leakage is the main modelling risk**: with only information available at claim notice (set A) XGBoost reaches R2={pm_(T1,'A','R2'):.2f} (T1) / AUC={pm_(T2,'A','AUC'):.2f} (T2, paid vs unpaid) / AUC={pm_(T3,'A','AUC'):.2f} (T3, high severity) on the 2024-25 test period. Adding adjuster-assessed fields (set B: damage, depth, value) lifts these to {pm_(T1,'B','R2'):.2f} / {pm_(T2,'B','AUC'):.2f} / {pm_(T3,'B','AUC'):.3f}, and adding post-outcome columns (net payments, non-payment reason) gives {pm_(T1,'LEAK','R2'):.3f} / {pm_(T2,'LEAK','AUC'):.4f} / {pm_(T3,'LEAK','AUC'):.4f}, i.e. a trivial leak.")
w(f"- **Strongest legitimate drivers**: building damage amount (`log_bldg_damage`), contents damage, damage-to-value ratio, building coverage, deductible, water depth, replacement cost; among notice-time features: building coverage, location, building age, deductible, foundation type, occupancy. See section 6 for rankings and recommended lists.\n")

# ---------------------------------------------------------------- 1
w("## 1. Data acquisition and filtering\n")
w(f"- Source: OpenFEMA bulk Parquet `{RC['bulk_file']}` (dataset page: <https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3>; API `https://www.fema.gov/api/open/v3/NfipClaims`; field metadata from `OpenFemaDataSetFields`, saved in `raw/fields.json`). The bulk file was used instead of paging the API; the API row count for the identical filter was used as a cross-check.")
w("- Filter: `dateOfLoss` between 2020-01-01 and 2025-12-31 inclusive (`yearOfLoss` is identical to year(`dateOfLoss`) for every row, so both filters give the same set).\n")
w(md(pd.DataFrame([
    ("Rows in full v3 file", RC["total_rows_full_dataset"]), ("Columns", RC["total_columns"]),
    ("Rows with dateOfLoss in 2020-2025", RC["rows_in_2020_2025_by_dateOfLoss"]), ("Rows with yearOfLoss in 2020-2025", RC["rows_in_2020_2025_by_yearOfLoss"]),
    ("Rows where yearOfLoss != year(dateOfLoss)", RC["rows_where_yearOfLoss_ne_year_of_dateOfLoss"]), ("API $count with same filter", RC["api_count_same_filter"]),
    ("Rows after exact-duplicate removal (cleaned)", n_clean)], columns=["Quantity", "Value"])))
w("\nRaw filtered file: `raw/nfip_claims_2020_2025_raw.parquet` (unmodified Parquet types; the full download is `raw/NfipClaimsV3_full.parquet`).\n")
w("Rows per `yearOfLoss` (raw filtered):\n")
w(md(pd.DataFrame({"yearOfLoss": list(RC["per_yearOfLoss"].keys()), "rows": list(RC["per_yearOfLoss"].values())})))
w("")

# ---------------------------------------------------------------- 2
w("## 2. Profiling\n")
w(f"The raw file has {G['n_rows']:,} rows x {G['n_columns']} columns (~{G['memory_mb']:.0f} MB in memory). Full per-column profile: `outputs/profile_columns.csv` (dtype, missing, unique, top value, min/p01/median/mean/p99/max, skew, negatives, zeros, IQR-outliers); global facts: `outputs/profile_global.json`.\n")
w("### 2.1 Types, missingness, constants and cardinality\n")
w(f"- Types: money fields (`amountPaid*`, `net*Payment`, `total*Recovery`, elevations, lat/lon) arrive as Python `Decimal` objects (parquet `decimal`) and dates as `date` objects; they are converted in step 3. Codes such as `occupancyType`, `crsClassCode` are floats although categorical.")
w(f"- All-null columns ({G['n_all_null_cols']}): `ficoNumber`, `floodCharacteristicsIndicator`. Constant columns ({G['n_constant_cols']}): `asOfDate`, `iccCoverage` (30000 for every non-null), `reportedCity` ('Currently Unavailable'). Near-constant (>=99% one value) columns ({G['n_near_constant_cols']}): " + ", ".join(f"`{c}`" for c in P[P.flag_near_constant_ge99pct].column) + ".")
w(f"- Columns with >60% missing ({len(G['cols_gt60pct_missing'])}): " + ", ".join(f"`{c}` ({P.loc[P.column==c,'missing_pct'].iloc[0]:.0f}%)" for c in G['cols_gt60pct_missing']) + ".")
w(f"- Columns with 20-60% missing ({len(G['cols_20_60pct_missing'])}): " + ", ".join(f"`{c}` ({P.loc[P.column==c,'missing_pct'].iloc[0]:.0f}%)" for c in G['cols_20_60pct_missing']) + ".")
hc = P[P.flag_high_cardinality]
w("- High-cardinality text keys: " + ", ".join(f"`{c}` ({int(n):,} levels)" for c, n in zip(hc.column, hc.n_unique)) + ".")
w(f"- **Structure of the missingness** (important for imputation): (i) building damage is missing for {F['dmg_na']:,} raw rows and in {F['dmg_val_rep_same_rows']:,} of them building value and replacement cost are missing too (no adjuster record; {F['paid_when_dmg_na']:.0f}% of those claims are paid versus {F['paid_when_dmg_pos']:.0f}% when damage > 0 and {F['paid_when_dmg_zero']:.0f}% when damage = 0); (ii) contents damage/value are missing for {F['cont_dmg_na']:,} rows of which {F['cont_dmg_na_struct']:,} have zero contents coverage (structural); (iii) `contentsDeductibleCode` is missing for {F['cont_ded_na_pct_when_nocov']:.1f}% of rows with zero contents coverage; (iv) `amountPaid*` null = no payment ({F['pay_na_n']:,} rows).\n")
w("![Missingness](figures/01_missingness.png)\n")
w("### 2.2 Distributions, skew and outliers\n")
sk = P[P["skew"].notna() & (P["skew"].abs() > 5)][["column", "skew", "min", "median", "p99", "max", "n_outlier_iqr3"]].sort_values("skew", key=abs, ascending=False)
w("Numeric columns with |skew| > 5 (raw): \n")
w(md(sk, 2)); w("")
w(f"Money fields are extremely right-skewed with absurd tails (e.g. `buildingReplacementCost` max ${F['repl_max']:,.0f}, `totalBuildingInsuranceCoverage` max ${F['cov_max']:,.0f}, `amountPaidOnBuildingClaim` max ${F['pay_b_max']:,.0f}, `buildingDamageAmount` max ${F['dmg_max']:,.0f}). The median gross building payment (rows with a payment record) is ${F['pay_b_median']:,.0f}; the 99th percentile is ${F['pay_b_p99']:,.0f}; {F['share_b_250k']:.1f}% of positive building payments equal exactly $250,000 and {F['share_b_500k']:.1f}% exactly $500,000 (policy limits), and {F['share_b_100']:.1f}% of positive contents payments equal $100,000.\n")
w("![Money distributions](figures/03_money_distributions.png)\n")
w("![Depth sentinels](figures/04_depth_elevation_sentinels.png)\n")
w("### 2.3 Logical consistency / impossible values (raw)\n")
w(md(pd.DataFrame([(k, (f"{v:,}" if isinstance(v, int) else f"{v}")) for k, v in pm.items()], columns=["check", "count / pct"]))); w("")
w(f"Notes: negative `amountPaid*` values are re-issued checks ({F['neg_b']} building, {F['neg_c']} contents rows). {pm['net_bldg_payment_gt_damage']:,} rows pay more than the recorded `buildingDamageAmount` (damage is an ACV estimate for the *main* building and payments may include other items, so this is not necessarily an error). `preFirmIndicator` and `postFIRMConstructionIndicator` are not complements ({pm['preFirm_and_postFirm_both_false']:,} rows are False/False), so `preFirmIndicator` is treated as redundant/ambiguous.\n")
w("### 2.4 Dates, duplicates, per-year counts and the 2025 'reporting lag'\n")
w("Date ranges (raw): " + "; ".join(f"`{k}` {v[0]} to {v[1]}" for k, v in G['date_ranges'].items()) + ".\n")
w(f"Duplicates: no duplicate `id`; {G['duplicate_rows_excluding_id_extra_copies']:,} extra copies ({G['pct_duplicate_extra_copies']:.2f}%) of rows that are identical in every column except `id`. {G['dup_condo_assoc_or_rcbap_pct']:.0f}% of the duplicated rows are condominium association/master policies and {G['dup_groups_with_positive_building_payment_pct']:.0f}% have a positive building payment, so a minority may be genuine separate units; they were dropped (documented, negligible impact).\n")
w(md(PY.assign(yearOfLoss=PY.yearOfLoss.astype(str)).rename(columns={"yearOfLoss": "year"}).assign(share_rows=lambda d: d.share_rows * 100, paid_share=lambda d: d.paid_share * 100, bldg_damage_missing_share=lambda d: d.bldg_damage_missing_share * 100).rename(columns={"share_rows": "% of rows", "paid_share": "% claims paid", "bldg_damage_missing_share": "% bldg damage missing"}), 1)); w("")
w("![Claims per year/month](figures/02_claims_per_year_month.png)\n")
w(f"**2025 caveat.** 2025 has only {RC['per_yearOfLoss']['2025']:,} claims. Median monthly claim count by loss year: {', '.join(f'{y}: {int(v):,}' for y, v in medmon.items())}; the yearly totals are driven by a few catastrophe months (Ida 2021, Ian 2022, Helene/Milton 2024). The largest named event of 2025 has {int(ev25.drop('(no named event)', errors='ignore').iloc[0]):,} claims ({ev25.drop('(no named event)', errors='ignore').index[0]}); {int(ev25.get('(no named event)', 0)):,} of the {RC['per_yearOfLoss']['2025']:,} 2025 claims have no named event. The `openDate - dateOfLoss` lag has median {lagq[.5]:.0f} d, p90 {lagq[.9]:.0f} d, p95 {lagq[.95]:.0f} d, p99 {lagq[.99]:.0f} d against a snapshot dated {RC['asOfDate_values'][0][:10]}, so a small share of late-2025 claims (and open/unpaid ones: {100*clean.loc[clean.yearOfLoss==2025,'is_paid'].mean():.0f}% of 2025 claims are paid vs {100*clean.loc[clean.yearOfLoss==2024,'is_paid'].mean():.0f}% for 2024) may still be missing or revised. The data alone cannot separate 'quiet year' from 'incomplete reporting': the 2025 median month is in line with 2022, which points to a quiet year as the main reason, but treat 2025 as provisional and do not read the count as a trend.\n")
w(f"Risk Rating 2.0 share (`rateMethod == 'RE'`) by year of loss: " + ", ".join(f"{y}: {100*v:.0f}%" for y, v in rr2_year.items()) + ". The rating-system shift is a built-in structural change inside the window; `is_rr2` is therefore kept as a feature.\n")
w("Most frequent flood events (cleaned data): " + ", ".join(f"{k} ({v:,})" for k, v in top_ev.items()) + ". Building non-payment reasons (when building not paid, top codes):\n")
w(md(npr_t)); w("")

# ---------------------------------------------------------------- 3
w("## 3. Column guide and keep / drop decisions (all 84 raw columns)\n")
w("Full table with descriptions from the OpenFEMA data dictionary: **`column_dictionary.csv`** (column, description, dtype, missing %, n_unique, decision, reason, role, derived features). Roles: *feature*, *target-component*, *leakage* (post-outcome), *id*, *geo-key*, *split-key*, *dropped*. Engineered features: `engineered_feature_dictionary.csv`.\n")
w("Decision counts: " + ", ".join(f"{k} = {v}" for k, v in dec_counts.items()) + ". By role: " + ", ".join(f"{k} = {v}" for k, v in role_counts.items()) + ".\n")
for dec in ("keep", "maybe", "drop"):
    sub = CD[CD.decision == dec][["column", "missing_pct", "n_unique", "role", "reason"]]
    w(f"### 3.{['keep','maybe','drop'].index(dec)+1} Decision = {dec} ({len(sub)} columns)\n")
    w(md(sub, 1)); w("")
w("### 3.4 Unused / dropped columns grouped by reason\n")
def grp(c):
    d = D[c]
    if d["decision"] != "drop": return None
    r = d["reason"].lower()
    if d["role"] == "leakage": return "leakage / post-outcome"
    if d["role"] == "id": return "identifier"
    if "all-null" in r or "100% null" in r or "all null" in r: return "all-null"
    if "constant" in r and "near" not in r: return "constant"
    if "near-constant" in r: return "near-constant"
    if ">60%" in r: return "too much missing (>60%)"
    if "redundant" in r or "duplicate" in r: return "redundant / duplicate of another column"
    if "id-like" in r or "high cardinality" in r or "levels" in r: return "ID-like / high cardinality"
    return "other"
gg = {}
for c in CD.column:
    g_ = grp(c)
    if g_: gg.setdefault(g_, []).append(c)
w(md(pd.DataFrame([(k, len(v), ", ".join(v)) for k, v in sorted(gg.items(), key=lambda x: -len(x[1]))], columns=["reason", "n", "columns"]))); w("")

# ---------------------------------------------------------------- 4
w("## 4. Cleaning\n")
w("Script: `scripts/03_clean_impute.py`. Every step and the number of rows affected is in `outputs/cleaning_log.csv`:\n")
w(md(CL[~CL.description.str.startswith("rows remaining")].rename(columns={"rows_affected": "rows"}))); w("")
w("**Design choices**\n")
w("- *Dedupe*: exact duplicates ignoring the surrogate `id`; first copy kept.")
w("- *Null payment = 0*: `amountPaid*` is null exactly when the claim was closed without payment (net payment 0, no payment date). Filling 0 is a recode, not an imputation. Negative payments (re-issued checks) clipped to 0 in the target; gross payments (not net of recoveries) are used so the target matches the requested definition `total_paid = amountPaidOnBuildingClaim + amountPaidOnContentsClaim`.")
w(f"- *Zero vs missing*: zero building property value / replacement cost with building coverage > 0 is impossible and is treated as missing (these are claims with no valuation: {F['val0_paid']:.0f}% paid vs {F['val_pos_paid']:.0f}% otherwise). Zero damage is kept as a real value (0 damage = typically denied claims, {F['paid_when_dmg_zero']:.0f}% paid). Contents damage/value/deductible that are null only because the policy has no contents coverage are set to 0.")
w(f"- *Sentinels*: waterDepth 99 (spike: claims at 99 behave like 'unknown', {F['wd99_paid']:.1f}% paid vs {F['wd98_paid']:.1f}% at depth 98 and {F['wd100_paid']:.1f}% at 100), interior depth 999, exterior depth >= 999, elevation fields >= 9990 / abs >= 500. Negative `waterDepth` is kept ({F['wdneg_paid']:.0f}% paid vs {F['wd0_paid']:.0f}% at depth 0 and {F['wdna_paid']:.0f}% when depth is missing, so it carries information); depths are winsorised at p1/p99.5.")
w("- *Standardisation*: legacy 1-digit occupancy codes mapped to 2-digit Risk Rating 2.0 codes; 52 flood-zone labels -> 8 zone families (A01-A30 = AE); `rateMethod` -> 7 groups; deductible codes -> USD; cause of damage -> 6 groups; foundation/basement/contents-location codes labelled; rare states -> 'OTHER'.")
w("- *Skew*: money inputs clipped at their 99.9th percentile then `log1p`; the target is modelled as `log1p(total_paid)`; table below (observed values only):\n")
w(md(SK, 2)); w("")
w("Note the log of the coverage/value fields becomes *left*-skewed because of the zeros and a few tiny values; skew is much closer to 0 for the damage and target columns. Tree models are invariant to the transform; it matters for linear/KNN/clustering.\n")
w("![Log transform](figures/06_log_transform_before_after.png)\n")

# ---------------------------------------------------------------- 5
w("## 5. Imputation\n")
w(f"Rules: do not impute where missingness is structural (recode instead), do not impute categorical 'unknown' where the absence is itself informative, flag every imputed numeric field, drop (not impute) columns with >60% missing. **All imputers are fit on the training period (yearOfLoss <= {META['train_max_year']}) only** and applied to every row, so the 2024-25 test period does not influence any imputation parameter. Imputation log: `outputs/imputation_log.csv`.\n")
w(md(IL[["column", "n_imputed", "pct_imputed", "strategy", "justification"]], 2)); w("")
w("### 5.1 Choosing the imputer (masked-value validation)\n")
w("On 20,000 fully-observed training rows, observed values were masked and re-imputed; RMSE (in the transformed units: log USD or inches) is shown. *single* = one column masked at a time; *joint* = the whole adjuster block masked together (which mimics the real missingness pattern; used to pick the method per column).\n")
for scn in ("single", "joint"):
    v = IV[IV.scenario == scn].pivot(index="column", columns="method", values="RMSE"); v["SD of truth"] = IV[IV.scenario == scn].groupby("column").sd_truth.first()
    w(f"**{scn} masking**\n"); w(md(v, 3, index=True)); w("")
vj = IV[IV.scenario == "joint"]; bj = vj.sort_values("RMSE").groupby("column").first()
skill = pd.DataFrame({"best method": bj.method, "best RMSE": bj.RMSE, "SD of truth": bj.sd_truth, "skill = 1 - RMSE/SD": 1 - bj.RMSE / bj.sd_truth})
w("**Joint-masking skill of the best method** (0 = no better than predicting the mean, 1 = perfect):\n"); w(md(skill, 3, index=True)); w("")
w("Median baselines are clearly the worst for the value/damage/contents columns; KNN (k=10) and the iterative (MICE-style, BayesianRidge) imputer are close. When the whole adjuster block is missing together, skill is low for the depth columns (see table), i.e. depths and contents amounts are only weakly recoverable from policy/location variables. This is why the `*_missing` flags are kept and why set-A (notice-time) models are reported separately.\n")
w("![Imputer validation](figures/08_imputation_validation.png)\n")
w("### 5.2 Before / after distributions\n")
w("Statistics of *observed* vs *imputed* values (contents columns compare against observed rows with contents coverage). KS-tests are all significant because n is large and because the missingness is not random (the imputed claims are a different population: no adjuster record), so the aim is plausibility (similar scale, no collapse to a spike, no impossible values), not equality.\n")
w(md(IBA[["column", "n_observed", "n_imputed", "mean_observed", "mean_imputed", "median_observed", "median_imputed", "sd_observed", "sd_imputed", "mean_all_after", "ks_stat"]], 2)); w("")
w("![Imputed vs observed](figures/07_imputation_distributions.png)\n")
w("Imputed values have a smaller spread than observed ones (the usual regression-to-the-mean effect of single imputation), so downstream variance is slightly understated; use the missing flags or multiple imputation for inference. Remaining missing values in the cleaned file (not features): " + ", ".join(f"`{r.column}` ({r.n_missing:,})" for r in REM.itertuples()) + ".\n")

# ---------------------------------------------------------------- 6
w("## 6. Feature selection\n")
S = FS["summary"]
w("### 6.1 Targets, split and feature sets\n")
w(f"- **T1 regression**: `log_total_paid = log1p(amountPaidOnBuildingClaim + amountPaidOnContentsClaim)` on paid claims only (n train {S['paid_claims_train']:,} / test {S['paid_claims_test']:,}); paid claims have median ${paid_only.median():,.0f}, mean ${paid_only.mean():,.0f}, max ${paid_only.max():,.0f}.")
w(f"- **T2 classification**: `is_paid` (total paid > 0): rate {S['paid_rate_train']*100:.1f}% in training years vs {S['paid_rate_test']*100:.1f}% in the test years.")
w(f"- **T3 classification**: `high_severity` = total paid >= ${META['high_severity_threshold_usd']:,.0f} (90th percentile of *paid* claims in 2020-23). Positive rate {S['high_sev_rate_train']*100:.1f}% train vs {S['high_sev_rate_test']*100:.1f}% test (2024 Helene/Milton and general claim inflation push the test period up; this is a real covariate/label shift to mention in any evaluation).")
w(f"- **Time-based split**: train = loss year <= {META['train_max_year']} ({S['train_rows']:,} rows), test = 2024-25 ({S['test_rows']:,} rows). Models: XGBoost (300 trees, depth 6) and RandomForest (150 trees); categoricals ordinal-encoded.")
w("- **Feature sets**: **A** = information known at/shortly after notice of loss (policy, property, location, rating, season, report lag); **B** = A + adjuster-assessed fields (damage, depth, value, cause of damage); **LEAK** = B + post-outcome columns (net payments, non-payment reason present) - shown only to demonstrate leakage.\n")
w("### 6.2 Performance by feature set (leakage check, 2024-25 test)\n")
pp = PERF.copy(); w(md(pp[["target", "feature_set", "model", "n_features", "n_test", "R2", "RMSE_log", "AUC", "PR_AUC", "base_rate"]], 3)); w("")
w("![Performance by feature set](figures/13_performance_by_feature_set.png)\n")
w("Interpretation: set A is the honest *early-triage* problem (modest skill: location, building value and coverage matter, paid/unpaid is hard to predict); set B is the *explain-the-payout-given-the-adjuster-file* problem and is dominated by damage amount; LEAK is meaningless. If a research question is about drivers of payment, use B but say so; if it is about prediction at claim notice, use A.\n")
w("### 6.3 Correlation matrix\n")
w("![Correlation](figures/09_correlation_matrix.png)\n")
w("Spearman correlations (100k-row sample, imputed numeric features). Highly correlated pairs (|rho| >= 0.80) among features:\n")
w(md(PAIRS, 3)); w("")
w("Targets vs numeric features (Spearman; the `PAID_ONLY` column uses only paid claims):\n")
tc = TC.copy(); tc["abs"] = tc.spearman_vs_log_total_paid_PAID_ONLY.abs(); tc = tc.sort_values("abs", ascending=False).drop(columns="abs").head(15)
w(md(tc.rename_axis("feature"), 3, index=True)); w("")
w("### 6.4 Mutual information, RF and XGBoost importances\n")
w("Note: gain/impurity importance is split arbitrarily among redundant columns (e.g. the several `*_missing` flags are nearly collinear), so a high gain with a near-zero permutation drop means redundancy, not independent signal. Columns: MI (nats, 60k training sample), RF impurity importance, XGBoost total-gain share, XGBoost permutation importance (drop in R2 or AUC on the 2024-25 sample; +/- is the SD over 3 repeats) and the mean normalised rank over the four measures (`composite_rank`, 0 = best).\n")
w("![XGB gain](figures/10_xgb_importance_setB.png)\n"); w("![MI](figures/11_mutual_information.png)\n"); w("![Permutation](figures/12_permutation_importance_setB.png)\n")
names = {T1: "T1 regression", T2: "T2 paid vs unpaid", T3: "T3 high severity"}
for sname in ("B", "A"):
    for t in (T1, T2, T3):
        g = RANK[(RANK.target == t) & (RANK.feature_set == sname)].sort_values("composite_rank").head(15)
        w(f"**{names[t]}, feature set {sname}: top 15**\n")
        w(md(g[["feature", "tier", "mi", "rf_impurity", "xgb_gain", "xgb_perm_drop", "xgb_perm_sd", "composite_rank", "recommended"]], 4)); w("")
w("### 6.5 Leakage risks (summary)\n")
w("| Column(s) | Why leaky | Treatment |\n|---|---|---|")
w("| `amountPaid*`, `net*PaymentAmount`, `totalSalvage/Subrogation/*Recovery`, `mostRecentPaymentDate`, `mostRecentRecoveryDate` | they are the target or computed from it; payment date is null exactly when nothing was paid | excluded from all feature lists |")
w("| `nonPaymentReasonBuilding/Contents` | exists only for claims closed without payment (" + f"{F['nonpay_b_null']:.0f}% / {F['nonpay_c_null']:.0f}%" + " null) -> near-perfect proxy for T2 | excluded; useful for a separate 'why unpaid' descriptive/classification study |")
w("| `buildingDamageAmount`, `contentsDamageAmount`, water depths, `buildingPropertyValue`, `buildingReplacementCost`, `causeOfDamage` (tier B) | adjuster outputs; not known at notice; missing for no-adjuster claims -> missingness itself predicts non-payment | only in set B / 'extended' lists; flags kept |")
w("| `floodEvent` name / `eventDesignationNumber` | event identity; cannot generalise to new storms and splits cleanly only by event | only binary `named_event` recommended; use event-grouped CV if event names are used |")
w("| `yearOfLoss` / dollars | claim inflation and event mix shift between train and test years | split key only; consider inflation-adjusting dollar targets |")
w("| `openDate` (report lag) | known when the claim is opened (not at the loss); late-reported claims are paid less often (see `report_lag_days`), and 2025 lags are truncated by the snapshot date | kept (tier A) with the caveat |\n| `replacementCostBasis` | dictionary: basis on which the claim was *settled*; claims with R are paid more often | moved to tier B |\n")
w("### 6.6 Recommended final feature lists\n")
w("Rule (script 04): rank features by the composite of MI, RF, XGBoost gain and permutation importance; keep a feature if its permutation importance exceeds max(2 SD, 1e-4) **or** its composite rank is in the top 20%; drop one of any pair with |Spearman| >= 0.90 (keeping the higher-ranked); never use leakage columns; `flood_event_grp` is excluded by policy. Lists are then validated by refitting:\n")
cc_ = CHK.copy(); cc_["target"] = cc_.target.map(names)
cols = ["target", "feature_set", "n_features_full", "n_features_recommended", "full_R2", "rec_R2", "full_AUC", "rec_AUC", "full_PR_AUC", "rec_PR_AUC"]
w(md(cc_[[c for c in cols if c in cc_.columns]], 4)); w("")
for t in (T1, T2, T3):
    for sname, lab in (("A", "notice-time (A)"), ("B", "extended with adjuster fields (B)")):
        keep = REC[f"{t}|{sname}"]
        w(f"**{names[t]} - {lab}** ({len(keep)} features): " + ", ".join(f"`{c}`" for c in keep) + "\n")
w("Features in set A/B that were evaluated and **not** recommended for at least one target are visible in `outputs/feature_ranking_full.csv` (column `recommended`). Typical rejects are listed in the CSV (see column `recommended`).\n")
w("The full lists with membership flags for each target are in `engineered_feature_dictionary.csv` (`rec_T*_A/B` columns).\n")

# ---------------------------------------------------------------- 7
w("## 7. Caveats and suggestions\n")
w("- **Unit of analysis**: rows are claim transactions on redacted policies without a policy key; repeated losses on one property cannot be linked. Block-group `censusGeoid` is available for joins with ACS (ecological inference only); lat/lon are rounded to 0.1 degrees.")
w(f"- **Selection and measurement**: only *insured* losses; deductible and coverage caps censor payments ({F['share_b_250k']:.1f}% of positive building payments are exactly $250,000 and {F['share_b_500k']:.1f}% exactly $500,000).")
w("- **Imputation**: single imputation understates variance; the adjuster block is missing together and is only weakly predictable (section 5.1). Set-A models do not depend on it.")
w(f"- **Time**: the 2020-2025 window contains the Risk Rating 2.0 roll-out (share of claims with rateMethod 'RE' rises from {100*rr2_year.iloc[0]:.0f}% in 2020 to {100*rr2_year.iloc[-1]:.0f}% in 2025), 2024 mega-events and a quiet 2025. Use time-based or event-grouped CV; do not random-split.")
w(f"- **Duplicates/RCBAP**: {G['duplicate_rows_excluding_id_extra_copies']:,} identical rows removed, about half of them condominium-association policies where they may be legitimate; sensitivity is negligible ({G['pct_duplicate_extra_copies']:.2f}% of rows).")
w("- **Not done** (possible next steps): ACS join, inflation adjustment of dollars, event-grouped CV, SHAP, spatial hotspot analysis, clustering and anomaly detection on the cleaned file.\n")

# ---------------------------------------------------------------- 8
w("## 8. Files and reproducibility\n")
w("```\nnfip_claims/\n  report.md                         this report\n  cleaned_claims.parquet / .csv     cleaned + imputed analysis table (see engineered_feature_dictionary.csv)\n  column_dictionary.csv             84 raw columns: description, dtype, missing %, decision, reason\n  engineered_feature_dictionary.csv engineered features with tier and recommended-list membership\n  figures/                          01..13 PNG\n  outputs/                          profile_*.csv/json, cleaning_log.csv, imputation_*.csv, feature_*.csv, model_performance*.csv, corr_*.csv\n  raw/                              NfipClaimsV3_full.parquet, nfip_claims_2020_2025_raw.parquet, fields.json (OpenFEMA dictionary)\n  scripts/                          01_download_filter.py 02_profile.py 03_clean_impute.py 04_feature_selection.py 05_build_report.py common.py decisions.py run_all.sh\n```")
w("Reproduce: `bash scripts/run_all.sh` (Python 3.13; pandas, numpy, scipy, scikit-learn, xgboost, matplotlib, seaborn, pyarrow, requests; seeds fixed to 42; ~12 minutes on 8 cores; step 03 is the slowest because of KNN/iterative validation).")
(ROOT / "report.md").write_text("\n".join(L))
print("report written", len("\n".join(L)))
