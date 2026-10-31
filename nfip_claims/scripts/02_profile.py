"""Step 2 - profiling of the raw filtered file: dtypes, missingness, uniqueness, constants, cardinality,
distributions, outliers, skew, date ranges, duplicates, per-year counts. Writes outputs/profile_*.csv + figures."""
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from scipy import stats
from common import *
sns.set_theme(style="whitegrid")

raw = pd.read_parquet(FILTERED)
n = len(raw)
# -- type normalisation only for profiling (Decimal -> float); the real cleaning is in step 3
df = raw.copy()
for c in df.columns:
    if df[c].dtype == object and df[c].dropna().map(lambda x: type(x).__name__).eq("Decimal").all() and df[c].notna().any():
        df[c] = dec_to_float(df[c])
date_cols = ["dateOfLoss", "originalConstructionDate", "originalNBDate", "openDate", "mostRecentRecoveryDate", "mostRecentPaymentDate"]
for c in date_cols: df[c] = pd.to_datetime(df[c], errors="coerce")

rows = []
for c in raw.columns:
    s = df[c]
    nn = s.dropna()
    vc = nn.astype(str).value_counts() if s.dtype != "datetime64[ns, UTC]" else nn.astype(str).value_counts()
    top_share = (vc.iloc[0] / n) if len(vc) else np.nan   # share of ALL rows held by the most common non-null value
    top_share_nn = (vc.iloc[0] / len(nn)) if len(vc) else np.nan
    r = dict(column=c, raw_dtype=str(raw[c].dtype), dtype_for_analysis=str(s.dtype), n_missing=int(s.isna().sum()),
             missing_pct=round(100 * s.isna().mean(), 2), n_unique=int(nn.nunique()),
             top_value=vc.index[0] if len(vc) else None, top_value_share_of_nonnull=round(100 * top_share_nn, 2) if len(vc) else np.nan)
    r["flag_all_null"] = bool(s.isna().all())
    r["flag_constant"] = bool(nn.nunique() <= 1 and not s.isna().all())
    r["flag_near_constant_ge99pct"] = bool(len(vc) and top_share_nn >= 0.99 and nn.nunique() > 1)
    r["flag_high_cardinality"] = bool(nn.nunique() > 1000 and s.dtype in (object, "str", "string") or (str(s.dtype) in ("str", "string", "object") and nn.nunique() > 0.05 * n))
    if pd.api.types.is_numeric_dtype(s) and s.dtype != bool and len(nn):
        q = nn.quantile([.01, .25, .5, .75, .99])
        iqr = q[.75] - q[.25]
        r.update(min=nn.min(), p01=q[.01], median=q[.5], mean=nn.mean(), p99=q[.99], max=nn.max(), std=nn.std(),
                 skew=stats.skew(nn), n_negative=int((nn < 0).sum()), n_zero=int((nn == 0).sum()),
                 n_outlier_iqr3=int(((nn < q[.25] - 3 * iqr) | (nn > q[.75] + 3 * iqr)).sum()) if iqr > 0 else np.nan)
    if str(s.dtype).startswith("datetime"):
        r.update(min=str(nn.min()), max=str(nn.max()))
    rows.append(r)
prof = pd.DataFrame(rows)
prof.to_csv(OUT / "profile_columns.csv", index=False)

# ---- global stats
dup_cols = [c for c in raw.columns if c != "id"]
dup_mask = raw.duplicated(subset=dup_cols, keep="first")
dup_all = raw.duplicated(subset=dup_cols, keep=False)
extra = {
    "n_rows": n, "n_columns": raw.shape[1],
    "memory_mb": round(raw.memory_usage(deep=True).sum() / 1e6, 1),
    "duplicate_id": int(raw["id"].duplicated().sum()),
    "duplicate_rows_excluding_id_extra_copies": int(dup_mask.sum()),
    "duplicate_rows_excluding_id_all_members": int(dup_all.sum()),
    "pct_duplicate_extra_copies": round(100 * dup_mask.mean(), 3),
    "dup_groups_with_positive_building_payment_pct": round(100 * (dec_to_float(raw.loc[dup_all, "netBuildingPaymentAmount"]) > 0).mean(), 1),
    "dup_condo_assoc_or_rcbap_pct": round(100 * raw.loc[dup_all, "condominiumCoverageTypeCode"].isin(["A", "L", "H"]).mean(), 1),
    "date_ranges": {c: [str(df[c].min().date()), str(df[c].max().date())] for c in date_cols},
    "n_all_null_cols": int(prof.flag_all_null.sum()), "n_constant_cols": int(prof.flag_constant.sum()),
    "n_near_constant_cols": int(prof.flag_near_constant_ge99pct.sum()),
    "cols_gt60pct_missing": prof.loc[prof.missing_pct > 60, "column"].tolist(),
    "cols_20_60pct_missing": prof.loc[(prof.missing_pct > 20) & (prof.missing_pct <= 60), "column"].tolist(),
    "cols_0_20pct_missing": prof.loc[(prof.missing_pct > 0) & (prof.missing_pct <= 20), "column"].tolist(),
    "cols_no_missing": prof.loc[prof.missing_pct == 0, "column"].tolist(),
}
# logical/consistency checks
dl, oc, nb, od, pay = df.dateOfLoss, df.originalConstructionDate, df.originalNBDate, df.openDate, df.mostRecentPaymentDate
extra["consistency"] = {
    "construction_date_before_1850": int((oc.dt.year < 1850).sum()),
    "construction_date_after_loss": int((oc > dl).sum()),
    "construction_date_is_Jan1_pct": round(100 * ((oc.dt.month == 1) & (oc.dt.day == 1)).sum() / oc.notna().sum(), 1),
    "policy_start_after_loss_date": int((nb > dl).sum()), "policy_start_before_1970": int((nb.dt.year < 1970).sum()),
    "openDate_before_loss": int((od < dl).sum()), "openDate_gt_365d_after_loss": int(((od - dl).dt.days > 365).sum()),
    "paymentDate_before_openDate": int((pay < od).sum()),
    "paid_building_negative": int((df.amountPaidOnBuildingClaim < 0).sum()), "paid_contents_negative": int((df.amountPaidOnContentsClaim < 0).sum()),
    "paid_building_null_but_net_building_zero_pct": round(100 * (df.loc[df.amountPaidOnBuildingClaim.isna(), "netBuildingPaymentAmount"] == 0).mean(), 1),
    "waterDepth_eq_99_count (sentinel suspect)": int((df.waterDepth == 99).sum()),
    "waterDepth_negative": int((df.waterDepth < 0).sum()),
    "interiorWaterDepth_eq_999": int((df.interiorWaterDepth == 999).sum()),
    "exteriorWaterDepth_ge_999": int((df.exteriorWaterDepth >= 999).sum()),
    "baseFloodElevation_ge_9990": int((df.baseFloodElevation >= 9990).sum()),
    "lowestFloorElevation_abs_ge_9990": int((df.lowestFloorElevation.abs() >= 9990).sum()),
    "lowestAdjacentGrade_abs_ge_9990": int((df.lowestAdjacentGrade.abs() >= 9990).sum()),
    "elevationDifference_abs_ge_500": int((df.elevationDifference.abs() >= 500).sum()),
    "buildingPropertyValue_zero": int((df.buildingPropertyValue == 0).sum()), "buildingReplacementCost_zero": int((df.buildingReplacementCost == 0).sum()),
    "building_damage_zero": int((df.buildingDamageAmount == 0).sum()),
    "net_bldg_payment_gt_coverage": int((df.netBuildingPaymentAmount > df.totalBuildingInsuranceCoverage + 1).sum()),
    "net_bldg_payment_gt_damage": int((df.netBuildingPaymentAmount > df.buildingDamageAmount).sum()),
    "preFirm_and_postFirm_both_true": int((df.preFirmIndicator & df.postFIRMConstructionIndicator).sum()),
    "preFirm_and_postFirm_both_false": int((~df.preFirmIndicator & ~df.postFIRMConstructionIndicator).sum()),
    "contents_cov_zero_and_contentsDeductible_null_pct": round(100 * df.loc[df.totalContentsInsuranceCoverage == 0, "contentsDeductibleCode"].isna().mean(), 1),
    "contents_cov_zero_and_contentsDamage_null_pct": round(100 * df.loc[df.totalContentsInsuranceCoverage == 0, "contentsDamageAmount"].isna().mean(), 1),
    "zip_code_len_not_5": int((df.reportedZipCode.str.len() != 5).sum()),
    "state_blank": int((df.state.fillna("").str.strip() == "").sum()),
}
# per-year profile
tot_paid = df.amountPaidOnBuildingClaim.fillna(0) + df.amountPaidOnContentsClaim.fillna(0)
py = df.assign(paid=(tot_paid > 0), tot=tot_paid, dmg_missing=df.buildingDamageAmount.isna()).groupby("yearOfLoss").agg(
    n_claims=("id", "size"), paid_share=("paid", "mean"), mean_total_paid=("tot", "mean"), median_total_paid=("tot", "median"),
    total_paid_usd=("tot", "sum"), bldg_damage_missing_share=("dmg_missing", "mean"))
py["share_rows"] = py.n_claims / py.n_claims.sum()
py.to_csv(OUT / "profile_per_year.csv")
extra["per_year"] = py.round(4).reset_index().to_dict("records")
monthly = df.groupby(df.dateOfLoss.dt.to_period("M")).size()
monthly.index = monthly.index.astype(str); monthly.to_csv(OUT / "profile_monthly_counts.csv", header=["n_claims"])
extra["claims_2025_by_month"] = {k: int(v) for k, v in monthly.items() if k.startswith("2025")}
extra["top_floodEvents"] = df.floodEvent.value_counts().head(12).to_dict()
extra["top_states"] = df.state.value_counts().head(8).to_dict()
extra["lag_open_minus_loss_days"] = df.assign(l=(df.openDate - df.dateOfLoss).dt.days).l.describe().round(1).to_dict()
save_json(extra, "profile_global.json")

# ---------------- figures
# 1 missingness
m = prof.sort_values("missing_pct", ascending=False)
fig, ax = plt.subplots(figsize=(8, 15))
colors = np.where(m.missing_pct > 60, "#c0392b", np.where(m.missing_pct > 20, "#e67e22", np.where(m.missing_pct > 0, "#f1c40f", "#27ae60")))
ax.barh(m.column, m.missing_pct, color=colors); ax.invert_yaxis(); ax.set_xlabel("% missing (raw)")
ax.axvline(60, ls="--", c="k", lw=.8); ax.set_title("Missingness per column (raw, 2020-2025)\nred >60%, orange 20-60%, yellow 0-20%, green none")
ax.tick_params(axis="y", labelsize=7); plt.tight_layout(); plt.savefig(FIG / "01_missingness.png", dpi=130); plt.close()
# 2 per-year & monthly counts
fig, ax = plt.subplots(1, 2, figsize=(14, 4.5))
py.n_claims.plot.bar(ax=ax[0], color="#2980b9"); ax[0].set_title("Claim records by yearOfLoss"); ax[0].set_ylabel("rows")
for i, v in enumerate(py.n_claims): ax[0].text(i, v, f"{v:,}", ha="center", va="bottom", fontsize=8)
monthly.index = pd.PeriodIndex(monthly.index, freq="M").to_timestamp()
ax[1].plot(monthly.index, monthly.values, c="#2980b9"); ax[1].set_title("Claim records by month of loss (recent months incomplete = reporting lag)")
ax[1].axvline(pd.Timestamp("2025-01-01"), ls="--", c="r"); plt.tight_layout(); plt.savefig(FIG / "02_claims_per_year_month.png", dpi=130); plt.close()
# 3 money distributions raw vs log1p
money = ["amountPaidOnBuildingClaim", "amountPaidOnContentsClaim", "buildingDamageAmount", "contentsDamageAmount",
         "buildingPropertyValue", "buildingReplacementCost", "totalBuildingInsuranceCoverage", "totalContentsInsuranceCoverage"]
fig, axes = plt.subplots(4, 4, figsize=(18, 13))
for i, c in enumerate(money):
    v = df[c].dropna(); v = v[v > 0]
    a = axes[(i // 2) * 1 if False else i // 2, (i % 2) * 2]; b = axes[i // 2, (i % 2) * 2 + 1]
    a.hist(v.clip(upper=v.quantile(.99)), bins=60, color="#7f8c8d"); a.set_title(f"{c}\n(>0, clipped at p99)  skew={stats.skew(v):.1f}", fontsize=9)
    b.hist(np.log10(v), bins=60, color="#16a085"); b.set_title("log10 scale (>0 only)", fontsize=9)
plt.tight_layout(); plt.savefig(FIG / "03_money_distributions.png", dpi=110); plt.close()
# 4 water depth / sentinel spikes
fig, ax = plt.subplots(1, 3, figsize=(16, 4))
df.waterDepth.dropna().clip(-20, 140).hist(bins=161, ax=ax[0]); ax[0].set_title("waterDepth (clipped -20..140): spike at 99 = sentinel")
df.interiorWaterDepth.dropna().clip(-50, 1100).hist(bins=100, ax=ax[1]); ax[1].set_yscale("log"); ax[1].set_title("interiorWaterDepth (log y): spike at 999")
df.elevationDifference.dropna().clip(-60, 60).hist(bins=121, ax=ax[2]); ax[2].set_title("elevationDifference (clipped)")
plt.tight_layout(); plt.savefig(FIG / "04_depth_elevation_sentinels.png", dpi=120); plt.close()
# 5 paid vs unpaid by year, deductible
fig, ax = plt.subplots(1, 2, figsize=(13, 4))
py.paid_share.plot.bar(ax=ax[0], color="#8e44ad"); ax[0].set_title("Share of claims with total paid > 0 (gross paid, NaN as 0)"); ax[0].set_ylim(0, 1)
df["total_paid_tmp"] = tot_paid
g = df[tot_paid > 0].groupby("yearOfLoss").total_paid_tmp.median(); g.plot.bar(ax=ax[1], color="#d35400"); ax[1].set_title("Median total paid among paid claims (USD)")
plt.tight_layout(); plt.savefig(FIG / "05_paid_share_by_year.png", dpi=120); plt.close()
print(prof[["column", "missing_pct", "n_unique", "flag_constant", "flag_near_constant_ge99pct", "flag_high_cardinality", "skew"]].to_string())
print(json.dumps(extra["consistency"], indent=1)); print({k: v for k, v in extra.items() if k not in ("consistency", "per_year")})
