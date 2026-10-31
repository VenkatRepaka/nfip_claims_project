# NFIP claims fields: ignore, keep, or impute

Scope: US NFIP Redacted Claims v3, year of loss 2020–2025, 84-column extract. Local keep/drop/impute rules in `nfip_claims/scripts/decisions.py` and `scripts/03_clean_impute.py` are treated as hypotheses. Missingness percentages below are those hypotheses (this extract), not published FEMA rates. Another notebook owns the MICE / KNN / missForest comparison; this note names a method only where the field’s meaning picks it.

Label used throughout: **FEMA** = the published definition requires this handling. **Judgment** = a modeling choice that the definition allows but does not require.

## Which of these decisions match FEMA field definitions and standard missing-data guidance, and which should change?

### Takeaway

Most drops are right: snapshot and city fields, post-2020 FICO, IDs, net payments and recoveries, and elevation numbers as model features. Several local rules should change. Do not blank every January 1 construction date. Do not drop `obstructionType` only because it is more than 60% empty. Do not median-impute CRS class. Do not mode-fill `replacementCostBasis`. Fill latitude and longitude inside state plus county, and do not round coordinates that FEMA already truncated. A payment NaN filled with 0 is an extract cross-check, not a sentence in the dictionary. A building value of 0 when building coverage is positive is the opposite case: it is not a real zero.

### Cited Findings

- OpenFEMA NfipClaims v3 is the current redacted claims file (entity `NfipClaims`, API `/v3/NfipClaims`). v2 is deprecated and frozen as of 2026-06-01. The v3 dictionary is the field authority for this extract. — [NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3); [v2 deprecation notice](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v2)
- `reportedCity`: “the city value is currently unavailable.” `asOfDate` is “the date when the record was created or last updated in the source system,” not a claim attribute. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `ficoNumber` is the pre-June 2020 event id. “EDN has replaced FICO … after June 2020, flood events are assigned EDN number designations.” The FAQ repeats that NFIP tracks events with an EDN or a FICO number, and that if no EDN was assigned the program searches by state and date of loss. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3); [NFIP data FAQ](https://agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)
- `floodCharacteristicsIndicator` is still defined (1 velocity, 2 low-velocity or ponding, 3 wave, 4 mudflow, 5 erosion). The dictionary does not retire it. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `elevationCertificateIndicator` empty “indicates that this information was not reported; it does not necessarily mean that these properties do not have an Elevation Certificate.” `elevationDifference` of 9999.0 “indicates the field is not reported and/or used for this policy.” Base flood elevation, lowest floor elevation, and lowest adjacent grade are elevations from the certificate or the grade; the dictionary does not name 9990, −9999, or −9980 for those three. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- FAQ: where the legacy system required a value that was not reported, “99XX” is often a placeholder. FEMA “doesn’t recommend using this data as input for commercial catastrophic models” because truncated coordinates shift location and “can affect how accurate the elevation data is.” “FEMA recommends using the provided elevation data as a descriptive field for aggregation only.” — [NFIP data FAQ](https://agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)
- `obstructionType` “describes the type of obstruction that may be present under an elevated building.” `elevatedBuildingIndicator` is the NFIP elevated-building flag (no basement; elevated floor or lowest horizontal member above grade; anchored; specified foundation). — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `disasterAssistanceCoverageRequired`: 0 = Not Required, 1 = SBA, 2 = FEMA, 4 = HHS (canceled 2009-10-01), 5 = Other. `floodWaterDuration` is “number of hours flood water remained in the insured building.” — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `netBuildingPaymentAmount` “is the amountPaidOnBuildingClaim minus any recovery amounts.” `netContentsPaymentAmount` is the contents analogue. Salvage, building-payment recovery, contents recovery, ICC recovery, and subrogation are amounts recovered after payment. `mostRecentRecoveryDate` is the date of that recovery. `mostRecentPaymentDate` is “the most recent date a claim payment was issued.” `nonPaymentReasonBuilding` and `nonPaymentReasonContents` are “Claim Closed Without Payment Reason,” with codes such as below deductible, not a flood, and no demonstrable damage. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `amountPaidOnBuildingClaim` and `amountPaidOnContentsClaim` are dollar amounts paid. “A negative amount may appear which occurs when a check issued to a policy holder is not cashed and has to be re-issued.” The dictionary does not say null means zero. ICC paid “provides up to $30,000” toward mitigation. `iccCoverage` is that coverage limit in whole dollars. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- The FAQ says “the current claims data doesn’t include payment date or claims transaction” and that users cannot report disbursements by calendar year from this dataset. v3 nonetheless publishes `mostRecentPaymentDate`. — [NFIP data FAQ](https://agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data); contradicted on field existence by [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- Community numbers are six-digit NFIP community ids (state code plus FEMA community number, optional suffix for a FIRM revision). `id` is “unique ID assigned to the record.” Latitude and longitude are “approximate … (to 1 decimal place)” and “may result in a point location that exists in an incorrect county or state. Use the state and county fields for record aggregation.” The FAQ says coordinates are “accurate to within approximately six miles,” and that “census tract and county fields are best used for aggregation.” `countyCode` is illustrated as `011` for Broward County, so it is the county portion, not a full state-and-county FIPS. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3); [NFIP data FAQ](https://agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)
- `basementEnclosureCrawlspaceType`: 0 None, 1 finished, 2 unfinished, 3 crawlspace, 4 subgrade crawlspace. `locationOfContents` is codes 1–7 (basement only through manufactured home). `buildingDescriptionCode` is codes 01–20 (main house through townhouse). `foundationType` lists 1 slab through 6 elevated with enclosure not on posts; 0 is not a listed code. `crsClassCode` is classes 1–10, and class 10 is a real class (0% SFHA and 0% non-SFHA discount), not a missing-value code. Beginning with Risk Rating 2.0, “there are no longer percentage discount differences for policies inside and outside of the SFHA.” — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `numberOfFloorsInTheInsuredBuilding` is a code (1 one floor, 2 two, 3 three or more, 4 split-level, 5 manufactured home, 6 townhouse/rowhouse), not a story count. `condominiumCoverageTypeCode`: N not a condominium, U unit, A association, H high-rise RCBAP, L low-rise RCBAP. `numberOfUnits` is “the number of residential and nonresidential units covered by the Condominium Master Policy.” `policyCount` counts insured units; an RCBAP is one contract that “represent[s] multiple policies.” — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `buildingPropertyValue` is “the actual cash value of the main building prior to any flooding event … as estimated by an adjuster.” `buildingReplacementCost` is “estimated cost … to replace the building as reported by the insurer,” not market value, and excludes land and contents. `buildingDamageAmount` is “the actual cash value amount of damage.” `contentsPropertyValue`, `contentsDamageAmount`, and `contentsReplacementCost` are the contents analogues; personal property in a basement is not covered. `replacementCostBasis` “indicates whether the building claim was settled” on R (replacement cost) or A (actual cash value). `causeOfDamage` includes 0 other, 1 tidal, 2 stream/river/lake, 4 rainfall or snowmelt, and B/C/D expedited or remote adjustment; “more than one cause of loss code can be selected,” though the published field is a single code. `waterDepth` is “depth of flood water in inches,” with “instances where measurements were provided in feet.” Exterior and interior depth are inches “relative to the lowest adjacent grade.” — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `postFIRMConstructionIndicator` is yes when start of construction or substantial improvement was after 1974-12-31 or on or after the initial FIRM, whichever is later. `preFirmIndicator` is titled Pre-FIRM, but the same page’s explanatory sentence describes a building “for which construction or substantial improvement occurred after December 31, 1974, or on or after the effective date of an initial FIRM,” which is the post-FIRM rule. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `originalConstructionDate` in the v3 dictionary is only “the original date of the construction of the building.” The legacy TRRP plan, which the FAQ says still applies to historical reporting, is more specific: it is the building-permit date or construction date; for a post-FIRM substantial improvement it is the substantial-improvement date; if the date is before 1900, “report this date as January 1, 1900.” `originalNBDate` is “the original date of the flood policy.” The dataset narrative dates the NFIP to the 1968 Act. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3); [NFIP data FAQ](https://agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data); [TRRP Change 23.1, Original Construction Date](https://nfipservices.floodsmart.gov/sites/default/files/media/document/2025-07/trrp_changes_23.1.pdf)
- `ratedFloodZone` is the FIRM zone used to rate the property (formerly `floodZone`). `floodZoneCurrent` is the zone “where the insured property is currently located.” The FAQ: “the NFIP has some policy rules that allow properties to be rated in one flood zone, while physically located in another.” — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3); [NFIP data FAQ](https://agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)
- `totalBuildingInsuranceCoverage` and `totalContentsInsuranceCoverage` are “total insurance amount in whole dollars” on building and contents. Deductible codes map to fixed dollars (0 = $500, 1 = $1,000, … H = $200 for group policies). — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- Nofal, Gagnon, and others, using OpenFEMA claims: replacement values “are often not reported or are reported as zero or very large numbers”; about 18% of their file had no building replacement value; residential building and contents limits are generally $250,000 and $100,000 “for most” policies, but some reported coverage exceeds those caps; they removed zero replacement values and clearly erroneous fields rather than filling them. They also write that the TRRP manual reports negative water depth when only a basement floods, and that the file also contains negative depth on buildings without a basement. Contents coverage “has to be bought separately,” and contents claims are fewer because contents replacement value is often missing. They treat post-FIRM as the compliance indicator (after 1974 or the community’s initial FIRM). — [Nofal et al., FLOODrisk 2020](https://doi.org/10.3311/floodrisk2020.11.14)
- van Buuren: the missing-indicator method (fill a constant, add a response indicator) “can yield severely biased regression estimates, even under MCAR and for low amounts of missing data.” It is not a general fix. Listwise deletion needs MCAR except for special MNAR cases. Regression imputation can be unbiased under MAR. — [van Buuren, Flexible Imputation of Missing Data, §1.3.7](https://stefvanbuuren.name/fimd/missing-data-pattern.html)
- Harrell: an extra “missing” category “causes serious biases” and “fails even under MCAR,” but “may be OK if values are missing because of not applicable,” and “may be OK for pure prediction” when coefficients are not the goal and future data will be missing in the same pattern. Setting a continuous predictor to a constant and adding an indicator has the same problem. When the fraction missing is small and n is large, simple imputation is enough for prediction. Do not impute an outcome just to keep the row if the imputation model is the thing you would have to trust. — [Harrell, Regression Modeling Strategies, ch. 3](https://hbiostat.org/rmsc/missing)
- Donders et al.: for a categorical predictor the indicator method is an extra category; they do not recommend it for association estimates even under MCAR. — [Donders et al. 2006, Journal of Clinical Epidemiology](https://www.jclinepi.com/article/S0895-4356(06)00197-1/fulltext)

### Inferences

Confirmed, with the procedure a notebook should code. “Drop from features” means the column is not in `X`. It may still be printed in EDA.

**Drop from features, no imputation (FEMA plus no variance or no rows).**

- `asOfDate`, `reportedCity`. FEMA: file timestamp, and city “currently unavailable.” In this extract they are constant. Drop. No flag.
- `agricultureStructureIndicator`, `houseWorship`, `nonProfitIndicator`, `stateOwnedIndicator`, `floodproofedIndicator`. FEMA: real policy-application flags. Judgment: this extract is near-constant or constant, so they have no feature variance. Show a one-line frequency in EDA. Do not recode the rare Trues and do not impute. Constant `floodproofedIndicator` may be under-reporting (FEMA requires a floodproofing certificate before the rating credit); that is a data-quality sentence, not a filled column.
- `ficoNumber`. FEMA: replaced by EDN after June 2020. A 2020–2025 extract that is 100% null matches that. Drop. No flag.
- `floodCharacteristicsIndicator`. FEMA still defines it. Judgment: 100% null in this extract, so drop. Say “unpopulated here,” not “FEMA retired this field.”
- `iccCoverage`. FEMA: ICC limit, “up to $30,000.” Judgment: if every non-missing value is 30000, the column has no variance. Drop. Do not fill the missing rows with 30000 and then use it. The missing rows are not documented as “no ICC eligibility.”
- `eventDesignationNumber`. FEMA: event id, and missing when no EDN was assigned. Judgment: high cardinality and event-specific, so not a feature that will apply to the next storm. Keep only as a grouping key. A binary `named_event` / `has_edn` is the feature. Do not impute an event code. `floodEvent` (the name) gets the same treatment: binary feature, name as a cross-validation group, not a dummy.
- `nfipRatedCommunityNumber`, `nfipCommunityNumberCurrent`, `nfipCommunityName`, `id`. FEMA: community identifier or row id. Judgment: drop from `X`. `id` stays the row key.
- `reportedZipCode`, `countyCode`, `censusGeoid`. FEMA and FAQ: county and census geography are for aggregation; coordinates are privacy-truncated. Judgment: join keys only (ACS, county tables). Do not one-hot thousands of levels. Zip is a quasi-identifier at this cardinality.
- `floodZoneCurrent`. FEMA/FAQ: current map zone, which can differ from the rated zone on purpose. Judgment: not a substitute for `ratedFloodZone`. Show a rated-vs-current crosstab in EDA. Do not impute it to the rated zone (that would hide map revisions). Missing stays missing in that crosstab.
- `preFirmIndicator`. Change the reason. It is not a clean duplicate of `postFIRMConstructionIndicator`. The v3 sentence under the pre-FIRM title restates the post-FIRM rule, and the extract hypothesis (both false on a large block, both true on a small block) means they are not logical negations. Drop `preFirmIndicator`. Use `postFIRMConstructionIndicator`, whose definition matches the 1974 / initial-FIRM rule Nofal uses. No imputation.

**Drop the numbers from features; keep one availability flag; do not impute the numbers (elevation block).**

- Fields: `elevationCertificateIndicator`, `elevationDifference`, `baseFloodElevation`, `lowestAdjacentGrade`, `lowestFloorElevation`.
- FEMA: empty certificate indicator means “not reported,” not “no certificate.” The only documented numeric sentinel is `elevationDifference == 9999`. FAQ: elevation is for description and aggregation, not a cat-model input, because truncated coordinates make an elevation match unreliable.
- Procedure: in EDA, set `elevationDifference == 9999` to missing. Also set the extract’s other impossible codes (hypothesis: 9990, ±9999, −9980) to missing and label those as suspected 99XX placeholders, not as FEMA-named codes. Then plot. For the model, do not impute feet. Store `elevation_reported` = 1 when any of the four numeric fields is present after sentinel removal. Store `elevation_cert_code_reported` = 1 when the indicator is non-null. Name the second flag that way so it is not read as “has a certificate.” The 60% cutoff is a student heuristic. It lines up with FEMA’s “descriptive only” guidance here. It is not itself a FEMA rule.

**Change: do not drop `obstructionType` only for the 60% rule.**

- FEMA: the code applies under an elevated building. Missing on a non-elevated building is “not applicable,” which Harrell treats as a legitimate extra level.
- Procedure: if `elevatedBuildingIndicator` is false, set category `not_applicable` (structural, do not impute). If elevated and blank, set `unknown`. Use that three-way column in EDA and, if the model is predictive, as a feature. Do not mode-impute enclosure type onto slab houses.

**Drop from features with no model flag (high missingness and almost no variation). EDA counts only.**

- `disasterAssistanceCoverageRequired`. FEMA: 0 means Not Required, so blank is not 0. Judgment: if 99% of non-missing rows are already 0, a model flag is an indicator-method column with nothing to predict (van Buuren / Harrell). Drop from `X`. In EDA, tabulate 0/1/2/5 versus missing. Do not fill missing with 0.
- `floodWaterDuration`. FEMA: hours water remained, an inspection measurement. At the hypothesis rate (~94% missing), imputation would invent the column. Drop from `X`. Optional EDA histogram of the observed hours. No indicator in the model.

**Leakage: drop from features. Do not impute. The null is the outcome.**

- `netBuildingPaymentAmount`, `netContentsPaymentAmount`, `netIccPaymentAmount`. FEMA: net equals gross minus recoveries. If the target is gross paid, these leak. If the target is net paid, the gross `amountPaid*` fields leak the other way. Pick one family.
- `totalSalvageRecovery`, `totalBldgClaimPmtRecovery`, `totalContentsClaimPmtRecovery`, `totalIccClaimPmtRecovery`, `totalSubrogationRecovery`, `mostRecentRecoveryDate`. FEMA: post-payment recoveries. Hypothesis ~99.85% missing. Descriptive rate only.
- `mostRecentPaymentDate`. Post-issuance. Null lines up with no payment in the hypothesis, so a “date was missing” flag would leak the target. The FAQ’s claim that the file has no payment date conflicts with v3; follow v3 for existence, and still keep it out of `X`. Do not parse 1900-01-01 into a lag.
- `nonPaymentReasonBuilding`, `nonPaymentReasonContents`. FEMA: reason the claim was closed without payment. Populated only in that case, so missing means “this reason does not apply,” which is outcome information. Descriptive bar chart only. Do not mode-impute code 01.
- `amountPaidOnIncreasedCostOfComplianceClaim`. FEMA: an ICC payment, rare relative to the $30,000 cap. Not a predictor of building or contents payment. Keep a `paid_icc` column only in a descriptive total-outlay table.

**Target construction, not a feature imputation.**

- `amountPaidOnBuildingClaim`, `amountPaidOnContentsClaim`. Procedure in the dollar section below. They are the target (or its parts), never `X`.

**Keep as category `unknown`. Do not mode-impute or mean-impute.**

- `basementEnclosureCrawlspaceType` (~49% missing). FEMA code 0 already means None. Filling 0 or the mode calls an unreported basement “none” or “finished.”
- `locationOfContents` (~45%). Codes 1–7 are places. Missing is not a place.
- `buildingDescriptionCode` (~26%). Codes 01–20. Missing → `unknown`. Rare codes (guest house, barn, pool house) may be grouped to `other` after the unknown level exists. That grouping is judgment.
- `foundationType`. No nulls in the hypothesis, but code 0 is not in the 1–6 list. Map 0 → `unknown`. Do not map 0 → slab.
- `crsClassCode` (~30%). Change the script. `decisions.py` says an unknown category; `03_clean_impute.py` median-imputes class by state and keeps `crs_unknown`. Class 10 already means 0% discount. A state median invents a participating class. Procedure: category `unknown` (or an ordinal with a missing level). Do not median-impute. Interpretation caveat, FEMA: under Risk Rating 2.0 the inside/outside SFHA discount gap was removed, so class is not the same price effect on RE-rated policies as on older methods.
- `ratedFloodZone` (~0.02% missing) → `unknown`, then collapse labels to families (A, AE including A1–A30, AH, AO, A99, V including VE, X including B and C, D). Collapse is judgment; the zone definitions are FEMA.
- `causeOfDamage`: code 0 is “other causes,” not missing. True nulls (~1%) → `unknown`. Group tidal / river / rainfall / other / expedited for EDA. This field is Tier B (below).

**Structural fill with 0 (the zero is a coverage or payment state).**

- Contents damage, contents property value, and contents deductible: where `totalContentsInsuranceCoverage` is 0, NaN → 0. FEMA: contents coverage is a separate limit; there is no contents adjustment and no contents deductible. Remaining NaNs (coverage > 0 but no inspection amount) stay missing, with a flag, and are Tier B.
- `amountPaid*` NaN → 0 only after the notebook shows the cross-check in the dollar section.
- `numberOfUnits`: FEMA defines it for the condominium master policy. Where `condominiumCoverageTypeCode` is N, 0 is structural. Do not median-impute units across all policies (the script’s global median does that). If type is A/H/L and units are missing (~1%), fill the training-set mode inside those types.

**Rare gaps: mode, and only because the hole is tiny.**

- `numberOfFloorsInTheInsuredBuilding` (~0.12%). It is a code 1–6. Mode is acceptable (Harrell: small fraction, large n). Do not average codes. An `unknown` level is also fine; at this rate it will not matter.
- `condominiumCoverageTypeCode` (~0.13%). Mode is acceptable only if the mode is not forced onto contradictory occupancy. If occupancy is 15 or 16 (condo association or condo unit), leave `unknown` rather than filling N (“not a condominium”).
- `buildingDeductibleCode` (~2.4%). Map known codes to the FEMA dollar table. Mode plus a missing flag is acceptable at this rate. Prefer an `unknown` deductible level if the model is a tree and the flag is easy to forget.

**Group median.**

- `latitude`, `longitude` (~0.66% missing). FEMA already publishes one decimal place (~6 miles in the FAQ; 0.1° of latitude is about 11 km). Do not round again. Impute with the training-period median of (state, `countyCode`), then state, then global. County code alone is wrong: FEMA’s example `011` is the county part and repeats across states. Fit medians on the training years only.
- Building age and policy tenure, after the date rules in the January 1 section: median within occupancy × state group, training years only, plus a missingness indicator. The median is for the small invalid set, not for every January 1.

**Sentinel → missing, then indicator, then impute only inside Tier B.**

- `waterDepth` 99, and exterior or interior depth ≥ 999. FEMA does not name these sentinels. The FAQ’s 99XX practice plus the extract spike are the evidence. Recode to NaN before any imputer. Keep a missingness indicator. Negatives stay (see EDA section). The v3 note that some `waterDepth` values were entered in feet means the unit is mixed; do not multiply or divide by 12 without a separate rule. Show that caveat in EDA.
- Elevation sentinels: as in the elevation block. Indicator only; do not impute feet.

**Clip then log1p is a transform, not an imputation.** Apply it after structural zeros and after impossible zeros are set to missing. Do not clip coverage at $250,000 / $100,000: Nofal notes those are the usual residential caps and that some rows legally sit above them (non-residential, RCBAP, other). A high percentile cap (the script uses p99.9) is judgment for skew, and it must be fit on the training years.

**Tier B, known after inspection. Show in EDA. Exclude from an early-triage model.**

- `buildingDamageAmount`, `buildingPropertyValue`, `buildingReplacementCost`, `contentsDamageAmount`, `contentsPropertyValue`, `contentsReplacementCost`, `waterDepth`, `exteriorWaterDepth`, `interiorWaterDepth`, `causeOfDamage`, `replacementCostBasis`, `floodWaterDuration`.
- FEMA: damage, property value, and water marks are adjuster figures; replacement cost is “as reported by the insurer”; basis is how the claim “was settled”; cause codes B/C/D are expedited or remote adjustment.
- `contentsReplacementCost`: dropping it as redundant with contents value is judgment for a payment model. Nofal needed it for a contents loss ratio and found it often missing. Keep it in the Tier B EDA table. Do not put it in an early-triage `X`.

**What should change relative to the local hypotheses**

1. January 1 construction dates: do not set all of them to missing. See that section.
2. `obstructionType`: `not_applicable` / `unknown`, not a hard drop.
3. `crsClassCode`: unknown category, matching `decisions.py`, not the script’s state median.
4. `replacementCostBasis`: unknown category, not mode. Missingness is informative (hypothesis: paid rate much lower when blank). The field is settlement-stage.
5. Lat/lon: median by state and county, no second rounding.
6. `numberOfUnits`: structural 0 for non-master policies, not a global median.
7. Condo type: do not mode-fill N onto condo occupancy codes.
8. `preFirmIndicator`: drop as ill-defined and inconsistent with post-FIRM, not as a perfect copy.
9. Elevation and duration and disaster-assistance: the 60% rule is not FEMA policy. Elevation drops because FEMA says descriptive use and because 9999 means not reported. Duration and disaster-assistance drop because imputation would invent a column and a flag would be almost constant. Do not hang a model flag on those two.
10. `amountPaid` NaN → 0 is justified only with the cross-check shown in the notebook. Dictionary text does not say it.
11. `buildingPropertyValue` 0 with coverage > 0 → missing, never a structural zero. Nofal deleted those rows; for a prediction notebook, recode to missing and flag, and do not delete the claim.
12. Payment-date FAQ vs v3: the field exists in v3. Still keep it out of `X`.

### Gaps

- No published FEMA table says `waterDepth == 99` or exterior/interior `999` are official sentinels. The 99XX FAQ is generic.
- No published rate confirms this extract’s missingness percentages. They stay hypotheses until the notebook prints them.
- The FAQ sentence that claims data have no payment date is not reconciled with the v3 field list in any FEMA note found here.
- No source found that documents −9980 or 9990 as official elevation codes. Only 9999.0 on `elevationDifference` is named.
- Whether `countyCode` in v3 is always three digits, or sometimes a five-digit FIPS, was not checked against a live API row. The dictionary example is three digits.

## For categorical codes, why is an explicit unknown level preferable to mode imputation when missingness is 20-50%?

### Takeaway

At 20–50% missing, mode imputation labels a huge block of policies as the most common building, which the codebook already distinguishes from “not reported.” An explicit unknown level is the right notebook procedure when blank means not applicable or not reported. It is a poor way to estimate a regression coefficient, and the notebook should say that.

### Cited Findings

- Basement code 0 is already “None.” CRS class 10 is already the 0% discount class. Foundation codes are listed as 1–6 only. Contents location and building description are finite legal lists with no “missing” code. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- Empty elevation-certificate indicator means not reported, not the absence of a certificate. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- An extra missing category biases associations even under MCAR (Allison, Jones, Vach and Blettner, Donders, Knol, van der Heijden). The same device “may be OK” when the blank means not applicable, and for pure prediction if missingness in new data will look like missingness in training. — [Harrell, Regression Modeling Strategies, ch. 3](https://hbiostat.org/rmsc/missing); [Donders et al. 2006](https://www.jclinepi.com/article/S0895-4356(06)00197-1/fulltext)
- Filling a constant and adding an indicator “can yield severely biased regression estimates, even under MCAR.” — [van Buuren, Flexible Imputation of Missing Data, §1.3.7](https://stefvanbuuren.name/fimd/missing-data-pattern.html)

### Inferences

Mode imputation answers “what would this code have been if it were missing at random and looked like the majority?” These fields fail that question.

- The codebook already has the value mode imputation would be tempted to use. Basement 0 is none. CRS 10 is no discount. Writing the mode into a blank cell merges “not on the application” with a real structural class.
- At 20–50% the filled class is large enough to move splits and coefficients. The hypothesis rates are about 49% (basement), 45% (contents location), 30% (CRS), 26% (building description). A mode fill would assign the majority code to that entire share.
- The blank is tied to reporting practice (legacy vs Risk Rating 2.0, contents not covered, certificate not used), so it is not a small MCAR gap. Harrell’s “not applicable” case fits `obstructionType` on non-elevated buildings and contents location when there are no contents. The “not reported” case fits CRS and basement type. Both are states the model is allowed to see. They are not values to be guessed.
- Procedure: `fillna("unknown")` after mapping documented codes. Do not `fillna(mode)`. For a linear model, dummy-encode `unknown` and do not interpret that coefficient as the effect of a basement or a CRS discount. For a tree, the extra level is enough; a separate indicator is redundant.
- Do not use this pattern on the 0.12% floor-code gap. There, mode and unknown are almost the same, and Harrell’s small-fraction guidance applies.
- The script’s CRS state-median fights this rule. A numeric median also pretends class is an interval (class 6 is not “halfway” in a way the median respects once class 10 and “non-participant” are different). Keep the category.

### Gaps

- No NFIP-specific study was found that measures the bias from mode-imputing CRS or basement type. The recommendation uses the codebook plus the general missing-data results.
- Whether CRS blanks are non-participating communities or failed extracts is not stated in the v3 dictionary. Class 10 is defined as a class, not as the blank.

## For dollar amounts, when is filling 0 correct versus treating 0 as missing (buildingPropertyValue 0 when coverage > 0)?

### Takeaway

Fill 0 when the field is a payment that did not happen, or a contents amount on a policy with no contents limit. Recode 0 to missing when the field is an adjuster’s value of a building that the policy actually covers. Clip-and-log1p comes after that decision and does not fill holes.

### Cited Findings

- Building and contents payments are dollar amounts paid; a negative “occurs when a check … is not cashed and has to be re-issued.” Net payment is gross minus recoveries. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- Building property value is adjuster-estimated actual cash value of the main building before the flood, in whole dollars. Replacement cost is the insurer’s estimated cost to reconstruct the building, excluding land and contents. Building coverage is the insurance limit in whole dollars. Contents value, contents damage, and contents coverage are separate. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- Nofal et al.: replacement values are often missing, zero, or extremely large; they removed zero and “clearly erroneous” replacement values. Usual residential limits are $250,000 building and $100,000 contents, with exceptions above those caps in the file. Contents coverage is purchased separately, and contents replacement value is reported less often. — [Nofal et al., FLOODrisk 2020](https://doi.org/10.3311/floodrisk2020.11.14)
- Setting a missing continuous predictor to a constant such as zero and adding an indicator biases coefficients. That warning is about filling unknown measurements with 0. It is not a ban on a zero that the coverage rules make true. — [Harrell, Regression Modeling Strategies, ch. 3](https://hbiostat.org/rmsc/missing)

### Inferences

**Fill NaN with 0 (structural).**

- `amountPaidOnBuildingClaim` and `amountPaidOnContentsClaim`. The dictionary does not define null. The hypothesis says null rows have net payment 0, and `mostRecentPaymentDate` is null on the same share (~7.85%). Procedure: in the notebook, cross-tabulate `amountPaid` null against net payment and against payment-date null. If that identity holds, fill 0 for a “dollars paid” target. If it fails on a slice, stop and leave those rows missing. Clip negatives to 0 only for that target, and print the count, because FEMA says those negatives are reissued checks, not negative loss. Do not add a “payment was null” feature beside a target you just filled with 0; that flag is the target.
- Contents damage, contents ACV, and contents deductible where contents coverage is 0. No contents limit means no contents loss to measure and no contents deductible. Fill 0. This is the structural case, not the indicator method Harrell criticizes.
- Building coverage 0 is a real contents-only limit. Keep 0 and add `no_building_cov`. Contents coverage 0: keep 0 and add `no_contents_cov`. `log1p(0) = 0` is correct. The flag matters because a mass at 0 is “no cover,” not “a very small limit.”

**Recode 0 to missing (impossible value).**

- `buildingPropertyValue == 0` while `totalBuildingInsuranceCoverage > 0`. An adjuster ACV of a building the policy covers cannot be zero dollars. Set to NaN, set `bldg_value_was_zero`, and impute only in a Tier B model. Nofal dropped these rows. Dropping changes who is in the sample. For this notebook, recode and flag; mention complete-case deletion as a sensitivity, not the main path.
- `buildingReplacementCost == 0` with building coverage > 0. Same rule. FEMA: replacement cost excludes land, but it is still the cost to reconstruct a building.
- Where building coverage is 0, building value 0 and replacement cost 0 can stay 0 (contents-only; there is no building limit to contradict).
- `contentsPropertyValue < 0` (hypothesis includes −1): invalid. Set to NaN. Do not clip a negative ACV to 0 without looking; a negative is not a structural zero.
- Contents damage NaN while contents coverage > 0: not a structural zero. That is “not valued.” Indicator plus Tier B impute. Do not fill 0, or every uninspected contents claim looks like no damage.

**Do not fill, because the zero would be a fake measurement (Harrell / van Buuren).**

- Do not set missing building damage to 0. FEMA defines it as ACV damage; 0 is a valid inspected amount (no payable damage), and NaN is “no figure.” Keep them apart. Indicator if the column is used at all.
- Do not set missing water depth to 0. Depth 0 is a real measurement.

**Transform, after the zero rule.**

- Right skew: clip at a training-set high percentile, then `log1p`. Do this on coverage, policy count, and (Tier B only) damage, ACV, and replacement cost. The clip is not an imputer. Fit the cap on training years only.

**Deductibles.**

- Map the FEMA code table to dollars. Contents deductible NaN and contents coverage 0 → 0. Other deductible holes: unknown level, or mode plus flag if under about 5%.

### Gaps

- The null-payment identity (NaN amount paid ⇔ net payment 0 ⇔ no payment date) is an extract hypothesis. It was not found as a FEMA sentence.
- No source states the share of `buildingPropertyValue == 0` with positive coverage in 2020–2025 v3. Nofal’s 18% missing replacement figure is an older, longer claims file, not this extract.

## What should the notebook do with January 1 construction-date placeholders and invalid dates?

### Takeaway

Only January 1, 1900 is a required FEMA placeholder, and it means “before 1900,” not “this building’s birthday.” Other January 1 dates should keep the year for age, with a flag that the day is untrustworthy. Dates after the loss, unparseable dates, and policy dates before 1968 become missing and are then filled with a training-set group median.

### Cited Findings

- v3: `originalConstructionDate` is “the original date of the construction of the building.” `originalNBDate` is “the original date of the flood policy.” `dateOfLoss` is “the date on which water first entered the insured building.” `openDate` is “the date the claim was opened.” — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- TRRP, which the FAQ says still governs historical reporting practice: original construction date is the permit date or the construction date; for post-FIRM substantial improvement it is the substantial-improvement date; for pre-FIRM substantial improvement (rule as of 2015-04-01) it remains the original construction date. “If the construction date is a date prior to the year 1900, report this date as January 1, 1900.” Mobile-home dates can be the park-facilities date or the date of permanent placement. — [TRRP Change 23.1](https://nfipservices.floodsmart.gov/sites/default/files/media/document/2025-07/trrp_changes_23.1.pdf); [NFIP data FAQ on TRRP](https://agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)
- The program narrative dates the NFIP to the National Flood Insurance Act of 1968. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- Post-FIRM means start of construction or substantial improvement after 1974-12-31 or on or after the initial FIRM, whichever is later. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)

### Inferences

The hypothesis “55% are January 1 placeholders, set the date missing” overstates FEMA. A search of the TRRP text found January 1 required only for pre-1900 dates. No FEMA page found here says that January 1 of 1978 or 2005 means “day unknown.”

Procedure:

1. Parse with errors coerced to NaT.
2. If the construction date is exactly 1900-01-01, set it to missing and set `construction_pre_1900 = 1`. Do not compute an age of `year(loss) − 1900`. Those buildings are older than 1900 by an unknown amount. FEMA requires this code for that case.
3. If month and day are January 1 in any other year, keep the date. Set `construction_date_jan1 = 1`. Age uses that year. The flag tells the model the day may be a default. Do not drop these rows and do not median-impute their age. Doing so would throw away the year on about half the file (the hypothesis rate) for a day-level problem. Age error from a wrong day is at most about one year, which is small next to a building-age effect.
4. Set construction date to missing if it is after `dateOfLoss` or the year is before 1850 and it is not the 1900-01-01 code already handled. Before 1900 should already have been reported as 1900-01-01, so a surviving 1850–1899 date is an edit failure. The 1850 cutoff is judgment; FEMA’s written cutoff for the sentinel is 1900.
5. Age = (`dateOfLoss` − construction date) in years, on the dates that remain. Call it rating age in the write-up. TRRP says a post-FIRM substantial improvement stores the improvement date, and a manufactured home may store a park or placement date. It is not always the year the first stick was nailed.
6. Policy date `originalNBDate`: missing if after the loss or before 1968 (NFIP start; judgment anchored to the 1968 Act, not to a field edit in v3). Tenure in years from the dates that remain. No TRRP rule found here that January 1 policy dates are placeholders. If the notebook finds a January 1 spike, use the same pattern as construction: keep the year, add `policy_date_jan1`, do not blank the year.
7. Impute remaining missing age and tenure with the training median inside occupancy × state group, then global training median. Add `age_invalid` and `tenure_invalid`. Fit on training years only. This median covers the small invalid set, not the January 1 majority.
8. `openDate`: `report_lag_days = openDate − dateOfLoss`. Negative lag → missing (opened before the water entered: impossible under the definitions). If that count is tiny, fill the training median. Winsorize a long right tail for the model (the script uses 365 days) and show the raw tail in EDA. This lag is known early. It is not a payment field.
9. `mostRecentPaymentDate`: do not repair 1900-01-01 and do not build a lag. Out of the feature set (leakage section).

`postFIRMConstructionIndicator` is the compliance flag and does not depend on repairing the construction date. Keep it even when age is missing.

### Gaps

- No FEMA source found that says non-1900 January 1 construction dates are official placeholders. The 55% figure is the extract hypothesis.
- TRRP does not give an edit that rejects construction dates before 1850 specifically. The 1850 rule is local judgment.
- No v3 edit text was found that says `originalNBDate` before 1968 is invalid. 1968 is the statute year in the dataset narrative.

## What belongs only in descriptive EDA and must not be used as a model feature?

### Takeaway

EDA can show anything that explains the file. The feature matrix cannot include payment outcomes, recovery, closed-without-payment reasons, row and community ids, or adjuster measurements if the question is what was knowable at first notice. Elevation belongs in EDA after sentinels are removed, and FEMA says to use it for description and aggregation rather than as model input.

### Cited Findings

- Net payments subtract recoveries. Non-payment reasons are why a claim was closed without payment. Recovery fields are salvage, subrogation, and claim-payment recoveries. Payment date is the date a payment was issued. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- Damage amount, building and contents property value, and exterior and interior water marks are adjuster measurements. Replacement-cost basis is how the claim was settled. Cause-of-damage codes B, C, and D are expedited handling, follow-up inspection, and remote adjustment. Water depth is an inspection measurement and is sometimes in feet. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- FAQ: do not use the redacted file as cat-model input; use elevation “as a descriptive field for aggregation only”; coordinates are truncated to about six miles; county and census tract are the aggregation fields. City is unavailable. PII, full address, and precise coordinates are withheld. — [NFIP data FAQ](https://agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)
- Empty elevation-certificate indicator means not reported. `elevationDifference` 9999.0 means not reported or not used. — [NfipClaims v3 dictionary](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- Nofal et al. use depth, occupancy, basement, flood zone, and post-FIRM status in descriptive loss-ratio plots, and they remove zero and erroneous replacement values and some impossible negative depths before those plots. They note negative depth on buildings that have no basement. — [Nofal et al., FLOODrisk 2020](https://doi.org/10.3311/floodrisk2020.11.14)
- van Buuren and Harrell: a missingness indicator is not a harmless extra covariate when the goal is a coefficient. It is more defensible when the blank is a real state (not applicable) or when the only goal is prediction and future claims will be missing the same way. — [van Buuren §1.3.7](https://stefvanbuuren.name/fimd/missing-data-pattern.html); [Harrell ch. 3](https://hbiostat.org/rmsc/missing)

### Inferences

Put these in EDA and keep them out of `X`.

**Outcomes and anything computed from them.**

- Gross building, contents, and ICC payments, once they are the target or part of it. A model that predicts `paid_building` cannot also see `amountPaidOnBuildingClaim`.
- All net payment fields, all five recovery amounts, `mostRecentRecoveryDate`, `mostRecentPaymentDate`.
- `nonPaymentReasonBuilding` and `nonPaymentReasonContents`. Plot the code frequencies. They describe why dollars are zero. Using them to predict those dollars is circular. Do not impute them; missingness is the signal that the closed-without-payment reason does not apply, and that signal is the outcome.

**Identifiers and geography that identify or will not generalize.**

- `id`, community number, current community number, community name.
- Zip, county code, and census block-group id as dummies. Maps and joins are fine. FAQ prefers county and tract for aggregation, so a county choropleth belongs in EDA.
- Raw latitude and longitude as if they were parcel points. A coarse coordinate may stay in a predictive model after the state-and-county median fill, with the six-mile caveat in the text. Do not treat a 0.1° point as the building.

**Elevation, as numbers.**

- After removing 9999 and the suspected 99XX codes, plot elevation difference, BFE, lowest floor, and lowest adjacent grade, and the certificate-indicator codes. FEMA’s aggregation advice is the reason they stay out of `X`. The availability flags may enter a predictive model; they mean “a number was reported,” not “the building has a certificate” and not “the floor is high.”

**Process and inspection fields, when the task is early triage.**

The notebook should still plot them, with a written warning that they are known after inspection or settlement. Exclude them from the early-triage feature list.

- Building and contents damage, building and contents ACV, building and contents replacement cost, all three water depths, `causeOfDamage`, `replacementCostBasis`, `floodWaterDuration`.
- `replacementCostBasis` is the strongest leakage of this group: FEMA says it is the basis on which the claim was settled. The hypothesis that missing rows are paid far less often means the blank itself leaks closure. Unknown category in EDA. Not a mode-filled feature.
- Cause code 0 stays “other” in the plot. Codes B/C/D are the expedited path; show their payment rate in EDA so the leakage is visible.
- Water depth: histogram with a callout at 99 and at negative values. Keep negatives in the plot. Nofal/TRRP: negative depth can mean basement-only water measured from the first floor, but Nofal also found negatives on buildings without basements, so add a small table of negative `waterDepth` by basement type rather than deleting them all. Exterior and interior depth are relative to lowest adjacent grade in v3, so a negative there is not the same TRRP sentence. Do not zero them.
- Damage equal to 0 stays 0 in the plot (inspected, no dollar damage). Damage NaN is a separate bar.

**Near-constants, one frequency table, then stop.**

- Agricultural structure, house of worship, nonprofit, state-owned, floodproofed, ICC limit, disaster-assistance code. No model flag unless the student first shows the rare level has enough rows to estimate. Disaster-assistance code 0 must not be merged with blank.

**Data-quality exhibits, not features.**

- Crosstab of `preFirmIndicator` vs `postFIRMConstructionIndicator`.
- Rated zone vs current zone.
- Construction-date calendar: count of 1900-01-01 vs other January 1 vs other days.
- `waterDepth` unit caveat (some records in feet).
- Coverage above $250,000 / $100,000, as a note that those caps are not hard clips (Nofal).

**Safe in a first-notice feature list (no imputation beyond the rules above).**

- Policy and building, known from the application: occupancy (legacy 1-digit harmonized to the Risk Rating 2.0 two-digit codes), `ratedFloodZone` family, `postFIRMConstructionIndicator`, `elevatedBuildingIndicator`, foundation type with 0 as unknown, basement type with unknown, obstruction as not-applicable vs unknown, floors code, condo type, policy count, number of units with the structural-zero rule, building and contents limits with the no-coverage flags, deductibles, primary residence, rental, small business, rate method collapsed to a few groups plus an RR2 flag, CRS as unknown category, state (rare states grouped), named-event binary, rating age and policy tenure with the date rules, report lag.
- `yearOfLoss` and `dateOfLoss` are the split and the seasonality source (month), not a raw datetime dumped into the model. A linear year trend fit on 2020–2023 will not be a stable feature for 2024–2025; use year as the split key.

If the course model is allowed to see the closed file, including adjuster fields, put them in a second feature set and score it separately. Do not mix the two sets in one importance chart without the label. Missingness indicators on Tier B fields are themselves post-inspection information (“no adjuster figure on a closed claim”). For early triage, drop both the value and the indicator. For a post-inspection prediction task, Harrell’s prediction exception allows the indicator, as long as the write-up does not read it as a causal effect of depth or damage.

### Gaps

- FEMA does not publish an official “first notice of loss” field list for v3. The Tier A / Tier B split is judgment from the dictionary verbs (policy application vs adjuster vs settled).
- No source found that quantifies how often `causeOfDamage` stores only one code when the manual says more than one can be selected.
- The FAQ’s “no payment date” sentence and the v3 `mostRecentPaymentDate` field were not reconciled by FEMA in the pages read here.
