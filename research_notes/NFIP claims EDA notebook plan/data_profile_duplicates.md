# Descriptive statistics, presence checks, and duplicate-row definitions for NFIP Redacted Claims v3 (yearOfLoss 2020–2025)

Scope of these notes: what the EDA notebook should compute and state. Imputation methods and chart design are out of scope. Facts under **Cited Findings** were read from the linked page in this research pass. Rates, cardinalities, and keep/drop labels under **Local extract** come from the project column dictionary for the 2020–2025 extract (`nfip_claims/column_dictionary.csv`, 84 columns). That file is an inventory, not a FEMA citation. The notebook should recompute every local rate rather than paste it.

Full OpenFEMA file, as listed on the dataset page when retrieved: entity `NfipClaims`, API `https://www.fema.gov/api/open/v3/NfipClaims`, update frequency `R/P1M`, CSV and Parquet record count **2,725,989**. The local extract is the year-of-loss slice, not that full file. The dictionary records **277,033** distinct `id` values.

The FEMA PDF linked from the dataset page, `https://www.fema.gov/sites/default/files/documents/fema_nfip-data-faqs.pdf`, timed out on fetch. The same questions and answers were read on the live Floodsmart NFIP data FAQ and on a Norfolk open-data PDF titled “National Flood Insurance Program (NFIP) Data Frequently Asked Questions.” Where those two agree, both are cited. They are not a substitute for a line-by-line check of the FEMA PDF.

---

## What is the official row grain of OpenFEMA NfipClaims v3, and how should the notebook define duplicate rows?

### Takeaway

Each published row is one redacted NFIP **claims record** with its own `id`. FEMA calls these “claims transactions,” but the NFIP data FAQ says the claims file is not a payment-by-payment transaction ledger and that loss year, not disbursement year, is the supported way to total payments. The row is not a policy term, not one insured unit, and not a claim that can be joined to a policy. The notebook should count three different things and keep the labels distinct: exact duplicate rows, duplicate `id`s, and repeated geographic or date keys.

### Cited Findings

- The dataset “provides details on flood insurance claims transactions,” is “redacted to protect policy holder personally identifiable information,” “represents more than 2.7 million claims transactions,” is updated approximately monthly, and “will have a lag with the system of record.” It “is not intended to be an official federal report.” — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- The same page lists CSV and Parquet record count 2,725,989 and field `id` as “Unique ID assigned to the record.” — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `policyCount` is “Insured units in an active status.” “Residential Condominium Building Association Policy (RCBAP) contracts are stored as a single policy contract but insure multiple units and therefore represent multiple policies.” — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `dateOfLoss` is the “Date on which water first entered the insured building.” `yearOfLoss` is the “Year in which the flood loss occurred.” `openDate` is the “Date the claim was opened in the system.” `mostRecentPaymentDate` is “The most recent date a claim payment was issued to insured.” — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- The NFIP data FAQ says: “The current claims data doesn’t include payment date or claims transaction. Users can use the date of loss to understand the magnitude of losses for a given calendar year, but it isn’t possible to report disbursements by year from this dataset.” — [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data); same sentence in the [Norfolk copy of the NFIP data FAQ PDF](https://data.norfolk.gov/api/views/suf7-r643/files/ba36ea87-3e70-47b9-a3c0-1e78b85098b7?download=true&filename=NFIP_Data_Frequently_Asked_Questions_FAQs+%281%29.pdf)
- That FAQ statement conflicts with the v3 field list, which includes `mostRecentPaymentDate`. The dataset page and the FAQ were both live when read. Neither page explains the conflict. — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3); [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)
- Policy grain is defined separately, and only for the policy file: “The policy file is a transactional dataset which shows when a policy was in force and eventually terminated.” A point-in-time in-force count keeps rows whose effective date is on or before the date and whose termination date is after it. “Each subsequent renewal constitutes a separate record in this dataset.” — [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)
- “NFIP policy data released on OpenFEMA is designed to provide insight into NFIP operations and flood risk in aggregate. It’s not designed to track individual structures or policy holders over time. Our program doesn’t provide a record ID which allows users to associate a specific claim with a specific policy record at this time.” Withheld identifiers include policy number and repetitive-loss ID. — [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)
- `pandas.DataFrame.duplicated(subset=None, keep="first")` flags duplicate rows, optionally on a column subset. `keep="first"` or `"last"` flags only the extra copies. `keep=False` flags every row that participates in a duplicate group. — [pandas.DataFrame.duplicated](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.duplicated.html)
- ydata-profiling’s overview reports dataset-level duplicate rows, and its alerts include duplicates. Its own write-up treats “Duplicates: N duplicate rows” as a data-quality alert whose meaning depends on whether identical rows are possible in that domain. — [ydata-profiling overview](https://github.com/data-centric-ai-community/fg-data-profiling); [ydata worked example](https://ydata.ai/resources/advanced-data-visualisation-with-pandas-profiling.html)

### Local extract (inventory, not a citation)

- `id`: missing 0%, `n_unique` 277,033, role `id`, reason “unique row identifier (kept as key only).”
- `asOfDate`: missing 0%, `n_unique` 1, reason “constant (single snapshot timestamp 2026-09-08).”
- `yearOfLoss`: missing 0%, `n_unique` 6, reason “identical to year(dateOfLoss).”
- `dateOfLoss`: missing 0%, `n_unique` 2,176. `openDate`: missing 0%, `n_unique` 2,279.
- `policyCount`: missing 0%, `n_unique` 258.

### Inferences

- Notebook grain statement, in this order: one row is one redacted claims record; `id` is the record key FEMA describes as unique; a row is not a policy term; `policyCount` can make one row stand for more than one insured unit; the file is not documented as one row per payment installment. The word “transaction” on the dataset page should be printed next to the FAQ sentence that says the file has no payment-level claims transaction. The notebook should not resolve that conflict by dropping rows.
- Duplicate checks to code, and the names to print:
  1. **Exact-row duplicates.** `df.duplicated(keep=False).sum()` is the number of rows that sit in an all-column duplicate group. `df.duplicated().sum()` is the number of extra copies after keeping one. Report both. Include every column, including `id`.
  2. **Exact duplicates ignoring the surrogate id.** Same two counts on `df.drop(columns=["id"])`. If `asOfDate` is a single timestamp, say so and also run the check with `asOfDate` removed so a snapshot column cannot hide otherwise identical rows. These are the duplicates that would indicate a repeated extract, not a second claim.
  3. **Duplicate ids.** `df["id"].duplicated(keep=False).sum()` and `df["id"].is_unique`. The dictionary calls `id` unique and records 277,033 distinct values, but it does not store the row count, so uniqueness is still a check. Any duplicate `id` is a load failure relative to the published definition, not a second transaction to be modeled.
  4. **Repeated claim keys, counted and not dropped.** There is no public claim number. Do not define a duplicate as a repeated `(dateOfLoss, state, reportedZipCode)`, `(dateOfLoss, censusGeoid)`, or `(dateOfLoss, latitude, longitude)`. Rounding, redaction, RCBAP `policyCount` greater than 1, and distinct properties in the same block group all produce repeated keys. Report `nunique` of the key, the share of rows whose key occurs more than once, and the max multiplicity. Label the table “repeated keys under redaction,” not “duplicate claims.”
- A reasonable substantive-key diagnostic, still not a dedupe rule, is the multiplicity of `(dateOfLoss, state, reportedZipCode, censusGeoid, occupancyType, totalBuildingInsuranceCoverage, totalContentsInsuranceCoverage, originalConstructionDate)`. Print it as a cardinality check. Do not delete rows that share it.
- `pandas` duplicate checks treat the row as the unit. They do not implement NFIP claim identity. The notebook should say that.

### Gaps

- No OpenFEMA page retrieved here states whether two rows can be successive versions of one claim (original report and later adjustment) or whether `id` is stable across monthly refreshes.
- The FEMA PDF at the URL on the dataset page was not retrieved (fetch timeout), so a newer sentence that reconciles “claims transactions” with “does not include payment date or claims transaction” may exist there and was not read.
- Whether `duplicated` treats NaN as equal to NaN was not re-read in the pandas page captured here. The notebook should confirm that behavior on the installed pandas version before interpreting exact-duplicate counts on sparse columns.

---

## Which profile statistics should be computed for numeric, categorical, boolean, date, and dollar fields?

### Takeaway

Use `DataFrame.describe` as the base, then add the presence, cardinality, tail, shape, zero, negative, and sentinel rates that `describe` does not return. For dollar amounts, also report a five-number summary and the same statistics on the positive subset, because a point mass at zero and a heavy right tail make the mean a poor standalone summary. Coded fields with a published legal-value list get a full frequency table, not a top-5 list.

### Cited Findings

- `DataFrame.describe` excludes NaN. For numeric data the index is `count`, `mean`, `std`, `min`, the lower percentile, `50`, the upper percentile, and `max`. The default percentiles are 25, 50, and 75. The 50th percentile is the median. For object data the index is `count`, `unique`, `top`, and `freq`. `top` is the most common value and `freq` is that value’s count. If several values tie for most common, `top` is chosen arbitrarily among them. Default `describe` on a mixed frame returns numeric columns only; `include="all"` returns the union of type-specific rows. — [pandas.DataFrame.describe](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.describe.html)
- ydata-profiling numeric summaries add median, variance, median absolute deviation, coefficient of variation, interquartile range, range, skewness, kurtosis, sum, and the quantiles 5%, 25%, 50%, 75%, and 95%, plus counts and proportions of zeros, negatives, and infinities. Default numeric quantiles are `[0.05, 0.25, 0.5, 0.75, 0.95]`. The default skewness warning threshold is 20. A numeric series with fewer than 5 distinct values is, by default, also treated as categorical (`low_categorical_threshold`). — [ydata numeric statistics](https://deepwiki.com/ydataai/ydata-profiling/3.1-basic-statistics-and-descriptions); [ydata settings](https://docs.profiling.ydata.ai/4.5/advanced_settings/available_settings/)
- Categorical settings: cardinality warning default 50 distinct values; imbalance warning default 0.5; number of categories displayed by default is 5 (`vars.cat.n_obs`). Boolean settings: imbalance warning default 0.5; 3 values displayed by default. Non-hashable series are limited to `n`, `count`, `n_missing`, and `p_missing`. Dataset alerts called out in the tool’s own description include missing data, skewness, constant values, and duplicates. — [ydata settings](https://docs.profiling.ydata.ai/4.5/advanced_settings/available_settings/); [ydata-profiling features](https://github.com/data-centric-ai-community/fg-data-profiling); [ydata worked example](https://ydata.ai/resources/advanced-data-visualisation-with-pandas-profiling.html)
- Tukey’s five-number summary is the minimum, lower hinge, median, upper hinge, and maximum. The lower hinge is the median of the lower half and the upper hinge is the median of the upper half. Hinges are close to the first and third quartiles but are not the same calculation; when the number of observations is odd, the median is included in both halves. The summary is used because the center half and the extremes are readable when outliers are present. Sample quartiles from `describe` are not guaranteed to equal hinges. — [Tukey hinges and the five-number summary](https://sdrfoundation.org/tukey-hinges-guide); [pandas describe versus Tukey fivenum](https://github.com/ad-si/RosettaGit/blob/master/content/tasks/fivenum.md)
- Building and contents payments are decimal dollar amounts, and “a negative amount may appear” when a check is not cashed and has to be reissued. Several other money fields are defined as whole dollars (coverage, damage, property value, replacement cost). `amountPaidOnIncreasedCostOfComplianceClaim` “provides up to $30,000.” Net building payment “is the amountPaidOnBuildingClaim minus any recovery amounts.” Net contents payment is the contents amount paid minus recoveries. — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- The FAQ’s supported annual total is by date of loss, not by year of disbursement. — [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)

### Inferences

Statistics to compute for **every column**, before the type-specific block:

- `n_rows` (the filtered frame).
- `n_nonnull` (`count` in `describe`), `n_missing`, `pct_missing` = `n_missing / n_rows`. Count blank strings separately from nulls on text columns; the dictionary does not say whether OpenFEMA uses blanks, nulls, or both.
- `n_unique` including only non-null values, and `n_unique` with missing counted as its own level.

**Numeric codes and continuous numbers** (`policyCount`, `crsClassCode`, `elevationDifference`, `baseFloodElevation`, `lowestAdjacentGrade`, `lowestFloorElevation`, `numberOfFloorsInTheInsuredBuilding`, `obstructionType`, `occupancyType`, `floodWaterDuration`, `waterDepth`, `exteriorWaterDepth`, `interiorWaterDepth`, `numberOfUnits`, `latitude`, `longitude`, and any other numeric):

- From `describe`: count, mean, std, min, 25%, 50%, 75%, max.
- Add p01, p05, p95, p99.
- Add skewness and excess kurtosis (the usual pandas `skew` / `kurt` definitions). ydata’s skewness alert fires only above 20, which is a software default, not a claim that smaller skew is unimportant. Print the numbers; do not suppress them because they fail that threshold.
- Add `n_zeros`, `pct_zeros`, `n_negative`, `pct_negative`, `n_infinite`.
- Add a sentinel block: counts and percents of the candidate codes listed in the next section. Compute those rates on the raw values, before any replacement with missing.
- Add the five-number summary (min, lower hinge or at least p25, median, upper hinge or at least p75, max) beside the mean and std. Claim sizes and elevation fields are where the mean and the median will diverge; the notebook should show both.
- If `n_unique` is at or below the ydata low-categorical threshold of 5, also emit the categorical frequency table. Published code lists that are longer than 5 (occupancy, obstruction, CRS, floors, cause of damage) still get a full frequency table. The threshold is only a trigger to add the table, not a reason to skip it for longer code lists.

**Categorical / text codes** (`ratedFloodZone`, `floodZoneCurrent`, `rateMethod`, `causeOfDamage`, `condominiumCoverageTypeCode`, `buildingDeductibleCode`, `contentsDeductibleCode`, `eventDesignationNumber`, `floodEvent`, `replacementCostBasis`, `buildingDescriptionCode`, `state`, `reportedCity`, `reportedZipCode`, `countyCode`, `censusGeoid`, `foundationType`, `elevationCertificateIndicator`, `nonPaymentReasonBuilding`, `nonPaymentReasonContents`, community number and name fields):

- `describe(include="object")` supplies count, unique, top, and freq. Also report `pct_top` = freq / n_rows, and freq / n_nonnull.
- If `n_unique` is small enough that the published legal values fit on a page (zone, cause, deductible code, rate method, condo type, foundation, state, non-payment reason, replacement-cost basis, CRS, floors, occupancy), print every level with count and percent of rows, and print an explicit missing row. ydata’s default of 5 displayed categories is too small for these codebooks.
- If `n_unique` is large (ZIP, community number, community name, census block group, EDN), print `n_unique`, missing percent, top 15 with count and percent, the percent of rows outside the top 15, and the number of levels that occur once. ydata warns at 50 distinct values; that warning is the high-cardinality flag to show, not a collapse rule.
- For fields whose dictionary lists legal values, add a row for observed values that are not in that list, with their counts. `foundationType` is documented as 1 through 6 only. `causeOfDamage` says more than one cause code can be selected on a single claim, so the notebook should report whether stored values are single codes or concatenations rather than assume one code per row.
- Deductible fields are codes, not dollars. Profile the code. A dollar mapping is a derived column (`0` = $500, `1` = $1,000, …, `H` = $200). Do not run `describe` on the raw code as if it were a dollar amount.

**Boolean:**

- Value counts and percent for True, False, and missing. The documented `describe` contract retrieved here covers numeric and object columns and does not spell out a boolean index, so the notebook should not rely on `describe` alone for flags.
- Flag imbalance. ydata’s default boolean imbalance threshold is 0.5, which will flag almost every NFIP indicator. Also print a stricter near-constant flag when one class exceeds 95% and when it exceeds 99%, so the profile separates “imbalanced” from “effectively constant.” Those cutoffs are notebook labels, not FEMA rules.
- Boolean fields in this file, from the data dictionary: `agricultureStructureIndicator`, `elevatedBuildingIndicator`, `houseWorship`, `nonProfitIndicator`, `postFIRMConstructionIndicator`, `smallBusinessIndicatorBuilding`, `primaryResidenceIndicator`, `floodproofedIndicator`, `stateOwnedIndicator`, `rentalPropertyIndicator`, `preFirmIndicator`.

**Dates:**

- `n_nonnull`, `n_missing`, `pct_missing`, `n_unique`, min, max.
- Counts of sentinel or placeholder dates the next section names (`1900-01-01`, any year-01-01, dates before 1968, dates after `dateOfLoss`, dates after `asOfDate`).
- Agreement check: share of rows where `yearOfLoss` does not equal the calendar year of `dateOfLoss`. The local reason text says they are identical; the notebook should measure the mismatch count.
- `openDate` minus `dateOfLoss` in days: n, missing, n negative, and min, p50, p75, p95, p99, max. That is the reporting-lag profile. It is a table of numbers, not a chart specification.
- The `describe` page retrieved here does not document the datetime index. Compute min, max, and the percentiles explicitly.

**Dollar fields** (treat as numeric, and add these rows): `amountPaidOnBuildingClaim`, `amountPaidOnContentsClaim`, `amountPaidOnIncreasedCostOfComplianceClaim`, `netBuildingPaymentAmount`, `netContentsPaymentAmount`, `netIccPaymentAmount`, `totalBuildingInsuranceCoverage`, `totalContentsInsuranceCoverage`, `buildingDamageAmount`, `contentsDamageAmount`, `buildingPropertyValue`, `contentsPropertyValue`, `buildingReplacementCost`, `contentsReplacementCost`, `iccCoverage`, `totalSalvageRecovery`, `totalBldgClaimPmtRecovery`, `totalContentsClaimPmtRecovery`, `totalIccClaimPmtRecovery`, `totalSubrogationRecovery`.

- All of the numeric statistics above, including zeros, negatives, and percentiles through p99.
- Sum of non-null values, labeled nominal dollars, plus the count of nulls excluded from the sum. Do not fill nulls before summing.
- A second block restricted to values strictly greater than zero: n, mean, std, five-number summary, p95, p99, max, skewness. The unrestricted mean mixes the zero mass with the size of paid or damaged claims. The FAQ supports talking about magnitude of losses; it does not define this split. The split is the notebook’s way of keeping those two magnitudes readable. Label it as a descriptive split.
- Share of rows at the observed maximum, so a cap or a constant coverage amount is visible. The v3 text documents an ICC ceiling of $30,000. It does not, in the text retrieved here, state a building or contents coverage cap.
- Negatives are a documented payment event for the two gross building and contents payment fields (reissued checks). Still count negatives on every dollar field. Do not drop them in the profile.

**Presence table the notebook should print once**, one row per column: dtype, `n_missing`, `pct_missing`, `n_unique`, `pct_top` or percent True, `pct_zero` where numeric, sentinel count, and a blank column the author can later mark keep/drop. Sort by `pct_missing` descending, then by `n_unique` ascending, so empty, constant, and identifier columns surface first.

### Gaps

- The pandas documentation page retrieved here does not specify `describe` output for boolean or datetime dtypes, and pandas was not importable in the environment used to check. Those two types are specified above as explicit calculations.
- No ISO, AAIS, or CAS statistical-plan page was retrieved in this pass, so the zero-versus-positive dollar split is not attributed to a named actuarial standard. It follows from the documented possibility of zero and negative payments plus Tukey’s reason for reporting a resistant summary next to the mean.
- ydata’s skewness threshold of 20 and cardinality threshold of 50 are tool defaults, not NFIP data-quality standards.

---

## What do the OpenFEMA dictionary and the NFIP data FAQ say about redaction, rounded coordinates, sentinels, constants, and city?

### Takeaway

Privacy redaction removes direct identifiers and truncates latitude and longitude to one decimal place, about six miles, which can place the point in the wrong county or state and can pile many records on one printed point. The only sentinel the v3 claims dictionary defines by name is `elevationDifference` 9999.0, meaning not reported or not used. The FAQ adds that “99XX” is often a placeholder when a required value was not reported. `reportedCity` is documented as currently unavailable. `9990`, `-9980`, and `1900-01-01` do not appear in the v3 field text that was read.

### Cited Findings

- “Personal identifiable information (PII) is redacted. Data is also anonymized to the census tract, reported zip code and to one decimal point (.1) digit of latitude and longitude. If mapped, flood insurance policies and claims may appear to be clustered at a particular location due to this anonymization.” — [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data). The Norfolk FAQ uses the same anonymization sentence. — [Norfolk NFIP data FAQ PDF](https://data.norfolk.gov/api/views/suf7-r643/files/ba36ea87-3e70-47b9-a3c0-1e78b85098b7?download=true&filename=NFIP_Data_Frequently_Asked_Questions_FAQs+%281%29.pdf)
- Latitude and longitude “are truncated to one decimal point.” “The level of data that FEMA provides for users will find the coordinates accurate to within approximately six miles.” — [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)
- v3 field text: latitude and longitude are “Approximate … (to 1 decimal place).” “The precision has been lessened to ensure individual privacy. This may result in a point location that exists in an incorrect county or state. Use the state and county fields for record aggregation for these dimensions.” — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- “Census tract and county fields are best used for aggregation” because they are derived from the policy or claim geocode. — [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data). The coordinate fields themselves say to use state and county. Both instructions are on official pages. The notebook should name both and not treat them as one instruction.
- “FEMA doesn’t recommend using this data as input for commercial catastrophic models. The truncated coordinates don’t provide an accurate depiction of spatial risk.” Truncation “may shift the location of risks, which can affect how accurate the elevation data is.” “FEMA recommends using the provided elevation data as a descriptive field for aggregation only.” — [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)
- Withheld as PII: name, policy number, repetitive-loss ID, street address, ZIP+4, census block, and complete geocoordinates. Also withheld: WYO company, agency, agent, and adjusting firm. `censusGeoid` on v3 is titled “Census Block Group FIPS,” which is coarser than a census block. The FAQ’s “census block” item is the withheld finer geography, not a statement that block group is absent. — [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data); [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- “Where our system requires a value but isn’t always reported, ‘99XX’ is often used as placeholder or default value.” Legacy reporting rules are the TRRP manuals. — [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)
- `elevationDifference`: “The value of 9999.0 indicates the field is not reported and/or used for this policy.” — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3). The same 9999.0 sentence is on the v2 policies dictionary. — [FIMA NFIP Redacted Policies v2](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-policies-v2)
- Archived TRRP data-element text for lowest floor elevation: “Value 9999 indicates the field is not reported and/or used for this policy.” In the claims master file, whole feet are used. — [Archived April 2018 NFIP/WYO data-element bulletin W-05022](https://agents.floodsmart.gov/sites/default/files/bulletins/W-05022/w-05022.pdf)
- The same archived bulletin’s Mortgage Portfolio Protection Program defaults, applied by the system even if the company sends something else, include base flood elevation `+9999`, elevation difference `+999`, lowest floor elevation `+9999`, and original construction date `00000000`. — [W-05022](https://agents.floodsmart.gov/sites/default/files/bulletins/W-05022/w-05022.pdf). These are legacy policy-reporting defaults, not a v3 claims code list. The dataset page points at a 2015 consolidated TRRP manual that was not opened in this pass: `https://nfipservices.floodsmart.gov/manuals/jan_2015_consolidated_trrp.pdf`.
- `reportedCity`: “This is the city of the insured property as reported to us by our Write Your Own (WYO) partners. NOTE: the city value is currently unavailable.” — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `baseFloodElevation`, `lowestAdjacentGrade`, and `lowestFloorElevation` are defined as elevations or grade differences. The v3 sentences retrieved here do **not** define 9999, 9990, or -9980 for those three fields. — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `waterDepth` is “Depth of flood water in inches. Note: there are instances where measurements were provided in feet.” No code 99 or 999 is defined. Exterior and interior depth are also “in inches” relative to lowest adjacent grade, with no sentinel defined. — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `originalConstructionDate` is “The original date of the construction of the building.” No placeholder date is named. `mostRecentPaymentDate` has no placeholder named. — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `elevationCertificateIndicator`: “An empty value indicates that this information was not reported; it does not necessarily mean that these properties do not have an Elevation Certificate.” — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `ficoNumber` and `eventDesignationNumber`: “Prior to June 2020 flood events were assigned FICO number designations; after June 2020, flood events are assigned EDN number designations.” — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `countyCode` is a FIPS code “associated with the project.” “The County Code field may not reflect the individual county the property is located as projects can be associated with more than one county.” — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `preFirmIndicator` is titled pre-FIRM, but the definition text that follows says “A building for which construction or substantial improvement occurred after December 31, 1974, or on or after the effective date of an initial FIRM,” which is the post-FIRM rule. `postFIRMConstructionIndicator` uses that same post-FIRM rule and is titled consistently with it. — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- FEMA “omitted fields in the current OpenFEMA dataset for which there is low confidence” (Floodsmart wording) / “low fidelity” (Norfolk FAQ wording). Data are released “as it was reported and recorded.” Critical transaction errors are rejected; some minor or non-critical errors are allowed. More recent geography is more reliable than older geography. The FAQ does not name the omitted fields. — [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data); [Norfolk NFIP data FAQ PDF](https://data.norfolk.gov/api/views/suf7-r643/files/ba36ea87-3e70-47b9-a3c0-1e78b85098b7?download=true&filename=NFIP_Data_Frequently_Asked_Questions_FAQs+%281%29.pdf)
- `asOfDate` is “the effective date of the data in the file,” the date the record was created or last updated in the source system. — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)

### Local extract (inventory, not a citation)

Candidate sentinels the dictionary already flags, which the notebook should count on the raw column:

- `elevationDifference`: examples given are `-9980` and `987` (the reason text groups “outliers/sentinels”). Missing 65.84%. `n_unique` 107.
- `baseFloodElevation`: sentinel `9990`. Missing 66.04%. `n_unique` 4,535.
- `lowestAdjacentGrade` and `lowestFloorElevation`: sentinels `+/-9999`. Missing 64.49% and 64.21%. `n_unique` 6,068 and 6,054.
- `waterDepth`: value `99` flagged, with 12,509 rows named in the reason text. Missing 15.05%. Official text does not define 99, and it warns that some measurements were stored in feet, so 99 may be a code, 99 inches, or a feet/inches mix. Count it; do not label it an official sentinel.
- `exteriorWaterDepth`: `>=999` flagged. Missing 17.56%.
- `interiorWaterDepth`: `999` flagged. Missing 35.27%.
- `originalConstructionDate`: “55% are Jan-1 placeholders”; invalid dates are described as before 1850 or after the loss. Missing 0.71%. `n_unique` 17,411.
- `mostRecentPaymentDate`: “1900-01-01 placeholders.” Missing 7.85%.
- `reportedCity`: `n_unique` 1, value “Currently Unavailable,” missing 0%. This matches the dictionary NOTE.
- `foundationType`: reason text says code `0` is undocumented. Official legal values retrieved are 1 through 6. Missing 0%. `n_unique` 7.
- `rateMethod`: the local reason text says `'RE'` is Risk Rating 2.0 and is 90% of rows. The published legal value is `RatingEngine`, not `RE`. The notebook should frequency-count the raw stored value.

### Inferences

- Sentinel block to print, each as count and percent of rows, on raw values: `elevationDifference` in `{999, 9999, -9980}`; `baseFloodElevation` in `{9990, 9999, -9999}`; `lowestAdjacentGrade` and `lowestFloorElevation` in `{9999, -9999, 9990}`; `waterDepth` equal to 99; exterior and interior depth equal to 999 and greater than or equal to 999. Split the table into “defined by FEMA or TRRP” (`elevationDifference` 9999.0; legacy lowest-floor 9999; legacy MPPP `+9999` / `+999` / `00000000`; FAQ “99XX”) and “observed in this extract, not defined in the v3 sentences read here” (the rest). `987` on elevation difference should be listed as an extreme value, not as a sentinel, unless a code list is found.
- `1900-01-01` is not the MPPP default. That bulletin’s construction-date default is `00000000`. A parsed null and a parsed `1900-01-01` may be the same upstream code or two different codes. Count them separately. Also count January 1 of any year on `originalConstructionDate`, because the local inventory’s 55% figure is “Jan-1 placeholders,” which is broader than 1900.
- City: print the distinct values of `reportedCity`. Expect a single “Currently Unavailable” level. Do not geocode city. Do not treat that string as a real municipality.
- Coordinates: print missing percent, `n_unique`, and the share of rows sharing their rounded `(latitude, longitude)`. State the six-mile FAQ accuracy and the wrong-county-or-state warning beside those numbers. Aggregation fields to prefer, per the two official wordings, are county and census geography, with state used when the coordinate falls outside the recorded state.
- Constants FEMA actually documents, as opposed to constants that merely show up in this extract: city unavailable; `id` unique per record; `asOfDate` is a file effective date (one value per snapshot). A field that is constant only in the 2020–2025 slice is a measured fact about the slice, not a dictionary constant.
- `preFirmIndicator`’s definition contradicts its title. The presence notebook should crosstab it against `postFIRMConstructionIndicator` and print both dictionary sentences. The local reason text says both are False on 118k rows and both are True on 220 rows. Recompute that crosstab; do not describe `preFirm` as the logical opposite of `postFIRM`.

### Gaps

- `9990` and `-9980` were not found in the v3 claims field descriptions, the v2 policies `elevationDifference` sentence, or the searched portions of W-05022. They stay candidate codes until the 2015 TRRP manual or the FEMA FAQ PDF is read.
- The FEMA PDF URL on the dataset page was not fetched. The Floodsmart page and the Norfolk PDF agree on redaction, six-mile coordinates, 40–60 day refresh, and the payment-year limitation. Wording differs on “low confidence” versus “low fidelity,” and the Norfolk copy still describes API access as future tense, so it is an older snapshot of the FAQ.
- No official sentence was found that the stored city string is the literal `Currently Unavailable`. The dictionary says the city “is currently unavailable.” The literal string is a local measurement.

---

## Which fields are near-constant, 100% null, ID-like, or post-outcome leakage, and what published documentation supports showing them in EDA while leaving them out of a model?

### Takeaway

Show every one of the 84 columns in the presence table. Leave out of a feature matrix the columns that are empty or constant in this extract, the record identifiers and other quasi-identifiers, and the fields whose own definitions are settlement, payment, recovery, or a reason for non-payment. The published support for that split is FEMA’s field definitions plus the leakage rule that a feature has to be knowable at prediction time, not the project’s keep/drop file.

### Cited Findings

- Leakage is “the introduction of information about the target of a data mining problem that should not be legitimately available to mine from.” A legitimate feature’s values are observable earlier than the prediction time of the associated target. Using the target itself is the trivial case. — Kaufman, Rosset, Perlich, and Stitelman, 2012, [Leakage in Data Mining](https://doi.org/10.1145/2382577.2382579), as quoted from the [ACM PDF](https://dl.acm.org/doi/pdf/10.1145/2382577.2382579)
- ydata-profiling is built to surface constants, missingness, duplicates, and high cardinality in a profile. The worked example treats a constant column as something the profile reveals and that model development then drops, while saying the decision depends on the domain. — [ydata-profiling features](https://github.com/data-centric-ai-community/fg-data-profiling); [ydata worked example](https://ydata.ai/resources/advanced-data-visualisation-with-pandas-profiling.html)
- Payment, net payment, recovery, non-payment reason, settlement basis, and payment date are defined as outcomes of the claim, not as attributes of the building at inception:
  - `netBuildingPaymentAmount`: “Net building payment amount made to insured in dollars. It is the amountPaidOnBuildingClaim minus any recovery amounts.”
  - `netContentsPaymentAmount`: contents amount paid minus recoveries.
  - `nonPaymentReasonBuilding` / `nonPaymentReasonContents`: “Reason why a claim was not paid,” TRRP name “Claim Closed Without Payment Reason,” with legal values from `01` (less than deductible) through `99` (erroneous assignment).
  - `replacementCostBasis`: “whether the building claim was settled” on replacement cost (`R`) or actual cash value (`A`).
  - `mostRecentPaymentDate`: date a claim payment was issued.
  - `mostRecentRecoveryDate`: date an overpayment “was recovered from insured.”
  - `totalSalvageRecovery`, `totalBldgClaimPmtRecovery`, `totalContentsClaimPmtRecovery`, `totalIccClaimPmtRecovery`, `totalSubrogationRecovery`: amounts recovered after the loss was paid or determined.
  - `buildingPropertyValue`: actual cash value “as estimated by an adjuster.”
  - `buildingDamageAmount`: “actual cash value amount of damage.”
  - `amountPaidOnBuildingClaim`, `amountPaidOnContentsClaim`, `amountPaidOnIncreasedCostOfComplianceClaim`: dollar amounts paid.
  — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `id` is the record id. No claim-to-policy id is published. Policy number is withheld PII. — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3); [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)
- Community numbers are six-digit NFIP community designations, not claim attributes with a small codebook. ZIP is the reported five-digit ZIP. `censusGeoid` is a block-group id. — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- FICO was replaced by EDN after June 2020. — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- FEMA already dropped some low-confidence fields before publication and still warns that minor errors remain. That supports profiling what was published, including empty columns, rather than assuming every published column has usable variation. — [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)

### Local extract (inventory, not a citation)

Percents are the dictionary’s `missing_pct` and the reason text. `n_unique` is the dictionary’s. The notebook should recompute them.

**100% null in this extract**

| Field | missing_pct | n_unique | Dictionary reason |
|---|---:|---:|---|
| `ficoNumber` | 100 | 0 | 100% null; FICO replaced by EDN in 2020 |
| `floodCharacteristicsIndicator` | 100 | 0 | 100% null |

`floodCharacteristicsIndicator` has published codes 1–5 (velocity, low-velocity or ponding, wave action, mudflow, erosion). Emptiness is a property of this extract, not a statement in the field definition that the column is unused.

**Constant or near-constant in this extract**

| Field | missing_pct | n_unique | What the reason text says |
|---|---:|---:|---|
| `asOfDate` | 0 | 1 | single timestamp 2026-09-08 |
| `reportedCity` | 0 | 1 | constant `Currently Unavailable` |
| `iccCoverage` | 5.08 | 1 | 30000 on every non-missing row |
| `floodproofedIndicator` | 0 | 2 | reason says 100% False |
| `agricultureStructureIndicator` | 0 | 2 | 99.97% False |
| `stateOwnedIndicator` | 0 | 2 | 99.97% False |
| `houseWorship` | 0 | 2 | 99.89% False |
| `nonProfitIndicator` | 0 | 2 | 99.2% False |
| `smallBusinessIndicatorBuilding` | 0 | 2 | 97.3% False |
| `amountPaidOnIncreasedCostOfComplianceClaim` | 7.85 | 449 | 99.6% zeros |
| `netIccPaymentAmount` | 0 | 450 | 99.6% zeros |
| `disasterAssistanceCoverageRequired` | 66.52 | 5 | 99.4% of non-missing values are 0 |

`floodproofedIndicator` cannot be both strictly constant and `n_unique` 2. The reason text says 100% False while `n_unique` is 2. Print the value counts. A rounded “100%” can hide a handful of True rows.

`iccCoverage` is a coverage limit, not a payment. The payment field’s text says ICC “provides up to $30,000.” A constant 30000 in this extract is consistent with that ceiling and still has no variance to fit.

**ID-like or join-key cardinality in this extract**

| Field | missing_pct | n_unique | Role in the project file |
|---|---:|---:|---|
| `id` | 0 | 277,033 | id |
| `censusGeoid` | 0.91 | 39,628 | geo-key |
| `reportedZipCode` | 0 | 11,898 | geo-key |
| `nfipRatedCommunityNumber` | 0 | 7,884 | dropped as ID-like |
| `nfipCommunityNumberCurrent` | 20.17 | 7,293 | dropped as ID-like |
| `nfipCommunityName` | 0 | 6,775 | dropped as a duplicate of the community number |
| `countyCode` | 0.16 | 2,235 | geo-key |
| `floodEvent` | 20.39 | 68 | event name; project keeps a binary named-event flag and uses the name as a group key |
| `eventDesignationNumber` | 33.52 | 67 | dropped as ID-like and redundant with `floodEvent` |
| `latitude` | 0.66 | 298 | rounded coordinate |
| `longitude` | 0.66 | 619 | rounded coordinate |
| `state` | 0 | 56 | two-letter state, including whatever extra codes are present beyond 50 states |

**Post-outcome or adjustment-time fields in this extract** (profile them; do not put them on the feature side of a model whose target is payment or whose prediction time is first notice)

| Field | missing_pct | n_unique | Why the project file flags it |
|---|---:|---:|---|
| `amountPaidOnBuildingClaim` | 7.85 | 195,669 | target component; reason text says null means closed with no payment because net payment is 0 on every such row |
| `amountPaidOnContentsClaim` | 7.85 | 71,148 | target component; same null note |
| `amountPaidOnIncreasedCostOfComplianceClaim` | 7.85 | 449 | payment, 99.6% zeros |
| `netBuildingPaymentAmount` | 0 | 195,650 | net of recoveries |
| `netContentsPaymentAmount` | 0 | 71,140 | net of recoveries |
| `netIccPaymentAmount` | 0 | 450 | net ICC payment |
| `nonPaymentReasonBuilding` | 80.17 | 23 | closed-without-payment reason |
| `nonPaymentReasonContents` | 64.9 | 23 | same, and the reason text says it is still useful descriptively |
| `mostRecentPaymentDate` | 7.85 | 2,317 | payment date; null “exactly when no payment”; 1900-01-01 placeholders |
| `mostRecentRecoveryDate` | 99.85 | 274 | recovery date |
| `totalSalvageRecovery` | 99.85 | 33 | recovery |
| `totalBldgClaimPmtRecovery` | 99.85 | 219 | recovery |
| `totalContentsClaimPmtRecovery` | 99.85 | 53 | recovery |
| `totalIccClaimPmtRecovery` | 99.85 | 3 | recovery |
| `totalSubrogationRecovery` | 99.85 | 3 | recovery |
| `replacementCostBasis` | 0.59 | 2 | dictionary: basis on which the claim was settled |
| `buildingDamageAmount` | 11.04 | 114,064 | adjuster damage; project calls it usable only after inspection |
| `contentsDamageAmount` | 37.76 | 47,609 | adjuster damage |
| `buildingPropertyValue` | 11.04 | 169,043 | adjuster actual cash value |
| `contentsPropertyValue` | 37.76 | 36,824 | adjuster contents value |
| `buildingReplacementCost` | 11.04 | 175,285 | insurer replacement cost, collinear with building value in the project notes |
| `contentsReplacementCost` | 37.76 | 6,316 | dropped as redundant with contents value |
| `waterDepth`, `exteriorWaterDepth`, `interiorWaterDepth`, `causeOfDamage` | 15.05, 17.56, 35.27, 1.29 | 260, 438, 380, 12 | project labels these as recorded at adjustment |

The shared 7.85% missingness on the three gross payment fields and on `mostRecentPaymentDate` is a local pattern. The claim that those nulls are exactly the closed-without-payment rows is a local cross-check against the net-payment fields, not a FEMA definition of null. The notebook should print that cross-tab (`amountPaid` null versus `net` payment equal to 0, and versus `mostRecentPaymentDate` null) and label it a measured association.

### Inferences

- EDA shows all 84 fields. A later feature matrix drops a field only for one of these documented reasons, which the profile table should stamp in a column:
  1. **No variation in this extract** (`ficoNumber`, `floodCharacteristicsIndicator`, `asOfDate`, `reportedCity`, `iccCoverage`, and any boolean whose recomputed rate is effectively constant). A constant cannot separate outcomes. Kaufman-style leakage is a different reason; constants are a variance reason. ydata’s practice is to show the constant in the profile and then drop it from modeling.
  2. **Identifier or quasi-identifier.** `id` is the record key. Community number, community name, ZIP, county, and census block group are join or aggregation keys. FEMA withholds the true policy and address keys and says the release is for aggregate insight. High cardinality is the ydata alert to display. Leaving them out of a model is about non-generalizing identifiers and re-identification risk, not about them being “uninteresting” in the profile. `floodEvent` and `eventDesignationNumber` are event labels: show frequencies; if they are used later, the documented use is as a group, not as a 60-level dummy that memorizes particular catastrophes.
  3. **Not yet observable at the prediction time.** If the target is payment, the paid amounts are the target, and net payment, recoveries, non-payment reasons, and payment dates are functions of that outcome. If the prediction time is notice of loss, adjuster damage, depth, cause of damage, property value, and settlement basis are also later information. The profile should tag those columns `outcome` or `known after inspection` and still print their distributions. The tag depends on the prediction time. The notebook does not have to choose the prediction time in order to print the tag beside the definition.
- `nonPaymentReason*` is the clearest case of “show in EDA, exclude from features”: the legal values are reasons a claim was closed without payment, and the project reason text already says the contents reason is for descriptive analysis. The frequency table of those codes, with missing as its own level, belongs in the notebook.
- Do not import the project’s “drop if missingness exceeds 60%” rule into the notebook as if it were FEMA or ydata guidance. ydata reports missingness; it does not set a 60% drop rule. Those high-missing columns (`elevationCertificateIndicator` 87.68%, `obstructionType` 78.44%, `floodWaterDuration` 94.37%, elevation fields around 64–66%, `disasterAssistanceCoverageRequired` 66.52%) stay in the presence table. `elevationCertificateIndicator`’s empty value is defined as “not reported,” not as “no certificate.”
- `smallBusinessIndicatorBuilding` at 97.3% False is near-constant on the variance criterion and was still marked `maybe` in the project file. The profile should show 97.3% and leave the modeling argument for later. The EDA job is the rate.

### Gaps

- No FEMA page says which published columns are forbidden as model features. The modeling exclusion is an inference from the field definitions and from Kaufman et al., applied to a prediction time the dataset itself does not name.
- `floodCharacteristicsIndicator` is 100% null locally and fully coded in the dictionary. No sentence was found that says FEMA stopped populating it.
- The local “null payment means closed without payment” rule was not confirmed against FEMA. It has to be recomputed in the notebook.

---

## What reporting-lag and 2025-incompleteness issues should the notebook state when profiling yearOfLoss?

### Takeaway

`yearOfLoss` is the calendar year water first entered the building, not the year of payment. The published file is refreshed about monthly, lags the system of record, and is revised for several months after an error, so totals for a loss year change after the year ends. A profile of 2025, and any loss year close to the file’s `asOfDate`, is a partial count as of that snapshot. The local snapshot date recorded in the dictionary is 2026-09-08.

### Cited Findings

- The dataset “will be updated approximately monthly and will have a lag with the system of record.” Update frequency on the page is `R/P1M`. The page says the file is not an official federal report. — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- “Our program aims to ensure this data is refreshed every 40–60 days.” The Norfolk FAQ says FEMA will update the data every 40 to 60 days. — [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data); [Norfolk NFIP data FAQ PDF](https://data.norfolk.gov/api/views/suf7-r643/files/ba36ea87-3e70-47b9-a3c0-1e78b85098b7?download=true&filename=NFIP_Data_Frequently_Asked_Questions_FAQs+%281%29.pdf)
- “The NFIP legacy system operates on a monthly cycle and it takes several months to resolve flood insurance policies and claims errors. As a result, NFIP reporting is not static and totals will vary from one month to the next.” — [Norfolk NFIP data FAQ PDF](https://data.norfolk.gov/api/views/suf7-r643/files/ba36ea87-3e70-47b9-a3c0-1e78b85098b7?download=true&filename=NFIP_Data_Frequently_Asked_Questions_FAQs+%281%29.pdf). The live Floodsmart page says the system of record rejects critical errors but may allow minor errors, and that abnormal data can persist into the OpenFEMA release. It does not repeat the “several months” sentence. — [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)
- Claims history posted to OpenFEMA runs “from 1978 to present.” “All NFIP claims records are preserved.” That sentence is about retention, not about the newest loss year already being complete. — [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data)
- Annual magnitude is defined on date of loss. The FAQ says disbursements cannot be reported by calendar year from this dataset. v3 nevertheless publishes `mostRecentPaymentDate`. — [Floodsmart NFIP data FAQ](https://www.agents.floodsmart.gov/flood-maps-and-data/faqs-about-nfip-data); [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- `asOfDate` is the effective date of the data in the file. `openDate` is when the claim was opened. Negative payment amounts exist because checks are reissued, so a published payment is not a frozen final number. — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)
- There is no claim-status or “open/closed” field in the v3 field list that was read. Closed-without-payment is represented, when populated, by `nonPaymentReasonBuilding` and `nonPaymentReasonContents`. — [OpenFEMA NFIP Redacted Claims v3](https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3)

### Local extract (inventory, not a citation)

- `asOfDate` has one value, 2026-09-08.
- `yearOfLoss` has 6 distinct values and 0% missing. The project reason text says it matches the year of `dateOfLoss`.
- `dateOfLoss` has 2,176 distinct values and 0% missing. `openDate` has 2,279 distinct values and 0% missing, so a report-lag day count can be computed on every row of this extract.
- `mostRecentPaymentDate` is 7.85% missing, the same missing rate as the three gross payment amounts.

### Inferences

- Print `asOfDate` (min and max, and `n_unique`) in the header of the year-of-loss section. Every sentence about 2025 completeness is relative to that date. For this extract the dictionary says that date is 2026-09-08, which is after the end of calendar 2025, so 2025 is not an open calendar year at the snapshot. It can still be incomplete as a claims year.
- Reasons a loss year can still be short, or its dollars still low, after the calendar year has ended, all of which are supported by the citations above or by fields that exist to record later activity:
  1. The extract lags the system of record and is refreshed on a monthly / 40–60 day cycle.
  2. Error correction continues for months, and published totals change from release to release.
  3. A claim can be opened after the loss date (`openDate` versus `dateOfLoss`). Claims opened after the snapshot are absent. The lag distribution by `yearOfLoss` shows whether recent years have a shorter right tail, which is what a truncated reporting window looks like.
  4. Payment amounts and `mostRecentPaymentDate` can still change after the snapshot. A negative amount is a documented reissue, so even a non-null payment is not final.
  5. There is no open-claim flag. A 2025 row with null payments may be closed without payment, not yet paid, or not yet adjusted. The notebook should not interpret a lower 2025 mean payment as a lower severity.
- Table to compute, one row per `yearOfLoss` from 2020 through 2025: row count; distinct `dateOfLoss`; distinct `floodEvent`; sum of non-null `amountPaidOnBuildingClaim` and `amountPaidOnContentsClaim`; percent of rows with those payments null; percent with `mostRecentPaymentDate` null; median and p95 of `openDate - dateOfLoss` in days; number of negative lags; max `dateOfLoss`; max `openDate`; max `mostRecentPaymentDate`. Put `asOfDate` in the table caption.
- State in prose next to that table: 2025 counts and sums are as of the snapshot, subject to late notices, later payments, and later corrections. Do not rank 2025 against 2020–2023 as a finished year. 2024 can still move for the same reasons, especially for losses late in 2024. The FAQ’s “several months” line is the Norfolk wording; the live Floodsmart page is softer, so attribute the “several months” claim to that FAQ copy.
- Also state the FAQ rule the table can violate if someone sums by payment date: the endorsed annual total is loss year. If the notebook also counts rows by year of `mostRecentPaymentDate`, label that column as a payment-date profile that the FAQ does not treat as an annual disbursement total, and keep the 7.85% null payment dates out of every year rather than imputing them.
- `yearOfLoss` equal to 6 levels is the filter working, not a property of the national file. The national file’s loss years run from 1978 to present. A reader who compares these year counts with published all-years NFIP totals is comparing a slice plus a lagged snapshot with a different release.
- Calendar coverage check: 2020–2025 contain 2,192 calendar dates (365×6 plus leap days 2020 and 2024). The extract has 2,176 distinct `dateOfLoss` values. The notebook should list the calendar dates in range with zero claims, as a presence check, not as evidence that no flood occurred.

### Gaps

- No FEMA page retrieved here gives a numeric development factor, a recommended cutoff such as “ignore the latest N months,” or a statement that 2025 is incomplete by a stated percent.
- The live Floodsmart FAQ and the older Norfolk FAQ disagree in strength on how long corrections take. The dataset page says only that there is a lag. The notebook should quote the lag and the non-static totals, and should not invent a maturity window.
- Whether `id` values and payment amounts for 2020–2024 have already stopped changing between monthly files was not measured. Only one snapshot is described in the local dictionary.
