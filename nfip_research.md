# Research & Resources for a Data-Mining Project on NFIP Flood Insurance Data (2020–2025)

Compiled 2026-10-17. Every URL below was returned by web search/fetch during this session. Details marked "(abstract)" come from the abstract/landing page; I did not read every paper in full. Check each paper's page before citing.

---
## 0. Data sources: OpenFEMA docs and API (verified live today)

| Dataset | Docs / data dictionary | API endpoint |
|---|---|---|
| **NFIP Redacted Claims v3** (`NfipClaims`, ~2.7M claim transactions, monthly refresh) | https://www.fema.gov/openfema-data-page/nfip-redacted-claims-v3 | https://www.fema.gov/api/open/v3/NfipClaims |
| **NFIP Redacted Policies v3** (`NfipPolicies`, ~74.7M policy transactions, refreshed 2026-09-09) | https://www.fema.gov/openfema-data-page/nfip-redacted-policies-v3 | https://www.fema.gov/api/open/v3/NfipPolicies |
| Old v2 policies (`FimaNfipPolicies`) | https://www.fema.gov/openfema-data-page/fima-nfip-redacted-policies-v2 | **Deprecated. Its page says it is removed by 2026-10-15 and frozen at 2026-06-01. Use v3.** |

- Field-list metadata API (v3 claims): https://www.fema.gov/api/open/v1/OpenFemaDataSetFields?$filter=openFemaDataSet%20eq%20%27NfipClaims%27%20and%20datasetVersion%20eq%203
- Field-list metadata API (v3 policies): https://www.fema.gov/api/open/v1/OpenFemaDataSetFields?$filter=openFemaDataSet%20eq%20%27NfipPolicies%27%20and%20datasetVersion%20eq%203
- OpenFEMA API docs: https://www.fema.gov/about/openfema/api (the fetch timed out, but it is the standard docs page). Also see the FAQ https://www.fema.gov/about/openfema/faq and the terms/citation page https://www.fema.gov/about/openfema/terms-conditions
- FEMA's NFIP data FAQ (PDF): https://www.fema.gov/sites/default/files/documents/fema_nfip-data-faqs.pdf
- Both datasets are available as CSV/Parquet bulk files (the policies file is multi-GB). Prefer Parquet or the API with `$filter`/`$select` and pandas/DuckDB.

**Tested API calls**
```
https://www.fema.gov/api/open/v3/NfipClaims?$filter=yearOfLoss ge 2020 and yearOfLoss le 2025&$top=1000
https://www.fema.gov/api/open/v3/NfipPolicies?$filter=policyEffectiveDate ge '2020-01-01T00:00:00.000Z'&$top=1000
```
The API caps each call at 1000 rows, so page with `$skip`, or download the Parquet file.

**Key v3 claims columns:** dateOfLoss, yearOfLoss, state, countyCode, censusGeoid, reportedZipCode, latitude/longitude (rounded to 0.1°), ratedFloodZone, floodZoneCurrent, occupancyType, buildingDamageAmount, contentsDamageAmount, amountPaidOnBuildingClaim, amountPaidOnContentsClaim, netBuildingPaymentAmount, buildingPropertyValue, buildingReplacementCost, totalBuildingInsuranceCoverage, causeOfDamage, floodEvent, eventDesignationNumber, waterDepth / interiorWaterDepth / exteriorWaterDepth, floodWaterDuration, foundationType, elevatedBuildingIndicator, basementEnclosureCrawlspaceType, numberOfFloorsInTheInsuredBuilding, originalConstructionDate, postFIRMConstructionIndicator, primaryResidenceIndicator, buildingDeductibleCode, nonPaymentReasonBuilding, crsClassCode, nfipCommunityName, ficoNumber, rateMethod, totalSubrogationRecovery, openDate, mostRecentPaymentDate.

**Key v3 policies columns:** policyEffectiveDate, policyTerminationDate, cancellationDateOfFloodPolicy, cancellationVoidanceReasonCode, originalNBDate, rolloverTransferCode (N = new, R = renewal, T = transfer, O = RR2.0 opt-in), policyCost, totalInsurancePremiumOfThePolicy, **fullRiskPremium**, totalBuildingInsuranceCoverage, totalContentsInsuranceCoverage, buildingDeductibleCode, rateMethod (value `RatingEngine` = Risk Rating 2.0), subsidizedRateType, grandfatheringTypeCode, mandatoryPurchaseFlag, ratedFloodZone, floodZoneCurrent, occupancyType, primaryResidenceIndicator, rentalPropertyIndicator, buildingReplacementCost, foundationType, elevationDifference, crsClassCode, propertyState, reportedZipCode, censusGeoid, latitude/longitude.

**Gotchas to mention in the report**
- Claims have a reporting lag, so 2025 losses are likely incomplete.
- Records are *transactions*, not unique policies. Build a policy ID proxy, or dedupe on property keys plus dates.
- Coordinates are rounded to 0.1° (about 11 km), so use `censusGeoid` or ZIP for spatial work. The Virginia portal also warns that redaction can create artificial point clusters.
- Many fields have NaNs or sentinel values (e.g. 9999 for elevationDifference).
- Damage and payment are correlated with the deductible and coverage caps. Payments are capped at $250k for buildings and $100k for contents. Watch for leakage if you predict payment from damage.
- The Risk Rating 2.0 break (new business 2021-10-01, renewals 2022-04-01) falls in the middle of 2020–2025. This is a built-in natural experiment, and it is also a structural break for any model.

---
## 1. Claim amount / severity prediction with ML

| Title | Authors | Year | Finding | Method | URL |
|---|---|---|---|---|---|
| Learning inter-annual flood loss risk models from historical flood insurance claims | (authors not captured; see page) | 2023 (J. Environ. Manage.) | Predicts flood financial damage from NFIP claims 2000–2020. XGBoost best; R²=0.807 across coastal counties in 8 southern states. A Burr distribution plus bias correction improves match to the observed claim distribution. | CGAN, XGBoost, Gaussian-process regression | https://doi.org/10.1016/j.jenvman.2023.118862 |
| Predicting flood damage using the flood peak ratio and Giovanni Flooded Fraction | (Ghaedi et al. is cited in related work; check page for authors) | 2022 | Two-stage model: a classifier decides if a county-event has claims, then a regression predicts the number of residential NFIP claims. Variable importance shows flood-intensity features matter. State-level models beat the national model. | Statistical learning, two-stage classify then regress | https://pmc.ncbi.nlm.nih.gov/articles/PMC9348728/ |
| Stacked Ensemble Approach for Predicting Residential Flood Insurance Claims Across Indiana (AGU Fall Meeting abstract) | (see page) | 2020 | Mean building claims per subwatershed from about 65 predictors. Stacked ensemble reached adj. R² of only about 0.22 on validation, so this problem is hard. | GLM, RF, GBM, ANN, H2O stacked ensemble | https://studio.m-anage.com/agu/fm20/webprogram/Paper760856.html |
| FloodGenome: interpretable ML for property flood-risk predisposition | Chenyue Liu, Ali Mostafavi | 2024 (arXiv); 2025 (IOP, Env. Research: Infrastructure & Sustainability) | Uses NFIP claims as a flood-risk signal at block-group level. Clusters and classifies risk levels, with interpretable feature importance. | k-means clustering, random forest | https://arxiv.org/abs/2403.10625 / https://doi.org/10.1088/2634-4505/adb800 |
| FloodDamageCast: Building flood damage nowcasting with ML and data augmentation | Chia-Fu Liu, Lipai Huang, Kai Yin, Sam Brody, Ali Mostafavi | 2024 (Int. J. Disaster Risk Reduction) | Harris County, TX. Combines 41,606 NFIP claims with 66,366 FEMA IA records to cover uninsured damage. Nowcasts building-level damage. | ML with data augmentation | https://arxiv.org/abs/2405.14232 / https://doi.org/10.1016/j.ijdrr.2024.104971 |
| Uncovering Drivers of Atmospheric River Flood Damage Using Interpretable ML | Corinne Bowers, Katherine A. Serafin, Jack W. Baker | 2024 (Natural Hazards Review 25(3)) | Census-tract claim occurrence from storm and exposure features. Interpretable drivers identified. | Random forest classification, interpretability | https://doi.org/10.1061/nhrefo.nheng-1995 / https://www.jackwbaker.com/Publications/Bowers_et_al_(2024)_ML_Damage_Drivers,_NHR.pdf |
| Random forest damage prediction for California NFIP claims (AMS poster) | Boschee (poster) | 2024 | Binary and regression random forests at 1° resolution. Plans a classify-then-regress design. | RF | https://ams.confex.com/ams/104ANNUAL/webprogram/Handout/Paper440366/AMSPoster_Boschee.pdf |
| Developing a Spatial Regression Model Framework for Insured Flood Losses in Houston | Lily L. Kraft, Gabriele Villarini, Jeffrey Czajkowski, Dale Zimmerman, Renato S. Amorim | 2025 (ASCE OPEN 3(1)) | Spatial regression of tract-level insured flood losses, using Moran's I and spatial filtering. | Spatial regression, eigenvector spatial filtering | https://ascelibrary.org/doi/10.1061/AOMJAH.AOENG-0044 |
| Examining Flood Insurance Claims in the United States: Six Key Findings | Carolyn Kousky, Erwann Michel-Kerjan | 2017 (J. Risk & Insurance 84(3):819–850) | Analysis of more than 1M claims from 1978–2012. Covers flood-zone effects, property characteristics, mitigation, repetitive-loss properties and determinants of claim payment. A classic baseline for features. | Descriptive statistics and regression | https://doi.org/10.1111/jori.12106 (also https://www.rff.org/publications/journal-articles/examining-flood-insurance-claims-in-the-united-states-six-key-findings/) |
| hurricaneDamagePredictions (GitHub) | dylan-turner25 | n.d. | Cleans NFIP claims and trains XGBoost to predict household hurricane damage. Includes clustering of storm conditions to simulate damage distributions. | XGBoost, clustering, Jupyter notebooks | https://github.com/dylan-turner25/hurricaneDamagePredictions |
| fema-nfip-claims (GitHub) | mebauer | n.d. | Claims data repository (I only saw it in search results, not its contents). | n/a | https://github.com/mebauer/fema-nfip-claims |
| From point to probabilistic gradient boosting for claim frequency and severity prediction | (see page) | 2024 | **General actuarial** reference for frequency/severity GBM modelling. Not NFIP-specific, but useful as a method template. | Gradient boosting (frequency and severity) | https://arxiv.org/html/2412.14916 |

---
## 2. Policy take-up, lapse and retention

| Title | Authors | Year | Finding | Method | URL |
|---|---|---|---|---|---|
| Policy Tenure Under the U.S. NFIP | (authors not captured; see page) | Risk Analysis (DOI 10.1111/j.1539-6924.2011.01671.x; year not confirmed, about 2011/12) | First tenure analysis, over 2001–2009. Median tenure of new policies is 2–4 years. Small prior claims increase retention and large claims reduce it. | Survival/tenure analysis | https://onlinelibrary.wiley.com/doi/10.1111/j.1539-6924.2011.01671.x |
| Who Holds On to Their Flood Insurance? | Resources for the Future blog post (author and date not captured; it uses 2012 data) | n.d. | In 2012, about 13% of single-family policies were new that year and about 30% had lasted more than 6 years. | Descriptive | https://www.resources.org/common-resources/who-holds-on-to-their-flood-insurance/ |
| Voluntary purchases and adverse selection in the market for flood insurance | Jacob Bradt, Carolyn Kousky, Oliver Wing | 2021 (JEEM 110) | Voluntary take-up is about 4.1% outside the SFHA and 29.9% inside. Demand is more price-sensitive outside the SFHA. Suggestive adverse selection. | Panel econometrics with policy and claims data | https://doi.org/10.1016/j.jeem.2021.102515 |
| Effects of Risk-based Pricing Reform on Flood Insurance Uptake | Jesse Gourevitch, Max Snyder, Carolyn Kousky | 2025 (J. Catastrophe Risk & Resilience) | Risk Rating 2.0 caused an 11–39% drop in new policies and 5–13% in existing ones, by October 2024. Effects are larger in low-income ZIPs. | DiD with dynamic treatment effects on OpenFEMA policies (ZIP-month PIF) | https://journalofcrr.com/research/03-07-gourevitch-et-al/ |
| To Improve Is to Change? The Effects of Risk Rating 2.0 on Flood Insurance Demand | Francesc Ortega, Ivan Petkov | 2024 WP; 2025 (JEEM 134:103228) | Policy histories 2019–2023 show RR2 raised exits and cut entry. Renewers outside the flood zone saw average premiums rise about 12%, while flood-zone renewers saw them fall 2.7%. Deductibles rose. NFIP revenue fell 3.7%. | Individual insurance-history panel, DiD | https://docs.iza.org/dp17021.pdf / https://doi.org/10.2139/ssrn.4844528 / https://ideas.repec.org/a/eee/jeeman/v134y2025ics0095069625001123.html |
| Biggert-Waters / SRL reform renewal study | (AEA 2020 conference paper; authors not captured) | 2020 | About a quarter of severe-repetitive-loss owners stopped insuring after the 2012 reform. Renewal rate fell from about 80% to about 60%. | DiD on monthly renewals | https://www.aeaweb.org/conference/2020/preliminary/paper/75eBbzHn |
| Flood Insurance Coverage of FHA Single-Family Homes | HUD PD&R | n.d. | Defines policy-drop and policy-take-up rates by year, inside and outside the SFHA. Useful variable definitions. | Descriptive | https://www.huduser.gov/portal/sites/default/files/pdf/MDRT-Flood-Insurance-Coverage-of-FHA-SFH.pdf |
| Measuring flood underinsurance in the USA | Natee Amornsiripanitch, Siddhartha Biswas, John Orellana-Li, David Zink | 2025 (Nature Climate Change 15) | About $17.1B/yr (70%) of expected flood losses to single-family homes are uninsured. Underinsurance affects 88% of at-risk households. | First Street risk data merged with NFIP policies | https://www.nature.com/articles/s41558-025-02396-w |

---
## 3. Risk pricing and Risk Rating 2.0 effects

| Title | Authors | Year | Finding | Method | URL |
|---|---|---|---|---|---|
| Risk Rating without Information Provision | Philip Mulder, Carolyn Kousky | 2023 (AEA P&P 113:299–303) | More than two-thirds of policyholders face higher premiums under RR2.0. The largest increases hit those FEMA maps as low risk. The in/out-of-floodplain premium gap is narrowing. | Descriptive on NFIP data | https://www.aeaweb.org/articles?id=10.1257/pandp.20231102 |
| Ortega & Petkov (above) | | | | | https://docs.iza.org/dp17021.pdf |
| Gourevitch et al. (above) | | | | | https://journalofcrr.com/research/03-07-gourevitch-et-al/ |
| The Cap Trap: How Rate Ceilings Redirected the Repricing Burden Under NFIP RR2.0 | (AI-generated "APE" working paper series; **not peer-reviewed, treat with caution**) | 2026 | Policy-level data for 5 states, 2019–2024. Grandfathered policies saw lower relative premiums and lapse rates after Oct 2021, because of the 18% cap. | DiD on grandfathered vs non-grandfathered policies | https://ape.socialcatalystlab.org/papers/apep_1407/v1/paper.pdf |
| NFIP: FEMA's New Rate-Setting Methodology Improves Actuarial Soundness but Highlights Need for Broader Program Reform (GAO-23-105977) | U.S. GAO | 2023 | Recommends means-tested affordability assistance rather than premium caps. | Government audit | https://www.gao.gov/products/gao-23-105977 |
| The Cost of Climate: America's Growing Flood Risk | First Street Foundation | 2021 | 4.3M homes face substantial flood risk. NFIP rates would need to rise about 4.5× to cover the risk. Outside SFHAs, estimated annual loss is $2,484 vs a $478 average NFIP premium. | Property-level flood model, AAL vs premium | https://assets.firststreet.org/uploads/2021/02/The_Cost_of_Climate_FSF20210219-1.pdf / https://doi.org/10.5281/zenodo.4742591 |
| The Pricing of Flood Insurance | Kousky, Lingle, Shabman | 2017 (J. Extreme Events 4(2)) | Pre-RR2.0 baseline on how NFIP prices are set. Cited by Gourevitch et al. | Review | https://doi.org/10.1142/s2345737617500014 |

---
## 4. Spatial clustering, hotspots, and equity / disparity

| Title | Authors | Year | Finding | Method | URL |
|---|---|---|---|---|---|
| Flood Risk Mapping and the Distributional Impacts of Climate Information | Single author (Federal Reserve FEDS 2023-066; name not captured, check page) | 2023 | Post-Katrina map updates removed 1.4M properties from the floodplain and wrongly omitted 5.2M, mostly in neighborhoods with more Black and Hispanic residents. Take-up fell and racial disparities widened. | Staggered-timing event study | https://www.federalreserve.gov/econres/feds/files/2023066pap.pdf |
| Flood Insurance in Communities at Risk of Flooding | CBO | (year not captured) | Lower-income communities have less NFIP coverage and fewer discounts. Details by race and ethnicity. | Descriptive, community-level | https://www.cbo.gov/publication/60510 |
| Are Underserved Populations Left Out of National Flood Mitigation Efforts…? A Method to Assess Racial Inequality at the Census Tract Level | (see page) | (journal CRCJ, year not confirmed) | NFIP claims 1975–2019. Majority non-White tracts in majority-White counties have disproportionately fewer NFIP claims. | Tract-to-county disproportionality | https://doi.org/10.1162/crcj_a_00005 |
| Federal Disaster Policy Reforms… Should Center Racial and Economic Equity | Urban Institute | n.d. | New Orleans. Majority-White tracts get larger claim awards and lower nonpayment rates. Damage severity is not controlled for. | Descriptive | https://www.urban.org/urban-wire/federal-disaster-policy-reforms-including-flood-insurance-treatment-should-center-racial-and-economic-equity |
| Insurance Coverage and Flood Exposure in the Gulf of Mexico | (see page) | 2024 (Water 16(20):2968) | Tracts with more Black residents have lower coverage and greater exposure. | Tract-level modelling of coverage, exposure and social vulnerability | https://www.mdpi.com/2073-4441/16/20/2968 |
| Environmental justice implications of NFIP pricing reforms: Miami-Dade | Wesley Martin (undergraduate project, Temple) | 2025 | Wealthier ZIPs get larger average increases, but low-income ZIPs still carry a higher cost burden. | ZIP-level comparison with income data | https://scholarshare.temple.edu/handle/20.500.12613/11012 |
| Pluvial flood impacts and policyholder responses throughout the United States | Benjamin Nelson-Mercer, Taeho Kim, Vinh Ngoc Tran, Valeriy Ivanov | 2025 (npj Natural Hazards) | 216,994 non-SFHA NFIP claims (1978–2021). 87.1% were pluvial-driven. | Spatial and descriptive | https://www.nature.com/articles/s44304-025-00058-7 |
| Harnessing AI-driven space-time clustering (DBSCAN) poster (UNC) | (see page) | 2025 | Space-time clustering with DBSCAN; check whether it uses NFIP data. | DBSCAN | https://our.unc.edu/wp-content/uploads/sites/1148/gravity_forms/159-ad7f2de0736bb70ca3f153a194936e1c/2025/04/rhouck_DBSCANposter_final.pdf |

---
## 5. Repetitive loss and climate change

| Title | Authors | Year | Finding | Method | URL |
|---|---|---|---|---|---|
| One- to Four-Family Properties with Multiple Losses Insured by the NFIP | RAND | (year not confirmed) | Multiple-loss properties are a disproportionate share of claims cost. Characterises structure type, cause of loss and mitigation options. | Descriptive and policy analysis | https://doi.org/10.7249/rra2462-1 |
| Closing the Flood Insurance Gap (CRS) | Congressional Research Service | n.d. | About 40% of NFIP claims are outside SFHAs. | Report | https://www.congress.gov/crs-product/IN10890 |
| The Cost of Climate | First Street Foundation | 2021 | Annual expected losses rise 61% (from $20.0B to $32.2B) in 30 years. | Climate-adjusted flood model | https://doi.org/10.5281/zenodo.4742591 |

---
## 6. Text and other
- No NFIP-specific NLP paper turned up in OpenFEMA public data. The redacted claims data has no free text. Text would have to come from other sources (FEMA/GAO reports, news, policy documents).
- **Data documentation paper:** Dombrowski, Ratnadiwakara, Slawson (2021), "The FIMA NFIP's Redacted Policies and Redacted Claims Datasets," *J. Real Estate Literature* 28(2):190–212, with sample code and merges with ACS and Zillow data. https://doi.org/10.1080/09277544.2021.1876435
- **Data cleaning / anomaly detection:** "Refining FEMA's NFIP V2 Open Flood Insurance Claims Dataset: … Feature Engineering, Anomaly Detection and Multiple Imputation" (AGU 2023 abstract). Uses Isolation Forest and XGBoost outlier detection. https://ui.adsabs.harvard.edu/abs/2023AGUFM.U53B0527T/abstract
- **Practitioner example:** "Analyzing FEMA NFIP Claims Data for Floodplain Ordinance Update Discussions" (LinkedIn). Aggregates by year and tract, fills zeros and adjusts for inflation. https://www.linkedin.com/pulse/analyzing-fema-nfip-claims-data-floodplain-ordinance-l%C3%B3pez-hern%C3%A1ndez-8khde
- **Time-series association rules:** SAX-ARM (IPOL 2022), a method template for mining deviant events. https://www.ipol.im/pub/art/2022/437/article.pdf
- **Stochastic analysis:** "Preliminary stochastic analysis of the Redacted Claims data set by the USA NFIP" (EGU 2022). https://doi.org/10.5194/egusphere-egu22-2934
- **Kaggle:** "US Climate Disaster & Flood Insurance Risk" dataset, https://www.kaggle.com/datasets/darkmatternet/us-climate-disaster-flood-insurance-risk. I found no widely-known Kaggle NFIP notebook.
- State-level one-pager: https://www.psirc.psu.edu/assets/NFIP_Insights_for_PA.one-pager.2022-09-05.pdf

**Gap you can claim as novelty:** I found no result that combines 2020–2025 NFIP data with Risk Rating 2.0 *and* ML or data-mining methods (classification, clustering, association rules). Most RR2 work is econometric DiD on policy counts, and most ML work uses data up to about 2020. Check Google Scholar again before claiming novelty.

---
## 7. Suggested research questions (2020–2025)

1. **Classification: which policies lapse?** Target: `lapsed` = 1 if the policy is not renewed within about 60 days of `policyTerminationDate` (build the renewal link from property keys such as censusGeoid + ZIP + originalNBDate + construction date + occupancy; use `rolloverTransferCode` = R as the renewal marker). Features: policyCost, fullRiskPremium, premium/fullRiskPremium ratio, YoY premium change, rateMethod (RR2.0), grandfatheringTypeCode, mandatoryPurchaseFlag, ratedFloodZone, occupancyType, primaryResidenceIndicator, deductible code, coverage amounts, tenure (policyEffectiveDate − originalNBDate), state. Models: logistic regression, random forest, XGBoost. Report SHAP, AUC and PR-AUC with a time-based split. Link: Ortega & Petkov; Gourevitch et al.
2. **Classification: what drives non-payment or claims closed without payment?** Target: whether `nonPaymentReasonBuilding` is non-null, or amountPaid = 0. Features: damage amount, deductible, coverage, cause of damage, waterDepth, foundationType, flood zone, state, year. Link: the Urban Institute New Orleans finding.
3. **Regression: building claim severity.** Target: log(amountPaidOnBuildingClaim) or log(buildingDamageAmount), restricted to positive claims. Features: waterDepth, floodWaterDuration, occupancyType, buildingReplacementCost, elevated/basement/foundation, floodEvent, causeOfDamage, deductible, postFIRM, zone, building age. Compare linear regression, random forest and gradient boosting (also a Tweedie/gamma model). Use CV by event or year. Link: the Learning inter-annual flood loss paper (XGBoost, R² 0.81) and the Indiana stacked-ensemble result (R² 0.22).
4. **Two-stage model: (a) Will a county or tract have any claim in an event? (b) How much will it cost?** Aggregate by floodEvent × censusGeoid. Link: Bowers et al., PMC9348728.
5. **Clustering: claim archetypes.** Run k-means, GMM or HDBSCAN on claim features (damage ratio, depth, duration, replacement cost, coverage, construction year). Profile the clusters by flood zone, event and state. Link: FloodGenome. Check whether cluster composition shifts in 2021 (Ida) and 2022 (Ian) compared with 2020.
6. **Spatial hotspots: where do claims concentrate?** Compute claim counts per policy-in-force by tract, then run Getis-Ord Gi\* or LISA (Moran's I), or DBSCAN on ZIP centroids. Compare 2020–22 with 2023–25. Question: are hotspots outside the SFHA? Link: Kraft et al.; Nelson-Mercer et al.
7. **Equity: tract demographics vs coverage or payout.** Join `censusGeoid` to ACS income and race, then model claim payment, nonpayment rate or policy density. Use mixed-effects or tree models, and cluster tracts. Link: CBO; Fed FEDS; Urban Institute. Caveat: it is ecological inference, so control for damage depth.
8. **Association rules (Apriori / FP-Growth): which policy attributes co-occur with lapse or high-severity claims?** Discretise premium change, zone, deductible, foundation, occupancy, rateMethod, mandatoryPurchaseFlag. Report lift and confidence for rules whose consequent is `lapsed=1` or `severity=high`. Compare pre- vs post-RR2.0.
9. **Anomaly detection: unusual claims or policies.** Use Isolation Forest, LOF or an autoencoder on damage-to-value ratio, payment-to-damage ratio and damage-to-coverage ratio. Also flag payment above damage and implausible depths. Link: the AGU 2023 data-refinement abstract.
10. **Time series: monthly policies-in-force, new business and claims count or paid (2020–2025).** Forecast with Prophet, ARIMA or LSTM, test for a structural break at Oct 2021 / Apr 2022 (RR2.0), and decompose seasonality (hurricane season). Detect change points (e.g. `ruptures`). Link: Gourevitch et al. for the PIF series.
11. **(Bonus) Premium vs risk: how well does the premium track `fullRiskPremium`?** Regress or cluster the ratio of policyCost to fullRiskPremium by zone, state and income, to see who is still subsidised or capped. Link: Mulder & Kousky; First Street.

**Evaluation tips:** use time-based splits (train on 2020–2023, test on 2024–25) and event-grouped CV. Handle class imbalance. Beware leakage, such as damage predicting payment. Compare against simple baselines, and always report the share of the data you dropped.
