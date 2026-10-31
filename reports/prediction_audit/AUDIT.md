# NFIP notebook prediction audit — 21 October 2026

**The project has useful predictive signal, but several conclusions overstate what was tested.** Exact payment prediction across future storms remains difficult. Severity ranking is more successful, and an independently prepared classifier improves the saved benchmark. The numerical outputs are mostly internally consistent; the largest problems are evaluation design, inconsistent targets, and interpreting retrospective associations as deployable confidence or flood risk.

This audit inspected all **23 existing notebooks**: **14 NFIP analyses** and **9 lecture workbooks**. The original notebooks were preserved. `notebook_inventory.json` records their hashes, execution status and saved errors; `saved_notebook_evidence.json` preserves all 600 NFIP cells and their text/HTML outputs. Cell numbers below are **zero-based notebook cell indices**. Tables styled with pandas contain real HTML results even when their plain-text representation only says `Styler`.

## What was verified by running models

### Payment regression: same cohort and original dollar target

Raw data were aligned by unique `id` to the 276,570-row simple-notebook cohort. There are 119,746 paid training claims through 2023 and 93,186 paid test claims in 2024–2025. All errors below score original, uncapped dollars on those same test claims.

| Model | Dollar R² | Mean absolute error | Median absolute error |
|---|---:|---:|---:|
| Training median | −0.175 | $72,976 | $32,291 |
| Training mean | −0.024 | $68,729 | $52,833 |
| Original notebook 04 Ridge, log target | **0.183** | $58,873 | $31,872 |
| Original notebook 04 Poisson, 15 features | −0.059 | $57,036 | $30,378 |
| New Poisson, selected by forward validation | 0.169 | **$54,186** | **$29,814** |

The original Ridge and Poisson results were reproduced, rather than inferred from prose. The revised model reduces MAE by **5.0% versus the original Poisson** and **8.0% versus Ridge**. Its R² improves over the original Poisson but **does not beat Ridge**. There is no single winner across objectives. Keep Ridge as the existing dollar-R² benchmark; the new model is a better MAE candidate on this retrospective evaluation.

The new model was selected using two forward folds: 2020–2021 → 2022 and 2020–2022 → 2023. Its mean validation R² was 0.160. It was then fit through 2023 and evaluated on 2024–2025. New preprocessing learns numeric medians and category vocabularies inside each fit. It uses a prespecified broad policy/building feature set, explicit unknown categories and missingness indicators; it excludes target histories, event identity, current flood zone and rating method. No random early-stopping holdout is used. More regularization, changed feature availability and preprocessing were changed together, so their individual causal contributions have **not** been isolated.

The new national model's test R² is 0.141 in Florida, 0.103 in the Gulf states, and 0.273 in the Northeast; it is 0.163 for 2024 and 0.144 for 2025. These are scores of the national model on subsets, **not independently tuned regional models**.

Additional experiments compared direct squared loss, log-target squared loss, coordinates, and payment/coverage targets. The naive coverage-ratio models fail badly: zero/low coverage and anomalous payment-to-coverage ratios make the target unstable. A bounded ratio sensitivity check excludes zero-coverage training cases and caps the training ratio at one, while still scoring uncapped dollars. It also fails forward validation (mean R² about −0.96 for squared loss and −0.81 for Poisson), so it was **not promoted to a new future-year test**. Normalizing by coverage is not a verified improvement here.

Artifacts: `original_regression_reproduction.csv`, `regression_forward_validation.csv`, `regression_selected_test.csv`, `regression_bounded_ratio_validation.csv`.

### Severity classification: corrected target and explicit threshold policy

The 80th percentile of observed training-year damage is **$84,724.20**. The simple pipeline uses the **all-year** percentile of **$106,127**. Correcting this changes **16,780 labels** among claims with observed damage. On the corrected definition the severe share is 20% in training, 36.2% in 2024–2025; separately, 38.6% in 2024 and 17.8% in 2025. Missing damage remains unlabeled.

The new classifier uses raw policy/building features, preprocessing fit through 2022, and 2023 for model selection, sigmoid calibration and threshold choice. Its threshold and calibration are frozen before evaluating 103,511 labeled 2024–2025 claims. The severity definition is fixed from the broader training period through 2023. Calibration and threshold optimization share 2023, so calibration-fit metrics are not independent estimates; only later-year metrics are reported as test performance.

| Probability policy | Precision | Recall | F1 | Brier loss |
|---|---:|---:|---:|---:|
| Unweighted boosting, raw probability ≥ 0.50 | 0.764 | 0.555 | 0.643 | 0.159 |
| Same model, calibrated probability ≥ 0.50 | 0.744 | 0.614 | 0.673 | **0.155** |
| Same calibrated model, validation-selected cutoff **0.23** | 0.598 | **0.831** | **0.695** | 0.155 |

**Transfer is still unstable:** at the frozen 0.23 cutoff, F1 is **0.711 in 2024 but only 0.325 in 2025**; 2025 ROC-AUC is 0.712 and average precision 0.344. The combined result is dominated by 2024 and must not be presented as stable performance across future years.

All three policies have ROC-AUC **0.840** and average precision **0.735**. A monotone calibration changes probability quality and decisions, not ranking. The lower cutoff increases recall at a real precision cost: approximately **40% of flagged cases are not severe**, and half of claims are flagged. This is useful triage, not confident case-by-case identification.

The closest saved benchmark with the **same training-derived damage cutoff and same 103,511 test labels** is notebook 02's all-tier-A gradient boosting: average precision **0.670**, ROC-AUC **0.834**, versus 0.735 and 0.840 here. Notebook 02 uses 10 extra rows in the overall cleaned cohort and more training years; its features/preprocessing differ. The ranking gain is verified as a retrospective comparison, **not a controlled one-change ablation or prospective guarantee**. Do not compare the new F1 directly against notebook 04's F1 ≈ 0.60 as if the labels were unchanged: notebook 04 uses the higher all-year cutoff.

**Confidence does not come free by raising the cutoff.** Fixed diagnostic cutoffs on calibrated test probabilities give:

| Positive probability cutoff | Positive calls | Observed precision | Severe claims recovered |
|---|---:|---:|---:|
| 0.70 | 15,313 | 81.5% | 33.3% |
| 0.80 | 4,221 | 85.7% | 9.7% |
| 0.90 | 10 | 100% | 0.027% |

The 0.90 result is only ten cases and is not evidence of guaranteed confidence. No cutoff satisfied the prespecified **80% precision with at least 100 positive calls on 2023 validation**, so the implementation does **not** select an operational “80% precision” policy. The high-precision test subsets above are descriptive diagnostics and must not be selected after seeing test results. Abstention accuracy can also be dominated by easy negatives; report severe-case recall and retained share alongside it.

Artifacts: `classification_validation.csv`, `classification_test.csv`, `classification_by_year.csv`, `classification_reliability.csv`, `classification_fixed_confidence.csv`.

### History features and clustering: additional reproductions

The original notebook 06 random-split model reproduces dollar R² **0.6489**. Replacing its self-inclusive training history features with out-of-fold training histories gives **0.6385**; removing target histories gives **0.6049**. Thus, the entire 0.65 score is **not explained by self-inclusion**. It remains a retrospective same-event score: claims later in an event can inform earlier ones, and availability of finalized payments is not checked. All three runs retain the original cleaned data, global category filtering and county frequency, deliberately isolating the history treatment. They are not clean prospective pipelines.

Reproduced clustering on the exact 20,000-row sample and representation:

| Inputs | k | Silhouette |
|---|---:|---:|
| Building profiles | 2 | 0.462 |
| Building profiles | 7 | 0.251 |
| Coordinates | 4 | 0.682 |
| Zone + foundation codes | 8 | 0.725 |

These match the saved notebooks. Better geometric separation of locations or repeated codes is not evidence of new predictive risk classes. The k=2 building split puts **97.5%** in one group and mostly isolates mobile homes. Damage and payment should remain profile outcomes, not be used as early-prediction inputs just to obtain stronger scores.

Artifacts: `history_encoding_diagnostic.csv`, `clustering_reproduction.csv`.

## Problems established by code and saved outputs

| Priority | Verified issue and evidence | Consequence / correction |
|---|---|---|
| High | Notebook 03 fits fills globally (cells 49, 55 and 63), defines severity with all-year `quantile(.80)` (cell 63), and samples/ranks features across all years (cell 75 onward). Notebook 04 and later consumers reuse the exported table/lists. | Future-year features **and labels used in selection** contaminate claims of untouched evaluation. Rebuild raw-data preprocessing and selection within each training split. Calling this leak “small” has no measured justification. |
| High | The three standalone regional notebooks learn fills, severity cutoffs and feature rankings from a random training subset containing 2024–2025. Their “new storms” checks later reuse those features and targets (cells 20, 25, 30–33, 39). | The random-split evaluation is a distinct retrospective task. Its subsequent future-year check is not a clean forward evaluation, even though the final model fit uses older rows. Rebuild the entire pipeline separately for each evaluation setting. |
| High | Notebooks 05 RQ1 and 06 call `history(key, tr_idx, tr_idx)` for training features (cell 5). Training targets contribute to their own predictors. “Earlier claims” is not implemented: no date or payment-availability ordering exists. County frequency and rare-level retention use the full paid cohort. | Use out-of-fold encoding for retrospective same-event tests; use strictly prior finalized claims for live event triage. Fit frequency encoding and category retention on each training fold. The diagnostic above quantifies self-inclusion without claiming the other issues are fixed. |
| High | Notebook 04 says the best national model explains only 18%; notebook 05 says no combination exceeds about 8%; notebook 06 says no national model predicts new storms. These refer to different limited candidate sets. | The saved numbers are consistent, but the broad impossibility claim is contradicted by the existing Ridge result. State the model, target, split and metric explicitly. “Limited transfer with the tested features” is justified; “cannot predict” is not. |
| High | Notebook 04 describes clusters as “real flood-risk profiles” (cell 42) and a mitigation conclusion (cell 51), while foundation alone has V=0.26 versus clustering V=0.24. Notebook 05 RQ3 correctly challenges that interpretation. | Use “descriptive claimant/building groups.” Claim-only data lack the insured-building exposure denominator needed for flood-occurrence risk. Severity associations are not causal estimates of elevation's benefit. |
| Medium | Random KFold / StratifiedKFold are used for model and tree-depth comparisons inside historical training periods. Notebooks 05/06 also use GroupKFold by loss year. | Random folds mix storms. GroupKFold removes years but may train on later years to predict earlier years; it is not a forward simulation. Year groups are only proxies for independent events. Use rolling forward validation plus event holdouts for a separate transfer stress test. |
| Medium | Feature selection is performed before the CV comparison in notebook 03 and the standalone regional notebooks. Notebook 02 has a more careful fit/validation design, but inherits through-2023 fills, caps and threshold from notebook 01 before internal CV/validation. | Fold scores do not include independent re-estimation of the whole preprocessing/selection process. Put the transforms and feature selection inside each fold; distinguish final test validity from internal validation optimism. |
| Medium | Notebook 04 and regional classification use balanced weights with default `.predict()` thresholds. “Best” models are identified using test F1. Original supervised notebooks contain no calibrated confidence policy, prediction intervals or coverage guarantees. | Weighted probabilities are not automatically calibrated posterior probabilities. Choose models and operating thresholds on validation only; assess calibration, AP, precision, recall and workload separately. F1 or R² is not confidence. |
| Medium | RQ1 uses paid claims only. Blank payments with zero net payment are structurally set to zero in the simple pipeline (cell 49). | The regression answers **payment size conditional on positive payment**, not expected payment for every incoming claim. Here 21,558 raw building payments are missing and none has positive net building payment; that supports the observed zero-association but does not establish that all are finalized denials. Preserve unknown status for an all-claims model. |
| Medium | R² log and R² dollars coexist. Notebook 02's 0.329 is log-payment R²; notebook 06's 0.649 is dollar R². Median baselines are used even when judging squared-error R². | These scores cannot be compared directly. Show train-mean and train-median baselines, dollar MAE/RMSE, and label log metrics. The zero point of R² is the **test** mean, not a deployable model that knows that mean. |
| Medium | Notebook 08 excludes ratios >1 as “impossible,” discards negative depths, requires complete observed outcome/cost features and forces k≥3. | These are cohort/modeling assumptions, not established truth. Audit's cached dictionary says waterDepth is in inches **with some values reported in feet**. Investigate units, record vintage and valuation definitions before treating spikes or exclusions as physical archetypes. Negative depths and ratios >1 need documented semantics, not automatic physical impossibility claims. |
| Medium | Notebook 09 mines and evaluates rules on all labeled years. Rule confidence is empirical severity frequency within a selected antecedent; no out-of-period validation is performed. The declared 1–3 antecedent rule limit is not applied in `severity_rules` (cell 9); four-item rules appear. | Rules are descriptive discoveries, not validated predictive confidence. Mine on training years, freeze antecedents and bins, score support/confidence/lift on later years, and account for event dependence and rule multiplicity. Apply the advertised length filter if retaining that requirement. |
| Medium | Notebook 09 itself notes rating method is a later snapshot attribute. Earlier notebooks list rating indicators as “pre-loss.” | Availability is not established just because a field is about a policy. A policy snapshot may reflect updates after loss. Validate point-in-time availability; the new model omits rating method and current flood zone. Other building/policy fields remain subject to that limitation. |
| Low | Repeatedly evaluated 2024–2025 claims are called “used once” across notebook development. Notebook 06 attributes the 0.571 → 0.618 median-fill difference to fold noise although it uses the same folds and changed preprocessing. | These are reused retrospective tests; no untouched prospective holdout remains. Treat the fill result as an unresolved paired preprocessing effect; it is not ordinary fold resampling noise. |

Global preprocessing is acceptable for a clearly labeled descriptive EDA or full-sample clustering. The issue is reusing those transforms to claim independent supervised evaluation.

## Notebook-by-notebook disposition

| Notebook | Audit finding |
|---|---|
| `notebooks/01_EDA_NFIP_claims.ipynb` | More careful train-only outer preprocessing and a train-only severity cutoff than notebook 03. Missingness diagnostics are valuable. Observational associations with paid status do not identify MAR versus MNAR, and masked observed cells do not reproduce true missingness. Through-2023 preprocessing must still be refit inside earlier validation folds. |
| `notebooks/02_feature_selection.ipynb` | Saved summary matches output CSVs: log R² 0.326 → 0.329 with selection; AP 0.670 → 0.666. The claim that selection simplifies rather than boosts accuracy is supported. The leakage demonstration is valid for notice-time availability, but log R² must stay labeled. Internal preprocessing/selection is not completely nested. |
| `notebooks/03_EDA_and_feature_selection_simple.ipynb` | Main source of global fills, all-year severity cutoff, and future-label feature selection. Its EDA can remain descriptive; its exports should not be the source of independent forecast claims. |
| `notebooks/04_research_questions.ipynb` | Original national regression values reproduced; classification prose matches the saved HTML table. Leakage caveats are present but minimized without evidence. RQ3's affirmative risk-profile conclusion is stronger than its own comparative evidence. |
| `notebooks/05_rq1_feature_search.ipynb` | Saved final table matches prose: random all-groups R² 0.649, future all-groups −0.370, future forward-selected 0.083. Search uses training partitions, but the inherited 15-feature comparator is already contaminated and history is not truly prior history. “No combination predicts” overgeneralizes. |
| `notebooks/05_rq3_check.ipynb` | Largely sound, appropriately critical interpretation: foundation/zone/region patterns do not establish distinct universal risk types. Reproduced its key geometric scores. Published-study numerical claims were not independently re-audited. |
| `notebooks/06_rq1_national_and_regional.ipynb` | Saved regional/CV tables match the cited 0.57 national and 0.67 Florida random-fold results. Reproduced random R² and isolated history self-inclusion. “Known storms” means retrospective shared-event data, not proven live nowcasting. Future-year negatives concern these models, not every viable model. |
| `notebooks/07_clusters_that_separate.ipynb` | Main silhouettes reproduced. “Zone + foundation pairs” labels sometimes name a modal category only: the largest AE/slab cluster is 91.5% AE, not entirely AE. Avoid exact-pair language for mixed clusters. These are geometry illustrations. |
| `notebooks/08_claim_archetypes.ipynb` | Explicitly post-loss descriptive clustering, appropriately reports disagreement and overlap. Selection, depth units, ratio filters and all-year clustering affect temporal comparisons. State-matched contrasts still do not isolate storm causality; chi-square independence is doubtful for event-correlated claims. Expensive full GMM/HDBSCAN sweeps not rerun. |
| `notebooks/09_association_rules.ipynb` | Co-occurrence findings match saved tables and causal caveats are partly present. No predictive holdout; all-year target inherited; antecedent-length mismatch. Calendar before/after comparisons cannot identify rating effects. RR2.0 phase-in dates and regulatory interpretations were not independently verified in this audit. |
| `regional_analysis/regional_research_questions.ipynb` | Outer time split is appropriate, but inherits simple preprocessing and selected features. County target averages include training rows' own outcomes. Number of regions is selected partly using test sample counts: unlabeled test information, not label leakage, but still not a fully training-only design. Regression scores are log scale. |
| `florida_peninsula/florida_peninsula_analysis.ipynb` | Saved random dollar R² 0.628 and forward-check 0.076 without coordinates are consistent with prose. Random training cutoffs/fills/rankings are reused in the forward check. Florida's severity cutoff $136,560 differs from the national one. |
| `gulf_coast/gulf_coast_analysis.ipynb` | Saved random dollar R² 0.389 and forward-check 0.044 without coordinates support limited transfer. Same reuse issue. Regional cutoff $69,481 means classification results measure a different target. |
| `mid_atlantic_northeast/mid_atlantic_northeast_analysis.ipynb` | Saved random dollar R² ≈0.335 and forward-check ≈0.170 support modest signal. Coordinates actually help here, so a universal claim that coordinates always harm transfer is false. Same reuse issue; local cutoff $44,302. |

The **nine class notebooks** (`Code_from_lecture_03` through `11`) were inspected as teaching references. They use Ames Housing, built-in datasets and synthetic clustering examples rather than the NFIP cohort; their scores do not support or contradict NFIP conclusions. Lectures 03 and 11 have no executed code outputs; 06 has eight unexecuted code cells and 10 has one. No notebook contains a saved error output. Examples of preprocessing and feature selection before CV in lecture material should not be copied as independent predictive evaluation. These reference notebooks were not rerun or modified.

## Implemented path and remaining hypotheses

**Implemented and measured:** raw-data reconstruction on the existing cohort, fold-fit imputation/encoding, forward regression validation, multiple target/loss comparisons, class-weight comparison, held-year calibration, validation-only threshold selection, reliability/workload diagnostics, target-history cross-fitting diagnostic, and representative clustering reruns. The executable entry points are `scripts/audit_predictions.py`, `scripts/audit_supplemental.py` and `scripts/audit_history_encoding.py`. `notebooks/10_prediction_audit.ipynb` displays the measured outputs and provides an explicit full-rerun switch.

**Highest-value hypotheses still requiring new data or testing:**

1. Join observed/forecast surge, rain, river level and terrain using a prediction-time contract. Event intensity may improve transfer, but this audit has **not measured** that gain. Redacted rounded coordinates and water-depth unit mixtures constrain these joins. Rain measured after the intended forecast time would answer a different question.
2. For all incoming claims, separately model payment occurrence and positive amount, combining expected payment as `P(paid) × E(payment | paid)`. Verify label finalization first; current regression only covers paid claims. This two-part model has **not been validated here**.
3. Separate single-property claims from multi-policy condominium/master-policy records, and assess per-unit targets with dollar reconstruction. Broad coverage-ratio normalization failed; a carefully defined occupancy/policy-count model remains a hypothesis, not a promised fix.
4. If the real use is triage during an active storm, evaluate chronological arrivals with finalized-payment history available only before each prediction. Random splits do not simulate that. If the real use is future-storm forecasting, retain forward/event holdouts and omit same-storm target history.
5. Consider quantile regression and held-period conformal intervals for amount uncertainty, reporting empirical event/year coverage and interval width. Exchangeability-based coverage is not guaranteed under storm shifts. There are **no validated payment intervals in this delivery**.
6. A rule model or Gower/k-prototypes comparison could improve interpretability of mixed data; higher silhouette alone would not establish better prediction. Repeating algorithm searches until a pleasant cluster score appears is not a solution to low predictive confidence.

Reused 2024–2025 data cannot become fresh again. Freeze the selected pipeline now and reserve genuinely later, sufficiently mature claims or an independently held-out event for the next confirmatory evaluation. Do not tune against these reported test metrics.

## Reproduction and provenance

Run from the project root (the raw file is already available locally):

```sh
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/audit_predictions.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/audit_supplemental.py
OMP_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 python scripts/audit_history_encoding.py
```

Run the scripts **sequentially** because the supplemental comparison consumes the completed main validation table. Set `NFIP_RAW_PATH` to override the existing absolute raw-file path. This run used Python 3.13.9, pandas 2.3.3, NumPy 2.3.5 and scikit-learn 1.7.2. `run_metadata.json` records the split, labels, model choices and limitations; `input_manifest.json` records source-file hashes. No missing XGBoost installation was required, and the original full XGBoost pipeline was not rerun. Original deduplication keys are reused to maintain the same cohort; redacted near-identical rows are not proven to be the same property. An independent raw deduplication-policy sensitivity analysis remains outstanding.

External methodology reference: scikit-learn documents why training target encoding requires [internal cross-fitting](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_target_encoder_cross_val.html). Cross-fitting prevents self-target inclusion, but does not itself enforce calendar-time availability. Field semantics were checked against the project's cached `nfip_claims/column_dictionary.csv`; the current FEMA v3 web page could not be fetched (HTTP 403), so its present content and the notebooks' literature/regulatory claims are not independently certified here.
