# NFIP flood claims: student notebook sequence

**DSC 411/511 semester project · Venkat Repaka and Ritik Gaire**

We study FEMA flood insurance claims with loss years 2020–2025. We combine claims with rainfall, Atlantic hurricane tracks and Census neighborhood data. The notebooks use simple English and a gradual research sequence: inspect a problem, try a simple method, measure what changes, and explain the next decision.

Start with the [learning path and class-workbook map](LEARNING_PATH.md). It links lectures 03–12 to the exact project experiments and explains why some methods are kept, revised or discarded.

## Reading order

| Notebook | Question and purpose |
|---|---|
| [01 · Data and EDA](01_EDA.ipynb) | What do the sources contain? How are they joined? Which values are missing or uncertain? |
| [02 · RQ1: payment](02_RQ1_payment_amount.ipynb) | How well can we predict positive building payments? Which features and model designs help? |
| [03 · RQ2: large losses](03_RQ2_high_severity.ipynb) | Can we flag damage above a threshold defined using fitting years only? |
| [04 · RQ3: profiles](04_RQ3_risk_clusters.ipynb) | Do building and policy features form useful groups? Do external features help? |
| [05 · Payment timelines](05_payment_speed.ipynb) | Are larger events associated with more days to the latest recorded payment? |
| [06 · Floor elevation](06_freeboard.ipynb) | Is floor height above base flood elevation associated with lower damage? |
| [07 · Early claim counts](07_early_warning.ipynb) | Can early counts estimate the recorded event total better than a simple completion rule? |
| [08 · Low coverage](08_underinsurance.ipynb) | How often is building coverage low compared with reported replacement cost? |
| [09 · Neighborhood income](09_census_income.ipynb) | How do payment shares and coverage differ by neighborhood income under several adjustment models? |
| [10 · RQ4: unusual claims](10_anomaly_detection.ipynb) | Does accounting for building size and type change which claims anomaly detection selects? |
| [11 · Association patterns](11_association_patterns.ipynb) | Can readable fact combinations describe major damage, and do frozen rules transfer to later years? Why are property sequences unsupported? |

Notebooks 02–11 read the preparation from 01. They do not require variables left in another notebook's kernel. Run each from the `student_notebooks` folder or use the runner below. Shared data and model code is in `project_utils.py`.

## Data combination

1. Start with the FEMA claims table. Keep unique claim IDs, including similar redacted records with different IDs. Preserve missing outcomes.
2. Join the PRISM feature cache to claims by ID. The builder links the loss date and approximate location to daily rainfall grids. Windows include the day after loss.
3. Join the HURDAT2 feature cache by ID. The builder uses a block-group population center, or rounded coordinates, and searches Atlantic tracks within two days of loss.
4. Join ACS 2020–2024 five-year block-group estimates using the twelve-digit Census code. Many claims can share one neighborhood.
5. Use left joins and validate the key relationships. The claim count must stay the same. Missing matches stay visible. Weather values at unknown fallback locations are excluded.
6. Count the claims that came from the same block group, county and state in the same storm. These claim counts (no payments, no damage) are a rough sign of how widely a storm flooded an area. They are known only after claims arrive, so models that use them are studies of past storms, not first-day forecasts.

Notebook 01 reports coverage by year and summaries across sources. A block-group center is not a house location. HURDAT2 is revised after storms, and ACS describes neighborhoods over five years. External-feature prediction is a **retrospective comparison**, not a verified first-day service.

## How to run

Python 3.11 or newer is recommended. Install the packages once:

```bash
cd student_notebooks
pip install -r requirements.txt
```

Set the raw file path if needed:

```bash
export NFIP_RAW_FILE="/path/to/NfipClaimsV3.parquet"
```

The helper also checks for `data/NfipClaimsV3.parquet` and the original adjacent `project` folder. Raw data are not downloaded by the notebooks.

If the three external feature caches are already present, run the sequence:

```bash
python run_notebooks.py --start 1 --end 11
```

To rebuild from the raw file on a new machine:

```bash
python prepare_data.py
python build_rain_features.py
python build_track_features.py
python build_census_features.py
python run_notebooks.py --start 1 --end 11
```

Rainfall preparation can take about 70 minutes on the first run. Completed days are cached in `data/prism_days/`. Keep this cache: PRISM limits repeated daily downloads. The other builders are shorter. The main prediction notebooks are the slowest analyses; runtime depends on the computer. The runner prints progress and saves executed notebook outputs. Use `--start` and `--end` to rerun a smaller range.

The existing weather caches may miss IDs restored after the previous duplicate-removal rule. These are retained as unmatched claims. Rerun the builders if complete coverage of the restored IDs is needed; notebook 01 reports the current match rates.

## Why the results stay modest

In simple words: the claims file says a lot about the *building and the policy*, and very little about *how much water came into the building*. Water depth is what mostly decides the damage. Rain, storm tracks and Census data describe the weather and the area, not the water inside one home. Notebooks 02 and 03 test this idea directly:

- more training data barely changes the score;
- a fact that an early model cannot know (the inspector's damage estimate) changes the score a lot;
- a few very large condo buildings hold most of the payment variation, so they decide the overall R².

The best gain came from a new clue made from the claims themselves: how many claims came from the same storm and place. It helps because it is a rough measure of how widely the water spread.

## Evaluation rules

- Compare model and feature choices using fitting years 2020–2021 and validation years 2022–2023.
- Freeze the choice, refit on 2020–2023, and evaluate 2024–2025.
- Learn missing-value replacements and category encodings inside each fitting pipeline.
- Define the large-loss cutoff using only the fitting period. The final cutoff uses 2020–2023, never 2024–2025.
- Report useful baselines, several metrics, regional results and event sensitivity. A median error can hide large dollar mistakes.
- Treat leave-one-event-out checks as sensitivity analysis when fitting includes later events.
- State that earlier project versions already explored these years. The later-year evaluation is not a newly collected external test.

## A gradual learning sequence

- **Extensive EDA:** field meanings and attribute types; missing-value patterns and deletion cost; duplicates versus redacted matches; noise versus genuine large losses; time, place, foundation and building mix; Pearson/Spearman correlation; leakage; imputation, bins, encoding, scaling and sampling; join checks; and rain/track/Census exploration.
- **Payment prediction:** mean/median baselines → dollar/log ridge → shallow/deeper trees → ensembles → row/event validation → fitting-only filters and PCA → external sources → coverage-share prediction → later-year and excluded-storm tests. The main development table contains 24 declared candidates.
- **Classification:** a common teaching sample compares decision trees, Naive Bayes, raw/scaled KNN, k and distance, PCA, logistic regression and SVM. A separate main comparison contains 16 candidates. It then freezes the model, chooses a validation cutoff and evaluates later years, regions and storms.
- **Clustering:** distance representation → k and silhouette diagrams → hierarchy and linkage → seed/sample stability → a noise reference → PCA → DBSCAN radius sensitivity → external-feature profiles.
- **Focused studies:** timelines, elevation, early counts, coverage and income build from raw comparisons to more careful definitions or adjusted models. Income models A–E are fitted in separate visible stages.
- **Unusual records and rules:** KDE precedes the multivariate anomaly extension. The final notebook starts with support counts, compares Apriori/FP-Growth and several rule measures, then checks frozen rules on later claims.

The notebooks contain measured checkpoints and keep/discard ledgers. A checkpoint may show a genuine improvement, an unchanged result or a failure. Forests, boosting, coverage-share regression, Isolation Forest and grouped uncertainty estimates are explained as extensions to the course methods.

The small classifier teaching sample supports fair, practical comparisons of the course methods. The main experiment uses engineered inputs and declared fitting budgets. It does not claim that every method received unlimited tuning. See the learning path for those distinctions.

## Auditing small model differences

Notebooks 02 and 03 show event-block bootstrap bands for every main validation candidate, paired score differences for stated comparisons, and the effect of omitting each evaluation event without refitting. Whole events stay together rather than treating thousands of related claims as independent evidence. These checks describe fixed predictions and the observed event mix. They do not include model-fitting uncertainty or correct for selecting the highest score among many candidates. With few storms, the limits can themselves be unstable.

Notebook 02 also completes the later-year dollar/share × core/track comparison and reports each model separately for condos and other buildings. It shows squared-error concentration, including the ten largest mistakes. These are additional audits after seeing test results; they do not create a new untouched test set or change the original development choice. The original share models use more boosting rounds. Two additional matched dollar rows use the same settings and positive-coverage fitting population as the share models. The audit also breaks the overall R² change into condo and other-building contributions using one common denominator. Track comparisons within each design hold settings fixed.

## How to read weak results

Extra data can help on one storm and fail on another. Rain does not measure storm surge or river levels. Track wind is not wind or flood depth at the house. Census values describe areas. Closely related features may repeat information, and rare large condo losses can dominate squared errors.

The notebooks report these limits instead of claiming that a larger feature table must improve prediction. They show what was tested; they do not prove that all further improvement is impossible.

The supporting analyses also have limits. The latest payment is not confirmed final settlement. Low coverage below a cap does not reveal an owner's motive. An adjusted income gap does not prove equal treatment. An anomaly score does not prove an error or fraud. These distinctions are part of the project findings.

## Saved results

Each notebook stores tables in `results/` as CSV files, alongside its inline charts and outputs. The section below is updated from the final execution.

The full runner calls `refresh_summary.py` to update the plain-English findings and this summary from the CSV tables. After rerunning only part of the sequence, run `python refresh_summary.py` to refresh the written findings too. `results/environment.json` records package versions and hashes of the supporting Python files.

<!-- RESULTS_SUMMARY -->

- **Payment amount:** on the later years the chosen model has R² **0.136**. A typical error is **$27,998**. It does worse than a plain average for 2 of six left-out storms, so it is not reliable for every storm.
- **Large losses:** F1 **0.708**, precision **64.9%**, recall **77.7%**. Florida is easy to warn about; other regions are harder.
- **Early counts:** the simple completion rule has a median error of **11.0%** on day 7. The regression has **14.7%**.
- **Income:** the payment gap between rich and poor areas moves from **2.65** points to **0.76** as we add controls. The answer depends on the model.
- **Unusual claims:** the general list is **84.3%** condos; the peer list is **35.3%**. Flags are not confirmed errors.

These numbers use new cleaning rules, cutoffs set on earlier years, and a model chosen on earlier years. Do not compare them with older versions as if only the algorithm changed.

The added payment audit gives later-year R² **-0.006** for core share prediction and **0.415** for core + track dollar prediction. The combined headline does not establish a share-target benefit. Read the matched dollar rows and condo contributions in 02. The original development choice remains unchanged. The event-bootstrap bands in 02 and 03 describe fixed predictions; they do not correct the optimism from choosing among many candidates.


<!-- END_RESULTS_SUMMARY -->

## Sources

- [FEMA NFIP Redacted Claims v3 and data dictionary](https://www.fema.gov/openfema-data-page/fima-nfip-redacted-claims-v3)
- [PRISM Climate Group](https://prism.oregonstate.edu/)
- [NOAA hurricane data](https://www.nhc.noaa.gov/data/) and [HURDAT2 format and post-storm analysis](https://www.nhc.noaa.gov/data/hurdat/hurdat2-format-nov2019.pdf)
- [Census block-group population centers](https://www2.census.gov/geo/docs/reference/cenpop2020/blkgrp/)
- [ACS 2020–2024 five-year table files](https://www2.census.gov/programs-surveys/acs/summary_file/2024/table-based-SF/)
- [NFIP residential coverage guidance](https://www.floodsmart.gov/get-insured/buy-a-policy)
