# Student notebooks: NFIP flood claims 2020-2025

DSC 411/511 Semester Project · Venkat Repaka, Ritik Gaire

| Notebook | What it does | Run time |
|---|---|---|
| `01_EDA.ipynb` | Load, explore and clean the data; saves `data/claims_clean.parquet` | about 1 minute |
| `build_rain_features.py` | Adds PRISM daily rainfall for each claim; saves `data/rain_features.parquet` (needed by 02 and 03) | about 70 minutes, once |
| `build_track_features.py` | Adds NOAA HURDAT2 hurricane track features for each claim; saves `data/track_features.parquet` (needed by 02) | about 2 minutes |
| `build_census_features.py` | Downloads ACS 2020-2024 5-year block-group data (income, home value, renters); saves `data/acs_blockgroups.parquet` (needed by 09) | under 1 minute |
| `02_RQ1_payment_amount.ipynb` | RQ1 (regression): predict the dollar amount paid on a building claim | about 20 minutes |
| `03_RQ2_high_severity.ipynb` | RQ2 (classification): predict if a claim is high severity (top 20% of damage) | about 5 minutes |
| `04_RQ3_risk_clusters.ipynb` | RQ3 (K-means, hierarchical, DBSCAN, PCA): do claims form groups with similar flood risk? | about 3 minutes |
| `05_payment_speed.ipynb` | Extra question: are claims paid more slowly after mega-storms? (uses `openDate` and `mostRecentPaymentDate`, no new data) | about 2 minutes |
| `06_freeboard.ipynb` | Extra question: is each foot of the lowest floor above the base flood elevation worth it? (uses `elevationDifference`, no new data) | about 2 minutes |
| `07_early_warning.ipynb` | Extra question: can the claims of the first week predict a mega-storm? (follows notebook 05; no new data) | about 1 minute |
| `08_underinsurance.ipynb` | Extra question: who is underinsured, and how much damage was above the coverage? (uses `buildingReplacementCost`, no new data) | about 1 minute |
| `09_census_income.ipynb` | Extra question: are claims in lower-income neighbourhoods paid less, or closed without payment more often, at the same damage? | about 2 minutes |

## How to run

1. Open the notebooks **from this folder** (`student_notebooks/`), because they read and write `data/claims_clean.parquet`.
2. Run `01_EDA.ipynb` first.
3. Run `python build_rain_features.py` (needs `pip install tifffile imagecodecs requests`). It downloads one PRISM file per day
   and keeps finished days in `data/prism_days/`, so it can be stopped and restarted. PRISM allows each file to be downloaded only
   twice a day, so do not delete `data/prism_days/` and run it again on the same day.
4. Run `python build_track_features.py` (one small download from NOAA).
   Run `python build_census_features.py` (three files, about 63 MB, from the US Census Bureau; no API key needed).
5. Then run 02, 03 and 04 in any order.
6. The raw file is read from `/Users/home/workspace/School/DataMining/project/NfipClaimsV3.parquet`. Change `RAW_FILE` in notebook 01, `build_rain_features.py` and `build_track_features.py` if it is somewhere else.

Python packages: pandas, numpy, matplotlib, seaborn, scikit-learn (1.4 or newer), scipy, pyarrow; for the rain script also requests, tifffile, imagecodecs.

Rainfall data: PRISM Climate Group, Oregon State University, https://prism.oregonstate.edu. Hurricane tracks: NOAA National Hurricane Center, HURDAT2. Block-group locations: US Census Bureau, 2020 Centers of Population. Income, home value and renters: US Census Bureau, American Community Survey 2020-2024 5-year estimates.

## Our answers in short

- **RQ1:** only partly. R² 0.52 for storms already in the data, but only 0.08 for new storms (2024-2025). The payment depends on how hard the flood hit, which our features cannot see. Adding rainfall raises new-storm R² to 0.24 (helps rainy storms, not storm-surge storms); adding hurricane tracks raises it to 0.37 (average over 6 left-out storms: 0.12 to 0.25).
- **RQ2:** yes, quite well. On 2024-2025 the random forest finds 81% of high-severity claims (F1 0.64, ROC-AUC 0.83). Best in Florida, weaker in other regions. Rainfall did not help (most severe claims were storm surge).
- **RQ3:** partly. The clusters overlap, but their range of severities (damage as a share of building value) is very different: mobile homes and slab homes in zone AE lose a median of 35-39% of their value, elevated homes only 3%.
- **Payment speed (notebook 05):** yes. Storms with more than 20,000 claims were paid about twice as slowly (median 102 days vs about 50; only 44% paid within 90 days). Helene, Ian and Milton were the slowest; Hurricane Ida was big but fast (61 days).
- **Freeboard (notebook 06):** yes, mostly the first two feet. The median damage ratio falls from 16.8% at the flood line to 6.6% at 2 feet above it (about 60% less for slab buildings in zone AE and in Helene, Ian and Ida). With other features held the same, each foot goes with about 1.5 points less damage ratio (about $3,650 for a typical building in one flood).
- **Early warning (notebook 07):** yes. About 80% of a storm's claims are opened in the first week; first-week claims predict the final number with a 13% median error (vs 55% for a baseline). The model would have flagged all 4 mega-storms by day 14 (Helene and Ian by day 3) with no false alarms.
- **Underinsurance (notebook 08):** more than half of flooded buildings were insured below 80% of their rebuilding cost (median 74%). 34.8% of claims were at the NFIP limit (could not buy more) and 20.7% were underinsured by choice (median $100,000 coverage for a $243,000 building). $0.56 billion of damage was above the coverage (61% at the limit, 28% by choice). Cheaper homes are underinsured by choice 3.6 times more often than the most expensive ones.
- **Census income (notebook 09):** no sign of different treatment. At the same damage ratio, zone, building, storm and place, the richest neighbourhoods get about 3 points more of their damage paid, but the gap disappears (0.0 points) when the damage is compared in dollars: the fixed deductible takes a bigger share of the smaller claims of cheaper buildings. Non-payment is about the same (about 15%). The real gap is before the flood: homes in the poorest neighbourhoods are underinsured by choice 3.3 times more often (33.7% vs 10.1%).
