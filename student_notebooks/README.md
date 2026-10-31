# Student notebooks: NFIP flood claims 2020-2025

DSC 411/511 Semester Project · Venkat Repaka, Ritik Gaire

| Notebook | What it does | Run time |
|---|---|---|
| `01_EDA.ipynb` | Load, explore and clean the data; saves `data/claims_clean.parquet` | about 1 minute |
| `02_RQ1_payment_amount.ipynb` | RQ1 (regression): predict the dollar amount paid on a building claim | about 5 minutes |
| `03_RQ2_high_severity.ipynb` | RQ2 (classification): predict if a claim is high severity (top 20% of damage) | about 5 minutes |
| `04_RQ3_risk_clusters.ipynb` | RQ3 (K-means, hierarchical, DBSCAN, PCA): do claims form groups with similar flood risk? | about 3 minutes |

## How to run

1. Open the notebooks **from this folder** (`student_notebooks/`), because they read and write `data/claims_clean.parquet`.
2. Run `01_EDA.ipynb` first. Then run 02, 03 and 04 in any order.
3. The raw file is read from `/Users/home/workspace/School/DataMining/project/NfipClaimsV3.parquet`. Change `RAW_FILE` in notebook 01 if it is somewhere else.

Python packages: pandas, numpy, matplotlib, seaborn, scikit-learn, scipy, pyarrow.

## Our answers in short

- **RQ1:** only partly. R² 0.52 for storms already in the data, but only 0.08 for new storms (2024-2025). The payment depends on how hard the flood hit, which our features cannot see.
- **RQ2:** yes, quite well. On 2024-2025 the random forest finds 81% of high-severity claims (F1 0.64, ROC-AUC 0.83). Best in Florida, weaker in other regions.
- **RQ3:** partly. The clusters overlap, but their range of severities (damage as a share of building value) is very different: mobile homes and slab homes in zone AE lose a median of 35-39% of their value, elevated homes only 3%.
