# mid atlantic northeast flood claims

`mid_atlantic_northeast_analysis.ipynb` is self-contained: it reads the raw OpenFEMA file
(`/Users/home/workspace/School/DataMining/project/NfipClaimsV3.parquet`) and does loading, cleaning, missing values,
duplicates, dropped columns, imputation, outliers, new features, encoding, feature selection and RQ1-RQ3 for this region only.

- Split: random 70/30 over 2020-2025, seed 42. All fills and cut-offs come from the training part.
- Latitude and longitude are left out of the models; Section 9.1 shows why, and also how the models do on new storms (2024-2025).
- Runs in about 1 minute.
