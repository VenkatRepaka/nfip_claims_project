# Regional analysis

`regional_research_questions.ipynb` answers the three research questions **per region** instead of with one national model.

- **Regions:** states are grouped by their type of flood (storm surge, river overflow, heavy rain) and their buildings
  (basements, elevated homes, flood zones), using K-means on 2020-2023 claims. The result is 3 regions:
  Hurricane coast, Heavy-rain floods, Inland river floods.
- **Local features:** five county features built from the training years only (past claims, storm-surge share,
  V-zone share, past payments, past share of severe claims).
- **Comparison:** national vs national + local vs regional vs regional + local, trained on 2020-2023 and tested on 2024-2025.

## How to run

Run these first, from the `notebooks/` folder, so the inputs exist in `notebooks/outputs/`:

1. `03_EDA_and_feature_selection_simple.ipynb` (writes `nfip_claims_simple_clean.parquet`, `nfip_claims_simple_keys.parquet`,
   `simple_feature_lists.json`)

Then run `regional_research_questions.ipynb` from this folder (about 1 minute).
