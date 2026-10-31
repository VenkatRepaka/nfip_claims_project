"""Build the review notebook without altering the original analyses."""
from pathlib import Path
import nbformat as nb
ROOT=Path(__file__).resolve().parents[1]
n=nb.v4.new_notebook()
md=nb.v4.new_markdown_cell;code=nb.v4.new_code_cell
n.cells=[
md('''# Prediction audit: verified results and corrected experiments

Audit of **23 original notebooks**, 21 October 2026. Original analyses are preserved.

**Verified:** severity ranking can improve, but exact payment transfer remains limited. Combined classification results conceal substantial weakness in 2025. A classifier score is not confidence in a particular claim.

The full findings and cell-level evidence are in `reports/prediction_audit/AUDIT.md` and `saved_notebook_evidence.json`. This notebook displays actual results from the scripts, rather than manually entered performance tables.'''),
code('''from pathlib import Path
import json, sys
import pandas as pd
from IPython.display import display, Markdown
ROOT = Path.cwd().resolve()
if ROOT.name == 'notebooks':
    ROOT = ROOT.parent
assert (ROOT / 'scripts/audit_predictions.py').exists(), 'Open from the project root or notebooks directory.'
OUT = ROOT / 'reports/prediction_audit'
sys.path.insert(0, str(ROOT / 'scripts'))
pd.set_option('display.max_columns', 20)
pd.set_option('display.width', 180)
print('Project:', ROOT)
print('Output directory:', OUT)'''),
md('''## Reproduce the experiments

The runs below use the raw snapshot, aligned by unique claim ID to the original simple cohort. `NFIP_RAW_PATH` can override the raw path. The default displays already computed outputs. Set `RERUN=True` to recompute models sequentially; it may take several minutes depending on CPU resources.

Preprocessing is learned within each regression fit. Regression model selection uses forward validation through 2023, followed by a fit through 2023 and evaluation on 2024–2025. Classification is fit through 2022, calibrated and threshold-selected on 2023, then scored on 2024–2025. The severity definition uses observed damage in 2020–2023. No future-year score chooses a model or threshold in these scripts.

These are retrospective comparisons on years already examined in the original project. The existing deduplication keys and policy/building snapshot fields are retained; full point-in-time field availability is not verified.'''),
code('''RERUN = False
if RERUN:
    from threadpoolctl import threadpool_limits
    from audit_predictions import main
    from audit_supplemental import main_extra
    from audit_history_encoding import main_history
    with threadpool_limits(limits=4):
        main()
        main_extra()
        main_history()
meta = json.loads((OUT / 'run_metadata.json').read_text())
display(pd.Series({k: meta[k] for k in [
    'rows', 'target_cutoff_train_2020_2023', 'target_cutoff_all_years',
    'old_label_disagreements_with_train_cutoff', 'selected_regression',
    'selected_classification', 'classification_f1_threshold']}))'''),
md('''## Regression: comparable dollars and cohort

Paid claims only. The original Ridge and Poisson results were reproduced. The revised model improves MAE, but does **not** beat Ridge on dollar R². The unobserved payment size of unpaid/unknown claims is not the target of this experiment.'''),
code('''original = pd.read_csv(OUT / 'original_regression_reproduction.csv')
revised = pd.read_csv(OUT / 'regression_selected_test.csv')
display(pd.concat([original, revised[revised.scope.eq('National')].drop(columns=['scope', 'n'])], ignore_index=True).round(3))
display(revised.round(3))'''),
md('''## Forward validation and failed alternatives

All candidate choices below use older-year validation scores. Direct Poisson wins. Coordinates are not assumed universally harmful; they are evaluated as a candidate. Naive coverage normalization fails badly, and bounded-ratio sensitivity also fails. Neither result justifies discarding legitimate large payments to make metrics look better.'''),
code('''v = pd.read_csv(OUT / 'regression_forward_validation.csv')
bounded = pd.read_csv(OUT / 'regression_bounded_ratio_validation.csv')
display(pd.concat([v, bounded]).pivot(index='model', columns='validation_year', values='r2_dollars').round(3))
print((OUT / 'regression_bounded_ratio_decision.json').read_text())'''),
md('''## Classification: ranking, calibration and operating decisions

Labels differ from notebook 04: the correct training-derived cutoff is $84,724.20, not the all-year $106,127. The closest saved same-label benchmark is notebook 02 (AP 0.670, AUC 0.834), although its preprocessing/features and training cohort differ. It is a retrospective benchmark comparison, not a one-change ablation.

The 0.23 threshold is chosen on 2023 F1. It catches more severe claims while increasing workload and false positives. Calibration and threshold fitting share 2023; the displayed test metrics use later years.'''),
code('''cls = pd.read_csv(OUT / 'classification_test.csv')
display(cls.round(4))
display(pd.read_csv(OUT / 'classification_by_year.csv').round(4))
display(pd.read_csv(OUT / 'cohort_by_year.csv').round(4))'''),
md('''**The year split matters:** F1 is about 0.711 in 2024 and 0.325 in 2025. A combined AP of 0.735 does not establish stable future-year performance.

At higher probability cutoffs, high observed precision covers fewer severe claims. The 0.90 result has only ten positive predictions. No threshold met the validation requirement of 80% precision with at least 100 positive calls, so no such operational guarantee is claimed.'''),
code('''display(pd.read_csv(OUT / 'classification_fixed_confidence.csv').round(4))
display(pd.read_csv(OUT / 'classification_reliability.csv').round(4))'''),
md('''## What the history and clustering reruns establish

Cross-fitting training target histories changes random-split dollar R² from 0.649 to 0.638. This preserves most of the retrospective signal. It does not establish that the history was available at prediction time, and the diagnostic retains the original globally prepared features.

Building-profile silhouettes remain weak at k=7. Stronger location/code geometry is not independent evidence of flood-occurrence risk or causal mitigation effects.'''),
code('''display(pd.read_csv(OUT / 'history_encoding_diagnostic.csv').round(4))
display(pd.read_csv(OUT / 'clustering_reproduction.csv').round(4))'''),
md('''## Verified findings versus remaining hypotheses

**Verified:** global preprocessing and all-year label/feature selection contaminate the simple pipeline; regional future checks reuse random-split preparation; training history encodings include their own targets; exact dollar and log R² are different; confidence is not measured by F1 or cluster separation. The original numerical payment tables were reproduced.

**Implemented:** a separate raw-data pipeline with fold-fit encoders/imputers, forward regression selection, held-year probability calibration, validation-selected thresholds, and year/reliability/workload diagnostics. Original notebooks and outputs remain unchanged.

**Not yet verified:** additional gains from rain/surge/river data, an all-claims two-part payment model, per-unit/occupancy models, mixed-data clustering or payment prediction intervals. A new mature event/year holdout and point-in-time data are needed for confirmatory evaluation. See the full audit for per-notebook findings, source limitations, and exact evidence.'''),
code('''inventory = pd.read_json(OUT / 'notebook_inventory.json')
display(inventory[['path', 'role', 'code_cells', 'unexecuted_code_cells']])
assert len(inventory) == 23
assert all(not e for e in inventory.saved_errors)
print('All 23 original notebooks inventoried. No saved error outputs.')''')]
n.metadata={'kernelspec':{'display_name':'Python 3','language':'python','name':'python3'},'language_info':{'name':'python','version':'3.13.9'}}
nb.write(n,ROOT/'notebooks/10_prediction_audit.ipynb')
print('Created',ROOT/'notebooks/10_prediction_audit.ipynb')
