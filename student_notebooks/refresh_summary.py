"""Refresh plain-English findings from the saved result tables.

Run after notebook execution. No model is fitted and no result is changed.
"""
from pathlib import Path
import json
from importlib.metadata import version
from datetime import datetime, timezone
import hashlib

import pandas as pd
import nbformat

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / 'results'
TAG = 'generated-result-interpretation'


def read(name, levels=1):
    return pd.read_csv(RESULTS / name, index_col=list(range(levels)))


def update_notebook(number, text):
    path = next(ROOT.glob(f'{number:02d}_*.ipynb'))
    notebook = nbformat.read(path, as_version=4)
    notebook.cells = [c for c in notebook.cells if TAG not in c.metadata.get('tags', [])]
    notebook.cells.append(nbformat.v4.new_markdown_cell(
        '### Result in plain English\n\n' + text,
        metadata={'tags': [TAG]}))
    nbformat.validate(notebook)
    nbformat.write(notebook, path)


def checkpoint(number, heading, key, text):
    """Place a measured finding before the next learning step."""
    path = next(ROOT.glob(f'{number:02d}_*.ipynb'))
    notebook = nbformat.read(path, as_version=4)
    tag = 'learning-checkpoint-' + key
    notebook.cells = [c for c in notebook.cells if tag not in c.metadata.get('tags', [])]
    index = next((i for i, c in enumerate(notebook.cells)
                  if c.cell_type == 'markdown' and c.source.startswith(heading)), None)
    if index is None:
        return
    notebook.cells.insert(index, nbformat.v4.new_markdown_cell(
        '**Checkpoint · What the result changes**\n\n' + text,
        metadata={'tags': [tag]}))
    nbformat.write(notebook, path)


def add_model_audit_findings():
    """Keep the audit's limits next to the actual measured differences."""
    for number, key, label in [
        (2, 'core | PCA boosting minus core | boosting', 'PCA versus core boosting'),
        (3, 'core | boosting minus core | forest', 'Boosting versus the forest')]:
        name = f'{number:02d}_validation_paired_gaps.csv'
        if not (RESULTS / name).exists():
            continue
        gaps = read(name)
        row = gaps.loc[key]
        blocks = read(f'{number:02d}_validation_event_blocks.csv')
        omitted = read(f'{number:02d}_validation_omit_event_gaps.csv')
        relevant = omitted[omitted.comparison.eq(key)]
        weakest = relevant.loc[relevant.difference.idxmin()]
        crossing = row['paired 2.5%'] <= 0 <= row['paired 97.5%']
        checkpoint(number, '## 8.', 'candidate-uncertainty',
            f'{label} has a point difference of **{row["difference"]:+.3f}**, with a paired event-bootstrap '
            f'band **[{row["paired 2.5%"]:+.3f}, {row["paired 97.5%"]:+.3f}]**. '
            f'There are **{len(blocks)} evaluation blocks**, many small, and the largest contains '
            f'**{blocks.claims.max()/blocks.claims.sum():.1%}** of validation claims. '
            + ('The band crosses zero, so this check does not show a clear gain. ' if crossing else
               'The band does not cross zero, but this does not remove the selection problem. ')
            + f'Removing **{weakest["omitted event"]}** changes the gap to **{weakest.difference:+.3f}**. '
            + 'We keep the original highest-scoring model as a working choice. These fixed-model bands '
            'do not correct for searching many candidates or prove performance on a new storm.')
    if not all((RESULTS / name).exists() for name in [
        '02_later_target_track_ablation.csv', '02_later_condo_ablation.csv',
        '02_later_ablation_paired_gaps.csv', '02_later_R2_contributions.csv']):
        return
    ablation = read('02_later_target_track_ablation.csv')
    groups = read('02_later_condo_ablation.csv', 2)
    gaps = read('02_later_ablation_paired_gaps.csv')
    track = gaps.loc['core + track share minus core share']
    matched_target = gaps.loc['core + track share minus core + track dollar matched']
    if matched_target['paired 97.5%'] < 0:
        target_reading = 'This check supports the dollar target over the share target in these later years.'
    elif matched_target['paired 2.5%'] > 0:
        target_reading = 'This check supports the share target in these later years, with the stated selection limits.'
    else:
        target_reading = 'The paired band crosses zero, so this check does not show a clear target-design gain.'
    if matched_target['paired 97.5%'] < 0 and matched_target['omit-event maximum'] < 0:
        target_decision = ('The share target scores lower than this matched dollar model in every event-deletion check, '
                           'and its paired event band is below zero. We discard the claim that the share target helped in later years. ')
    else:
        target_decision = target_reading + ' Read the event-deletion results before interpreting the target change. '
    condo = groups.loc[('selected model', 'condo building')]
    other = groups.loc[('selected model', 'other building')]
    checkpoint(2, '### 9.1', 'later-attribution',
        f'Later-year R² is **{ablation.loc["core dollar","R2"]:.3f}** for core dollar prediction, '
        f'**{ablation.loc["core share","R2"]:.3f}** for core share prediction, and '
        f'**{ablation.loc["core + track share","R2"]:.3f}** for share prediction with tracks. '
        f'The track addition within the share design has a difference of **{track["difference"]:+.3f}**, '
        f'with paired band **[{track["paired 2.5%"]:+.3f}, {track["paired 97.5%"]:+.3f}]**. '
        'R² can be negative, so a difference or its band can exceed one. '
        f'The matched dollar rows give R² **{ablation.loc["core dollar matched","R2"]:.3f}** with core inputs '
        f'and **{ablation.loc["core + track dollar matched","R2"]:.3f}** with tracks. '
        'These rows match the share models in boosting settings and fitting population. '
        f'With tracks, share minus matched-dollar R² is **{matched_target["difference"]:+.3f}**, '
        f'with band **[{matched_target["paired 2.5%"]:+.3f}, {matched_target["paired 97.5%"]:+.3f}]**. '
        + target_reading + '\n\n'
        f'For the selected model, condos make up **{condo.claims/(condo.claims+other.claims):.1%}** of claims '
        f'and **{condo["share of model squared error"]:.1%}** of squared error. '
        f'Its R² is **{condo.R2:.3f}** within condos and **{other.R2:.3f}** within other buildings. '
        'These subgroup R² values use different outcome variation; they are not additive. '
        'The audit describes the viewed test years and does not change the earlier development choice.')
    contribution = read('02_later_R2_contributions.csv', 2)
    chosen = contribution.loc['selected model', 'contribution to overall R2 gain vs core dollar']
    reg_val = read('02_validation.csv')
    curve = read('02_learning_curve.csv') if (RESULTS / '02_learning_curve.csv').exists() else None
    missing = read('02_missing_information_test.csv') if (RESULTS / '02_missing_information_test.csv').exists() else None
    limits = ''
    if curve is not None and missing is not None:
        limits = (
            '\n\n**Why the score stays modest.** Four times more training data raises R² by '
            f'**{curve.iloc[-1,0]-curve.iloc[0,0]:+.3f}**, so more data helps a little. Adding the inspector\'s damage estimate, a fact an early model cannot know, '
            f'raises R² by **{missing.iloc[1,0]-missing.iloc[0,0]:+.3f}**. The main limit is therefore missing information about the water '
            'inside each building, not the choice of method.')
    check_path = RESULTS / '02_later_claim_count_check.csv'
    if check_path.exists():
        check = pd.read_csv(check_path, index_col=[0, 1])
        two = check.loc[('two-part: condos without counts, others with counts', 'mixed'), 'all claims R2']
        limits += (
            '\n\n**Claim counts.** They helped on the 2022–2023 validation years and they help ordinary buildings on the later years: '
            f'R² for other buildings goes from **{check.loc[("core + track","coverage fraction"),"other buildings R2"]:.3f}** to '
            f'**{check.loc[("core + track + claim counts","coverage fraction"),"other buildings R2"]:.3f}**. '
            'But for condos they can make predictions much worse, so the model chosen on validation scores lower overall than plain track inputs '
            f'(**{ablation.loc["selected model","R2"]:.3f}** against **{ablation.loc["core + track dollar","R2"]:.3f}**). '
            f'A two-part model, chosen after seeing the test years, reaches **{two:.3f}**. We report this as an exploration, not as a new result.')
    update_notebook(2,
        f'The model chosen on the 2022–2023 validation years is **{reg_val.index[0]}**. On the later years (2024–2025) its R² is '
        f'**{ablation.loc["selected model","R2"]:.3f}**, compared with **{ablation.loc["core dollar","R2"]:.3f}** for the first core model. '
        f'A typical error is still large, so one claim can be far from its prediction.\n\n'
        f'The three audits in section 9 show what helped. Storm-track inputs help a lot '
        f'(core R² **{ablation.loc["core dollar","R2"]:.3f}** → **{ablation.loc["core + track dollar","R2"]:.3f}**). '
        f'The share target alone did not help (**{ablation.loc["core share","R2"]:.3f}**). '
        + target_decision + 'Condos make up '
        f'**{condo.claims/(condo.claims+other.claims):.1%}** of claims but **{condo["share of model squared error"]:.1%}** of the squared error, '
        f'and they account for **{chosen.loc["condo building"]:+.3f}** of the R² gain; other buildings account for **{chosen.loc["other building"]:+.3f}**. '
        'These are extra checks made after the test years were seen, so they do not change the model we froze.'
        + limits)


def add_learning_checkpoints(reg_val, income):
    def value(family):
        return reg_val.loc['core | '+family,'R2']
    checkpoint(2,'## 3. Let a tree','linear',
        f'Dollar ridge gives validation R² **{value("dollar ridge"):.3f}**, while log-target ridge gives '
        f'**{value("linear"):.3f}** after conversion to dollars. A log transformation did not automatically '
        'solve this task. We keep both as baselines and next allow nonlinear groups.')
    checkpoint(2,'## 4. Compare ensembles','tree-depth',
        f'Tree R² is **{value("tree depth 2"):.3f}** at depth 2, **{value("tree depth 6"):.3f}** at depth 6, '
        f'and **{value("tree depth 12"):.3f}** at depth 12. More depth did not steadily improve validation. '
        'Next we combine trees rather than assuming the deepest single tree is best.')
    checkpoint(2,'## 5. Can fewer columns','ensembles',
        f'Core boosting gives R² **{value("boosting"):.3f}**, the forest **{value("forest"):.3f}**, '
        f'and more flexible boosting **{value("flexible boosting"):.3f}**. These results do not support '
        'a simple “more complexity is better” conclusion. Next we change the representation.')
    checkpoint(2,'## 6. Return to','reduction',
        f'With core boosting, validation R² changes from **{value("boosting"):.3f}** to '
        f'**{value("top 10 boosting"):.3f}** using ten selected columns and **{value("PCA boosting"):.3f}** '
        'using PCA. This is the result for this preparation and period. We retain these candidates and '
        'next test information missing from the core table.')
    checkpoint(2,'## 8. Freeze','coverage-share',
        f'Core coverage-share prediction gives R² **{value("coverage fraction"):.3f}**. Adding tracks '
        f'under that design gives **{reg_val.loc["core + track | coverage fraction","R2"]:.3f}**. '
        'These are point estimates. Read the paired event audit before calling either change a clear gain. '
        'The frozen working choice must now face later-year evaluation.')
    def v(table, name, metric='R2'):
        return table.loc[name, metric]
    checkpoint(2,'## 7. Change the target scale','claim-counts',
        f'Core boosting scores R² **{v(reg_val,"core | boosting"):.3f}**. Adding claim counts for the same storm and place gives '
        f'**{v(reg_val,"core + claim counts | boosting"):.3f}**. Adding storm tracks and ratios as well gives '
        f'**{v(reg_val,"best mix | boosting"):.3f}**. The claim counts help most. They tell the model how widely the storm flooded the area, '
        'which rain, tracks and Census values do not. Remember that the counts are known only after claims arrive.')
    cv0 = read('03_validation.csv')
    checkpoint(3,'### Before choosing a winner','claim-counts',
        f'Core boosting has PR-AUC **{v(cv0,"core | boosting","PR-AUC"):.3f}**. Adding claim counts gives '
        f'**{v(cv0,"core + claim counts | boosting","PR-AUC"):.3f}** with boosting and '
        f'**{v(cv0,"core + claim counts | linear","PR-AUC"):.3f}** with a straight-line model. '
        'Mixing in every source does not do better than the claim counts alone. More columns are not always better.')
    course=read('03_course_classifiers.csv')
    cv=read('03_validation.csv')
    checkpoint(3,'## 4. Logistic regression','knn',
        f'On the common teaching sample, scaling KNN changes ROC-AUC from '
        f'**{course.loc["KNN raw units","ROC-AUC"]:.3f}** to **{course.loc["KNN scaled","ROC-AUC"]:.3f}**, '
        f'while F1 changes from **{course.loc["KNN raw units","F1"]:.3f}** to '
        f'**{course.loc["KNN scaled","F1"]:.3f}**. Ranking and a fixed-cutoff warning score can move differently. '
        f'At the tuned settings, PCA changes F1 from **{course.loc["KNN tuned","F1"]:.3f}** to '
        f'**{course.loc["KNN with PCA","F1"]:.3f}**. Even then, the warning performance can remain weak.')
    checkpoint(3,'## 6. Define the main','course-models',
        f'The strongest teaching-sample model by PR-AUC is **{course["PR-AUC"].idxmax()}**, '
        f'with **{course["PR-AUC"].max():.3f}**. This is not yet a national comparison. '
        'Next we change to the engineered core inputs and full validation population. '
        'Settings carried from the teaching experiment are reasonable candidates, not proven optimal settings for that new representation.')
    course_keys=['core | '+name for name in ['course tree','Naive Bayes','KNN','linear','SVM']]
    best=cv.loc[course_keys,'PR-AUC'].idxmax()
    checkpoint(3,'## 7. Add sources','course-extensions',
        f'The best main course candidate is **{best}** (PR-AUC **{cv.loc[best,"PR-AUC"]:.3f}**). '
        f'Core boosting gives **{cv.loc["core | boosting","PR-AUC"]:.3f}**. '
        'Read the fitting budgets and printed runtimes too. The next step keeps the boosting learner fixed and changes the information supplied to it.')
    checkpoint(3,'## 8. Freeze','external-classification',
        f'Boosting PR-AUC is **{cv.loc["core | boosting","PR-AUC"]:.3f}** with core features, '
        f'**{cv.loc["core + rain | boosting","PR-AUC"]:.3f}** with rain, '
        f'**{cv.loc["core + track | boosting","PR-AUC"]:.3f}** with tracks, and '
        f'**{cv.loc["all sources | boosting","PR-AUC"]:.3f}** with all sources. '
        'A successful join is not automatically a useful predictive feature. We keep the weaker combinations visible.')
    checkpoint(9,'### Adjustment D','income-linear',
        f'The income contrast changes from **{income.loc["A","gap Q5 minus Q1 (points)"]:.2f} points** in A '
        f'to **{income.loc["C","gap Q5 minus Q1 (points)"]:.2f}** in C. Dollar loss size matters in this model. '
        'Before declaring the difference explained, we challenge the straight-line damage adjustment.')
    checkpoint(9,'### Adjustment E','income-flexible',
        f'The flexible damage adjustment gives an income contrast of '
        f'**{income.loc["D","gap Q5 minus Q1 (points)"]:.2f} points** in D. '
        'The answer changes under a reasonable model choice. We discard a definitive equal-treatment '
        'conclusion and next test whether broad event exposure changes this estimate.')
    noise=read('04_noise_reference.csv')
    checkpoint(4,'## 6. Does PCA','cluster-noise',
        f'Observed-profile silhouette is **{noise.loc["observed profiles","silhouette"]:.3f}**, '
        f'compared with **{noise.loc["independent-column reference","silhouette"]:.3f}** after breaking field relationships. '
        'The small difference warns that category frequencies and the chosen representation explain much of the apparent separation. '
        'Stability is useful, but it does not by itself prove natural flood-risk classes. Next we test PCA as another representation.')


def main():
    reg = read('02_later_years.csv')
    reg_val = read('02_validation.csv')
    storms = read('02_storm_sensitivity.csv')
    cls = read('03_later_years.csv')
    region = read('03_regions.csv')
    profiles = read('04_cluster_profiles.csv')
    timeline = read('05_event_timelines.csv')
    height = read('06_height_sensitivity.csv')
    early = read('07_chronological_predictions.csv')
    coverage = read('08_single_family_coverage.csv')
    income = read('09_adjusted_income_gap.csv')
    anomalies = read('10_anomaly_comparison.csv', 2)
    designs = read('10_design_sensitivity.csv')
    pooled = designs.loc['pooled ratios, no weather']
    peer_no_weather = designs.loc['peer ratios, no weather']
    r = reg.loc['selected on earlier years']
    c = cls.loc['selected on earlier years']
    negative = storms[storms.R2 < 0]
    g = anomalies.loc[(.01, 'general forest')]
    p = anomalies.loc[(.01, 'peer forest')]
    e7 = early[early.day == 7].groupby('method')['absolute % error'].median()
    h = height.loc['all', 'points per foot']
    low = coverage.loc['below 80%, below cap', 'share']
    sens = read('09_income_sensitivity.csv')
    gap_a = income.loc['A', 'gap Q5 minus Q1 (points)']
    gap_c = income.loc['C', 'gap Q5 minus Q1 (points)']
    gap_e = income.loc['E', 'gap Q5 minus Q1 (points)']
    update_notebook(1,
        'The joins keep every retained claim ID and make missing external matches visible. '
        'The EDA checks field meanings, missingness, unusual values, time, place and preparation choices. '
        'Rain and tracks support prediction, cluster, elevation and anomaly checks; Census supports neighborhood comparisons. '
        'These sources add event and neighborhood context, but they do not measure flood depth at each house. '
        'Different claims, locations and reporting periods make this a study of observed insurance claims, '
        'not all buildings exposed to flooding.')
    flag_all = cls.loc['flag every claim'] if 'flag every claim' in cls.index else None
    update_notebook(3,
        f'The model chosen on the earlier years finds **{c.recall:.1%}** of the large-loss claims in the later years. '
        f'**{c.precision:.1%}** of its warnings are correct (F1 **{c.F1:.3f}**, ROC-AUC **{c["ROC-AUC"]:.3f}**). '
        + (f'Warning about every claim gives F1 **{flag_all.F1:.3f}**, so the model adds **{c.F1-flag_all.F1:+.3f}**. ' if flag_all is not None else '')
        + f'Recall is **{region.loc["Florida", "recall"]:.1%}** in Florida and **{region.loc["Gulf Coast", "recall"]:.1%}** on the Gulf Coast.\n\n'
        'The national result hides a regional gap. Florida has many large losses, so a warning there is easy to get right. '
        'Other regions have fewer large losses and the rule works less well. Changing the cutoff moves the balance between missed losses and false alarms; '
        'it does not add new information. The comparison bands do not allow for choosing the winner from many models, '
        'so a higher validation score is a hint, not proof.')
    update_notebook(4,
        f'The largest group contains **{profiles.claims.max()/profiles.claims.sum():.1%}** of the sample. '
        f'Cluster median damage ratios range from **{profiles.median_damage_ratio.min():.1%}** '
        f'to **{profiles.median_damage_ratio.max():.1%}**. These differences can describe building profiles, '
        'but they do not establish separate natural risk classes.\n\n'
        'Compare method agreement, external-feature results and the foundation-only baseline. '
        'If a simple known category has a similar severity link, the cluster label adds limited information. '
        'Overlapping mixed building types and the chosen distance measure limit separation.')
    update_notebook(5,
        f'The median timeline to the latest recorded payment is '
        f'**{timeline.loc["Hurricane Helene", "median_days"]:.0f} days for Helene**, '
        f'**{timeline.loc["Hurricane Ian", "median_days"]:.0f} for Ian**, '
        f'**{timeline.loc["Hurricane Milton", "median_days"]:.0f} for Milton** and '
        f'**{timeline.loc["Hurricane Ida", "median_days"]:.0f} for Ida**. '
        'Ida shows why event size alone is not a complete explanation. '
        'Claim complexity, location and supplements may also matter. These dates do not establish first-payment time, '
        'final closure or a staffing capacity threshold.')
    update_notebook(6,
        f'With the listed building, event, rain and track controls, one extra foot is associated with '
        f'**{abs(h):.2f} percentage points lower damage ratio** in the selected sample. '
        'The storm-removal and slab-only checks show how the estimate changes with the sample. '
        'The association remains a comparison between different claims. '
        'It is not a measured dollar return from raising a house, because local water conditions and buildings that stayed dry are missing.')
    update_notebook(7,
        f'On the chronological 2024 evaluation, the day-seven completion-factor rule has median absolute error '
        f'**{e7.loc["completion factor"]:.1f}%**, compared with '
        f'**{e7.loc["log-linear regression"]:.1f}%** for regression. '
        'The simple rule is the stronger benchmark in this comparison. '
        'The start-date sensitivity check is essential: a very early loss record can move the warning window. '
        'A high early-count correlation alone is not enough, because the early claims are already part of the total. '
        'The small event sample and revised dates prevent a strong claim about operational warning reliability.')
    update_notebook(8,
        f'Among usable single-family claims, **{low:.1%}** have coverage below 80% of reported replacement cost '
        'and below the $250,000 benchmark. This describes low recorded NFIP coverage. '
        'It does not show whether affordability, advice, other cover or an owner preference explains the amount. '
        'The reported damage above coverage is also narrower than total unpaid flood loss.')
    update_notebook(9,
        f'The estimated highest-minus-lowest income payment-share gap is **{gap_a:.2f} points** in model A, '
        f'**{gap_c:.2f}** after the straight-line log-damage adjustment in C, and '
        f'**{gap_e:.2f}** with the flexible damage curve and event exposure in E.\n\n'
        f'When the deductible is added back to the payment, the gap is only **{sens.loc["deductible added back (cap 1.5)","gap points"]:.2f}** points. '
        'So the gap changes with the model, and most of what is left looks like a deductible effect. '
        'This does not prove that claims are handled equally: we only see neighborhood averages, some Census matches are missing, and we cannot see the claim process.')
    peer_weather = designs.loc['peer ratios + weather']
    update_notebook(10,
        f'At the 1% review budget, condos are **{g["condo share"]:.1%}** of the list from the general detector and '
        f'**{p["condo share"]:.1%}** of the list from the peer detector. At least one review flag is true for '
        f'**{g["any review flag"]:.1%}** and **{p["any review flag"]:.1%}** of these claims.\n\n'
        f'Adding weather to the peer detector makes the list **{peer_weather["condo share"]:.1%}** condos and '
        f'**{peer_weather["missing weather"]:.1%}** of its claims have missing weather. The detector is then finding missing data, not unusual floods, '
        'so weather is kept only as a test and not in the main detector.\n\n'
        'The review lists show how the meaning of "unusual" changes the result. They are not proof of errors. '
        'Only a check of the original documents can show whether a flagged claim is wrong.')
    add_learning_checkpoints(reg_val, income)
    add_model_audit_findings()
    if (RESULTS / '11_later_rules.csv').exists():
        rules = read('11_later_rules.csv')
        better = int(rules["beats best single fact later"].sum()) if "beats best single fact later" in rules else 0
        update_notebook(11,
            f'**{int(rules["later lift"].gt(1).sum())} of {len(rules)}** frozen rules still have a lift above 1 in the later years, '
            f'but only **{better}** of {len(rules)} beat their own best single fact. '
            'The rules describe claims we can see, not the chance that any building floods. '
            'We keep support, confidence, lift and excess counts together, and we do not mine building histories, '
            'because the file has no stable building ID.')
    audit_summary = ''
    if (RESULTS / '02_later_R2_contributions.csv').exists():
        audit = read('02_later_target_track_ablation.csv')
        audit_summary = (
            f'\nThe added payment audit gives later-year R² **{audit.loc["core share","R2"]:.3f}** '
            f'for core share prediction and **{audit.loc["core + track dollar","R2"]:.3f}** '
            'for core + track dollar prediction. The combined headline does not establish a share-target benefit. '
            'Read the matched dollar rows and condo contributions in 02. The original development choice remains unchanged. '
            'The event-bootstrap bands in 02 and 03 describe fixed predictions; they do not correct '
            'the optimism from choosing among many candidates.\n')
    summary = f'''- **Payment amount:** on the later years the chosen model has R² **{r.R2:.3f}**. A typical error is **${r['median error $']:,.0f}**. It does worse than a plain average for {len(negative)} of six left-out storms, so it is not reliable for every storm.
- **Large losses:** F1 **{c.F1:.3f}**, precision **{c.precision:.1%}**, recall **{c.recall:.1%}**. Florida is easy to warn about; other regions are harder.
- **Early counts:** the simple completion rule has a median error of **{e7.loc['completion factor']:.1f}%** on day 7. The regression has **{e7.loc['log-linear regression']:.1f}%**.
- **Income:** the payment gap between rich and poor areas moves from **{gap_a:.2f}** points to **{gap_e:.2f}** as we add controls. The answer depends on the model.
- **Unusual claims:** the general list is **{g['condo share']:.1%}** condos; the peer list is **{p['condo share']:.1%}**. Flags are not confirmed errors.

These numbers use new cleaning rules, cutoffs set on earlier years, and a model chosen on earlier years. Do not compare them with older versions as if only the algorithm changed.
{audit_summary}
'''
    readme = ROOT / 'README.md'
    text = readme.read_text()
    start = '<!-- RESULTS_SUMMARY -->'
    end = '<!-- END_RESULTS_SUMMARY -->'
    if end in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + start + '\n\n' + summary + '\n' + end + after
    else:
        text = text.replace(start, start + '\n\n' + summary + '\n' + end)
    readme.write_text(text)
    manifest = {'updated_utc': datetime.now(timezone.utc).isoformat(), 'seed': 42,
                'packages': {name: version(name) for name in ['pandas', 'numpy', 'scikit-learn', 'scipy', 'pyarrow', 'statsmodels', 'nbclient', 'seaborn', 'mlxtend']},
                'source_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                  for path in sorted(ROOT.glob('*.py'))}}
    (RESULTS / 'environment.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print('Updated plain-English findings, README results and package manifest.')


if __name__ == '__main__':
    main()
