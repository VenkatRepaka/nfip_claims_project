# Learning path and connection to the class workbooks

This project follows a research process: observe a problem, test a simple response, measure what changed, and explain what remains uncertain. The notebooks are intended to be read in numbered order. Their smaller cells separate ideas rather than hide an entire experiment in one loop.

## How the course topics enter the project

| Class workbook | Topic | Project application | What we do not assume |
|---|---|---|---|
| [03](../class_workbooks/Code_from_lecture_03.ipynb) | Missing values, duplicates, outliers and noise | 01 keeps the raw view, measures deletion cost, checks related blanks and distinguishes unusual from incorrect records | A blank is not always zero; matching redacted records are not proven duplicates |
| [03](../class_workbooks/Code_from_lecture_03.ipynb) | Encoding, transformation, bins, aggregation, sampling and distances | 01 compares raw/log plots, bins, neighbors before/after scaling, sampling and event summaries | More processing is not automatically better |
| [04](../class_workbooks/Code_from_lecture_04.ipynb) | PCA, loadings and filter selection | 02 inspects fitting-only variance and correlation, reads a scree curve and compares all columns, top ten and PCA | Input variance is not the same as predictive value |
| [05](../class_workbooks/Code_from_lecture_05.ipynb) | Decision trees and Naive Bayes | 03 starts with a readable tree, tunes depth inside fitting events, then tests Naive Bayes | A deeper tree or a simple distributional assumption must be checked |
| [06](../class_workbooks/Code_from_lecture_06.ipynb) | KNN, scaling, k, distance and PCA | 03 compares raw/scaled distance, k values, Manhattan/Euclidean distance and PCA | Scaling may improve ranking without improving F1 at a fixed cutoff |
| [07](../class_workbooks/Code_from_lecture_07.ipynb) | Logistic regression, SVM, evaluation, thresholds and leakage | 02 compares row/event validation; 03 traces a logistic score, evaluates SVM, compares metrics and freezes a cutoff before later years | A national score is not evidence of equal performance in every region |
| [08](../class_workbooks/Code_from_lecture_08.ipynb) | K-means, hierarchy, linkage and stability | 04 compares distance choices, k, a dendrogram, linkage and repeat fits | An algorithm returning groups does not prove natural classes |
| [09](../class_workbooks/Code_from_lecture_09.ipynb) | KDE, density scores and DBSCAN | 04 inspects neighbor distances and DBSCAN radius sensitivity; 10 begins with KDE before Isolation Forest | Low density or an unassigned label is not proof of an error |
| [10](../class_workbooks/Code_from_lecture_10.ipynb) | Cluster validation and noise references | 04 adds silhouette diagrams, Davies–Bouldin, sample stability and an independently shuffled-column reference | A high silhouette can be driven by the representation or excluded cases |
| [11](../class_workbooks/Code_from_lecture_11.ipynb) | Association rules, Apriori and FP-Growth | 11 turns claims into stated facts, checks support manually and compares the two miners | Confidence alone does not establish an informative rule |
| [12](../class_workbooks/Code_from_lecture_12.ipynb) | Lift, leverage, conviction, chi-squared and sequences | 11 compares rule measures and evaluates frozen rules on later years | Claim IDs and rounded locations cannot establish building histories |

## What each notebook adds to the previous one

1. **01: Understand the records.** Complete the extensive EDA before choosing models. Keep the field dictionary, data-quality checks and plots visible. Show how the four sources are joined, follow one claim through the join, and check what each source adds.
2. **02: Predict an amount.** Begin with mean/median guesses, linear models and shallow trees. Test ensembles, validation design, reduced inputs and external sources. Then add claim counts per storm and place, try a payment share, and finish with three simple tests of why the score stays modest.
3. **03: Change to a warning question.** Learn the course classifiers on a common sample before a full development comparison. Test input combinations, including claim counts. Separate ranking, cutoff choice, calibration and regional performance, and compare with the plan of warning about every claim.
4. **04: Remove the outcome from fitting.** Ask whether building profiles exist, then challenge the answer using different methods, samples and a noise reference.
5. **05: Study event burden.** Aggregate payment timelines and compare claim sizes before discussing possible capacity limits.
6. **06: Study a misleading mixture.** Start with the overall elevation curve, discover foundation differences, then add controls and uncertainty checks.
7. **07: Test a practical use of early counts.** Move from correlation to meaningful baselines and chronological evaluation. Challenge the event start date.
8. **08: Refine the coverage definition.** A simple ratio needs a cap-aware comparison, cutoff sensitivity and a careful distinction from owner motive.
9. **09: Add social context.** Read raw income differences, then fit A, B, C, D and E in separate stages. Keep the specification that challenges the earlier conclusion visible.
10. **10: Study unusual records.** Start with one-dimensional density, then add multivariate amounts, ratios, peers and weather. Test whether missing data dominate a review list.
11. **11: Explain combinations with rules.** Add readable conditional patterns, compare several measures and check later years. Reject unsupported property-sequence mining.

## How to read a checkpoint

Each stage should answer four questions:

- What limitation did the previous step reveal?
- What is changed in the next comparison?
- Which score or plot tells us whether it helped?
- Do we keep the step, retain it only as a baseline, or reject a stronger interpretation?

A discarded model remains in the tables. A result-specific checkpoint reports the current data rather than a promised improvement. The models selected in development are frozen before later-year evaluation.

## Course methods and project extensions

The basic classifiers and clustering methods come directly from the workbooks. Forests, gradient boosting, coverage-share regression, Isolation Forest and grouped uncertainty estimates extend those ideas. The notebooks label and explain these additions.

Not every taught algorithm belongs in every study. Sequential mining requires a reliable repeated-entity history. This file does not provide one for buildings or households. We use event-level timing in 07 instead of creating unsupported property histories.

The teaching samples in 03 let us inspect method behavior at a practical cost. The main comparison uses engineered core inputs and larger samples where feasible. Settings carried from the teaching experiment are reasonable candidates, not a claim that every algorithm received an exhaustive search.

## Reading small score differences carefully

Before selecting a model, 02 and 03 now show paired event-block uncertainty and event-deletion sensitivity. A higher score is a working choice, not proof that the method is better. Notebook 02 then separates the later-year target/track combinations and checks error concentration in condos. These checks address uncertainty and interpretation; they do not erase the earlier exploration of these years.
