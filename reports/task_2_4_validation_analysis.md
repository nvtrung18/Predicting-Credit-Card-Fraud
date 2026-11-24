# Task 2.4 — Validation analysis

## Method

Compared the four saved validation prediction sets on the shared seed-9 stratified holdout: MLP baseline, TabNet baseline, MLP with class weighting, and MLP with categorical embeddings. Selection used highest ROC-AUC, with Average Precision as the tie-breaker. Test labels and the public leaderboard were not used. Cross-validation was not run.

## Results

| Model | Validation ROC-AUC | Average Precision | Best epoch |
|---|---:|---:|---:|
| MLP baseline | 0.944622 | 0.484340 | 9 |
| MLP with class weighting | 0.944596 | 0.480716 | 5 |
| MLP with categorical embeddings | 0.944339 | 0.482583 | 10 |
| TabNet baseline | 0.943122 | 0.479909 | 14 |

Selected model: **MLP baseline**. It had the highest ROC-AUC and Average Precision among these validation runs.

At a 0.5 probability threshold, the selected model produced 661 true positives, 137,433 true negatives, 221 false positives, and 1,677 false negatives. This threshold is used only for error interpretation; it is not used to compute ROC-AUC or create the submission.

## Artifacts

- Kaggle notebook: [Credit Card Fraud 2.4 Validation Analysis, Version 2](https://www.kaggle.com/code/nvtqx1/credit-card-fraud-2-4-validation-analysis/versions/2)
- Model comparison: [comparison.csv](../2.DL/results/task_2_4_validation_analysis/comparison.csv)
- Selection decision: [selected_model.json](../2.DL/results/task_2_4_validation_analysis/selected_model.json)
- Summary: [analysis_summary.md](../2.DL/results/task_2_4_validation_analysis/analysis_summary.md)
- ROC and probability distributions: [roc_and_probability_distributions.png](../2.DL/results/task_2_4_validation_analysis/roc_and_probability_distributions.png)
- Learning curves: [learning_curves_comparison.png](../2.DL/results/task_2_4_validation_analysis/learning_curves_comparison.png)

The Kaggle notebook is private. The local artifacts in this repository contain the comparison results used for Task 2.5.
