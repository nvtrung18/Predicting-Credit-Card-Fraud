# ML

## Credit Card Fraud — Kaggle

The DL branch is evaluated on the shared stratified 80/20 train-validation split with seed 9. Validation results below are not leaderboard scores.

| Task | Model | Validation ROC-AUC | Average Precision |
|---|---|---:|---:|
| 2.1 | MLP baseline | 0.944622 | 0.484340 |
| 2.2 | TabNet baseline | 0.943122 | 0.479909 |
| 2.3 | MLP with class weighting | 0.944596 | 0.480716 |
| 2.3 | MLP with categorical embeddings | 0.944339 | 0.482583 |

Task 2.4 selected the MLP baseline by validation ROC-AUC, using Average Precision as the tie-breaker. Cross-validation was not run; model selection used the single shared holdout.

Task 2.5 refit the selected MLP on all 699,960 training rows for 9 epochs on a Kaggle Tesla T4. The generated submission contains 299,984 test rows with columns id and is_fraud. The user subsequently reported a Kaggle score of 0.94263; this is distinct from the validation results above.

## Notebooks and reports

- 2.1: [Kaggle notebook](2.DL/src/credit-card-fraud-2-1-mlp-baseline.ipynb), [report](reports/task_2_1_mlp.md)
- 2.2: [Kaggle notebook](2.DL/src/credit-card-fraud-2-2-tabnet-baseline.ipynb)
- 2.3: [Kaggle notebook](2.DL/src/credit-card-fraud-2-3-mlp-imbalance-experiments.ipynb)
- 2.4: [Kaggle notebook, Version 2](https://www.kaggle.com/code/nvtqx1/credit-card-fraud-2-4-validation-analysis/versions/2), [report](reports/task_2_4_validation_analysis.md)
- 2.5: [Kaggle notebook, Version 1](https://www.kaggle.com/code/nvtqx1/credit-card-fraud-2-5-final-dl-submission/versions/1), [report](reports/task_2_5_final_submission.md), [submission.csv](2.DL/results/task_2_5_final/submission.csv)
- Shared split IDs: [split_ids_seed9.csv](shared/splits/split_ids_seed9.csv)

## DL optimization — tasks 2.6–2.11

The optimization notebooks are implemented but have not yet been trained on Kaggle. No improved score is claimed. Training is guarded to run only on Kaggle GPU; task 2.11 performs inference from saved CV checkpoints and does not train again.

| Task | Source | Required notebook outputs as Kaggle inputs |
|---|---|---|
| 2.6 | [Experiment foundation](2.DL/src/credit-card-fraud-2-6-experiment-foundation.ipynb) | 2.1 |
| 2.7 | [MLP tuning](2.DL/src/credit-card-fraud-2-7-mlp-tuning.ipynb) | 2.6 |
| 2.8 | [DL feature experiments](2.DL/src/credit-card-fraud-2-8-dl-feature-experiments.ipynb) | 2.6, 2.7 |
| 2.9 | [DL cross-validation](2.DL/src/credit-card-fraud-2-9-dl-cross-validation.ipynb) | 2.6, 2.8 |
| 2.10 | [OOF DL ensemble](2.DL/src/credit-card-fraud-2-10-dl-oof-ensemble.ipynb) | 2.6, 2.9 |
| 2.11 | [Optimized DL submission](2.DL/src/credit-card-fraud-2-11-optimized-dl-submission.ipynb) | 2.6, 2.9, 2.10 |

Attach competition data to every notebook. Start with 2.6, save its completed output, then follow the dependency table. Full instructions, experiment budgets and viva explanations: [DL optimization Kaggle guide](2.DL/docs/dl-optimization-kaggle-guide.md).
