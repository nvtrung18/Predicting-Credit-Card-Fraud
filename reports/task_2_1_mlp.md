# Task 2.1 — MLP baseline

## Kaggle run

- Notebook version: [Credit Card Fraud — 2.1 MLP Baseline, Version 1](https://www.kaggle.com/code/nvtqx1/credit-card-fraud-2-1-mlp-baseline/versions/1)
- Runtime: Kaggle GPU, Tesla T4.
- Validation split: stratified 80/20 with `random_state=9`.
- Rows: 559,968 train and 139,992 validation.
- Preprocessing: excluded `id`; standardized numeric and boolean features; one-hot encoded five categorical features. The encoder was fitted on the training split only.
- Model: 54 inputs → 128 → 64 → 1, ReLU activations, dropout 0.20/0.10.
- Loss: unweighted `BCEWithLogitsLoss`.
- Training: best checkpoint at epoch 9; 12 epochs completed, early stopping patience 3.

## Validation results

| Metric | Result |
|---|---:|
| ROC-AUC | 0.944622 |
| Average Precision | 0.484340 |

These are holdout validation results, not a Kaggle leaderboard score. No public test predictions or competition submission were produced in Task 2.1.

## Files

- Kaggle notebook export: [credit-card-fraud-2-1-mlp-baseline.ipynb](../2.DL/src/credit-card-fraud-2-1-mlp-baseline.ipynb)
- Shared split membership: [split_ids_seed9.csv](../shared/splits/split_ids_seed9.csv)
- Metrics: [metrics.json](../2.DL/results/task_2_1_mlp/metrics.json)
- Epoch history: [training_history.csv](../2.DL/results/task_2_1_mlp/training_history.csv)
- Validation probabilities: [validation_predictions_seed9.csv](../2.DL/results/task_2_1_mlp/validation_predictions_seed9.csv)
- Learning curves: [learning_curves.png](../2.DL/results/task_2_1_mlp/learning_curves.png)
- Model checkpoint: [best_model.pt](../2.DL/checkpoints/task_2_1_mlp/best_model.pt)
- Fitted preprocessing pipeline: [preprocessor.joblib](../2.DL/checkpoints/task_2_1_mlp/preprocessor.joblib)

Use `split_ids_seed9.csv` in the ML branch so both branches evaluate on the same rows. The Kaggle notebook version is private, so collaborators need access to the Kaggle notebook or the local files in this repository.

## Reuse in Kaggle

The `.ipynb` file is exported from Kaggle Version 1 and is the runnable project source. Upload/open it as a Kaggle notebook, attach the competition data if needed, enable the Kaggle GPU, then use **Run All**.
