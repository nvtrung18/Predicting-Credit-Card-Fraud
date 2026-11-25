import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from metrics import compute_roc_auc, compute_pr_auc

# Load train and test data
print("Loading data for Task 1.3 (Stratified 5-Fold Cross Validation)...")
train_df = pd.read_csv("data/train.csv")
test_df = pd.read_csv("data/test.csv")

num_cols = [
    "amount_usd", "hours_since_last_txn", "txn_count_last_24h",
    "distance_from_home_km", "card_age_months", "customer_age",
    "account_balance_usd", "cvv_retry_count", "velocity_score",
    "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes"
]

bool_cols = [
    "is_foreign_transaction", "is_new_merchant", "used_vpn",
    "ip_country_mismatch", "billing_shipping_mismatch",
    "is_ai_generated_scam_attempt"
]

cat_cols = [
    "merchant_category", "card_type", "auth_method", "channel", "device_type"
]

for col in bool_cols:
    train_df[col] = train_df[col].astype(int)
    test_df[col] = test_df[col].astype(int)

for col in cat_cols:
    train_df[col] = train_df[col].astype("category").cat.codes
    test_df[col] = test_df[col].astype("category").cat.codes

# Apply engineered features from Task 1.2
print("Applying optimized features to Train and Test...")

# Amount relative to category mean
cat_mean = train_df.groupby("merchant_category")["amount_usd"].mean()
train_df["amt_to_cat_mean"] = train_df["amount_usd"] / train_df["merchant_category"].map(cat_mean)
test_df["amt_to_cat_mean"] = test_df["amount_usd"] / test_df["merchant_category"].map(cat_mean)

# Late-night transaction flag
train_df["is_night_txn"] = train_df["time_of_day_hour"].apply(lambda h: 1 if (h >= 23 or h <= 5) else 0)
test_df["is_night_txn"] = test_df["time_of_day_hour"].apply(lambda h: 1 if (h >= 23 or h <= 5) else 0)

# Sum of security risk flags
train_df["risk_flag_sum"] = train_df[bool_cols].sum(axis=1)
test_df["risk_flag_sum"] = test_df[bool_cols].sum(axis=1)

# Amount to balance ratio
train_df["amt_to_balance_ratio"] = train_df["amount_usd"] / (train_df["account_balance_usd"] + 1.0)
test_df["amt_to_balance_ratio"] = test_df["amount_usd"] / (test_df["account_balance_usd"] + 1.0)

# Log-transformed distance
train_df["log_distance_km"] = np.log1p(train_df["distance_from_home_km"])
test_df["log_distance_km"] = np.log1p(test_df["distance_from_home_km"])

# Transaction velocity rate
train_df["txn_velocity_rate"] = train_df["txn_count_last_24h"] / (train_df["hours_since_last_txn"] + 0.1)
test_df["txn_velocity_rate"] = test_df["txn_count_last_24h"] / (test_df["hours_since_last_txn"] + 0.1)

# Merchant risk and velocity interaction
train_df["merchant_velocity_risk"] = train_df["merchant_risk_score"] * train_df["velocity_score"]
test_df["merchant_velocity_risk"] = test_df["merchant_risk_score"] * test_df["velocity_score"]

new_features = [
    "amt_to_cat_mean", "is_night_txn", "risk_flag_sum",
    "amt_to_balance_ratio", "log_distance_km", "txn_velocity_rate",
    "merchant_velocity_risk"
]

all_features = num_cols + bool_cols + cat_cols + new_features

X = train_df[all_features].values
y = train_df["is_fraud"].values
X_test = test_df[all_features].values

# Stratified 5-Fold Cross Validation
print("\nStarting Stratified 5-Fold Cross Validation (seed=9)...")
from sklearn.model_selection import StratifiedKFold

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=9)

oof_preds = np.zeros(len(train_df))
test_preds = np.zeros(len(test_df))
fold_scores = []

# Tuned LightGBM hyperparameters
tuned_params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 45,
    "max_depth": 8,
    "feature_fraction": 0.85,
    "bagging_fraction": 0.85,
    "bagging_freq": 1,
    "min_child_samples": 30,
    "scale_pos_weight": 3.0,
    "random_state": 9,
    "verbose": -1,
    "n_jobs": -1
}

t_start = time.time()

for fold, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
    X_train_f, y_train_f = X[train_idx], y[train_idx]
    X_val_f, y_val_f = X[val_idx], y[val_idx]
    
    train_data = lgb.Dataset(X_train_f, label=y_train_f)
    val_data = lgb.Dataset(X_val_f, label=y_val_f, reference=train_data)
    
    model = lgb.train(
        tuned_params,
        train_data,
        num_boost_round=350,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
    )
    
    val_pred = model.predict(X_val_f, num_iteration=model.best_iteration)
    oof_preds[val_idx] = val_pred
    
    fold_auc = compute_roc_auc(y_val_f, val_pred)
    fold_scores.append(fold_auc)
    print(f"Fold {fold}/5 -> ROC-AUC: {fold_auc:.5f} (Best Iter: {model.best_iteration})")
    
    # Average test predictions across 5 folds
    test_preds += model.predict(X_test, num_iteration=model.best_iteration) / 5.0

cv_time = time.time() - t_start
overall_oof_auc = compute_roc_auc(y, oof_preds)
overall_oof_pr = compute_pr_auc(y, oof_preds)

print("\n" + "=" * 55)
print("     TỔNG KẾT TASK 1.3: 5-FOLD CROSS VALIDATION")
print("=" * 55)
print(f"Điểm từng Fold: {[round(s, 5) for s in fold_scores]}")
print(f"Mean Fold AUC:  {np.mean(fold_scores):.5f} (+/- {np.std(fold_scores):.5f})")
print(f"Overall OOF ROC-AUC: {overall_oof_auc:.5f}")
print(f"Overall OOF PR-AUC : {overall_oof_pr:.5f}")
print(f"Tổng thời gian 5 Folds: {cv_time:.2f} giây")
print("=" * 55)

# Save OOF predictions and final submission
print("\nSaving Out-Of-Fold (OOF) predictions and Final Test Submission...")

# Save OOF predictions for ensembling
oof_df = pd.DataFrame({"id": train_df["id"], "oof_pred_ml": oof_preds})
oof_df.to_csv("oof_preds_ml.csv", index=False)
print("Saved oof_preds_ml.csv!")

# Save test submission
sub = pd.DataFrame({"id": test_df["id"], "is_fraud": test_preds})
sub.to_csv("submission_task_1_3_kfold.csv", index=False)
print(f"Saved submission_task_1_3_kfold.csv successfully ({len(sub):,} rows)!")
