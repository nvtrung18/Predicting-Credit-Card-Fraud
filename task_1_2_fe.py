import time
import pandas as pd
import numpy as np
import lightgbm as lgb
from metrics import compute_roc_auc, compute_pr_auc

# Load data and split
print("Loading data from data/train_split.csv...")
df = pd.read_csv("data/train_split.csv")

train_mask = (df["is_val"] == False)
val_mask = (df["is_val"] == True)

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
    df[col] = df[col].astype(int)

for col in cat_cols:
    df[col] = df[col].astype("category").cat.codes

# Feature engineering
print("Engineering new domain-specific features...")

# Amount relative to merchant category mean
cat_mean = df.loc[train_mask].groupby("merchant_category")["amount_usd"].mean()
df["amt_to_cat_mean"] = df["amount_usd"] / df["merchant_category"].map(cat_mean)

# Late-night transaction flag (23:00 - 05:00)
df["is_night_txn"] = df["time_of_day_hour"].apply(lambda h: 1 if (h >= 23 or h <= 5) else 0)

# Sum of security risk flags
df["risk_flag_sum"] = df[bool_cols].sum(axis=1)

# Amount to account balance ratio
df["amt_to_balance_ratio"] = df["amount_usd"] / (df["account_balance_usd"] + 1.0)

# Log-transformed distance from home
df["log_distance_km"] = np.log1p(df["distance_from_home_km"])

# Transaction velocity rate per hour
df["txn_velocity_rate"] = df["txn_count_last_24h"] / (df["hours_since_last_txn"] + 0.1)

# Merchant risk and velocity interaction
df["merchant_velocity_risk"] = df["merchant_risk_score"] * df["velocity_score"]

new_features = [
    "amt_to_cat_mean", "is_night_txn", "risk_flag_sum",
    "amt_to_balance_ratio", "log_distance_km", "txn_velocity_rate",
    "merchant_velocity_risk"
]

all_features = num_cols + bool_cols + cat_cols + new_features
print(f"Total features: {len(all_features)} (Original: {len(num_cols + bool_cols + cat_cols)}, Added: {len(new_features)})")

X_train = df.loc[train_mask, all_features]
y_train = df.loc[train_mask, "is_fraud"].values

X_val = df.loc[val_mask, all_features]
y_val = df.loc[val_mask, "is_fraud"].values

# Train LightGBM with class weighting
print("\nTraining LightGBM with scale_pos_weight (Task 1.2)...")
t0 = time.time()

train_data = lgb.Dataset(X_train, label=y_train, categorical_feature=cat_cols)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data, categorical_feature=cat_cols)

params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "scale_pos_weight": 5.0,  # Handle class imbalance
    "random_state": 9,
    "verbose": -1,
    "n_jobs": -1
}

lgb_model = lgb.train(
    params,
    train_data,
    num_boost_round=300,
    valid_sets=[val_data],
    callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)]
)

train_time = time.time() - t0
val_preds = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
roc_1_2 = compute_roc_auc(y_val, val_preds)
pr_1_2 = compute_pr_auc(y_val, val_preds)

print(f"Task 1.2 Result -> ROC-AUC: {roc_1_2:.5f} | PR-AUC: {pr_1_2:.5f} | Best Iteration: {lgb_model.best_iteration}")

# Predict test set and save submission
print("\nGenerating submission file for Task 1.2...")
test_df = pd.read_csv("data/test.csv")

for col in bool_cols:
    test_df[col] = test_df[col].astype(int)

for col in cat_cols:
    test_df[col] = test_df[col].astype("category").cat.codes

# Apply same engineered features to test set
test_df["amt_to_cat_mean"] = test_df["amount_usd"] / test_df["merchant_category"].map(cat_mean)
test_df["is_night_txn"] = test_df["time_of_day_hour"].apply(lambda h: 1 if (h >= 23 or h <= 5) else 0)
test_df["risk_flag_sum"] = test_df[bool_cols].sum(axis=1)
test_df["amt_to_balance_ratio"] = test_df["amount_usd"] / (test_df["account_balance_usd"] + 1.0)
test_df["log_distance_km"] = np.log1p(test_df["distance_from_home_km"])
test_df["txn_velocity_rate"] = test_df["txn_count_last_24h"] / (test_df["hours_since_last_txn"] + 0.1)
test_df["merchant_velocity_risk"] = test_df["merchant_risk_score"] * test_df["velocity_score"]

X_test = test_df[all_features]
test_preds = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)

sub = pd.DataFrame({"id": test_df["id"], "is_fraud": test_preds})
sub.to_csv("submission_task_1_2.csv", index=False)
print(f"Saved submission_task_1_2.csv successfully ({len(sub):,} rows)!")
