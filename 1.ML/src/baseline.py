import pandas as pd
import numpy as np

# Load data
df = pd.read_csv("data/train_split.csv")

# Feature groups
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

# Convert booleans to 0/1
for col in bool_cols:
    df[col] = df[col].astype(int)

# Split train and validation set
train_df = df[df["is_val"] == False]
val_df = df[df["is_val"] == True]

y_train = train_df["is_fraud"].values
y_val = val_df["is_fraud"].values

# Quick check
print(f"Train shape: {train_df.shape}, Val shape: {val_df.shape}")
print(f"Train fraud: {y_train.mean():.4%}, Val fraud: {y_val.mean():.4%}")

from sklearn.preprocessing import StandardScaler

# One-hot encoding for categorical columns
train_cat = pd.get_dummies(train_df[cat_cols], drop_first=True, dtype=int)
val_cat = pd.get_dummies(val_df[cat_cols], drop_first=True, dtype=int)

# Align columns (make sure train and val have exact same columns)
train_cat, val_cat = train_cat.align(val_cat, join="left", axis=1, fill_value=0)

# Scale numerical columns
scaler = StandardScaler()
X_train_num = scaler.fit_transform(train_df[num_cols])
X_val_num = scaler.transform(val_df[num_cols])

# Combine all features into final matrix X
X_train = np.hstack([X_train_num, train_df[bool_cols].values, train_cat.values])
X_val = np.hstack([X_val_num, val_df[bool_cols].values, val_cat.values])

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape:   {X_val.shape}")

from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
from metrics import compute_roc_auc, compute_pr_auc
# 1. Train Logistic Regression (Linear Baseline)
print("\nTraining Logistic Regression...")
lr = LogisticRegression(max_iter=500, class_weight="balanced", random_state=9)
lr.fit(X_train, y_train)
val_preds_lr = lr.predict_proba(X_val)[:, 1]
print(f"Logistic Regression -> ROC-AUC: {compute_roc_auc(y_val, val_preds_lr):.5f}")
# 2. Train LightGBM (Gradient Boosting Baseline)
print("\nTraining LightGBM...")
train_data = lgb.Dataset(X_train, label=y_train)
val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
params = {
    "objective": "binary",
    "metric": "auc",
    "boosting_type": "gbdt",
    "learning_rate": 0.05,
    "num_leaves": 31,
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
val_preds_lgb = lgb_model.predict(X_val, num_iteration=lgb_model.best_iteration)
roc_lgb = compute_roc_auc(y_val, val_preds_lgb)
pr_lgb = compute_pr_auc(y_val, val_preds_lgb)
print(f"LightGBM -> ROC-AUC: {roc_lgb:.5f} | PR-AUC: {pr_lgb:.5f}")

print("\nPredicting on test set and generating submission file...")
test_df = pd.read_csv("data/test.csv")
for col in bool_cols:
    test_df[col] = test_df[col].astype(int)
test_cat = pd.get_dummies(test_df[cat_cols], drop_first=True, dtype=int)
train_cat, test_cat = train_cat.align(test_cat, join="left", axis=1, fill_value=0)
X_test_num = scaler.transform(test_df[num_cols])
X_test = np.hstack([X_test_num, test_df[bool_cols].values, test_cat.values])
# Predict probabilities on test set
test_preds = lgb_model.predict(X_test, num_iteration=lgb_model.best_iteration)
# Save submission file
sub = pd.DataFrame({"id": test_df["id"], "is_fraud": test_preds})
sub.to_csv("submission_ml.csv", index=False)
print(f"Saved submission_ml.csv successfully ({len(sub):,} rows)!")