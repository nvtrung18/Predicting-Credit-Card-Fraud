"""
=============================================================================
Kaggle PS S6E11 - Consensus Co-Training Pipeline (Pure ML + Pure DL)
Semi-Supervised Multi-View Consensus Learning
Inputs: submission_task_1_4.csv (ML) + submission_trung.csv (DL)
Produces: submission_co_training.csv, oof_co_training.csv
=============================================================================
"""
import time
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy.stats import rankdata
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

warnings.filterwarnings("ignore")
SEED = 9
N_FOLDS = 5
TRAIN = "data/train.csv"
TEST = "data/test.csv"
FILE_ML = "submission_task_1_4.csv"
FILE_DL = "submission_trung.csv"
OUT_SUB = "submission_co_training.csv"
OUT_OOF = "oof_co_training.csv"

# Consensus thresholds between ML and DL
POS_RANK_THRESH = 0.985   # Top 1.5% highest rank in both -> fraud (1)
NEG_RANK_THRESH = 0.100   # Bottom 10.0% rank in both -> non-fraud (0)

NUMS = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
        "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
        "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes"]
BOOLS = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
         "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
SPLINE_COLS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "card_age_months",
               "customer_age", "account_balance_usd", "velocity_score", "merchant_risk_score"]
DISCRETE = ["txn_count_last_24h", "cvv_retry_count", "prior_disputes", "time_of_day_hour", "day_of_week"]
LOG_COLS = ["amount_usd", "distance_from_home_km", "hours_since_last_txn"]


def log(msg):
    print(time.strftime("[%H:%M:%S] ") + msg, flush=True)


def load_data_and_consensus_pseudo():
    log("Dang doc du lieu Train va Test...")
    tr, te = pd.read_csv(TRAIN), pd.read_csv(TEST)
    for d in (tr, te):
        for c in BOOLS:
            d[c] = d[c].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(int)

    log(f"Dang doc file ML ({FILE_ML}) va file DL ({FILE_DL}) de tim su dong thuan...")
    df_ml = pd.read_csv(FILE_ML)
    df_dl = pd.read_csv(FILE_DL)

    n_te = len(te)
    r_ml = (rankdata(df_ml["is_fraud"]) - 1) / (n_te - 1)
    r_dl = (rankdata(df_dl["is_fraud"]) - 1) / (n_te - 1)

    # Filter high-confidence consensus samples
    pos_consensus = (r_ml >= POS_RANK_THRESH) & (r_dl >= POS_RANK_THRESH)
    neg_consensus = (r_ml <= NEG_RANK_THRESH) & (r_dl <= NEG_RANK_THRESH)
    consensus_mask = pos_consensus | neg_consensus

    te_pseudo = te[consensus_mask].copy()
    te_pseudo["is_fraud"] = np.where(pos_consensus[consensus_mask], 1, 0)

    log(f"  Co-Training POS Consensus (Ca 2 bao Gian lan): {pos_consensus.sum():,} mau")
    log(f"  Co-Training NEG Consensus (Ca 2 bao An toan):  {neg_consensus.sum():,} mau")
    log(f"  Tong so mau dong thuan sieu sach nap vao:     {len(te_pseudo):,} mau ({len(te_pseudo)/n_te*100:.1f}% tap Test)")

    full_train = pd.concat([tr, te_pseudo], ignore_index=True)
    log(f"  Tong tap Train moi sau khi Co-Training:       {len(full_train):,} dong (Goc: {len(tr):,})")

    full_all = pd.concat([full_train.drop(columns="is_fraud"), te], ignore_index=True)
    for c in CATS:
        full_all[c] = full_all[c].astype("category")

    n_tr_orig = len(tr)
    n_tr_full = len(full_train)

    X_train_full = full_all.iloc[:n_tr_full].reset_index(drop=True)
    X_test = full_all.iloc[n_tr_full:].reset_index(drop=True)
    y_train_full = full_train["is_fraud"].values.astype(int)

    return X_train_full, X_test, y_train_full, te["id"].values, n_tr_orig, n_tr_full


def tree_frame(d):
    X = d[NUMS + BOOLS + CATS].copy()
    X["n_flags"] = d[BOOLS].sum(1)
    return X


def codes_frame(d):
    X = tree_frame(d)
    for c in CATS:
        X[c] = X[c].cat.codes
    return X


class LinearFE:
    def fit(self, d):
        self.sp = SplineTransformer(n_knots=8, degree=3, knots="quantile").fit(d[SPLINE_COLS])
        self.ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(d[CATS + DISCRETE].astype(str))
        self.sc = StandardScaler().fit(self._raw(d))
        return self

    def _raw(self, d):
        return np.hstack([self.sp.transform(d[SPLINE_COLS]), self.ohe.transform(d[CATS + DISCRETE].astype(str)),
                          d[BOOLS].values, np.log1p(d[LOG_COLS].values)])

    def transform(self, d):
        return self.sc.transform(self._raw(d)).astype(np.float32)


LGB_BASE = dict(objective="binary", metric="auc", bagging_freq=1, verbose=-1, n_jobs=-1,
                min_data_per_group=200, cat_l2=10)
LGB_CONFIGS = {
    "lgb_nl3": dict(learning_rate=0.06, num_leaves=3, min_child_samples=200, feature_fraction=0.8,
                    bagging_fraction=0.8, lambda_l2=10, cat_smooth=20, seed=SEED),
    "lgb_nl4": dict(learning_rate=0.06, num_leaves=4, min_child_samples=200, feature_fraction=0.7,
                    bagging_fraction=0.8, lambda_l2=10, cat_smooth=20, seed=SEED + 1),
    "lgb_nl8": dict(learning_rate=0.04, num_leaves=8, min_child_samples=300, feature_fraction=0.6,
                    bagging_fraction=0.7, lambda_l2=30, lambda_l1=1, cat_smooth=30, seed=SEED + 2),
}


def fit_lgb(name, Xtr, ytr, Xva, yva, Xte):
    p = {**LGB_BASE, **LGB_CONFIGS[name]}
    m = lgb.train(p, lgb.Dataset(Xtr, ytr), 6000, valid_sets=[lgb.Dataset(Xva, yva)],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    it = m.best_iteration
    return m.predict(Xva, num_iteration=it), m.predict(Xte, num_iteration=it)


def fit_linear(Atr, ytr, Ava, yva, Ate):
    m = LogisticRegression(C=0.1, max_iter=400).fit(Atr, ytr)
    return m.predict_proba(Ava)[:, 1], m.predict_proba(Ate)[:, 1]


def fit_xgb(Xtr, ytr, Xva, yva, Xte):
    import xgboost as xgb
    p = dict(objective="binary:logistic", eval_metric="auc", eta=0.06, max_depth=3, min_child_weight=20,
             subsample=0.8, colsample_bytree=0.7, reg_lambda=10, tree_method="hist", nthread=-1, seed=SEED)
    dtr, dva, dte = xgb.DMatrix(Xtr, ytr), xgb.DMatrix(Xva, yva), xgb.DMatrix(Xte)
    m = xgb.train(p, dtr, 6000, evals=[(dva, "v")], early_stopping_rounds=150, verbose_eval=False)
    r = (0, m.best_iteration + 1)
    return m.predict(dva, iteration_range=r), m.predict(dte, iteration_range=r)


def rank01(x):
    return (rankdata(x) - 1) / (len(x) - 1)


def tune_weights(R, y, iters=300, seed=0):
    rng = np.random.default_rng(seed)
    k = R.shape[1]
    best_w = np.ones(k) / k
    best = roc_auc_score(y, R @ best_w)
    for _ in range(iters):
        w = np.clip(best_w + rng.normal(0, 0.08, k), 0, None)
        if w.sum() == 0:
            continue
        w /= w.sum()
        s = roc_auc_score(y, R @ w)
        if s > best + 1e-7:
            best, best_w = s, w
    return best_w, best


def main():
    t0 = time.time()
    Xtr_full, Xte, y_full, test_id, n_orig, n_total = load_data_and_consensus_pseudo()

    Xtr_t, Xte_t = tree_frame(Xtr_full), tree_frame(Xte)
    lin = LinearFE().fit(pd.concat([Xtr_full, Xte], ignore_index=True))
    Atr, Ate = lin.transform(Xtr_full), lin.transform(Xte)
    Xtr_c, Xte_c = codes_frame(Xtr_full), codes_frame(Xte)

    models = list(LGB_CONFIGS) + ["spline_lr", "xgb"]
    log("Danh sach mo hinh chay: " + ", ".join(models))

    oof = {m: np.zeros(n_orig) for m in models}
    pte = {m: np.zeros(len(Xte)) for m in models}

    skf = StratifiedKFold(N_FOLDS, shuffle=True, random_state=SEED)
    pseudo_indices = np.arange(n_orig, n_total)
    y_orig = y_full[:n_orig]

    for f, (tr_idx, val_idx) in enumerate(skf.split(Xtr_full.iloc[:n_orig], y_orig)):
        full_tr_idx = np.concatenate([tr_idx, pseudo_indices])
        ya, yb = y_full[full_tr_idx], y_orig[val_idx]

        log(f"--- Bat dau FOLD {f}/4 (Train: {len(full_tr_idx):,} mau [gom {len(pseudo_indices):,} Co-Training pseudo] | Val nhan that: {len(val_idx):,} mau) ---")
        for m in models:
            t_m0 = time.time()
            if m in LGB_CONFIGS:
                o, p = fit_lgb(m, Xtr_t.iloc[full_tr_idx], ya, Xtr_t.iloc[val_idx], yb, Xte_t)
            elif m == "spline_lr":
                o, p = fit_linear(Atr[full_tr_idx], ya, Atr[val_idx], yb, Ate)
            elif m == "xgb":
                o, p = fit_xgb(Xtr_c.iloc[full_tr_idx], ya, Xtr_c.iloc[val_idx], yb, Xte_c)

            oof[m][val_idx] = o
            pte[m] += rank01(p) / N_FOLDS
            elapsed = time.time() - t_m0
            log(f"  Fold {f} | {m:10s} | Val AUC (nhan that) = {roc_auc_score(yb, o):.5f} | ({elapsed:.1f}s)")

    log("=== TONG KET OOF AUC CO-TRAINING (TREN NHAN THAT GOC) ===")
    for m in models:
        log(f"  {m:10s}: OOF AUC = {roc_auc_score(y_orig, oof[m]):.5f}")

    # Rank Blending
    R = np.column_stack([rank01(oof[m]) for m in models])
    T = np.column_stack([pte[m] for m in models])
    eq_score = roc_auc_score(y_orig, R.mean(1))
    w, tuned_score = tune_weights(R, y_orig)
    log(f"Equal-weight Blend OOF AUC = {eq_score:.5f}")
    log(f"Tuned Blend OOF AUC        = {tuned_score:.5f}")

    w = 0.7 * w + 0.3 / len(models)
    final_score = roc_auc_score(y_orig, R @ w)
    log(f"Final Shrunk Blend OOF AUC = {final_score:.5f}")
    log("Trong so toi uu: " + ", ".join(f"{m}={x:.3f}" for m, x in zip(models, w)))

    # Save OOF predictions and submission file
    pd.DataFrame({"id": pd.read_csv(TRAIN)["id"], "is_fraud": y_orig, **{m: oof[m] for m in models}}).to_csv(OUT_OOF, index=False)
    sub = pd.DataFrame({"id": test_id, "is_fraud": T @ w})
    assert len(sub) == 299984
    sub.to_csv(OUT_SUB, index=False)
    log(f"DA XUAT FILE NOP CO-TRAINING: {OUT_SUB} ({len(sub):,} dong) trong {(time.time() - t0) / 60:.1f} phut!")


if __name__ == "__main__":
    main()
