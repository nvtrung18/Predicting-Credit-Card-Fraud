"""Shared data loading, feature engineering, folds and prediction I/O for the ML + DL ensemble.

Feature logic is taken unchanged from 1.ML/src/task_1_5_decimal_fe.py so that the ML models
can be reproduced on the shared folds and the DL models can reuse the same engineered features.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, SplineTransformer, StandardScaler

SEED = 9
N_FOLDS = 5
N_TRAIN = 699_960
N_TEST = 299_984
TARGET = "is_fraud"

NUMS = ["amount_usd", "hours_since_last_txn", "txn_count_last_24h", "distance_from_home_km",
        "card_age_months", "customer_age", "account_balance_usd", "cvv_retry_count",
        "velocity_score", "time_of_day_hour", "day_of_week", "merchant_risk_score", "prior_disputes"]
BOOLS = ["is_foreign_transaction", "is_new_merchant", "used_vpn", "ip_country_mismatch",
         "billing_shipping_mismatch", "is_ai_generated_scam_attempt"]
CATS = ["merchant_category", "card_type", "auth_method", "channel", "device_type"]
SPLINE_COLS = ["amount_usd", "hours_since_last_txn", "distance_from_home_km", "card_age_months",
               "customer_age", "account_balance_usd", "velocity_score", "merchant_risk_score"]
DISCRETE = ["txn_count_last_24h", "cvv_retry_count", "prior_disputes", "time_of_day_hour", "day_of_week"]
CONTINUOUS = [c for c in NUMS if c not in DISCRETE]
LOG_COLS = ["amount_usd", "distance_from_home_km", "hours_since_last_txn"]

DECIMAL_SCALES = {
    "amount_usd": 100,
    "account_balance_usd": 100,
    "hours_since_last_txn": 100,
    "distance_from_home_km": 100,
    "velocity_score": 10,
    "merchant_risk_score": 10,
}
EXTRA_FEATURES = (["n_flags"] + [f"{c}_fraction" for c in DECIMAL_SCALES]
                  + ["log_amt_rel_balance", "velocity_x_risk"])


# --------------------------------------------------------------------------- paths
def find_file(name: str, roots: tuple[str, ...] = ("/kaggle/input", "data/raw", "data", ".")) -> Path:
    """Return the single file called `name` under the given roots (Kaggle input or local repo)."""
    hits: list[Path] = []
    for root in roots:
        if os.path.isdir(root):
            hits.extend(p for p in Path(root).rglob(name) if p.is_file())
        if hits:
            break
    if len(hits) != 1:
        raise FileNotFoundError(f"Expected exactly one '{name}', found {len(hits)}: {hits[:5]}")
    return hits[0]


# --------------------------------------------------------------------------- data
def load(train_path=None, test_path=None):
    """Load train/test, booleans -> 0/1, categories built on train+test so codes agree.

    Returns (train_X, test_X, y, train_id, test_id). Both frames keep the `id` column.
    """
    tr = pd.read_csv(train_path or find_file("train.csv"))
    te = pd.read_csv(test_path or find_file("test.csv"))
    for d in (tr, te):
        for c in BOOLS:
            d[c] = d[c].astype(str).str.lower().isin(["true", "1", "1.0"]).astype(int)
    full = pd.concat([tr.drop(columns=TARGET), te], ignore_index=True)
    for c in CATS:
        full[c] = full[c].astype("category")
    n = len(tr)
    tr_x = full.iloc[:n].reset_index(drop=True)
    te_x = full.iloc[n:].reset_index(drop=True)
    return tr_x, te_x, tr[TARGET].to_numpy(dtype=int), tr["id"].to_numpy(), te["id"].to_numpy()


# --------------------------------------------------------------------------- features
def add_extra_features(d: pd.DataFrame) -> pd.DataFrame:
    """Engineered columns from ML task 1.5: flag count, decimal fractions, two interactions."""
    X = pd.DataFrame(index=d.index)
    X["n_flags"] = d[BOOLS].sum(1)
    for name, scale in DECIMAL_SCALES.items():
        digits = np.rint(np.abs(d[name].to_numpy(dtype="float64")) * scale)
        X[f"{name}_fraction"] = ((digits % scale) / scale).astype(np.float32)
    amt = np.maximum(d["amount_usd"].to_numpy(dtype="float32"), 0)
    bal = np.maximum(d["account_balance_usd"].to_numpy(dtype="float32"), 0)
    X["log_amt_rel_balance"] = np.log1p(amt) - np.log1p(bal)
    X["velocity_x_risk"] = d["velocity_score"] * d["merchant_risk_score"]
    return X


def add_count_features(tr: pd.DataFrame, te: pd.DataFrame, cols=None):
    """cnt_<col> = how often each value of <col> occurs in train+test (unsupervised, no labels used).

    Synthetic Playground data often reuses generator values; their frequency can carry signal.
    Returns (train_counts, test_counts) as float32 frames.
    """
    cols = cols or NUMS
    both = pd.concat([tr[cols], te[cols]], ignore_index=True)
    out = pd.DataFrame({f"cnt_{c}": both[c].map(both[c].value_counts()).astype(np.float32) for c in cols})
    return out.iloc[:len(tr)].reset_index(drop=True), out.iloc[len(tr):].reset_index(drop=True)


def tree_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Feature frame for LightGBM (categories kept as pandas category)."""
    return pd.concat([d[NUMS + BOOLS + CATS].copy(), add_extra_features(d)], axis=1)


def codes_frame(d: pd.DataFrame) -> pd.DataFrame:
    """Feature frame for XGBoost (categories as integer codes)."""
    X = tree_frame(d)
    for c in CATS:
        X[c] = X[c].cat.codes
    return X


class LinearFE:
    """Spline + one-hot + log features for the linear model (ML task 1.4/1.5)."""

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


# --------------------------------------------------------------------------- folds
def make_folds(train_id, y, n_folds: int = N_FOLDS, seed: int = SEED) -> pd.DataFrame:
    """Same splitter as ML tasks 1.3-1.5: StratifiedKFold(5, shuffle=True, random_state=9) on train order."""
    fold = np.full(len(y), -1, dtype=int)
    skf = StratifiedKFold(n_folds, shuffle=True, random_state=seed)
    for f, (_, va) in enumerate(skf.split(np.zeros(len(y)), y)):
        fold[va] = f
    return pd.DataFrame({"id": train_id, "fold": fold})


def load_folds(train_id, path=None) -> np.ndarray:
    """Read the shared fold file and align it to `train_id`; returns fold index per train row."""
    folds = pd.read_csv(path or find_file("folds_5_seed9.csv"))
    fold = pd.Series(folds["fold"].to_numpy(), index=folds["id"]).reindex(train_id)
    if fold.isna().any():
        raise ValueError(f"{int(fold.isna().sum())} train ids missing from fold file")
    return fold.to_numpy(dtype=int)


def iter_folds(fold: np.ndarray):
    for f in range(fold.max() + 1):
        yield f, np.flatnonzero(fold != f), np.flatnonzero(fold == f)


# --------------------------------------------------------------------------- metrics / blending
def rank01(x) -> np.ndarray:
    return (rankdata(x) - 1) / (len(x) - 1)


def tune_weights(R, y, iters: int = 300, seed: int = 0):
    """Random-walk search for non-negative blend weights maximising ROC-AUC (from ML task 1.4)."""
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


def rank_within_folds(oof, fold) -> np.ndarray:
    """Rank OOF predictions inside each fold (removes scale differences between fold models)."""
    r = np.empty(len(oof), dtype=float)
    for f in range(fold.max() + 1):
        m = fold == f
        r[m] = rank01(oof[m])
    return r


def hill_climb(R, y, max_steps: int = 40, init_best: bool = True):
    """Caruana ensemble selection on rank columns: greedily add (with replacement) the column that most
    improves ROC-AUC of the mean; returns weights summing to 1."""
    k = R.shape[1]
    counts = np.zeros(k)
    if init_best:
        counts[int(np.argmax([roc_auc_score(y, R[:, j]) for j in range(k)]))] = 1
    current = R @ counts
    best = roc_auc_score(y, current / counts.sum())
    for _ in range(max_steps):
        scores = [roc_auc_score(y, (current + R[:, j]) / (counts.sum() + 1)) for j in range(k)]
        j = int(np.argmax(scores))
        if scores[j] <= best + 1e-7:
            break
        best, counts[j], current = scores[j], counts[j] + 1, current + R[:, j]
    return counts / counts.sum(), best


def cv_alpha_gain(base, new, y, fold, alphas=np.round(np.arange(0, 0.51, 0.025), 3)):
    """Blend (1-a)*base + a*new (both ranks); choose a on 4 folds, score on the 5th.
    Returns (fold-mean AUC of base, fold-mean AUC of blend, chosen alphas)."""
    nf = fold.max() + 1
    grid = np.array([[roc_auc_score(y[fold == f], (1 - a) * base[fold == f] + a * new[fold == f])
                      for f in range(nf)] for a in alphas])
    scores, chosen = [], []
    for f in range(nf):
        best = int(np.argmax(grid[:, [g for g in range(nf) if g != f]].mean(1)))
        chosen.append(float(alphas[best]))
        scores.append(grid[best, f])
    return float(grid[0].mean()), float(np.mean(scores)), chosen


def fold_metrics(y, oof, fold) -> dict:
    per_fold = [float(roc_auc_score(y[fold == f], oof[fold == f])) for f in range(fold.max() + 1)]
    return {
        "oof_roc_auc": float(roc_auc_score(y, oof)),
        "oof_average_precision": float(average_precision_score(y, oof)),
        "fold_roc_auc": per_fold,
        "fold_roc_auc_mean": float(np.mean(per_fold)),
        "fold_roc_auc_std": float(np.std(per_fold)),
    }


# --------------------------------------------------------------------------- prediction I/O
def save_predictions(name: str, train_id, oof, test_id, test_pred, y, fold, out_dir="3.ENSEMBLE",
                     extra: dict | None = None) -> dict:
    """Write oof/<name>.csv, test/<name>.csv and results/<name>_metrics.json after sanity checks."""
    oof, test_pred = np.asarray(oof, dtype=float), np.asarray(test_pred, dtype=float)
    assert len(oof) == len(train_id) == len(y) and len(test_pred) == len(test_id)
    assert pd.Series(train_id).is_unique and pd.Series(test_id).is_unique
    assert np.isfinite(oof).all() and np.isfinite(test_pred).all()
    out = Path(out_dir)
    for sub in ("oof", "test", "results"):
        (out / sub).mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": train_id, "pred": oof}).to_csv(out / "oof" / f"{name}.csv", index=False)
    pd.DataFrame({"id": test_id, "pred": test_pred}).to_csv(out / "test" / f"{name}.csv", index=False)
    metrics = {"model": name, **fold_metrics(y, oof, fold), **(extra or {})}
    (out / "results" / f"{name}_metrics.json").write_text(json.dumps(metrics, indent=2))
    return metrics


def find_prediction(kind: str, name: str, roots=("/kaggle/input", "3.ENSEMBLE")) -> Path:
    """Locate `<kind>/<name>.csv` (kind = 'oof' or 'test') under the first root that has it; must be unique."""
    for root in roots:
        if os.path.isdir(root):
            hits = [p for p in Path(root).rglob(f"{name}.csv") if p.parent.name == kind]
            if len(hits) > 1:
                raise FileNotFoundError(f"Ambiguous {kind}/{name}.csv: {hits}")
            if hits:
                return hits[0]
    raise FileNotFoundError(f"{kind}/{name}.csv not found under {roots}")


def load_predictions(names, train_id, test_id, roots=("/kaggle/input", "3.ENSEMBLE")):
    """Load OOF/test predictions for `names`, aligned to train_id/test_id. Returns (OOF, TEST) arrays."""
    oofs, tests = [], []
    for n in names:
        o = pd.read_csv(find_prediction("oof", n, roots))
        t = pd.read_csv(find_prediction("test", n, roots))
        oofs.append(pd.Series(o["pred"].to_numpy(), index=o["id"]).reindex(train_id).to_numpy())
        tests.append(pd.Series(t["pred"].to_numpy(), index=t["id"]).reindex(test_id).to_numpy())
    O, T = np.column_stack(oofs), np.column_stack(tests)
    if np.isnan(O).any() or np.isnan(T).any():
        raise ValueError("Missing ids when aligning predictions")
    return O, T
