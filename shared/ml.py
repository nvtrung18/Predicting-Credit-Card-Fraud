"""Tree / linear model fitting used by the ML notebooks (3.1 config from ML task 1.5, plus CatBoost for 3.5)."""
from __future__ import annotations

import numpy as np

SEED = 9

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
XGB_PARAMS = dict(objective="binary:logistic", eval_metric="auc", eta=0.06, max_depth=3, min_child_weight=20,
                  subsample=0.8, colsample_bytree=0.7, reg_lambda=10, tree_method="hist", nthread=-1, seed=SEED)


def fit_lgb(config, Xa, ya, Xb, yb, Xt, seed=None):
    """LightGBM with early stopping on the validation fold. Returns (valid_pred, test_pred, best_iteration)."""
    import lightgbm as lgb
    params = {**LGB_BASE, **LGB_CONFIGS[config]}
    if seed is not None:
        params["seed"] = seed
    m = lgb.train(params, lgb.Dataset(Xa, ya), 6000, valid_sets=[lgb.Dataset(Xb, yb)],
                  callbacks=[lgb.early_stopping(150, verbose=False)])
    it = m.best_iteration
    return m.predict(Xb, num_iteration=it), m.predict(Xt, num_iteration=it), it


def fit_xgb(Xa, ya, Xb, yb, Xt):
    import xgboost as xgb
    da, db, dt = xgb.DMatrix(Xa, ya), xgb.DMatrix(Xb, yb), xgb.DMatrix(Xt)
    m = xgb.train(XGB_PARAMS, da, 6000, evals=[(db, "v")], early_stopping_rounds=150, verbose_eval=False)
    r = (0, m.best_iteration + 1)
    return m.predict(db, iteration_range=r), m.predict(dt, iteration_range=r), m.best_iteration + 1


def fit_linear(Aa, ya, Ab, yb, At):
    from sklearn.linear_model import LogisticRegression
    m = LogisticRegression(C=0.1, max_iter=400).fit(Aa, ya)
    return m.predict_proba(Ab)[:, 1], m.predict_proba(At)[:, 1], int(m.n_iter_[0])


def fit_catboost(Xa, ya, Xb, yb, Xt, cat_features, depth=6, seed=SEED, task_type="CPU"):
    """CatBoost with native categorical handling (ordered target statistics), early stopping on AUC."""
    from catboost import CatBoostClassifier, Pool
    m = CatBoostClassifier(iterations=6000, learning_rate=0.06, depth=depth, l2_leaf_reg=10,
                           eval_metric="AUC", od_type="Iter", od_wait=150, random_seed=seed,
                           task_type=task_type, thread_count=-1, verbose=500, allow_writing_files=False)
    m.fit(Pool(Xa, ya, cat_features=cat_features), eval_set=Pool(Xb, yb, cat_features=cat_features),
          use_best_model=True)
    return (m.predict_proba(Pool(Xb, cat_features=cat_features))[:, 1],
            m.predict_proba(Pool(Xt, cat_features=cat_features))[:, 1], int(m.get_best_iteration()) + 1)
