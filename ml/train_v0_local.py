"""Model v0 without MLflow or Databricks: same splits, model and gate check as notebooks/02_train_v0.py.

    python ml/train_v0_local.py out/ml/authors_range data/ml/authors_range [report_dir] [--seed 0]

Writes v0_results.csv and feature_importance.csv to report_dir (default <run_dir>/report).
"""
import argparse
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from features import FEATURES, build_table, check_estimates  # noqa: E402

EPS, DELTA = [0.10, 0.25], 0.05
PARAMS = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def split_ids(qids, seed):
    """50/20/30 split of query ids."""
    u = np.random.default_rng(seed).permutation(np.unique(qids))
    n = len(u)
    return set(u[:n // 2]), set(u[n // 2:int(.7 * n)]), set(u[int(.7 * n):])


def cutoff(score, err, eps, delta):
    """Largest cutoff t such that, among queries with score <= t, at most delta have error > eps."""
    order = np.argsort(score, kind="stable")
    s, bad = score[order], err[order] > eps
    viol = np.cumsum(bad) / np.arange(1, len(s) + 1)
    ok = np.where((viol <= delta) & np.r_[s[1:] != s[:-1], True])[0]
    return -np.inf if len(ok) == 0 else s[ok.max()]


def gate(score, err, eps, t):
    """Coverage (share answered from synopses) and violation rate (share of those over eps)."""
    answered = score <= t
    viol = float((err[answered] > eps).mean()) if answered.any() else 0.0
    return float(answered.mean()), viol


def scores(df, pred):
    """Higher = predicted worse. 'oracle' is the actual error (coverage ceiling)."""
    return {"learned": pred, "skew_align": df.skew_pred.to_numpy(), "query_size": -df.area_cells.to_numpy(),
            "edge_bound": df.bound_rel.to_numpy(), "oracle": df.relErr.to_numpy()}


def train(df, dataset, seed=0):
    """Returns (results table, feature importance table) for one dataset."""
    tr_ids, ca_ids, te_ids = split_ids(df.qId, seed)
    va_ids = set(list(sorted(tr_ids))[::10])                    # 10% of training queries for early stopping
    tr = df[df.qId.isin(tr_ids - va_ids)]
    va, ca, te = df[df.qId.isin(va_ids)], df[df.qId.isin(ca_ids)], df[df.qId.isin(te_ids)]

    model = lgb.LGBMRegressor(random_state=seed, **PARAMS)
    model.fit(tr[FEATURES], tr.label, eval_set=[(va[FEATURES], va.label)],
              callbacks=[lgb.early_stopping(100, verbose=False)])

    s_ca, s_te = scores(ca, model.predict(ca[FEATURES])), scores(te, model.predict(te[FEATURES]))
    err_ca, err_te = ca.relErr.to_numpy(), te.relErr.to_numpy()
    results = []
    for eps in EPS:
        for name in s_te:
            t = cutoff(s_ca[name], err_ca, eps, DELTA)
            cov, viol = gate(s_te[name], err_te, eps, t)
            auc = roc_auc_score(err_te > eps, s_te[name])
            results.append({"dataset": dataset, "eps": eps, "score": name, "auc": round(auc, 3),
                            "coverage": round(cov, 3), "violation": round(viol, 3)})
    importance = pd.DataFrame({"feature": FEATURES, "gain": model.booster_.feature_importance("gain")}) \
        .sort_values("gain", ascending=False)
    return pd.DataFrame(results), importance, model.best_iteration_


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("run_dir")
    ap.add_argument("data_dir")
    ap.add_argument("report_dir", nargs="?")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    report_dir = a.report_dir or os.path.join(a.run_dir, "report")

    df = build_table(a.run_dir, a.data_dir)
    print(f"{len(df):,} rows, {df.attrs['dropped_small_exact']:,} dropped (exact < 10), "
          f"estimate gap {check_estimates(df):.2e}")
    res, imp, best = train(df, os.path.basename(os.path.normpath(a.run_dir)), a.seed)
    os.makedirs(report_dir, exist_ok=True)
    res.to_csv(os.path.join(report_dir, "v0_results.csv"), index=False)
    imp.to_csv(os.path.join(report_dir, "feature_importance.csv"), index=False)
    print(f"best iteration {best}")
    print(res.to_string(index=False))
    print(imp.head(10).to_string(index=False))
    print("wrote", report_dir)
