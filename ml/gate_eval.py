"""Training and gate evaluation shared by notebooks 02 and 06.

A gate answers a query from the merged synopsis when its score is at or below a cutoff, and reads raw data
otherwise. The cutoff is chosen on calibration queries so that at most `delta` of the answered queries exceed
the error limit `eps`, then applied to test queries.

Two ways to apply it:
  pooled     one cutoff over all merge rates together (notebook 02, v0)
  per rate   one cutoff per merge rate, then averaged over rates. This matches deployment: a query is answered
             from one merged synopsis, so the gate only has to rank queries within that synopsis.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

EPS, DELTA = [0.10, 0.25], 0.05
PARAMS = dict(n_estimators=4000, learning_rate=0.1, num_leaves=63, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)


def split_ids(qids, seed):
    """50/20/30 split of query ids (train / calibration / test); a query never appears in two splits."""
    u = np.random.default_rng(seed).permutation(np.unique(qids))
    n = len(u)
    return set(u[:n // 2]), set(u[n // 2:int(.7 * n)]), set(u[int(.7 * n):])


def fit(train, features, seed, params=PARAMS, stop=100):
    """LightGBM on `label`; every 10th training query is held back for early stopping."""
    ids = np.array(sorted(train.qId.unique()))
    va_ids = set(ids[::10])
    va, tr = train[train.qId.isin(va_ids)], train[~train.qId.isin(va_ids)]
    model = lgb.LGBMRegressor(random_state=seed, **params)
    model.fit(tr[features], tr.label, eval_set=[(va[features], va.label)],
              callbacks=[lgb.early_stopping(stop, verbose=False)])
    return model


def cutoff(score, err, eps, delta=DELTA):
    """Largest cutoff t such that, among queries with score <= t, at most delta have error > eps."""
    order = np.argsort(score, kind="stable")
    s, bad = score[order], err[order] > eps
    viol = np.cumsum(bad) / np.arange(1, len(s) + 1)
    ok = np.where((viol <= delta) & np.r_[s[1:] != s[:-1], True])[0]
    return -np.inf if len(ok) == 0 else s[ok.max()]


def gate(score, err, eps, t):
    """Coverage (share answered from the synopsis) and violation rate (share of those over eps)."""
    answered = score <= t
    viol = float((err[answered] > eps).mean()) if answered.any() else 0.0
    return float(answered.mean()), viol


def baseline_scores(df):
    """Higher = predicted worse. `oracle` is the true error and gives the coverage ceiling."""
    s = {"skew_align": df.skew_pred.to_numpy(), "edge_bound": df.bound_rel.to_numpy()}
    if "src_bound_rel" in df:
        s["src_bound"] = df.src_bound_rel.to_numpy()
    s["oracle"] = df.relErr.to_numpy()
    return s


def evaluate(cal, test, cal_scores, test_scores, eps_list=EPS, delta=DELTA):
    """One row per (score, eps): pooled AUC / coverage / violation and their per-merge-rate averages."""
    rows = []
    rates = np.sort(test.mergeRate.unique())
    cal_rate, test_rate = cal.mergeRate.to_numpy(), test.mergeRate.to_numpy()
    err_c, err_t = cal.relErr.to_numpy(), test.relErr.to_numpy()
    for name in test_scores:
        sc, st = cal_scores[name], test_scores[name]
        for eps in eps_list:
            bad = err_t > eps
            r = {"score": name, "eps": eps, "auc": roc_auc_score(bad, st)}
            r["coverage"], r["violation"] = gate(st, err_t, eps, cutoff(sc, err_c, eps, delta))
            aucs, covs, viols = [], [], []
            for m in rates:
                c, t = cal_rate == m, test_rate == m
                if 0 < bad[t].mean() < 1:
                    aucs.append(roc_auc_score(bad[t], st[t]))
                cov, viol = gate(st[t], err_t[t], eps, cutoff(sc[c], err_c[c], eps, delta))
                covs.append(cov)
                viols.append(viol)
            r["auc_per_rate"], r["coverage_per_rate"], r["violation_per_rate"] = (
                float(np.mean(aucs)), float(np.mean(covs)), float(np.mean(viols)))
            rows.append(r)
    return pd.DataFrame(rows)


def run(df, features, seed, train_mask=None, test_mask=None, params=PARAMS):
    """Split by query, train on `features`, evaluate learned + baseline scores on the test queries.

    train_mask / test_mask (boolean Series over df) restrict which rows may be used for training and
    testing, for the generalization tests; calibration rows follow test_mask."""
    tr_ids, ca_ids, te_ids = split_ids(df.qId, seed)
    train_mask = pd.Series(True, index=df.index) if train_mask is None else train_mask
    test_mask = pd.Series(True, index=df.index) if test_mask is None else test_mask
    tr = df[df.qId.isin(tr_ids) & train_mask]
    ca = df[df.qId.isin(ca_ids) & test_mask]
    te = df[df.qId.isin(te_ids) & test_mask]
    model = fit(tr, features, seed, params)
    cal_scores = {"learned": model.predict(ca[features]), **baseline_scores(ca)}
    test_scores = {"learned": model.predict(te[features]), **baseline_scores(te)}
    res = evaluate(ca, te, cal_scores, test_scores)
    res.insert(0, "seed", seed)
    res["best_iteration"] = model.best_iteration_
    importance = pd.DataFrame({"feature": features, "gain": model.booster_.feature_importance("gain")})
    return res, importance, model
