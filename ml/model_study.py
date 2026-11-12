"""Model study helpers (notebook 10): which model, objective and evaluation split give the best gate?

Everything not listed here matches notebook 06: the v1 inputs, the label log(max(relErr, 1e-3)), exact counts of at
least 10, a 50/20/30 train/calibration/test split, and one cutoff per merge rate chosen on calibration queries so that
at most `delta` of accepted answers exceed the error limit.

Two ways to split the workload:
  query  by query ID, as in notebooks 02-06. The workload has about 5 shifted queries per source-histogram cell, so
         most test queries have a sibling from the same cell in training.
  cell   by source cell: all queries generated from one cell stay in the same split, so every test query is at a
         location the model never saw. This measures how well the model predicts error from the synopsis alone.

Two training objectives:
  reg    regress the log relative error (notebooks 02-06); one model serves every error limit.
  clf    classify "relative error > eps" directly; one model per error limit. The score is P(error > eps).
"""
import time

import lightgbm as lgb
import numpy as np
import pandas as pd

from gate_eval import PARAMS, split_ids

# name: (library, objective). Scores are always "higher = predicted worse".
MODELS = {
    "lgbm_reg": ("lightgbm", "reg"),      # notebook 06 model
    "lgbm_clf": ("lightgbm", "clf"),
    "xgb_reg": ("xgboost", "reg"),
    "xgb_clf": ("xgboost", "clf"),
    "rf_reg": ("random_forest", "reg"),
    "mlp_reg": ("mlp", "reg"),
    "linear_reg": ("linear", "reg"),
}


def source_cells(run_dir, data_dir):
    """qId -> source cell the query was generated from ("<file>_<i>_<j>").

    Each query is one cell of one source histogram, shifted 0-22% of a cell, so its centre stays inside that cell.
    The source file is the one whose cell size matches the query size."""
    q = pd.read_csv(f"{data_dir}/queries-Ans.txt", header=None)
    width, height = (q[1] - q[0]).to_numpy(), (q[3] - q[2]).to_numpy()
    clon, clat = ((q[0] + q[1]) / 2).to_numpy(), ((q[2] + q[3]) / 2).to_numpy()
    grids = []
    k = 0
    while True:
        try:
            lines = open(f"{run_dir}/src_{k}.txt").read().split("\n")
        except FileNotFoundError:
            break
        lon, lat = np.array(lines[1].split(","), float), np.array(lines[2].split(","), float)
        grids.append((lon, lat, np.diff(lon).mean(), np.diff(lat).mean()))
        k += 1
    size_gap = np.array([[abs(w - g[2]) + abs(h - g[3]) for g in grids] for w, h in zip(width, height)])
    src = size_gap.argmin(axis=1)
    i = np.array([np.searchsorted(grids[s][0], x) - 1 for s, x in zip(src, clon)])
    j = np.array([np.searchsorted(grids[s][1], y) - 1 for s, y in zip(src, clat)])
    return pd.DataFrame({"qId": np.arange(len(q)), "cell": [f"{s}_{a}_{b}" for s, a, b in zip(src, i, j)]})


def make_split(df, seed, by="query", train_fraction=1.0):
    """Train / calibration / test frames. `by` is "query" (same split as notebooks 02-06 for a given seed) or
    "cell". `train_fraction` < 1 keeps a random share of the training groups (learning curve); calibration and
    test are unchanged."""
    key = "qId" if by == "query" else "cell"
    tr, ca, te = split_ids(df[key], seed)
    if train_fraction < 1:
        groups = np.array(sorted(tr))
        keep = np.random.default_rng(100 + seed).choice(groups, int(len(groups) * train_fraction), replace=False)
        tr = set(keep)
    return df[df[key].isin(tr)], df[df[key].isin(ca)], df[df[key].isin(te)]


def sibling_share(train, test):
    """Share of test queries whose source cell also has a query in training."""
    cells = set(train.cell)
    t = test.drop_duplicates("qId")
    return float(t.cell.isin(cells).mean())


def sibling_lookup(train, cal, test):
    """Reference score with no model: a query's predicted log error is the mean label of its training siblings
    (same source cell, same merge rate), or the merge rate's median when it has none. It shows how much of a
    model's ranking could come from remembering locations. With the cell split no query has siblings."""
    look = train.groupby(["cell", "mergeRate"]).label.mean().rename("sib").reset_index()
    fallback = train.groupby("mergeRate").label.median().rename("fb").reset_index()

    def score(d):
        x = d[["cell", "mergeRate"]].merge(look, on=["cell", "mergeRate"], how="left").merge(fallback, on="mergeRate")
        return x.sib.fillna(x.fb).to_numpy()
    return score(cal), score(test)


def _held_back(train):
    """Every 10th training query is held back for early stopping (same rule as gate_eval.fit)."""
    ids = np.array(sorted(train.qId.unique()))
    va = train.qId.isin(set(ids[::10]))
    return train[~va], train[va]


def fit_scores(name, train, cal, test, features, seed, eps=None, params=None):
    """Fit model `name` and return (calibration scores, test scores, info). For "clf" models `eps` is required."""
    library, objective = MODELS[name]
    if objective == "clf" and eps is None:
        raise ValueError(f"{name} needs an error limit")
    y = (lambda d: (d.relErr > eps).astype(int)) if objective == "clf" else (lambda d: d.label)
    t0 = time.time()
    info = {"best_iteration": np.nan}
    if library == "lightgbm":
        p = dict(PARAMS, **(params or {}))
        tr, va = _held_back(train)
        model = (lgb.LGBMClassifier if objective == "clf" else lgb.LGBMRegressor)(random_state=seed, **p)
        model.fit(tr[features], y(tr), eval_set=[(va[features], y(va))],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        info["best_iteration"] = model.best_iteration_
    elif library == "xgboost":
        import xgboost as xgb
        p = dict(n_estimators=4000, learning_rate=0.1, max_leaves=63, grow_policy="lossguide", tree_method="hist",
                 min_child_weight=50 if objective == "reg" else 1, subsample=0.8, colsample_bytree=0.8,
                 early_stopping_rounds=100, random_state=seed, n_jobs=-1)
        p.update(params or {})
        tr, va = _held_back(train)
        model = (xgb.XGBClassifier if objective == "clf" else xgb.XGBRegressor)(**p)
        model.fit(tr[features], y(tr), eval_set=[(va[features], y(va))], verbose=False)
        info["best_iteration"] = model.best_iteration
    elif library == "random_forest":
        from sklearn.ensemble import RandomForestRegressor
        p = dict(n_estimators=200, min_samples_leaf=20, max_features=0.5, n_jobs=-1, random_state=seed)
        p.update(params or {})
        model = RandomForestRegressor(**p).fit(train[features], y(train))
    elif library == "mlp":
        from sklearn.neural_network import MLPRegressor
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import QuantileTransformer
        p = dict(hidden_layer_sizes=(256, 128), batch_size=1024, learning_rate_init=1e-3, max_iter=60,
                 early_stopping=True, validation_fraction=0.1, n_iter_no_change=6, random_state=seed)
        p.update(params or {})
        model = make_pipeline(QuantileTransformer(output_distribution="normal", subsample=200_000, random_state=seed),
                              MLPRegressor(**p)).fit(train[features], y(train))
    elif library == "linear":
        from sklearn.linear_model import Ridge
        from sklearn.pipeline import make_pipeline
        from sklearn.preprocessing import QuantileTransformer
        model = make_pipeline(QuantileTransformer(output_distribution="normal", subsample=200_000, random_state=seed),
                              Ridge(alpha=1.0)).fit(train[features], y(train))
    else:
        raise ValueError(library)
    info["fit_seconds"] = time.time() - t0
    if objective == "clf":
        return model.predict_proba(cal[features])[:, 1], model.predict_proba(test[features])[:, 1], info
    return model.predict(cal[features]), model.predict(test[features]), info


def aurc(test, score, eps):
    """Area under the risk-coverage curve at error limit eps, averaged over merge rates (lower is better).

    Within a merge rate, answers are accepted in order of score; risk at coverage k/n is the share of the first k
    accepted answers with error > eps. AURC averages that risk over every coverage level, so it scores the whole
    ranking rather than one cutoff."""
    out = []
    err, rate, score = test.relErr.to_numpy(), test.mergeRate.to_numpy(), np.asarray(score)
    for m in np.unique(rate):
        sel = rate == m
        bad = err[sel][np.argsort(score[sel], kind="stable")] > eps
        out.append(float(np.mean(np.cumsum(bad) / np.arange(1, bad.size + 1))))
    return float(np.mean(out))
