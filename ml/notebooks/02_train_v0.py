# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "lightgbm",
# ]
# ///
# MAGIC %md
# MAGIC # 02 · Model v0 (M3)
# MAGIC
# MAGIC Trains the error predictor on the shared sample (notebook 01) and logs one MLflow run per dataset.
# MAGIC
# MAGIC * **Label:** log(max(relative error, 0.001)); queries with exact count below 10 dropped (DEC2).
# MAGIC * **Model:** LightGBM regressor (DEC5), early stopping on a validation slice of the training queries.
# MAGIC * **Splits:** by query id, 50% train / 20% calibration / 30% test (a query never appears in two splits).
# MAGIC * **Gate check:** cutoff picked on calibration queries for at most 5% violations, applied to test queries.
# MAGIC   This is a plain empirical cutoff; the conformal version is C2.
# MAGIC
# MAGIC Needs an ML runtime (LightGBM, MLflow).

# COMMAND ----------

# MAGIC %pip install lightgbm

# COMMAND ----------

# MAGIC %md
# MAGIC **Settings.** Run the next cell on its own first: it adds the input boxes (widgets) at the top of the notebook.
# MAGIC `datasets` is a comma-separated list of folders under `data_root` and `sample_root` (default `authors_range`,
# MAGIC written by notebook 01 and the `ErrorPredictionLogExp` job task).
# MAGIC For the development fixture, set `datasets` to `world,europe`, `data_root` to `…/data/ml/dev_fixture`,
# MAGIC `sample_root` to `…/output/ml/dev_fixture`, and `experiment` to a path in your own user folder, so fixture runs
# MAGIC stay out of the shared experiment. Then run all.
# MAGIC
# MAGIC Each dataset's gate table and feature importance are also written to `<sample_root>/<dataset>/report/`
# MAGIC (`v0_results.csv`, `feature_importance.csv`) for notebook 04.

# COMMAND ----------

import os

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"


def _repo_root():
    """Folder that holds ml/ (this notebook is …/ml/notebooks/02_train_v0)."""
    try:
        nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
        root = nb.split("/ml/notebooks/")[0]
        return root if root.startswith("/Workspace/") else "/Workspace" + root
    except Exception:
        return os.getcwd()


try:
    dbutils.widgets.text("volume", VOL)  # noqa: F821
    dbutils.widgets.text("repo_root", _repo_root())  # noqa: F821
    dbutils.widgets.text("data_root", VOL + "/data/ml")  # noqa: F821  (dev fixture: .../data/ml/dev_fixture)
    dbutils.widgets.text("sample_root", VOL + "/output/ml")  # noqa: F821  (dev fixture: .../output/ml/dev_fixture)
    dbutils.widgets.text("datasets", "authors_range")  # noqa: F821  (dev fixture: world,europe)
    dbutils.widgets.text("experiment", "/Shared/synopsislake-error-predictor")  # noqa: F821
    dbutils.widgets.text("seed", "0")  # noqa: F821
    get = dbutils.widgets.get  # noqa: F821
except NameError:
    def get(name):
        return os.environ[name.upper()]

# COMMAND ----------

import sys
import tempfile

import lightgbm as lgb
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

volume, repo, seed = get("volume"), get("repo_root"), int(get("seed"))
data_root, sample_root = get("data_root"), get("sample_root")
sys.path.insert(0, os.path.join(repo, "ml"))
from features import FEATURES, build_table, check_estimates  # noqa: E402

mlflow.set_experiment(get("experiment"))
DATASETS = [d.strip() for d in get("datasets").split(",") if d.strip()]
EPS, DELTA = [0.10, 0.25], 0.05
PARAMS = dict(n_estimators=2000, learning_rate=0.05, num_leaves=63, min_child_samples=50,
              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
print("features from", os.path.join(repo, "ml"), "| data", data_root, "| sample", sample_root,
      "| datasets", DATASETS, "| experiment", get("experiment"))

# COMMAND ----------


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


# COMMAND ----------

results = []
for ds in DATASETS:
    ds_results = []
    df = build_table(f"{sample_root}/{ds}", f"{data_root}/{ds}")
    check_estimates(df)
    tr_ids, ca_ids, te_ids = split_ids(df.qId, seed)
    va_ids = set(list(sorted(tr_ids))[::10])                    # 10% of training queries for early stopping
    tr = df[df.qId.isin(tr_ids - va_ids)]
    va, ca, te = df[df.qId.isin(va_ids)], df[df.qId.isin(ca_ids)], df[df.qId.isin(te_ids)]

    with mlflow.start_run(run_name=f"v0-{ds}-seed{seed}"):
        mlflow.log_params({"dataset": ds, "seed": seed, "features": ",".join(FEATURES), "min_exact": 10,
                           "rows_train": len(tr), "rows_test": len(te),
                           "dropped_small_exact": df.attrs["dropped_small_exact"], **PARAMS})
        model = lgb.LGBMRegressor(random_state=seed, **PARAMS)
        model.fit(tr[FEATURES], tr.label, eval_set=[(va[FEATURES], va.label)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
        mlflow.log_metric("best_iteration", model.best_iteration_)

        s_ca, s_te = scores(ca, model.predict(ca[FEATURES])), scores(te, model.predict(te[FEATURES]))
        err_ca, err_te = ca.relErr.to_numpy(), te.relErr.to_numpy()
        for eps in EPS:
            tag = f"eps{int(eps * 100)}"
            for name in s_te:
                t = cutoff(s_ca[name], err_ca, eps, DELTA)
                cov, viol = gate(s_te[name], err_te, eps, t)
                auc = roc_auc_score(err_te > eps, s_te[name])
                mlflow.log_metrics({f"{name}_auc_{tag}": auc, f"{name}_coverage_{tag}": cov,
                                    f"{name}_violation_{tag}": viol})
                ds_results.append({"dataset": ds, "eps": eps, "score": name, "auc": round(auc, 3),
                                   "coverage": round(cov, 3), "violation": round(viol, 3)})

        importance = pd.DataFrame({"feature": FEATURES, "gain": model.booster_.feature_importance("gain")}) \
            .sort_values("gain", ascending=False)
        with tempfile.TemporaryDirectory() as tmp:
            model.booster_.save_model(os.path.join(tmp, "model.txt"))      # LightGBM text format (DEC9)
            importance.to_csv(os.path.join(tmp, "importance.csv"), index=False)
            mlflow.log_artifacts(tmp)

    report_dir = f"{sample_root}/{ds}/report"                          # read by notebook 04 and CI
    os.makedirs(report_dir, exist_ok=True)
    pd.DataFrame(ds_results).to_csv(f"{report_dir}/v0_results.csv", index=False)
    importance.to_csv(f"{report_dir}/feature_importance.csv", index=False)
    print("wrote", report_dir)
    results += ds_results

table = pd.DataFrame(results).pivot_table(index=["dataset", "eps", "score"],
                                          values=["auc", "coverage", "violation"])
try:
    display(table.reset_index())  # noqa: F821
except NameError:
    print(table.to_string())