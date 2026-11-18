# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "lightgbm",
# ]
# ///
# MAGIC %md
# MAGIC # 12 · Learned error predictor vs Skewness-Align, offline sweep
# MAGIC
# MAGIC Compares the gate's scores on the logged rows (`rows.csv`, `arm_*`, `src_*`); no Java and no raw-data scans, so none of
# MAGIC the fixed costs of notebook 09 get in the way, and every query and seed can be used.
# MAGIC
# MAGIC 1. **Coverage at equal guarantee, across `eps`.** For each error limit, every score gets a cutoff per merge rate
# MAGIC    calibrated for at most `delta` violations; the plot shows the share answered from the synopsis and the violation
# MAGIC    on test queries. `oracle` (the true error) is the ceiling.
# MAGIC 2. **Risk-coverage curve.** Answer the lowest-scored share of queries from the synopsis and plot the mean real error of
# MAGIC    those answers. A lower curve is better at every operating point, and no cutoff is involved.
# MAGIC
# MAGIC Scores: the learned model (all 28 v1 inputs, trained per seed), Skewness-Align, two simple single-input baselines
# MAGIC (`edge_bound`, `src_bound`) and the oracle. Each seed re-splits queries 50/20/30 (train / calibration / test) and
# MAGIC retrains; tables give the mean and spread over seeds.
# MAGIC
# MAGIC Locally: `VOLUME=… REPO_ROOT=… DATA_ROOT=… SAMPLE_ROOT=… DATASET=… python ml/notebooks/12_gate_eps_sweep.py`.

# COMMAND ----------

# MAGIC %pip install lightgbm

# COMMAND ----------

import os
import sys

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"


def _repo_root():
    try:
        nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
        root = nb.split("/ml/notebooks/")[0]
        return root if root.startswith("/Workspace/") else "/Workspace" + root
    except Exception:
        try:
            return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        except NameError:
            return os.getcwd()


def _defaults(volume):
    return {"volume": volume, "repo_root": _repo_root(), "data_root": volume + "/data/ml",
            "sample_root": volume + "/output/ml", "dataset": "authors_range", "seeds": "0,1,2,3,4",
            "eps_list": "0.05,0.1,0.15,0.2,0.25,0.3,0.4,0.5", "delta": "0.05"}


try:
    for k, v in _defaults(VOL).items():
        dbutils.widgets.text(k, v)  # noqa: F821
    get = dbutils.widgets.get  # noqa: F821
except NameError:
    _env_defaults = _defaults(os.environ.get("VOLUME", VOL))

    def get(name):
        return os.environ.get(name.upper(), _env_defaults[name])


def show(df):
    try:
        display(df)  # noqa: F821
    except NameError:
        print(df.to_string())

# COMMAND ----------

import warnings  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore", message=".*eval_set.*")
sys.path.insert(0, os.path.join(get("repo_root"), "ml"))
from features_v1 import FEATURES_V1, build_table_v1  # noqa: E402
from gate_eval import baseline_scores, evaluate, fit, split_ids  # noqa: E402

dataset = get("dataset")
run_dir, data_dir = f"{get('sample_root')}/{dataset}", f"{get('data_root')}/{dataset}"
report_dir = f"{run_dir}/report"
os.makedirs(report_dir, exist_ok=True)
SEEDS = [int(s) for s in get("seeds").split(",")]
EPS_LIST = [float(e) for e in get("eps_list").split(",")]
DELTA = float(get("delta"))

df = build_table_v1(run_dir, data_dir)
print(f"{len(df):,} rows, {df.qId.nunique():,} queries, {df.mergeRate.nunique()} merge rates, seeds {SEEDS}")

# COMMAND ----------

NAMES = {"learned": "Learned", "skew_align": "Skewness-Align", "edge_bound": "edge_bound", "src_bound": "src_bound",
         "oracle": "Oracle (true error)"}
COVER_GRID = np.round(np.arange(0.02, 1.0001, 0.02), 2)
sweep, curves = [], []

for seed in SEEDS:
    tr_ids, ca_ids, te_ids = split_ids(df.qId, seed)
    tr, ca, te = (df[df.qId.isin(ids)] for ids in (tr_ids, ca_ids, te_ids))
    model = fit(tr, FEATURES_V1, seed)
    cal_scores = {"learned": model.predict(ca[FEATURES_V1]), **baseline_scores(ca)}
    test_scores = {"learned": model.predict(te[FEATURES_V1]), **baseline_scores(te)}
    res = evaluate(ca, te, cal_scores, test_scores, EPS_LIST, DELTA)
    res.insert(0, "seed", seed)
    sweep.append(res)

    # risk-coverage: within each merge rate, answer the lowest-scored share of test queries from the synopsis
    rate, err = te.mergeRate.to_numpy(), te.relErr.to_numpy()
    for name, s in test_scores.items():
        acc = []
        for m in np.unique(rate):
            e = err[rate == m][np.argsort(s[rate == m], kind="stable")]
            run_mean = np.cumsum(e) / np.arange(1, len(e) + 1)
            acc.append([run_mean[max(int(round(c * len(e))), 1) - 1] for c in COVER_GRID])
        curves.append(pd.DataFrame({"seed": seed, "score": name, "coverage": COVER_GRID,
                                    "mean_error": np.mean(acc, axis=0)}))
    print(f"seed {seed}: best iteration {model.best_iteration_}")

sweep = pd.concat(sweep, ignore_index=True)
curves = pd.concat(curves, ignore_index=True)
sweep.to_csv(f"{report_dir}/gate_eps_sweep.csv", index=False)
curves.to_csv(f"{report_dir}/gate_risk_coverage.csv", index=False)

# COMMAND ----------

# MAGIC %md ## 1 · Coverage at equal guarantee
# MAGIC
# MAGIC Per merge rate, averaged over rates, then mean (± std) over seeds. `violation` should stay near `delta`.

# COMMAND ----------

cols = ["coverage_per_rate", "violation_per_rate", "auc_per_rate"]
agg = sweep.groupby(["score", "eps"])[cols].agg(["mean", "std"])
agg.columns = [f"{c}_{s}" for c, s in agg.columns]
agg = agg.reset_index()
table = agg.pivot(index="eps", columns="score", values="coverage_per_rate_mean")[list(NAMES)].round(3)
print("coverage (share answered from the synopsis)")
show(table.reset_index())
vtable = agg.pivot(index="eps", columns="score", values="violation_per_rate_mean")[list(NAMES)].round(3)
print("violation (share of answered queries above eps)")
show(vtable.reset_index())

learned = agg[agg.score == "learned"].set_index("eps")
skew = agg[agg.score == "skew_align"].set_index("eps")
ratio = (learned.coverage_per_rate_mean / skew.coverage_per_rate_mean.replace(0, np.nan)).round(2)
print("learned coverage / Skewness-Align coverage, by eps:", ratio.to_dict())

fig, ax = plt.subplots(1, 2, figsize=(13, 4.5))
style = {"learned": ("#1f5fbf", "-"), "skew_align": ("#8d94a3", "-"), "edge_bound": ("#b58a4a", ":"),
         "src_bound": ("#4a9b8e", ":"), "oracle": ("#222222", "--")}
for name, (color, ls) in style.items():
    a = agg[agg.score == name].sort_values("eps")
    for k, col in enumerate(("coverage_per_rate", "violation_per_rate")):
        ax[k].plot(a.eps, a[f"{col}_mean"], color=color, ls=ls, marker="o", ms=3, label=NAMES[name])
        ax[k].fill_between(a.eps, a[f"{col}_mean"] - a[f"{col}_std"].fillna(0),
                           a[f"{col}_mean"] + a[f"{col}_std"].fillna(0), color=color, alpha=0.15)
ax[0].set_ylabel("share answered from the synopsis")
ax[1].set_ylabel("violation (share of answers above eps)")
ax[1].axhline(DELTA, color="red", lw=0.8, ls="--")
for a_ in ax:
    a_.set_xlabel("error limit eps")
ax[0].legend()
ax[0].set_title(f"Coverage at violation target {DELTA:.0%}")
ax[1].set_title("Violation on test queries")
plt.tight_layout()
fig.savefig(f"{report_dir}/gate_eps_sweep.png", dpi=150)
plt.show()

# COMMAND ----------

# MAGIC %md ## 2 · Risk-coverage curve
# MAGIC
# MAGIC Mean real error of the answers when the gate answers the lowest-scored `coverage` share of queries from the
# MAGIC synopsis (averaged over merge rates and seeds). Lower is better; the oracle is the floor.

# COMMAND ----------

rc = curves.groupby(["score", "coverage"]).mean_error.mean().reset_index()
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, (color, ls) in style.items():
    a = rc[rc.score == name]
    ax.plot(a.coverage, a.mean_error, color=color, ls=ls, label=NAMES[name])
ax.set_xlabel("share answered from the synopsis")
ax.set_ylabel("mean relative error of those answers")
ax.legend()
plt.tight_layout()
fig.savefig(f"{report_dir}/gate_risk_coverage.png", dpi=150)
plt.show()

pick = rc[rc.coverage.round(2).isin([0.1, 0.25, 0.5, 0.75])].pivot(index="coverage", columns="score", values="mean_error")
show(pick[list(NAMES)].round(4).reset_index())
print(f"wrote gate_eps_sweep.csv/.png and gate_risk_coverage.csv/.png to {report_dir}")
