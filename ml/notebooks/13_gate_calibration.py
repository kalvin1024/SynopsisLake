# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "lightgbm",
# ]
# ///
# MAGIC %md
# MAGIC # 13 · Does the gate keep its violation target at every merge rate?
# MAGIC
# MAGIC The gate's cutoff is calibrated per merge rate so that at most `delta` of the answers it accepts have error above
# MAGIC `eps`. End-to-end runs (notebook 09) showed violation above `delta` at some merge rates, while the average over
# MAGIC merge rates in notebook 12 sits near `delta`. This notebook asks whether that is noise or a bias, and what fixing
# MAGIC it costs. It works offline on the logged rows: no Java, no raw scans.
# MAGIC
# MAGIC For every seed the queries are split into train / calibration / test (`50/20/30`, or `50/30/20` with a larger
# MAGIC calibration share), a model is trained, and three cutoff rules are applied per merge rate and `eps`:
# MAGIC
# MAGIC | rule | cutoff |
# MAGIC |---|---|
# MAGIC | `empirical` | largest score cutoff whose calibration violation is at most `delta` (`gate_eval.cutoff`, what the notebooks use) |
# MAGIC | `ucb` | largest cutoff whose **upper confidence bound** (Clopper-Pearson, one-sided, level `conf`) on the calibration violation is at most `delta`; more conservative, most so when few calibration answers back the cutoff |
# MAGIC | `empirical_cal30` | the `empirical` rule with calibration on 30% of queries instead of 20% (same training queries; test is the 20% tail) |
# MAGIC
# MAGIC **How to read the results.** Test violation is pooled over all seeds (all accepted test answers of a merge rate
# MAGIC together), with a 95% confidence interval; `flag` is `HIGH` when the interval lies entirely above `delta`, which
# MAGIC means the miss is more than sampling noise. `coverage` is the share of test queries answered from the synopsis,
# MAGIC so the gap between rules is what meeting the target costs.
# MAGIC
# MAGIC Locally: `VOLUME=… REPO_ROOT=… DATA_ROOT=… SAMPLE_ROOT=… DATASET=… python ml/notebooks/13_gate_calibration.py`.

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
            "sample_root": volume + "/output/ml", "dataset": "authors_range", "variant": "v1_no_skew",
            "seeds": "0,1,2,3,4,5,6,7,8,9", "eps_list": "0.1,0.25", "delta": "0.05", "conf": "0.9",
            "slack": "0.01"}


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
from scipy.stats import beta as beta_dist  # noqa: E402

warnings.filterwarnings("ignore", message=".*eval_set.*")
sys.path.insert(0, os.path.join(get("repo_root"), "ml"))
from features import FEATURES  # noqa: E402
from features_v1 import FEATURES_V1, GROUPS, build_table_v1  # noqa: E402
from gate_eval import cutoff, fit  # noqa: E402

VARIANTS = {"v0": FEATURES, "v0_no_skew": [f for f in FEATURES if f not in GROUPS["skew"]],
            "v1": FEATURES_V1, "v1_no_skew": [f for f in FEATURES_V1 if f not in GROUPS["skew"]]}
variant = get("variant")
if variant not in VARIANTS:
    raise ValueError(f"variant must be one of {list(VARIANTS)}, got {variant!r}")
feats = VARIANTS[variant]

dataset = get("dataset")
run_dir, data_dir = f"{get('sample_root')}/{dataset}", f"{get('data_root')}/{dataset}"
report_dir = f"{run_dir}/report"
os.makedirs(report_dir, exist_ok=True)
SEEDS = [int(s) for s in get("seeds").split(",")]
EPS_LIST = [float(e) for e in get("eps_list").split(",")]
DELTA, CONF, SLACK = float(get("delta")), float(get("conf")), float(get("slack"))

df = build_table_v1(run_dir, data_dir)
rates = np.sort(df.mergeRate.unique())
print(f"{len(df):,} rows, {df.qId.nunique():,} queries, {len(rates)} merge rates | variant {variant} "
      f"({len(feats)} inputs), seeds {SEEDS}, delta {DELTA}, ucb level {CONF}")

# COMMAND ----------


def cutoff_ucb(score, err, eps, delta, conf):
    """Largest cutoff t such that the Clopper-Pearson upper bound (one-sided, level conf) on the violation rate
    among calibration queries with score <= t is at most delta. -inf when there is none."""
    order = np.argsort(score, kind="stable")
    s, bad = score[order], err[order] > eps
    k, n = np.cumsum(bad), np.arange(1, len(s) + 1)
    upper = np.where(n > k, beta_dist.ppf(conf, k + 1, np.maximum(n - k, 1)), 1.0)
    ok = np.where((upper <= delta) & np.r_[s[1:] != s[:-1], True])[0]
    return -np.inf if len(ok) == 0 else s[ok.max()]


def splits(qids, seed):
    """Same permutation as gate_eval.split_ids, so the training queries do not change: train = first 50%;
    calibration = next 20% (default) or 30%; test = the rest (30% or 20%)."""
    u = np.random.default_rng(seed).permutation(np.unique(qids))
    n = len(u)
    return {"default": (set(u[:n // 2]), set(u[n // 2:int(.7 * n)]), set(u[int(.7 * n):])),
            "cal30": (set(u[:n // 2]), set(u[n // 2:int(.8 * n)]), set(u[int(.8 * n):]))}


RULES = [("empirical", "default", lambda s, e, eps: cutoff(s, e, eps, DELTA)),
         ("ucb", "default", lambda s, e, eps: cutoff_ucb(s, e, eps, DELTA, CONF)),
         ("empirical_cal30", "cal30", lambda s, e, eps: cutoff(s, e, eps, DELTA))]

cells = []
for seed in SEEDS:
    sp = splits(df.qId, seed)
    tr = df[df.qId.isin(sp["default"][0])]
    model = fit(tr, feats, seed)
    score = model.predict(df[feats])
    for rule, which, make_cutoff in RULES:
        _, ca_ids, te_ids = sp[which]
        ca, te = df.qId.isin(ca_ids).to_numpy(), df.qId.isin(te_ids).to_numpy()
        for m in rates:
            at = (df.mergeRate == m).to_numpy()
            c, t = ca & at, te & at
            for eps in EPS_LIST:
                cut = make_cutoff(score[c], df.relErr.to_numpy()[c], eps)
                cal_ans = score[c] <= cut
                test_ans = score[t] <= cut
                cells.append({"seed": seed, "rule": rule, "mergeRate": m, "eps": eps, "cutoff": cut,
                              "cal_answered": int(cal_ans.sum()),
                              "cal_bad": int((df.relErr.to_numpy()[c][cal_ans] > eps).sum()),
                              "test_queries": int(t.sum()), "test_answered": int(test_ans.sum()),
                              "test_bad": int((df.relErr.to_numpy()[t][test_ans] > eps).sum())})
    print(f"seed {seed}: best iteration {model.best_iteration_}")
cells = pd.DataFrame(cells)
cells.to_csv(f"{report_dir}/gate_calibration_cells.csv", index=False)

# COMMAND ----------

# MAGIC %md ## Test violation per merge rate
# MAGIC
# MAGIC Pooled over seeds, with a 95% Clopper-Pearson interval. `flag` = `HIGH` when the whole interval is above `delta`.

# COMMAND ----------


def pooled(g):
    ans, bad, tq = g.test_answered.sum(), g.test_bad.sum(), g.test_queries.sum()
    lo = beta_dist.ppf(0.025, bad, ans - bad + 1) if bad > 0 else 0.0
    hi = beta_dist.ppf(0.975, bad + 1, ans - bad) if ans > bad else 1.0
    per_cell = g.test_bad / g.test_answered.replace(0, np.nan)
    return pd.Series({"answered": int(ans), "violation": bad / ans if ans else np.nan, "ci_low": lo, "ci_high": hi,
                      "coverage": ans / tq,
                      "cells_over": float((per_cell > DELTA + SLACK).sum() / per_cell.notna().sum())
                      if per_cell.notna().any() else np.nan})


summary = pd.DataFrame([{"eps": k[0], "rule": k[1], "mergeRate": k[2], **pooled(g).to_dict()}
                        for k, g in cells.groupby(["eps", "rule", "mergeRate"])])
summary["flag"] = np.where(summary.ci_low > DELTA, "HIGH", "")
summary.to_csv(f"{report_dir}/gate_calibration_summary.csv", index=False)

for eps in EPS_LIST:
    s = summary[summary.eps == eps]
    print(f"\neps = {eps}   (delta = {DELTA}, slack {SLACK}; cells_over = share of (seed, merge rate) cells above delta + slack)")
    view = s.pivot(index="mergeRate", columns="rule", values="violation").round(3)
    view = view.join(s.pivot(index="mergeRate", columns="rule", values="coverage").round(3).add_prefix("cov_"))
    view = view.join(s.pivot(index="mergeRate", columns="rule", values="flag").add_prefix("flag_"))
    show(view.sort_index(ascending=False).reset_index())
    avg = s.groupby("rule")[["violation", "coverage", "cells_over"]].mean().round(3)
    print("mean over merge rates:")
    show(avg.reset_index())
    print("merge rates flagged HIGH:", s[s.flag == "HIGH"].groupby("rule").size().to_dict() or "none")

# COMMAND ----------

fig, ax = plt.subplots(len(EPS_LIST), 2, figsize=(13, 4 * len(EPS_LIST)), squeeze=False)
colors = {"empirical": "#1f5fbf", "ucb": "#d1603d", "empirical_cal30": "#4a9b8e"}
for i, eps in enumerate(EPS_LIST):
    for rule, color in colors.items():
        s = summary[(summary.eps == eps) & (summary.rule == rule)].sort_values("mergeRate")
        ax[i, 0].plot(s.mergeRate, s.violation, color=color, marker="o", ms=3, label=rule)
        ax[i, 0].fill_between(s.mergeRate, s.ci_low, s.ci_high, color=color, alpha=0.15)
        ax[i, 1].plot(s.mergeRate, s.coverage, color=color, marker="o", ms=3, label=rule)
    ax[i, 0].axhline(DELTA, color="red", lw=0.8, ls="--")
    ax[i, 0].set_title(f"eps {eps}: test violation (95% interval)")
    ax[i, 1].set_title(f"eps {eps}: coverage")
    for a in ax[i]:
        a.set_xlabel("merge rate")
ax[0, 0].legend()
plt.tight_layout()
fig.savefig(f"{report_dir}/gate_calibration.png", dpi=150)
plt.show()
print(f"wrote gate_calibration_cells.csv, gate_calibration_summary.csv and gate_calibration.png to {report_dir}")
