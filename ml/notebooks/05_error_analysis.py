# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 05 · Where the error comes from, and whether Skewness-Align sees it
# MAGIC
# MAGIC Reads the logger output for one dataset (default `authors_range`, written by the `ml_xy_pipeline` job) and answers:
# MAGIC
# MAGIC 1. How much of the error is already there without merging (the source histograms), and how much merging adds.
# MAGIC 2. How error varies with query size and with the number of source files a query touches.
# MAGIC 3. Within one merged synopsis, does Skewness-Align rank bad queries above good ones?
# MAGIC
# MAGIC Every row is one query answered from one ARM-merged histogram. `srcSumEst` is the same query answered from the
# MAGIC source histograms without merging (the authors' `ReadOpt`). Section 1 uses all rows, like the authors' averages;
# MAGIC sections 2 and 3 use the training table (exact count of at least 10).
# MAGIC
# MAGIC Locally: `VOLUME=… REPO_ROOT=… DATA_ROOT=… SAMPLE_ROOT=… DATASET=… python ml/notebooks/05_error_analysis.py`.

# COMMAND ----------

import os
import sys

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"


def _repo_root():
    """Folder that holds ml/ (this notebook is …/ml/notebooks/05_error_analysis)."""
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
            "sample_root": volume + "/output/ml", "dataset": "authors_range"}


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

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

volume, repo, dataset = get("volume"), get("repo_root"), get("dataset")
sys.path.insert(0, os.path.join(repo, "ml"))
from features import build_table  # noqa: E402

run_dir, data_dir = f"{get('sample_root')}/{dataset}", f"{get('data_root')}/{dataset}"
report_dir = f"{run_dir}/report"
os.makedirs(report_dir, exist_ok=True)

rows = pd.read_csv(f"{run_dir}/rows.csv")
rows = rows[rows.method == "ARM"].copy()
df = build_table(run_dir, data_dir)
for t in (rows, df):
    t["nm_err"] = (t.srcSumEst - t["gt"]).abs() / t["gt"]      # error with no merging (sources answered directly)
    t["added"] = t.relErr - t.nm_err                           # what merging added
print(f"{len(rows):,} rows ({rows.qId.nunique():,} queries x {rows.mergeRate.nunique()} merge rates); "
      f"training table {len(df):,} rows")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Error by merge rate: source histograms vs merging
# MAGIC
# MAGIC `arm_mean` is the mean relative error of the merged answer, which is the authors' `ourLH` (shown next to it when
# MAGIC the authors' file is on the volume). `no_merge_mean` is the authors' `ReadOpt`, which doesn't change with merge
# MAGIC rate. `merge_share` is the part of `arm_mean` that merging adds.

# COMMAND ----------

by_rate = (rows.groupby("mergeRate")
           .agg(lonRes=("lonRes", "first"), latRes=("latRes", "first"),
                arm_mean=("relErr", "mean"), no_merge_mean=("nm_err", "mean"),
                arm_median=("relErr", "median"), no_merge_median=("nm_err", "median"),
                share_within_10pct=("relErr", lambda e: (e <= 0.10).mean()),
                share_within_25pct=("relErr", lambda e: (e <= 0.25).mean()),
                alpha=("betaAlpha", "first"))
           .reset_index().sort_values("mergeRate", ascending=False))
by_rate["merge_share"] = 1 - by_rate.no_merge_mean / by_rate.arm_mean
by_rate.insert(0, "mergeRate_pct", (by_rate.pop("mergeRate") * 100).round(1))
authors = f"{volume}/data/range_query/dataDrivenQuery/results/DataDrivenQueryAvgRelError.txt"
if os.path.exists(authors):
    a = pd.read_csv(authors)[["MergeRate(%)", "ourLH", "ReadOpt"]]
    by_rate = by_rate.merge(a.rename(columns={"MergeRate(%)": "mergeRate_pct", "ourLH": "authors_ourLH",
                                              "ReadOpt": "authors_ReadOpt"}), on="mergeRate_pct", how="left")
show(by_rate.round(3))
by_rate.to_csv(f"{report_dir}/error_by_rate.csv", index=False)

fig, ax = plt.subplots(figsize=(7, 4))
ax.plot(by_rate.mergeRate_pct, by_rate.arm_mean, marker="o", label="merged (ARM)")
ax.plot(by_rate.mergeRate_pct, by_rate.no_merge_mean, ls="--", label="no merge (source histograms)")
ax.set_xlabel("merge rate (% of canonical cells kept)")
ax.set_ylabel("mean relative error")
ax.set_title("Error added by merging, on top of the source histograms' error")
ax.legend()
plt.tight_layout()
fig.savefig(f"{report_dir}/error_by_rate.png", dpi=150)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Error by query size and by number of source files touched
# MAGIC
# MAGIC `scale` ranks the distinct query widths (1 = smallest). `n_src_overlap` counts the source files whose extent the
# MAGIC query overlaps; the authors' files are overlapping regional tiles, so a query can touch 1, 2 or 4 of them.

# COMMAND ----------

agg = dict(rows=("qId", "size"), queries=("qId", "nunique"), arm_median=("relErr", "median"),
           no_merge_median=("nm_err", "median"), share_within_25pct=("relErr", lambda e: (e <= 0.25).mean()))
show(df.groupby("scale").agg(**agg).reset_index().round(3))
show(df.groupby("n_src_overlap").agg(**agg).reset_index().round(3))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Does Skewness-Align rank bad queries within one merged synopsis?
# MAGIC
# MAGIC A query is always answered from one merged synopsis, so what matters is ranking queries *within* a merge rate.
# MAGIC Pooled over all rates, a score can look useful just by knowing that coarser synopses are worse.
# MAGIC
# MAGIC * `spearman_err`: rank correlation of Skewness-Align with the real error (+1 = same order)
# MAGIC * `auc_10pct`: chance a random query over 10% error scores higher than one under it (0.5 = coin flip)
# MAGIC * `spearman_added`: rank correlation with the error merging added
# MAGIC * `auc_10pct_bound`: the same AUC for the histogram's own min–max bound, as a simple reference

# COMMAND ----------

within = []
for m, s in df.groupby("mergeRate"):
    bad = s.relErr > 0.10
    within.append({"mergeRate_pct": round(m * 100, 1),
                   "alpha": s.betaAlpha.iloc[0],
                   "spearman_err": spearmanr(s.skew_pred, s.relErr).correlation,
                   "auc_10pct": roc_auc_score(bad, s.skew_pred),
                   "spearman_added": spearmanr(s.skew_pred, s.added).correlation,
                   "auc_10pct_bound": roc_auc_score(bad, s.bound_rel)})
within = pd.DataFrame(within).sort_values("mergeRate_pct", ascending=False)
show(within.round(3))
within.to_csv(f"{report_dir}/skew_align_within_rate.csv", index=False)
print(f"Pooled over all rates: AUC {roc_auc_score(df.relErr > 0.10, df.skew_pred):.3f}; "
      f"rank correlation of Skewness-Align with merge rate {spearmanr(df.skew_pred, df.mergeRate).correlation:+.2f}")
print(f"Skewness-Align exactly 0 on {(df.skew_pred == 0).mean():.1%} of rows; "
      f"alpha ranges {df.betaAlpha.min():.3f} to {df.betaAlpha.max():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to look for (authors' data, from the prototype run)
# MAGIC
# MAGIC * **Source error is a floor.** With no merging, mean error is 0.286 and median 0.186 at every rate. Only 30% of
# MAGIC   queries are within 10% even at full resolution, and 61% within 25%.
# MAGIC * **Merging adds most of the error at low rates.** Mean error rises to 0.95 at 5.5%, where 70% of it is
# MAGIC   added by merging. Section 1's `arm_mean` should match the authors' `ourLH` to about 3 decimals from 100% down to
# MAGIC   21%. Below that it runs lower (0.946 vs 1.047 at 5.5%) unless the logger ran with `-reproduceAuthors true`.
# MAGIC * **Every query is one source cell, shifted 0–22% off that cell's grid.** So query size barely matters (median 0.25
# MAGIC   for the smallest size, 0.21 for the largest), while queries touching 2 or 4 overlapping files are worse
# MAGIC   than those touching one.
# MAGIC * **Skewness-Align barely ranks queries within a synopsis.** Within each merge rate its AUC is 0.57–0.62. Its
# MAGIC   pooled AUC of 0.64 comes mostly from tracking merge rate (rank correlation about −0.70). Alpha stays
# MAGIC   between 1.54 and 1.61 and it is never 0 on this data, unlike the practice data.