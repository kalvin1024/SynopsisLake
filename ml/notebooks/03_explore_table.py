# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03 · Look at the training table before training
# MAGIC
# MAGIC One row = one range query answered from one ARM-merged histogram (one merge rate).
# MAGIC
# MAGIC * `gt` exact count, `est` estimate from the merged histogram, `relErr` = |est − gt| / gt (what the model predicts)
# MAGIC * `skew_pred` SynopsisLake's own error estimate (Skewness-Align); `label` = log(max(relErr, 0.001))
# MAGIC * `scale` query size (1 = smallest), `mergeRate` share of the canonical resolution kept (1.0 = no reshaping loss)
# MAGIC * every name in `features.FEATURES` is a model input, computed from the histogram and the query rectangle only
# MAGIC
# MAGIC Runs on the logger output (`ml_xy_pipeline` job: `output/ml/<dataset>` and `data/ml/<dataset>`) by default.
# MAGIC Set `SAMPLE = "dev_fixture"` for the development fixture (look at it, don't report its numbers).

# COMMAND ----------

import os
import sys

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"


def _ml_dir():
    """Folder that holds features.py (this notebook is …/ml/notebooks/03_explore_table)."""
    try:
        nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
        root = nb.split("/ml/notebooks/")[0]
        return (root if root.startswith("/Workspace/") else "/Workspace" + root) + "/ml"
    except Exception:
        return os.path.join(os.getcwd(), "ml")


ML_DIR = _ml_dir()
SAMPLE = "logger"              # "logger" (ErrorPredictionLogExp output), "sample", or "dev_fixture"
DATASET = "authors_range"      # dev fixture: "world" or "europe"

sys.path.insert(0, ML_DIR)
from features import FEATURES, build_table  # noqa: E402

if SAMPLE == "dev_fixture":
    run_dir, data_dir = f"{VOL}/output/ml/dev_fixture/{DATASET}", f"{VOL}/data/ml/dev_fixture/{DATASET}"
elif SAMPLE == "logger":
    run_dir, data_dir = f"{VOL}/output/ml/{DATASET}", f"{VOL}/data/ml/{DATASET}"
else:
    run_dir, data_dir = f"{VOL}/output/ml/sample/{DATASET}", f"{VOL}/data/ml/{DATASET}"

df = build_table(run_dir, data_dir)
print(f"{len(df):,} rows | {df.qId.nunique():,} queries | {df.mergeRate.nunique()} merge rates | "
      f"{df.attrs['dropped_small_exact']:,} rows dropped (exact count < 10)")

# COMMAND ----------

# MAGIC %md ## 1 · What a row looks like

# COMMAND ----------

display(df[["qId", "scale", "mergeRate", "lonRes", "gt", "est", "relErr", "skew_pred",  # noqa: F821
            "area_cells", "bound_rel", "label"]].head(20))

# COMMAND ----------

# MAGIC %md ## 2 · How wrong are the estimates, by merge rate and by query size?

# COMMAND ----------

def error_summary(by):
    return (df.groupby(by)
              .agg(rows=("qId", "size"),
                   median_err=("relErr", "median"),
                   share_over_10pct=("relErr", lambda e: (e > 0.10).mean()),
                   share_over_25pct=("relErr", lambda e: (e > 0.25).mean()))
              .round(3).reset_index())


display(error_summary("mergeRate"))  # noqa: F821
display(error_summary("scale"))  # noqa: F821

# COMMAND ----------

# MAGIC %md ## 3 · Does Skewness-Align's estimate track the real error?
# MAGIC
# MAGIC Rank correlation: +1 means the score orders queries exactly by real error. AUC: chance that a random
# MAGIC query with error above 10% gets a higher score than a random query below it (0.5 = coin flip, 1.0 = perfect).
# MAGIC Two simple scores are included for comparison: the histogram's own min/max bound and "smaller query = worse".

# COMMAND ----------

from scipy.stats import spearmanr  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

bad = df.relErr > 0.10
for name, score in [("skew_pred", df.skew_pred), ("bound_rel", df.bound_rel), ("small query", -df.area_cells)]:
    rho = spearmanr(score, df.relErr).correlation
    print(f"{name:12s} rank correlation {rho:+.2f} | AUC for error > 10%: {roc_auc_score(bad, score):.2f}")
print(f"skew_pred is exactly 0 on {(df.skew_pred == 0).mean():.0%} of rows; "
      f"median real error on those rows: {df.relErr[df.skew_pred == 0].median():.2f}")

# COMMAND ----------

# MAGIC %md ## 4 · One query across all merge rates

# COMMAND ----------

q = df[df.scale == df.scale.max()].qId.sample(1, random_state=1).iloc[0]   # largest query size
display(df[df.qId == q][["mergeRate", "lonRes", "gt", "est", "relErr", "skew_pred"]]  # noqa: F821
        .sort_values("mergeRate"))