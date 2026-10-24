# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 04 · Data report
# MAGIC
# MAGIC Writes `<sample_root>/<dataset>/report/report.md` (input checks, error by merge rate and query scale,
# MAGIC simple baselines, and the v0 results from notebook 02 when present) and returns its path as the
# MAGIC notebook exit value. CI downloads the file and shows it on the GitHub Actions run summary.
# MAGIC
# MAGIC Locally: `VOLUME=… REPO_ROOT=… DATA_ROOT=… SAMPLE_ROOT=… DATASET=… python ml/notebooks/04_data_report.py`.

# COMMAND ----------

import os
import sys

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"


def _repo_root():
    """Folder that holds ml/ (this notebook is …/ml/notebooks/04_data_report)."""
    try:
        nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
        return "/Workspace" + nb.split("/ml/notebooks/")[0]
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

# COMMAND ----------

volume, repo, dataset = get("volume"), get("repo_root"), get("dataset")
sys.path.insert(0, os.path.join(repo, "ml"))
from report import build_report  # noqa: E402

run_dir = f"{get('sample_root')}/{dataset}"
data_dir = f"{get('data_root')}/{dataset}"
report_dir = f"{run_dir}/report"
md = build_report(run_dir, data_dir,
                  authors_avg=f"{volume}/data/range_query/dataDrivenQuery/results/DataDrivenQueryAvgRelError.txt",
                  v0_csv=f"{report_dir}/v0_results.csv", importance_csv=f"{report_dir}/feature_importance.csv")
os.makedirs(report_dir, exist_ok=True)
path = f"{report_dir}/report.md"
with open(path, "w") as f:
    f.write(md)
print(md)
print("wrote", path)

# COMMAND ----------

try:
    dbutils.notebook.exit(path)  # noqa: F821
except NameError:
    pass
