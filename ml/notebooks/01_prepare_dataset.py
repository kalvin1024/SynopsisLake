# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 · Prepare the query dataset
# MAGIC
# MAGIC Turns the authors' range-query file (`range_query/dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv`,
# MAGIC uploaded to the `data` volume by CI) into `<volume>/data/ml/<dataset>/queries-Ans.txt` and
# MAGIC `queries-meta.csv` (formats: `ml/SCHEMA.md`). The logger task (`ErrorPredictionLogExp`) writes the
# MAGIC matching run folder `<volume>/output/ml/<dataset>/`.
# MAGIC
# MAGIC Locally: `VOLUME=… REPO_ROOT=… DATASET=authors_range python ml/notebooks/01_prepare_dataset.py`.

# COMMAND ----------

import os
import sys

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"


def _repo_root():
    """Folder that holds ml/ (this notebook is …/ml/notebooks/01_prepare_dataset)."""
    try:
        nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
        root = nb.split("/ml/notebooks/")[0]
        return root if root.startswith("/Workspace/") else "/Workspace" + root
    except Exception:
        try:
            return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        except NameError:
            return os.getcwd()


DEFAULTS = {"volume": VOL, "repo_root": _repo_root(), "dataset": "authors_range"}
try:
    for k, v in DEFAULTS.items():
        dbutils.widgets.text(k, v)  # noqa: F821
    get = dbutils.widgets.get  # noqa: F821
except NameError:
    def get(name):
        return os.environ.get(name.upper(), DEFAULTS[name])

# COMMAND ----------

volume, repo, dataset = get("volume"), get("repo_root"), get("dataset")
sys.path.insert(0, os.path.join(repo, "ml"))
from prepare_authors_range import prepare  # noqa: E402

src = f"{volume}/data/range_query/dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv"
out = f"{volume}/data/ml/{dataset}"
meta = prepare(src, out)
print(f"{len(meta):,} queries from {src} -> {out}")
print("queries per scale:", meta.scale.value_counts().sort_index().to_dict())
