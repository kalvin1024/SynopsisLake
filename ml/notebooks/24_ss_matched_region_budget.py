# Databricks notebook source
# MAGIC %md
# MAGIC # 24 · Spend the Source Shape storage budget on a new east-side cohort
# MAGIC
# MAGIC Each method has a total synopsis cap of 403,472 bytes on Authors’ Range and 2,609,480 bytes on Europe.
# MAGIC These are the measured Source Shape storage totals from notebook 23, including the merged histogram,
# MAGIC source metadata and gate. A fine grid uses the largest resolution that fits its own remaining budget.
# MAGIC Actual serialized bytes and unused bytes are reported separately.
# MAGIC
# MAGIC | Method | Answer used when accepted | Synopsis files charged |
# MAGIC |---|---|---|
# MAGIC | `source_shape` | Original merged estimate H, assessed by Source Shape | H, native source summaries, detail packet and gate |
# MAGIC | `fine_reference` | Original merged estimate H, assessed against a fine grid | H, fine count grid and gate |
# MAGIC | `fine_direct` | Fine-grid estimate F, assessed with its count interval | Fine count grid and gate |
# MAGIC | `exact` | Exact count from raw points | Raw data is common to all methods and accounted separately |
# MAGIC
# MAGIC Notebook 23's archived IDs define which eastern queries to exclude. The run keeps the archived H,
# MAGIC region cut and western calibration memberships. Every method and seed uses the same 2,000 fresh
# MAGIC queries per dataset. Each rectangle must lie wholly east of the cut. Gates are frozen on western
# MAGIC labels before exact eastern labels are computed. The relative error tolerance is 25%; the calibration
# MAGIC target is at most 5% violations among accepted answers.

# COMMAND ----------

# MAGIC %pip install lightgbm pyarrow scipy pandas numpy scikit-learn matplotlib

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run in the existing Databricks Git folder
# MAGIC
# MAGIC Pull the updated branch and use the same Java-capable compute and shared Volume as notebook 23.
# MAGIC Leave `prior_results_zip` empty to use the archived 2026-11-26 run in the shared Volume. If the
# MAGIC archive has been moved, enter the full `/Volumes/.../region_budget_results.zip` path to that run's
# MAGIC archive. The preflight checks its lineage before starting an expensive run.
# MAGIC
# MAGIC Select **Run All** with the defaults to measure both datasets on this compute. Set `dataset` to
# MAGIC `europe_range` or `authors_range` to measure one dataset at its fixed cap. Path overrides accept
# MAGIC identical data stored elsewhere. `run_dir` contains `arm_*.txt`
# MAGIC and `src_*.txt`, and `data_dir` contains the raw numbered point files. The repo root contains this
# MAGIC notebook and the Java sources. `math3_path` defaults to the existing shared commons-math3 JAR.
# MAGIC
# MAGIC A new run directory is created under `<volume>/output/ml/ss_matched_region_budget`. The notebook
# MAGIC prints its full path and displays the per-seed table. Upload the **new** `region_budget_results.zip`
# MAGIC from that directory to share the evidence.

# COMMAND ----------

import importlib
import os
from pathlib import Path
import sys


VOLUME = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"
PRIOR_RUN = "region-budget-20261126T131336-6096c254"


def _repo_root():
    try:
        path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
        marker = "/ml/notebooks/"
        if marker in path:
            root = path.split(marker, 1)[0]
            return root if root.startswith("/Workspace/") else "/Workspace" + root
    except Exception:
        pass
    try:
        return str(Path(__file__).resolve().parents[2])
    except NameError:
        cwd = Path.cwd()
        return str(cwd.parents[1] if cwd.name == "notebooks" and cwd.parent.name == "ml" else cwd)


DEFAULTS = {
    "volume": VOLUME,
    "repo_root": "",
    "prior_results_zip": "",
    "output_root": "",
    "dataset": "both",
    "java": "java",
    "javac": "javac",
    "math3_path": "",
    "classpath": "",
    **{f"{dataset}_{key}": "" for dataset in ("authors", "europe")
       for key in ("run_dir", "dataset_dir", "data_dir", "queries_file")},
}
try:
    for name, default in DEFAULTS.items():
        try:
            dbutils.widgets.get(name)  # noqa: F821
        except Exception:
            dbutils.widgets.text(name, default)  # noqa: F821

    def get(name):
        return dbutils.widgets.get(name).strip()  # noqa: F821
except NameError:
    def get(name):
        return os.environ.get(name.upper(), DEFAULTS[name]).strip()


volume = get("volume").rstrip("/")
repo_root = Path(get("repo_root") or _repo_root()).expanduser().resolve()
selection = get("dataset").lower()
storage_caps = {"authors_range": 403472, "europe_range": 2609480}
if selection not in (*storage_caps, "both"):
    raise ValueError("dataset must be both, authors_range or europe_range")
datasets = []
for name, prefix, raw_folder in (
    ("authors_range", "authors", "range_query"),
    ("europe_range", "europe", "europe_range"),
):
    if selection not in ("both", name):
        continue
    item = {
        "dataset": name,
        "run_dir": get(f"{prefix}_run_dir") or f"{volume}/output/ml/{name}",
        "dataset_dir": get(f"{prefix}_dataset_dir") or f"{volume}/data/ml/{name}",
        "data_dir": get(f"{prefix}_data_dir") or f"{volume}/data/{raw_folder}",
    }
    if get(f"{prefix}_queries_file"):
        item["queries_file"] = get(f"{prefix}_queries_file")
    datasets.append(item)

config = dict(
    repo_root=str(repo_root), datasets=datasets,
    prior_results_zip=get("prior_results_zip") or (
        f"{volume}/output/ml/region_budget_comparison/{PRIOR_RUN}/region_budget_results.zip"),
    output_root=get("output_root") or f"{volume}/output/ml/ss_matched_region_budget",
    dataset_budget_bytes={item["dataset"]: storage_caps[item["dataset"]] for item in datasets},
    methods=["source_shape", "fine_reference", "fine_direct"],
    seeds=[0, 1, 2], epsilon=0.25, delta=0.05, merge_rate=0.2,
    n_test_queries=2000, n_calibration_queries=0, calibration_fraction=2 / 7,
    repeats=3, chunk_size=250000, shape_packet_budget_bytes=0,
    java=get("java"), javac=get("javac"), classpath=get("classpath"),
    math3_path=get("math3_path") or f"{volume}/jars/commons-math3-3.6.1.jar",
)

# COMMAND ----------

os.environ.setdefault("MPLCONFIGDIR", "/tmp/synopsislake-region-budget-matplotlib")
helper_path = repo_root / "ml" / "region_budget_comparison.py"
if not helper_path.is_file():
    raise FileNotFoundError(f"Notebook 24 runner is missing at {helper_path}. Check repo_root and the Git branch.")
sys.path.insert(0, str(repo_root))
importlib.invalidate_caches()
from ml import region_budget_comparison  # noqa: E402

loaded_path = Path(region_budget_comparison.__file__).resolve()
if loaded_path != helper_path:
    raise RuntimeError(
        f"Python loaded {loaded_path}; the selected checkout has {helper_path}. "
        "Restart Python and open notebook 24 in the updated Git folder.")

vars(region_budget_comparison).pop("MATCHED_REGION_PROTOCOL", None)
runner = importlib.reload(region_budget_comparison)
if getattr(runner, "MATCHED_REGION_PROTOCOL", None) != 1:
    raise RuntimeError(f"Notebook 24 needs the matched-budget runner. Pull the updated branch at {repo_root}.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validate the previous run and fresh query IDs
# MAGIC
# MAGIC This check reads the archived protocol and IDs and opens the required input paths. It prints the
# MAGIC selected H rate, frozen cut, fixed byte cap and fresh cohort count for each dataset.

# COMMAND ----------

import pandas as pd


def show(table):
    display(table) if "display" in globals() else print(table.to_string(index=False))  # noqa: F821


resolved_config = runner._config(config)
inputs = [runner._dataset_inputs(item, resolved_config) for item in datasets]
for item in inputs:
    paths = [Path(item["queries_file"]), item["H_path"], *item["sources"], *item["raw_paths"]]
    for path in paths:
        with path.open("rb") as stream:
            stream.read(1)
prior_zip = Path(config["prior_results_zip"])
if not prior_zip.is_absolute() or not prior_zip.is_file():
    raise FileNotFoundError(f"Set prior_results_zip to the absolute path of the notebook 23 archive. Checked {prior_zip}")
lineage = runner._matched_prior(resolved_config, inputs)
show(pd.DataFrame([
    dict(dataset=name, actual_merge_rate=entry["actual_rate"], region_cut=entry["region_cut"],
         total_budget_bytes=entry["budget"], previous_queries=len(entry["previous_ids"]),
         fresh_queries=len(entry["test_ids"]), overlapping_queries=len(set(entry["previous_ids"]) & set(entry["test_ids"])))
    for name, entry in lineage.items()
]))
print("Prior archive:", prior_zip)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the Java comparison and read the per-seed results
# MAGIC
# MAGIC Each seed has three timed batches. Each timed batch includes payload loading, scoring, and real
# MAGIC exact fallback. The always-exact baseline uses the same fresh query batch for that dataset.
# MAGIC The table shows the median batch time and seconds saved. The observed violation rate uses only
# MAGIC accepted queries as its denominator. With zero acceptances, the rate is undefined.

# COMMAND ----------

result_dir = Path(runner.run_suite(config)).resolve()
print("Completed:", result_dir)
print("Paper table:", result_dir / "paper_table.csv")
print("Paper Markdown:", result_dir / "paper_table.md")
print("Review archive:", result_dir / "region_budget_results.zip")
show(pd.read_csv(result_dir / "paper_table.csv"))
print((result_dir / "analysis_zh.md").read_text(encoding="utf-8"))
