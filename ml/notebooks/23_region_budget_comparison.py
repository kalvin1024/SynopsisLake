# Databricks notebook source
# MAGIC %md
# MAGIC # 23 · Unseen-region comparison under one total storage budget
# MAGIC
# MAGIC This experiment reuses notebook 22's Source Shape summaries, the team's full-v1 model family, and 09b's
# MAGIC shared Java batch/fallback approach. It does **not** overwrite those notebooks or reuse models trained on
# MAGIC the test region. It compares, on the same eastern query rectangles:
# MAGIC
# MAGIC | Method | Candidate returned when accepted | Scoring information |
# MAGIC |---|---|---|
# MAGIC | `source_shape` | Original ARM answer H | Notebook 22's source-shape reference R |
# MAGIC | `v1` | Original ARM answer H | Full 28-feature v1, retrained on western TRAIN only |
# MAGIC | `fine_direct` | Finer count-grid answer F | Its count-derived worst-relative-error bound |
# MAGIC | `fine_reference` (optional control) | Original ARM answer H | The same role as 22: discrepancy between H and a fine-grid reference |
# MAGIC | `exact` | Exact raw-point count | No synopsis gate |
# MAGIC
# MAGIC **Region:** the longitude cut uses query geometry only. TRAIN and calibration rectangles must lie wholly
# MAGIC west of the cut; test rectangles lie wholly east. Rectangles crossing the cut are excluded and recorded.
# MAGIC Unlike the older region test, **eastern labels never train the model, select storage budgets, or choose
# MAGIC cutoffs**. All methods use the same western calibration IDs and empirical tied-prefix cutoff rule at
# MAGIC epsilon=25%, delta=5%. That source-side target need not hold on the new region; test violations remain visible.
# MAGIC The v1 control uses the team's28 features/learner refit on this one H and western TRAIN; it is not the
# MAGIC original saved joint-multi-rate model. Source metadata may describe the full indexed raw dataset,
# MAGIC but eastern QUERY labels never select a method or operating point.
# MAGIC
# MAGIC **Budget:** one common total byte ceiling includes each method's required H/count grid, source summaries,
# MAGIC model/properties, SS packet and gate data. Actual bytes and unused bytes are reported; there is no padding
# MAGIC to pretend payloads are equal. Fine direct does not retain H; fine reference must pay for H as well as F.
# MAGIC A zero budget asks the runner to derive a common ceiling from western model sizes and minimum required
# MAGIC representations before consulting eastern labels. It is not a test-selected quality optimum.
# MAGIC
# MAGIC **Timing:** median whole batches include Java startup, reading the required summaries/models, raw loading,
# MAGIC scoring, actual exact fallback and output conversion. Raw points are loaded once per JVM and rejected
# MAGIC queries scan those real in-memory arrays. OS cache is uncontrolled. These are batch measurements, not
# MAGIC independent cloud-storage requests. Preparation, training and calibration costs are recorded separately.

# COMMAND ----------

# MAGIC %pip install lightgbm pyarrow scipy pandas numpy scikit-learn matplotlib

# COMMAND ----------

# MAGIC %md
# MAGIC ## Settings
# MAGIC
# MAGIC Start with `dataset=authors_range` or `europe_range` to inspect one dataset. `both` compares the two
# MAGIC datasets under the suite's common budget and writes a dataset-contrast table. The formal defaults are
# MAGIC seeds `0,1,2`, 2,000 eastern queries and three timing repeats. For an explicitly labelled execution check,
# MAGIC use seeds `0`, 200 queries and one repeat; those preview numbers are not the formal three-seed result.
# MAGIC
# MAGIC `run_dir` means the existing histogram export containing `arm_*.txt` and `src_*.txt`, **not** the Git folder.
# MAGIC The requested merge rate selects the nearest existing H without using test accuracy. The source raw
# MAGIC files and stored query geometry must describe that export. Leave a region cut blank to use the existing
# MAGIC notebook 06 geometry-only center-extent midpoint, with the stricter whole-rectangle membership above.
# MAGIC
# MAGIC A JDK with `java`/`javac` and commons-math3 are required. The runner compiles the new entry point and
# MAGIC its dependencies directly from this checkout into a local temporary cache; no Maven build or replacement
# MAGIC of the existing Volume JAR is needed. Alternatively, an explicit `classpath` must contain the new class
# MAGIC and its dependencies. Databricks keeps widget values across pulls. Leave `repo_root` blank to
# MAGIC use this notebook's Git folder or enter the path to the checkout you want to use.

# COMMAND ----------

import os
from pathlib import Path

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"


def _repo_root():
    try:
        path = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
        root = path.split("/ml/notebooks/")[0]
        return root if root.startswith("/Workspace/") else "/Workspace" + root
    except Exception:
        try:
            return str(Path(__file__).resolve().parents[2])
        except NameError:
            cwd = Path.cwd()
            if cwd.name == "notebooks" and cwd.parent.name == "ml":
                return str(cwd.parents[1])
            return str(cwd)


DEFAULTS = {
    "volume": VOL,
    "repo_root": _repo_root(),
    "dataset": "both",
    "authors_run_dir": "",
    "authors_dataset_dir": "",
    "authors_data_dir": "",
    "authors_queries_file": "",
    "authors_region_cut": "",
    "europe_run_dir": "",
    "europe_dataset_dir": "",
    "europe_data_dir": "",
    "europe_queries_file": "",
    "europe_region_cut": "",
    "seeds": "0,1,2",
    "merge_rate": "0.2",
    "eps": "0.25",
    "delta": "0.05",
    "n_test_queries": "2000",
    "n_calibration_queries": "0",
    "max_train_queries": "0",
    "repeats": "3",
    "total_budget_bytes": "0",
    "shape_packet_budget_bytes": "0",
    "include_fine_reference": "true",
    "chunk_size": "250000",
    "java": "java",
    "math3_path": "",
    "classpath": "",
    "output_root": "",
}
try:
    for name, value in DEFAULTS.items():
        try:
            dbutils.widgets.get(name)  # noqa: F821
        except Exception:
            dbutils.widgets.text(name, value)  # noqa: F821
    get = dbutils.widgets.get  # noqa: F821
except NameError:
    def get(name):
        return os.environ.get(name.upper(), DEFAULTS[name])


def _integer(name):
    value = int(str(get(name)).strip())
    if value < 0:
        raise ValueError(name + " must be nonnegative")
    return value


def _boolean(name):
    value = str(get(name)).strip().lower()
    if value not in ("true", "false"):
        raise ValueError(name + " must be true or false")
    return value == "true"


volume = str(get("volume")).strip().rstrip("/")
repo_root = Path(str(get("repo_root")).strip() or _repo_root()).expanduser().resolve()
selection = str(get("dataset")).strip().lower()
if selection not in ("authors_range", "europe_range", "both"):
    raise ValueError("dataset must be authors_range, europe_range or both")
datasets = []
for name, prefix, default_raw in (
    ("authors_range", "authors", f"{volume}/data/range_query"),
    ("europe_range", "europe", f"{volume}/data/europe_range"),
):
    if selection not in ("both", name):
        continue
    item = {
        "dataset": name,
        "run_dir": str(get(prefix + "_run_dir")).strip() or f"{volume}/output/ml/{name}",
        "dataset_dir": str(get(prefix + "_dataset_dir")).strip() or f"{volume}/data/ml/{name}",
        "data_dir": str(get(prefix + "_data_dir")).strip() or default_raw,
    }
    queries_file = str(get(prefix + "_queries_file")).strip()
    if queries_file:
        item["queries_file"] = queries_file
    region_cut = str(get(prefix + "_region_cut")).strip()
    if region_cut:
        item["region_cut"] = float(region_cut)
    datasets.append(item)

seeds = [int(value.strip()) for value in str(get("seeds")).split(",") if value.strip()]
if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
    raise ValueError("seeds must contain unique nonnegative integers")
config = {
    "repo_root": str(repo_root),
    "datasets": datasets,
    "seeds": seeds,
    "merge_rate": float(get("merge_rate")),
    "epsilon": float(get("eps")),
    "delta": float(get("delta")),
    "n_test_queries": _integer("n_test_queries"),
    "n_calibration_queries": _integer("n_calibration_queries"),
    "max_train_queries": _integer("max_train_queries"),
    "calibration_fraction": 2 / 7,
    "train_min_exact": 10,
    "total_budget_bytes": _integer("total_budget_bytes"),
    "shape_packet_budget_bytes": _integer("shape_packet_budget_bytes"),
    "repeats": _integer("repeats"),
    "chunk_size": _integer("chunk_size"),
    "include_fine_reference": _boolean("include_fine_reference"),
    "java": str(get("java")).strip(),
    "math3_path": str(get("math3_path")).strip() or f"{volume}/jars/commons-math3-3.6.1.jar",
    "classpath": str(get("classpath")).strip(),
    "output_root": str(get("output_root")).strip() or f"{volume}/output/ml/region_budget_comparison",
}

# COMMAND ----------

import importlib
import json
import sys

os.environ.setdefault("MPLCONFIGDIR", "/tmp/synopsislake-region-budget-matplotlib")

helper_path = repo_root / "ml" / "region_budget_comparison.py"
if not helper_path.is_file():
    raise FileNotFoundError(f"Notebook 23 runner is missing at {helper_path}. Set repo_root to the updated Git folder.")
sys.path.insert(0, str(repo_root))
importlib.invalidate_caches()
from ml import region_budget_comparison  # noqa: E402

loaded_path = Path(region_budget_comparison.__file__).resolve()
if loaded_path != helper_path:
    raise RuntimeError(
        f"Python loaded {loaded_path}, but repo_root points to {helper_path}. "
        "Open Notebook 23 in the updated Git folder, set repo_root for that folder, and restart Python.")

# A pull can update the file while this session retains the old Python module.
vars(region_budget_comparison).pop("MODEL_EXPORT_MODE", None)
region_budget_comparison = importlib.reload(region_budget_comparison)
if getattr(region_budget_comparison, "MODEL_EXPORT_MODE", None) != "single-write":
    raise RuntimeError(
        f"Notebook 23 found an older runner at {helper_path}. Pull experiments/region-budget-comparison "
        "in that Git folder and check the repo_root widget before rerunning.")

print("Notebook 23 code:", helper_path)
print(json.dumps(config, indent=2))
print("Western labels only: model fitting, budget selection and calibration.")
print("Eastern query geometry is frozen before fitting; eastern labels are evaluated after the gates are frozen.")
print("All-query results are primary; truth>=10 results are a separate evaluation slice.")
if seeds != [0, 1, 2] or config["n_test_queries"] != 2000 or config["repeats"] != 3:
    print("Non-default run: label it with these actual seeds/query counts/repeats, rather than the formal default protocol.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check shared data access and model export before training
# MAGIC
# MAGIC Git folders store the experiment code. The existing raw data, query files and histogram exports live
# MAGIC in the shared Unity Catalog Volume. A personal Git folder in the same Databricks workspace can read
# MAGIC those paths when the current user/compute has access. Use the same workspace and compute that ran
# MAGIC Europe notebook09b. This check opens each required input without parsing query labels, and stops before
# MAGIC expensive preparation if a path is missing or unreadable. A small synthetic model also checks that
# MAGIC the model writer works in the configured output directory.

# COMMAND ----------

import pandas as pd

input_checks = []
for dataset_config in config["datasets"]:
    try:
        resolved = region_budget_comparison._dataset_inputs(dataset_config, config)
        paths = [Path(resolved["queries_file"]), resolved["H_path"],
                 *resolved["sources"], *resolved["raw_paths"]]
        for path in paths:
            with path.open("rb") as stream:
                stream.read(1)  # Access check only; no query labels are parsed.
        input_checks.append(dict(dataset=resolved["dataset"], status="READABLE",
            raw_source_files=len(resolved["raw_paths"]), actual_merge_rate=resolved["actual_rate"],
            raw_data_dir=resolved["data_dir"], queries_file=resolved["queries_file"], detail=""))
    except (OSError, ValueError, KeyError) as error:
        input_checks.append(dict(dataset=dataset_config["dataset"], status="MISSING_OR_UNREADABLE",
            raw_source_files=None, actual_merge_rate=None,
            raw_data_dir=dataset_config.get("data_dir", ""),
            queries_file=dataset_config.get("queries_file", ""), detail=str(error)))
input_check_table = pd.DataFrame(input_checks)
display(input_check_table) if "display" in globals() else print(input_check_table.to_string(index=False))
if not input_check_table.status.eq("READABLE").all():
    raise RuntimeError(
        "Input access check failed before training. Check the displayed Volume paths and use the same "
        "Databricks workspace/compute as the earlier Europe09b run. If a path exists but is denied, "
        "ask the Volume owner to check USE CATALOG, USE SCHEMA and READ VOLUME for this user. "
        "Keep the existing shared dataset in its Volume; the Git folder only needs the code.")
print("All selected dataset inputs are readable.")
region_budget_comparison.check_model_export(config["output_root"])
print("Model export check: PASS (single write)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Execute the frozen comparison
# MAGIC
# MAGIC This is a new experiment, not a replay of the old query-split numbers. It prepares the required summaries,
# MAGIC trains v1 only on western TRAIN, chooses the common total byte ceiling without eastern labels, calibrates
# MAGIC every score on the same western calibration queries, freezes the gates, and then evaluates/times the
# MAGIC eastern queries. It records actual payload files and hashes. Existing completed runs are not overwritten.

# COMMAND ----------

result_dir = Path(region_budget_comparison.run_suite(config)).resolve()
print("Completed:", result_dir)
print("Summary:", result_dir / "summary.csv")
print("Dataset contrast:", result_dir / "dataset_contrast.csv")
print("Payload accounting:", result_dir / "payload_costs.csv")
print("Review archive:", result_dir / "region_budget_results.zip")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read acceptance, violations, bytes and time together
# MAGIC
# MAGIC Coverage is the fraction answered from the method's candidate histogram. A direct fine-grid candidate
# MAGIC differs from H; SS, v1 and the fine-reference control assess H. Accepted violations are accepted answers
# MAGIC with error above 25%, divided by accepted count. Zero accepted answers have undefined violation, not 0%.
# MAGIC Read observed risk with accepted/bad counts; the 5% source-calibration target is not a guarantee on east.
# MAGIC
# MAGIC Each dataset has its own always-exact baseline. Time saved is that dataset's exact batch median minus
# MAGIC the method median, so negative savings mean slower execution. Compare fallback counts, raw-load and
# MAGIC raw-scan phases before attributing cross-dataset differences to data size. Seeds reuse eastern geometries
# MAGIC and must not be counted as independent query datasets.

# COMMAND ----------

import pandas as pd


def show(table):
    display(table) if "display" in globals() else print(table.to_string(index=False))  # noqa: F821


for filename in ("summary.csv", "dataset_contrast.csv", "payload_costs.csv"):
    file = result_dir / filename
    if file.is_file():
        print("\n" + filename)
        show(pd.read_csv(file))
print("Full run evidence:", result_dir)
print("Timing and phase records:", result_dir / "timing.csv", result_dir / "phase_timing.csv")
print("Frozen region memberships:", result_dir / "region_splits.csv")
print("Frozen western calibration gates:", result_dir / "frozen_gates.csv")
print("Upload region_budget_results.zip after a completed run for analysis; it excludes raw data and large model weights.")
