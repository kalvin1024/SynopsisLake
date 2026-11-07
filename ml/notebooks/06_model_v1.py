# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "lightgbm==4.6.0",
# ]
# ///
# MAGIC %md
# MAGIC # 06 - Source-feature subset ablation
# MAGIC
# MAGIC The v0/v1 baseline comparison has already been reproduced. This experiment asks whether
# MAGIC **one or two source-level features recover most of the benefit of all seven source features**.
# MAGIC It reuses the feature extraction, LightGBM training, query splits, and calibration logic in
# MAGIC `features_v1.py` and `gate_eval.py`.
# MAGIC
# MAGIC | Default variant | Features | Purpose |
# MAGIC | --- | ---: | --- |
# MAGIC | `v0` | 19 | Baseline without the additional source features |
# MAGIC | `v0_src_bound` | 20 | Add only `src_bound_rel` |
# MAGIC | `v0_src_contrast` | 20 | Add only `src_edge_contrast` |
# MAGIC | `v0_src_bound_contrast` | 21 | Add both features |
# MAGIC | `v1_no_shift` | 26 | Reference using all seven additional source features |
# MAGIC
# MAGIC All five variants exclude merge-shift features. The default run uses **seed 0 and five model fits**.
# MAGIC GitHub Actions and `DATABRICKS_TOKEN` are not required.
# MAGIC
# MAGIC Inputs are read-only. Each run creates a new personal output directory and, by default, snapshots
# MAGIC the required inputs with SHA-256 fingerprints. Models, feature order, splits, and cutoffs are saved.
# MAGIC This is an **offline answer-quality experiment**. It does not execute raw-data fallback or measure
# MAGIC end-to-end query latency. Using fewer model inputs does not itself reduce source-histogram I/O.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.6.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Settings
# MAGIC
# MAGIC Run the next cell to create widgets, then check their values before continuing.
# MAGIC
# MAGIC * `repo_root`: the full path to your personal Git folder.
# MAGIC * `data_root`: the root containing `<dataset>/queries-Ans.txt` and `queries-meta.csv`.
# MAGIC * `sample_root`: the logger-output root containing `<dataset>/rows.csv`, `arm_*.txt`, and `src_*.txt`.
# MAGIC * `output_root`: a personal results directory outside the repo, input directories, and `/Workspace/Shared`.
# MAGIC * `experiment`: a personal MLflow path, such as `/Users/<your-account>/synopsislake-model-ablation`.
# MAGIC * Keep `seeds=0` and `generalization=false` for the first source-feature study.
# MAGIC * `copy_inputs=true` snapshots the inputs. Missing files or permissions must be resolved before training.
# MAGIC * Set `mlflow_enabled=false` to save files without MLflow. Requested MLflow failures are not silently ignored.
# MAGIC
# MAGIC **Existing Databricks widgets retain their previous values after a Git pull.** If `variants` still
# MAGIC shows `v0,v1`, replace that one field with:
# MAGIC
# MAGIC ```text
# MAGIC v0,v0_src_bound,v0_src_contrast,v0_src_bound_contrast,v1_no_shift
# MAGIC ```
# MAGIC
# MAGIC Other settings and personal paths can stay as they were. The plan below should report **5 model fits**.

# COMMAND ----------

import os
from pathlib import Path
import sys

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"
SOURCE_STUDY_VARIANTS = (
    "v0,v0_src_bound,v0_src_contrast,v0_src_bound_contrast,v1_no_shift"
)


def _repo_root():
    try:
        name = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
        root = name.split("/ml/notebooks/")[0]
        return root if root.startswith("/Workspace/") else "/Workspace" + root
    except Exception:
        try:
            return str(Path(__file__).resolve().parents[2])
        except NameError:
            return os.getcwd()


def _defaults():
    repo = _repo_root()
    output, experiment = "", ""
    parts = Path(repo).parts
    if "Users" in parts and len(parts) > parts.index("Users") + 1:
        i = parts.index("Users") + 1
        home = Path(*parts[:i + 1])
        output = str(home / "synopsislake-results")
        experiment = f"/Users/{parts[i]}/synopsislake-model-ablation"
    volume = os.environ.get("VOLUME", VOL)
    return dict(repo_root=repo, data_root=volume + "/data/ml",
                sample_root=volume + "/output/ml", output_root=output,
                dataset="authors_range", seeds="0", variants=SOURCE_STUDY_VARIANTS, generalization="false",
                experiment=experiment, mlflow_enabled="true", copy_inputs="true",
                epsilons="0.25", delta="0.05", max_trees="4000", num_threads="2")


DEFAULTS = _defaults()
try:
    for key, value in DEFAULTS.items():
        dbutils.widgets.text(key, value)  # noqa: F821
    get = dbutils.widgets.get  # noqa: F821
except NameError:
    def get(name):
        return os.environ.get(name.upper(), DEFAULTS[name])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Review the training plan
# MAGIC
# MAGIC The default study has five variants and one seed: **5 fits**.
# MAGIC Using `seeds=0,1,2` produces 15 fits. Selecting `all` produces 10 variants per seed.
# MAGIC Enabling generalization adds four fits per seed.
# MAGIC All variants share the same feature table, query split, LightGBM parameters, and early-stopping rules.

# COMMAND ----------

import importlib

repo = Path(get("repo_root")).resolve()
if not (repo / "ml/model_ablation.py").is_file():
    raise FileNotFoundError("Check repo_root and switch to the experiments/model-ablation branch")
sys.path.insert(0, str(repo / "ml"))
# A session may have cached code before a branch switch or a local edit.
# Re-import this small module set from the explicitly selected Git folder.
for module_name in ("model_ablation", "features_v1", "features", "gate_eval"):
    sys.modules.pop(module_name, None)
importlib.invalidate_caches()
from model_ablation import run_experiment, variants  # noqa: E402

CONFIG = {key: get(key) for key in DEFAULTS}
requested = list(variants()) if CONFIG["variants"] == "all" else [v.strip() for v in CONFIG["variants"].split(",")]
if set(requested) - variants().keys():
    raise ValueError(f"Unknown variants; available choices: {list(variants())}")
seeds = [int(s.strip()) for s in CONFIG["seeds"].split(",")]
fits = len(requested) * len(seeds)
if CONFIG["generalization"].lower() == "true":
    fits += 4 * len(seeds)
print("repo:", repo)
print("variants:", requested, "seeds:", seeds, "model fits:", fits)
print("read-only data:", CONFIG["data_root"])
print("read-only logger exports:", CONFIG["sample_root"])
print("personal output root:", CONFIG["output_root"] or "Not set; enter a personal directory first")
print("MLflow experiment:", CONFIG["experiment"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and evaluate
# MAGIC
# MAGIC Missing volumes, files, or permissions produce an error with the relevant path.
# MAGIC This run uses actual inputs; it does not substitute fabricated data or historical results.
# MAGIC Queries with an exact count below 10 are excluded, matching the existing pipeline.
# MAGIC Results therefore do not assess zero-count or small-count queries.

# COMMAND ----------

result_dir = run_experiment(CONFIG)
print("Results for this run:", result_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read this run's results
# MAGIC
# MAGIC Compare each subset with `v0` and with `v1_no_shift`. A useful subset retains much of the
# MAGIC acceptance improvement of all seven source features at a comparable observed violation rate.
# MAGIC Higher coverage alone is not sufficient if the quality target is exceeded.
# MAGIC
# MAGIC * `coverage_per_rate_mean`: average acceptance after calibrating a separate gate for each merge rate.
# MAGIC * `violation_per_rate_mean`: average risk among rates with accepted answers. **Zero acceptance has undefined risk.**
# MAGIC * `violation_pooled_per_rate_gates_mean`: violations divided by accepted cases across the per-rate gates;
# MAGIC   this uses different weights from the preceding average.
# MAGIC * `gate_details.csv` contains the exact counts and cutoffs. `quality_target_met` is an observation on test data,
# MAGIC   not a statistical guarantee. Individual rates can fail even when the average risk is below 5%.
# MAGIC * Cutoffs use calibration data only. Do not retune them using these test outcomes.
# MAGIC * A single seed has no between-seed standard deviation; empty `std` values are expected.
# MAGIC * `oracle` uses true error and is an offline reference, not a deployable model.
# MAGIC
# MAGIC Each `models/<variant>/seed_<seed>/` directory contains `model.txt`, `schema.json`, `cutoffs.json`,
# MAGIC `cohorts.csv`, and `test_predictions.csv`. Cutoffs are in **predicted log-error score** space, not relative-error units.
# MAGIC `run.json` must have `status=complete` before treating the full run as completed.

# COMMAND ----------

import json
import pandas as pd

summary = pd.read_csv(result_dir / "v1_summary.csv")
details = pd.read_csv(result_dir / "gate_details.csv")
try:
    display(summary)  # noqa: F821
    display(details[(details.score == "learned") & (details.scope == "per_rate")])  # noqa: F821
except NameError:
    print(summary.to_string(index=False))
print("Gate details:", result_dir / "gate_details.csv")
print("Run metadata:", result_dir / "run.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Follow-up experiments
# MAGIC
# MAGIC After this first study, use the same five variants with `seeds=0,1,2` to assess variability (15 fits).
# MAGIC Keep the same input cohort and hyperparameters. Preserve earlier results as independent run directories.
# MAGIC
# MAGIC To reproduce the older group-level ablation, select `v0,v1,v1_no_source,v1_no_shift` explicitly.
# MAGIC `v1_no_source` still retains merge-shift features and therefore still depends on source estimates.
# MAGIC
# MAGIC After editing widgets, rerun from **Review the training plan**. Each execution creates a new result directory.
# MAGIC This pipeline still builds the complete source-feature table before training: selecting fewer model inputs
# MAGIC does not demonstrate reduced I/O, memory use, or query latency. Those require separate systems experiments.
# MAGIC
# MAGIC Optional `generalization=true` tests high-rate to low-rate and west-to-east transfer, with calibration
# MAGIC queries from the evaluated side. Effective row cohorts are saved for every model.
