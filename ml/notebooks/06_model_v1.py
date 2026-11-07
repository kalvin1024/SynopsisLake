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
# MAGIC All five variants exclude merge-shift features. This continuation reuses the completed **seed 0**
# MAGIC run, trains **seeds 1 and 2**, and compares all three seeds. It performs **10 new model fits**.
# MAGIC GitHub Actions and `DATABRICKS_TOKEN` are not required.
# MAGIC
# MAGIC Inputs are read-only. Each run creates a new personal output directory and, by default, snapshots
# MAGIC the required inputs with SHA-256 fingerprints. Models, feature order, splits, and cutoffs are saved.
# MAGIC This is an **offline answer-quality experiment**. It does not execute raw-data fallback or measure
# MAGIC end-to-end query latency. Using fewer model inputs does not itself reduce source-histogram I/O.
# MAGIC A previous result is combined only after checking the input fingerprints, feature/training code,
# MAGIC parameters, package versions, and feature order. A mismatch stops the comparison.

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
# MAGIC * Use `seeds=1,2` and `generalization=false` to extend the completed seed 0 study.
# MAGIC * `baseline_run=auto` finds the single compatible completed seed 0 run in `output_root`.
# MAGIC   If more than one matches, enter the exact seed 0 run directory instead. Set it to `none` to run
# MAGIC   a separate experiment without combining earlier results.
# MAGIC * `copy_inputs=true` snapshots the inputs. Missing files or permissions must be resolved before training.
# MAGIC * Set `mlflow_enabled=false` to save files without MLflow. Requested MLflow failures are not silently ignored.
# MAGIC
# MAGIC **Existing Databricks widgets retain their previous values after a Git pull.** Change `seeds`
# MAGIC from `0` to `1,2`. If `variants` is not already the following list, update that field too:
# MAGIC
# MAGIC ```text
# MAGIC v0,v0_src_bound,v0_src_contrast,v0_src_bound_contrast,v1_no_shift
# MAGIC ```
# MAGIC
# MAGIC Other settings and personal paths can stay as they were. The plan below should report **10 model fits**
# MAGIC and identify the previous seed 0 directory. It stops before training if the comparison is incompatible.

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
                dataset="authors_range", seeds="1,2", variants=SOURCE_STUDY_VARIANTS, generalization="false",
                experiment=experiment, mlflow_enabled="true", copy_inputs="true",
                epsilons="0.25", delta="0.05", max_trees="4000", num_threads="2", baseline_run="auto")


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
# MAGIC The continuation uses five variants and two new seeds: **10 fits**. The completed seed 0 models
# MAGIC are reused, not retrained. The comparison contains the same five variants across seeds 0, 1, and 2.
# MAGIC An independent run can be configured with `baseline_run=none`; it will not create a three-seed comparison.
# MAGIC Enabling generalization adds four fits per seed and is not part of the three-seed ablation comparison.
# MAGIC All variants share the same feature table, query split, LightGBM parameters, and early-stopping rules.

# COMMAND ----------

import importlib

repo = Path(get("repo_root")).resolve()
if not (repo / "ml/model_ablation.py").is_file():
    raise FileNotFoundError("Check repo_root and switch to the experiments/model-ablation branch")
sys.path.insert(0, str(repo / "ml"))
# A session may have cached code before a branch switch or a local edit.
# Re-import this small module set from the explicitly selected Git folder.
for module_name in ("model_ablation", "compare_ablation_runs", "features_v1", "features", "gate_eval"):
    sys.modules.pop(module_name, None)
importlib.invalidate_caches()
from model_ablation import run_experiment, variants  # noqa: E402
from compare_ablation_runs import combine_runs, find_compatible_baseline  # noqa: E402

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
baseline = None
if CONFIG["baseline_run"].strip().lower() != "none":
    baseline = find_compatible_baseline(CONFIG)
    print("Compatible seed 0 run:", baseline)
else:
    print("Independent run: no prior results will be combined.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and evaluate
# MAGIC
# MAGIC Missing volumes, files, or permissions produce an error with the relevant path.
# MAGIC This run uses actual inputs; it does not substitute fabricated data or historical results.
# MAGIC Queries with an exact count below 10 are excluded, matching the existing pipeline.
# MAGIC Results therefore do not assess zero-count or small-count queries.

# COMMAND ----------

if {key: get(key) for key in DEFAULTS} != CONFIG:
    raise ValueError("Settings changed; rerun the Review the training plan cell before training")
result_dir = run_experiment(CONFIG)
print("Results for this run:", result_dir)
comparison_dir = None
if baseline is not None:
    comparison_summary, comparison_rates, comparison_dir = combine_runs(baseline, result_dir)
    print("Three-seed comparison:", comparison_dir)

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
# MAGIC When a seed 0 baseline is selected, the tables below combine seeds 0, 1, and 2. The new
# MAGIC `comparison/` directory contains per-seed rows, three-seed summaries, per-rate summaries,
# MAGIC and a provenance manifest. The old seed 0 directory is never modified.
# MAGIC Repeated seeds reuse queries across different splits; their standard deviation is descriptive,
# MAGIC not a confidence interval or a statistical guarantee.

# COMMAND ----------

import json
import pandas as pd

summary = comparison_summary if comparison_dir else pd.read_csv(result_dir / "v1_summary.csv")
details = comparison_rates if comparison_dir else pd.read_csv(result_dir / "gate_details.csv")
try:
    display(summary)  # noqa: F821
    if comparison_dir:
        display(details)  # noqa: F821
    else:
        display(details[(details.score == "learned") & (details.scope == "per_rate")])  # noqa: F821
except NameError:
    print(summary.to_string(index=False))
print("Gate details:", comparison_dir / "combined_gate_details.csv" if comparison_dir else result_dir / "gate_details.csv")
print("Run metadata:", result_dir / "run.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Follow-up experiments
# MAGIC
# MAGIC The combined three-seed comparison describes variability across the existing query splits. It is not
# MAGIC a new independent dataset, and its standard deviation is not a confidence interval.
# MAGIC The original seed 0 and new seed 1/2 runs remain separate, with a derived comparison in the new run.
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
