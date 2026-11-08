# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "lightgbm==4.6.0",
# ]
# ///
# MAGIC %md
# MAGIC # 07 - Leave source features out of full v1
# MAGIC
# MAGIC Notebook 06 adds source features to v0. This complementary experiment asks whether a source
# MAGIC feature still provides useful information when the model already has the other v1 features.
# MAGIC Each reduced model is **retrained** using its remaining features; no feature is replaced with zero.
# MAGIC
# MAGIC | Default variant | Features | Removed from full v1 |
# MAGIC | --- | ---: | --- |
# MAGIC | `v1` | 28 | None |
# MAGIC | `v1_no_src_contrast` | 27 | `src_edge_contrast` only |
# MAGIC | `v1_no_src_bound` | 27 | `src_bound_rel` only |
# MAGIC
# MAGIC Every variant retains both `merge_shift` and `abs_merge_shift`. The 26-feature `v1_no_shift`
# MAGIC from notebook 06 is a different model and cannot be reused as full v1.
# MAGIC This study uses the same seeds **0, 1, 2**, query splits, LightGBM parameters, early stopping,
# MAGIC empirical calibration, and error definitions as notebook 06. There are at most **9 new fits**.
# MAGIC A compatible saved full-v1 model can be reused separately for each seed.
# MAGIC
# MAGIC This is an offline quality study. Fewer model features do not demonstrate reduced source I/O
# MAGIC or query latency. The existing 5% empirical calibration rule does not guarantee 5% test risk.
# MAGIC Previously inspected test results remain exploratory evidence, not a new independent holdout.

# COMMAND ----------

# MAGIC %pip install lightgbm==4.6.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## Settings
# MAGIC
# MAGIC Run the next cell, then check the input and personal output paths.
# MAGIC Leave `seeds=0,1,2`, `epsilons=0.25`, `delta=0.05`, and the default three variants unchanged
# MAGIC for this study. `max_trees=4000` and `num_threads=2` match the earlier experiment.
# MAGIC
# MAGIC Keep the same input roots as notebook 06 and ensure their contents are unchanged.
# MAGIC Each study fingerprints its actual inputs; compare `input_manifest.json` when comparing studies.
# MAGIC A candidate reused full-v1 model must match the current inputs exactly.
# MAGIC
# MAGIC * `baseline_run=auto`: search personal results for compatible full-v1 models, using provenance
# MAGIC   rather than test performance. Reuse available seeds and train the missing ones.
# MAGIC * `baseline_run=none`: train all nine models in a new run.
# MAGIC * An explicit `baseline_run` directory can provide full v1 for one or more seeds. An incompatible
# MAGIC   explicit run stops before training. An ambiguous automatic match requires an explicit directory.
# MAGIC * `copy_inputs=true`: save input snapshots and fingerprints in the new result directory.
# MAGIC * `mlflow_enabled=true`: log to your personal experiment. Set false to save files without MLflow.
# MAGIC
# MAGIC To test removal of both features instead, use
# MAGIC `v1,v1_no_src_contrast,v1_no_src_bound_contrast` in `variants`.
# MAGIC The last variant has 26 features **including both shift features**, unlike `v1_no_shift`.
# MAGIC Databricks retains existing widget values. Review them before continuing.

# COMMAND ----------

import os
from pathlib import Path
import sys


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
        output = str(Path(*parts[:i + 1]) / "synopsislake-results")
        experiment = f"/Users/{parts[i]}/synopsislake-v1-feature-ablation"
    volume = os.environ.get("VOLUME", "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake")
    return dict(repo_root=repo, data_root=volume + "/data/ml", sample_root=volume + "/output/ml",
                output_root=output, dataset="authors_range", seeds="0,1,2",
                variants="v1,v1_no_src_contrast,v1_no_src_bound", baseline_run="auto",
                epsilons="0.25", delta="0.05", max_trees="4000", num_threads="2",
                copy_inputs="true", mlflow_enabled="true", experiment=experiment)


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
# MAGIC The plan checks the inputs and any candidate full-v1 results before training. Compatibility
# MAGIC requires matching feature order, data fingerprints, training code, packages, parameters, and splits.
# MAGIC Reused model artifacts are copied into this study; previous runs remain unchanged.
# MAGIC The original training and feature modules remain unchanged so notebook 06 still recognizes its prior runs.

# COMMAND ----------

import importlib
import pandas as pd

repo = Path(get("repo_root")).resolve()
if not (repo / "ml/v1_feature_ablation.py").is_file():
    raise FileNotFoundError("Check repo_root and pull the experiments/model-ablation branch")
sys.path.insert(0, str(repo / "ml"))
for module_name in ("v1_feature_ablation", "compare_ablation_runs", "model_ablation", "features_v1", "features", "gate_eval"):
    sys.modules.pop(module_name, None)
importlib.invalidate_caches()
from v1_feature_ablation import plan_experiment, run_experiment, variants  # noqa: E402

CONFIG = {key: get(key) for key in DEFAULTS}
PLAN = plan_experiment(CONFIG)
print("Repo:", repo)
print("Read-only inputs:", CONFIG["data_root"], CONFIG["sample_root"])
print("Personal output root:", CONFIG["output_root"])
print("Seeds:", PLAN["seeds"])
print("New model fits:", PLAN["fit_count"])
for seed, source in PLAN["baselines"].items():
    print(f"Full v1 seed {seed}: " + (f"reuse {source}" if source else "train a new model"))
for reason in PLAN["baseline_diagnostics"][:5]:
    print("Skipped incompatible baseline:", reason)
print(pd.DataFrame([dict(variant=name, features=len(variants()[name])) for name in PLAN["variants"]]).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train and evaluate
# MAGIC
# MAGIC The same query IDs are used for all variants within each seed. Early stopping uses training
# MAGIC queries only; each model's cutoff is selected using calibration queries only. Test outcomes
# MAGIC do not select reused models, cutoffs, or feature sets automatically.
# MAGIC Exact counts below 10 are excluded, matching the existing pipeline; zero-count queries are not assessed.
# MAGIC Missing inputs or incompatible settings stop the run. There is no substitution of synthetic data.

# COMMAND ----------

if {key: get(key) for key in DEFAULTS} != CONFIG:
    raise ValueError("Settings changed; rerun Review the training plan before training")
result_dir = run_experiment(CONFIG, plan=PLAN)
print("Results for this study:", result_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Read the results
# MAGIC
# MAGIC `summary.csv` reports the learned models' means and sample standard deviations over seeds.
# MAGIC `per_rate_summary.csv` shows each merge rate, the number of seeds with defined risk, and test
# MAGIC target exceedances. `gate_details.csv` preserves the individual accepted and violating counts.
# MAGIC Zero accepted answers have undefined risk and must not be interpreted as meeting the target.
# MAGIC `oracle` uses true error and is an offline reference only.
# MAGIC
# MAGIC `paired_deltas.csv` compares each reduced model with **full v1 at the same seed and epsilon**.
# MAGIC Deltas are reduced minus full: a negative coverage delta means fewer accepted answers;
# MAGIC a positive risk delta means more violations among accepted answers. Compare both quantities.
# MAGIC AUC measures ranking, not answer accuracy. Seed standard deviations are descriptive, not confidence intervals.
# MAGIC
# MAGIC The new directory contains every model's schema, cutoffs, cohorts, predictions, and source
# MAGIC provenance. `run.json` must report `status=complete`. Reused training times describe the original
# MAGIC fit; the new study records reuse and current wall time separately. No end-to-end speedup is measured.

# COMMAND ----------

import json

metadata = json.loads((result_dir / "run.json").read_text())
if metadata["status"] != "complete":
    raise ValueError("This study is incomplete; inspect run.json")
summary = pd.read_csv(result_dir / "summary.csv")
deltas = pd.read_csv(result_dir / "paired_deltas.csv")
rates = pd.read_csv(result_dir / "per_rate_summary.csv")
try:
    display(summary)  # noqa: F821
    display(deltas)  # noqa: F821
    display(rates)  # noqa: F821
except NameError:
    print(summary.to_string(index=False))
    print(deltas.to_string(index=False))
print("Result files:", result_dir)
print("Share summary.csv, by_seed.csv, paired_deltas.csv, per_rate_summary.csv, gate_details.csv, and run.json for analysis.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Interpretation
# MAGIC
# MAGIC A feature may improve v0 when added alone but add little to full v1 if other features provide
# MAGIC similar information. This leave-out experiment tests conditional usefulness after retraining;
# MAGIC it does not prove a feature is universally necessary or causally responsible for a change.
# MAGIC Keep observed test risk and sample counts alongside coverage. This experiment deliberately
# MAGIC preserves the earlier calibration method for comparability; risk-control improvements are a separate study.
# MAGIC Do not repeatedly retune delta against these test results. Confirm final claims with a reserved workload.
# MAGIC Removing a model input still builds the complete source feature table and therefore makes no I/O-saving claim.
