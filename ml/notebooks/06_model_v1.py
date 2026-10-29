# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "lightgbm",
# ]
# ///
# MAGIC %md
# MAGIC # 06 · Model v1: new inputs, ablations, generalization
# MAGIC
# MAGIC Builds on model v0 (notebook 02) with the same label, splits and gate, and asks three questions:
# MAGIC
# MAGIC 1. **Do source-level inputs help?** v1 adds 9 inputs to v0's 19 (`ml/features_v1.py`):
# MAGIC     * 7 that describe the query on each per-file source histogram
# MAGIC     * 2 that measure how far the merged answer moved away from the no-merge answer (`merge_shift`)
# MAGIC
# MAGIC     All of them come from synopses only.
# MAGIC 2. **Which inputs matter?** Ablations drop one input group at a time: Skewness-Align, source-level, merge
# MAGIC    shift; there is also a geometry-only model.
# MAGIC 3. **Does it transfer?**
# MAGIC     * Train on high merge rates (40% and up), test on low ones.
# MAGIC     * Train on the western half of the map, test on the eastern half.
# MAGIC
# MAGIC     Each is compared with a model trained on everything and tested on the same queries.
# MAGIC
# MAGIC **How to read the results.** The gate is evaluated *per merge rate*: one cutoff per merged synopsis, set on
# MAGIC calibration queries for at most 5% violations, then averaged over the 18 rates. That matches deployment, where a
# MAGIC query is answered from one merged synopsis. `coverage_per_rate` is the share of test queries the gate answers from
# MAGIC the synopsis. The pooled numbers (one cutoff over all rates, as in notebook 02) are kept for comparison.
# MAGIC
# MAGIC **Runtime:** about 33 model fits (7 variants × 3 seeds, plus 2 tests × 2 settings × 3 seeds). Use the `seeds`,
# MAGIC `variants` and `generalization` widgets to run less.
# MAGIC
# MAGIC Locally: `VOLUME=… REPO_ROOT=… DATA_ROOT=… SAMPLE_ROOT=… DATASET=… python ml/notebooks/06_model_v1.py`.

# COMMAND ----------

# MAGIC %pip install lightgbm

# COMMAND ----------

import os
import sys

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"


def _repo_root():
    """Folder that holds ml/ (this notebook is …/ml/notebooks/06_model_v1)."""
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
            "sample_root": volume + "/output/ml", "dataset": "authors_range", "seeds": "0,1,2",
            "variants": "all", "generalization": "true", "experiment": "/Shared/synopsislake-error-predictor"}


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

import time  # noqa: E402
import warnings  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore", message=".*eval_set.*")   # newer LightGBM renamed the argument; still works
sys.path.insert(0, os.path.join(get("repo_root"), "ml"))
from features import FEATURES, check_estimates  # noqa: E402
from features_v1 import FEATURES_V1, GROUPS, build_table_v1, check_source_estimates  # noqa: E402
from gate_eval import PARAMS, run  # noqa: E402

try:
    import mlflow
    mlflow.set_experiment(get("experiment"))
except Exception as e:                                         # runs without MLflow too (locally)
    print("MLflow off:", str(e)[:80])
    mlflow = None

dataset = get("dataset")
run_dir, data_dir = f"{get('sample_root')}/{dataset}", f"{get('data_root')}/{dataset}"
report_dir = f"{run_dir}/report"
os.makedirs(report_dir, exist_ok=True)
SEEDS = [int(s) for s in get("seeds").split(",")]

df = build_table_v1(run_dir, data_dir)
queries = pd.read_csv(f"{data_dir}/queries-Ans.txt", header=None)
df = df.merge(((queries[0] + queries[1]) / 2).rename("center_lon").rename_axis("qId").reset_index(), on="qId")
print(f"{len(df):,} rows, {df.qId.nunique():,} queries | estimate gap {check_estimates(df):.1e} | "
      f"no-merge estimate gap {check_source_estimates(df):.1e} | {len(FEATURES)} v0 inputs, {len(FEATURES_V1)} v1 inputs")

# COMMAND ----------

# MAGIC %md ## 1 · Ablations

# COMMAND ----------


def drop(features, group):
    return [f for f in features if f not in GROUPS[group]]


VARIANTS = {
    "v0": FEATURES,
    "v0_no_skew": drop(FEATURES, "skew"),
    "v1": FEATURES_V1,
    "v1_no_skew": drop(FEATURES_V1, "skew"),
    "v1_no_shift": drop(FEATURES_V1, "shift"),
    "v1_no_source": drop(FEATURES_V1, "source"),
    "geometry_only": GROUPS["shape"] + GROUPS["merge"],
}
if get("variants") != "all":
    VARIANTS = {k: VARIANTS[k] for k in get("variants").split(",")}


def log_run(name, seed, features, res):
    if mlflow is None:
        return
    with mlflow.start_run(run_name=f"v1-{dataset}-{name}-seed{seed}"):
        mlflow.log_params({"dataset": dataset, "variant": name, "seed": seed, "n_features": len(features),
                           "features": ",".join(features), **PARAMS})
        for r in res.itertuples():
            tag = f"{r.score}_eps{int(r.eps * 100)}"
            mlflow.log_metrics({f"{tag}_auc_per_rate": r.auc_per_rate, f"{tag}_coverage_per_rate": r.coverage_per_rate,
                                f"{tag}_violation_per_rate": r.violation_per_rate, f"{tag}_coverage_pooled": r.coverage})
        mlflow.log_metric("best_iteration", int(res.best_iteration.iloc[0]))


results, importances = [], []
for name, feats in VARIANTS.items():
    for seed in SEEDS:
        t0 = time.time()
        res, imp, _ = run(df, feats, seed)
        res.insert(0, "variant", name)
        res.insert(1, "n_features", len(feats))
        results.append(res)
        importances.append(imp.assign(variant=name, seed=seed))
        log_run(name, seed, feats, res)
        print(f"{name} seed {seed}: {time.time() - t0:.0f}s, best iteration {res.best_iteration.iloc[0]}")
ablations = pd.concat(results, ignore_index=True)
ablations.to_csv(f"{report_dir}/v1_ablations.csv", index=False)
pd.concat(importances).to_csv(f"{report_dir}/v1_importance.csv", index=False)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Summary: mean (and spread) over seeds
# MAGIC
# MAGIC The learned rows show one line per variant. Baselines don't depend on the variant, so they are shown once.
# MAGIC `oracle` uses the true error and is the ceiling.

# COMMAND ----------

cols = ["auc_per_rate", "coverage_per_rate", "violation_per_rate", "auc", "coverage", "violation"]
learned = ablations[ablations.score == "learned"]
summary = learned.groupby(["eps", "variant"])[cols].agg(["mean", "std"]).round(3)
summary.columns = [f"{c}_{s}" for c, s in summary.columns]
first = list(VARIANTS)[0]
base = (ablations[(ablations.score != "learned") & (ablations.variant == first)]
        .groupby(["eps", "score"])[cols].mean().round(3))
show(summary.reset_index())
show(base.reset_index())
summary.reset_index().to_csv(f"{report_dir}/v1_summary.csv", index=False)

eps = 0.25
s = summary.reset_index()
s = s[s.eps == eps].set_index("variant").reindex(list(VARIANTS))
fig, ax = plt.subplots(figsize=(8, 4))
ax.bar(s.index, s.coverage_per_rate_mean, yerr=s.coverage_per_rate_std, color="#4C78A8")
ax.axhline(base.loc[(eps, "oracle"), "coverage_per_rate"], ls="--", color="grey", label="oracle (true error)")
ax.axhline(base.loc[(eps, "skew_align"), "coverage_per_rate"], ls=":", color="black", label="Skewness-Align")
ax.set_ylabel("share answered from synopsis")
ax.set_title(f"Gate coverage at {int(eps * 100)}% error limit, at most 5% violations (per merge rate)")
ax.legend()
plt.xticks(rotation=20)
plt.tight_layout()
fig.savefig(f"{report_dir}/v1_coverage.png", dpi=150)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Generalization
# MAGIC
# MAGIC * `shifted`: trained only on the training side of the split, then tested on the other side.
# MAGIC * `in_distribution`: trained on all training queries, then tested on the same test queries.
# MAGIC
# MAGIC The gap between the two is the cost of never having seen that part of the data.

# COMMAND ----------

if get("generalization").lower() == "true":
    mid = (df.center_lon.min() + df.center_lon.max()) / 2
    tests = {"rate_high_to_low": (df.mergeRate >= 0.40, df.mergeRate < 0.40),
             "west_to_east": (df.center_lon < mid, df.center_lon >= mid)}
    gen = []
    for tname, (train_side, test_side) in tests.items():
        for seed in SEEDS:
            for setting, train_mask in [("shifted", train_side), ("in_distribution", None)]:
                res, _, _ = run(df, FEATURES_V1, seed, train_mask=train_mask, test_mask=test_side)
                res.insert(0, "test", tname)
                res.insert(1, "setting", setting)
                gen.append(res)
                print(tname, setting, seed, "done")
    gen = pd.concat(gen, ignore_index=True)
    gen.to_csv(f"{report_dir}/v1_generalization.csv", index=False)
    g = (gen[gen.score == "learned"].groupby(["test", "eps", "setting"])
         [["auc_per_rate", "coverage_per_rate", "violation_per_rate"]].mean().round(3))
    show(g.reset_index())
    print(f"west/east split at longitude {mid:.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to look for (authors' data, from the prototype run)
# MAGIC
# MAGIC These numbers come from the local prototype, as means over seeds 0–2, evaluated per merge rate. Yours should be close;
# MAGIC library versions shift them slightly.
# MAGIC
# MAGIC | variant | AUC | coverage at 25% | coverage at 10% |
# MAGIC |---|---|---|---|
# MAGIC | `v1` | 0.880 | 16.0% | 3.3% |
# MAGIC | `v1_no_shift` (v0 + source inputs) | 0.876 | 15.8% | 2.9% |
# MAGIC | `v1_no_skew` | 0.874 | 12.1% | 1.2% |
# MAGIC | `v1_no_source` (v0 + merge shift) | 0.852 | 11.7% | 2.7% |
# MAGIC | `v0` | 0.822 | 10.4% | 2.7% |
# MAGIC | `v0_no_skew` | 0.819 | 7.8% | 1.3% |
# MAGIC | `geometry_only` | 0.613 | 0.6% | 0.2% |
# MAGIC | Skewness-Align alone | 0.605 | 0.2% | 0.0% |
# MAGIC | oracle | 1.000 | 54.4% | 25.3% |
# MAGIC
# MAGIC * **Source-level inputs are the main gain.** AUC rises from 0.82 to 0.88, and coverage at 25% from 10% to 16%.
# MAGIC   Merge shift adds little once the source inputs are in.
# MAGIC * **Skewness-Align is useful as an input, though weak as a score on its own.** Removing it costs 3–4 points of
# MAGIC   coverage.
# MAGIC * **Geometry alone (query shape and merge rate) predicts almost nothing**, so the histogram counts carry the signal.
# MAGIC * **Violations average 5–6%, slightly above the 5% target.** The cutoff is empirical, so a guarantee needs
# MAGIC   conformal calibration.
# MAGIC * **Every model except geometry-only reached the 4,000-tree cap**, so more capacity may help further.
# MAGIC * **Generalization (v1, AUC at the 25% limit):**
# MAGIC     * High to low merge rates: 0.81 shifted vs 0.90 in distribution. Coverage falls from 11.6% to 4.9%.
# MAGIC     * West to east: 0.72 vs 0.89. Coverage falls from 16.7% to 2.1%. The east uses different source files and
# MAGIC       resolutions (32 and 48 cells) from the west (40 and 56), so this tests a change of geography and of synopsis
# MAGIC       together.
# MAGIC     * In both tests the shifted model still ranks better than Skewness-Align on the same queries (0.61–0.62).