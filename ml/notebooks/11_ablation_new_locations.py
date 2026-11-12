# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "lightgbm",
#   "xgboost",
# ]
# ///
# MAGIC %md
# MAGIC # 11 · Input ablations at new locations
# MAGIC
# MAGIC Notebooks 06, 06_source_feature_ablation and 07_v1_feature_ablation measured which inputs help, using the query split, where most test queries
# MAGIC have a sibling from the same source cell in training (notebook 10). This notebook repeats those ablations on both
# MAGIC splits, so each input group can be judged by what it adds **at locations the model has not seen** (`cell` split),
# MAGIC not only at familiar ones (`query` split).
# MAGIC
# MAGIC Variants (all retrained from scratch):
# MAGIC
# MAGIC | variant | inputs | from |
# MAGIC |---|---|---|
# MAGIC | `v1` | all 28 | 06 |
# MAGIC | `v1_no_shift`, `v1_no_skew`, `v1_no_source` | v1 minus one group | 06 |
# MAGIC | `v1_no_src_contrast`, `v1_no_src_bound` | v1 minus one source input | 07_v1_feature_ablation |
# MAGIC | `v0`, `v0_no_skew` | the 19 merged-histogram inputs, with and without Skewness-Align | 06 |
# MAGIC | `v0_src_contrast`, `v0_src_bound` | v0 plus one source input | 06_source_feature_ablation |
# MAGIC | `geometry_only` | query shape and merge rate | 06 |
# MAGIC
# MAGIC The model is chosen with the `model` widget (any name from `ml/model_study.py`). The default is the LightGBM
# MAGIC classifier, the best model in notebook 10; `lgbm_reg` reproduces notebook 06 on the query split.
# MAGIC
# MAGIC **Runtime** (2 cores): about 30 seconds per fit for `lgbm_clf` at one error limit, plus about 3 minutes to build
# MAGIC the table. Defaults (11 variants, 2 splits, seed 0, limit 25%) take about 15–20 minutes; `seeds = 0,1,2` about
# MAGIC 45 minutes.
# MAGIC
# MAGIC **Read AUC and AURC first on the `cell` split.** Coverage there is only 1–5%, a few dozen accepted answers per
# MAGIC merge rate, so it moves a lot between variants and seeds.

# COMMAND ----------

# MAGIC %pip install lightgbm xgboost

# COMMAND ----------

import os
import sys

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"


def _repo_root():
    """Folder that holds ml/ (this notebook is …/ml/notebooks/11_ablation_new_locations)."""
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
            "sample_root": volume + "/output/ml", "dataset": "authors_range", "seeds": "0", "splits": "query,cell",
            "model": "lgbm_clf", "limits": "0.25", "variants": "all",
            "experiment": "/Shared/synopsislake-error-predictor"}


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

import warnings  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

warnings.filterwarnings("ignore", message=".*eval_set.*")
sys.path.insert(0, os.path.join(get("repo_root"), "ml"))
from features import FEATURES  # noqa: E402
from features_v1 import FEATURES_V1, GROUPS, build_table_v1  # noqa: E402
from gate_eval import baseline_scores, evaluate  # noqa: E402
from model_study import MODELS, aurc, fit_scores, make_split, source_cells  # noqa: E402

try:
    import mlflow
    mlflow.set_experiment(get("experiment"))
except Exception as e:
    print("MLflow off:", str(e)[:80])
    mlflow = None

dataset = get("dataset")
run_dir, data_dir = f"{get('sample_root')}/{dataset}", f"{get('data_root')}/{dataset}"
report_dir = f"{run_dir}/report"
os.makedirs(report_dir, exist_ok=True)
SEEDS = [int(s) for s in get("seeds").split(",")]
SPLITS = get("splits").split(",")
MODEL = get("model")
LIMITS = [float(e) for e in get("limits").split(",")]


def without(features, names):
    return [f for f in features if f not in names]


VARIANTS = {
    "v1": FEATURES_V1,
    "v1_no_shift": without(FEATURES_V1, GROUPS["shift"]),
    "v1_no_skew": without(FEATURES_V1, GROUPS["skew"]),
    "v1_no_source": without(FEATURES_V1, GROUPS["source"]),
    "v1_no_src_contrast": without(FEATURES_V1, ["src_edge_contrast"]),
    "v1_no_src_bound": without(FEATURES_V1, ["src_bound_rel"]),
    "v0": FEATURES,
    "v0_no_skew": without(FEATURES, GROUPS["skew"]),
    "v0_src_contrast": FEATURES + ["src_edge_contrast"],
    "v0_src_bound": FEATURES + ["src_bound_rel"],
    "geometry_only": GROUPS["shape"] + GROUPS["merge"],
}
if get("variants") != "all":
    VARIANTS = {k: VARIANTS[k] for k in get("variants").split(",")}

df = build_table_v1(run_dir, data_dir).merge(source_cells(run_dir, data_dir), on="qId")
print(f"{len(df):,} rows | model {MODEL} | {len(VARIANTS)} variants x {len(SPLITS)} splits x {len(SEEDS)} seeds")

# COMMAND ----------

# MAGIC %md ## 1 · Ablations on both splits

# COMMAND ----------

rows = []
for split in SPLITS:
    for seed in SEEDS:
        tr, ca, te = make_split(df, seed, by=split)
        base = evaluate(ca, te, baseline_scores(ca), baseline_scores(te), eps_list=LIMITS)
        base["aurc"] = [aurc(te, baseline_scores(te)[s], e) for s, e in zip(base.score, base.eps)]
        rows.append(base.rename(columns={"score": "variant"}).assign(split=split, seed=seed, n_features=0))
        for name, feats in VARIANTS.items():
            fits = LIMITS if MODELS[MODEL][1] == "clf" else [None]
            for eps in fits:
                sc, st, info = fit_scores(MODEL, tr, ca, te, feats, seed, eps=eps)
                res = evaluate(ca, te, {name: sc}, {name: st}, eps_list=[eps] if eps else LIMITS)
                res["aurc"] = [aurc(te, st, e) for e in res.eps]
                res = res.rename(columns={"score": "variant"}).assign(split=split, seed=seed, n_features=len(feats),
                                                                     model=MODEL, **info)
                rows.append(res)
                if mlflow is not None:
                    with mlflow.start_run(run_name=f"11-{dataset}-{name}-{split}-seed{seed}"):
                        mlflow.log_params({"dataset": dataset, "variant": name, "split": split, "seed": seed,
                                           "model": MODEL, "n_features": len(feats)})
                        for r in res.itertuples():
                            tag = f"eps{int(r.eps * 100)}"
                            mlflow.log_metrics({f"{tag}_auc_per_rate": r.auc_per_rate,
                                                f"{tag}_coverage_per_rate": r.coverage_per_rate,
                                                f"{tag}_violation_per_rate": r.violation_per_rate, f"{tag}_aurc": r.aurc})
                print(f"{split} split, seed {seed}, {name}: "
                      + ", ".join(f"coverage at {e:.0%} {c:.3f}" for e, c in zip(res.eps, res.coverage_per_rate)))
abl = pd.concat(rows, ignore_index=True)
abl.to_csv(f"{report_dir}/ablation_by_split.csv", index=False)

# COMMAND ----------

# MAGIC %md ## 2 · Summary: what each input group adds on each split

# COMMAND ----------

cols = ["auc_per_rate", "coverage_per_rate", "violation_per_rate", "aurc"]
summary = abl.groupby(["eps", "split", "variant"])[cols].mean().round(3).reset_index()
summary.to_csv(f"{report_dir}/ablation_by_split_summary.csv", index=False)
wide = summary.pivot_table(index=["eps", "variant"], columns="split", values=["auc_per_rate", "coverage_per_rate"])
wide.columns = [f"{m}_{s}" for m, s in wide.columns]
order = list(VARIANTS) + ["skew_align", "edge_bound", "src_bound", "oracle"]
wide = wide.reset_index()
wide["variant"] = pd.Categorical(wide.variant, order, ordered=True)
show(wide.sort_values(["eps", "variant"]))

eps = LIMITS[-1]
fig, axes = plt.subplots(1, 2, figsize=(11, 3.8))
s = summary[summary.eps == eps]
for ax, metric, label in [(axes[0], "auc_per_rate", "AUC"), (axes[1], "coverage_per_rate", "coverage (%)")]:
    names = [v for v in order if v != "oracle" and v in s.variant.values]
    x = range(len(names))
    for j, split in enumerate(SPLITS):
        vals = s[s.split == split].set_index("variant").reindex(names)[metric] * (100 if metric != "auc_per_rate" else 1)
        ax.bar([i + (j - (len(SPLITS) - 1) / 2) * 0.4 for i in x], vals, 0.4, label=f"{split} split")
    ax.set_xticks(list(x))
    ax.set_xticklabels(names, rotation=60, ha="right", fontsize=8)
    ax.set_ylabel(f"{label} at {eps:.0%} limit")
axes[0].set_ylim(0.5, 1)
axes[0].legend()
plt.tight_layout()
fig.savefig(f"{report_dir}/ablation_by_split.png", dpi=150)
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to look for (authors' data, local run, seed 0, `lgbm_clf`, limit 25%)
# MAGIC
# MAGIC | variant | AUC, query split | AUC, cell split | coverage, query | coverage, cell |
# MAGIC |---|---|---|---|---|
# MAGIC | `v1` | 0.902 | 0.741 | 20.1% | 4.5% |
# MAGIC | `v1_no_shift` | 0.900 | 0.724 | 21.7% | 3.2% |
# MAGIC | `v1_no_skew` | 0.896 | 0.725 | 14.8% | 1.2% |
# MAGIC | `v1_no_source` | 0.877 | 0.720 | 12.3% | 2.9% |
# MAGIC | `v1_no_src_contrast` | 0.889 | 0.734 | 14.6% | 3.5% |
# MAGIC | `v1_no_src_bound` | 0.905 | 0.728 | 20.7% | 3.4% |
# MAGIC | `v0` | 0.855 | 0.646 | 10.9% | 2.7% |
# MAGIC | `v0_no_skew` | 0.853 | 0.621 | 6.8% | 0.8% |
# MAGIC | `v0_src_contrast` | 0.875 | 0.660 | 14.2% | 2.7% |
# MAGIC | `v0_src_bound` | 0.856 | 0.672 | 13.6% | 2.1% |
# MAGIC | `geometry_only` | 0.619 | 0.579 | 0.7% | 0.2% |
# MAGIC | Skewness-Align alone | 0.608 | 0.608 | 0.0% | 0.1% |
# MAGIC
# MAGIC * **The v1 inputs help more at new locations.** v0 → v1 raises AUC by 0.05 on the query split but by 0.10 on the
# MAGIC   cell split (0.646 → 0.741). The source-level and merge-shift inputs carry information that transfers.
# MAGIC * **At new locations every group matters a little and they overlap.** Removing shift, Skewness-Align or the
# MAGIC   source group each costs 0.016–0.021 AUC; no single group dominates.
# MAGIC * **Skewness-Align is a useful input on both splits.** Removing it costs 5.3 points of coverage on the query
# MAGIC   split and most of the coverage at new locations (4.5% → 1.2%).
# MAGIC * **The contrast vs bound result depends on the split.** On the query split contrast adds more (as in
# MAGIC   07_v1_feature_ablation); at new locations bound adds more AUC (0.672 vs 0.660 when added to v0). One seed, so confirm with
# MAGIC   three.