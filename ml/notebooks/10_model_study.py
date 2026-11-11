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
# MAGIC # 10 · Model study: which model, objective and evaluation split?
# MAGIC
# MAGIC Notebooks 06, 06_source_feature_ablation and 07_v1_feature_ablation changed the **inputs**. This notebook keeps the 28 v1 inputs fixed and changes
# MAGIC the **model** and the **evaluation split**. It asks three questions:
# MAGIC
# MAGIC 1. **How much of the score comes from having seen the same location?** The workload has about 5 shifted queries
# MAGIC    per source-histogram cell. With the query split used so far, most test queries have a sibling from the same
# MAGIC    cell in training. The `cell` split keeps all queries of a cell together, so every test location is new.
# MAGIC 2. **Which model and objective work best?** LightGBM, XGBoost, random forest, a neural network (MLP) and a linear
# MAGIC    model. `reg` models predict the log error (as in 06); `clf` models predict directly whether the error is over
# MAGIC    the limit, with one model per limit.
# MAGIC 3. **Would more training data help?** A learning curve trains on 10–100% of the training groups.
# MAGIC
# MAGIC Everything else matches notebook 06: label, exact counts ≥ 10, 50/20/30 split, and one cutoff per merge rate set
# MAGIC on calibration queries for at most 5% violations. With `split = query` and `model = lgbm_reg` the numbers equal
# MAGIC notebook 06's v1 row for the same seed.
# MAGIC
# MAGIC **New metric — AURC** (area under the risk-coverage curve): accept answers in order of score and track the share
# MAGIC over the limit at every coverage level; AURC is the average of that share. It scores the whole ranking rather
# MAGIC than one cutoff, so it is steadier than coverage. Lower is better.
# MAGIC
# MAGIC **Runtime** (2 cores): building the table takes about 3 minutes; all seven models take about 8 minutes per seed
# MAGIC and split (random forest and the MLP take most of it); the learning curve about 10 minutes. The defaults
# MAGIC (`seeds = 0`, both splits, learning curve on) take about 30 minutes. For the final numbers set `seeds = 0,1,2`.

# COMMAND ----------

# MAGIC %pip install lightgbm xgboost

# COMMAND ----------

import os
import sys

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"
ALL_MODELS = "lgbm_reg,lgbm_clf,xgb_reg,xgb_clf,rf_reg,mlp_reg,linear_reg"


def _repo_root():
    """Folder that holds ml/ (this notebook is …/ml/notebooks/10_model_study)."""
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
            "models": ALL_MODELS, "learning_curve": "true", "lc_models": "lgbm_reg,lgbm_clf",
            "fractions": "0.1,0.25,0.5,1.0", "experiment": "/Shared/synopsislake-error-predictor"}


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
from features_v1 import FEATURES_V1, build_table_v1  # noqa: E402
from gate_eval import baseline_scores, evaluate  # noqa: E402
from model_study import (MODELS, aurc, fit_scores, make_split, sibling_lookup, sibling_share,  # noqa: E402
                         source_cells)

try:
    import mlflow
    mlflow.set_experiment(get("experiment"))
except Exception as e:
    print("MLflow off:", str(e)[:80])
    mlflow = None

EPS = [0.10, 0.25]
dataset = get("dataset")
run_dir, data_dir = f"{get('sample_root')}/{dataset}", f"{get('data_root')}/{dataset}"
report_dir = f"{run_dir}/report"
os.makedirs(report_dir, exist_ok=True)
SEEDS = [int(s) for s in get("seeds").split(",")]
SPLITS = get("splits").split(",")
MODEL_NAMES = [m for m in get("models").split(",") if m]

df = build_table_v1(run_dir, data_dir).merge(source_cells(run_dir, data_dir), on="qId")
print(f"{len(df):,} rows, {df.qId.nunique():,} queries from {df.cell.nunique():,} source cells "
      f"({df.qId.nunique() / df.cell.nunique():.1f} queries per cell)")
for split in SPLITS:
    tr, ca, te = make_split(df, SEEDS[0], by=split)
    print(f"split = {split:5s}: {sibling_share(tr, te):.1%} of test queries have a sibling from the same source cell "
          "in training")

# COMMAND ----------

# MAGIC %md ## 1 · Models and objectives on both splits

# COMMAND ----------


def score_rows(cal, test, cal_scores, test_scores, eps_list):
    """gate_eval.evaluate plus AURC, one row per (score, eps)."""
    res = evaluate(cal, test, cal_scores, test_scores, eps_list=eps_list)
    res["aurc"] = [aurc(test, test_scores[s], e) for s, e in zip(res.score, res.eps)]
    return res


def log_run(row_frame, name, split, seed):
    if mlflow is None:
        return
    with mlflow.start_run(run_name=f"10-{dataset}-{name}-{split}-seed{seed}"):
        mlflow.log_params({"dataset": dataset, "model": name, "split": split, "seed": seed,
                           "features": "v1", "n_features": len(FEATURES_V1)})
        for r in row_frame.itertuples():
            tag = f"eps{int(r.eps * 100)}"
            mlflow.log_metrics({f"{tag}_auc_per_rate": r.auc_per_rate, f"{tag}_coverage_per_rate": r.coverage_per_rate,
                                f"{tag}_violation_per_rate": r.violation_per_rate, f"{tag}_aurc": r.aurc,
                                "fit_seconds": r.fit_seconds})


rows = []
for split in SPLITS:
    for seed in SEEDS:
        tr, ca, te = make_split(df, seed, by=split)
        sib_c, sib_t = sibling_lookup(tr, ca, te)
        base = score_rows(ca, te, {**baseline_scores(ca), "sibling_lookup": sib_c},
                          {**baseline_scores(te), "sibling_lookup": sib_t}, EPS)
        rows.append(base.rename(columns={"score": "model"}).assign(split=split, seed=seed, objective="baseline"))
        for name in MODEL_NAMES:
            objective = MODELS[name][1]
            limits = EPS if objective == "clf" else [None]
            for eps in limits:
                sc, st, info = fit_scores(name, tr, ca, te, FEATURES_V1, seed, eps=eps)
                res = score_rows(ca, te, {name: sc}, {name: st}, [eps] if eps else EPS)
                res = res.rename(columns={"score": "model"}).assign(split=split, seed=seed, objective=objective, **info)
                rows.append(res)
                log_run(res, name, split, seed)
                label = name if eps is None else f"{name} (limit {eps:.0%})"
                summary_txt = ", ".join(f"coverage at {e:.0%} {c:.3f}" for e, c in zip(res.eps, res.coverage_per_rate))
                print(f"{split} split, seed {seed}, {label}: {info['fit_seconds']:.0f}s, {summary_txt}")
study = pd.concat(rows, ignore_index=True)
study.to_csv(f"{report_dir}/model_study.csv", index=False)

# COMMAND ----------

# MAGIC %md ## 2 · Summary: mean over seeds
# MAGIC
# MAGIC Baselines need no training: `skew_align`, `edge_bound` and `src_bound` as in notebook 06; `oracle` ranks by the
# MAGIC true error and is the ceiling; `sibling_lookup` predicts a query's error from its training siblings (same source
# MAGIC cell and merge rate) with no model, so it shows how much ranking is available just from remembering locations.

# COMMAND ----------

cols = [c for c in ["auc_per_rate", "coverage_per_rate", "violation_per_rate", "aurc", "fit_seconds"] if c in study]
summary = (study.groupby(["split", "eps", "model"])[cols].mean().round(3)
           .join(study.groupby(["split", "eps", "model"]).coverage_per_rate.std().round(3).rename("coverage_sd")))
summary = summary.reset_index()
summary.to_csv(f"{report_dir}/model_study_summary.csv", index=False)
show(summary[summary.eps == 0.25].sort_values(["split", "coverage_per_rate"], ascending=[True, False]))
show(summary[summary.eps == 0.10].sort_values(["split", "coverage_per_rate"], ascending=[True, False]))

order = MODEL_NAMES + ["sibling_lookup", "skew_align"]
fig, axes = plt.subplots(1, len(SPLITS), figsize=(5.5 * len(SPLITS), 3.6), sharey=True, squeeze=False)
for ax, split in zip(axes[0], SPLITS):
    s = summary[(summary.split == split) & (summary.eps == 0.25)].set_index("model").reindex(order)
    ax.bar(s.index, s.coverage_per_rate * 100, yerr=s.coverage_sd.fillna(0) * 100, color="#4C78A8")
    orc = summary[(summary.split == split) & (summary.eps == 0.25) & (summary.model == "oracle")]
    ax.axhline(orc.coverage_per_rate.iloc[0] * 100, ls="--", color="grey", label="oracle")
    ax.set_title(f"split = {split}")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
axes[0][0].set_ylabel("coverage at 25% limit (%)")
plt.tight_layout()
fig.savefig(f"{report_dir}/model_study_coverage.png", dpi=150)
plt.show()

# COMMAND ----------

# MAGIC %md ## 3 · Learning curve
# MAGIC
# MAGIC Trains on a share of the training groups (queries or cells); calibration and test stay the same. If coverage keeps
# MAGIC rising at 100%, more data would help. Compare the two splits: with the query split, more training data also means
# MAGIC more test queries with a sibling in training.

# COMMAND ----------

if get("learning_curve").lower() == "true":
    curve = []
    for split in SPLITS:
        for seed in SEEDS:
            tr_full, ca, te = make_split(df, seed, by=split)
            for frac in [float(f) for f in get("fractions").split(",")]:
                tr, _, _ = make_split(df, seed, by=split, train_fraction=frac)
                share = sibling_share(tr, te)
                for name in get("lc_models").split(","):
                    limits = EPS if MODELS[name][1] == "clf" else [None]
                    for eps in limits:
                        sc, st, info = fit_scores(name, tr, ca, te, FEATURES_V1, seed, eps=eps)
                        res = score_rows(ca, te, {name: sc}, {name: st}, [eps] if eps else EPS)
                        curve.append(res.rename(columns={"score": "model"}).assign(
                            split=split, seed=seed, fraction=frac, train_queries=tr.qId.nunique(),
                            sibling_share=share, **info))
                print(f"{split} seed {seed} fraction {frac}: done ({share:.0%} of test queries have a sibling)")
    curve = pd.concat(curve, ignore_index=True)
    curve.to_csv(f"{report_dir}/learning_curve.csv", index=False)
    c = (curve[curve.eps == 0.25].groupby(["split", "model", "fraction"])
         [["train_queries", "sibling_share", "auc_per_rate", "coverage_per_rate", "aurc"]].mean().round(3).reset_index())
    show(c)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6))
    fracs = sorted(c.fraction.unique())
    for ax, metric, label, scale in [(axes[0], "auc_per_rate", "AUC at 25% limit", 1),
                                     (axes[1], "coverage_per_rate", "coverage at 25% limit (%)", 100)]:
        for (split, name), g in c.groupby(["split", "model"]):
            ax.plot(g.fraction * 100, g[metric] * scale, marker="o", ls="-" if split == "query" else "--",
                    label=f"{name}, {split} split")
        ax.set_xscale("log")
        ax.set_xticks([f * 100 for f in fracs])
        ax.set_xticklabels([f"{f * 100:g}" for f in fracs])
        ax.minorticks_off()
        ax.set_xlabel("share of training data used (%)")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8)
    plt.tight_layout()
    fig.savefig(f"{report_dir}/learning_curve.png", dpi=150)
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## What to look for (authors' data, local run, seed 0)
# MAGIC
# MAGIC Yours should match closely; library versions shift the last digit. Limit 25%, per-merge-rate averages.
# MAGIC
# MAGIC | model | AUC, query split | coverage, query split | AUC, cell split | coverage, cell split |
# MAGIC |---|---|---|---|---|
# MAGIC | `lgbm_clf` | 0.902 | 20.1% | 0.741 | 4.5% |
# MAGIC | `xgb_clf` | 0.893 | 18.5% | 0.734 | 3.2% |
# MAGIC | `rf_reg` | 0.881 | 16.7% | 0.747 | 3.3% |
# MAGIC | `lgbm_reg` (notebook 06's v1) | 0.886 | 14.8% | 0.723 | 1.5% |
# MAGIC | `xgb_reg` | 0.872 | 14.7% | 0.718 | 1.5% |
# MAGIC | `mlp_reg` | 0.814 | 5.1% | 0.686 | 1.3% |
# MAGIC | `linear_reg` | 0.695 | 0.5% | 0.696 | 1.1% |
# MAGIC | `sibling_lookup` (no model) | 0.879 | 2.5% | 0.500 | 0.0% |
# MAGIC | `skew_align` | 0.608 | 0.0% | 0.608 | 0.1% |
# MAGIC | `oracle` | 1.000 | 54.5% | 1.000 | 55.3% |
# MAGIC
# MAGIC * **Most of the query-split score comes from remembering locations.** 92% of test queries have a training sibling
# MAGIC   from the same source cell. Looking up the siblings' errors, with no model, already reaches AUC 0.879. At new
# MAGIC   locations every tree model loses 0.13–0.16 AUC and most of its coverage.
# MAGIC * **The linear model scores the same on both splits** (0.695 vs 0.696): it cannot memorize locations. At new
# MAGIC   locations the trees beat it by only 0.02–0.05 AUC.
# MAGIC * **Classifying "error over the limit" beats regressing the error at 25%** on both splits (LightGBM: 14.8% → 20.1%
# MAGIC   and 1.5% → 4.5%). At the 10% limit the objectives are about equal.
# MAGIC * **More trees do not help.** With up to 20,000 trees, LightGBM stops at about 4,400 with the same result; training
# MAGIC   AUC is 0.99 against 0.89 on test, so the model already fits its training data.
# MAGIC * **More data helps at familiar locations, barely at new ones.** Learning curve for `lgbm_clf`, 10% → 100% of the
# MAGIC   training data: on the query split AUC rises 0.740 → 0.902, tracking the share of test queries with a training
# MAGIC   sibling (18% → 92%); on the cell split it rises only 0.698 → 0.741 and is flattening. Gains at new locations
# MAGIC   need better inputs, not more of the same data.
# MAGIC * **Violation rates at new locations are noisy** because so few answers are accepted per merge rate.