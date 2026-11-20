# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "lightgbm",
#   "pyarrow",
# ]
# ///
# MAGIC %md
# MAGIC # 09b · Skewness-Align vs exact execution vs every learned error predictor (v0, v1, v1_no_skew)
# MAGIC
# MAGIC Notebook 09 with several learned models. Runs the **same held-out queries** on the **same dataset** and merge rates
# MAGIC through Skewness-Align, exact execution and **each model in `models`** (default `v0,v1,v1_no_skew`), and compares
# MAGIC latency, error and coverage. Nothing is trained: every model is read from the catalog.
# MAGIC
# MAGIC | Setup | How it answers |
# MAGIC |---|---|
# MAGIC | **Skewness-Align** | `ApproxQueryExp -mode gated`, the original Skewness-Align score decides synopsis or raw data (its cutoffs are calibrated here, on the same calibration queries and with the same rule as the learned model's) |
# MAGIC | **Exact** | `ApproxQueryExp -mode exact`: every query is counted from the raw points |
# MAGIC | **One row per model** (`v0`, `v1`, `v1_no_skew`, ...) | `ApproxQueryExp -mode gated -modelPath`: that model decides, using **artifacts that already exist** in `<run_dir>/models/<name>/` (`model.txt`, `model.properties`, `cutoffs.csv`; notebook 06 `export_models`). `v0` falls back to `<run_dir>/gate` when `models/v0` does not exist. Nothing is trained here |
# MAGIC
# MAGIC **What is measured**
# MAGIC * **Latency**: wall-clock seconds from starting `ApproxQueryExp` on the query file to having the answers as a
# MAGIC   Parquet file (Java run + CSV to Parquet), median of `repeats` runs. A run includes starting the JVM, loading the
# MAGIC   synopsis (and the raw points for `gated` / `exact`), so it is the cost of producing the file, not just of
# MAGIC   answering. The per-query gate / synopsis / raw time that Java logs is reported separately.
# MAGIC * **Error**: relative error of the answer each setup returns (a raw-data answer has error 0), its 95th
# MAGIC   percentile, and the share of synopsis-answered queries whose error is above `eps` (must be near or below
# MAGIC   `delta`).
# MAGIC * **Coverage**: share of queries answered from the synopsis instead of the raw data. Exact is 0% by definition.
# MAGIC
# MAGIC **Needs** the run folder from notebook 07 and the model folders above (or any dataset prepared the same way), and the raw
# MAGIC data in `<volume>/data/range_query`. A classic dedicated cluster on an ML runtime, as for notebook 07.

# COMMAND ----------

# MAGIC %pip install lightgbm pyarrow

# COMMAND ----------

import os

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"


def _repo_root():
    """Folder that holds ml/ (this notebook is …/ml/notebooks/09_compare_skewness_exact_learned)."""
    try:
        nb = dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()  # noqa: F821
        root = nb.split("/ml/notebooks/")[0]
        return root if root.startswith("/Workspace/") else "/Workspace" + root
    except Exception:
        try:
            return os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        except NameError:
            return os.getcwd()


DEFAULTS = {
    "volume": VOL,
    "repo_root": _repo_root(),
    "dataset": "europe_range",
    "run_dir": "",              # default <volume>/output/ml/<dataset>
    "models": "v0,v1,v1_no_skew",   # learned models to run; each is a folder <models_root>/<name> with model.txt, model.properties, cutoffs.csv
    "models_root": "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake/ml_artifacts",          # default <run_dir>/models
    "data_dir": "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake/data/europe_range",             # default <volume>/data/range_query (0.csv, 1.csv, ... tab-separated lon lat)
    "seed": "0",
    "eps": "0.10",              # an answer is "good" if its relative error is at most eps
    "delta": "0.05",            # at most this share of synopsis answers may be above eps
    "merge_rates": "1,0.8,0.6,0.4,0.2",   # the nearest available merge rate is used for each
    "n_queries": "2000",        # held-out queries to answer (a random sample, same for every setup); "all" for every one
    "repeats": "2",             # timing repeats per setup and merge rate
    "java": "java",
}
try:
    for k, v in DEFAULTS.items():
        dbutils.widgets.text(k, v)  # noqa: F821
    get = dbutils.widgets.get  # noqa: F821
except NameError:
    def get(name):
        return os.environ.get(name.upper(), DEFAULTS[name])

# COMMAND ----------

import glob
import shutil
import subprocess
import sys
import time
import zipfile

import numpy as np
import pandas as pd

volume, repo, dataset, java = get("volume"), get("repo_root"), get("dataset"), get("java")
seed, eps, delta = int(get("seed")), float(get("eps")), float(get("delta"))
repeats = max(1, int(get("repeats")))
wanted_rates = [float(x) for x in get("merge_rates").split(",") if x.strip()]

jar = f"{volume}/jars/synopsislake.jar"
data_dir = get("data_dir") or f"{volume}/data/range_query"
ds_dir = f"{volume}/data/ml/{dataset}"
run_dir = get("run_dir") or f"{volume}/output/ml/{dataset}"
models_root = get("models_root") or f"{run_dir}/models"
model_names = [m.strip() for m in get("models").split(",") if m.strip()]
model_dirs = {m: f"{models_root}/{m}" for m in model_names}
if "v0" in model_dirs and not os.path.exists(f"{model_dirs['v0']}/model.txt") and os.path.exists(f"{run_dir}/gate/model.txt"):
    model_dirs["v0"] = f"{run_dir}/gate"            # the v0 model of notebooks 02 / 07
skew_dir = f"{run_dir}/compare/skew_gate"
out_dir = f"{run_dir}/compare"
for d in (skew_dir, out_dir):
    os.makedirs(d, exist_ok=True)
sys.path.insert(0, os.path.join(repo, "ml"))

n_files = len([f for f in glob.glob(f"{data_dir}/*.csv") if os.path.basename(f)[:-4].isdigit()])
print("volume", volume, "| dataset", dataset, "| models", model_names, "| merge rates", wanted_rates, "| eps", eps, "delta", delta,
      "| repeats", repeats, "| source files", n_files)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1 · Check the inputs

# COMMAND ----------

problems = []
if shutil.which(java) is None:
    problems.append(f"`{java}` not found on this cluster (needs a classic cluster with a JDK)")
if not os.path.exists(jar):
    problems.append(f"{jar} is missing: run the CI workflow, or upload the jar built with `mvn -DskipTests package`")
else:
    with zipfile.ZipFile(jar) as z:
        if "sigspatial25/ApproxQueryExp.class" not in z.namelist():
            problems.append(f"{jar} has no ApproxQueryExp: it was built from a commit without the query path")
if n_files == 0:
    problems.append(f"no raw data files {data_dir}/0.csv ...")
if not glob.glob(f"{run_dir}/arm_*.txt") or not os.path.exists(f"{run_dir}/rows.csv"):
    problems.append(f"no run folder at {run_dir} (arm_*.txt and rows.csv): run notebook 07 first")
if not model_names:
    problems.append("`models` is empty")
for m, d in model_dirs.items():
    for name in ("model.txt", "model.properties", "cutoffs.csv"):
        if not os.path.exists(f"{d}/{name}"):
            problems.append(f"{d}/{name} is missing: model {m} needs the artifacts from notebook 06 (export_models) or 07")
if not os.path.exists(f"{ds_dir}/queries-Ans.txt"):
    problems.append(f"{ds_dir}/queries-Ans.txt is missing (notebook 07 step 2 or notebook 01)")
try:
    import pyarrow  # noqa: F401
except ImportError:
    problems.append("pyarrow is not installed (%pip install pyarrow)")


def find_math3(volume):
    """commons-math3 jar for the classpath: the cluster's own copy if there is one, else one cached in the volume,
    else downloaded once from Maven Central into <volume>/jars (upload it there by hand if the cluster is offline)."""
    patterns = ["/databricks/jars/**/*commons-math3*.jar", "/databricks/**/commons-math3*.jar",
                "/usr/share/java/commons-math3*.jar", os.path.expanduser("~/.m2/**/commons-math3-*.jar"),
                f"{volume}/jars/commons-math3*.jar"]
    for pat in patterns:
        found = sorted(glob.glob(pat, recursive=True))
        if found:
            return found[0]
    target = f"{volume}/jars/commons-math3-3.6.1.jar"
    import urllib.request
    urllib.request.urlretrieve("https://repo1.maven.org/maven2/org/apache/commons/commons-math3/3.6.1/"
                               "commons-math3-3.6.1.jar", target)
    return target


try:
    math3 = [find_math3(volume)]
except Exception as e:
    math3 = []
    problems.append(f"no commons-math3 jar found or downloadable ({e}); upload commons-math3-3.6.1.jar to {volume}/jars/")
if problems:
    raise RuntimeError("fix before continuing:\n - " + "\n - ".join(problems))
classpath = f"{jar}:{math3[0]}"

# every model's artifacts must hold a cutoff for this eps and delta
for m, d in model_dirs.items():
    cut_m = pd.read_csv(f"{d}/cutoffs.csv")
    if cut_m[np.isclose(cut_m.eps, eps) & np.isclose(cut_m.delta, delta)].empty:
        raise RuntimeError(f"{d}/cutoffs.csv has no rows for eps={eps}, delta={delta}; it has "
                           f"{sorted(set(zip(cut_m.eps, cut_m.delta)))}")
print("ok. artifacts:", model_dirs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · One query set for every setup
# MAGIC
# MAGIC The held-out **test** queries (30% of query ids, the split the artifacts were calibrated against; queries with
# MAGIC fewer than 10 true points are excluded as in training). A random sample of `n_queries` is written once and used
# MAGIC by every setup.

# COMMAND ----------

from features import build_table  # noqa: E402
from gate_eval import cutoff, split_ids  # noqa: E402

df = build_table(run_dir, ds_dir)          # one row per (query, merge rate), with relErr and skew_pred
tr_ids, ca_ids, te_ids = split_ids(df.qId, seed)
all_q = pd.read_csv(f"{ds_dir}/queries-Ans.txt", header=None, names=["minLon", "maxLon", "minLat", "maxLat", "gt"])
test_ids = np.array(sorted(te_ids))
if get("n_queries").strip().lower() != "all" and int(get("n_queries")) < len(test_ids):
    test_ids = np.array(sorted(np.random.default_rng(seed).choice(test_ids, int(get("n_queries")), replace=False)))
query_file = f"{out_dir}/queries.csv"
all_q.loc[test_ids].to_csv(query_file, header=False, index=False)
print(f"{len(test_ids):,} held-out queries -> {query_file}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Skewness-Align cutoffs
# MAGIC
# MAGIC A gate needs a cutoff on the score. The learned model's cutoffs were calibrated on the calibration queries so that
# MAGIC at most `delta` of the answered queries exceed `eps`. Skewness-Align gets the same treatment: for every merge rate,
# MAGIC the largest cutoff on its score with that property on the **same calibration queries** (`gate_eval.cutoff`).

# COMMAND ----------

cal = df[df.qId.isin(ca_ids)]
rows = []
for rate in np.sort(cal.mergeRate.unique()):
    c = cal[cal.mergeRate == rate]
    rows.append({"mergeRate": rate, "eps": eps, "delta": delta,
                 "cutoff": cutoff(c.skew_pred.to_numpy(), c.relErr.to_numpy(), eps, delta)})
skew_cutoffs = f"{skew_dir}/cutoffs.csv"
pd.DataFrame(rows).to_csv(skew_cutoffs, index=False)
print(f"wrote {skew_cutoffs} ({len(rows)} merge rates, calibrated on {cal.qId.nunique():,} queries)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Run every setup
# MAGIC
# MAGIC Each run is timed from the start of `ApproxQueryExp` to the finished Parquet file. Setups and merge rates are
# MAGIC interleaved within each repeat so that cluster noise does not favour one setup. Exact does not depend on the merge
# MAGIC rate, so it runs once per repeat.

# COMMAND ----------

available = np.sort(df.mergeRate.unique())
rates = sorted({float(available[np.argmin(np.abs(available - r))]) for r in wanted_rates}, reverse=True)
print("merge rates used:", [round(r, 4) for r in rates])


def java_args(setup, rate, out_csv):
    a = ["-queryFile", query_file, "-outFile", out_csv, "-dataDir", data_dir + "/", "-numFile", n_files]
    if setup == "exact":
        return ["-mode", "exact"] + a
    a += ["-mode", "gated", "-runDir", run_dir, "-eps", eps, "-delta", delta, "-mergeRate", repr(rate)]
    if setup in model_dirs:
        a += ["-modelPath", f"{model_dirs[setup]}/model.txt", "-cutoffs", f"{model_dirs[setup]}/cutoffs.csv"]
    else:
        a += ["-cutoffs", skew_cutoffs]
    return a


def run_one(setup, rate, repeat):
    tag = "exact" if setup == "exact" else f"{setup}_{rate:.4f}"
    out_csv = f"{out_dir}/{tag}_r{repeat}.csv"
    t0 = time.perf_counter()
    r = subprocess.run([java, "-cp", classpath, "sigspatial25.ApproxQueryExp", *map(str, java_args(setup, rate, out_csv))],
                       capture_output=True, text=True)
    t1 = time.perf_counter()
    if r.returncode != 0:
        print(r.stderr[-4000:])
        raise RuntimeError(f"{setup} at merge rate {rate} exited with {r.returncode}")
    pq = out_csv[:-4] + ".parquet"
    pd.read_csv(out_csv).to_parquet(pq, index=False)        # what ml/to_parquet.py does
    t2 = time.perf_counter()
    return {"setup": setup, "rate": None if setup == "exact" else rate, "repeat": repeat,
            "java_s": t1 - t0, "parquet_s": t2 - t1, "total_s": t2 - t0, "parquet": pq}


pd.DataFrame({"warm_up": [1.0]}).to_parquet(f"{out_dir}/_warm_up.parquet", index=False)   # first use loads pyarrow
runs = []
for repeat in range(repeats):
    for setup in ("skewness", *model_names):
        for rate in rates:
            runs.append(run_one(setup, rate, repeat))
            print(f"repeat {repeat} {setup:12s} merge rate {rate:.3f}: {runs[-1]['total_s']:.1f}s")
    runs.append(run_one("exact", None, repeat))
    print(f"repeat {repeat} exact                    : {runs[-1]['total_s']:.1f}s")
timing = pd.DataFrame(runs)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Results
# MAGIC
# MAGIC Error, coverage and per-query times come from the answers of the last repeat (every repeat gives the same
# MAGIC answers); latency is the median over repeats.

# COMMAND ----------

from sklearn.metrics import roc_auc_score  # noqa: E402

def score_auc(a, rate):
    """AUC of the setup's error score at telling queries whose synopsis answer is over eps from those that are not.
    Uses every query (also the ones sent to raw data), so it does not depend on the cutoff or on coverage."""
    rows = df[np.isclose(df.mergeRate.astype(float), rate)][["qId", "relErr"]]
    # the answers' qId is the line in the sampled query file; map it back to the dataset's query id
    m = a[["qId", "errorScore"]].assign(qId=test_ids[a.qId.to_numpy()]).merge(rows, on="qId")
    bad = m.relErr > eps
    return float(roc_auc_score(bad, m.errorScore)) if bad.nunique() == 2 else np.nan


records = []
SETUPS = ["skewness", *model_names, "exact"]
for setup in SETUPS:
    for rate in ([None] if setup == "exact" else rates):
        t = timing[(timing.setup == setup) & (timing.rate.isna() if rate is None else np.isclose(timing.rate.astype(float), rate))]
        a = pd.read_parquet(t.sort_values("repeat").parquet.iloc[-1])
        syn = a[a.fromSynopsis] if setup != "exact" else a.iloc[0:0]
        records.append({
            "setup": setup, "merge_rate": np.nan if rate is None else round(rate, 4), "queries": len(a),
            "coverage": float(a.fromSynopsis.mean()) if setup != "exact" else 0.0,
            "mean_error": float(a.relErr.mean()), "p95_error": float(a.relErr.quantile(0.95)),
            "violation": float((syn.relErr > eps).mean()) if len(syn) else np.nan,
            "synopsis_queries": len(syn),
            "synopsis_mean_error": float(syn.relErr.mean()) if len(syn) else np.nan,
            "synopsis_p95_error": float(syn.relErr.quantile(0.95)) if len(syn) else np.nan,
            "score_auc": score_auc(a, rate) if setup != "exact" else np.nan,
            "latency_s": float(t.total_s.median()), "latency_min_s": float(t.total_s.min()),
            "latency_max_s": float(t.total_s.max()), "java_s": float(t.java_s.median()),
            "parquet_s": float(t.parquet_s.median()),
            "gate_ms_per_query": float(a.gateNs.mean() / 1e6), "synopsis_ms_per_query": float(a.synNs.mean() / 1e6),
            "raw_ms_per_query": float(a.rawNs.mean() / 1e6),
        })
summary = pd.DataFrame(records)
summary.to_csv(f"{out_dir}/summary.csv", index=False)

counts = summary.groupby("setup").queries.unique()
if any(len(v) != 1 for v in counts) or len({int(v[0]) for v in counts}) != 1:
    raise RuntimeError(f"the setups answered different numbers of queries: {counts.to_dict()}")
exact_err = summary[summary.setup == "exact"].mean_error.iloc[0]
if exact_err > 1e-6:
    print(f"WARNING: exact mode has mean relative error {exact_err:.2e} against the stored counts; "
          "check that -dataDir holds the same points the query counts were made from")
print(f"{int(summary.queries.iloc[0]):,} queries per setup")
show = summary.round(4)
display(show) if "display" in globals() else print(show.to_string())  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ### Side by side
# MAGIC
# MAGIC One table per metric, merge rates down, setups across. For the gated setups the guarantee is that the
# MAGIC `violation` stays near or below `delta`; with that held, more coverage means fewer raw-data scans.
# MAGIC
# MAGIC **Read `mean_error` and `p95_error` with coverage in mind.** They are taken over all queries, and a query sent
# MAGIC to raw data has error 0, so a gate that accepts almost nothing looks very accurate. `synopsis_*` columns use only
# MAGIC the queries answered from the synopsis (`synopsis_queries` of them; few queries make the numbers noisy).
# MAGIC `score_auc` does not depend on the cutoff or coverage: it is how well each setup's error score ranks queries whose
# MAGIC synopsis answer is over `eps` above those that are not (0.5 = no better than chance).

# COMMAND ----------

gated = summary[summary.setup != "exact"]
for metric, fmt in (("coverage", "{:.1%}"), ("violation", "{:.1%}"), ("score_auc", "{:.3f}"),
                    ("synopsis_queries", "{:.0f}"), ("synopsis_mean_error", "{:.4f}"),
                    ("synopsis_p95_error", "{:.4f}"), ("mean_error", "{:.4f}"),
                    ("p95_error", "{:.4f}"), ("latency_s", "{:.2f}")):
    p = gated.pivot(index="merge_rate", columns="setup", values=metric).sort_index(ascending=False)
    if metric in ("mean_error", "p95_error", "latency_s"):
        p["exact"] = summary[summary.setup == "exact"][metric].iloc[0]
    print(f"\n{metric}")
    print(p.to_string(float_format=lambda v: fmt.format(v)))

lp = gated.pivot(index="merge_rate", columns="setup", values="coverage")
vp = gated.pivot(index="merge_rate", columns="setup", values="violation")
for m in model_names:
    print(f"\n{m}: higher coverage than Skewness-Align at {(lp[m] > lp.skewness).sum()} of {len(lp)} merge rates; "
          f"violation above delta={delta:.0%} (+1 point slack) at {(vp[m] > delta + 0.01).sum()} "
          f"(Skewness-Align: {(vp.skewness > delta + 0.01).sum()}) of {len(vp)}")

# COMMAND ----------

import matplotlib.pyplot as plt  # noqa: E402

palette = ["#1f5fbf", "#d1603d", "#4a9b8e", "#b58a4a", "#7b5ea7", "#3d8f3d"]
colors = {"skewness": "#8d94a3", "exact": "#222222", **{m: palette[i % len(palette)] for i, m in enumerate(model_names)}}
fig, ax = plt.subplots(1, 3, figsize=(15, 4))
for setup in ("skewness", *model_names):
    g = gated[gated.setup == setup].sort_values("merge_rate")
    ax[0].plot(g.merge_rate, g.coverage * 100, "o-", color=colors[setup], label=setup)
    ax[1].plot(g.merge_rate, g.mean_error, "o-", color=colors[setup], label=setup)
    ax[2].plot(g.merge_rate, g.latency_s, "o-", color=colors[setup], label=setup)
ex = summary[summary.setup == "exact"].iloc[0]
ax[1].axhline(ex.mean_error, color=colors["exact"], ls="--", label="exact")
ax[2].axhline(ex.latency_s, color=colors["exact"], ls="--", label="exact")
for a_, title, yl in zip(ax, ("Coverage", "Mean relative error of the returned answers", "Latency to the Parquet file"),
                         ("% answered from the synopsis", "mean relative error", "seconds (median)")):
    a_.set_title(title); a_.set_xlabel("merge rate"); a_.set_ylabel(yl); a_.invert_xaxis(); a_.legend()
plt.tight_layout()
display(fig) if "display" in globals() else plt.savefig(f"{out_dir}/comparison.png", dpi=120)  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ### Where the time goes
# MAGIC
# MAGIC Wall-clock latency splits into the Java run (JVM start, loading synopses and raw points, answering) and the Parquet
# MAGIC conversion. The per-query columns are what Java logged for the gate, the synopsis lookup and the raw-data scan,
# MAGIC averaged over all queries.

# COMMAND ----------

split = summary[["setup", "merge_rate", "java_s", "parquet_s", "gate_ms_per_query", "synopsis_ms_per_query",
                 "raw_ms_per_query"]].round(4)
display(split) if "display" in globals() else print(split.to_string())  # noqa: F821
print(f"\nresults saved to {out_dir}/summary.csv; answers per run are in {out_dir}/*.parquet")