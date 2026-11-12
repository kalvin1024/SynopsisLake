# Databricks notebook source
# MAGIC %md
# MAGIC # 07 · End to end: train the error predictor and answer queries approximately
# MAGIC
# MAGIC One notebook that does everything needed to run SynopsisLake with a trained error predictor and get an
# MAGIC **approximate answer plus its estimated error** for every query:
# MAGIC
# MAGIC 1. check the inputs (jar, data, Java)
# MAGIC 2. prepare the query dataset (`queries-Ans.txt`)
# MAGIC 3. build the merged synopses for every merge rate (`ErrorPredictionLogExp`)
# MAGIC 4. train the LightGBM error predictor and calibrate the gate (`model.txt`, `model.properties`, `cutoffs.csv`)
# MAGIC 5. answer held-out queries from the synopsis with `ApproxQueryExp` (`approx.csv`)
# MAGIC 6. compare the estimated error with the true error, and ask your own query
# MAGIC
# MAGIC **Needs a classic dedicated (single user) cluster on an ML runtime** (LightGBM, a `java` on the PATH and a
# MAGIC commons-math3 jar, which is downloaded into the volume if the cluster has none). Serverless and shared clusters cannot start `java` with jars under
# MAGIC `/databricks/jars`.
# MAGIC
# MAGIC **Inputs** (uploaded by the CI workflow `build-upload-and-run.yml`, or by hand):
# MAGIC * `<volume>/jars/synopsislake.jar`, built from a commit that contains `sigspatial25/ApproxQueryExp`
# MAGIC * `<volume>/data/range_query/0.csv … 3.csv` and `dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv`
# MAGIC
# MAGIC **Outputs** are under `<volume>/output/ml/<dataset>/`: the run folder (`rows.csv`, `src_*`, `arm_*`),
# MAGIC `gate/` (the trained model and cutoffs) and `answers/` (`approx.csv`, `plan.csv`).
# MAGIC
# MAGIC Run the first code cell on its own so the input boxes appear at the top, adjust them, then run all.
# MAGIC Re-running is cheap: the synopses and the model are reused unless you turn on `rebuild_synopses` /
# MAGIC `retrain`.

# COMMAND ----------

# MAGIC %pip install lightgbm

# COMMAND ----------

import os

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"


def _repo_root():
    """Folder that holds ml/ (this notebook is …/ml/notebooks/07_end_to_end_approx)."""
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
    "dataset": "authors_range",
    "seed": "0",
    "eps": "0.10",             # an answer is "good" if its relative error is at most eps
    "delta": "0.05",           # the gate tolerates at most this share of answers above eps
    "merge_rate": "0.5",       # which synopsis to answer from: a merge rate (1.0 = largest), or "plan" to let the
                               # planner choose the smallest one that meets eps/delta
    "mode": "approx",          # approx: synopsis only | gated: synopsis, raw data for rejected queries
                               # | exact: raw data only (slow, the ground truth)
    "rebuild_synopses": "false",
    "retrain": "false",
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
import zipfile

import numpy as np
import pandas as pd

volume, repo, dataset = get("volume"), get("repo_root"), get("dataset")
seed, eps, delta = int(get("seed")), float(get("eps")), float(get("delta"))
merge_rate, mode, java = get("merge_rate").strip(), get("mode").strip(), get("java")
if mode not in ("approx", "gated", "exact"):
    raise ValueError("mode must be approx, gated or exact")
use_plan = merge_rate.lower() == "plan"

jar = f"{volume}/jars/synopsislake.jar"
data_dir = f"{volume}/data/range_query"
src_query_file = f"{data_dir}/dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv"
ds_dir = f"{volume}/data/ml/{dataset}"           # queries-Ans.txt, queries-meta.csv
run_dir = f"{volume}/output/ml/{dataset}"        # rows.csv, src_*.txt, arm_*.txt
gate_dir = f"{run_dir}/gate"                     # model.txt, model.properties, cutoffs.csv
ans_dir = f"{run_dir}/answers"                   # approx.csv, plan.csv
for d in (gate_dir, ans_dir):
    os.makedirs(d, exist_ok=True)

sys.path.insert(0, os.path.join(repo, "ml"))
print("volume", volume, "| dataset", dataset, "| mode", mode, "| merge_rate", merge_rate,
      "| eps", eps, "delta", delta, "| seed", seed)

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
        classes = set(z.namelist())
    for c in ("sigspatial25/ErrorPredictionLogExp.class", "sigspatial25/ApproxQueryExp.class"):
        if c not in classes:
            problems.append(f"{jar} has no {c}: it was built from a commit without the query path (rebuild it)")
n_files = 4
for i in range(n_files):
    if not os.path.exists(f"{data_dir}/{i}.csv"):
        problems.append(f"{data_dir}/{i}.csv is missing (gunzip the .gz files if you uploaded those)")
if not os.path.exists(src_query_file):
    problems.append(f"{src_query_file} is missing")

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
print("ok. classpath:", classpath)


def run_java(main_class, *args, ok=(0,)):
    """Run a SynopsisLake main class; print its summary lines. Returns the exit code."""
    cmd = [java, "-cp", classpath, main_class, *map(str, args)]
    print(" ".join(cmd))
    r = subprocess.run(cmd, capture_output=True, text=True)
    out = [ln for ln in r.stdout.splitlines() if ln.strip()]
    print("\n".join(out[-12:]))
    if r.returncode not in ok:
        print(r.stderr[-5000:])
        raise RuntimeError(f"{main_class} exited with {r.returncode}")
    return r.returncode

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2 · Prepare the query dataset
# MAGIC
# MAGIC Copies the authors' range queries (`minLon,maxLon,minLat,maxLat,exactCount`) into the dataset folder; the
# MAGIC line number is the query id.

# COMMAND ----------

from prepare_authors_range import prepare  # noqa: E402

meta = prepare(src_query_file, ds_dir)
print(f"{len(meta):,} queries -> {ds_dir}; per scale: {meta.scale.value_counts().sort_index().to_dict()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3 · Build the merged synopses
# MAGIC
# MAGIC `ErrorPredictionLogExp` builds the source histograms and one merged histogram per merge rate, saves them as
# MAGIC `src_*.txt` / `arm_*.txt`, and logs `rows.csv` (the true error of every query on every synopsis, which is the
# MAGIC training data). Skipped when `rows.csv` already exists, unless `rebuild_synopses` is true.

# COMMAND ----------

rows_csv = f"{run_dir}/rows.csv"
if os.path.exists(rows_csv) and get("rebuild_synopses").lower() != "true":
    print("reusing", run_dir, "(set rebuild_synopses=true to redo)")
else:
    run_java("sigspatial25.ErrorPredictionLogExp",
             "-dataDir", data_dir + "/", "-numFile", n_files, "-srcRes", "40,56,32,48",
             "-queryFile", src_query_file, "-outDir", run_dir + "/", "-end", 42, "-step", 8,
             "-reproduceAuthors", "true", "-fixStride", "false")
rows = pd.read_csv(rows_csv, usecols=["mergeRate", "qId", "relErr"])
print(f"{rows.qId.nunique():,} queries x {rows.mergeRate.nunique()} merge rates; "
      f"mean relErr per rate:\n{rows.groupby('mergeRate').relErr.mean().round(3).to_string()}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4 · Train the error predictor and calibrate the gate
# MAGIC
# MAGIC * Label: log relative error; queries with fewer than 10 true points are dropped.
# MAGIC * Split by query id: 50% train, 20% calibration, 30% test (no query appears in two splits).
# MAGIC * `gate_artifacts` writes `cutoffs.csv` (one cutoff per merge rate, calibrated on the calibration queries so that
# MAGIC   at most `delta` of the answered queries exceed `eps`) and `model.properties`.
# MAGIC
# MAGIC Skipped when the model files exist, unless `retrain` is true.

# COMMAND ----------

from features import FEATURES, build_table  # noqa: E402
from gate_artifacts import write_gate_artifacts  # noqa: E402
from gate_eval import PARAMS, fit, split_ids  # noqa: E402

df = build_table(run_dir, ds_dir)
tr_ids, ca_ids, te_ids = split_ids(df.qId, seed)
model_path = f"{gate_dir}/model.txt"
cutoffs_path = f"{gate_dir}/cutoffs.csv"
props_path = f"{gate_dir}/model.properties"

if all(os.path.exists(p) for p in (model_path, cutoffs_path, props_path)) and get("retrain").lower() != "true":
    print("reusing the model in", gate_dir, "(set retrain=true to retrain)")
else:
    model = fit(df[df.qId.isin(tr_ids)], FEATURES, seed, PARAMS)
    model.booster_.save_model(model_path)                       # LightGBM text format, read by Java
    report = write_gate_artifacts(model, FEATURES, df, gate_dir, seed, eps_list=sorted({eps, 0.10, 0.25}), delta=delta)
    print(f"trained on {len(tr_ids):,} queries ({model.best_iteration_} trees); wrote model.txt, model.properties, "
          f"cutoffs.csv, gate_report.csv to {gate_dir}")
    print("held-out check (test queries): share answered from the synopsis and share of those above eps")
    display(report[report.eps == eps].round(4)) if "display" in globals() else print(report.round(4).to_string())  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5 · Answer queries with `ApproxQueryExp`
# MAGIC
# MAGIC The held-out **test** queries are answered from the synopsis. Each gets the approximate count and the model's
# MAGIC estimated relative error, and `fromSynopsis` says whether the gate trusts it (estimated error within the
# MAGIC cutoff). The queries carry their true count only so we can score the estimates afterwards; the answers do not
# MAGIC use it.
# MAGIC
# MAGIC With `merge_rate = plan` the test queries are split in half: one half is used by the planner to pick the
# MAGIC smallest merge rate that meets eps/delta, the other half is answered. (Calibration queries are not reused for
# MAGIC planning because the cutoffs were fitted on them.)

# COMMAND ----------

all_q = pd.read_csv(f"{ds_dir}/queries-Ans.txt", header=None, names=["minLon", "maxLon", "minLat", "maxLat", "gt"])
test_ids = sorted(te_ids)
rng = np.random.default_rng(seed)
shuffled = rng.permutation(test_ids)
plan_ids, eval_ids = sorted(shuffled[:len(shuffled) // 2]), sorted(shuffled[len(shuffled) // 2:])
if not use_plan:
    eval_ids = test_ids

query_file = f"{ans_dir}/queries.csv"
plan_query_file = f"{ans_dir}/plan_queries.csv"
all_q.loc[eval_ids].to_csv(query_file, header=False, index=False)
all_q.loc[plan_ids].to_csv(plan_query_file, header=False, index=False)
print(f"{len(eval_ids):,} queries to answer" + (f", {len(plan_ids):,} for planning" if use_plan else ""))

approx_csv = f"{ans_dir}/approx.csv"
args = ["-mode", mode, "-queryFile", query_file, "-outFile", approx_csv]
if mode != "exact":
    args += ["-runDir", run_dir, "-modelPath", model_path, "-cutoffs", cutoffs_path, "-eps", eps, "-delta", delta]
    args += ["-plan", "true", "-planQueryFile", plan_query_file] if use_plan else ["-mergeRate", merge_rate]
if mode in ("gated", "exact"):
    args += ["-dataDir", data_dir + "/", "-numFile", n_files]
code = run_java("sigspatial25.ApproxQueryExp", *args, ok=(0, 2))
if code == 2:
    print("The planner found no merge rate that meets eps/delta. Plan table:")
    display(pd.read_csv(approx_csv + ".plan.csv")) if "display" in globals() else None  # noqa: F821
    raise SystemExit("rerun with a larger eps, mode=gated, or a fixed merge_rate")
if use_plan:
    plan = pd.read_csv(approx_csv + ".plan.csv")
    print("planner table (one row per merge rate):")
    display(plan) if "display" in globals() else print(plan.to_string())  # noqa: F821

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6 · Results
# MAGIC
# MAGIC Columns of `approx.csv`: `answer` (the count returned), `fromSynopsis` (the gate trusts the synopsis),
# MAGIC `estRelError` (the model's estimate of the answer's relative error), `errorScore`/`cutoff` (what the gate
# MAGIC compared), `gateNs`/`synNs`/`rawNs` (timings), and `exact`/`relErr` (the true count and error, for scoring).

# COMMAND ----------

res = pd.read_csv(approx_csv)
show = ["qId", "mergeRate", "answer", "fromSynopsis", "estRelError", "exact", "relErr"]
display(res[show].head(20)) if "display" in globals() else print(res[show].head(20).to_string())  # noqa: F821

trusted = res[res.fromSynopsis]
summary = {
    "queries": len(res),
    "answered from synopsis (trusted)": f"{len(trusted) / len(res):.1%}",
    f"trusted answers with true error > {eps}": f"{(trusted.relErr > eps).mean():.1%}" if len(trusted) else "n/a",
    "mean true error, trusted": f"{trusted.relErr.mean():.4f}" if len(trusted) else "n/a",
    "mean true error, all": f"{res.relErr.mean():.4f}",
    "mean estimated error, all": f"{res.estRelError.mean():.4f}" if res.estRelError.notna().any() else "n/a",
    "median gate ns / synopsis ns": f"{res.gateNs.median():.0f} / {res.synNs.median():.0f}",
}
print("\n".join(f"{k:45s} {v}" for k, v in summary.items()))
print(f"\ngoal: trusted answers above eps should be at most delta = {delta:.0%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Does the estimated error track the true error?
# MAGIC
# MAGIC Bucket the queries by estimated error and compare with the true error in each bucket.

# COMMAND ----------

ok = res.dropna(subset=["estRelError", "relErr"])
if len(ok) >= 20:
    ok = ok.assign(bucket=pd.qcut(ok.estRelError, 5, duplicates="drop"))
    calib = ok.groupby("bucket", observed=True).agg(queries=("qId", "size"), mean_estimated=("estRelError", "mean"),
                                                    mean_true=("relErr", "mean"),
                                                    share_above_eps=("relErr", lambda s: (s > eps).mean()))
    display(calib.round(4).reset_index()) if "display" in globals() else print(calib.round(4).to_string())  # noqa: F821
    print(f"rank correlation, estimated vs true error: {ok.estRelError.corr(ok.relErr, method='spearman'):.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Ask your own query
# MAGIC
# MAGIC Edit the bounds and run this cell to get the approximate answer and its estimated error for one range.
# MAGIC Uses the merge rate above (or 0.5 if you chose `plan`).

# COMMAND ----------

MIN_LON, MAX_LON, MIN_LAT, MAX_LAT = -10.0, 30.0, 35.0, 60.0   # edit me

one_q = f"{ans_dir}/one_query.csv"
with open(one_q, "w") as f:
    f.write(f"{MIN_LON},{MAX_LON},{MIN_LAT},{MAX_LAT}\n")
one_out = f"{ans_dir}/one_answer.csv"
run_java("sigspatial25.ApproxQueryExp", "-mode", "approx", "-runDir", run_dir, "-modelPath", model_path,
         "-cutoffs", cutoffs_path, "-eps", eps, "-delta", delta, "-mergeRate", 0.5 if use_plan else merge_rate,
         "-queryFile", one_q, "-outFile", one_out)
a = pd.read_csv(one_out).iloc[0]
print(f"approximate count {a.answer:,.0f}, estimated relative error {a.estRelError:.1%}, "
      f"{'trusted' if a.fromSynopsis else 'NOT trusted (read the raw data for an exact count)'}")
