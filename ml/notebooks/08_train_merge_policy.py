# Databricks notebook source
# MAGIC %md
# MAGIC # 08 · Train the learned merge policy and compare it with ARM
# MAGIC
# MAGIC ARM reshapes the aligned histogram by dropping boundaries in the order of a hand-written score. This notebook
# MAGIC trains a LightGBM model that picks the drops instead (`ml/merge_policy.py`), then runs the Java pipeline with
# MAGIC it (`ErrorPredictionLogExp -mergePolicy`) and compares the two on held-out queries.
# MAGIC
# MAGIC 1. check the inputs (jar with `predictor/MergePolicy`, a run folder, the queries)
# MAGIC 2. train on queries with known counts (the dataset's own, or uniformly placed synthetic ones)
# MAGIC 3. write `merge_policy.txt`
# MAGIC 4. rebuild the synopses in Java with the policy and compare mean relative error per merge rate
# MAGIC
# MAGIC **Needs** the run folder from notebook 07 (or the `run_logger` job task) and `<volume>/data/ml/<dataset>`
# MAGIC (notebook 01). A classic dedicated cluster on an ML runtime, as for notebook 07.
# MAGIC
# MAGIC **Training data.** The model learns from the true error of every candidate boundary drop on training queries,
# MAGIC so it needs queries with exact counts. `synthetic = true` trains on uniformly placed queries of many sizes
# MAGIC (exact counts from the raw points in `synthetic_data`), which does not assume the production workload:
# MAGIC on the datasets tried this beat ARM at every merge rate. `synthetic = false` trains on the dataset's own
# MAGIC queries, which is better when the workload is known, and can be much worse than ARM when it changes.
# MAGIC The policy is trained per dataset.

# COMMAND ----------

# MAGIC %pip install lightgbm

# COMMAND ----------

import os

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"


def _repo_root():
    """Folder that holds ml/ (this notebook is …/ml/notebooks/08_train_merge_policy)."""
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
    "train_queries": "4000",
    "synthetic": "true",                         # train on uniformly placed synthetic queries (see above)
    "synthetic_data": VOL + "/data/range_query",   # folder with 0.csv.. tab-separated lon lat (for exact counts)
    "synthetic_width": "0.5,12.0",               # degrees, log-uniform; cover (and exceed) the real query sizes
    "synthetic_height": "0.25,6.0",
    "src_res": "40,56,32,48",
    "end": "42",
    "step": "8",
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

volume, repo, dataset, seed = get("volume"), get("repo_root"), get("dataset"), int(get("seed"))
synthetic = get("synthetic").lower() == "true"
run_dir = f"{volume}/output/ml/{dataset}"
ds_dir = f"{volume}/data/ml/{dataset}"
out_dir = f"{run_dir}/merge_policy"
model_path = f"{out_dir}/merge_policy.txt"
jar = f"{volume}/jars/synopsislake.jar"
data_dir = f"{volume}/data/range_query"
os.makedirs(out_dir, exist_ok=True)
sys.path.insert(0, os.path.join(repo, "ml"))

problems = []
if shutil.which(get("java")) is None:
    problems.append(f"`{get('java')}` not found (needs a classic cluster with a JDK)")
if not os.path.exists(jar):
    problems.append(f"{jar} is missing")
else:
    with zipfile.ZipFile(jar) as z:
        if "predictor/MergePolicy.class" not in z.namelist():
            problems.append(f"{jar} was built without predictor/MergePolicy: rebuild it from the learned-merge-policy branch")
if not glob.glob(f"{run_dir}/arm_*.txt"):
    problems.append(f"no run folder at {run_dir} (run notebook 07 or the run_logger task first)")
if not os.path.exists(f"{ds_dir}/queries-Ans.txt"):
    problems.append(f"{ds_dir}/queries-Ans.txt is missing (notebook 01)")
if synthetic and not glob.glob(f"{get('synthetic_data')}/*.csv"):
    problems.append(f"synthetic = true needs raw points in {get('synthetic_data')}/0.csv ..")
math3 = sorted(glob.glob("/databricks/jars/*commons-math3*.jar")
               + glob.glob("/databricks/jars/**/*commons-math3*.jar", recursive=True)
               + glob.glob(os.path.expanduser("~/.m2/**/commons-math3-*.jar"), recursive=True))
if not math3:
    problems.append("no commons-math3 jar under /databricks/jars")
if problems:
    raise RuntimeError("fix before continuing:\n - " + "\n - ".join(problems))
classpath = f"{jar}:{math3[0]}"
print("ok:", run_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train
# MAGIC
# MAGIC Starting from the aligned histogram, the training run drops boundaries one at a time, always the one whose
# MAGIC removal raises the training queries' error the least, and records every candidate's features and true error
# MAGIC increase. A LightGBM regressor learns that increase from the features alone. The test queries are the same
# MAGIC 30% split by query id used for the error predictor, and are never used for training.

# COMMAND ----------

import merge_policy as mp  # noqa: E402

aligned, resolutions = mp.Aligned.from_run_dir(run_dir)
train_labels, test, test_ids = mp.load_queries(ds_dir, seed, int(get("train_queries")))
if synthetic:
    files = sorted(glob.glob(f"{get('synthetic_data')}/*.csv"))
    points = np.vstack([pd.read_csv(f, sep="\t", header=None, usecols=[0, 1]).to_numpy() for f in files])
    width = [float(x) for x in get("synthetic_width").split(",")]
    height = [float(x) for x in get("synthetic_height").split(",")]
    q, gt = mp.synthetic_queries(points, 6 * int(get("train_queries")), np.random.default_rng(seed), width, height)
    keep = np.random.default_rng(seed).permutation(len(q))[:int(get("train_queries"))]
    train_labels = (q[keep], gt[keep])
    print(f"training on {len(keep):,} synthetic queries from {len(points):,} points")
else:
    print(f"training on {len(train_labels[1]):,} of the dataset's own queries")

model, is_lgb, rows = mp.train(aligned, train_labels, seed, resolutions)
if not is_lgb:
    raise RuntimeError("lightgbm is required to write a model Java can read (pip install lightgbm)")
model.booster_.save_model(model_path)
print(f"trained on {rows:,} candidate rows; wrote {model_path}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Quick check in Python
# MAGIC
# MAGIC Mean relative error on the held-out test queries at each resolution: ARM (the Java run's logged rows), the
# MAGIC learned policy, and evenly spaced boundaries.

# COMMAND ----------

py = mp.evaluate(aligned, resolutions, model, test, run_dir, test_ids)
try:
    display(py.reset_index().round(4))  # noqa: F821
except NameError:
    print(py.round(4).to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ## Run the Java pipeline with the policy
# MAGIC
# MAGIC `ErrorPredictionLogExp -mergePolicy` rebuilds the synopses, builds the learned histograms at every merge rate
# MAGIC (`learned_<res>.txt`) and logs `rows_learned.csv` next to ARM's `rows.csv`, on the same queries. Takes a minute or
# MAGIC two. Output goes to `<run folder>/merge_policy/java/`, leaving the main run folder untouched.

# COMMAND ----------

java_out = f"{out_dir}/java"
cmd = [get("java"), "-cp", classpath, "sigspatial25.ErrorPredictionLogExp",
       "-dataDir", data_dir + "/", "-numFile", "4", "-srcRes", get("src_res"),
       "-queryFile", f"{data_dir}/dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv", "-outDir", java_out + "/",
       "-end", get("end"), "-step", get("step"), "-reproduceAuthors", "true", "-fixStride", "false",
       "-mergePolicy", model_path]
print(" ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True)
print("\n".join(ln for ln in r.stdout.splitlines() if ln.startswith(("canonical", "mergeRate")))[-3000:])
if r.returncode != 0:
    print(r.stderr[-5000:])
    raise RuntimeError(f"ErrorPredictionLogExp exited with {r.returncode}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Results on held-out queries
# MAGIC
# MAGIC Mean relative error of the answers per merge rate (queries with at least 10 true points), ARM against the learned
# MAGIC merge policy, both computed in Java.

# COMMAND ----------

keep_ids = set(int(i) for i in test_ids)


def held_out(path):
    d = pd.read_csv(path, usecols=["lonRes", "qId", "relErr", "gt"])
    d = d[d.qId.isin(keep_ids) & (d["gt"] >= mp.MIN_EXACT)]
    return d.groupby("lonRes").relErr.mean()


cmp_ = pd.DataFrame({"ARM": held_out(f"{java_out}/rows.csv"), "learned": held_out(f"{java_out}/rows_learned.csv")})
cmp_["change"] = cmp_.learned / cmp_.ARM - 1
cmp_ = cmp_.sort_index(ascending=False)
try:
    display(cmp_.reset_index().round(4))  # noqa: F821
except NameError:
    print(cmp_.round(4).to_string())
full = cmp_.index.max()
low = cmp_[cmp_.index <= 0.73 * full]
print(f"merge rates at or below 73% of full resolution: ARM {low.ARM.mean():.4f}, learned {low.learned.mean():.4f} "
      f"({low.learned.mean() / low.ARM.mean() - 1:+.1%}); learned is lower at {(low.learned < low.ARM).sum()} of "
      f"{len(low)} resolutions")
