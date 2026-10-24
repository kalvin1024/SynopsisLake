# Databricks notebook source
# MAGIC %md
# MAGIC # 01b · Run the logger from a notebook (optional)
# MAGIC
# MAGIC The `ml_xy_pipeline` job runs `sigspatial25.ErrorPredictionLogExp` as a JAR task. This notebook is the
# MAGIC manual alternative: it calls `java` in a subprocess with the CI-built jar plus the cluster's
# MAGIC commons-math3 jar, and writes the same run folder `<volume>/output/ml/<dataset>/`.
# MAGIC
# MAGIC **Classic dedicated (single user) cluster only.** Serverless and shared clusters do not let a
# MAGIC notebook start `java` with the jars under `/databricks/jars`.
# MAGIC
# MAGIC Inputs: `<volume>/data/range_query/0.csv … 3.csv` and the query file (uploaded by CI).

# COMMAND ----------

import glob
import os
import subprocess

import pandas as pd

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"
dbutils.widgets.text("volume", VOL)  # noqa: F821
dbutils.widgets.text("dataset", "authors_range")  # noqa: F821
dbutils.widgets.dropdown("reproduce_authors", "true", ["true", "false"])  # noqa: F821
dbutils.widgets.dropdown("fix_stride", "false", ["true", "false"])  # noqa: F821
get = dbutils.widgets.get  # noqa: F821

volume, dataset = get("volume"), get("dataset")
data_dir = f"{volume}/data/range_query"
query_file = f"{data_dir}/dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv"
out_dir = f"{volume}/output/ml/{dataset}/"

# COMMAND ----------

jar = f"{volume}/jars/synopsislake.jar"
math3 = sorted(glob.glob("/databricks/jars/*commons-math3*.jar") + glob.glob("/databricks/jars/**/*commons-math3*.jar",
                                                                              recursive=True))
if not math3:
    raise FileNotFoundError("no commons-math3 jar under /databricks/jars (needs a classic dedicated cluster)")
cmd = ["java", "-cp", f"{jar}:{math3[0]}", "sigspatial25.ErrorPredictionLogExp",
       "-dataDir", data_dir + "/", "-numFile", "4", "-srcRes", "40,56,32,48", "-queryFile", query_file,
       "-outDir", out_dir, "-end", "42", "-step", "8",
       "-reproduceAuthors", get("reproduce_authors"), "-fixStride", get("fix_stride")]
print(" ".join(cmd))
r = subprocess.run(cmd, capture_output=True, text=True)
print("\n".join(ln for ln in r.stdout.splitlines() if ln.startswith(("canonical", "mergeRate", "dataDir"))))
if r.returncode != 0:
    print(r.stderr[-5000:])
    raise RuntimeError(f"logger exited with {r.returncode}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reproduction check
# MAGIC
# MAGIC Mean relative error per merge rate against the authors' `ourLH` column. With `reproduce_authors = true`
# MAGIC the difference is 0.000 at every rate.

# COMMAND ----------

rows = pd.read_csv(out_dir + "rows.csv")
ours = (rows[rows.method == "ARM"].groupby("mergeRate")
        .agg(lonRes=("lonRes", "first"), mean_relErr=("relErr", "mean")).reset_index())
ours["MergeRate(%)"] = (ours.mergeRate * 100).round(1)
authors = pd.read_csv(f"{data_dir}/dataDrivenQuery/results/DataDrivenQueryAvgRelError.txt")[["MergeRate(%)", "ourLH"]]
check = ours.merge(authors, on="MergeRate(%)", how="left").sort_values("mergeRate", ascending=False)
check["diff"] = (check.mean_relErr - check.ourLH).abs().round(3)
check["mean_relErr"] = check.mean_relErr.round(3)
display(check[["MergeRate(%)", "lonRes", "mean_relErr", "ourLH", "diff"]])  # noqa: F821
print(f"max difference {check['diff'].max():.3f} (reproduce_authors = {get('reproduce_authors')})")
