# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # Environment check
# MAGIC
# MAGIC Read-only apart from one test file it writes and deletes in the `output` volume.
# MAGIC `OK` = present or working, `--` = missing or failed.

# COMMAND ----------

import datetime
import glob
import importlib
import os
import subprocess
import zipfile

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"


def ok(label, cond, detail=""):
    print(("OK  " if cond else "--  ") + label + (f"  ({detail})" if detail else ""))


# 1. Volumes you can read
for v in ["jars", "data", "output"]:
    try:
        ok(f"volume {v} readable", True, ", ".join(sorted(os.listdir(f"{VOL}/{v}"))[:8]))
    except Exception as e:
        ok(f"volume {v} readable", False, str(e)[:80])

# 2. The CI-built jar: when it was built, and whether it has the logging class (F2)
jar = f"{VOL}/jars/synopsislake.jar"
if os.path.exists(jar):
    built = datetime.datetime.fromtimestamp(os.path.getmtime(jar))
    ok("jar exists", True, f"last built {built:%Y-%m-%d %H:%M} UTC")
    ok("jar has ErrorPredictionLogExp (F2)", any("ErrorPredictionLogExp" in n for n in zipfile.ZipFile(jar).namelist()))
else:
    ok("jar exists", False)

# 3. Can you write to the output volume?
test = f"{VOL}/output/_write_test_{os.getpid()}"
try:
    with open(test, "w") as f:
        f.write("x")
    os.remove(test)
    ok("can write to output volume", True)
except Exception as e:
    ok("can write to output volume", False, str(e)[:80])

# 4. Is anything already there?
for p in ["data/ml", "data/range", "output/ml/dev_fixture", "output/ml/sample", "output/ml-dev-fixture.zip"]:
    ok(f"{p} exists", os.path.exists(f"{VOL}/{p}"))

# 5. This compute: runtime, Java, commons-math3, Python packages
try:
    ok("runtime", True, spark.conf.get("spark.databricks.clusterUsageTags.sparkVersion"))  # noqa: F821
except Exception as e:
    ok("runtime", False, str(e)[:60])
try:
    r = subprocess.run(["java", "-version"], capture_output=True, text=True)
    version = next((ln for ln in (r.stderr + r.stdout).splitlines() if "version" in ln), "")
    ok("java available (notebook 01 needs it)", r.returncode == 0, version)
except Exception as e:
    ok("java available (notebook 01 needs it)", False, str(e)[:60])
ok("commons-math3 in /databricks/jars", bool(glob.glob("/databricks/jars/*commons-math3*")))
for m in ["lightgbm", "mlflow", "sklearn", "scipy", "pandas"]:
    try:
        ok(f"import {m}", True, getattr(importlib.import_module(m), "__version__", ""))
    except Exception as e:
        ok(f"import {m}", False, str(e)[:60])

# COMMAND ----------

# What is in the data and output volumes, two levels deep (first 20 entries per folder)
for v in ["data", "output"]:
    root = f"{VOL}/{v}"
    for name in sorted(os.listdir(root))[:20]:
        p = f"{root}/{name}"
        if os.path.isdir(p):
            kids = sorted(os.listdir(p))
            print(f"{v}/{name}/  ({len(kids)} entries)  {', '.join(kids[:6])}{' …' if len(kids) > 6 else ''}")
        else:
            print(f"{v}/{name}  ({os.path.getsize(p):,} bytes)")

# COMMAND ----------

  names = zipfile.ZipFile(f"{VOL}/jars/synopsislake.jar").namelist()
  print("commons-math3 inside the jar:", any(n.startswith("org/apache/commons/math3/") for n in names))