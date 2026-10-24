# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 00 · Unpack the development fixture
# MAGIC
# MAGIC `ml-dev-fixture.zip` holds a small world and Europe sample in the `ml/SCHEMA.md` formats, generated
# MAGIC outside Databricks. It exists only so workstreams can write and test code before notebook 01 produces
# MAGIC the real sample on Databricks (task F4). **Do not report numbers from it.**
# MAGIC
# MAGIC Before running: upload `ml-dev-fixture.zip` to the `output` volume
# MAGIC (Catalog → cs598_ccc_research_… → synopsis_lake → Volumes → output → Upload to this volume).

# COMMAND ----------

import os
import zipfile

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"
try:
    dbutils.widgets.text("volume", VOL)  # noqa: F821
    get = dbutils.widgets.get  # noqa: F821
except NameError:
    def get(name):
        return os.environ[name.upper()]

volume = get("volume")
zip_path = f"{volume}/output/ml-dev-fixture.zip"

# The zip holds output/ml/dev_fixture/<dataset>/... and data/ml/dev_fixture/<dataset>/...
with zipfile.ZipFile(zip_path) as z:
    for member in z.namelist():
        if member.endswith("/"):
            continue
        target = os.path.join(volume, member)
        os.makedirs(os.path.dirname(target), exist_ok=True)
        with z.open(member) as src, open(target, "wb") as dst:
            dst.write(src.read())

for root in ["output/ml/dev_fixture", "data/ml/dev_fixture"]:
    for ds in sorted(os.listdir(os.path.join(volume, root))):
        print(f"{root}/{ds}: {len(os.listdir(os.path.join(volume, root, ds)))} files")