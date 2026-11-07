# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# dependencies = [
#   "lightgbm==4.6.0",
# ]
# ///
# MAGIC %md
# MAGIC # 06 · 個人模型訓練與 feature ablation
# MAGIC
# MAGIC 這份 notebook 重用 `features_v1.py` 與 `gate_eval.py` 的訓練／split／calibration 邏輯。
# MAGIC 預設只訓練 **v0、v1，seed 0**。不需要 GitHub Actions 或 `DATABRICKS_TOKEN`。
# MAGIC
# MAGIC **第一次操作：** 先執行套件與「設定」兩格，確認上方 widgets，再執行後續 cells。
# MAGIC 不修改團隊資料。`data_root`、`sample_root` 只讀；每次在個人 `output_root` 下建立新的 run 目錄。
# MAGIC 預設複製必要輸入到該 run，保存 SHA-256、實際 split、模型、feature 順序及 calibration cutoffs。
# MAGIC
# MAGIC 這是**離線答案品質實驗**。它不執行 raw-data fallback，也不量測端到端查詢延遲。

# COMMAND ----------

# MAGIC %pip install lightgbm==4.6.0

# COMMAND ----------

# MAGIC %md
# MAGIC ## 設定：先執行下一格，再填 widgets
# MAGIC
# MAGIC * `repo_root`：你目前這份個人 Git folder 的完整路徑。
# MAGIC * `data_root`：內含 `<dataset>/queries-Ans.txt`、`queries-meta.csv` 的資料根目錄。
# MAGIC * `sample_root`：內含 `<dataset>/rows.csv`、`arm_*.txt`、`src_*.txt` 的 logger 輸出根目錄。
# MAGIC * `output_root`：個人結果目錄，不能放在 repo、輸入目錄或 `/Workspace/Shared` 內。
# MAGIC * `experiment`：個人 MLflow 路徑，例如 `/Users/你的帳號/synopsislake-model-ablation`。
# MAGIC * 第一次維持 `variants=v0,v1`、`seeds=0`、`generalization=false`。
# MAGIC * `copy_inputs=true` 將輸入快照保存在個人 run；若沒有資料權限，請先取得授權，不要跳過檢查。
# MAGIC * `mlflow_enabled=false` 可只保存本次檔案，錯誤不會被悄悄當成 MLflow 成功。

# COMMAND ----------

import os
from pathlib import Path
import sys

VOL = "/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake"


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
        home = Path(*parts[:i + 1])
        output = str(home / "synopsislake-results")
        experiment = f"/Users/{parts[i]}/synopsislake-model-ablation"
    volume = os.environ.get("VOLUME", VOL)
    return dict(repo_root=repo, data_root=volume + "/data/ml",
                sample_root=volume + "/output/ml", output_root=output,
                dataset="authors_range", seeds="0", variants="v0,v1", generalization="false",
                experiment=experiment, mlflow_enabled="true", copy_inputs="true",
                epsilons="0.25", delta="0.05", max_trees="4000", num_threads="2")


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
# MAGIC ## 檢查計畫
# MAGIC
# MAGIC `all` 包含 10 個版本；3 個 seeds 會訓練 30 次，另開 generalization 再加 12 次。
# MAGIC 請先看下面的訓練次數，再決定是否執行。
# MAGIC 所有版本共用同一份 feature table、相同 query split、LightGBM 參數與 early stopping 規則。

# COMMAND ----------

import importlib

repo = Path(get("repo_root")).resolve()
if not (repo / "ml/model_ablation.py").is_file():
    raise FileNotFoundError("repo_root 不正確，或尚未切換到 experiments/model-ablation")
sys.path.insert(0, str(repo / "ml"))
# A session may have cached code before a branch switch or a local edit.
# Re-import this small module set from the explicitly selected Git folder.
for module_name in ("model_ablation", "features_v1", "features", "gate_eval"):
    sys.modules.pop(module_name, None)
importlib.invalidate_caches()
from model_ablation import run_experiment, variants  # noqa: E402

CONFIG = {key: get(key) for key in DEFAULTS}
requested = list(variants()) if CONFIG["variants"] == "all" else [v.strip() for v in CONFIG["variants"].split(",")]
if set(requested) - variants().keys():
    raise ValueError(f"不支援的 variants；可選 {list(variants())}")
seeds = [int(s.strip()) for s in CONFIG["seeds"].split(",")]
fits = len(requested) * len(seeds)
if CONFIG["generalization"].lower() == "true":
    fits += 4 * len(seeds)
print("repo:", repo)
print("variants:", requested, "seeds:", seeds, "model fits:", fits)
print("read-only data:", CONFIG["data_root"])
print("read-only logger exports:", CONFIG["sample_root"])
print("personal output root:", CONFIG["output_root"] or "尚未設定，請先填寫")
print("MLflow experiment:", CONFIG["experiment"])

# COMMAND ----------

# MAGIC %md
# MAGIC ## 執行訓練
# MAGIC
# MAGIC 若缺少 Volume／檔案／權限，這格會報出路徑，不會虛構資料或沿用歷史結果。
# MAGIC 預設排除真值 `<10` 的 query，與團隊現有 pipeline 相同；這輪不能推論零答案的品質。

# COMMAND ----------

result_dir = run_experiment(CONFIG)
print("本次結果：", result_dir)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 查看這次實際結果
# MAGIC
# MAGIC * `coverage_per_rate_mean`：每個 merge rate 各自校準後，接受比例的平均。
# MAGIC * `violation_per_rate_mean`：有接受答案的 rates，其違規率平均；**接受 0 筆時是空值，不是 0%**。
# MAGIC * `violation_pooled_per_rate_gates_mean`：把各 rate gate 的接受／違規案例加總後計算，與前者不同。
# MAGIC * 精確的分子、分母與 cutoff 在 `gate_details.csv`。`quality_target_met` 是 test 的觀察結果，沒有統計保證。
# MAGIC * 門檻只看 calibration，不可用 test 反覆調整。
# MAGIC * `oracle` 使用真實誤差，只是離線參考，不是可部署的模型。
# MAGIC
# MAGIC `models/<variant>/seed_<seed>/` 包含 `model.txt`、`schema.json`、`cutoffs.json`、
# MAGIC `cohorts.csv`、`test_predictions.csv`。cutoff 是模型的 **log-error score**，不是直接的相對誤差。
# MAGIC `run.json` 的 status 必須為 complete，才代表本輪全部完成。

# COMMAND ----------

import json
import pandas as pd

summary = pd.read_csv(result_dir / "v1_summary.csv")
details = pd.read_csv(result_dir / "gate_details.csv")
try:
    display(summary)  # noqa: F821
    display(details[(details.score == "learned") & (details.scope == "per_rate")])  # noqa: F821
except NameError:
    print(summary.to_string(index=False))
print("Gate details:", result_dir / "gate_details.csv")
print("Run metadata:", result_dir / "run.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 下一輪可做的消融
# MAGIC
# MAGIC **重現內建實驗：** `variants=v0,v1,v1_no_source,v1_no_shift`、`seeds=0,1,2`。
# MAGIC `v1_no_source` 仍有 merge-shift，因此仍依賴 source histogram 的估計。
# MAGIC
# MAGIC **少量來源資訊是否足夠：**
# MAGIC `variants=v0,v0_src_bound,v0_src_contrast,v0_src_bound_contrast,v1_no_shift`。
# MAGIC 這三個子集分別加入來源的 bounds、邊界 density contrast，以及兩者。
# MAGIC 與全 7 個來源特徵比較，找出哪些資訊最有用。
# MAGIC
# MAGIC 修改 widgets 後，從「檢查計畫」往下重跑。每次會建立新 run，不覆蓋上次的結果。
# MAGIC 改 variants 不會減少這份離線程式預先建立的來源特徵，也不代表已量測到 I/O 或查詢加速。
# MAGIC `generalization=true` 另外測 high-rate→low-rate 與 west→east；使用評估側的 calibration queries。
# MAGIC 訓練限制與測試限制是行過濾，另存每個模型實際用到的 cohorts，方便檢查資料切分。
