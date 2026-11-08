# 個人訓練與 ablation

若要從完整 v1 移除個別來源特徵，請改用新增的
[`07_v1_feature_ablation`](V1_FEATURE_ABLATION.md)。下方保留 notebook 06 的執行說明。

使用分支 `experiments/model-ablation` 的 `ml/notebooks/06_model_v1.py`。
這是手動 Databricks 訓練流程，不需要 GitHub Actions、PR 或 `DATABRICKS_TOKEN`。
不會呼叫部署、建立 cluster 或執行 raw-data query。

## 在 Databricks 執行

1. 在 **自己的 Users Git folder** 切換至 `experiments/model-ablation` 並 Pull。
2. 打開 `ml/notebooks/06_model_v1`，連接有權限使用的 Compute。
3. 先執行套件安裝與設定 cells，確認 widgets。若舊 notebook 已有 widgets，舊值可能保留，請重新核對。
4. 填好路徑，再從 **Review the training plan** 往下執行。先確認 model fits 顯示 10，並找到原來的 seed 0 run。
   Notebook 內的標題、說明與執行訊息已全部改為英文。
   Databricks 會保留舊 widget 值；把 `seeds` 從 `0` 改為 `1,2`。若 `variants` 仍是 `v0,v1`，請換成下表的五版本字串。

| Widget | 第一次設定 |
|---|---|
| `repo_root` | 你個人 Git folder 的完整路徑 |
| `data_root` | 團隊既有 query dataset 根目錄，預設 `<volume>/data/ml` |
| `sample_root` | 團隊既有 logger 輸出根目錄，預設 `<volume>/output/ml` |
| `dataset` | `authors_range` |
| `output_root` | 個人結果目錄，預設 `/Workspace/Users/<帳號>/synopsislake-results` |
| `experiment` | 個人 MLflow experiment，預設 `/Users/<帳號>/synopsislake-model-ablation` |
| `variants` | `v0,v0_src_bound,v0_src_contrast,v0_src_bound_contrast,v1_no_shift` |
| `seeds` | `1,2`，只補跑兩個 seeds；舊的 seed 0 不重跑 |
| `baseline_run` | `auto`，自動找到唯一相容的 seed 0；也可填該 run 的完整路徑 |
| `epsilons` / `delta` | `0.25` / `0.05` |
| `generalization` | `false` |
| `copy_inputs` | `true` |
| `mlflow_enabled` | `true`；本機無 MLflow 可明確選 false |
| `max_trees` / `num_threads` | `4000` / `2`；小資料正確性測試可降低樹數，不作正式比較 |

套件使用 LightGBM 4.6.0，並需要 Runtime 的 pandas、scipy、scikit-learn；開啟 MLflow 時也需要 mlflow。
LightGBM 在 driver 上訓練，不會自動分散到所有 Spark workers。
此 notebook 沒有 `volume` 控制路徑的 widget；請明確設定 `data_root`、`sample_root`。

需要的既有檔案：

```text
data_root/dataset/queries-Ans.txt
data_root/dataset/queries-meta.csv
sample_root/dataset/rows.csv
sample_root/dataset/arm_<resolution>.txt
sample_root/dataset/src_<source-id>.txt
```

Git clone 不會生成這些 logger 輸出。缺檔或權限錯誤時會停止，需先取得相同 run 的輸入。
原 query 欄位順序是 `xmin,xmax,ymin,ymax,truth`，不要另行重排匯出檔。
`copy_inputs=true` 自動建立唯讀來源的個人快照，不必再手動複製。
不修改 `sample_root/dataset/report`；CSV、模型全部寫到新的個人 run。

## 可選實驗

| 比較 | variants |
|---|---|
| 重現既有 v0/v1 基線 | `v0,v1` |
| 重現主要 ablation | `v0,v1,v1_no_source,v1_no_shift` |
| 目前預設：少量來源資訊是否足夠 | `v0,v0_src_bound,v0_src_contrast,v0_src_bound_contrast,v1_no_shift` |
| Skewness 作用 | `v0,v0_no_skew,v1,v1_no_skew` |
| 幾何資訊 baseline | `v0,v1,geometry_only` |

`v0_src_bound` 只加 `src_bound_rel`；`v0_src_contrast` 只加 `src_edge_contrast`；第三個加兩者。
目前的預設接續先前五版本的 seed 0，只補跑 seed 1 和 2（共 10 fits）。
舊結果與新結果保存在不同 run 目錄。每個 variant 使用相同 split、超參數與 early-stopping 規則。
`all` 包含 10 個版本；3 seeds 為 30 fits。開啟 generalization 另外增加 12 fits。
`generalization=true` 會跑 high merge rate→low 與 west→east，calibration 來自評估側；不能宣稱是不需目標資料的轉移。

v0 的 19 features、v1 的 28 features 沿用團隊程式。沒有更換模型。
`v1_no_source` 還有 merge-shift，因此仍依賴來源估計。
所有版本目前都預先建立完整來源特徵表；少用模型輸入不等於少讀來源資訊。

## 結果與保存

每次產生唯一的 `<output_root>/<dataset>-<run-id>/`，不覆蓋之前的 run。
`baseline_run=auto` 只依設定與資料身分尋找唯一相容的 seed 0，並非根據 test 成績選擇。
多個相符時請在 widget 填入要使用的 seed 0 run 完整路徑；找不到則不會自動重訓 seed 0。
訓練前會比對輸入與快照的 SHA-256、features/訓練程式 hash、特徵順序、模型參數、epsilon/delta 和環境版本。
訓練完成後再核對本次輸入，以偵測檢查後資料變更；不相容的 run 不會被合併。
若必須使用不同版本的資料或套件，應使用相同條件重新跑三個 seeds：設定 `baseline_run=none`、`seeds=0,1,2`。這會做 15 fits，舊檔案仍保留。

| 產物 | 內容 |
|---|---|
| `run.json` | status、設定、版本、程式 hash、git 狀態、輸入／特徵處理時間 |
| `input_manifest.json`、`inputs/` | 輸入 SHA-256／來源位置及預設保存的快照 |
| `splits.csv` | 既有 50/20/30 query-ID 分配與 training early-stopping 標記 |
| `v1_ablations.csv`、`v1_summary.csv` | 各版本、seed、epsilon 的指標與 seed 間平均／標準差 |
| `gate_details.csv` | 每 rate／pooled cutoff、接受數、違規數、樣本數及 observed target 是否達標 |
| `models/<variant>/seed_<seed>/model.txt` | 實際訓練的 LightGBM Booster |
| `schema.json`、`cutoffs.json` | feature 順序、參數、label 定義、模型 score 空間中的門檻 |
| `cohorts.csv`、`test_predictions.csv` | 該模型實際使用的 row cohorts 與離線 test 預測 |
| `comparison/summary_3_seeds.csv` | 舊 seed 0 加新 seed 1、2 的平均與標準差 |
| `comparison/combined_by_seed.csv` | 分開列出每個 seed 的結果與來源 run |
| `comparison/per_rate_3_seeds.csv` | 每個 merge rate 的三 seed 統計、有效風險 seed 數與超標次數 |
| `comparison/combined_gate_details.csv`、`provenance.json` | 原始 gate 細節與相容性檢查資訊 |

MLflow 開啟時會記錄各模型的 params、metrics 與模型／評估 artifacts。
MLflow 設定或寫入失敗會明確失敗，不會悄悄宣稱有記錄。
Databricks 未提供 notebook 的普通 `.py` 檔案時，仍保存 module hashes，並在 run.json 註明 notebook hash 無法取得。

模型之後可透過 `lightgbm.Booster(model_file=...)` 載入；必須保持 schema 的 feature 順序和一致的特徵計算語義。
`cutoffs.json` 的數值是 predicted log-error score；`null` 配合 `accept_none=true` 表示拒絕全部。
`test_predictions.csv` 僅供離線評估，不得用來替代未來線上模型推論。

## 如何解讀品質

使用團隊既有 split_ids，先排除真值 `<10`，所以不評估零／小答案。
模型 early stopping 只使用 training 的 validation slice；接受門檻只使用 calibration，不使用 test。
每個 merge rate 的接受／違規數是 query–histogram cases；跨 rate 的同一 query 並非獨立新樣本。
三個 seed 的 split 也可能包含重疊 query；seed 間標準差只是描述穩定性，不是信賴區間或顯著性檢定。

`coverage_per_rate` 是各 rate 接受比例的平均。
`violation_per_rate` 只平均有接受答案的 rates；`rates_with_acceptances` 說明分母。
`violation_pooled_per_rate_gates` 是各 rate gate 的 accepted/violation cases 加總後計算，與 pooled 單一 cutoff 不同。
零接受時 violation 留空，不宣稱通過。單一標籤類別的 AUC 也留空。
5% 是 empirical calibration 目標，不保證 test 達標；要一起看接受數、違規數與 observed risk。
`oracle` 使用真實誤差，只是離線參考，不可部署，也不代表一般意義的保證上界。

這份 notebook 不執行 exact fallback，不提供查詢速度／系統加速結論。
時間分成 input preparation、feature build、fit、prediction/evaluation，以及整輪 wall time；不是逐查詢 E2E latency。

## 本機驗證

在裝有上述套件的 Python 環境，從 repo 根目錄執行：

```bash
python -m unittest discover -s ml/tests -v
```

也可用本機路徑與個人輸出目錄執行同一份 notebook：

```bash
REPO_ROOT="$PWD" DATA_ROOT=/path/to/data/ml SAMPLE_ROOT=/path/to/output/ml \
OUTPUT_ROOT=/tmp/my-ablation-results DATASET=authors_range \
VARIANTS=v0,v1 SEEDS=0 MLFLOW_ENABLED=false \
python ml/notebooks/06_model_v1.py
```

本次以 160 個合成 query、兩個 sources、兩個 merged resolutions 做正確性驗證；不當成研究結果。
測試實際訓練／重載模型、門檻隔離、ties、零接受風險、單一類別 AUC、split 與寫入保護。
完整 notebook 也以小資料跑過全部 10 variants 加 4 個 generalization fits。
尚未宣稱已在你的 Databricks、團隊正式資料或 MLflow 執行成功。

## 上週 main 的紅叉

`a42efb6` 的 merge 已成功，並且個人 fork 與團隊 main 一致，不需要重做 merge。
[Actions run](https://github.com/kywang0906/SynopsisLake/actions/runs/42024731947) 中 Maven package 成功；
`Verify Databricks auth` 回傳 HTTP 401，後續上傳、deploy、job run 都 skipped。
這是 GitHub→Databricks 認證問題，與刪除個人 Git folder 無關。

手動訓練不需要修該 workflow。要啟用自動部署才需要另行處理核准的 CI 憑證／權限；本分支沒有重跑、部署或修改 secrets。
`experiments/model-ablation` 不在現有自動 push workflow 的分支名單中。
