# Unseen-region comparison with a common total storage budget

Entry points:

- Databricks: `ml/notebooks/23_region_budget_comparison.py`.
- CLI: `python -m ml.region_budget_comparison --config CONFIG.json`.

This adds one focused experiment using notebook 22's source-shape representation, the team's full-v1 learner,
and 09b's Java execution/fallback components. Notebooks 22 and 09b retain their existing behavior. The new
entry point is necessary because the earlier region test calibrates on the tested side and the earlier 22
reference comparison does not match trained-model total storage.

## What is compared

| Output method | Accepted answer | Quality score | Required representation |
|---|---|---|---|
| `source_shape` | Original merged H | `abs(H-R)/max(R,1)` | H, native source grids/moments, selected child packet, gate data |
| `v1` | Original merged H | Full-v1 predicted error | H, original source histogram exports, model/properties, gate data |
| `fine_direct` | Its own finer count answer F | Worst relative-error bound from exact fine-bin count intervals | Fine count grid and gate data; no H or source-shape files |
| `fine_reference` | Original merged H | `abs(H-F)/max(F,1)` | H, fine count grid and gate data |
| `exact` | Exact raw-point count | No gate | Raw points; common raw data are accounted separately from synopsis storage |

The fine-reference row is the role already evaluated in notebook 22. It is not the direct-answer alternative.
Fine direct can improve the candidate answer itself, while SS and v1 decide whether to retain H. Its own
calibration errors therefore use F versus truth, not H versus truth. A larger direct count grid must not be
described as merely another error predictor for H.

Fine direct uses a prespecified count-only score. Fully contained bin counts give a lower count L; counts
from all possibly intersected bins give an upper count U. Given F, the worst relative error is evaluated at
the interval endpoints, with explicit zero-count handling. Inclusive query boundaries are covered
conservatively. The same empirical western-calibration rule selects its operational cutoff; relaxing a
bound-score cutoff empirically is not a distribution-free risk guarantee.

## Strict region and label protocol

1. Read query geometry and fix the cut without using eastern labels. The default cut is the midpoint of the
   smallest/largest query-center longitude, preserving the older region test's cut-selection rule.
2. Assign a rectangle west only when its entire longitude extent lies west of the cut (`maxLon < cut`).
   Assign east when `minLon >= cut`. Exclude crossing rectangles and preserve their IDs in the protocol.
   This is stricter than assigning locations by query center.
3. Select common eastern evaluation IDs by geometry/IDs before fitting or choosing a budget. The same
   eastern geometries are used for every method and seed. Using multiple seeds does not create independent
   geographic test datasets.
4. Divide western IDs into TRAIN and calibration. Fit the full28-feature v1 model on western TRAIN only,
   with the team's log-error label,63 leaves,4000-tree cap and TRAIN-only early stopping. No unverified
   pre-existing model trained over both sides is accepted as a substitute. The training eligibility rule
   retains exact counts of at least10, matching the team's learner.
   This focused implementation refits on the single tested H, like notebook22's matched model-family
   control. It is not the old saved joint-18-merge-rate model. The training scope is explicit so the result
   is not described as reproducing the published multi-rate model.
   TRAIN features are exported by the same Java FeatureRegistry used for timed inference, avoiding
   notebook22's Python/Java feature-rounding contract difference.
5. Construct SS/fine representations from raw point data offline. This is allowed data summarization, not
   use of eastern query labels. Source-only SS cell selection does not consume test query errors.
6. Determine the common budget from declared settings and western-trained model sizes. Freeze the
   candidate representations, model weights and budget before eastern labels are evaluated.
7. Compute all scores on the same western calibration queries. Apply the same complete-tie empirical
   cutoff rule at epsilon25%, delta5%. Each method's own candidate answer determines its bad labels.
8. Freeze the gates, then compute eastern truth with exact execution and evaluate without retuning.

The primary test table includes all selected eastern queries. A truth-at-least10 slice is an additional
evaluation table, not a test-label filter used to choose the test geometry, budget or threshold. Stored
eastern labels are not used to train or choose operating points. Exact fallback and zero-truth handling are
checked explicitly. Any observed violation above5% on east remains a result to report.

## Total storage accounting

`total_budget_bytes` is a common upper ceiling. With `0`, the suite derives a shared ceiling large enough
for the west-trained v1 payloads and SS's required representation, before evaluating east. An explicit cap
that cannot fit a required method fails rather than silently shrinking that model or hiding its weights.

Charge every file needed for query-time execution:

- Original H where the method returns or assesses H.
- Native source counts, boundaries and moments; SS child packet.
- Original source histogram exports used by v1 features.
- Model weights and label-transform properties.
- Persisted gate parameters.
- The direct/reference count grid, including its header and count storage.

Report component bytes, total actual bytes and unused budget. A representation can saturate before filling
the ceiling; no padding is counted as useful information. The packet cap is not total SS storage. Each
seed's deployed model is charged separately, rather than adding three alternative models as if one query
loads all of them. All methods use the same suite ceiling; exact execution is a separately identified raw
baseline. Raw files, query files, evaluation outputs, source code and JARs are identified separately and are
not hidden inside a synopsis-only byte comparison. Memory consumption is not interchangeable with these
serialized bytes.

`shape_packet_budget_bytes=0` allocates the bytes remaining after H/native/gate charges to the existing
one-level child representation. It may select every positive-priority source cell and still leave unused
space. Setting49152 explicitly reproduces notebook22's incremental packet ceiling; that ceiling is not
the full SS cost. Neither choice is selected from eastern errors.

## Databricks execution

Use a classic runtime with Java and access to the actual Volume. Import/open notebook23 in the updated
Git checkout. Run the dependency cell, review widgets, then run the remaining cells. Python modules from
another checkout must not remain cached; restart Python after pulling changed implementation files.

The notebook's input-access check runs before model fitting. It confirms the raw sources, matching query
file and selected H are readable without parsing query labels. A personal Git folder does not need its
own copy of Europe: in the same Databricks workspace, it can use the existing shared Volume under the
current user's permissions. Prefer the same compute used for the successful Europe09b run. If access is
denied, the Volume owner must check USE CATALOG, USE SCHEMA and READ VOLUME; changing Git branches
does not grant data permissions. The check reports missing/unreadable paths before expensive training.

The standard Volume root is:

```text
/Volumes/cs598_ccc_research_7474657373242960/synopsis_lake
```

Default inputs are explicit and differ by dataset:

| Setting | Authors' Range | Europe |
|---|---|---|
| `run_dir` | `<volume>/output/ml/authors_range` | `<volume>/output/ml/europe_range` |
| `dataset_dir` | `<volume>/data/ml/authors_range` | `<volume>/data/ml/europe_range` |
| `data_dir` | `<volume>/data/range_query` | `<volume>/data/europe_range` |
| Query file | `<dataset_dir>/queries-Ans.txt` | `<dataset_dir>/queries-Ans.txt` |

Override `authors_queries_file` when the actual AUTHORS file remains at its original prepared location,
for example `data/range_query/dataDrivenQuery/srcQuery-40-56-32-48-Ans.csv`. Do not use AUTHORS raw data for
Europe or point `run_dir` at the Git folder. Histogram exports and raw numbered files must correspond.

The runner compiles `sigspatial25.RegionBudgetQueryExp` and its reachable dependencies directly from the
current checkout using `javac` and commons-math3, placing classes in a hash-keyed local temporary cache.
The normal path requires neither Maven nor replacing the existing Volume JAR. If `javac` is unavailable,
build the updated project elsewhere and provide an explicit matching JAR/classpath:

```bash
cd SynopsisLake
mvn -DskipTests package
```

`math3_path` defaults to `<volume>/jars/commons-math3-3.6.1.jar`. An explicit `classpath` must contain the
updated classes/JAR plus commons-math3. No hard-coded personal result directory or private model export is
required.

Use `dataset=both` in one invocation for a single budget ceiling shared across both datasets. Running
each dataset separately with automatic budgets may select different ceilings; do not describe those runs
as a shared cross-dataset budget. An explicit `total_budget_bytes` can instead be reused after verifying
that it fits both source-trained model/representation minima.

### Execution check

For a small explicitly labelled input/execution check, use:

```ini
dataset = authors_range
seeds = 0
n_test_queries = 200
repeats = 1
eps = 0.25
delta = 0.05
merge_rate = 0.2
total_budget_bytes = 0
```

Keep any training/calibration subsampling explicit in the run configuration. This check is not the formal
three-seed/two-dataset result, and its timing must not replace the formal table.

### Formal comparison

```ini
dataset = both
seeds = 0,1,2
n_test_queries = 2000
n_calibration_queries = 0
max_train_queries = 0
repeats = 3
eps = 0.25
delta = 0.05
merge_rate = 0.2
total_budget_bytes = 0
shape_packet_budget_bytes = 0
include_fine_reference = true
```

Here, zero query limits mean use all available western training/calibration IDs where applicable, and a
zero storage setting means the declared automatic budget policy. Test query limits and eligible counts are
recorded; do not label a smaller available set as2000. The nearest available H merge rate is recorded.

## Command-line execution

An example configuration is provided in `ml/region_budget_config.example.json`:

```bash
python -m ml.region_budget_comparison --config ml/region_budget_config.example.json
```

Edit its input/output paths for the actual machine. This command reads real existing datasets; it does not
generate replacement synthetic research data. The local test suite uses small fixtures only for software
validation. Access to the full Europe files and a valid Java runtime is required for Europe's measured
result. An example command/configuration is not evidence that the experiment has already run.

## Outputs and interpretation

The notebook prints the exact unique run directory under:

```text
<volume>/output/ml/region_budget_comparison/<unique-run-directory>/
```

Completed runs are preserved. The main files are:

| File | Purpose |
|---|---|
| `summary.csv` | Per dataset/seed/method: query count, coverage, accepted count, accepted bad count/rate, fallback count, batch time and seconds saved versus exact |
| `dataset_contrast.csv` | Each dataset's own exact time, raw rows/load/scan information, fallback counts and measured time savings |
| `payload_costs.csv`, `payload_files.csv` | Component totals, actual files/hashes, common ceiling and unused bytes |
| `timing.csv`, `phase_timing.csv` | Per-repeat whole-batch and stage records; source/model reads and exact fallback remain charged |
| `region_splits.csv` | Geometry-only cut and retained/excluded region memberships |
| `calibration.csv`, `frozen_gates.csv` | Western-only calibration scores/labels and fixed operational cutoffs |
| `protocol.json`, `manifest.json` | Settings, input identities, ordering of preparation/freeze/evaluation, versions and completion status |
| `analysis_en.md` | Concise interpretation of the measured run, including risk failures |
| `region_budget_results.zip` | Review evidence without raw data and large model weights; weights remain at paths recorded in the manifest |

Coverage is the share using the method's candidate answer. Accepted bad count is how many accepted
answers exceed25% true error; accepted violation is that count divided by accepted count. It is undefined
when nothing is accepted. Fallback count is total queries minus accepted count. Exact execution uses raw
data for every query, but is labelled separately from a rejected-query fallback.

Batch time is the median over actual repeats. Time saved is **the same dataset's exact median minus the
method median**; negative values mean slower. Every dataset receives its own exact baseline. Data loading,
source summaries, model initialization and real fallback scans are included; preparation/training and
calibration remain separately recorded offline costs. Component medians need not add to the whole-batch
median, so use per-repeat phase records for decomposition.

Differences in exact point counts, raw loading, number of fallbacks and scan work can help explain why
AUTHORS and Europe save different amounts of time. Merely observing that Europe has more points does not
establish causality. OS caches are not flushed. These are local JVM/in-memory-scan batch costs, not Spark
distributed execution or per-query cold cloud I/O.

The experiment answers a stricter question than milestone4's earlier region study: its operational gates
never use target-region calibration labels. SS has no trained error predictor but still requires a cutoff;
it can therefore suffer calibration transfer failure. Better ranking, higher coverage and faster execution
must be read with accepted violations and storage. Neither a5% calibration target nor one completed test
establishes a future risk guarantee or research novelty by itself.
