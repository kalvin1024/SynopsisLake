# Full-v1 leave-one-source-feature-out study

This study complements notebook 06's additive source-feature experiment. It asks whether a source
feature remains useful when the other full-v1 features are already available. It retrains reduced
models; it does not set a trained model's input column to zero.

Run the **English** Databricks source notebook `ml/notebooks/07_v1_feature_ablation.py` from your
personal Git folder on `experiments/model-ablation`. No PR, GitHub Actions, or Databricks token is needed.

## Default experiment

| Variant | Features | Removed from full v1 |
| --- | ---: | --- |
| `v1` | 28 | None |
| `v1_no_src_contrast` | 27 | `src_edge_contrast` |
| `v1_no_src_bound` | 27 | `src_bound_rel` |

All three retain `merge_shift` and `abs_merge_shift`. The earlier `v1_no_shift` is a different,
26-feature model and cannot serve as the full-v1 reference. Feature order otherwise stays identical.

The optional `v1_no_src_bound_contrast` removes both source features and retains both shift features.
To compare full v1, removal of contrast, and removal of both, set
`variants=v1,v1_no_src_contrast,v1_no_src_bound_contrast`.

Seeds `0,1,2`, the existing query-ID train/calibration/test split, training-only early stopping,
LightGBM parameters, epsilon `0.25`, and empirical calibration target `0.05` match notebook 06.
This experiment does not change calibration, split algorithms, models, or histogram code.

## Databricks

1. Pull `experiments/model-ablation` in your personal Users Git folder.
2. Open `ml/notebooks/07_v1_feature_ablation`, attach the same compute/runtime, and run the installation
   and settings cells. Keep the same dataset, input roots, model parameters, and package environment.
3. Check the personal output root, `seeds=0,1,2`, the default variant list, and `baseline_run=auto`.
4. Run **Review the training plan**. It prints the exact new fit count and full-v1 reuse decision per seed.
   There are at most 9 fits. A reused full-v1 seed saves one fit; both reduced models are retrained.
5. Run the remaining cells. The final output prints the new result directory.

Paths default to the same existing logger exports and personal `synopsislake-results` directory used
in notebook 06. The defaults for `max_trees=4000` and `num_threads=2` match that experiment.
Databricks widgets keep previous values, so review settings before every study.

Keep the original input roots unchanged for comparison with notebook 06. Each study records its
actual input fingerprints; compare `input_manifest.json` across studies. Any reused full-v1 model
must match the current input files exactly. If the shared files changed, use an identically restored
dataset layout rather than comparing results across different inputs.

`baseline_run=auto` discovers reusable full-v1 results by provenance, without selecting on test metrics.
`none` requests a fresh full-v1 fit for all seeds. An explicit prior run path can supply one or more seeds;
the remaining seeds are trained. An incompatible explicit result or ambiguous automatic match stops
before training. Never substitute `v1_no_shift`, and never choose a reused model by its test score.

Reuse requires compatible input fingerprints/snapshots, four original training-module hashes,
Python/package versions, model parameters, epsilon/delta, full-v1 feature schema, splits/cohorts,
and complete artifacts/results. Reused artifacts are copied into the new study and their source is
recorded. Missing volumes, inputs, or permissions fail clearly. Previous results and shared inputs are read-only.

## Saved results

Each study creates a unique personal `<dataset>-v1-feature-<run-id>/` directory.

| File | Contents |
| --- | --- |
| `by_seed.csv` | Existing quality metrics and counts for each variant, seed, and scorer |
| `summary.csv` | Learned-model means, sample SDs, and valid-seed counts |
| `gate_details.csv` | Per-rate and pooled cutoffs, calibration/test counts, accepted/violating counts, observed target status |
| `per_rate_summary.csv` | Per-rate seed summaries and observed target exceedance counts |
| `paired_deltas.csv` | Reduced minus full-v1 metrics at the same seed and epsilon |
| `models/<variant>/seed_<seed>/` | Model, schema, cutoffs, query cohorts, predictions, and feature importance |
| `run.json` | Complete/failed status, settings, code/package/input provenance, timings and trained/reused model origins |
| `splits.csv`, `input_manifest.json`, `inputs/` | Common query splits, input fingerprints, and default input snapshot |

To request analysis, upload `by_seed.csv`, `summary.csv`, `gate_details.csv`, `per_rate_summary.csv`,
`paired_deltas.csv`, and `run.json`. No raw data or tokens are required.

For paired deltas, negative coverage means fewer answers accepted after removing a feature; positive
risk means a higher violating fraction among accepted answers. AUC describes ranking. Evaluate these
together; models can accept different queries and have different risks.

The original four training/feature modules and notebook 06 are unchanged, preserving existing
code-hash compatibility checks. The new study module and optional notebook hash are recorded separately.

## Interpretation limits

- Exact counts below 10 remain excluded. This study does not evaluate zero-count queries.
- Zero accepted answers have undefined risk, not a successful 0% violation result.
- The 5% target is the existing empirical calibration rule, not a test or per-rate simultaneous guarantee.
- Seeds and merge rates reuse queries. Seed SDs are descriptive, not confidence intervals or significance tests.
- Previously examined test outcomes remain exploratory; validate final claims on a reserved workload.
- `oracle` uses true error and is an offline reference, not a deployable gate.
- Full source features are still computed. No source-I/O, memory, fallback, or query-latency savings are measured.
- Reused fit times describe past fits; reuse and current wall time must be reported separately.

## Local correctness checks

With LightGBM 4.6.0, numpy, pandas, scipy and scikit-learn installed:

```bash
python -m unittest discover -s ml/tests -v
```

The integration tests train actual small LightGBM models using synthetic logger exports; those metrics
are correctness fixtures and must not be reported as team research results. On a local fixture, the
same notebook accepts uppercase environment variables such as `REPO_ROOT`, `DATA_ROOT`, `SAMPLE_ROOT`,
`OUTPUT_ROOT`, `DATASET`, `MAX_TREES`, `MLFLOW_ENABLED=false`, and `BASELINE_RUN=none`.
