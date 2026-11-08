"""Leave-one-feature-out experiments using the frozen model-ablation helpers.

The original runner is intentionally unchanged so earlier notebook 06 runs keep
their provenance identity. This study removes only named source inputs from full
v1, retains both merge-shift features, and does not change training or calibration.
"""
import math
from pathlib import Path
import platform
import shutil
import subprocess
import time
from uuid import uuid4

import lightgbm as lgb
import numpy as np
import pandas as pd

from compare_ablation_runs import (
    _code_hashes, _expected_splits, _identity, _live_inputs, _manifest_entries,
    _read_frame, _read_json, _validate_frames, _validate_snapshot, _versions,
    _DETAIL_COLS, _RESULT_COLS,
)
from features import check_estimates
from features_v1 import FEATURES_V1, build_table_v1, check_source_estimates
from gate_eval import PARAMS
from model_ablation import (
    _bool, _hash, _inputs, _log_mlflow, _record_model, _snapshot, _write_json,
    make_split_manifest, validate_output_root,
)


DEFAULT_VARIANTS = "v1,v1_no_src_contrast,v1_no_src_bound"
_METRICS = ["auc_per_rate", "coverage_per_rate", "violation_per_rate",
            "violation_pooled_per_rate_gates", "fit_seconds"]


def variants():
    """Keep the order of full v1 and remove only the named source input(s)."""
    return {
        "v1": list(FEATURES_V1),
        "v1_no_src_contrast": [f for f in FEATURES_V1 if f != "src_edge_contrast"],
        "v1_no_src_bound": [f for f in FEATURES_V1 if f != "src_bound_rel"],
        "v1_no_src_bound_contrast": [f for f in FEATURES_V1
                                     if f not in ("src_bound_rel", "src_edge_contrast")],
    }


def _normalize(config):
    cfg = dict(config)
    for name, value in dict(variants=DEFAULT_VARIANTS, seeds="0,1,2", epsilons="0.25",
                            delta="0.05", baseline_run="auto", generalization="false",
                            max_trees="4000", num_threads="2", copy_inputs="true",
                            mlflow_enabled="true").items():
        cfg.setdefault(name, value)
    dataset = str(cfg["dataset"]).strip()
    if not dataset or dataset in (".", "..") or "/" in dataset or "\\" in dataset:
        raise ValueError("dataset must be a single directory name")
    cfg["dataset"] = dataset
    validate_output_root(cfg["output_root"], cfg["repo_root"], cfg["data_root"], cfg["sample_root"])
    names = [v.strip() for v in cfg["variants"].split(",")]
    if (not names or len(names) != len(set(names)) or "v1" not in names
            or set(names) - variants().keys()):
        raise ValueError(f"Choose distinct variants including v1 from {list(variants())}")
    seeds = [int(s.strip()) for s in str(cfg["seeds"]).split(",")]
    eps = [float(s.strip()) for s in str(cfg["epsilons"]).split(",")]
    delta = float(cfg["delta"])
    if not seeds or len(set(seeds)) != len(seeds) or min(seeds) < 0:
        raise ValueError("seeds must be distinct nonnegative integers")
    if (not eps or len(set(eps)) != len(eps) or any(not math.isfinite(e) or e <= 0 for e in eps)
            or not 0 <= delta < 1):
        raise ValueError("Invalid epsilon or delta")
    if _bool(cfg["generalization"]):
        raise ValueError("This leave-out study requires generalization=false")
    _bool(cfg["copy_inputs"])
    _bool(cfg["mlflow_enabled"])
    params = dict(PARAMS, n_estimators=int(cfg["max_trees"]), n_jobs=int(cfg["num_threads"]))
    if params["n_estimators"] < 1 or params["n_jobs"] < 1:
        raise ValueError("max_trees and num_threads must be positive")
    cfg["model_params"] = params
    return cfg, names, seeds, eps, delta, params


def _baseline(folder, seed, cfg, eps, delta, params, identity):
    """Validate one full-v1 model, including its saved calibration decisions."""
    state = _read_json(folder / "run.json")
    saved = state.get("config", {})
    if state.get("status") != "complete" or saved.get("dataset") != cfg["dataset"]:
        raise ValueError("Run is incomplete or uses another dataset")
    if (saved.get("model_params") != params
            or [float(e) for e in str(saved.get("epsilons", "")).split(",")] != eps
            or float(saved.get("delta", -1)) != delta
            or _bool(saved.get("generalization", False))):
        raise ValueError("Model parameters, epsilon/delta, or generalization differ")
    if (state.get("package_versions") != identity["package_versions"]
            or state.get("python") != identity["python"]
            or any(state.get("code_sha256", {}).get(k) != v
                   for k, v in identity["code_sha256"].items())):
        raise ValueError("Training code or Python/package versions differ")
    records = _manifest_entries(folder / "input_manifest.json")
    if {f"{key[0]}/{key[1]}": row["sha256"] for key, row in records.items()} != identity["input_sha256"]:
        raise ValueError("Input fingerprints differ")
    _validate_snapshot(folder, records, state)
    model = folder / "models" / "v1" / f"seed_{seed}"
    schema = _read_json(model / "schema.json")
    if (schema.get("features") != FEATURES_V1 or schema.get("variant") != "v1"
            or schema.get("seed") != seed or schema.get("params") != params
            or schema.get("min_exact") != 10):
        raise ValueError("Full-v1 model feature schema differs")
    booster = lgb.Booster(model_file=str(model / "model.txt"))
    if booster.feature_name() != FEATURES_V1:
        raise ValueError("Saved model feature order differs")
    summary = _read_frame(model / "results.csv", _RESULT_COLS)
    details = _read_frame(folder / "gate_details.csv", _DETAIL_COLS)
    details = details[(details.variant == "v1") & (details.seed == seed)].copy()
    predictions = _read_frame(model / "test_predictions.csv", ["qId", "mergeRate", "relErr", "score"])
    if (predictions.empty or predictions.duplicated(["qId", "mergeRate"]).any()
            or not np.isfinite(predictions[["mergeRate", "relErr", "score"]].to_numpy(float)).all()):
        raise ValueError("Missing, duplicated, or nonfinite saved test predictions")
    rates = set(predictions.mergeRate)
    _validate_frames(summary, details, ["v1"], [seed], eps, {("v1", seed): rates})
    splits = _read_frame(folder / "splits.csv", ["seed", "qId", "split", "early_stopping"])
    splits = splits[splits.seed == seed]
    _expected_splits(splits, [seed])
    cohorts = _read_frame(model / "cohorts.csv", ["qId", "mergeRate", "split", "early_stopping"])
    expected = splits.drop(columns="seed").merge(pd.DataFrame({"mergeRate": sorted(rates)}), how="cross")
    order = ["qId", "mergeRate"]
    try:
        pd.testing.assert_frame_equal(
            cohorts[expected.columns].sort_values(order).reset_index(drop=True),
            expected.sort_values(order).reset_index(drop=True), check_dtype=False,
        )
    except AssertionError as error:
        raise ValueError("Model cohorts differ from saved query split") from error
    expected_test = expected[expected.split == "test"]
    if set(map(tuple, predictions[order].to_numpy())) != set(map(tuple, expected_test[order].to_numpy())):
        raise ValueError("Saved test predictions do not match test cohorts")
    cutoffs = pd.DataFrame(_read_json(model / "cutoffs.json"))
    learned = details[details.score == "learned"]
    columns = ["eps", "scope", "mergeRate", "cutoff", "accept_none", "calibration_count",
               "calibration_accepted_count", "calibration_violation_count", "test_count",
               "accepted_count", "violation_count", "coverage", "violation"]
    try:
        pd.testing.assert_frame_equal(
            cutoffs[columns].sort_values(["eps", "scope", "mergeRate"]).reset_index(drop=True),
            learned[columns].sort_values(["eps", "scope", "mergeRate"]).reset_index(drop=True),
            check_dtype=False,
        )
    except (AssertionError, KeyError) as error:
        raise ValueError("Saved cutoffs differ from gate decisions") from error
    for row in learned.itertuples():
        test = predictions if row.scope == "pooled" else predictions[predictions.mergeRate == row.mergeRate]
        accepted = test.score <= row.cutoff if not row.accept_none else pd.Series(False, index=test.index)
        if (int(accepted.sum()) != row.accepted_count
                or int(((test.relErr > row.eps) & accepted).sum()) != row.violation_count
                or len(test) != row.test_count):
            raise ValueError("Saved gate counts disagree with predictions/cutoffs")
    importance = _read_frame(model / "importance.csv", ["feature", "gain", "variant", "seed"])
    if (list(importance.feature) != FEATURES_V1 or set(importance.variant) != {"v1"}
            or set(importance.seed) != {seed}):
        raise ValueError("Saved feature importance schema differs")
    return summary, details, importance


def plan_experiment(config):
    """Select compatible full-v1 baselines by provenance, never test performance."""
    started = time.perf_counter()
    cfg, names, seeds, eps, delta, params = _normalize(config)
    identity = dict(input_sha256={f"{k[0]}/{k[1]}": v for k, v in _live_inputs(cfg).items()},
                    code_sha256=_code_hashes(cfg["repo_root"]),
                    package_versions=_versions(), python=platform.python_version())
    selection = str(cfg["baseline_run"]).strip()
    output = Path(cfg["output_root"]).resolve()
    explicit = selection.lower() not in ("auto", "none")
    if explicit:
        folder = Path(selection).expanduser().resolve()
        if not folder.is_dir() or not folder.is_relative_to(output):
            raise ValueError("baseline_run must be an existing run under output_root")
        candidates = [folder]
    else:
        candidates = [] if selection.lower() == "none" else sorted(output.glob(cfg["dataset"] + "-*"))
    baselines = {str(seed): None for seed in seeds}
    diagnostics = []
    for folder in candidates:
        if not folder.is_dir():
            continue
        available = [seed for seed in seeds if (folder / "models" / "v1" / f"seed_{seed}").is_dir()]
        if explicit and not available:
            raise ValueError("Chosen baseline contains no requested full-v1 seed; v1_no_shift is not full v1")
        for seed in available:
            try:
                _baseline(folder, seed, cfg, eps, delta, params, identity)
            except (OSError, ValueError, TypeError, KeyError, lgb.basic.LightGBMError) as error:
                if explicit:
                    raise ValueError(f"Chosen full-v1 baseline is incompatible for seed {seed}: {error}") from error
                diagnostics.append(f"{folder.name}/seed_{seed}: {error}")
                continue
            if baselines[str(seed)] is not None:
                raise ValueError(f"Multiple compatible full-v1 baselines for seed {seed}; set baseline_run explicitly")
            baselines[str(seed)] = str(folder.resolve())
    reuse = sum(path is not None for path in baselines.values())
    return dict(config=cfg, variants=names, seeds=seeds, baselines=baselines,
                fit_count=len(names) * len(seeds) - reuse, reuse_count=reuse,
                baseline_diagnostics=diagnostics, preflight_seconds=time.perf_counter() - started, **identity)


def _tables(results, details, destination):
    learned = results[results.score == "learned"]
    grouped = learned.groupby(["eps", "variant"])
    summary = grouped[_METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().merge(grouped.seed.nunique().rename("n_seed").reset_index())
    summary["n_seeds"] = summary.n_seed
    for metric in _METRICS:
        summary = summary.merge(grouped[metric].count().rename(metric + "_n").reset_index())
    counts = grouped[["accepted_cases_per_rate", "violating_cases_per_rate"]].sum().reset_index()
    counts["case_weighted_violation"] = (
        counts.violating_cases_per_rate / counts.accepted_cases_per_rate.where(counts.accepted_cases_per_rate > 0)
    )
    summary = summary.merge(counts, on=["eps", "variant"], validate="one_to_one")
    summary.to_csv(destination / "summary.csv", index=False)
    rate = details[(details.score == "learned") & (details.scope == "per_rate")].copy()
    rate["above_target"] = (rate.accepted_count > 0) & ~rate.quality_target_met.fillna(False).astype(bool)
    rate.groupby(["eps", "variant", "mergeRate"]).agg(
        n_seed=("seed", "nunique"), coverage_mean=("coverage", "mean"), coverage_std=("coverage", "std"),
        violation_mean=("violation", "mean"), violation_std=("violation", "std"),
        defined_risk_seeds=("violation", "count"), above_target_seeds=("above_target", "sum"),
        accepted_cases=("accepted_count", "sum"), violating_cases=("violation_count", "sum"),
    ).reset_index().to_csv(destination / "per_rate_summary.csv", index=False)
    metrics = ["coverage_per_rate", "violation_per_rate", "auc_per_rate", "violation_pooled_per_rate_gates",
               "accepted_cases_per_rate", "violating_cases_per_rate", "unique_test_queries", "test_cases"]
    reference = learned[learned.variant == "v1"][["seed", "eps", *metrics]].rename(
        columns={metric: "baseline_" + metric for metric in metrics})
    paired = learned[learned.variant != "v1"].merge(reference, on=["seed", "eps"], validate="many_to_one")
    for metric in metrics:
        paired["delta_" + metric] = paired[metric] - paired["baseline_" + metric]
    paired.to_csv(destination / "paired_deltas.csv", index=False)


def _artifact_hashes(folder):
    return {str(path.relative_to(folder)): _hash(path)
            for path in sorted(folder.rglob("*")) if path.is_file()}


def run_experiment(config, plan=None):
    """Train missing models and write a new complete, independently saved study."""
    started = time.perf_counter()
    current = plan_experiment(config)
    if plan is not None:
        for key in ("config", "baselines", "input_sha256", "code_sha256", "package_versions", "python"):
            if plan.get(key) != current[key]:
                raise ValueError(f"Plan changed ({key}); rerun the training-plan cell")
    cfg, names, seeds, eps, delta, params = _normalize(config)
    mlflow = None
    if _bool(cfg["mlflow_enabled"]):
        if not str(cfg.get("experiment", "")).startswith("/Users/"):
            raise ValueError("Set a personal /Users/<account>/... MLflow experiment")
        import mlflow
        mlflow.set_experiment(cfg["experiment"])
    output = Path(cfg["output_root"]).resolve()
    destination = output / f"{cfg['dataset']}-v1-feature-{uuid4().hex[:12]}"
    destination.mkdir(parents=True, exist_ok=False)
    state = dict(status="running", study="full_v1_source_leaveout", config=cfg,
                 feature_definitions={name: variants()[name] for name in names}, model_sources=[],
                 timings={"preflight_seconds": current["preflight_seconds"]}, notes=[
                     "Offline answer-quality study; not query latency or an I/O saving.",
                     "All variants retain both merge-shift features and build the complete source table.",
                     "Existing empirical calibration is unchanged and does not guarantee 5% test risk.",
                     "Exact answers below 10 are excluded; zero-count quality is not evaluated.",
                     "Seed standard deviations are descriptive; queries overlap across seeds/rates.",
                     "Inherited fit_seconds describe the source model's original fit; current_fit_seconds excludes reused fits.",
                 ], package_versions=current["package_versions"], python=current["python"],
                 code_sha256={**current["code_sha256"], "v1_feature_ablation.py": _hash(Path(__file__))})
    try:
        try:
            state["code_sha256"]["notebooks/07_v1_feature_ablation.py"] = _hash(
                Path(__file__).parent / "notebooks/07_v1_feature_ablation.py")
        except OSError as error:
            state["notebook_hash_unavailable"] = type(error).__name__
        try:
            state["git_commit"] = subprocess.check_output(
                ["git", "-C", cfg["repo_root"], "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
            state["git_dirty"] = bool(subprocess.check_output(
                ["git", "-C", cfg["repo_root"], "status", "--porcelain"],
                text=True, stderr=subprocess.DEVNULL))
        except (OSError, subprocess.CalledProcessError):
            state["git_commit"] = None
        data = Path(cfg["data_root"]).resolve() / cfg["dataset"]
        samples = Path(cfg["sample_root"]).resolve() / cfg["dataset"]
        tick = time.perf_counter()
        manifest = _snapshot(_inputs(data, samples), data, samples, destination, _bool(cfg["copy_inputs"]))
        _write_json(destination / "input_manifest.json", manifest)
        captured = {f"{_identity(row['source'])[0]}/{Path(row['source']).name}": row["sha256"] for row in manifest}
        if captured != current["input_sha256"]:
            raise ValueError("Inputs changed after preflight; no models were trained")
        state["timings"]["input_preparation_seconds"] = time.perf_counter() - tick
        if _bool(cfg["copy_inputs"]):
            data, samples = destination / "inputs/data", destination / "inputs/run"
        tick = time.perf_counter()
        frame = build_table_v1(str(samples), str(data))
        if not len(frame) or frame.duplicated(["qId", "mergeRate"]).any():
            raise ValueError("Missing eligible rows or duplicate query/histogram cases")
        if not np.isfinite(frame[FEATURES_V1 + ["label", "relErr", "mergeRate"]].to_numpy(float)).all():
            raise ValueError("Nonfinite features/labels; check inputs")
        state.update(rows=len(frame), unique_queries=int(frame.qId.nunique()),
                     dropped_small_exact=frame.attrs.get("dropped_small_exact", 0),
                     estimate_gaps=dict(merged=check_estimates(frame), sources=check_source_estimates(frame)))
        state["timings"]["feature_build_seconds"] = time.perf_counter() - tick
        splits = pd.concat([make_split_manifest(frame, seed) for seed in seeds], ignore_index=True)
        splits.to_csv(destination / "splits.csv", index=False)
        _write_json(destination / "run.json", state)
        results, detailed, importances = [], [], []
        for name in names:
            for seed in seeds:
                folder = destination / "models" / name / f"seed_{seed}"
                source = current["baselines"][str(seed)] if name == "v1" else None
                tick = time.perf_counter()
                source_digests = None
                if source:
                    prior = Path(source)
                    prior_model = prior / "models" / "v1" / f"seed_{seed}"
                    source_digests = _artifact_hashes(prior_model)
                    res, det, imp = _baseline(prior, seed, cfg, eps, delta, params, current)
                    prior_splits = pd.read_csv(prior / "splits.csv")
                    try:
                        pd.testing.assert_frame_equal(
                            prior_splits[prior_splits.seed == seed].sort_values("qId").reset_index(drop=True),
                            splits[splits.seed == seed].sort_values("qId").reset_index(drop=True), check_dtype=False)
                    except AssertionError as error:
                        raise ValueError("Reused model's eligible query cohort differs from current data") from error
                    shutil.copytree(prior_model, folder)
                    if (_artifact_hashes(folder) != source_digests
                            or _artifact_hashes(prior_model) != source_digests):
                        raise ValueError("Baseline artifacts changed during validation/copy; reuse aborted")
                    _write_json(folder / "reuse.json", dict(source_run=source, seed=seed,
                                                          model_sha256=_hash(folder / "model.txt")))
                else:
                    res, det, imp = _record_model(frame, name, variants()[name], seed, params, eps, delta, folder)
                source_info = dict(variant=name, seed=seed, reused=bool(source), source_run=source,
                                   model_sha256=_hash(folder / "model.txt"),
                                   source_artifact_sha256=source_digests,
                                   preparation_seconds=time.perf_counter() - tick)
                state["model_sources"].append(source_info)
                res = res.assign(reused=bool(source), source_run=source or str(destination),
                                 current_fit_seconds=0.0 if source else res.fit_seconds)
                results.append(res)
                detailed.append(det)
                importances.append(imp)
                if mlflow:
                    _log_mlflow(mlflow, folder, res, cfg, variants()[name])
                print(f"{name} seed={seed}: {'reused ' + source if source else 'trained'}; saved {folder}")
                _write_json(destination / "run.json", state)
        combined = pd.concat(results, ignore_index=True)
        details = pd.concat(detailed, ignore_index=True)
        combined.to_csv(destination / "by_seed.csv", index=False)
        details.to_csv(destination / "gate_details.csv", index=False)
        pd.concat(importances, ignore_index=True).to_csv(destination / "importance.csv", index=False)
        _tables(combined, details, destination)
        state["timings"].update(experiment_wall_seconds=time.perf_counter() - started,
                                current_fit_seconds=sum(
                                    row.current_fit_seconds for row in
                                    combined[(combined.score == "learned") & (combined.eps == eps[0])].itertuples()))
        state["status"] = "complete"
        _write_json(destination / "run.json", state)
        print(f"COMPLETE: {destination}")
        return destination
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write_json(destination / "run.json", state)
        raise
