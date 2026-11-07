"""Personal ablation runs using the existing features and LightGBM training code.

Inputs are read-only. Every run writes to a new personal directory. These are
offline quality experiments, not online query-latency measurements.
"""
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import platform
import shutil
import subprocess
import time
from uuid import uuid4

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from features import FEATURES, check_estimates
from features_v1 import FEATURES_V1, GROUPS, build_table_v1, check_source_estimates
from gate_eval import PARAMS, baseline_scores, cutoff, fit, split_ids


def variants():
    def drop(names, group):
        return [name for name in names if name not in GROUPS[group]]
    return {
        "v0": list(FEATURES), "v0_no_skew": drop(FEATURES, "skew"),
        "v1": list(FEATURES_V1), "v1_no_skew": drop(FEATURES_V1, "skew"),
        "v1_no_shift": drop(FEATURES_V1, "shift"),
        "v1_no_source": drop(FEATURES_V1, "source"),
        "geometry_only": GROUPS["shape"] + GROUPS["merge"],
        "v0_src_bound": FEATURES + ["src_bound_rel"],
        "v0_src_contrast": FEATURES + ["src_edge_contrast"],
        "v0_src_bound_contrast": FEATURES + ["src_bound_rel", "src_edge_contrast"],
    }


def _bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() not in ("true", "false"):
        raise ValueError("boolean settings must be true or false")
    return str(value).lower() == "true"


def _jsonable(value):
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _write_json(path, value):
    Path(path).write_text(json.dumps(_jsonable(value), indent=2, allow_nan=False), encoding="utf-8")


def _hash(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_output_root(output_root, repo_root, data_root, sample_root):
    if not str(output_root).strip():
        raise ValueError("Set output_root to a personal results directory before running")
    output = Path(output_root).expanduser().resolve()
    if output.is_relative_to(Path("/Workspace/Shared")):
        raise ValueError("output_root must be personal, not /Workspace/Shared")
    for protected in (repo_root, data_root, sample_root):
        if output.is_relative_to(Path(protected).resolve()):
            raise ValueError("output_root must be outside the checkout and input directories")
    return output


def make_split_manifest(df, seed):
    train, cal, test = split_ids(df.qId, seed)
    early = set(sorted(train)[::10])
    return pd.DataFrame([
        {"seed": seed, "qId": int(qid), "split": split, "early_stopping": qid in early}
        for split, ids in (("train", train), ("calibration", cal), ("test", test))
        for qid in sorted(ids)
    ])


def _auc(error, score, eps):
    bad = np.asarray(error) > eps
    return float(roc_auc_score(bad, score)) if len(np.unique(bad)) == 2 else np.nan


def evaluate_scores(cal, test, cal_scores, test_scores, epsilons=(0.25,), delta=0.05):
    """Same empirical cutoff as gate_eval; count cases and leave empty risk undefined."""
    details, summaries = [], []
    if not len(cal) or not len(test):
        raise ValueError("calibration and test cohorts must both be nonempty")
    for name, test_score in test_scores.items():
        sc, st = np.asarray(cal_scores[name]), np.asarray(test_score)
        if len(sc) != len(cal) or len(st) != len(test) or not np.isfinite(sc).all() or not np.isfinite(st).all():
            raise ValueError("scores must be finite and match calibration/test rows")
        for eps in epsilons:
            start = len(details)
            scopes = [("pooled", None, np.ones(len(cal), bool), np.ones(len(test), bool))]
            scopes += [("per_rate", float(rate), cal.mergeRate.to_numpy() == rate,
                        test.mergeRate.to_numpy() == rate) for rate in sorted(test.mergeRate.unique())]
            for scope, rate, c, t in scopes:
                ec, et = cal.relErr.to_numpy()[c], test.relErr.to_numpy()[t]
                threshold = float(cutoff(sc[c], ec, eps, delta)) if c.any() else -np.inf
                ac = sc[c] <= threshold
                at = st[t] <= threshold
                count, bad = int(at.sum()), int(((et > eps) & at).sum())
                risk = bad / count if count else np.nan
                details.append(dict(
                    score=name, eps=eps, scope=scope, mergeRate=rate,
                    cutoff=threshold if math.isfinite(threshold) else None,
                    accept_none=not math.isfinite(threshold),
                    calibration_count=int(c.sum()), calibration_accepted_count=int(ac.sum()),
                    calibration_violation_count=int(((ec > eps) & ac).sum()),
                    test_count=int(t.sum()), accepted_count=count, violation_count=bad,
                    coverage=count / int(t.sum()), violation=risk,
                    quality_target_met=(risk <= delta) if count else None,
                    auc=_auc(et, st[t], eps),
                ))
            group = pd.DataFrame(details[start:])
            pooled, rates = group[group.scope == "pooled"].iloc[0], group[group.scope == "per_rate"]
            n, bad = int(rates.accepted_count.sum()), int(rates.violation_count.sum())
            summaries.append(dict(
                score=name, eps=eps, auc=pooled.auc, coverage=pooled.coverage,
                violation=pooled.violation, accepted_count=int(pooled.accepted_count),
                violation_count=int(pooled.violation_count),
                auc_per_rate=rates.auc.mean(), coverage_per_rate=rates.coverage.mean(),
                violation_per_rate=rates.violation.mean(), rates_with_acceptances=int((rates.accepted_count > 0).sum()),
                rates_total=len(rates), accepted_cases_per_rate=n, violating_cases_per_rate=bad,
                violation_pooled_per_rate_gates=bad / n if n else np.nan,
                unique_test_queries=int(test.qId.nunique()), test_cases=len(test),
            ))
    return pd.DataFrame(summaries), pd.DataFrame(details)


def _inputs(data_dir, run_dir):
    files = [data_dir / "queries-Ans.txt", data_dir / "queries-meta.csv", run_dir / "rows.csv"]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing input: {path}. Git clone does not create logger outputs.")
    sources = sorted(run_dir.glob("src_*.txt"))
    if not sources:
        raise FileNotFoundError(f"Missing source histogram exports: {run_dir}/src_*.txt")
    rows = pd.read_csv(run_dir / "rows.csv", usecols=["method", "lonRes"])
    resolutions = rows.loc[rows.method == "ARM", "lonRes"].unique()
    if not len(resolutions):
        raise ValueError("rows.csv has no ARM records")
    files += sources + [run_dir / f"arm_{int(res)}.txt" for res in sorted(resolutions)]
    for path in files:
        if not path.is_file():
            raise FileNotFoundError(f"Missing histogram export: {path}")
    return files


def _snapshot(files, data_dir, run_dir, destination, copy_inputs):
    manifest = []
    for source in files:
        before = source.stat()
        target = source
        if copy_inputs:
            kind = "data" if source.parent == data_dir else "run"
            target = destination / "inputs" / kind / source.name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source, target)
        checksum = _hash(target)
        after = source.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise RuntimeError(f"Input changed during snapshot: {source}")
        manifest.append(dict(source=str(source), used_path=str(target), size_bytes=before.st_size, sha256=checksum))
    return manifest


def _train(df, features, seed, params, train_mask=None, test_mask=None):
    tr_ids, ca_ids, te_ids = split_ids(df.qId, seed)
    train_mask = pd.Series(True, index=df.index) if train_mask is None else train_mask
    test_mask = pd.Series(True, index=df.index) if test_mask is None else test_mask
    train = df[df.qId.isin(tr_ids) & train_mask]
    cal = df[df.qId.isin(ca_ids) & test_mask]
    test = df[df.qId.isin(te_ids) & test_mask]
    if train.qId.nunique() < 3 or not len(cal) or not len(test):
        raise ValueError("Not enough query IDs in train/calibration/test; do not resplit to fix this")
    start = time.perf_counter()
    model = fit(train, features, seed, params=params)
    fit_seconds = time.perf_counter() - start
    start = time.perf_counter()
    sc = {"learned": model.predict(cal[features]), **baseline_scores(cal)}
    st = {"learned": model.predict(test[features]), **baseline_scores(test)}
    return model, train, cal, test, sc, st, fit_seconds, time.perf_counter() - start


def _record_model(df, name, features, seed, params, epsilons, delta, folder, train_mask=None, test_mask=None):
    model, train, cal, test, sc, st, fit_seconds, predict_seconds = _train(
        df, features, seed, params, train_mask, test_mask)
    start = time.perf_counter()
    summary, details = evaluate_scores(cal, test, sc, st, epsilons, delta)
    evaluate_seconds = time.perf_counter() - start
    folder.mkdir(parents=True, exist_ok=False)
    model.booster_.save_model(str(folder / "model.txt"))
    _write_json(folder / "schema.json", dict(
        features=features, variant=name, seed=seed, params=params,
        best_iteration=model.best_iteration_, min_exact=10,
        label="log(max(abs(estimate-truth)/truth, 0.001))",
        score="predicted log-error; lower is better; thresholds are in score space",
    ))
    _write_json(folder / "cutoffs.json", details[details.score == "learned"].to_dict("records"))
    predictions = test[["qId", "mergeRate", "relErr"]].copy()
    predictions["score"] = st["learned"]
    predictions.to_csv(folder / "test_predictions.csv", index=False)
    # Export effective row cohorts as well as the common global query split.
    early = set(sorted(train.qId.unique())[::10])
    cohorts = pd.concat([part[["qId", "mergeRate"]].assign(split=split)
                         for split, part in (("train", train), ("calibration", cal), ("test", test))], ignore_index=True)
    cohorts["early_stopping"] = (cohorts.split == "train") & cohorts.qId.isin(early)
    cohorts.to_csv(folder / "cohorts.csv", index=False)
    for frame in (summary, details):
        frame.insert(0, "variant", name)
        frame.insert(1, "seed", seed)
        frame.insert(2, "n_features", len(features))
    summary["fit_seconds"], summary["prediction_seconds"] = fit_seconds, predict_seconds
    summary["evaluation_seconds"], summary["best_iteration"] = evaluate_seconds, model.best_iteration_
    importance = pd.DataFrame(dict(feature=features, gain=model.booster_.feature_importance("gain"), variant=name, seed=seed))
    summary.to_csv(folder / "results.csv", index=False)
    importance.to_csv(folder / "importance.csv", index=False)
    return summary, details, importance


def _log_mlflow(mlflow, folder, result, config, features):
    with mlflow.start_run(run_name=f"{config['dataset']}-{folder.parent.name}-{folder.name}"):
        mlflow.log_params(dict(dataset=config["dataset"], variant=folder.parent.name,
                              seed=folder.name.removeprefix("seed_"),
                              features=",".join(features), **config["model_params"]))
        mlflow.set_tag("personal_result_dir", str(folder))
        for row in result.itertuples():
            for field in ("auc_per_rate", "coverage_per_rate", "violation_per_rate", "fit_seconds"):
                value = float(getattr(row, field))
                if math.isfinite(value):
                    mlflow.log_metric(f"{row.score}_eps{row.eps}_{field}", value)
        # Keep the data snapshot local; log only the model and its evaluation artifacts.
        mlflow.log_artifacts(str(folder), artifact_path="model_evaluation")


def run_experiment(config):
    """Run personal training and return a persistent unique result directory."""
    experiment_started = time.perf_counter()
    cfg = dict(config)
    dataset = cfg["dataset"].strip()
    if not dataset or dataset in (".", "..") or "/" in dataset or "\\" in dataset:
        raise ValueError("dataset must be a single directory name")
    output = validate_output_root(cfg["output_root"], cfg["repo_root"], cfg["data_root"], cfg["sample_root"])
    registry = variants()
    names = list(registry) if cfg.get("variants", "v0,v1") == "all" else [v.strip() for v in cfg.get("variants", "v0,v1").split(",")]
    if not names or len(names) != len(set(names)) or set(names) - registry.keys():
        raise ValueError(f"Invalid variants; choose from {list(registry)}")
    seeds = [int(v.strip()) for v in str(cfg.get("seeds", "0")).split(",")]
    epsilons = [float(v.strip()) for v in str(cfg.get("epsilons", "0.25")).split(",")]
    delta = float(cfg.get("delta", 0.05))
    if not seeds or min(seeds) < 0 or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be distinct nonnegative integers")
    if not epsilons or any(not math.isfinite(e) or e <= 0 for e in epsilons) or not 0 <= delta < 1:
        raise ValueError("invalid error/violation targets")
    params = dict(PARAMS, n_estimators=int(cfg.get("max_trees", 4000)), n_jobs=int(cfg.get("num_threads", 2)))
    if params["n_estimators"] < 1 or params["n_jobs"] < 1:
        raise ValueError("max_trees and num_threads must be positive")
    cfg["model_params"] = params
    data_dir, run_dir = Path(cfg["data_root"]).resolve() / dataset, Path(cfg["sample_root"]).resolve() / dataset
    files = _inputs(data_dir, run_dir)
    mlflow = None
    if _bool(cfg.get("mlflow_enabled", True)):
        if not str(cfg.get("experiment", "")).startswith("/Users/"):
            raise ValueError("Set experiment to your personal /Users/<account>/... MLflow experiment")
        import mlflow
        mlflow.set_experiment(cfg["experiment"])
    destination = output / f"{dataset}-{uuid4().hex[:12]}"
    destination.mkdir(parents=True, exist_ok=False)
    state = dict(status="running", config=cfg, timings={}, notes=[
        "Offline quality evaluation; not end-to-end query latency.",
        "All variants build the full source feature table; input ablation is not an I/O saving.",
        "Exact counts below 10 are excluded by the existing pipeline.",
        "Empirical calibration does not guarantee 5% test risk; empty accepted risk is undefined.",
    ])
    try:
        started = time.perf_counter()
        manifest = _snapshot(files, data_dir, run_dir, destination, _bool(cfg.get("copy_inputs", True)))
        _write_json(destination / "input_manifest.json", manifest)
        if _bool(cfg.get("copy_inputs", True)):
            data_dir, run_dir = destination / "inputs/data", destination / "inputs/run"
        state["timings"]["input_preparation_seconds"] = time.perf_counter() - started
        started = time.perf_counter()
        df = build_table_v1(str(run_dir), str(data_dir))
        dropped = df.attrs.get("dropped_small_exact", 0)
        if not len(df):
            raise ValueError("No eligible rows after the existing exact>=10 filter")
        gaps = dict(merged=check_estimates(df), sources=check_source_estimates(df))
        queries = pd.read_csv(data_dir / "queries-Ans.txt", header=None)
        center = ((queries[0] + queries[1]) / 2).rename("center_lon").rename_axis("qId").reset_index()
        df = df.merge(center, on="qId", validate="many_to_one")
        if df.duplicated(["qId", "mergeRate"]).any():
            raise ValueError("duplicate query/histogram cases")
        if not np.isfinite(df[FEATURES_V1 + ["label", "relErr", "mergeRate", "center_lon"]].to_numpy(float)).all():
            raise ValueError("non-finite features/labels; check input compatibility")
        state.update(rows=len(df), unique_queries=int(df.qId.nunique()), dropped_small_exact=dropped, estimate_gaps=gaps)
        state["timings"]["feature_build_seconds"] = time.perf_counter() - started
        state["package_versions"] = {p: importlib.metadata.version(p) for p in
                                     ("lightgbm", "numpy", "pandas", "scipy", "scikit-learn")}
        state["python"] = platform.python_version()
        state["code_sha256"] = {name: _hash(Path(__file__).parent / name) for name in
                                ("model_ablation.py", "features.py", "features_v1.py", "gate_eval.py")}
        # Some Databricks runtimes expose source notebooks as workspace objects,
        # not ordinary .py files. Module hashes remain mandatory and reproducible.
        try:
            state["code_sha256"]["notebooks/06_model_v1.py"] = _hash(Path(__file__).parent / "notebooks/06_model_v1.py")
        except OSError as error:
            state["notebook_hash_unavailable"] = type(error).__name__
        try:
            state["git_commit"] = subprocess.check_output(["git", "-C", cfg["repo_root"], "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL).strip()
            state["git_dirty"] = bool(subprocess.check_output(["git", "-C", cfg["repo_root"], "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL))
        except (OSError, subprocess.CalledProcessError):
            state["git_commit"] = None
        pd.concat([make_split_manifest(df, seed) for seed in seeds], ignore_index=True).to_csv(destination / "splits.csv", index=False)
        _write_json(destination / "run.json", state)
        results, details, importances = [], [], []
        for name in names:
            for seed in seeds:
                folder = destination / "models" / name / f"seed_{seed}"
                res, det, imp = _record_model(df, name, registry[name], seed, params, epsilons, delta, folder)
                results.append(res)
                details.append(det)
                importances.append(imp)
                if mlflow:
                    _log_mlflow(mlflow, folder, res, cfg, registry[name])
                print(f"{name} seed={seed}: fit={res.fit_seconds.iloc[0]:.2f}s; saved {folder}")
        ablations = pd.concat(results, ignore_index=True)
        ablations.to_csv(destination / "v1_ablations.csv", index=False)
        pd.concat(details, ignore_index=True).to_csv(destination / "gate_details.csv", index=False)
        pd.concat(importances, ignore_index=True).to_csv(destination / "v1_importance.csv", index=False)
        cols = ["auc_per_rate", "coverage_per_rate", "violation_per_rate", "violation_pooled_per_rate_gates", "fit_seconds"]
        summary = ablations[ablations.score == "learned"].groupby(["eps", "variant"])[cols].agg(["mean", "std"])
        summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
        summary.reset_index().to_csv(destination / "v1_summary.csv", index=False)
        if _bool(cfg.get("generalization", False)):
            mid = (df.center_lon.min() + df.center_lon.max()) / 2
            comparisons = {"rate_high_to_low": (df.mergeRate >= .4, df.mergeRate < .4),
                           "west_to_east": (df.center_lon < mid, df.center_lon >= mid)}
            generalization = []
            for test_name, (train_side, test_side) in comparisons.items():
                for setting, mask in (("shifted", train_side), ("in_distribution", None)):
                    for seed in seeds:
                        folder = destination / "generalization" / test_name / setting / f"seed_{seed}"
                        res, det, _ = _record_model(df, "v1", FEATURES_V1, seed, params, epsilons, delta, folder, mask, test_side)
                        generalization.append(res.assign(test=test_name, setting=setting))
                        det.to_csv(folder / "gate_details.csv", index=False)
            pd.concat(generalization, ignore_index=True).to_csv(destination / "v1_generalization.csv", index=False)
        state["timings"]["experiment_wall_seconds"] = time.perf_counter() - experiment_started
        state["status"] = "complete"
        _write_json(destination / "run.json", state)
        print(f"COMPLETE: {destination}")
        return destination
    except Exception as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}")
        _write_json(destination / "run.json", state)
        raise
