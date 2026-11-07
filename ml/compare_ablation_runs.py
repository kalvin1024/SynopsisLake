"""Compare completed source-feature runs without modifying or retraining them.

Only metadata and code/input fingerprints are used to select a baseline. The
previously observed test performance is never used for selection or calibration.
"""
import importlib.metadata
import json
import math
from pathlib import Path
import platform
import re
import shutil

import numpy as np
import pandas as pd

from gate_eval import PARAMS, split_ids
from model_ablation import DEFAULT_VARIANTS, _hash, _inputs, variants


_CODE_FILES = ("model_ablation.py", "features.py", "features_v1.py", "gate_eval.py")
_PACKAGES = ("lightgbm", "numpy", "pandas", "scipy", "scikit-learn")
_METRICS = ("auc_per_rate", "coverage_per_rate", "violation_per_rate",
            "violation_pooled_per_rate_gates", "fit_seconds")
_RESULT_COLS = ["variant", "seed", "n_features", "score", "eps", *_METRICS]
_DETAIL_COLS = ["variant", "seed", "score", "eps", "scope", "mergeRate", "test_count",
                "accepted_count", "violation_count", "coverage", "violation", "auc"]


def _read_json(path):
    try:
        with Path(path).open(encoding="utf-8") as stream:
            return json.load(stream)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read valid JSON at {path}: {error}") from error


def _values(value, cast):
    return [cast(item.strip()) for item in str(value).split(",")]


def _expected_config(config):
    names = [part.strip() for part in str(config.get("variants", DEFAULT_VARIANTS)).split(",")]
    expected = DEFAULT_VARIANTS.split(",")
    if names != expected:
        raise ValueError("This three-seed comparison requires the five source-feature variants in the documented order")
    if _values(config.get("seeds", "1,2"), int) != [1, 2]:
        raise ValueError("Set seeds to 1,2 to reuse the existing seed 0 run")
    if str(config.get("generalization", "false")).lower() != "false":
        raise ValueError("Set generalization=false for this three-seed comparison")
    eps = _values(config.get("epsilons", "0.25"), float)
    if len(eps) != 1 or not math.isfinite(eps[0]) or eps[0] <= 0:
        raise ValueError("This comparison requires one positive epsilon")
    delta = float(config.get("delta", "0.05"))
    if not 0 <= delta < 1:
        raise ValueError("Invalid delta")
    params = dict(PARAMS, n_estimators=int(config.get("max_trees", 4000)),
                  n_jobs=int(config.get("num_threads", 2)))
    if params["n_estimators"] < 1 or params["n_jobs"] < 1:
        raise ValueError("max_trees and num_threads must be positive")
    return names, eps, delta, params


def _identity(path):
    name = Path(path).name
    if name in ("queries-Ans.txt", "queries-meta.csv"):
        return ("data", name)
    if name == "rows.csv" or (name.startswith(("src_", "arm_")) and name.endswith(".txt")):
        return ("run", name)
    raise ValueError(f"Unexpected input file in manifest: {name}")


def _manifest_entries(path):
    records = _read_json(path)
    if not isinstance(records, list) or not records:
        raise ValueError(f"Empty or invalid input manifest: {path}")
    result = {}
    for item in records:
        try:
            key = _identity(item["source"])
            checksum = item["sha256"]
            if Path(item["used_path"]).name != key[1] or not re.fullmatch(r"[0-9a-f]{64}", checksum):
                raise ValueError("invalid filename or SHA-256")
            if key in result:
                raise ValueError(f"duplicate input {key}")
            result[key] = item
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Invalid input manifest {path}: {error}") from error
    return result


def _live_inputs(config):
    dataset = str(config["dataset"]).strip()
    if not dataset or dataset in (".", "..") or "/" in dataset or "\\" in dataset:
        raise ValueError("dataset must be a single directory name")
    data_dir = Path(config["data_root"]).resolve() / dataset
    run_dir = Path(config["sample_root"]).resolve() / dataset
    files = _inputs(data_dir, run_dir)
    result = {}
    for path in files:
        key = _identity(path)
        if key in result:
            raise ValueError(f"Duplicate input filename: {key}")
        before = path.stat()
        checksum = _hash(path)
        after = path.stat()
        if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
            raise ValueError(f"Input changed during verification: {path}")
        result[key] = checksum
    return result


def _versions():
    return {name: importlib.metadata.version(name) for name in _PACKAGES}


def _code_hashes(repo_root):
    directory = Path(repo_root).resolve() / "ml"
    try:
        return {name: _hash(directory / name) for name in _CODE_FILES}
    except OSError as error:
        raise ValueError(f"Cannot fingerprint the training code: {error}") from error


def _configuration(state, names, eps, delta, params, seed_values, versions, python, hashes, dataset):
    if state.get("status") != "complete":
        raise ValueError("run.json does not report status=complete")
    saved = state.get("config", {})
    if saved.get("dataset") != dataset:
        raise ValueError("dataset differs")
    if [p.strip() for p in str(saved.get("variants", "")).split(",")] != names:
        raise ValueError("variant list/order differs")
    if _values(saved.get("seeds", ""), int) != seed_values:
        raise ValueError("run has the wrong seeds")
    if _values(saved.get("epsilons", ""), float) != eps or float(saved.get("delta", -1)) != delta:
        raise ValueError("epsilon or delta differs")
    if str(saved.get("generalization", "false")).lower() != "false":
        raise ValueError("generalization setting differs")
    if saved.get("model_params") != params:
        raise ValueError("model parameters differ")
    if state.get("package_versions") != versions or state.get("python") != python:
        raise ValueError("Python or package versions differ")
    saved_hashes = state.get("code_sha256", {})
    if any(saved_hashes.get(name) != value for name, value in hashes.items()):
        raise ValueError("training/evaluation code fingerprints differ")


def _validate_models(folder, names, seeds, params):
    registry = variants()
    for name in names:
        for seed in seeds:
            model_dir = folder / "models" / name / f"seed_{seed}"
            schema = _read_json(model_dir / "schema.json")
            if (schema.get("features") != registry[name] or schema.get("variant") != name
                    or schema.get("seed") != seed or schema.get("params") != params):
                raise ValueError(f"model schema differs: {model_dir}")
            for filename in ("model.txt", "cutoffs.json", "cohorts.csv", "test_predictions.csv"):
                if not (model_dir / filename).is_file():
                    raise ValueError(f"Missing saved model artifact: {model_dir / filename}")


def _validate_snapshot(folder, records, state):
    copied = str(state.get("config", {}).get("copy_inputs", "true")).lower() == "true"
    if not copied:
        return
    base = (folder / "inputs").resolve()
    for key, item in records.items():
        used = Path(item["used_path"]).resolve()
        if not used.is_relative_to(base) or used.parent.name != key[0]:
            raise ValueError(f"Saved input snapshot is outside the run: {used}")
        try:
            checksum = _hash(used)
        except OSError as error:
            raise ValueError(f"Cannot read saved input snapshot {used}: {error}") from error
        if checksum != item["sha256"]:
            raise ValueError(f"Saved input snapshot changed: {used}")


def _expected_splits(frame, seeds):
    required = {"seed", "qId", "split", "early_stopping"}
    if not required.issubset(frame.columns):
        raise ValueError("split manifest is missing required columns")
    if set(frame.seed.unique()) != set(seeds):
        raise ValueError("split manifest has unexpected seeds")
    all_ids = None
    for seed in seeds:
        sample = frame[frame.seed == seed]
        if sample.qId.duplicated().any() or set(sample.split.unique()) != {"train", "calibration", "test"}:
            raise ValueError("split manifest has duplicate IDs or missing cohorts")
        ids = set(sample.qId)
        if all_ids is None:
            all_ids = ids
        if ids != all_ids:
            raise ValueError("eligible query cohort differs across seeds")
        tr, ca, te = split_ids(sample.qId, seed)
        for group, expected in (("train", tr), ("calibration", ca), ("test", te)):
            if set(sample.loc[sample.split == group, "qId"]) != expected:
                raise ValueError("saved split does not match the existing split algorithm")
        expected_early = set(sorted(tr)[::10])
        early = sample.early_stopping
        # CSV normally infers bool; reject unexpected text rather than truthy strings.
        if not early.isin([True, False]).all():
            raise ValueError("invalid early_stopping values")
        if set(sample.loc[early.astype(bool), "qId"]) != expected_early:
            raise ValueError("saved early-stopping cohort differs")
    return all_ids


def _candidate(folder, config, names, eps, delta, params, versions, python, hashes, live):
    state = _read_json(folder / "run.json")
    _configuration(state, names, eps, delta, params, [0], versions, python, hashes, config["dataset"])
    records = _manifest_entries(folder / "input_manifest.json")
    saved = {key: item["sha256"] for key, item in records.items()}
    if saved != live:
        raise ValueError("input file fingerprints differ from the previous run")
    _validate_snapshot(folder, records, state)
    _validate_models(folder, names, [0], params)
    for filename in ("v1_ablations.csv", "gate_details.csv", "splits.csv"):
        if not (folder / filename).is_file():
            raise ValueError(f"Missing prior result file: {folder / filename}")
    _expected_splits(pd.read_csv(folder / "splits.csv"), [0])
    frame = _read_frame(folder / "v1_ablations.csv", _RESULT_COLS)
    details = _read_frame(folder / "gate_details.csv", _DETAIL_COLS)
    _validate_frames(frame, details, names, [0], eps, _saved_rates(folder, names, [0]))
    return folder.resolve()


def find_compatible_baseline(config):
    """Identify one seed-0 run using provenance only, before any new model fits."""
    names, eps, delta, params = _expected_config(config)
    output = Path(config["output_root"]).resolve()
    live = _live_inputs(config)
    versions, python, hashes = _versions(), platform.python_version(), _code_hashes(config["repo_root"])
    selection = str(config.get("baseline_run", "auto")).strip()
    if selection and selection.lower() != "auto":
        candidate = Path(selection).expanduser().resolve()
        if not candidate.is_relative_to(output) or not candidate.is_dir():
            raise ValueError("baseline_run must be an existing run under your output_root")
        try:
            return _candidate(candidate, config, names, eps, delta, params, versions, python, hashes, live)
        except (OSError, ValueError, KeyError, TypeError) as error:
            raise ValueError(f"The chosen seed 0 run is incompatible: {error}") from error
    if not output.is_dir():
        raise FileNotFoundError(f"No prior results directory: {output}")
    found, errors = [], []
    for candidate in sorted(output.glob(str(config["dataset"]) + "-*")):
        if not candidate.is_dir():
            continue
        try:
            found.append(_candidate(candidate, config, names, eps, delta, params, versions, python, hashes, live))
        except (OSError, ValueError, KeyError, TypeError) as error:
            errors.append(f"{candidate.name}: {error}")
    if not found:
        raise FileNotFoundError("No compatible completed seed 0 run found. Check baseline_run, input snapshots, "
                                "versions, and settings. " + "; ".join(errors[:5]))
    if len(found) > 1:
        raise ValueError("Multiple compatible seed 0 runs found; set baseline_run explicitly: "
                         + ", ".join(str(p) for p in found))
    return found[0]


def _read_frame(path, required):
    try:
        frame = pd.read_csv(path)
    except (OSError, ValueError) as error:
        raise ValueError(f"Cannot read result CSV {path}: {error}") from error
    missing = set(required) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing columns in {path}: {sorted(missing)}")
    return frame


def _validate_frames(frame, details, names, seeds, eps, expected_rates=None):
    needed = {(name, seed, target) for name in names for seed in seeds for target in eps}
    learned = frame[frame.score == "learned"]
    keys = ["variant", "seed", "eps"]
    if learned.duplicated(keys).any() or set(map(tuple, learned[keys].itertuples(index=False, name=None))) != needed:
        raise ValueError("Missing or duplicate learned result for a variant, seed, or epsilon")
    if frame.duplicated(["variant", "seed", "eps", "score"]).any():
        raise ValueError("Duplicate result rows")
    if set(map(tuple, frame[["variant", "seed", "eps"]].itertuples(index=False, name=None))) != needed:
        raise ValueError("Unexpected variant, seed, or epsilon in result rows")
    keys = ["variant", "seed", "score", "eps", "scope", "mergeRate"]
    if details.duplicated(keys).any():
        raise ValueError("Duplicate gate detail rows")
    if set(details.scope.unique()) - {"pooled", "per_rate"}:
        raise ValueError("Unknown gate scope")
    rate_set = None
    # Validate every saved model and scorer has exactly one pooled record and
    # the same complete set of per-rate records. Never use observed risk to select.
    for variant, seed, score, target in frame[["variant", "seed", "score", "eps"]].itertuples(index=False, name=None):
        group = details[(details.variant == variant) & (details.seed == seed) & (details.score == score)
                        & (details.eps == target)]
        if len(group[group.scope == "pooled"]) != 1:
            raise ValueError("Missing pooled gate record")
        rates = group[group.scope == "per_rate"].mergeRate
        if rates.isna().any() or rates.duplicated().any() or rates.empty:
            raise ValueError("Missing or duplicate per-rate gate record")
        these = set(rates)
        if expected_rates is not None and these != expected_rates[(variant, seed)]:
            raise ValueError("Per-rate gate records differ from saved test predictions")
        if rate_set is None:
            rate_set = these
        if these != rate_set:
            raise ValueError("Per-rate coverage differs between models or seeds")
    expected_keys = set(map(tuple, frame[["variant", "seed", "score", "eps"]].itertuples(index=False, name=None)))
    actual_keys = set(map(tuple, details[["variant", "seed", "score", "eps"]].drop_duplicates().itertuples(index=False, name=None)))
    if actual_keys != expected_keys:
        raise ValueError("Gate records do not match the saved summaries")
    counts = details[["accepted_count", "violation_count", "test_count"]].apply(pd.to_numeric, errors="coerce")
    if (counts.isna().any().any() or (counts < 0).any().any()
            or (counts % 1 != 0).any().any()
            or (counts.violation_count > counts.accepted_count).any()
            or (counts.accepted_count > counts.test_count).any()
            or (counts.test_count == 0).any()):
        raise ValueError("Invalid test or accepted/violation counts in gate records")
    expected_risk = (counts.violation_count / counts.accepted_count).where(counts.accepted_count > 0)
    expected_coverage = counts.accepted_count / counts.test_count
    if (not np.allclose(details.violation, expected_risk, equal_nan=True)
            or not np.allclose(details.coverage, expected_coverage, equal_nan=False)):
        raise ValueError("Gate risk or coverage does not match its test counts")
    for row in learned.itertuples(index=False):
        if row.n_features != len(variants()[row.variant]):
            raise ValueError("Saved learned result has the wrong feature count")
        group = details[(details.variant == row.variant) & (details.seed == row.seed)
                        & (details.score == "learned") & (details.eps == row.eps)
                        & (details.scope == "per_rate")]
        n = group.accepted_count.sum()
        values = (group.auc.mean(), group.coverage.mean(), group.violation.mean(),
                  group.violation_count.sum() / n if n else np.nan)
        saved = (row.auc_per_rate, row.coverage_per_rate, row.violation_per_rate,
                 row.violation_pooled_per_rate_gates)
        if not np.allclose(values, saved, equal_nan=True):
            raise ValueError("Learned summary does not match its per-rate gate records")
    return rate_set


def _saved_rates(folder, names, seeds):
    result = {}
    for name in names:
        for seed in seeds:
            path = folder / "models" / name / f"seed_{seed}" / "test_predictions.csv"
            predictions = _read_frame(path, ["qId", "mergeRate"])
            if predictions.empty or predictions.mergeRate.isna().any():
                raise ValueError(f"Missing rates in test predictions: {path}")
            result[(name, seed)] = set(predictions.mergeRate)
    return result


def combine_runs(baseline, current):
    """Write a 0/1/2 seed comparison inside a completed NEW run only."""
    baseline, current = Path(baseline).resolve(), Path(current).resolve()
    if baseline == current:
        raise ValueError("baseline and current run must be different")
    previous, latest = _read_json(baseline / "run.json"), _read_json(current / "run.json")
    saved = latest.get("config", {})
    names, eps, delta, params = _expected_config(saved)
    versions, python, hashes = latest.get("package_versions"), latest.get("python"), latest.get("code_sha256", {})
    if not isinstance(versions, dict) or set(versions) != set(_PACKAGES) or not python:
        raise ValueError("Current run lacks Python/package provenance")
    if not all(hashes.get(name) for name in _CODE_FILES):
        raise ValueError("Current run lacks training/evaluation code fingerprints")
    code = {name: hashes[name] for name in _CODE_FILES}
    _configuration(previous, names, eps, delta, params, [0], versions, python, code, saved["dataset"])
    _configuration(latest, names, eps, delta, params, [1, 2], versions, python, code, saved["dataset"])
    records0 = _manifest_entries(baseline / "input_manifest.json")
    records12 = _manifest_entries(current / "input_manifest.json")
    if {key: value["sha256"] for key, value in records0.items()} != {key: value["sha256"] for key, value in records12.items()}:
        raise ValueError("Input changed between preflight and the new run; no comparison written")
    _validate_snapshot(baseline, records0, previous)
    _validate_snapshot(current, records12, latest)
    _validate_models(baseline, names, [0], params)
    _validate_models(current, names, [1, 2], params)
    splits0 = _read_frame(baseline / "splits.csv", ["seed", "qId", "split", "early_stopping"])
    splits12 = _read_frame(current / "splits.csv", ["seed", "qId", "split", "early_stopping"])
    if _expected_splits(splits0, [0]) != _expected_splits(splits12, [1, 2]):
        raise ValueError("Eligible query cohort differs between runs")
    frames, detailed = [], []
    for folder, seeds in ((baseline, [0]), (current, [1, 2])):
        frame = _read_frame(folder / "v1_ablations.csv", _RESULT_COLS)
        details = _read_frame(folder / "gate_details.csv", _DETAIL_COLS)
        _validate_frames(frame, details, names, seeds, eps, _saved_rates(folder, names, seeds))
        frames.append(frame)
        detailed.append(details)
    combined = pd.concat(frames, ignore_index=True)
    details = pd.concat(detailed, ignore_index=True)
    _validate_frames(combined, details, names, [0, 1, 2], eps)
    learned = combined[combined.score == "learned"]
    grouped = learned.groupby(["eps", "variant"], sort=True)
    summary = grouped[list(_METRICS)].agg(["mean", "std"])
    summary.columns = [f"{name}_{stat}" for name, stat in summary.columns]
    summary = summary.reset_index()
    counts = grouped.seed.nunique().rename("n_seed").reset_index()
    summary = summary.merge(counts, on=["eps", "variant"], validate="one_to_one")
    summary["n_seeds"] = summary["n_seed"]
    for metric in _METRICS:
        defined = grouped[metric].count().rename(f"{metric}_n").reset_index()
        summary = summary.merge(defined, on=["eps", "variant"], validate="one_to_one")
    rate = details[(details.score == "learned") & (details.scope == "per_rate")].copy()
    rate["above_target"] = (rate.accepted_count > 0) & (rate.violation > delta)
    rate_summary = (rate.groupby(["eps", "variant", "mergeRate"], sort=True)
                    .agg(n_seed=("seed", "nunique"), coverage_mean=("coverage", "mean"),
                         coverage_std=("coverage", "std"), violation_mean=("violation", "mean"),
                         violation_std=("violation", "std"), defined_risk_seeds=("violation", "count"),
                         above_target_seeds=("above_target", "sum"))
                    .reset_index())
    rate_summary["n_seeds"] = rate_summary["n_seed"]
    destination = current / "comparison"
    destination.mkdir(exist_ok=False)
    try:
        combined.to_csv(destination / "combined_by_seed.csv", index=False)
        summary.to_csv(destination / "summary_3_seeds.csv", index=False)
        details.to_csv(destination / "combined_gate_details.csv", index=False)
        rate_summary.to_csv(destination / "per_rate_3_seeds.csv", index=False)
        provenance = {
            "status": "complete", "baseline_run": str(baseline), "current_run": str(current),
            "seeds": [0, 1, 2], "variants": names, "epsilons": eps, "delta": delta,
            "input_sha256": {f"{key[0]}/{key[1]}": value["sha256"] for key, value in records0.items()},
            "code_sha256": code, "package_versions": versions, "python": python,
            "comparison_code_sha256": _hash(Path(__file__)),
            "notes": ["Selection uses provenance only, not observed test outcomes.",
                      "Means and standard deviations describe these three overlapping query splits; they are not confidence intervals.",
                      "The same query is evaluated at multiple histogram rates and must not be counted as independent data.",
                      "The empirical 5% calibration target does not guarantee test risk or provide a per-rate simultaneous guarantee."],
        }
        (destination / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False), encoding="utf-8")
    except Exception:
        shutil.rmtree(destination)
        raise
    return summary, rate_summary, destination
