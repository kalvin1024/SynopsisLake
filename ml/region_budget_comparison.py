"""A source-calibrated, held-out-region comparison with a common byte ceiling.

Reuses notebook22's SS codec, the team's full-v1 learner, and09b's Java
batch/Parquet timing scope. Target-region labels never select a representation,
model, byte budget or gate. Run ``python -m ml.region_budget_comparison --help``.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import json
import math
import os
import platform
from pathlib import Path
import shutil
import struct
import subprocess
import tempfile
import time
import traceback
from uuid import uuid4
from zipfile import ZipFile, ZIP_DEFLATED

import numpy as np
import pandas as pd

from . import features
from . import gate_eval
from . import region_budget_histogram as fine
from . import shape_e2e_inputs
from . import shape_team_model
from . import sidecar_workflow as prior

METHODS = ("source_shape", "v1", "fine_direct", "fine_reference")
MATCHED_REGION_PROTOCOL = 1
MATCHED_METHODS = ("source_shape", "fine_reference", "fine_direct")
MODEL_EXPORT_MODE = "single-write"
GATE_BYTES = 24  # little-endian float64 cutoff, epsilon, delta, actually loaded
DEFAULTS = dict(
    repo_root="", output_root="", datasets=[], seeds=[0, 1, 2],
    merge_rate=.2, epsilon=.25, delta=.05, n_test_queries=2000,
    n_calibration_queries=0, max_train_queries=0, calibration_fraction=2/7,
    train_min_exact=10, total_budget_bytes=0, shape_packet_budget_bytes=0,
    repeats=3, chunk_size=250000, include_fine_reference=True,
    java="java", javac="javac", classpath="", jar_path="", math3_path="",
    methods=None, prior_results_zip="", dataset_budget_bytes=None,
)


def _sha(path):
    return prior._sha(Path(path))


def _write_lightgbm_model(booster, path):
    # model_to_string includes LightGBM's pandas metadata. save_model reopens
    # the file in append mode to add that metadata, which fails on Volumes.
    payload = booster.model_to_string().encode("utf-8")
    digest = hashlib.sha256(payload).hexdigest()
    try:
        path.write_bytes(payload)
        if _sha(path) != digest:
            raise OSError(f"Model verification failed after writing {path}")
    except OSError as error:
        error.add_note(f"Notebook 23 could not write or read back its model at {path}")
        raise
    return digest


def check_model_export(output_root):
    """Verify the production writer with a tiny synthetic model at this destination."""
    import lightgbm as lgb

    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    data = lgb.Dataset(np.arange(4, dtype=float).reshape(-1, 1), label=[0., 0., 1., 1.])
    booster = lgb.train({"verbosity": -1, "num_threads": 1, "min_data_in_leaf": 1,
                         "min_data_in_bin": 1}, data, num_boost_round=1)
    with tempfile.TemporaryDirectory(prefix=".model-export-check-", dir=output) as scratch:
        _write_lightgbm_model(booster, Path(scratch)/"model.txt")


def _fit_and_publish_team_model(training_table, seed, destination):
    """Fit the same western v1 learner and write its artifacts sequentially."""
    # The older Shape comparison fingerprints its training helper for frozen
    # runs. Keep that helper intact and reuse the team's training entry point.
    model_name = "region_west_full_v1"
    names = list(shape_team_model.TEAM_FEATURES)
    matrix = shape_team_model._matrix(training_table, names)
    qids = training_table.qId.to_numpy()
    target = training_table.actual_error.to_numpy(float)
    if not training_table.split.eq("train").fillna(False).all():
        raise ValueError("Notebook 23 v1 may only fit western TRAIN rows")
    if (qids.dtype.kind not in "iu" or (qids < 0).any() or len(np.unique(qids)) < 12
            or not np.isfinite(target).all() or (target < 0).any()):
        raise ValueError("TRAIN needs at least 12 valid query IDs and finite nonnegative actual errors")

    table = matrix.copy()
    table["qId"] = qids
    table["label"] = np.log(np.maximum(target, features.LABEL_FLOOR))
    parameters = {**gate_eval.PARAMS, "n_jobs": 2}
    started = time.perf_counter()
    model = gate_eval.fit(table, names, int(seed), params=parameters)
    training_seconds = time.perf_counter() - started

    destination.mkdir(parents=True, exist_ok=False)
    model_sha = _write_lightgbm_model(model.booster_, destination/"model.txt")
    properties = destination/"model.properties"
    properties.write_text("label.transform=log\n", encoding="utf-8")
    schema = dict(schema_version=shape_team_model.SCHEMA_VERSION, family="team_v1_LightGBM",
        model_name=model_name, feature_names=names, augmented=False, label_transform="log",
        label_floor=features.LABEL_FLOOR, model_sha256=model_sha, model_properties_sha256=_sha(properties),
        implementation_sha256=shape_team_model._implementation_hashes(), seed=int(seed),
        best_iteration=int(model.best_iteration_), parameters=parameters,
        inner_validation="every tenth sorted outer-TRAIN qId, unchanged gate_eval.fit", refit_held_back=False)
    _json(destination/"model.schema.json", schema)
    fitted = shape_team_model.TeamModel(model.booster_, schema, destination)
    train_ids = sorted(map(int, np.unique(qids)))
    validation_ids = train_ids[::10]
    validation_rows = int(table.qId.isin(validation_ids).sum())
    fitted.training_metadata = dict(model_name=model_name, seed=int(seed), train_queries=len(train_ids),
        train_rows=len(table), inner_fit_rows=len(table)-validation_rows, inner_validation_rows=validation_rows,
        training_seconds=training_seconds, training_includes_inner_validation=True,
        best_iteration=schema["best_iteration"], runtime_model_bytes=fitted.model_bytes, feature_count=len(names),
        target_column="actual_error", outer_train_qIds=train_ids, inner_validation_qIds=validation_ids,
        target_sha256=hashlib.sha256(np.ascontiguousarray(target, np.float64).tobytes()).hexdigest(),
        feature_matrix_sha256=hashlib.sha256(np.ascontiguousarray(matrix, np.float64).tobytes()).hexdigest())
    _json(destination/"training.json", fitted.training_metadata)
    return fitted


def _safe(value):
    if isinstance(value, dict):
        return {str(k): _safe(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe(v) for v in value]
    if isinstance(value, (np.integer, np.bool_)):
        return value.item()
    if isinstance(value, (float, np.floating)) and not np.isfinite(value):
        return None if np.isnan(value) else ("Infinity" if value > 0 else "-Infinity")
    if isinstance(value, np.ndarray):
        return _safe(value.tolist())
    if isinstance(value, Path):
        return str(value)
    return value


def _json(path, value):
    Path(path).write_text(json.dumps(_safe(value), indent=2, allow_nan=False)+"\n")


def _now():
    return datetime.now(timezone.utc).isoformat()


def region_memberships(geometry, cut=None):
    """Legacy geometry-only cut, strict whole-rectangle regions, no label input."""
    columns = ["minLon", "maxLon", "minLat", "maxLat"]
    data = geometry[columns].copy()
    data.insert(0, "qId", geometry.qId.to_numpy() if "qId" in geometry else geometry.index.to_numpy())
    if (not len(data) or not np.isfinite(data[columns].to_numpy(float)).all()
            or (data.minLon >= data.maxLon).any() or (data.minLat >= data.maxLat).any()
            or data.qId.duplicated().any()):
        raise ValueError("Expected unique query IDs and finite positive query rectangles")
    centers = (data.minLon+data.maxLon)/2
    cut = float((centers.min()+centers.max())/2 if cut is None else cut)
    if not np.isfinite(cut):
        raise ValueError("region_cut must be finite")
    data["region"] = np.where(data.maxLon < cut, "west", np.where(data.minLon >= cut, "east", "crossing"))
    return data, cut


def source_roles(memberships, seed, calibration_fraction=2/7):
    """Use west only for TRAIN/CAL; east roles do not depend on seed or truth."""
    if not 0 < calibration_fraction < 1:
        raise ValueError("calibration_fraction must lie in (0,1)")
    rows = memberships[["qId", "region"]].copy()
    rows["split"] = np.where(rows.region.eq("east"), "test", "excluded")
    west = np.sort(rows.loc[rows.region.eq("west"), "qId"].to_numpy(int))
    if len(west) < 16:
        raise ValueError("Need at least 16 whole-rectangle west queries")
    shuffled = np.random.default_rng(int(seed)).permutation(west)
    ncal = max(1, int(len(west)*calibration_fraction))
    rows.loc[rows.qId.isin(shuffled[:ncal]), "split"] = "calibration"
    rows.loc[rows.qId.isin(shuffled[ncal:]), "split"] = "train"
    rows["seed"] = int(seed)
    return rows


def choose_cutoff(scores, errors, epsilon, delta):
    """Same complete-tie empirical prefix rule; +Inf scores always rejected."""
    scores, errors = np.asarray(scores, float), np.asarray(errors, float)
    if (scores.shape != errors.shape or not len(scores) or np.isnan(scores).any()
            or np.isnan(errors).any() or np.isneginf(scores).any() or (scores < 0).any()):
        raise ValueError("Invalid aligned calibration scores/errors")
    finite = np.isfinite(scores)
    if not finite.any():
        return -np.inf
    return float(gate_eval.cutoff(scores[finite], errors[finite], epsilon, delta))


def relative_error(answer, truth):
    answer, truth = np.asarray(answer, float), np.asarray(truth, float)
    result = np.full(answer.shape, np.inf)
    positive = truth > 0
    result[positive] = np.abs(answer[positive]-truth[positive])/truth[positive]
    result[(truth == 0) & (answer == 0)] = 0.
    result[np.isnan(answer)] = np.nan  # Exact has no approximate candidate.
    return result


def _accepted(frame):
    values = frame.fromSynopsis
    if values.dtype == bool:
        return values.to_numpy()
    if not values.astype(str).str.lower().isin(["true", "false"]).all():
        raise ValueError("Invalid Java acceptance flag")
    return values.astype(str).str.lower().eq("true").to_numpy()


def summarize(frame, truth, epsilon):
    """Each method is judged against its OWN returned synopsis candidate."""
    truth = np.asarray(truth, float)
    accepted = _accepted(frame)
    candidate_error = relative_error(frame.candidate, truth)
    returned_error = relative_error(frame.answer, truth)
    bad = candidate_error > epsilon
    count, total = int(accepted.sum()), len(frame)
    violations = int((accepted & bad).sum())
    return dict(queries=total, accepted_count=count, accepted_good_count=count-violations,
        accepted_bad_count=violations, fallback_count=total-count,
        coverage=count/total if total else np.nan,
        violation=violations/count if count else np.nan,
        candidate_mean_error=float(np.mean(candidate_error)) if total else np.nan,
        returned_mean_error=float(np.mean(returned_error)) if total else np.nan,
        synopsis_mean_error=float(np.mean(candidate_error[accepted])) if count else np.nan)


def _config(config):
    c = {**DEFAULTS, **dict(config)}
    c["repo_root"] = str(Path(c["repo_root"] or Path(__file__).resolve().parents[1]).resolve())
    c["output_root"] = str(Path(c["output_root"] or Path(c["repo_root"]).parent/"synopsislake-results"/"region_budget").resolve())
    if isinstance(c["seeds"], str):
        c["seeds"] = [int(v) for v in c["seeds"].split(",") if v.strip()]
    c["seeds"] = list(map(int, c["seeds"]))
    if not c["seeds"] or len(c["seeds"]) != len(set(c["seeds"])):
        raise ValueError("Need unique seeds")
    if not c["datasets"] or len({d["dataset"] for d in c["datasets"]}) != len(c["datasets"]):
        raise ValueError("datasets must list unique explicit dataset configurations")
    for key in ("n_test_queries", "n_calibration_queries", "max_train_queries", "train_min_exact",
                "total_budget_bytes", "shape_packet_budget_bytes", "repeats", "chunk_size"):
        if int(c[key]) != c[key] or c[key] < 0:
            raise ValueError(key+" must be a nonnegative integer")
        c[key] = int(c[key])
    if not c["repeats"] or not c["chunk_size"] or c["n_test_queries"] < 1:
        raise ValueError("Positive repeats/chunk_size/test-query count required")
    if not 0 < c["epsilon"] or not 0 <= c["delta"] < 1 or not 0 < c["calibration_fraction"] < 1:
        raise ValueError("Invalid error/calibration targets")
    if not 0 < c["merge_rate"] <= 1:
        raise ValueError("merge_rate must lie in (0,1]")
    if c["prior_results_zip"]:
        c["prior_results_zip"] = str(Path(c["prior_results_zip"]).expanduser().resolve())
        selected = tuple(c["methods"] if c["methods"] is not None else MATCHED_METHODS)
        if len(selected) != len(MATCHED_METHODS) or set(selected) != set(MATCHED_METHODS):
            raise ValueError("The SS-matched study requires source_shape, fine_reference and fine_direct")
        c["methods"] = list(MATCHED_METHODS)
        budgets = c["dataset_budget_bytes"]
        datasets = {d["dataset"] for d in c["datasets"]}
        if not isinstance(budgets, dict) or set(budgets) != datasets or any(
                type(value) is not int or value < 1 for value in budgets.values()):
            raise ValueError("Set one positive integer dataset_budget_bytes cap for each selected dataset")
        if c["total_budget_bytes"] or c["shape_packet_budget_bytes"]:
            raise ValueError("The matched study uses the verified per-dataset SS budget only")
    else:
        if c["dataset_budget_bytes"] is not None or c["methods"] is not None:
            raise ValueError("Per-dataset budgets and method selection require prior_results_zip")
        c["methods"] = list(METHODS if c["include_fine_reference"] else METHODS[:3])
    return c


def _build_java(c, root):
    if c["classpath"]:
        return c["classpath"], dict(status="provided", compile_seconds=None)
    math_path = Path(c["math3_path"]) if c["math3_path"] else None
    if not math_path or not math_path.is_file():
        candidates = list((Path.home()/".m2/repository/org/apache/commons/commons-math3").glob("*/*.jar"))
        candidates += list(Path(c["repo_root"]).parent.glob(".synopsislake/m2/org/apache/commons/commons-math3/*/*.jar"))
        if c["jar_path"]:
            candidates += list(Path(c["jar_path"]).parent.glob("commons-math3*.jar"))
        math_path = next((p for p in candidates if p.is_file()), None)
    if math_path is None:
        raise FileNotFoundError("Set math3_path to commons-math3.jar (already used by notebook09b)")
    if shutil.which(c["javac"]) is None:
        raise RuntimeError("A JDK with javac is needed to build the new entry point, or provide classpath containing it")
    source = Path(c["repo_root"])/"SynopsisLake/src/main/java"
    sources = sorted(source.rglob("*.java"))
    identity = hashlib.sha256("".join(_sha(p) for p in sources).encode()).hexdigest()
    classes = Path(tempfile.gettempdir())/"synopsislake-region-budget-build"/identity/"classes"
    marker = classes.parent/"complete.json"
    started = time.perf_counter()
    if not marker.is_file():
        classes.mkdir(parents=True, exist_ok=True)
        command = [c["javac"], "-source", "8", "-target", "8", "-cp", str(math_path),
                   "-sourcepath", str(source), "-d", str(classes),
                   str(source/"sigspatial25/RegionBudgetQueryExp.java")]
        result = subprocess.run(command, capture_output=True, text=True)
        (root/"java_build.log").write_text(result.stdout+result.stderr)
        if result.returncode:
            raise RuntimeError("New Java entry point failed to compile; see "+str(root/"java_build.log"))
        _json(marker, dict(source_sha256=identity, command=command))
    return str(classes)+os.pathsep+str(math_path), dict(status="compiled_current_source",
        source_sha256=identity, math3_sha256=_sha(math_path), compile_seconds=time.perf_counter()-started,
        cache=str(classes), scope="offline; excluded from measured query batches")


def _java(c, classpath, folder, method, query_file, *, seed=None, repeat=None,
          paths=None, cutoff=-np.inf, gate=None, score_only=False, feature_out=None,
          data_dir=None, n_files=None, timed=False):
    folder = Path(folder); folder.mkdir(parents=True, exist_ok=False)
    output, setup = folder/"answers.csv", folder/"setup.csv"
    arguments = ["-method", method, "-queryFile", str(query_file), "-outFile", str(output),
                 "-setupOut", str(setup), "-scoreOnly", str(bool(score_only)).lower(),
                 "-eps", str(c["epsilon"]), "-delta", str(c["delta"])]
    for key, value in (paths or {}).items():
        arguments += ["-"+key, str(value)]
    if gate is not None: arguments += ["-gatePath", str(gate)]
    else: arguments += ["-cutoff", str(float(cutoff))]
    if feature_out is not None: arguments += ["-featureOut", str(feature_out)]
    if not score_only:
        arguments += ["-dataDir", str(data_dir), "-numFile", str(n_files)]
    command = [c["java"], "-cp", classpath, "sigspatial25.RegionBudgetQueryExp", *arguments]
    _json(folder/"command.json", command)
    begin = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True)
    java_end = time.perf_counter()
    (folder/"stdout.log").write_text(result.stdout); (folder/"stderr.log").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(method+" Java execution failed: "+result.stderr[-3500:])
    frame = pd.read_csv(output)
    if timed: frame.to_parquet(folder/"answers.parquet", index=False)
    ended = time.perf_counter()
    record = dict(method=method, seed=seed, repeat=repeat, java_s=java_end-begin,
        parquet_s=ended-java_end, total_s=ended-begin, answers=str(output), setup=str(setup),
        raw_scan_s=float(frame.rawNs.sum()/1e9),gate_compute_s=float(frame.gateNs.sum()/1e9),
        synopsis_compute_s=float(frame.synNs.sum()/1e9))
    _json(folder/"timing.json", record)
    return frame, record


def _read_geometry(path):
    # Do not load the test truth column, even to choose eligibility or IDs.
    data = pd.read_csv(path, header=None, usecols=[0, 1, 2, 3], dtype=float)
    data.columns = ["minLon", "maxLon", "minLat", "maxLat"]
    data.insert(0, "qId", np.arange(len(data), dtype=int))
    return data


def _west_labels(path, ids):
    """Only parse truth fields for the explicitly permitted west IDs."""
    wanted, result = set(map(int, ids)), {}
    with Path(path).open(newline="") as stream:
        for qid, row in enumerate(csv.reader(stream)):
            if qid in wanted:
                if len(row) < 5: raise ValueError("West training/calibration queries require saved exact counts")
                value = float(row[4])
                if not np.isfinite(value) or value < 0: raise ValueError("Invalid west exact count")
                result[qid] = value
    if set(result) != wanted: raise ValueError("West IDs are missing from the query source")
    return result


def _select(ids, cap, seed):
    ids = np.sort(np.asarray(list(ids), dtype=int))
    return ids if not cap or cap >= len(ids) else np.sort(np.random.default_rng(seed).choice(ids, cap, replace=False))


def _query_file(path, members, ids):
    members.set_index("qId").loc[ids, ["minLon", "maxLon", "minLat", "maxLat"]].to_csv(path, header=False, index=False)
    pd.DataFrame({"qId": np.arange(len(ids)), "source_qId": ids}).to_csv(str(path)+".ids.csv", index=False)


def _dataset_inputs(config, c):
    item = dict(config)
    if not str(item.get("dataset", "")).strip(): raise ValueError("dataset name required")
    run = Path(item["run_dir"]).expanduser().resolve()
    raw = Path(item["data_dir"]).expanduser().resolve()
    dataset = Path(item.get("dataset_dir") or run).expanduser().resolve()
    queries = Path(item.get("queries_file") or dataset/"queries-Ans.txt").expanduser().resolve()
    if not queries.is_file(): raise FileNotFoundError("Explicit matching query file missing: "+str(queries))
    sources = sorted(run.glob("src_*.txt"), key=lambda p:int(p.stem.split("_")[-1]))
    if not sources: raise FileNotFoundError("Missing source exports in "+str(run))
    source_ids = [int(p.stem.split("_")[-1]) for p in sources]
    if source_ids != list(range(len(sources))): raise ValueError("Java raw files must have contiguous source IDs from0")
    raw_paths = []
    for sid in source_ids:
        p = next((v for v in (raw/f"{sid}.csv", raw/f"{sid}.csv.gz") if v.is_file()), None)
        if p is None: raise FileNotFoundError(f"Missing raw source {raw}/{sid}.csv[.gz]")
        raw_paths.append(p)
    arms = []
    for p in sorted(run.glob("arm_*.txt")):
        with p.open() as f: rate = float(f.readline())
        arms.append((abs(rate-c["merge_rate"]), str(p), rate))
    if not arms: raise FileNotFoundError("No ARM histogram exports in "+str(run))
    _, hpath, rate = min(arms)
    item.update(run_dir=str(run), data_dir=str(raw), queries_file=str(queries),
        sources=sources, raw_paths=raw_paths, H_path=Path(hpath), actual_rate=rate)
    return item


def _native_size(sources):
    total, cells = 0, 0
    for source in sources:
        _, _, _, counts = features.load_hist(source)
        nx, ny = counts.shape; cells += counts.size
        total += 24+8*(nx+ny+2+counts.size)+64+5*counts.size
    return int(total), int(cells)


def _matched_prior(c, inputs):
    from .region_budget_matched_inputs import load_prior

    return load_prior(c, inputs)


def _prepare_dataset(c, item, root, classpath):
    folder = root/item["dataset"]; folder.mkdir()
    payload = folder/"payload"; payload.mkdir()
    H = features.load_hist(item["H_path"])
    prior.save_histogram(payload/"H.hist", H)
    matched = item.get("matched_prior")
    if matched is not None and _sha(payload/"H.hist") != matched["h_sha256"]:
        raise ValueError("Serialized H differs from the previous run: " + item["dataset"])
    use_v1 = "v1" in c["methods"]
    if use_v1:
        source_dir = payload/"sources"; source_dir.mkdir()
        for p in item["sources"]: shutil.copyfile(p, source_dir/p.name)
    geometry = _read_geometry(item["queries_file"])
    cut = matched["region_cut"] if matched is not None else item.get("region_cut")
    members, cut = region_memberships(geometry, cut)
    members.to_csv(folder/"region_memberships.csv", index=False)
    if matched is None:
        test_ids = _select(members.loc[members.region.eq("east"), "qId"], c["n_test_queries"], 90127)
    else:
        test_ids = matched["test_ids"]
        pd.DataFrame({"source_qId": matched["previous_ids"]}).to_csv(
            folder/"previous_east_query_ids.csv", index=False)
    if not len(test_ids): raise ValueError("No wholly eastern test queries: "+item["dataset"])
    test_file = folder/"test_queries_geometry_only.csv"
    _query_file(test_file, members, test_ids)
    west_ids = members.loc[members.region.eq("west"), "qId"].to_numpy(int)
    west_truth = _west_labels(item["queries_file"], west_ids)
    audit_ids = _select(west_ids, min(32,len(west_ids)), 9311)
    audit_file=folder/"west_truth_audit_geometry_only.csv"
    _query_file(audit_file,members,audit_ids)
    audit,_=_java(c,classpath,folder/"west_truth_audit","exact",audit_file,
        data_dir=item["data_dir"],n_files=len(item["sources"]))
    saved=np.array([west_truth[int(q)] for q in audit_ids])
    if not np.array_equal(audit.answer.to_numpy(float),saved):
        raise ValueError("Western saved labels do not match real raw counts; correct the dataset before fitting")
    pd.DataFrame(dict(source_qId=audit_ids,saved_truth=saved,raw_truth=audit.answer)).to_csv(folder/"west_truth_parity.csv",index=False)
    if use_v1:
        west_file = folder/"west_queries_geometry_only.csv"
        _query_file(west_file, members, west_ids)
        feature_file = folder/"west_features_java.csv"
        answers, _ = _java(c, classpath, folder/"west_feature_extraction", "v1_features", west_file,
            score_only=True, paths={"answerHist":payload/"H.hist", "sourceDir":source_dir}, feature_out=feature_file)
        matrix = pd.read_csv(feature_file)
        if len(matrix) != len(west_ids) or not np.array_equal(matrix.qId, np.arange(len(west_ids))):
            raise ValueError("Java TRAIN feature rows are not aligned")
        matrix["qId"] = west_ids
        matrix["actual_error"] = relative_error(answers.candidate, np.array([west_truth[int(q)] for q in west_ids]))
        matrix["truth"] = [west_truth[int(q)] for q in west_ids]
    all_roles, models, calibration_ids, train_records = [], {}, {}, []
    for seed in c["seeds"]:
        roles = source_roles(members, seed, c["calibration_fraction"])
        cal = _select(roles.loc[roles.split.eq("calibration"), "qId"], c["n_calibration_queries"], seed+913)
        calibration_ids[seed] = cal
        _query_file(folder/f"calibration_seed_{seed}.csv", members, cal)
        if matched is not None:
            roles.loc[roles.split.eq("train"), "split"] = "west_unused"
            roles.loc[roles.region.eq("east"), "split"] = "east_unused"
            roles.loc[roles.qId.isin(matched["previous_ids"]), "split"] = "previous_test"
            roles.loc[roles.qId.isin(test_ids), "split"] = "test"
        all_roles.append(roles)
        if use_v1:
            train_ids = set(roles.loc[roles.split.eq("train"), "qId"])
            train = matrix[matrix.qId.isin(train_ids) & (matrix.truth >= c["train_min_exact"])].copy()
            selected = _select(train.qId, c["max_train_queries"], seed+117)
            train = train[train.qId.isin(selected)].copy(); train["split"] = "train"
            model_dir = payload/f"v1_seed_{seed}"
            print(f"{item['dataset']}: fitting v1 (seed {seed}, {len(train)} western TRAIN queries)", flush=True)
            model = _fit_and_publish_team_model(train, seed, model_dir)
            models[seed] = model_dir
            train_records.append(dict(seed=seed, train_queries=len(train), calibration_queries=len(cal),
                model_bytes=(model_dir/"model.txt").stat().st_size,
                training_seconds=model.training_metadata["training_seconds"]))
    pd.concat(all_roles, ignore_index=True).to_csv(folder/"region_splits.csv", index=False)
    if use_v1:
        pd.DataFrame(train_records).to_csv(folder/"training_costs.csv", index=False)
    native_bytes, cells = _native_size(item["sources"])
    hbytes = (payload/"H.hist").stat().st_size
    v1_floor = None
    if use_v1:
        source_bytes = sum(p.stat().st_size for p in source_dir.glob("src_*.txt"))
        v1_floor = max(hbytes+source_bytes+(p/"model.txt").stat().st_size+(p/"model.properties").stat().st_size+GATE_BYTES for p in models.values())
    ss_floor = hbytes+native_bytes+64+8*len(item["sources"])+(cells+7)//8+GATE_BYTES
    minimum = max(ss_floor, v1_floor or 0, GATE_BYTES+fine.payload_bytes(1)
                  + (hbytes if "fine_reference" in c["methods"] else 0))
    info = dict(item, folder=folder, payload=payload, H=H, models=models, calibration_ids=calibration_ids,
        west_truth=west_truth, test_ids=test_ids, test_file=test_file, region_cut=cut, members=members,
        native_bytes=native_bytes, source_cells=cells, budget_floor=minimum)
    _json(folder/"preparation.json", dict(dataset=item["dataset"], status="west_calibration_inputs_ready_no_east_labels_used",
        region_cut=cut, strict_whole_rectangle=True, region_counts=members.region.value_counts().to_dict(),
        test_queries=len(test_ids), test_geometry_sha256=_sha(test_file), actual_merge_rate=item["actual_rate"],
        prior_test_excluded=len(matched["previous_ids"]) if matched is not None else 0,
        v1_budget_floor=v1_floor, ss_budget_floor=ss_floor, train_calibration_side="west only",
        query_file_sha256=_sha(item["queries_file"]),original_H_sha256=_sha(item["H_path"]),
        source_histogram_sha256={p.name:_sha(p) for p in item["sources"]},
        test_label_access="none; final truth will come from frozen-phase Java exact scans"))
    return info


def _paths(info, method, seed):
    p = info["payload"]
    common = {"answerHist":p/"H.hist", "mergeRate":info["actual_rate"]}
    if method == "source_shape": return {**common, "shapeDir":p/"source_shape/native", "shapePacket":p/"source_shape/shape.bin"}
    if method == "v1": return {**common, "sourceDir":p/"sources", "modelPath":info["models"][seed]/"model.txt", "propsPath":info["models"][seed]/"model.properties"}
    if method == "fine_direct": return {"finePath":p/"fine_direct.counts"}
    if method == "fine_reference": return {**common, "finePath":p/"fine_reference.counts"}
    return {}


def _files_for(info, method, seed, gate):
    p=info["payload"]
    files=[("gate_bytes", gate)]
    if method != "fine_direct": files.append(("histogram_bytes", p/"H.hist"))
    if method == "v1":
        files += [("source_histogram_bytes", q) for q in sorted((p/"sources").glob("src_*.txt"))]
        files += [("model_bytes", info["models"][seed]/"model.txt"), ("model_properties_bytes", info["models"][seed]/"model.properties")]
    elif method == "source_shape":
        files += [("source_summary_bytes", q) for q in sorted((p/"source_shape/native").glob("*")) if q.is_file()]
        files += [("ss_packet_bytes", p/"source_shape/shape.bin")]
    else: files.append(("fine_histogram_bytes", p/("fine_direct.counts" if method == "fine_direct" else "fine_reference.counts")))
    return files


def _freeze_dataset(c, info, budget, classpath, methods):
    folder, p = info["folder"], info["payload"]
    matched = info.get("matched_prior")
    hbytes = (p/"H.hist").stat().st_size
    packet_cap = budget-hbytes-info["native_bytes"]-GATE_BYTES
    if c["shape_packet_budget_bytes"]: packet_cap=min(packet_cap, c["shape_packet_budget_bytes"])
    if matched is not None:
        # The prior allocation cap is also stored in the packet header. Reuse
        # it to reproduce the same bytes, then enforce the measured payload cap.
        packet_cap = matched["shape_packet_budget"]
    shape_path = shape_e2e_inputs.prepare_shape_metadata(info["run_dir"], info["data_dir"], p/"source_shape",
        packet_budget_bytes=packet_cap, chunk_size=c["chunk_size"])
    shape_meta = json.loads((shape_path/"manifest.json").read_text())
    if matched is not None:
        raw_hashes = {v["source_id"]: v["raw_sha256"] for v in shape_meta["source_inputs"]}
        shape_hashes = {name: digest for name, digest in shape_meta["artifact_sha256"].items()
                        if name == "shape.bin" or name.startswith("native/")}
        ss_bytes = hbytes + GATE_BYTES + sum((shape_path/name).stat().st_size for name in shape_hashes)
        if raw_hashes != matched["raw_sha256"] or shape_hashes != matched["shape_artifact_sha256"]:
            raise ValueError(f"{info['dataset']}: raw data or rebuilt SS payload differs from the prior run")
        if ss_bytes != budget:
            raise ValueError(f"{info['dataset']}: SS uses {ss_bytes} bytes, expected exactly {budget}")
    roots=[features.load_hist(v) for v in info["sources"]]
    root=(min(v[1][0] for v in roots),max(v[1][-1] for v in roots),min(v[2][0] for v in roots),max(v[2][-1] for v in roots))
    # Construct the largest direct grid, then stream once more only if the
    # reference's H charge changes the maximum feasible grid dimensions.
    fine_info = fine.build_fine_histogram(info["raw_paths"], root, budget-GATE_BYTES,
        p/"fine_direct.counts", chunk_size=c["chunk_size"])
    reference_info=None
    if "fine_reference" in methods:
        reference_info=fine.build_fine_histogram(info["raw_paths"], root, budget-hbytes-GATE_BYTES,
            p/"fine_reference.counts", chunk_size=c["chunk_size"])
    _json(folder/"fine_construction.json",dict(direct=fine_info,reference=reference_info))
    offline=[dict(dataset=info["dataset"],phase="source_shape_construction",seconds=shape_meta["construction_seconds"]),
             dict(dataset=info["dataset"],phase="fine_direct_construction",seconds=fine_info["construction_seconds"])]
    if reference_info:offline.append(dict(dataset=info["dataset"],phase="fine_reference_construction",seconds=reference_info["construction_seconds"]))
    gates, cal_records, costs, files, cal_scores = {}, [], [], [], []
    for seed in c["seeds"]:
        cal_ids = info["calibration_ids"][seed]
        truth = np.array([info["west_truth"][int(q)] for q in cal_ids])
        for method in methods:
            frame, cal_time = _java(c, classpath, folder/"calibration"/f"seed_{seed}"/method, method,
                folder/f"calibration_seed_{seed}.csv", seed=seed, paths=_paths(info,method,seed),score_only=True)
            error = relative_error(frame.candidate,truth)
            threshold = choose_cutoff(frame.errorScore,error,c["epsilon"],c["delta"])
            gate = p/f"gate_{method}_seed_{seed}.bin"
            gate.write_bytes(struct.pack("<ddd",threshold,c["epsilon"],c["delta"]))
            gates[seed,method]=gate
            accept=np.isfinite(frame.errorScore.to_numpy(float)) & (frame.errorScore.to_numpy(float)<=threshold)
            offline.append(dict(dataset=info["dataset"],phase="calibration_scores",seed=seed,method=method,seconds=cal_time["total_s"]))
            cal_scores.append(pd.DataFrame(dict(dataset=info["dataset"],seed=seed,method=method,source_qId=cal_ids,
                candidate=frame.candidate.to_numpy(),truth=truth,score=frame.errorScore.to_numpy(),actual_error=error,cutoff=threshold)))
            cal_records.append(dict(dataset=info["dataset"],seed=seed,method=method,cutoff=threshold,
                calibration_queries=len(frame),accepted_count=int(accept.sum()),accepted_bad_count=int((accept & (error>c["epsilon"])).sum()),
                violation=float((error[accept]>c["epsilon"]).mean()) if accept.any() else np.nan,
                labels_region="west only",gate_sha256=_sha(gate)))
            components={k:0 for k in ("histogram_bytes","source_histogram_bytes","source_summary_bytes","ss_packet_bytes","model_bytes","model_properties_bytes","fine_histogram_bytes","gate_bytes")}
            for component,path in _files_for(info,method,seed,gate):
                size=path.stat().st_size;components[component]+=size
                files.append(dict(dataset=info["dataset"],seed=seed,method=method,component=component,path=str(path),bytes=size,sha256=_sha(path)))
            used=sum(components.values())
            if used>budget: raise AssertionError("Runtime payload exceeds common total cap")
            costs.append(dict(dataset=info["dataset"],seed=seed,method=method,total_budget_bytes=budget,
                **components,used_bytes=used,unused_bytes=budget-used))
    pd.DataFrame(cal_records).to_csv(folder/"calibration.csv",index=False)
    pd.DataFrame(cal_records).to_csv(folder/"frozen_gates.csv",index=False)
    pd.DataFrame(costs).to_csv(folder/"payload_costs.csv",index=False)
    pd.DataFrame(files).to_csv(folder/"payload_files.csv",index=False)
    pd.DataFrame(offline).to_csv(folder/"offline_costs.csv",index=False)
    pd.concat(cal_scores,ignore_index=True).to_csv(folder/"calibration_scores.csv",index=False)
    info.update(gates=gates,shape_metadata=shape_meta,raw_rows=shape_meta["raw_rows"],raw_file_bytes=sum(q.stat().st_size for q in info["raw_paths"]))
    lock=dict(status="all_models_representations_budget_and_gates_frozen_before_east_labels",
        created_utc=_now(),test_geometry_sha256=_sha(info["test_file"]),total_budget_bytes=budget,
        payload_files=files,calibration_labels="west only",source_summaries="offline unlabelled source data from the full indexed dataset",
        target_region_labels_used_for_training=False,target_region_labels_used_for_budget=False,
        target_region_labels_used_for_calibration=False,shape_metadata=shape_meta,
        fine_direct=dict(answer="fine count estimate F",score="count-derived worst relative-error interval bound; finite west-calibrated threshold"),
        fine_reference=dict(answer="unchanged H",score="abs(H-F)/max(F,1)",separate_payload_budget=True),
        stored_test_labels_accessed=False,test_truth="independent Java raw scan after this lock")
    if matched is not None:
        lock.update(prior_archive_sha256=matched["archive_sha256"],
            test_id_mapping_sha256=_sha(str(info["test_file"])+".ids.csv"),
            previous_east_ids_sha256=_sha(folder/"previous_east_query_ids.csv"),
            previous_east_ids_excluded=len(matched["previous_ids"]),
            original_H_sha256=matched["h_sha256"], ss_payload_matches_prior=True)
    _json(folder/"protocol.json",lock)
    info["protocol_sha256"]=_sha(folder/"protocol.json")


def _phase_records(record, dataset):
    frame=pd.read_csv(record["setup"])
    for r in frame.to_dict("records"):
        yield dict(dataset=dataset,method=record["method"],seed=record["seed"],repeat=record["repeat"],**r)


def _evaluate_dataset(c,info,classpath,methods):
    folder=info["folder"];jobs=[]
    frozen=json.loads((folder/"protocol.json").read_text())
    if info.get("matched_prior") is not None and (
            _sha(info["test_file"]) != frozen["test_geometry_sha256"] or
            _sha(str(info["test_file"])+".ids.csv") != frozen["test_id_mapping_sha256"]):
        raise RuntimeError("Frozen test queries or source IDs changed")
    for item in frozen["payload_files"]:
        if Path(item["path"]).stat().st_size!=item["bytes"] or _sha(item["path"])!=item["sha256"]:
            raise RuntimeError("Frozen runtime payload changed: "+item["path"])
    for item in info["shape_metadata"]["source_inputs"]:
        if _sha(item["raw_path"])!=item["raw_sha256"]:
            raise RuntimeError("Raw data changed after source-summary construction: "+item["raw_path"])
    for repeat in range(c["repeats"]):
        jobs.append(("exact",None,repeat))
        for seed in c["seeds"]:
            jobs += [(method,seed,repeat) for method in methods]
    jobs=[jobs[i] for i in np.random.default_rng(82713).permutation(len(jobs))]
    _json(folder/"execution_order.json",jobs)
    records=[];frames={};phases=[]
    for method,seed,repeat in jobs:
        if _sha(folder/"protocol.json")!=info["protocol_sha256"]: raise RuntimeError("Frozen evaluation protocol changed")
        gate=info["gates"].get((seed,method))
        if gate is not None:
            expected=next(r["sha256"] for r in json.loads((folder/"protocol.json").read_text())["payload_files"] if r["path"]==str(gate))
            if _sha(gate)!=expected:raise RuntimeError("Frozen gate changed")
        label=f"{method}_seed_{seed}_repeat_{repeat}"
        frame,record=_java(c,classpath,folder/"timed"/label,method,info["test_file"],seed=seed,repeat=repeat,
            paths=_paths(info,method,seed),gate=gate,data_dir=info["data_dir"],n_files=len(info["sources"]),timed=True)
        if len(frame)!=len(info["test_ids"]):raise ValueError("Java returned a different test query count")
        frame["source_qId"]=info["test_ids"]
        if (method,seed) in frames:
            for key in ("candidate","answer","errorScore","fromSynopsis"):
                a,b=frames[method,seed][key].to_numpy(),frame[key].to_numpy()
                if key=="fromSynopsis":same=np.array_equal(a,b)
                else:same=np.allclose(a.astype(float),b.astype(float),rtol=0,atol=1e-10,equal_nan=True)
                if not same:raise ValueError("Repeated Java answers/scores differ: "+label+" "+key)
        frames[method,seed]=frame
        record["dataset"]=info["dataset"];records.append(record)
        phases.extend(_phase_records(record,info["dataset"]))
        print(f"{info['dataset']} {method} seed={seed} repeat={repeat}: {record['total_s']:.3f}s",flush=True)
    timing=pd.DataFrame(records);timing.to_csv(folder/"timing.csv",index=False)
    phase_table=pd.DataFrame(phases);phase_table.to_csv(folder/"phase_timing.csv",index=False)
    truth=frames["exact",None].answer.to_numpy(float)
    if not np.isfinite(truth).all() or (truth<0).any():raise ValueError("Invalid real exact counts")
    pd.DataFrame({"source_qId":info["test_ids"],"truth":truth}).to_csv(folder/"test_labels_after_freeze.csv",index=False)
    exact_time=float(timing.loc[timing.method.eq("exact"),"total_s"].median())
    costs=pd.read_csv(folder/"payload_costs.csv");summaries=[];quality=[];slices=[]
    for (method,seed),frame in frames.items():
        accepted=_accepted(frame)
        if method=="exact":
            if accepted.any() or not np.array_equal(frame.answer,truth):raise ValueError("Exact parity failed")
        else:
            threshold=struct.unpack("<ddd",info["gates"][seed,method].read_bytes())[0]
            score=frame.errorScore.to_numpy(float)
            if not np.array_equal(accepted,np.isfinite(score)&(score<=threshold)):raise ValueError("Gate reconstruction failed")
            if not np.allclose(frame.answer[accepted],frame.candidate[accepted],rtol=0,atol=1e-10):raise ValueError("Accepted candidate changed")
            if not np.array_equal(frame.answer[~accepted],truth[~accepted]):raise ValueError("Fallback is not the real exact answer")
        if ((frame.rawNs.to_numpy(float)[~accepted]<=0).any() or (frame.rawNs.to_numpy(float)[accepted]!=0).any()):
            raise ValueError("Raw scan timings disagree with fallback decisions")
        group=timing[timing.method.eq(method)]
        if method!="exact":group=group[group.seed.eq(seed)]
        total=float(group.total_s.median())
        stage=phase_table[phase_table.method.eq(method)]
        stage=stage[stage.seed.isna()] if seed is None else stage[stage.seed.eq(seed)]
        raw_load=stage[stage.stage.eq("raw_points_load")].elapsed_ns.to_numpy(float)/1e9
        loads=stage[~stage.stage.isin(["raw_points_load","query_file_load","answer_and_csv_output_loop"])]
        payload_load=loads.groupby("repeat").elapsed_ns.sum()/1e9
        result=dict(dataset=info["dataset"],seed=seed,method=method,merge_rate=info["actual_rate"],
            epsilon=c["epsilon"],delta_target=c["delta"],region_cut=info["region_cut"],**summarize(frame,truth,c["epsilon"]),
            batch_median_s=total,batch_min_s=float(group.total_s.min()),batch_max_s=float(group.total_s.max()),
            exact_batch_median_s=exact_time,seconds_saved_vs_exact=exact_time-total,speedup_vs_exact=exact_time/total,
            raw_load_median_s=float(np.median(raw_load)),payload_load_median_s=float(payload_load.median()) if len(payload_load) else 0.,
            raw_scan_batch_mean_s=float(group.raw_scan_s.mean()),gate_compute_batch_mean_s=float(group.gate_compute_s.mean()),
            raw_rows=info["raw_rows"],raw_file_bytes=info["raw_file_bytes"],
            answer_source="fine histogram F" if method=="fine_direct" else "raw exact" if method=="exact" else "original H")
        if method!="exact":
            cost=costs[costs.method.eq(method)&costs.seed.eq(seed)].iloc[0]
            result.update({k:cost[k] for k in costs.columns if k.endswith("bytes")})
        else:result.update(total_budget_bytes=int(costs.total_budget_bytes.iloc[0]),used_bytes=0,unused_bytes=int(costs.total_budget_bytes.iloc[0]))
        summaries.append(result)
        frame=frame.copy();frame["truth"]=truth;frame["dataset"]=info["dataset"];frame["seed"]=seed;frame["method"]=method
        frame["candidate_error"]=relative_error(frame.candidate,truth);frame["returned_error"]=relative_error(frame.answer,truth)
        quality.append(frame)
        for name,mask in [("truth_ge_10",truth>=10),("truth_lt_10",truth<10),("truth_zero",truth==0)]:
            if mask.any():slices.append(dict(dataset=info["dataset"],seed=seed,method=method,cohort=name,**summarize(frame.loc[mask],truth[mask],c["epsilon"])))
    summary=pd.DataFrame(summaries).sort_values(["method","seed"],na_position="last")
    summary.to_csv(folder/"summary.csv",index=False)
    pd.DataFrame(slices).to_csv(folder/"evaluation_slices.csv",index=False)
    pd.concat(quality,ignore_index=True).to_csv(folder/"per_query.csv",index=False)
    _json(folder/"result_manifest.json",dict(status="complete",protocol_sha256=info["protocol_sha256"],
        actual_east_query_count=len(truth),raw_rows=info["raw_rows"],original_H_unchanged=True,
        no_test_label_tuning=True,exact_truth_parity=True,completed_utc=_now()))
    return summary


def _report(root,summary,*,matched=False):
    contrast=[]
    for dataset,rows in summary.groupby("dataset",sort=False):
        exact=rows[rows.method.eq("exact")].iloc[0]
        for method,part in rows.groupby("method",sort=False):
            contrast.append(dict(dataset=dataset,method=method,raw_rows=int(exact.raw_rows),raw_file_bytes=int(exact.raw_file_bytes),
                test_queries=int(exact.queries),exact_batch_median_s=exact.exact_batch_median_s,
                mean_accepted_count=part.accepted_count.mean(),mean_accepted_bad_count=part.accepted_bad_count.mean(),
                mean_coverage=part.coverage.mean(),mean_violation=part.violation.mean(),mean_fallback_count=part.fallback_count.mean(),
                mean_method_batch_median_s=part.batch_median_s.mean(),mean_seconds_saved=part.seconds_saved_vs_exact.mean(),
                mean_raw_load_median_s=part.raw_load_median_s.mean(),mean_payload_load_median_s=part.payload_load_median_s.mean(),
                mean_raw_scan_batch_s=part.raw_scan_batch_mean_s.mean(),
                min_actual_bytes=part.used_bytes.min(),max_actual_bytes=part.used_bytes.max()))
    pd.DataFrame(contrast).to_csv(root/"dataset_contrast.csv",index=False)
    tables=summary[["dataset","seed","method","coverage","accepted_count","accepted_bad_count","violation","fallback_count","batch_median_s","exact_batch_median_s","seconds_saved_vs_exact","used_bytes"]].copy()
    design = (
        "Calibration uses only western query labels. Each dataset's three methods share its prior Source Shape byte ceiling. Eastern queries exclude the previous test IDs. Both datasets' gates freeze before Java scans the east for truth.\n\n"
        if matched else
        "Training and calibration use only whole-rectangle western queries. Eastern truth is generated by real Java exact scans after models, global budget, representations and gates are frozen. All rows share one total-byte ceiling.\n\n"
    )
    (root/"analysis_en.md").write_text(
        "# Held-out-region storage comparison\n\n" + design +
        "Crossing rectangles are excluded. Actual usage and unused bytes are reported. Fine direct returns F and uses a count-derived uncertainty score. Fine reference retains H and uses F only to score H.\n\n"
        "The primary cohort contains all frozen test IDs; truth>=10 is a post-freeze evaluation slice only. Zero accepted answers have undefined risk. Source-calibrated5% risk may not transfer east: do not retune on these test labels.\n\n"
        "Batch time is the median of Java startup, all method-required payload loads, raw-data loading, actual fallback scans and CSV-to-Parquet conversion. Raw points are loaded once per JVM; OS cache is uncontrolled. Offline model training, summary construction and calibration are reported separately. Repeated seeds share the eastern query geometry and are not independent datasets.\n\n"
        +tables.to_csv(index=False)+"\n\n"
        "Use dataset_contrast.csv with phase_timing.csv, raw point counts and fallback counts to inspect differences between AUTHORS and Europe. These observations do not isolate a causal effect of data volume or density. Do not combine these strict source-calibrated results with the old target-calibrated region test or prior09b timing cells.\n")
    css="body{font:15px system-ui;color:#17243f;margin:24px;background:#f7f7f4}table{border-collapse:collapse;background:white}td,th{border:1px solid #cdd5df;padding:7px;white-space:nowrap}th{background:#e5ebf4}.scroll{overflow:auto}p{max-width:1000px;line-height:1.6}"
    (root/"report.html").write_text('<!doctype html><meta charset="utf-8"><title>Region budget comparison</title><style>'+css+'</style><h1>Held-out-region comparison</h1><p>'+design.strip()+' Zero-acceptance risk is undefined.</p><div class="scroll">'+tables.to_html(index=False,na_rep='—',float_format=lambda v:f'{v:.4f}')+'</div><h2>Dataset contrast</h2><div class="scroll">'+pd.DataFrame(contrast).to_html(index=False,na_rep='—',float_format=lambda v:f'{v:.4f}')+'</div>')


def _archive(root,c):
    # Build locally, then sequentially copy onto a Databricks Volume. ZIP needs
    # seeks, which are not supported reliably by Volume/FUSE-backed writers.
    descriptor,name=tempfile.mkstemp(suffix=".zip");os.close(descriptor);tmp=Path(name)
    try:
        with ZipFile(tmp,"w",ZIP_DEFLATED) as z:
            for p in sorted(root.rglob("*")):
                rel=p.relative_to(root)
                if (p.is_file() and len(rel.parts)<=2 and p.suffix in {".csv",".json",".md",".html",".log"}):z.write(p,str(rel))
            for rel in ["ml/region_budget_comparison.py","ml/region_budget_histogram.py","SynopsisLake/src/main/java/sigspatial25/RegionBudgetQueryExp.java","SynopsisLake/src/main/java/predictor/CountHistogramPayload.java"]:
                p=Path(c["repo_root"])/rel
                if p.is_file():z.write(p,"implementation/"+rel)
            if c["prior_results_zip"]:
                for rel in ("ml/region_budget_matched_inputs.py", "ml/region_budget_matched_report.py"):
                    z.write(Path(c["repo_root"])/rel, "implementation/"+rel)
        dest=root/"region_budget_results.zip"
        with tmp.open("rb") as src,dest.open("wb") as out:shutil.copyfileobj(src,out)
    finally:tmp.unlink(missing_ok=True)


def run_suite(config):
    c=_config(config)
    # Fail on missing datasets before starting an expensive partial suite.
    inputs=[_dataset_inputs(item,c) for item in c["datasets"]]
    matched = bool(c["prior_results_zip"])
    if matched:
        prior_metadata = _matched_prior(c, inputs)
        for item in inputs:
            item["matched_prior"] = prior_metadata[item["dataset"]]
    output=Path(c["output_root"]);output.mkdir(parents=True,exist_ok=True)
    root=output/("region-budget-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid4().hex[:8])
    root.mkdir();started=time.perf_counter()
    state=dict(status="preparing",config=c,started_utc=_now(),measured_datasets=[],
        test_labels_used_for_selection=False, model_export_mode=MODEL_EXPORT_MODE)
    if matched:
        state.update(protocol_version=MATCHED_REGION_PROTOCOL, prior_archive_sha256=next(iter(prior_metadata.values()))["archive_sha256"],
            selected_methods=c["methods"], prior_test_ids_excluded=True)
    _json(root/"manifest.json",state)
    print("Run directory:",root,flush=True)
    try:
        classpath,build=_build_java(c,root);state["java_build"]=build
        state["java_version"]=subprocess.run([c["java"],"-version"],capture_output=True,text=True).stderr
        state["versions"]={p:importlib.metadata.version(p) for p in ("numpy","pandas","scipy","lightgbm","pyarrow","scikit-learn")}
        state["environment"]=dict(platform=platform.platform(),python=platform.python_version(),logical_cpus=os.cpu_count(),
            databricks_runtime=os.environ.get("DATABRICKS_RUNTIME_VERSION"),cache="raw loaded once per JVM; OS cache uncontrolled")
        state["implementation_sha256"]={name:_sha(Path(__file__).with_name(name)) for name in (
            "region_budget_comparison.py","region_budget_histogram.py","shape_e2e_inputs.py","shape_team_model.py","gate_eval.py")}
        if matched:
            for name in ("region_budget_matched_inputs.py", "region_budget_matched_report.py"):
                state["implementation_sha256"][name] = _sha(Path(__file__).with_name(name))
        info=[_prepare_dataset(c,item,root,classpath) for item in inputs]
        if matched:
            budgets = c["dataset_budget_bytes"]
            state.update(dataset_budget_bytes=budgets, budget_rule="prior Source Shape actual bytes per dataset")
        else:
            floor=max(v["budget_floor"] for v in info)
            budget=c["total_budget_bytes"] or int(math.ceil(floor/1024)*1024)
            budgets={item["dataset"]:budget for item in info}
            state.update(total_budget_bytes=budget,budget_floor=floor,
                budget_rule="explicit preset or1024-byte-rounded max of source-trained model/representation minimums across requested datasets")
        for item in info:
            cap = budgets[item["dataset"]]
            if cap < item["budget_floor"]:
                raise ValueError(f"{item['dataset']}: byte cap {cap} cannot fit the selected methods' minimum {item['budget_floor']}")
        state["status"] = "budget_frozen_before_east_labels"
        _json(root/"manifest.json",state)
        methods=c["methods"]
        for item in info:_freeze_dataset(c,item,budgets[item["dataset"]],classpath,methods)
        state["status"]="all_dataset_gates_frozen_before_east_labels";_json(root/"manifest.json",state)
        result=[]
        for item in info:
            result.append(_evaluate_dataset(c,item,classpath,methods));state["measured_datasets"].append(item["dataset"])
            _json(root/"manifest.json",state)
        summary=pd.concat(result,ignore_index=True);summary.to_csv(root/"summary.csv",index=False)
        for name in ("payload_costs.csv","payload_files.csv","timing.csv","phase_timing.csv","region_splits.csv","calibration.csv","frozen_gates.csv","training_costs.csv","offline_costs.csv","calibration_scores.csv"):
            if name == "training_costs.csv" and "v1" not in methods:
                continue
            pieces=[]
            for item in info:
                f=pd.read_csv(item["folder"]/name)
                if "dataset" not in f:f.insert(0,"dataset",item["dataset"])
                pieces.append(f)
            pd.concat(pieces,ignore_index=True).to_csv(root/name,index=False)
        _report(root,summary,matched=matched)
        if matched:
            from .region_budget_matched_report import write_report

            write_report(root, summary, c["delta"])
        state.update(status="complete",finished_utc=_now(),suite_seconds=time.perf_counter()-started,
            no_target_label_tuning=True,primary_cohort="all geometry-frozen whole-east test IDs",exact_truth_source="real raw scans after freeze")
        _json(root/"manifest.json",state)
        try:
            _archive(root,c)
        except OSError as error:
            # Measured CSVs remain valid if a Volume refuses the final export.
            state["export_warning"]=str(error);_json(root/"manifest.json",state)
            print("Results complete; ZIP export warning:",error,flush=True)
        print("Completed:",root,flush=True)
        return root
    except BaseException as error:
        state.update(status="failed",error=str(error),failed_utc=_now())
        try:
            _json(root/"manifest.json",state)
            (root/"failure.txt").write_text(traceback.format_exc(), encoding="utf-8")
        except OSError as log_error:
            error.add_note(f"Could not save failure details in {root}: {log_error}")
        raise


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",required=True,help="JSON suite configuration; never inline private credentials")
    args=parser.parse_args(argv)
    run_suite(json.loads(Path(args.config).read_text()))


if __name__=="__main__":main()
