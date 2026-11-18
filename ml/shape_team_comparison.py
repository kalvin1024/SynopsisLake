"""Frozen common-workload comparison of the team's full v1 learner and R10.

This module keeps the existing H, source packet, target, learner, and three
outer seed memberships fixed. The only learned-input contrast is full v1
versus full v1 plus the same two features from R10 or its byte-matched grid.
All models and empirical calibration gates freeze before new geometry/labels.
The optional Python-request benchmark executes live scorers and real raw scans.
It is not a reproduction of an unavailable private Databricks model artifact.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import shutil
import time
import traceback
from uuid import uuid4
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
import pandas as pd

from . import local_shape_workflow as original
from . import shape_team_model as team
from . import sidecar_workflow as prior
from .sidecar_e2e import Arm, Query, ScanCounter, ScoreResult, empirical_cutoff, run_benchmark


GEOMETRY = original.GEOMETRY
LEARNED = ("ML_v1", "ML_v1_shape", "ML_v1_fine")
METHODS = ("SA", "native_R08", "R10", "fine_matched", *LEARNED)
DEFAULTS = dict(phase="develop", source_development_dir="", histogram_dir="", preparation_dir="",
    development_dir="", output_root="", seeds=[0, 1, 2], train_queries=0,
    calibration_queries=512, epsilon=.25, delta=.05, min_exact=10,
    fresh_query_count=2048, fresh_query_seed=2026111901,
    timing_queries=32, timing_repeats=3, backend="compiled", bootstrap_repeats=2000)


def _config(config):
    cfg = {**DEFAULTS, **dict(config)}
    cfg["seeds"] = prior._list(cfg["seeds"], int)
    if not cfg["seeds"] or len(set(cfg["seeds"])) != len(cfg["seeds"]):
        raise ValueError("Seeds must be nonempty and unique")
    for name in ("train_queries", "calibration_queries", "fresh_query_count", "fresh_query_seed",
                 "timing_queries", "timing_repeats", "bootstrap_repeats"):
        value = cfg[name]
        if isinstance(value, bool) or int(value) != value or int(value) < 0:
            raise ValueError(name+" must be a nonnegative integer")
        cfg[name] = int(value)
    if not cfg["timing_repeats"] or not cfg["bootstrap_repeats"]:
        raise ValueError("Timing and bootstrap repetitions must be positive")
    if cfg["epsilon"] <= 0 or not 0 <= cfg["delta"] < 1 or cfg["min_exact"] != 10:
        raise ValueError("Invalid error/risk/cohort settings")
    if cfg["backend"] not in ("compiled", "indexed"):
        raise ValueError("Backend must explicitly be compiled or indexed")
    cfg["output_root"] = str(Path(cfg["output_root"] or Path.cwd().parent/"synopsislake-results").expanduser().resolve())
    return cfg


def _new(root, prefix):
    path = Path(root)/(prefix+"-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def _implementation():
    names = ("shape_team_comparison.py", "shape_team_model.py", "features.py", "features_v1.py", "gate_eval.py",
             "local_shape_workflow.py", "sidecar_workflow.py", "sidecar_e2e.py", "hierarchical_shape.py",
             "local_shape.py", "bounded_sidecar.py", "_distribution_math.py", "shape_runtime.py",
             "shape_kernel.py", "shape_kernel.cc", "shape_execution.py")
    return {name: prior._sha(Path(__file__).with_name(name)) for name in names}


def _state(path, state):
    prior._json(Path(path)/"run.json", state)


def _fail(path, state, error):
    state.update(status="failed", error=f"{type(error).__name__}: {error}")
    _state(path, state)
    (Path(path)/"failure.txt").write_text(traceback.format_exc())


def _shape_loader(path, packet=True):
    path = Path(path)
    state = json.loads((path/"run.json").read_text())
    packet_path = path/"shape.bin" if packet else None
    if state["config"]["backend"] == "compiled":
        from .shape_kernel import load_compiled
        return load_compiled(path/"native", packet_path)
    from .shape_runtime import load_indexed
    return load_indexed(path/"native", packet_path)


def _source_inputs(cfg):
    source = Path(cfg["source_development_dir"]).expanduser().resolve()
    protocol = original.verify_protocol_lock(source)
    state = json.loads((source/"run.json").read_text())
    if state.get("status") != "complete" or state["config"]["family"] != "hierarchical":
        raise ValueError("The source must be a completed frozen hierarchical development")
    if cfg["seeds"] != state["config"]["seeds"]:
        raise ValueError("Comparison seeds must match inherited source memberships")
    preparation = Path(cfg["preparation_dir"] or state["preparation_dir"]).expanduser().resolve()
    original._load_preparation(preparation)
    if prior._sha(preparation/"run.json") != protocol["source_preparation_manifest_sha256"]:
        raise ValueError("Inherited preparation differs from the source development")
    budget = str(state["candidate_extra_budget"])
    if int(budget) != 49152:
        raise ValueError("This prespecified comparison uses the 49,152-byte compact cap")
    if cfg["epsilon"] != state["config"]["epsilon"] or cfg["min_exact"] != state["config"]["min_exact"]:
        raise ValueError("Inherited source and comparison use different error/cohort definitions")
    return source, preparation, state


def _copy_inputs(cfg, destination):
    source, preparation, inherited = _source_inputs(cfg)
    histogram_dir = Path(cfg["histogram_dir"]).expanduser().resolve()
    base = prior.load_histogram(preparation/"original_H.hist")
    context = team.TeamV1Runtime(histogram_dir, merged_histogram=base)
    _check_source_identity(histogram_dir, preparation, source, destination)
    native_bytes = sum(file.stat().st_size for file in (source/"native").iterdir())
    packet = inherited["packets"][str(inherited["candidate_extra_budget"])]
    shutil.copytree(source/"native", destination/"native")
    shutil.copyfile(source/"packets"/Path(packet["path"]).name, destination/"shape.bin")
    shutil.copyfile(source/"original_H.hist", destination/"original_H.hist")
    fine = inherited["fine_controls"][str(packet["auxiliary_bytes"])]
    shutil.copyfile(source/"fine"/Path(fine["path"]).name, destination/"fine.counts")
    shutil.copyfile(preparation/"outer_splits.csv", destination/"outer_splits.csv")
    shutil.copyfile(preparation/"source_audit.csv", destination/"source_audit.csv")
    target = destination/"team_histograms"
    target.mkdir()
    for file in context.input_files:
        if Path(file).name.startswith("src_"):
            shutil.copyfile(file, target/Path(file).name)
    original._load_preparation(preparation)
    source_bytes = sum(file.stat().st_size for file in target.iterdir())
    alpha_path = destination/"sa_alpha.bin"
    np.asarray([context.alpha], dtype="<f8").tofile(alpha_path)
    return dict(source_development=str(source), source_protocol_sha256=prior._sha(source/"protocol.json"),
        preparation_dir=str(preparation), preparation_manifest_sha256=prior._sha(preparation/"run.json"),
        native_bytes=native_bytes, packet_bytes=(destination/"shape.bin").stat().st_size,
        shape_auxiliary_bytes=native_bytes+(destination/"shape.bin").stat().st_size,
        fine_bytes=(destination/"fine.counts").stat().st_size,
        team_source_text_bytes=source_bytes, common_H_bytes=(destination/"original_H.hist").stat().st_size,
        source_histogram_encoding="unchanged Java text exports; charges actual bytes, not a codec-optimal storage claim",
        budget_scope="fine grid matches analytic native+packet bytes; augmented model totals are separately charged, not forced equal")


def _check_source_identity(histogram_dir, preparation, source, destination):
    """Bind full-v1 source features to the very same frozen R10 sources."""
    from .features import load_hist
    prepared = json.loads((Path(preparation)/"run.json").read_text())
    manifests = [Path(preparation)/"input_manifest.json"]
    if prepared.get("source_r09_run"):
        inherited = Path(prepared["source_r09_run"])
        if prior._sha(inherited/"run.json") != prepared["source_r09_manifest_sha256"]:
            raise ValueError("Inherited R09 source manifest identity changed")
        manifests.append(inherited/"input_manifest.json")
    expected = {}
    for manifest in manifests:
        if manifest.is_file():
            for item in json.loads(manifest.read_text()):
                file = Path(item["path"])
                if file.name.startswith("src_") and file.suffix == ".txt":
                    expected[file.name] = item["sha256"]
    texts = sorted(Path(histogram_dir).glob("src_*.txt"), key=lambda p:int(p.stem.split("_")[-1]))
    native = sorted((Path(source)/"native").glob("src_*.hist"), key=lambda p:int(p.stem.split("_")[-1]))
    if not texts or len(texts) != len(native):
        raise ValueError("Team-v1 and R10 source histogram counts differ")
    rows = []
    for text, binary in zip(texts, native):
        left, right = load_hist(text), prior.load_histogram(binary)
        matches = all(np.array_equal(a, b) for a, b in zip(left, right))
        digest = prior._sha(text)
        known = expected.get(text.name)
        rows.append(dict(text_source=str(text), native_source=str(binary), semantic_identity=matches,
                         text_sha256=digest, native_sha256=prior._sha(binary),
                         inherited_text_sha256=known, inherited_hash_matches=known is None or digest == known))
        if not matches or (known is not None and digest != known):
            raise ValueError("Team-v1 source identity differs from frozen R10: "+text.name)
    pd.DataFrame(rows).to_csv(Path(destination)/"source_identity_audit.csv", index=False)


def _geometry_features(path, geometry):
    """Offline geometry-only batch; it cannot inspect a label or qId lookup."""
    if set(geometry) != {"qId", *GEOMETRY} or geometry.qId.duplicated().any():
        raise ValueError("Feature construction requires unique IDs and geometry only, without truth")
    path = Path(path)
    base = prior.load_histogram(path/"original_H.hist")
    context = team.TeamV1Runtime(path/"team_histograms", merged_histogram=base)
    rectangles = geometry[GEOMETRY].to_numpy(float)
    features = context.batch_features(rectangles)
    answers = np.asarray([prior.histogram_count(q, base) for q in rectangles])
    shape = _shape_loader(path)
    native = _shape_loader(path, packet=False)
    refs = dict(R10=shape.batch_reference(rectangles), native_R08=native.batch_reference(rectangles))
    fine = prior._load_fine(path/"fine.counts")
    refs["fine_matched"] = np.asarray([prior.histogram_count(q, fine) for q in rectangles])
    for name, values in refs.items():
        if values.shape != (len(geometry),) or not np.isfinite(values).all() or (values < 0).any():
            raise ValueError("Invalid source-only reference counts: "+name)
    return features, answers, refs


def _augment(features, answers, references):
    frame = features.copy()
    frame[team.AUX_FEATURES[0]] = np.log1p(references)
    frame[team.AUX_FEATURES[1]] = np.abs(answers-references)/np.maximum(references, 1.)
    return frame


def _score_geometry(path, geometry, seed, described=None):
    """Persisted pre-label scores; live benchmark scorers do not use this table."""
    features, answers, refs = described if described is not None else _geometry_features(path, geometry)
    rows = []
    scores = dict(SA=features.skew_pred.to_numpy(float))
    for name in ("native_R08", "R10", "fine_matched"):
        scores[name] = np.abs(answers-refs[name])/np.maximum(refs[name], 1.)
    raw_log = {}
    for method, ref in (("ML_v1", None), ("ML_v1_shape", "R10"), ("ML_v1_fine", "fine_matched")):
        predictor = team.load_team_model(Path(path)/"models"/f"seed_{seed}"/method)
        inputs = features if ref is None else _augment(features, answers, refs[ref])
        raw_log[method] = predictor.log_scores(inputs)
        scores[method] = np.exp(raw_log[method])
    for method in METHODS:
        reference = refs.get({"ML_v1_shape": "R10", "ML_v1_fine": "fine_matched"}.get(method, method))
        if not np.isfinite(scores[method]).all():
            raise ValueError("Nonfinite assessor scores: "+method)
        row = geometry.copy()
        row["seed"], row["method"], row["H"] = int(seed), method, answers
        row["score"] = scores[method]
        row["predicted_error"] = np.nan if method == "SA" else scores[method]
        row["raw_log_score"] = raw_log.get(method, np.full(len(row), np.nan))
        row["reference_count"] = np.nan if reference is None else reference
        rows.append(row)
    return pd.concat(rows, ignore_index=True)


def _assert_outer(outer, queries, seeds):
    for seed in seeds:
        selected = outer[outer.seed == seed]
        if selected.qId.duplicated().any() or set(selected.qId) != set(queries.qId):
            raise ValueError("Every original query must have exactly one inherited outer role per seed")
        for role in ("train", "calibration", "test"):
            if not (selected.split == role).any():
                raise ValueError("Empty inherited outer role: "+role)
        if "cell" in selected:
            grouped = selected.groupby("cell").split.nunique()
            if (grouped > 1).any():
                raise ValueError("Source-cell groups cross outer memberships")


def develop_experiment(config):
    """Fit three fixed-capacity controls on inherited TRAIN; freeze all gates."""
    cfg = _config(config)
    path = _new(cfg["output_root"], "team-shape-development")
    state = dict(status="preparing", study="team-shape-common-comparison", config=cfg,
                 started_utc=datetime.now(timezone.utc).isoformat())
    _state(path, state)
    started = time.perf_counter()
    implementation = _implementation()
    try:
        state.update(_copy_inputs(cfg, path))
        _state(path, state)
        from .shape_execution import _backend_setup
        state["executable_setup"] = _backend_setup(cfg["backend"])
        _state(path, state)
        preparation = Path(state["preparation_dir"])
        queries = pd.read_csv(preparation/"queries_offline.csv", float_precision="round_trip")
        outer = pd.read_csv(path/"outer_splits.csv")
        outer = outer[outer.seed.isin(cfg["seeds"])].copy()
        _assert_outer(outer, queries, cfg["seeds"])
        geometry = queries[["qId", *GEOMETRY]]
        feature_started = time.perf_counter()
        features, answers, refs = described = _geometry_features(path, geometry)
        state["offline_feature_seconds"] = time.perf_counter()-feature_started
        ids = queries.qId.to_numpy()
        targets = prior._actual_error(answers, queries.truth.to_numpy(float))
        cache = path/"feature_cache"
        cache.mkdir()
        features.assign(qId=ids).to_csv(cache/"original_query_features.csv", index=False)
        indexed = queries.set_index("qId")
        position = pd.Series(np.arange(len(queries)), index=queries.qId)
        training, assignments, calibrations, gates, costs = [], [], [], [], []
        for seed in cfg["seeds"]:
            roles = outer[outer.seed == seed]
            train_ids = prior._select(roles.loc[roles.split == "train", "qId"], cfg["train_queries"], seed, "team-full-v1-train")
            cal_ids = prior._select(roles.loc[roles.split == "calibration", "qId"], cfg["calibration_queries"], seed, "r10-calibration")
            if set(train_ids)&set(cal_ids):
                raise ValueError("TRAIN and calibration query IDs overlap")
            loc = position.loc[train_ids].to_numpy()
            held = set(sorted(train_ids)[::10])
            assignments.extend(dict(seed=seed, qId=qid, role="inner_validation" if qid in held else "fit") for qid in train_ids)
            assignments.extend(dict(seed=seed, qId=qid, role="calibration") for qid in cal_ids)
            for method, reference in (("ML_v1", None), ("ML_v1_shape", "R10"), ("ML_v1_fine", "fine_matched")):
                train_features = features.iloc[loc].reset_index(drop=True)
                if reference is not None:
                    train_features = _augment(train_features, answers[loc], refs[reference][loc])
                table = train_features.assign(qId=np.asarray(train_ids, np.int64), actual_error=targets[loc])
                predictor = team.fit_team_model(table, seed, path/"models"/f"seed_{seed}"/method,
                                                augmented=reference is not None, model_name=method)
                training.append(dict(predictor.training_metadata))
                print(f"Team comparison seed {seed}: trained {method} on {len(table):,} TRAIN queries", flush=True)
            cal_loc = position.loc[cal_ids].to_numpy()
            cal_geometry = indexed.loc[cal_ids].reset_index()[["qId", *GEOMETRY]]
            cal_described = (features.iloc[cal_loc].reset_index(drop=True), answers[cal_loc],
                             {name: values[cal_loc] for name, values in refs.items()})
            cal_started = time.perf_counter()
            scored = _score_geometry(path, cal_geometry, seed, cal_described)
            scored = attach_truth(scored, indexed.loc[cal_ids].reset_index())
            calibrations.append(scored)
            for method, part in scored.groupby("method", sort=False):
                gates.append(dict(seed=seed, method=method, cutoff=empirical_cutoff(part.score.to_numpy(),
                    part.actual_error.to_numpy(), cfg["epsilon"], cfg["delta"]), eps=cfg["epsilon"],
                    delta=cfg["delta"], calibration_queries=len(part)))
            costs.extend(_payload_costs(path, seed))
            state.setdefault("calibration_seconds", {})[str(seed)] = time.perf_counter()-cal_started
        pd.DataFrame(assignments).to_csv(path/"selected_query_ids.csv", index=False)
        pd.DataFrame(gates).to_csv(path/"frozen_gate_cutoffs.csv", index=False)
        pd.concat(calibrations, ignore_index=True).to_csv(path/"calibration_predictions.csv", index=False)
        pd.DataFrame(costs).to_csv(path/"payload_costs.csv", index=False)
        prior._json(path/"training_metadata.json", training)
        original._load_preparation(preparation)
        original.verify_protocol_lock(state["source_development"])
        if implementation != _implementation():
            raise ValueError("Implementation changed during common-model development")
        files = {str(file.relative_to(path)): prior._sha(file) for file in sorted(path.rglob("*"))
                 if file.is_file() and file.name != "run.json" and "feature_cache" not in file.parts}
        protocol = dict(status="models_gates_and_representation_frozen_before_new_geometry", config=cfg,
            implementation_sha256=implementation, files_sha256=files,
            preparation_manifest_sha256=state["preparation_manifest_sha256"],
            source_protocol_sha256=state["source_protocol_sha256"], methods=list(METHODS),
            comparison="same full-v1 learner/target/outer roles; only two reference features added",
            target="log(max(actual relative error, team LABEL_FLOOR=0.001)); exp before gates/MAE",
            label_policy="new query geometry and scores precede raw labels; no test-driven candidate selection",
            selected_packet="pre-existing fixed compact source-TV packet; no reallocation or test retuning")
        prior._json(path/"protocol.json", protocol)
        state.update(status="complete", protocol_sha256=prior._sha(path/"protocol.json"),
                     development_seconds=time.perf_counter()-started, implementation_sha256=implementation)
        _state(path, state)
        (path/"analysis_en.md").write_text("# Common-model development\n\nThe team's full 28-feature learner and two matched 30-feature reference augmentations use identical inherited TRAIN, the original log-error target and unchanged model capacity. Original calibration alone freezes gates. No fresh geometry or labels were generated during this phase. Source/model/metadata bytes are recorded separately; the fine reference matches the analytic source-sidecar budget, not every trained model's different total size. This is an aligned-family retraining, not the unavailable private Databricks model.\n")
        return path
    except BaseException as error:
        _fail(path, state, error)
        raise


def verify_development(path):
    path = Path(path).expanduser().resolve()
    state = json.loads((path/"run.json").read_text())
    if state.get("status") != "complete" or prior._sha(path/"protocol.json") != state["protocol_sha256"]:
        raise ValueError("Incomplete or changed common-comparison development")
    protocol = json.loads((path/"protocol.json").read_text())
    if protocol["implementation_sha256"] != _implementation():
        raise ValueError("Frozen common-comparison implementation changed")
    for relative, digest in protocol["files_sha256"].items():
        if prior._sha(path/relative) != digest:
            raise ValueError("Frozen development artifact changed: "+relative)
    preparation, _ = original._load_preparation(state["preparation_dir"])
    if prior._sha(preparation/"run.json") != protocol["preparation_manifest_sha256"]:
        raise ValueError("Frozen original preparation changed")
    original.verify_protocol_lock(state["source_development"])
    if prior._sha(Path(state["source_development"])/"protocol.json") != protocol["source_protocol_sha256"]:
        raise ValueError("Frozen original source development changed")
    return state, protocol


def _payload_costs(path, seed):
    path = Path(path)
    state = json.loads((path/"run.json").read_text())
    common = state["common_H_bytes"]
    for method in METHODS:
        model_bytes = source_text = native = packet = fine = alpha = 0
        files = [path/"original_H.hist"]
        if method == "SA":
            alpha = 8
            files.append(path/"sa_alpha.bin")
        elif method in ("native_R08", "R10", "ML_v1_shape"):
            native = state["native_bytes"]
            files.extend((path/"native").iterdir())
            if method != "native_R08":
                packet = state["packet_bytes"]
                files.append(path/"shape.bin")
        if method in ("fine_matched", "ML_v1_fine"):
            fine = state["fine_bytes"]
            files.append(path/"fine.counts")
        if method in LEARNED:
            model_dir = path/"models"/f"seed_{seed}"/method
            files.extend(model_dir/name for name in team.MODEL_FILES)
            model_bytes = sum((model_dir/name).stat().st_size for name in team.MODEL_FILES)
            files.extend((path/"team_histograms").iterdir())
            source_text = state["team_source_text_bytes"]
        yield dict(seed=seed, method=method, H_bytes=common, source_text_bytes=source_text,
            native_bytes=native, packet_bytes=packet, fine_bytes=fine, alpha_bytes=alpha,
            model_schema_weights_properties_bytes=model_bytes,
            auxiliary_bytes=source_text+native+packet+fine+alpha+model_bytes,
            total_serialized_bytes=sum(file.stat().st_size for file in dict.fromkeys(files)),
            file_paths="|".join(str(file.relative_to(path)) for file in dict.fromkeys(files)),
            accounting="actual required files; no free model/source metadata; text-source encoding is not codec optimal")


def load_arms(path, seed):
    """Live request callbacks: bounds/H/artifacts only; no IDs, scores or truth."""
    path = Path(path)
    original_h = path/"original_H.hist"
    costs = {row["method"]: row for row in _payload_costs(path, seed)}
    def native_score(rect, answer, runtime):
        reference = runtime.reference_count(rect)
        return ScoreResult(prior._proxy(answer, reference), reference)
    def fine_score(rect, answer, hist):
        reference = prior.histogram_count(rect, hist)
        return ScoreResult(prior._proxy(answer, reference), reference)
    arms = [Arm("SA", lambda rect, answer, alpha, histogram: prior._sa_score(rect, histogram, alpha),
                artifact_loader=lambda: float(np.fromfile(path/"sa_alpha.bin", dtype="<f8")[0]),
                metadata=costs["SA"], score_uses_histogram=True),
            Arm("native_R08", native_score, artifact_loader=lambda: _shape_loader(path, False), metadata=costs["native_R08"]),
            Arm("R10", native_score, artifact_loader=lambda: _shape_loader(path), metadata=costs["R10"]),
            Arm("fine_matched", fine_score, artifact_loader=lambda: prior._load_fine(path/"fine.counts"), metadata=costs["fine_matched"])]
    for method in LEARNED:
        def loader(method=method):
            context = team.TeamV1Runtime(path/"team_histograms", merged_path=original_h)
            predictor = team.load_team_model(path/"models"/f"seed_{seed}"/method)
            reference = (_shape_loader(path) if method == "ML_v1_shape" else
                         prior._load_fine(path/"fine.counts") if method == "ML_v1_fine" else None)
            return context, predictor, reference
        def score(rect, answer, artifact, method=method):
            context, predictor, reference = artifact
            if hasattr(context, "clear_query_cache"):
                context.clear_query_cache()
            value = (reference.reference_count(rect) if method == "ML_v1_shape" else
                     prior.histogram_count(rect, reference) if method == "ML_v1_fine" else None)
            described = context.describe(rect, reference=value)
            _, predicted_error = predictor.predict_one(described)
            return ScoreResult(predicted_error, value)
        arms.append(Arm(method, score, artifact_loader=loader, metadata=costs[method]))
    return arms


def attach_truth(scores, labels):
    if "truth" in scores or scores.duplicated(["seed", "method", "qId"]).any():
        raise ValueError("Pre-label scores must have unique keys and no truth")
    if labels.qId.duplicated().any():
        raise ValueError("Offline labels contain duplicate query IDs")
    result = scores.merge(labels[["qId", "truth"]], on="qId", how="left", validate="many_to_one")
    if result.truth.isna().any():
        raise ValueError("Missing offline truth for persisted geometry")
    result["actual_error"] = prior._actual_error(result.H, result.truth)
    return result


def quality_tables(quality, gates, cfg):
    rows, fixed = [], []
    for (seed, method), part in quality.groupby(["seed", "method"], sort=False):
        selected_gate = gates[(gates.seed == seed)&(gates.method == method)]
        if len(selected_gate) != 1:
            raise ValueError("Exactly one frozen gate is required per seed/method")
        cutoff = float(selected_gate.cutoff.iloc[0])
        cohorts = (("all", part), ("truth_ge_10", part[part.truth >= cfg["min_exact"]]),
                   ("positive_truth_lt_10", part[(part.truth > 0)&(part.truth < cfg["min_exact"])]),
                   ("zero_truth", part[part.truth == 0]))
        for cohort, selected in cohorts:
            score, error = selected.score.to_numpy(float), selected.actual_error.to_numpy(float)
            bad = error > cfg["epsilon"]
            accepted = np.isfinite(score)&(score <= cutoff)
            count, answered = len(selected), int(accepted.sum())
            wrong = int((accepted&bad).sum())
            estimate = selected.predicted_error.to_numpy(float)
            mae = np.nan if method == "SA" or not count else float(np.mean(abs(estimate-error))*100)
            rows.append(dict(seed=seed, method=method, cohort=cohort, queries=count,
                error_prediction_mae_pp=mae, bad_answer_auc=prior._auc(bad, score) if count else np.nan,
                accepted_count=answered, accepted_bad_count=wrong, accepted_good_count=answered-wrong,
                acceptance=answered/count if count else np.nan, accepted_violation=wrong/answered if answered else np.nan,
                good_answer_recall=(answered-wrong)/int((~bad).sum()) if (~bad).any() else np.nan,
                cutoff=cutoff, nonfinite_actual_errors=int((~np.isfinite(error)).sum())))
            if not count:
                continue
            for coverage in (.1, .25, .5, 1.):
                rank = max(1, math.ceil(count*coverage))
                threshold = np.sort(score)[rank-1]
                keep = np.isfinite(score)&(score <= threshold)
                n = int(keep.sum())
                b = int((keep&bad).sum())
                fixed.append(dict(seed=seed, method=method, cohort=cohort, nominal_coverage=coverage,
                    selected_count=n, bad_count=b, good_count=n-b, actual_coverage=n/count,
                    extra_tied_queries=n-rank, bad_fraction=b/n if n else np.nan))
    return pd.DataFrame(rows), pd.DataFrame(fixed)


def paired_intervals(quality, histogram, repetitions=2000):
    """Conditional fixed-grid block bootstrap; seed appearances first average."""
    rows = []
    root = np.array([histogram[1][0], histogram[2][0]])
    width = np.array([histogram[1][-1], histogram[2][-1]])-root
    finite = quality[(quality.truth >= 10)&np.isfinite(quality.actual_error)].copy()
    finite["loss"] = 100*abs(finite.predicted_error-finite.actual_error)
    for first, second in (("ML_v1_shape", "ML_v1"), ("ML_v1_shape", "ML_v1_fine"), ("R10", "fine_matched")):
        a = finite[finite.method == first][["seed", "qId", "loss", *GEOMETRY]]
        b = finite[finite.method == second][["seed", "qId", "loss"]]
        paired = a.merge(b, on=["seed", "qId"], validate="one_to_one", suffixes=("_a", "_b"))
        paired["delta"] = paired.loss_a-paired.loss_b
        one = paired.groupby("qId").agg(delta=("delta", "mean"), **{name: (name, "first") for name in GEOMETRY})
        if not len(one):
            continue
        xy = np.column_stack(((one.x0+one.x1)/2, (one.y0+one.y1)/2))
        cell = np.clip(((xy-root)/width*16).astype(int), 0, 15)
        one["block"] = cell[:, 0]*16+cell[:, 1]
        grouped = one.groupby("block").delta.agg(["sum", "count"]).to_numpy()
        draws = np.random.default_rng(20261118).integers(0, len(grouped), (int(repetitions), len(grouped)))
        bootstrap = grouped[draws, 0].sum(axis=1)/grouped[draws, 1].sum(axis=1)
        rows.append(dict(first=first, second=second, unique_queries=len(one), blocks=len(grouped),
            delta_mae_pp=float(one.delta.mean()), ci_low=float(np.quantile(bootstrap, .025)),
            ci_high=float(np.quantile(bootstrap, .975)), bootstrap_repeats=int(repetitions),
            scope="conditional descriptive 16x16 midpoint-block bootstrap; seed appearances are not independent samples"))
    return pd.DataFrame(rows)


def verify_confirmation(path):
    path = Path(path)
    state = json.loads((path/"run.json").read_text())
    if prior._sha(path/"protocol.json") != state["protocol_sha256"]:
        raise ValueError("Frozen confirmation protocol changed")
    protocol = json.loads((path/"protocol.json").read_text())
    parent, _ = verify_development(state["development_dir"])
    if (state["development_protocol_sha256"] != parent["protocol_sha256"] or
            protocol["development_protocol_sha256"] != parent["protocol_sha256"]):
        raise ValueError("Confirmation refers to a different frozen development")
    if protocol["implementation_sha256"] != _implementation():
        raise ValueError("Confirmation implementation changed")
    if prior._sha(path/"frozen_gate_cutoffs.csv") != protocol["gates_sha256"]:
        raise ValueError("Confirmation gates changed")
    lock_path = path/"score_lock.json"
    if lock_path.exists():
        lock = json.loads(lock_path.read_text())
        if lock["protocol_sha256"] != state["protocol_sha256"]:
            raise ValueError("Score lock refers to a different confirmation protocol")
        for filename, key in (("queries_geometry_only.csv", "geometry_sha256"),
                              ("scores_before_labels.csv", "scores_sha256")):
            if prior._sha(path/filename) != lock[key]:
                raise ValueError("Frozen score/geometry changed: "+filename)
    elif state.get("status") == "complete":
        raise ValueError("Completed confirmation needs a pre-label score lock")
    return state, protocol


def _timing(path, development, geometry, labeled, queries, seed, cfg, ids):
    gates = pd.read_csv(development/"frozen_gate_cutoffs.csv", float_precision="round_trip").rename(columns={"eps": "epsilon"})
    selected = pd.read_csv(development/"selected_query_ids.csv")
    cal_ids = selected.loc[(selected.seed == seed)&(selected.role == "calibration"), "qId"]
    calibration = queries.set_index("qId").loc[cal_ids].reset_index()
    timed = labeled.set_index("qId").loc[ids].reset_index()
    def records(frame):
        return [Query(int(row.qId), (row.x0, row.x1, row.y0, row.y1), int(row.truth)) for row in frame.itertuples(index=False)]
    state = json.loads((development/"run.json").read_text())
    tick = time.perf_counter()
    arrays = [np.array(points, copy=True) for points in original._raw_arrays(state["preparation_dir"])]
    counter = ScanCounter(arrays)
    counter.load_seconds = time.perf_counter()-tick-counter.initialization_seconds
    result = run_benchmark(queries=records(timed), calibration_queries=records(calibration),
        arms=load_arms(development, seed), histogram_loader=lambda: prior.load_histogram(development/"original_H.hist"),
        estimate_fn=prior.histogram_count, raw_counter=counter, output_dir=path/"e2e"/f"seed_{seed}",
        epsilon=cfg["epsilon"], delta=cfg["delta"], repeats=cfg["timing_repeats"], seed=seed,
        cache_modes=("resident", "per_query"),
        frozen_cutoffs=gates[gates.seed == seed].set_index("method").to_dict("index"),
        frozen_cutoff_provenance=dict(path=str(development/"frozen_gate_cutoffs.csv"),sha256=prior._sha(development/"frozen_gate_cutoffs.csv")),
        metadata=dict(backend="Python requests; explicit "+cfg["backend"]+" analytic references; original full-v1 model",
                      raw_access="every rejected/exact call scans every resident original point", labels_used_for_selection=False))
    from .shape_execution import verify_live_parity
    frozen = pd.read_csv(path/"scores_before_labels.csv", float_precision="round_trip")
    measured = pd.concat((result.per_query.assign(seed=seed), result.first_query.assign(seed=seed)), ignore_index=True)
    parity = verify_live_parity(measured, frozen, gates)
    parity.to_csv(path/f"live_parity_seed_{seed}.csv", index=False)
    if not parity.all_matches.all():
        raise ValueError("Live full-model outputs differ from frozen pre-label scores")
    return result.summary.assign(seed=seed)


def confirm_experiment(config):
    cfg = _config(config)
    development = Path(cfg["development_dir"]).expanduser().resolve()
    parent, parent_protocol = verify_development(development)
    if cfg["fresh_query_count"] < 1000:
        raise ValueError("New confirmation requires at least 1,000 fresh queries")
    for key in ("fresh_query_seed", "fresh_query_count", "timing_queries", "timing_repeats", "bootstrap_repeats"):
        if cfg[key] != parent["config"][key]:
            raise ValueError("Confirmation setting must be declared before development freeze: "+key)
    cfg = dict(parent["config"])
    path = _new(development/"confirmation", "team-shape-confirmation")
    state = dict(status="frozen_before_geometry", study="team-shape-fresh-confirmation", config=cfg,
                 development_dir=str(development), development_protocol_sha256=parent["protocol_sha256"])
    _state(path, state)
    try:
        preparation = Path(parent["preparation_dir"])
        queries = pd.read_csv(preparation/"queries_offline.csv", float_precision="round_trip")
        outer = pd.read_csv(development/"outer_splits.csv")
        donors = outer.loc[(outer.seed == cfg["seeds"][0])&(outer.split == "train"), "qId"].to_numpy()
        original_hist = prior.load_histogram(development/"original_H.hist")
        root = (original_hist[1][0], original_hist[1][-1], original_hist[2][0], original_hist[2][-1])
        donor_geometry = queries.set_index("qId").loc[donors, GEOMETRY].to_numpy(float)
        shutil.copyfile(development/"frozen_gate_cutoffs.csv", path/"frozen_gate_cutoffs.csv")
        protocol = dict(status="all_models_packets_gates_frozen_before_new_geometry", config=cfg,
            implementation_sha256=_implementation(), gates_sha256=prior._sha(path/"frozen_gate_cutoffs.csv"),
            development_protocol_sha256=parent["protocol_sha256"], donor_training_qIds=donors.tolist(),
            generator="original TRAIN width/height pairs uniformly shifted inside unchanged H root",
            timing_selection="fixed ID hashes before labels; repetitions are not independent samples",
            independent_dataset=False, class_or_geographic_generalization=False)
        prior._json(path/"protocol.json", protocol)
        state["protocol_sha256"] = prior._sha(path/"protocol.json")
        _state(path, state)
        rectangles = prior._fresh_rectangles(cfg["fresh_query_count"], root, cfg["fresh_query_seed"], donor_geometry)
        geometry = pd.DataFrame(rectangles, columns=GEOMETRY)
        geometry.insert(0, "qId", np.arange(len(geometry), dtype=np.int64)+int(queries.qId.max())+1)
        geometry.to_csv(path/"queries_geometry_only.csv", index=False)
        timing_ids = {seed: prior._select(geometry.qId, cfg["timing_queries"], seed, "team-shape-fresh-timing-v1")
                      for seed in cfg["seeds"]}
        pd.DataFrame([dict(seed=seed, qId=qid) for seed, ids in timing_ids.items() for qid in ids]).to_csv(path/"timing_ids_before_labels.csv", index=False)
        described = _geometry_features(development, geometry)
        scores = pd.concat([_score_geometry(development, geometry, seed, described) for seed in cfg["seeds"]], ignore_index=True)
        scores.to_csv(path/"scores_before_labels.csv", index=False)
        prior._json(path/"score_lock.json", dict(protocol_sha256=state["protocol_sha256"],
            geometry_sha256=prior._sha(path/"queries_geometry_only.csv"), scores_sha256=prior._sha(path/"scores_before_labels.csv")))
        verify_confirmation(path)
        arrays = original._raw_arrays(preparation)
        truth, label_seconds = prior._indexed_labels(arrays, rectangles)
        labeled = geometry.assign(truth=truth)
        labeled.to_csv(path/"query_labels_offline.csv", index=False)
        counter = ScanCounter(arrays)
        audits = []
        for row in labeled.iloc[[0, len(labeled)//2, len(labeled)-1]].itertuples(index=False):
            exact = counter.count((row.x0, row.x1, row.y0, row.y1))
            audits.append(dict(qId=int(row.qId), indexed_truth=int(row.truth), raw_scan_truth=int(exact), matches=exact == row.truth))
        pd.DataFrame(audits).to_csv(path/"fresh_raw_audit.csv", index=False)
        if not all(row["matches"] for row in audits):
            raise ValueError("Fresh exact labels disagree with independent raw scans")
        quality = attach_truth(scores, labeled)
        gates = pd.read_csv(path/"frozen_gate_cutoffs.csv", float_precision="round_trip")
        metrics, coverage = quality_tables(quality, gates, cfg)
        quality.to_csv(path/"per_query_quality.csv", index=False)
        metrics.to_csv(path/"metrics.csv", index=False)
        coverage.to_csv(path/"fixed_coverage.csv", index=False)
        paired_intervals(quality, original_hist, cfg["bootstrap_repeats"]).to_csv(path/"paired_mae_intervals.csv", index=False)
        timing = []
        if cfg["timing_queries"]:
            for seed in cfg["seeds"]:
                timing.append(_timing(path, development, geometry, labeled, queries, seed, cfg, timing_ids[seed]))
        if timing:
            pd.concat(timing, ignore_index=True).to_csv(path/"e2e_summary_by_seed.csv", index=False)
            pooled_timing(path).to_csv(path/"e2e_pooled.csv", index=False)
        verify_confirmation(path)
        state.update(status="complete", offline_raw_label_seconds=label_seconds,
                     raw_rows=sum(len(points) for points in arrays), fresh_unique_queries=len(geometry),
                     labels_used_for_model_or_gate_selection=False)
        _state(path, state)
        (path/"analysis_en.md").write_text("# Frozen common-workload confirmation\n\nEvery model, source packet, feature whitelist and calibration gate froze before new query geometry and raw labels. All predictions preceded exact labels; raw scans independently verify labels and serve every timed fallback. ML uses the team's full-v1 learner with its original log target, exponentiated before gates and prediction-MAE. All, small and zero truths remain visible; truth>=10 is reporting only. Three seeds reuse geometry and are not independent datasets. Source/model bytes and timing stages are separate. These are aligned-family retrainings, not a claim to beat an unavailable saved teammate model. Classical source-shape components do not establish global novelty or a five-percent guarantee.\n")
        package_results(path)
        return path
    except BaseException as error:
        _fail(path, state, error)
        raise


def pooled_timing(path):
    frames = []
    for file in sorted((Path(path)/"e2e").glob("seed_*/per_query.csv")):
        frames.append(pd.read_csv(file).assign(seed=int(file.parent.name.split("_")[-1])))
    if not frames:
        return pd.DataFrame()
    data = pd.concat(frames, ignore_index=True)
    rows = []
    for (mode, method), g in data.groupby(["cache_mode", "method"]):
        exact = data[(data.cache_mode == mode)&(data.method == "always_exact")]
        cases = g.drop_duplicates(["seed", "qId"])
        accepted = int(cases.accepted.sum())
        bad = int(cases.accepted_violation.sum())
        row = dict(cache_mode=mode, method=method, observations=len(g), unique_qIds=g.qId.nunique(),
            seed_query_cases=len(cases), accepted_count=accepted, accepted_bad_count=bad,
            timing_subset_risk=bad/accepted if accepted else np.nan,
            mean_ms=g.total_ns.mean()/1e6, p95_ms=g.total_ns.quantile(.95)/1e6,
            exact_mean_ms=exact.total_ns.mean()/1e6, speedup_ratio_of_pooled_means=exact.total_ns.mean()/g.total_ns.mean())
        for phase in ("histogram_load", "artifact_load", "estimate", "score", "gate", "raw"):
            row[phase+"_mean_ms"] = g[phase+"_ns"].mean()/1e6
        rows.append(row)
    return pd.DataFrame(rows)


def package_results(path):
    path = Path(path)
    archive = path/"team_shape_results.zip"
    with ZipFile(archive, "w", ZIP_DEFLATED) as zipped:
        for file in sorted(path.rglob("*")):
            if file.is_file() and file.suffix not in (".zip", ".so", ".pyc") and not set(file.relative_to(path).parts)&{"feature_cache", "raw_cache", "__pycache__"}:
                zipped.write(file, file.relative_to(path))
    return archive


def run_experiment(config):
    return confirm_experiment(config) if dict(config).get("phase") == "confirm" else develop_experiment(config)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path, help="JSON configuration; phase develop or confirm")
    args = parser.parse_args(argv)
    print(run_experiment(json.loads(args.config.read_text())))


if __name__ == "__main__":
    main()
