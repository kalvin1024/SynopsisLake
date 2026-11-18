"""Run existing Java query execution with separately frozen backend gates.

Java and Python source features can differ numerically. This supplement never
replaces Python gates or claims cross-backend model parity: it calibrates on
the same original calibration memberships using actual Java scores, then
freezes all Java assets before new geometry, scores and independent raw labels.
The only query engine invoked is the existing ``ApproxQueryExp`` CLI.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time
from uuid import uuid4

import numpy as np
import pandas as pd

from . import features
from . import shape_team_comparison as common
from . import sidecar_workflow as prior
from .sidecar_e2e import empirical_cutoff


METHODS = ("SA", "ML_v1", "R10", "ML_v1_shape")
GEOMETRY = common.GEOMETRY
MAIN_CLASS = "sigspatial25.ApproxQueryExp"


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024*1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _json(path, value):
    prior._json(Path(path), value)


def _new(root, prefix):
    path = Path(root)/(prefix+"-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def _config(config):
    cfg = dict(config)
    if not cfg.get("java") or not cfg.get("java_classpath"):
        raise ValueError("Provide an explicit Java executable and built Java classpath")
    executable = shutil.which(str(cfg["java"]))
    if executable is None:
        raise FileNotFoundError("Java executable is unavailable: "+str(cfg["java"]))
    cfg["java"] = str(Path(executable).resolve())
    cfg.setdefault("heap", "1g")
    cfg.setdefault("repo_root", str(Path(__file__).resolve().parents[1]))
    cfg.setdefault("output_root", str(Path.cwd().parent/"synopsislake-results"))
    cfg.setdefault("histogram_dir", "")
    cfg.setdefault("raw_data_dir", "")
    return cfg


def _code_identity(cfg):
    files = {str(Path(__file__).resolve()): _sha(__file__), cfg["java"]: _sha(cfg["java"])}
    for entry in str(cfg["java_classpath"]).split(":"):
        path = Path(entry).expanduser().resolve()
        if path.is_dir():
            for file in sorted(path.rglob("*.class")):
                files[str(file)] = _sha(file)
        elif path.is_file():
            files[str(path)] = _sha(path)
        else:
            raise FileNotFoundError("Classpath entry is unavailable: "+str(path))
    java_source = Path(cfg["repo_root"])/"SynopsisLake/src/main/java"
    for file in sorted(java_source.rglob("*.java")):
        files[str(file.resolve())] = _sha(file)
    version = subprocess.run([cfg["java"], "-version"], capture_output=True, text=True, check=True)
    return dict(files_sha256=files, java_version=(version.stdout+version.stderr).strip(),
                executable_code_bytes=sum(Path(file).stat().st_size for file in files),
                accounting="shared executable/JDK launcher/classes/library/source bytes; not dataset-dependent synopsis storage")


def _check_files(identity):
    for filename, digest in identity["files_sha256"].items():
        if _sha(filename) != digest:
            raise ValueError("Frozen Java executable/source identity changed: "+filename)


def _write_queries(path, frame):
    if "truth" in frame or "actual_error" in frame:
        raise ValueError("Java query inputs must omit offline labels")
    frame[GEOMETRY].to_csv(path, index=False, header=False)


def _java_args(cfg, development, method, seed, query_file, output_file, setup_file,
               *, mode="approx", cutoff_file=None, raw_data_dir=None):
    development = Path(development)
    args = [cfg["java"], "-Xmx"+str(cfg["heap"]), "-cp", cfg["java_classpath"], MAIN_CLASS,
            "-mode", mode, "-queryFile", str(query_file), "-outFile", str(output_file),
            "-setupFile", str(setup_file)]
    if method == "always_exact":
        if raw_data_dir is None:
            raise ValueError("Exact Java execution requires verified raw files")
        return args+["-dataDir", str(raw_data_dir), "-numFile", str(cfg["num_files"])]
    args += ["-runDir", cfg["histogram_dir"], "-mergeRate", repr(cfg["merge_rate"]),
             "-eps", repr(cfg["epsilon"]), "-delta", repr(cfg["delta"])]
    args += (["-cutoff", "0"] if cutoff_file is None else ["-cutoffs", str(cutoff_file)])
    if method in ("ML_v1", "ML_v1_shape"):
        args += ["-modelPath", str(development/"models"/f"seed_{seed}"/method/"model.txt")]
    if method in ("R10", "ML_v1_shape"):
        args += ["-shapeDir", str(development/"native"), "-shapePacket", str(development/"shape.bin")]
    if method == "ML_v1_shape":
        args += ["-augmentShape", "true"]
    if mode == "gated":
        if raw_data_dir is None:
            raise ValueError("Gated Java execution requires verified raw files")
        args += ["-dataDir", str(raw_data_dir), "-numFile", str(cfg["num_files"])]
    return args


def _run_java(cfg, development, method, seed, geometry, folder, *, mode="approx",
              cutoff_file=None, raw_data_dir=None):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=False)
    query_file, output_file, setup_file = folder/"queries.csv", folder/"answers.csv", folder/"setup.csv"
    _write_queries(query_file, geometry)
    pd.DataFrame(dict(java_row=np.arange(len(geometry)), qId=geometry.qId.to_numpy())).to_csv(folder/"query_id_mapping.csv", index=False)
    command = _java_args(cfg, development, method, seed, query_file, output_file, setup_file,
                         mode=mode, cutoff_file=cutoff_file, raw_data_dir=raw_data_dir)
    _json(folder/"command.json", dict(argv=command, queries=len(geometry), query_file_has_truth=False))
    started = time.perf_counter()
    result = subprocess.run(command, capture_output=True, text=True)
    seconds = time.perf_counter()-started
    (folder/"stdout.txt").write_text(result.stdout)
    (folder/"stderr.txt").write_text(result.stderr)
    if result.returncode:
        raise RuntimeError(f"Java {method} failed ({result.returncode}): "+result.stderr[-3000:])
    answers = pd.read_csv(output_file, float_precision="round_trip")
    if len(answers) != len(geometry) or not np.array_equal(answers.qId.to_numpy(), np.arange(len(geometry))):
        raise ValueError("Java output does not map exactly to the input query rows")
    answers["java_row"] = answers.qId
    answers["qId"] = geometry.qId.to_numpy()
    answers["seed"], answers["method"] = int(seed), method
    if answers.exact.notna().any() or answers.relErr.notna().any():
        raise ValueError("Java inference unexpectedly received offline truth")
    if mode == "approx" and (answers.rawNs != 0).any():
        raise ValueError("Geometry-only scoring unexpectedly executed raw scans")
    _json(folder/"process.json", dict(method=method, seed=int(seed), mode=mode,
        wall_seconds=seconds, output_bytes=output_file.stat().st_size,
        timing="subprocess wall to closed answer CSV; JVM/load/query/file output included; no Parquet conversion"))
    return answers, seconds


def _gate_costs(cfg, development, gate_files):
    common_files = sorted(Path(cfg["histogram_dir"]).glob("src_*.txt"))+sorted(Path(cfg["histogram_dir"]).glob("arm_*.txt"))
    rows = []
    for seed in cfg["seeds"]:
        for method in METHODS:
            files = list(common_files)
            if method in ("ML_v1", "ML_v1_shape"):
                model = development/"models"/f"seed_{seed}"/method
                files += [model/"model.txt", model/"model.properties"]
            if method in ("R10", "ML_v1_shape"):
                files += sorted((development/"native").iterdir())+[development/"shape.bin"]
            cutoff = gate_files[(seed, method)]
            files.append(cutoff)
            files = list(dict.fromkeys(files))
            rows.append(dict(seed=seed, method=method, actual_required_file_bytes=sum(p.stat().st_size for p in files),
                shared_histogram_text_bytes=sum(p.stat().st_size for p in common_files),
                cutoff_csv_bytes=cutoff.stat().st_size, cutoff_runtime_scalar_bytes=8,
                workload_query_bytes_policy="workload input files separate; no query table is an assessor artifact",
                required_files="|".join(str(p) for p in files),
                accounting="actual files used by current Java store/model; no model.schema/training metadata read; text is not codec optimal"))
    return pd.DataFrame(rows)


def prepare_profile(config):
    """Calibrate Java backend, then freeze geometry-only scores before labels."""
    cfg = _config(config)
    development = Path(cfg["development_dir"]).expanduser().resolve()
    parent, parent_protocol = common.verify_development(development)
    cfg.update({name: parent["config"][name] for name in ("seeds", "epsilon", "delta", "min_exact",
        "fresh_query_count", "fresh_query_seed", "timing_queries", "timing_repeats", "bootstrap_repeats")})
    cfg["development_dir"] = str(development)
    cfg["histogram_dir"] = str(Path(cfg["histogram_dir"] or parent["config"]["histogram_dir"]).expanduser().resolve())
    histogram = prior.load_histogram(development/"original_H.hist")
    candidates = sorted(Path(cfg["histogram_dir"]).glob("arm_*.txt"))
    matching = [p for p in candidates if all(np.array_equal(a,b) for a,b in zip(features.load_hist(p), histogram))]
    if len(matching) != 1:
        raise ValueError("Java store must contain exactly one text export matching the common fixed H")
    cfg["merge_rate"] = float(histogram[0])
    cfg["num_files"] = len(list((development/"native").glob("src_*.hist")))
    if cfg["fresh_query_count"] < 1000:
        raise ValueError("Common fresh confirmation requires at least 1000 geometries")
    path = _new(cfg["output_root"], "java-shape-profile")
    state = dict(status="calibrating_original_java_memberships", config=cfg,
        development_protocol_sha256=parent["protocol_sha256"], study="separately-calibrated Java backend profile",
        started_utc=datetime.now(timezone.utc).isoformat(), new_raw_labels_generated=False)
    _json(path/"run.json", state)
    code = _code_identity(cfg)
    _json(path/"java_executable_identity.json", code)
    preparation = Path(parent["preparation_dir"])
    queries = pd.read_csv(preparation/"queries_offline.csv", float_precision="round_trip")
    selected = pd.read_csv(development/"selected_query_ids.csv")
    gates, calibrations, parity_rows, gate_files = [], [], [], {}
    python_cal = pd.read_csv(development/"calibration_predictions.csv", float_precision="round_trip")
    for seed in cfg["seeds"]:
        ids = selected.loc[(selected.seed == seed)&(selected.role == "calibration"), "qId"].to_numpy()
        labeled = queries.set_index("qId").loc[ids].reset_index()
        geometry = labeled[["qId", *GEOMETRY]]
        for method in METHODS:
            answers, seconds = _run_java(cfg, development, method, seed, geometry,
                                        path/"calibration"/f"seed_{seed}"/method)
            expected_h = np.asarray([prior.histogram_count(q, histogram) for q in geometry[GEOMETRY].to_numpy(float)])
            if not np.isclose(answers.answer, expected_h, rtol=1e-9, atol=1e-9).all():
                raise ValueError("Java H differs materially from the common frozen answer")
            error = prior._actual_error(answers.answer, labeled.truth.to_numpy(float))
            threshold = empirical_cutoff(answers.errorScore.to_numpy(float), error, cfg["epsilon"], cfg["delta"])
            frame = geometry.assign(seed=seed, method=method, H=answers.answer.to_numpy(),
                score=answers.errorScore.to_numpy(), truth=labeled.truth.to_numpy(), actual_error=error,
                process_wall_seconds=seconds)
            calibrations.append(frame)
            gates.append(dict(seed=seed,method=method,mergeRate=cfg["merge_rate"],eps=cfg["epsilon"],
                delta=cfg["delta"],cutoff=threshold,calibration_queries=len(frame),backend="Java"))
            folder = path/"gates"/f"seed_{seed}"/method
            folder.mkdir(parents=True)
            file = folder/"cutoffs.csv"
            pd.DataFrame([dict(mergeRate=cfg["merge_rate"],eps=cfg["epsilon"],delta=cfg["delta"],cutoff=threshold)]).to_csv(file,index=False)
            gate_files[(seed,method)] = file
            old = python_cal[(python_cal.seed == seed)&(python_cal.method == method)].set_index("qId").loc[ids]
            python_cut = pd.read_csv(development/"frozen_gate_cutoffs.csv").query("seed == @seed and method == @method").cutoff.iloc[0]
            parity_rows.append(dict(seed=seed,method=method,queries=len(frame),
                max_H_abs_difference=float(abs(frame.H.to_numpy()-old.H.to_numpy()).max()),
                max_score_abs_difference=float(abs(frame.score.to_numpy()-old.score.to_numpy()).max()),
                scores_within_1e9_tolerance=bool(np.isclose(frame.score,old.score.to_numpy(),rtol=1e-9,atol=1e-9).all()),
                same_python_cutoff_decision_changes=int(((frame.score<=python_cut).to_numpy() != (old.score<=python_cut).to_numpy()).sum()),
                java_cutoff=threshold,python_cutoff=float(python_cut),
                interpretation="Java backend has its own original-calibration cutoff; Python model/features/gates were not edited"))
            print(f"Java calibration seed {seed} {method}: {len(frame)} original queries, cutoff {threshold}",flush=True)
    pd.concat(calibrations,ignore_index=True).to_csv(path/"java_calibration_predictions.csv",index=False)
    pd.DataFrame(gates).to_csv(path/"frozen_java_gate_cutoffs.csv",index=False)
    pd.DataFrame(parity_rows).to_csv(path/"calibration_backend_parity.csv",index=False)
    _gate_costs(cfg,development,gate_files).to_csv(path/"required_artifact_bytes.csv",index=False)
    _check_files(code)
    common.verify_development(development)
    frozen_assets = {str(file):_sha(file) for file in sorted(development.rglob("*"))
        if file.is_file() and file.suffix not in (".zip",) and "feature_cache" not in file.parts and "confirmation" not in file.parts}
    frozen_assets.update({str(file):_sha(file) for file in sorted(Path(cfg["histogram_dir"]).glob("*.txt"))})
    frozen_assets.update({str(file):_sha(file) for file in gate_files.values()})
    protocol = dict(status="Java gates and model/source/code identities frozen before new geometry",config=cfg,
        development_protocol_sha256=parent["protocol_sha256"],java_code_identity_sha256=_sha(path/"java_executable_identity.json"),
        assets_sha256=frozen_assets,gate_csv_sha256=_sha(path/"frozen_java_gate_cutoffs.csv"),
        backend_gate_policy="same original calibration IDs and empirical rule; actual Java scores, no fresh truth",
        methods=list(METHODS),raw_labels_generated=False,script_sha256=_sha(__file__),
        known_deployment_gap="Pre-fresh diagnostic found 33/256 full-v1 score mismatches per seed and Spearman 0.9908-0.9952; tiny merge_shift rounding can cross trained tree thresholds. Actual Java gets separately calibrated gates; this does not establish Python/Java model equivalence.",
        independence="new rectangles on the inspected same dataset; repeated seeds share geometry")
    _json(path/"protocol_before_geometry.json",protocol)
    state.update(status="frozen_before_new_geometry",protocol_sha256=_sha(path/"protocol_before_geometry.json"))
    _json(path/"run.json",state)
    outer = pd.read_csv(development/"outer_splits.csv")
    donors = outer.loc[(outer.seed == cfg["seeds"][0])&(outer.split == "train"),"qId"].to_numpy()
    root = (histogram[1][0],histogram[1][-1],histogram[2][0],histogram[2][-1])
    donor_geometry = queries.set_index("qId").loc[donors,GEOMETRY].to_numpy(float)
    rectangles = prior._fresh_rectangles(cfg["fresh_query_count"],root,cfg["fresh_query_seed"],donor_geometry)
    geometry = pd.DataFrame(rectangles,columns=GEOMETRY)
    geometry.insert(0,"qId",np.arange(len(geometry),dtype=np.int64)+int(queries.qId.max())+1)
    geometry.to_csv(path/"queries_geometry_only.csv",index=False)
    timing_ids = [dict(seed=seed,qId=qid) for seed in cfg["seeds"] for qid in
                  prior._select(geometry.qId,cfg["timing_queries"],seed,"team-shape-fresh-timing-v1")]
    pd.DataFrame(timing_ids).to_csv(path/"timing_ids_before_labels.csv",index=False)
    scores=[]
    for seed in cfg["seeds"]:
        for method in METHODS:
            answers,_ = _run_java(cfg,development,method,seed,geometry,path/"fresh_prelabel"/f"seed_{seed}"/method,
                                  cutoff_file=gate_files[(seed,method)])
            expected_h = np.asarray([prior.histogram_count(q,histogram) for q in rectangles])
            if not np.isclose(answers.answer,expected_h,rtol=1e-9,atol=1e-9).all():
                raise ValueError("Pre-label Java answers differ from the common fixed H")
            part=geometry.assign(seed=seed,method=method,H=answers.answer.to_numpy(),score=answers.errorScore.to_numpy(),
                accepted=answers.fromSynopsis.to_numpy(),cutoff=answers.cutoff.to_numpy(),
                predicted_error=np.nan if method=="SA" else answers.errorScore.to_numpy())
            if not np.isfinite(part.score).all():raise ValueError("Java emitted nonfinite pre-label scores")
            scores.append(part)
            print(f"Java fresh pre-label seed {seed} {method}: {len(part)} scores saved",flush=True)
    scored=pd.concat(scores,ignore_index=True)
    scored.to_csv(path/"java_scores_before_labels.csv",index=False)
    _json(path/"score_lock.json",dict(protocol_sha256=state["protocol_sha256"],
        geometry_sha256=_sha(path/"queries_geometry_only.csv"),scores_sha256=_sha(path/"java_scores_before_labels.csv"),
        timing_ids_sha256=_sha(path/"timing_ids_before_labels.csv"),raw_labels_generated=False))
    state.update(status="all_java_scores_frozen_before_labels",score_lock_sha256=_sha(path/"score_lock.json"),
                 new_geometry_count=len(geometry),assessor_score_rows=len(scored),new_raw_labels_generated=False)
    _json(path/"run.json",state)
    verify_profile(path)
    return path


def verify_profile(path):
    path=Path(path).expanduser().resolve()
    state=json.loads((path/"run.json").read_text())
    protocol=json.loads((path/"protocol_before_geometry.json").read_text())
    if _sha(path/"protocol_before_geometry.json")!=state["protocol_sha256"] or _sha(__file__)!=protocol["script_sha256"]:
        raise ValueError("Java profile protocol/script changed")
    _check_files(json.loads((path/"java_executable_identity.json").read_text()))
    if _sha(path/"java_executable_identity.json")!=protocol["java_code_identity_sha256"]:
        raise ValueError("Java executable manifest changed")
    for filename,digest in protocol["assets_sha256"].items():
        if _sha(filename)!=digest:raise ValueError("Frozen Java model/source/gate asset changed: "+filename)
    if _sha(path/"frozen_java_gate_cutoffs.csv")!=protocol["gate_csv_sha256"]:
        raise ValueError("Frozen Java gates changed")
    lock=json.loads((path/"score_lock.json").read_text())
    if _sha(path/"score_lock.json")!=state["score_lock_sha256"] or lock["protocol_sha256"]!=state["protocol_sha256"]:
        raise ValueError("Java pre-label score lock changed")
    for filename,key in (("queries_geometry_only.csv","geometry_sha256"),("java_scores_before_labels.csv","scores_sha256"),
                         ("timing_ids_before_labels.csv","timing_ids_sha256")):
        if _sha(path/filename)!=lock[key]:raise ValueError("Java frozen pre-label artifact changed: "+filename)
    common.verify_development(state["config"]["development_dir"])
    return state,protocol


def finalize_profile(config):
    path=Path(config["profile_dir"]).expanduser().resolve()
    state,protocol=verify_profile(path)
    confirmation=Path(config["confirmation_dir"]).expanduser().resolve()
    python_state,_=common.verify_confirmation(confirmation)
    if python_state.get("status")!="complete" or Path(python_state["development_dir"]).resolve()!=Path(state["config"]["development_dir"]):
        raise ValueError("Java profile requires the same completed common confirmation")
    java_geometry=pd.read_csv(path/"queries_geometry_only.csv",float_precision="round_trip")
    python_geometry=pd.read_csv(confirmation/"queries_geometry_only.csv",float_precision="round_trip")
    pd.testing.assert_frame_equal(java_geometry,python_geometry,check_exact=True)
    labels=pd.read_csv(confirmation/"query_labels_offline.csv",float_precision="round_trip")
    from .shape_execution import _geometry_labels
    _geometry_labels(java_geometry,labels)
    audit=pd.read_csv(confirmation/"fresh_raw_audit.csv")
    if len(audit)<3 or not audit.matches.all():raise ValueError("Independent original raw-label audits are missing")
    scored=pd.read_csv(path/"java_scores_before_labels.csv",float_precision="round_trip")
    quality=common.attach_truth(scored,labels)
    gates=pd.read_csv(path/"frozen_java_gate_cutoffs.csv",float_precision="round_trip")
    metrics,coverage=common.quality_tables(quality,gates,state["config"])
    quality.to_csv(path/"java_per_query_quality.csv",index=False)
    metrics.to_csv(path/"java_metrics.csv",index=False)
    coverage.to_csv(path/"java_fixed_coverage.csv",index=False)
    old=pd.read_csv(confirmation/"per_query_quality.csv",float_precision="round_trip")
    parity=quality.merge(old[["seed","method","qId","H","score"]],on=["seed","method","qId"],
                         validate="one_to_one",suffixes=("_java","_python"))
    rows=[]
    for (seed,method),part in parity.groupby(["seed","method"]):
        pc=float(pd.read_csv(Path(state["config"]["development_dir"])/"frozen_gate_cutoffs.csv").query("seed==@seed and method==@method").cutoff.iloc[0])
        rows.append(dict(seed=seed,method=method,queries=len(part),max_H_abs_delta=float(abs(part.H_java-part.H_python).max()),
            max_score_abs_delta=float(abs(part.score_java-part.score_python).max()),
            score_values_within_1e9_tolerance=bool(np.isclose(part.score_java,part.score_python,rtol=1e-9,atol=1e-9).all()),
            same_python_cutoff_decision_changes=int(((part.score_java<=pc)!=(part.score_python<=pc)).sum()),
            note="Backend differences reported; actual Java gates were calibrated before these labels"))
    pd.DataFrame(rows).to_csv(path/"fresh_backend_parity.csv",index=False)
    state.update(status="quality_finalized",source_python_confirmation=str(confirmation),
        source_python_protocol_sha256=_sha(confirmation/"protocol.json"),labels_sha256=_sha(confirmation/"query_labels_offline.csv"),
        java_quality_queries=java_geometry.qId.nunique(),new_raw_labels_generated=False)
    _json(path/"run.json",state)
    verify_profile(path)
    return path


def prepare_raw_files(path):
    path=Path(path)
    state,_=verify_profile(path)
    development=Path(state["config"]["development_dir"])
    audit=pd.read_csv(development/"source_audit.csv")
    target=path/"raw_timing"
    target.mkdir(exist_ok=False)
    rows=[]
    for row in audit.sort_values("source_id").itertuples(index=False):
        source=Path(row.raw_path)
        if _sha(source)!=row.raw_sha256:raise ValueError("Original raw input identity changed")
        file=target/f"{row.source_id}.csv"
        if source.suffix==".gz":
            with gzip.open(source,"rb") as inp,file.open("wb") as out:shutil.copyfileobj(inp,out)
        else:
            file.symlink_to(source.resolve())
        rows.append(dict(source_id=int(row.source_id),source_path=str(source),source_sha256=row.raw_sha256,
                         raw_rows=int(row.raw_rows),java_path=str(file),java_uncompressed_sha256=_sha(file),bytes=file.stat().st_size))
    _json(path/"raw_timing_manifest.json",rows)
    return target


def run_timing(config):
    """Use existing Java gated/exact execution; process and query-sum costs differ."""
    path=Path(config["profile_dir"]).expanduser().resolve()
    state,_=verify_profile(path)
    if state.get("status")!="quality_finalized":raise ValueError("Finalize audited quality before Java timing")
    cfg=state["config"]
    confirmation=Path(state["source_python_confirmation"])
    labels=pd.read_csv(confirmation/"query_labels_offline.csv",float_precision="round_trip")
    if _sha(confirmation/"query_labels_offline.csv")!=state["labels_sha256"]:raise ValueError("Frozen raw labels changed")
    raw=prepare_raw_files(path)
    timing_ids=pd.read_csv(path/"timing_ids_before_labels.csv")
    frozen=pd.read_csv(path/"java_scores_before_labels.csv",float_precision="round_trip")
    records=[]
    wall=[]
    for seed in cfg["seeds"]:
        ids=timing_ids.loc[timing_ids.seed==seed,"qId"].to_numpy()
        labeled=labels.set_index("qId").loc[ids].reset_index()
        geometry=labeled[["qId",*GEOMETRY]]
        for repeat in range(cfg["timing_repeats"]):
            order=np.random.default_rng(cfg["fresh_query_seed"]+seed+repeat).permutation([*METHODS,"always_exact"])
            for method in order:
                folder=path/"timing"/f"seed_{seed}"/f"repeat_{repeat}"/str(method)
                cutoff=None if method=="always_exact" else path/"gates"/f"seed_{seed}"/str(method)/"cutoffs.csv"
                actual,seconds=_run_java(cfg,Path(cfg["development_dir"]),str(method),seed,geometry,folder,
                    mode="exact" if method=="always_exact" else "gated",cutoff_file=cutoff,raw_data_dir=raw)
                actual["repeat"],actual["execution_order"]=repeat,int(np.where(order==method)[0][0])
                actual["truth"]=labeled.truth.to_numpy()
                if method=="always_exact":
                    if actual.fromSynopsis.any() or not np.array_equal(actual.answer,actual.truth):
                        raise ValueError("Java exact execution differs from verified original truth")
                else:
                    expected=frozen[(frozen.seed==seed)&(frozen.method==method)].set_index("qId").loc[ids]
                    if not np.isclose(actual.errorScore,expected.score,rtol=1e-9,atol=1e-9).all() or not np.array_equal(actual.fromSynopsis,expected.accepted):
                        raise ValueError("Timed Java scores/decisions differ from pre-label freeze")
                    accepted=actual.fromSynopsis.to_numpy(bool)
                    if not np.isclose(actual.answer[accepted],expected.H.to_numpy()[accepted],rtol=1e-9,atol=1e-9).all():
                        raise ValueError("Accepted Java answer differs from unchanged H")
                    if not np.array_equal(actual.answer[~accepted],actual.truth[~accepted]):
                        raise ValueError("Java fallback did not return the actual raw count")
                    if (actual.rawNs[accepted]!=0).any() or (actual.rawNs[~accepted]<=0).any():
                        raise ValueError("Java fallback timings contradict the decisions")
                actual["returned_relative_error"]=prior._actual_error(actual.answer,actual.truth)
                actual["accepted_bad"]=actual.fromSynopsis&(actual.returned_relative_error>cfg["epsilon"])
                actual["query_compute_ns"]=actual.gateNs+actual.synNs+actual.rawNs
                records.append(actual)
                wall.append(dict(seed=seed,method=method,repeat=repeat,queries=len(actual),process_wall_seconds=seconds,
                                 setup_csv=str(folder/"setup.csv"),output_csv=str(folder/"answers.csv")))
                print(f"Java live timing seed{seed} repeat{repeat} {method}: {seconds:.3f}s including JVM/raw load/output",flush=True)
    data=pd.concat(records,ignore_index=True)
    data.to_csv(path/"java_timed_requests.csv",index=False)
    pd.DataFrame(wall).to_csv(path/"java_process_wall.csv",index=False)
    summaries=[]
    for method,part in data.groupby("method"):
        exact=data[data.method=="always_exact"]
        one=part.drop_duplicates(["seed","qId"])
        count=int(one.fromSynopsis.sum())
        bad=int(one.accepted_bad.sum())
        rows=pd.DataFrame(wall)
        process=rows[rows.method==method].process_wall_seconds
        baseline=rows[rows.method=="always_exact"].process_wall_seconds
        summaries.append(dict(method=method,observations=len(part),unique_query_ids=part.qId.nunique(),seed_query_cases=len(one),
            accepted_seed_query_cases=count,accepted_bad_seed_query_cases=bad,timing_subset_risk=bad/count if count else np.nan,
            query_compute_mean_ms=part.query_compute_ns.mean()/1e6,query_compute_p95_ms=part.query_compute_ns.quantile(.95)/1e6,
            query_compute_speedup_ratio=exact.query_compute_ns.mean()/part.query_compute_ns.mean(),
            process_wall_mean_s=process.mean(),process_wall_median_s=process.median(),process_wall_p95_s=process.quantile(.95),
            process_wall_speedup_ratio_of_means=baseline.mean()/process.mean(),
            timing_scope="query compute=sum logged gate+synopsis+raw only; complete process wall reported separately; no Parquet"))
    pd.DataFrame(summaries).to_csv(path/"java_timing_summary.csv",index=False)
    state.update(status="timing_complete",timed_requests=len(data),real_raw_rows=int(pd.read_json(path/"raw_timing_manifest.json").raw_rows.sum()))
    _json(path/"run.json",state)
    verify_profile(path)
    return path


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config",type=Path)
    parser.add_argument("--phase",choices=("prepare","finalize","timing"),default="prepare")
    args=parser.parse_args(argv)
    cfg=json.loads(args.config.read_text())
    action={"prepare":prepare_profile,"finalize":finalize_profile,"timing":run_timing}[args.phase]
    print(action(cfg))


if __name__=="__main__":main()
