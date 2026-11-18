"""Explicitly post-hoc batch-amortization experiment using frozen Java assets.

Prepare an intent first, then run only after the caller gives exclusive timing
access. No source/profile/model/gate/label is changed. This is not preregistered
method evidence: it asks whether the existing 32-query JVM/IO overhead changes
at the peer notebook's approximately 2,000-query operating regime.
"""
from pathlib import Path
from datetime import datetime, timezone
import argparse
import hashlib
import json
import sys
import time
from uuid import uuid4

import numpy as np
import pandas as pd

from ml import shape_java_comparison as java
from ml import sidecar_workflow as prior


METHODS = (*java.METHODS, "always_exact")
REPETITIONS = 3
SEED = 0  # Existing team notebook09 default, not a selected best seed.


def sha(path):
    digest=hashlib.sha256()
    with Path(path).open("rb") as stream:
        for data in iter(lambda:stream.read(1024*1024),b""):digest.update(data)
    return digest.hexdigest()


def save(path,value):
    prior._json(path,value)


def prepare(profile_dir,output_root):
    profile=Path(profile_dir).expanduser().resolve()
    state,protocol=java.verify_profile(profile)
    if state.get("status")!="timing_complete":raise ValueError("The original formal32 profile must be complete")
    geometry=pd.read_csv(profile/"queries_geometry_only.csv",float_precision="round_trip")
    if len(geometry)!=2048 or geometry.qId.nunique()!=2048:
        raise ValueError("This fixed post-hoc operating regime uses ALL 2048 existing queries")
    confirmation=Path(state["source_python_confirmation"])
    labels=confirmation/"query_labels_offline.csv"
    if sha(labels)!=state["labels_sha256"]:raise ValueError("Previously verified labels changed")
    if SEED not in state["config"]["seeds"]:raise ValueError("The fixed default seed0 is unavailable")
    output=Path(output_root)/("java-full-batch-posthoc-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid4().hex[:8])
    output.mkdir(parents=True,exist_ok=False)
    order=[]
    for repeat in range(REPETITIONS):
        methods=np.random.default_rng(2026111902+repeat).permutation(METHODS)
        order.extend(dict(repeat=repeat,position=position,method=str(method)) for position,method in enumerate(methods))
    pd.DataFrame(order).to_csv(output/"predetermined_execution_order.csv",index=False)
    geometry.to_csv(output/"all_existing_queries_geometry_only.csv",index=False)
    unchanged=[profile/"run.json",profile/"protocol_before_geometry.json",profile/"score_lock.json",
               profile/"frozen_java_gate_cutoffs.csv",profile/"java_scores_before_labels.csv",
               profile/"java_timing_summary.csv",profile/"java_timed_requests.csv",labels]
    intent=dict(status="intent_frozen_before_posthoc_timing",posthoc=True,
        reason="After observing 32-query batch overhead, test the peer notebook's full ~2000-query operating regime; no method/threshold/model retuning",
        created_utc=datetime.now(timezone.utc).isoformat(),source_profile=str(profile),
        source_profile_protocol_sha256=sha(profile/"protocol_before_geometry.json"),
        source_label_sha256=sha(labels),source_confirmation=str(confirmation),
        seed=SEED,seed_rule="fixed team notebook09 default0, not a best-performing seed choice",
        queries=2048,query_rule="ALL same second-cohort geometries in their original order, no quality/subset selection",
        repeats=REPETITIONS,methods=list(METHODS),new_models=False,new_gates=False,new_labels=False,
        script_sha256=sha(__file__),java_module_sha256=sha(java.__file__),
        geometry_sha256=sha(output/"all_existing_queries_geometry_only.csv"),
        order_sha256=sha(output/"predetermined_execution_order.csv"),
        preserved_file_sha256={str(file):sha(file) for file in unchanged},
        timing_scope="complete process wall includes fresh JVM, all loading, requests, closed CSV output; query compute is logged gate+synopsis+raw only; no Parquet",
        reporting_rule="retain BOTH formal32 and posthoc2048 outcomes regardless of gain; same-seed32 comparison separate from three-seed32 aggregate")
    save(output/"intent_before_timing.json",intent)
    print(output,flush=True)
    return output


def verify(output):
    output=Path(output)
    intent=json.loads((output/"intent_before_timing.json").read_text())
    if sha(__file__)!=intent["script_sha256"] or sha(java.__file__)!=intent["java_module_sha256"]:
        raise ValueError("Frozen post-hoc execution source changed")
    for path,digest in intent["preserved_file_sha256"].items():
        if sha(path)!=digest:raise ValueError("Original formal experiment changed: "+path)
    if sha(output/"all_existing_queries_geometry_only.csv")!=intent["geometry_sha256"]:
        raise ValueError("Post-hoc workload changed")
    if sha(output/"predetermined_execution_order.csv")!=intent["order_sha256"]:
        raise ValueError("Post-hoc execution order changed")
    state,_=java.verify_profile(intent["source_profile"])
    return intent,state


def run(output):
    output=Path(output).resolve()
    intent,state=verify(output)
    if (output/"run.json").exists():raise ValueError("Do not overwrite an attempted timing run")
    cfg=dict(state["config"])
    cfg["seeds"]=[SEED]
    development=Path(cfg["development_dir"])
    profile=Path(intent["source_profile"])
    geometry=pd.read_csv(output/"all_existing_queries_geometry_only.csv",float_precision="round_trip")
    labels=pd.read_csv(Path(intent["source_confirmation"])/"query_labels_offline.csv",float_precision="round_trip")
    labels=labels.set_index("qId").loc[geometry.qId].reset_index()
    frozen=pd.read_csv(profile/"java_scores_before_labels.csv",float_precision="round_trip")
    raw=profile/"raw_timing"
    raw_manifest=json.loads((profile/"raw_timing_manifest.json").read_text())
    for item in raw_manifest:
        if sha(item["source_path"])!=item["source_sha256"] or sha(item["java_path"])!=item["java_uncompressed_sha256"]:
            raise ValueError("Verified raw files changed before post-hoc timing")
    save(output/"run.json",dict(status="timing",posthoc=True,started_utc=datetime.now(timezone.utc).isoformat()))
    wall=[]
    frames=[]
    started=time.perf_counter()
    order=pd.read_csv(output/"predetermined_execution_order.csv")
    for row in order.itertuples(index=False):
        method,repeat=str(row.method),int(row.repeat)
        folder=output/"measurements"/f"repeat_{repeat}"/method
        cutoff=None if method=="always_exact" else profile/"gates"/f"seed_{SEED}"/method/"cutoffs.csv"
        actual,seconds=java._run_java(cfg,development,method,SEED,geometry,folder,
            mode="exact" if method=="always_exact" else "gated",cutoff_file=cutoff,raw_data_dir=raw)
        actual["repeat"],actual["execution_order"],actual["truth"]=repeat,int(row.position),labels.truth.to_numpy()
        if method=="always_exact":
            if actual.fromSynopsis.any() or not np.array_equal(actual.answer,actual.truth):
                raise ValueError("Exact Java batch differs from original raw labels")
        else:
            expected=frozen[(frozen.seed==SEED)&(frozen.method==method)].set_index("qId").loc[geometry.qId]
            if (not np.isclose(actual.errorScore,expected.score,rtol=1e-9,atol=1e-9).all() or
                not np.array_equal(actual.fromSynopsis,expected.accepted)):
                raise ValueError("Post-hoc Java scores/decisions differ from the original frozen assessor")
            accepted=actual.fromSynopsis.to_numpy(bool)
            if not np.isclose(actual.answer[accepted],expected.H.to_numpy()[accepted],rtol=1e-9,atol=1e-9).all():
                raise ValueError("Accepted answer changed H")
            if not np.array_equal(actual.answer[~accepted],actual.truth[~accepted]):
                raise ValueError("Java fallback does not return the verified raw count")
            if (actual.rawNs[accepted]!=0).any() or (actual.rawNs[~accepted]<=0).any():
                raise ValueError("Raw scan timings contradict actual decisions")
        actual["returned_relative_error"]=prior._actual_error(actual.answer,actual.truth)
        actual["accepted_bad"]=actual.fromSynopsis&(actual.returned_relative_error>cfg["epsilon"])
        actual["query_compute_ns"]=actual.gateNs+actual.synNs+actual.rawNs
        frames.append(actual)
        wall.append(dict(seed=SEED,method=method,repeat=repeat,queries=len(actual),process_wall_seconds=seconds,
                         setup_csv=str(folder/"setup.csv")))
        print(f"POSTHOC full batch repeat{repeat} {method}: {seconds:.3f}s, all2048fixedqueries",flush=True)
    data=pd.concat(frames,ignore_index=True)
    data.to_csv(output/"posthoc_timed_requests.csv",index=False)
    times=pd.DataFrame(wall)
    times.to_csv(output/"posthoc_process_wall.csv",index=False)
    summary=_summary(data,times,"posthoc2048_seed0")
    old=pd.read_csv(profile/"java_timed_requests.csv")
    old_wall=pd.read_csv(profile/"java_process_wall.csv")
    formal=_summary(old[old.seed==SEED],old_wall[old_wall.seed==SEED],"formal32_seed0")
    pd.concat((formal,summary),ignore_index=True).to_csv(output/"batch_regime_comparison.csv",index=False)
    summary.to_csv(output/"posthoc_summary.csv",index=False)
    for item in raw_manifest:
        if sha(item["source_path"])!=item["source_sha256"] or sha(item["java_path"])!=item["java_uncompressed_sha256"]:
            raise ValueError("Raw inputs changed during post-hoc timing")
    verify(output)
    save(output/"run.json",dict(status="complete",posthoc=True,seed=SEED,unique_queries=2048,
        timed_requests=len(data),processes=len(wall),repeats=REPETITIONS,
        wall_seconds=time.perf_counter()-started,all_original_profiles_unchanged=True,
        all_live_scores_decisions_and_real_raw_returns_verified=True,
        all_raw_hashes_unchanged=True,methods_or_gates_selected_using_this_result=False,
        completed_utc=datetime.now(timezone.utc).isoformat()))
    print(summary.to_string(index=False),flush=True)
    return output


def _summary(data,wall,regime):
    exact=data[data.method=="always_exact"]
    baseline=wall[wall.method=="always_exact"].process_wall_seconds
    rows=[]
    for method,part in data.groupby("method"):
        one=part.drop_duplicates("qId")
        accepted=int(one.fromSynopsis.sum())
        bad=int(one.accepted_bad.sum())
        process=wall[wall.method==method].process_wall_seconds
        rows.append(dict(regime=regime,seed=SEED,method=method,queries=part.qId.nunique(),observations=len(part),
            repeats=part['repeat'].nunique(),accepted_count=accepted,accepted_bad_count=bad,
            accepted_risk=bad/accepted if accepted else np.nan,
            query_compute_mean_ms=part.query_compute_ns.mean()/1e6,
            query_compute_p95_ms=part.query_compute_ns.quantile(.95)/1e6,
            query_compute_speedup=exact.query_compute_ns.mean()/part.query_compute_ns.mean(),
            process_wall_mean_s=process.mean(),process_wall_median_s=process.median(),
            process_wall_min_s=process.min(),process_wall_max_s=process.max(),
            process_wall_speedup_ratio_of_means=baseline.mean()/process.mean(),posthoc=regime.startswith("posthoc")))
    return pd.DataFrame(rows)


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile-dir",type=Path)
    parser.add_argument("--output-root",type=Path)
    parser.add_argument("--run-dir",type=Path,help="Execute an already prepared intent; caller must provide exclusive timing access")
    args=parser.parse_args(argv)
    if args.run_dir:print(run(args.run_dir))
    elif args.profile_dir and args.output_root:prepare(args.profile_dir,args.output_root)
    else:parser.error("Supply profile/output to prepare only, or --run-dir to execute")


if __name__=="__main__":main()
