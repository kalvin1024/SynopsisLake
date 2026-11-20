"""Build notebook 22's source-shape payload for the shared Java benchmark.

Only source histogram exports and raw point files are inputs. No query geometry,
query labels, model weights, train/test roles, or histogram answers are read.
The packet uses the existing hierarchical R08 codec and source-only total-
variation priority. It is not a new learner or an accuracy guarantee.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
import re
import time

import numpy as np
import pandas as pd

from . import hierarchical_shape as hierarchical
from . import local_shape
from . import sidecar_workflow as prior
from .features import load_hist


_IMPLEMENTATION = (
    "shape_e2e_inputs.py", "sidecar_workflow.py", "features.py",
    "hierarchical_shape.py", "local_shape.py", "bounded_sidecar.py",
    "_distribution_math.py",
)


def _integer(value, name, minimum):
    if isinstance(value, bool) or int(value) != value or int(value) < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _source_files(run_dir, raw_data_dir):
    source = {}
    for file in Path(run_dir).glob("src_*.txt"):
        match = re.fullmatch(r"src_(\d+)\.txt", file.name)
        if match is None:
            raise ValueError(f"Invalid source histogram filename: {file.name}")
        sid = int(match.group(1))
        if file.name != f"src_{sid}.txt" or sid in source:
            raise ValueError(f"Source IDs must have canonical, unique filenames: {file.name}")
        source[sid] = file
    if not source or set(source) != set(range(len(source))):
        raise ValueError("Source exports must be contiguous src_0.txt .. src_N.txt")
    raw = {}
    for file in Path(raw_data_dir).iterdir():
        match = re.fullmatch(r"(\d+)\.csv(?:\.gz)?", file.name)
        if match:
            sid = int(match.group(1))
            if file.name not in (f"{sid}.csv", f"{sid}.csv.gz"):
                raise ValueError(f"Raw IDs must have canonical filenames: {file.name}")
            # The Java path consumes CSV; prefer it when a gzip copy also exists.
            if sid not in raw or file.suffix == ".csv":
                raw[sid] = file
    if set(raw) != set(source):
        raise ValueError("Numbered raw point files must match source histogram IDs exactly")
    return [(sid, source[sid], raw[sid]) for sid in sorted(source)]


def _write_manifest(path, state):
    (Path(path) / "manifest.json").write_text(
        json.dumps(state, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def prepare_shape_metadata(run_dir, raw_data_dir, output_dir, *,
                           packet_budget_bytes=49152, chunk_size=250000) -> Path:
    """Stream raw sources into unchanged native/child moment codecs.

    ``output_dir`` must not exist. Source row counts must reproduce every
    exported source cell exactly; this is an unweighted point-count contract.
    Additional raw columns are ignored like the Java raw counter (e.g. timestamps)
    and are never interpreted as weights. Uniform, half-open source grids are
    required by the existing moment codec; incompatible inputs fail explicitly.

    ``native`` and ``shape.bin`` are the only online payload files. Construction
    reads raw data offline and records its elapsed time separately. No raw cache
    is persisted and no ML model is trained. Completed output is immutable.
    """
    budget = _integer(packet_budget_bytes, "packet_budget_bytes", 1)
    chunk_size = _integer(chunk_size, "chunk_size", 1)
    run_dir, raw_data_dir = Path(run_dir).expanduser().resolve(), Path(raw_data_dir).expanduser().resolve()
    output = Path(output_dir).expanduser().resolve()
    inputs = _source_files(run_dir, raw_data_dir)
    output.mkdir(parents=True, exist_ok=False)
    state = dict(status="preparing", method="source_shape", notebook_number=22,
                 legacy_method="R10", packet_mode="hierarchical_r08",
                 selection_rule="source_only_parent_mass_weighted_quadrant_total_variation",
                 selection_uses_query_geometry=False, selection_uses_query_labels=False,
                 packet_budget_bytes=budget, chunk_size=chunk_size,
                 source_count=len(inputs), started_utc=datetime.now(timezone.utc).isoformat(),
                 run_dir=str(run_dir), raw_data_dir=str(raw_data_dir),
                 raw_semantics="unweighted rows; first two whitespace-separated fields; half-open source grids",
                 implementation_sha256={name: prior._sha(Path(__file__).with_name(name))
                                        for name in _IMPLEMENTATION})
    _write_manifest(output, state)
    started = time.perf_counter()
    try:
        native = output / "native"
        native.mkdir()
        histograms, encoded, children, child_encoded, audit, identities = [], [], [], [], [], []
        for sid, source, raw in inputs:
            tick = time.perf_counter()
            input_hash = prior._sha(source)
            raw_hash = prior._sha(raw)
            before = (raw.stat().st_size, raw.stat().st_mtime_ns)
            hist = prior._validate_hist(load_hist(source), exact=True)
            # Validate regularity before processing even an empty source.
            prior._sufficient(np.empty((0, 2)), hist)
            size = hist[3].size
            expected_rows = sum(map(int, hist[3].ravel()))
            counts = np.zeros(size, dtype=np.uint64)
            sums = np.zeros((size, 5), dtype=np.float64)
            child_counts = np.zeros((size, 4), dtype=np.uint64)
            child_sums = np.zeros((size, 4, 5), dtype=np.float64)
            rows, clipped = 0, 0
            if raw.stat().st_size:
                for points in prior.iter_raw(raw, chunk_size):
                    count, additive, rounding = prior._sufficient(points, hist)
                    quarter, quarter_sums = hierarchical.accumulate_child_moments(points, hist)
                    counts += count
                    sums += additive
                    child_counts += quarter
                    child_sums += quarter_sums
                    rows += len(points)
                    clipped += rounding
                    if rows > expected_rows:
                        raise ValueError(f"Raw rows exceed exported unweighted source mass: source {sid}")
            if rows != expected_rows or not np.array_equal(counts, hist[3].ravel()):
                raise ValueError(f"Raw counts do not reproduce exported source cells: source {sid}")
            if not np.array_equal(child_counts.sum(axis=1), counts):
                raise ValueError(f"Child counts disagree with exact parent counts: source {sid}")
            if before != (raw.stat().st_size, raw.stat().st_mtime_ns) or prior._sha(raw) != raw_hash:
                raise ValueError(f"Raw source changed during construction: source {sid}")
            if prior._sha(source) != input_hash:
                raise ValueError(f"Source histogram changed during construction: source {sid}")
            prior.save_histogram(native / f"src_{sid}.hist", hist)
            prior._save_structured_moments(native / f"src_{sid}-moments.bin", hist,
                                           prior._moments(counts, sums), sid)
            # Decode the persisted bytes rather than regenerate a distinct codec.
            payload = (native / f"src_{sid}-moments.bin").read_bytes()
            encoded.append(np.frombuffer(payload, np.uint8, offset=prior._MOMENT_HEADER.size)
                           .reshape(-1, 5).copy())
            histograms.append(hist)
            children.append(child_counts)
            child_encoded.append(hierarchical.encode_child_moments(child_counts, child_sums))
            record = dict(source_id=sid, histogram_path=str(source), histogram_sha256=input_hash,
                          raw_path=str(raw), raw_sha256=raw_hash, raw_bytes=raw.stat().st_size,
                          raw_rows=rows, source_cells=size, counts_match=True,
                          child_counts_match=True, arithmetic_clip_rows=clipped,
                          construction_seconds=time.perf_counter()-tick)
            audit.append(record)
            identities.append({key: record[key] for key in (
                "source_id", "histogram_path", "histogram_sha256", "raw_path", "raw_sha256", "raw_rows")})
        packet = hierarchical.build_selection(histograms, encoded, children, child_encoded,
                                               budget, mode="hierarchical_r08")
        selection = hierarchical.decode_selection(histograms, encoded, packet)
        (output / "shape.bin").write_bytes(packet)
        pd.DataFrame(audit).to_csv(output / "source_audit.csv", index=False)
        native_bytes = sum(file.stat().st_size for file in native.iterdir())
        files = [output / "shape.bin", output / "source_audit.csv", *sorted(native.iterdir())]
        state.update(status="complete", source_inputs=identities,
                     raw_rows=sum(row["raw_rows"] for row in audit),
                     source_cells=sum(row["source_cells"] for row in audit),
                     packet_bytes=len(packet), native_bytes=native_bytes,
                     auxiliary_bytes=native_bytes+len(packet), selected_cells=len(selection.selected_ids),
                     construction_seconds=time.perf_counter()-started,
                     artifact_sha256={str(file.relative_to(output)): prior._sha(file) for file in files})
        _write_manifest(output, state)
        verify_shape_metadata(output, run_dir=run_dir)
        return output
    except BaseException as error:
        state.update(status="failed", error=f"{type(error).__name__}: {error}",
                     construction_seconds=time.perf_counter()-started)
        _write_manifest(output, state)
        raise


def verify_shape_metadata(path, run_dir=None):
    """Verify completed payloads and optionally current exported source identity.

    Returns the manifest. This verifies recorded input hashes and persisted
    payload integrity; it does not rescan original raw files or trust an R10 run
    from another dataset. Current source exports must be supplied by callers
    reusing a payload with another Java ``run_dir``.
    """
    path = Path(path).expanduser().resolve()
    state = json.loads((path / "manifest.json").read_text())
    if (state.get("status") != "complete" or state.get("method") != "source_shape"
            or state.get("packet_mode") != "hierarchical_r08"
            or state.get("selection_uses_query_geometry") is not False
            or state.get("selection_uses_query_labels") is not False):
        raise ValueError("Source-shape metadata is not a completed source-only construction")
    required = {"shape.bin", "source_audit.csv"}
    source_count = state.get("source_count")
    if not isinstance(source_count, int) or isinstance(source_count, bool) or source_count < 1:
        raise ValueError("Invalid source count in source-shape metadata")
    for sid in range(source_count):
        required |= {f"native/src_{sid}.hist", f"native/src_{sid}-moments.bin"}
    hashes = state.get("artifact_sha256", {})
    if set(hashes) != required:
        raise ValueError("Source-shape artifact inventory changed")
    for name, expected in hashes.items():
        file = path / name
        if not file.is_file() or prior._sha(file) != expected:
            raise ValueError(f"Source-shape artifact changed or missing: {name}")
    native_files = {str(file.relative_to(path)) for file in (path / "native").iterdir()}
    if native_files != {name for name in required if name.startswith("native/")}:
        raise ValueError("Unexpected source-shape native file inventory")
    hists, encoded = local_shape.load_native(path / "native")
    packet = (path / "shape.bin").read_bytes()
    selection = hierarchical.decode_selection(hists, encoded, packet)
    native_bytes = sum((path / name).stat().st_size for name in required if name.startswith("native/"))
    if (state.get("native_bytes") != native_bytes or state.get("packet_bytes") != len(packet)
            or state.get("auxiliary_bytes") != native_bytes+len(packet)
            or state.get("packet_budget_bytes") != selection.budget_bytes
            or state.get("selected_cells") != len(selection.selected_ids)
            or len(hists) != source_count):
        raise ValueError("Source-shape codec and manifest accounting disagree")
    if run_dir is not None:
        run_dir = Path(run_dir).expanduser().resolve()
        paths = {file.name for file in run_dir.glob("src_*.txt")}
        expected_paths = {f"src_{sid}.txt" for sid in range(source_count)}
        if paths != expected_paths:
            raise ValueError("Source-shape payload source count differs from this Java run")
        records = state.get("source_inputs", [])
        if len(records) != source_count or {record["source_id"] for record in records} != set(range(source_count)):
            raise ValueError("Invalid source input identities")
        for record in records:
            source = run_dir / f"src_{record['source_id']}.txt"
            if prior._sha(source) != record["histogram_sha256"]:
                raise ValueError(f"Source-shape payload belongs to different source exports: {source.name}")
    return state
