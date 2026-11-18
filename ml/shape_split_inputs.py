"""Standalone grouped-cell split handoff for R10; no historical notebook runs.

The recipe reproduces ``model_study.source_cells`` and ``gate_eval.split_ids``
without importing a model/training library. Labels are used only for the declared
offline truth>=min_exact cohort, never for grouping, ranking or role selection.
The historical AUTHORS identity is checked against a recorded byte-level result;
other inputs are explicitly a new declared grouped cohort, not a replication.
"""
from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from . import shape_inputs
from . import sidecar_workflow as prior


ROLES = ("train", "calibration", "test")
RECIPE = "source-size-center-groups-v1_50-20-30"


def _seeds(values):
    values = values.split(",") if isinstance(values, str) else list(values)
    if not values or any(isinstance(v, bool) or int(v) != float(v) for v in values):
        raise ValueError("seeds must contain distinct integers")
    values = list(map(int, values))
    if len(values) != len(set(values)) or any(v < 0 for v in values):
        raise ValueError("seeds must contain distinct nonnegative integers")
    return values


def source_cell_ids(queries, histogram_dir):
    """Reproduce the original nearest-size, midpoint, searchsorted-left IDs.

    This is generating-cell reconstruction for the AUTHORS cell-sized query
    workload. For a different workload the same rule is only a declared size/
    midpoint grouping, not proof of geographic or population independence.
    """
    folder = Path(histogram_dir)
    paths = sorted(folder.glob("src_*.txt"), key=lambda p: int(p.stem.split("_")[-1]))
    if not paths or [int(p.stem.split("_")[-1]) for p in paths] != list(range(len(paths))):
        raise ValueError("The historical source-cell recipe requires contiguous src_0.txt ... exports")
    grids = []
    for path in paths:
        lines = path.read_text().splitlines()
        if len(lines) < 3:
            raise ValueError("Truncated source histogram: "+str(path))
        x, y = np.asarray(lines[1].split(","), float), np.asarray(lines[2].split(","), float)
        if (len(x) < 2 or len(y) < 2 or not np.isfinite(np.r_[x, y]).all()
                or (np.diff(x) <= 0).any() or (np.diff(y) <= 0).any()):
            raise ValueError("Invalid source boundaries: "+str(path))
        grids.append((x, y, np.diff(x).mean(), np.diff(y).mean()))
    width = (queries.x1-queries.x0).to_numpy(float)
    height = (queries.y1-queries.y0).to_numpy(float)
    cx = ((queries.x0+queries.x1)/2).to_numpy(float)
    cy = ((queries.y0+queries.y1)/2).to_numpy(float)
    sizes = np.asarray([[abs(w-g[2])+abs(h-g[3]) for g in grids] for w, h in zip(width, height)])
    sources = sizes.argmin(axis=1)
    ix = [np.searchsorted(grids[s][0], x)-1 for s, x in zip(sources, cx)]
    iy = [np.searchsorted(grids[s][1], y)-1 for s, y in zip(sources, cy)]
    # No clipping or alternative source choice: this preserves the inherited rule.
    cells = [f"{s}_{i}_{j}" for s, i, j in zip(sources, ix, iy)]
    return pd.Series(cells, index=queries.qId.to_numpy(), name="cell"), paths


def grouped_memberships(queries, histogram_dir, seeds=(0, 1, 2), min_exact=10):
    """Create deterministic memberships while preserving original query IDs."""
    seeds = _seeds(seeds)
    if isinstance(min_exact, bool) or int(min_exact) != min_exact or min_exact < 0:
        raise ValueError("min_exact must be a nonnegative integer")
    if not len(queries) or not {"qId", "x0", "x1", "y0", "y1", "truth"} <= set(queries):
        raise ValueError("Queries require qId, bounds and offline truth")
    values = queries[["qId", "x0", "x1", "y0", "y1", "truth"]].to_numpy(float)
    if (not np.isfinite(values).all() or queries.qId.duplicated().any()
            or not np.equal(values[:, 0], np.floor(values[:, 0])).all()
            or (values[:, 1] >= values[:, 2]).any() or (values[:, 3] >= values[:, 4]).any()
            or (values[:, 5] < 0).any() or not np.equal(values[:, 5], np.floor(values[:, 5])).all()):
        raise ValueError("Queries require unique integer IDs, finite positive-area bounds and nonnegative integer truth")
    cells, _ = source_cell_ids(queries, histogram_dir)
    qids = np.sort(queries.loc[queries.truth >= min_exact, "qId"].to_numpy(dtype=np.int64))
    available = cells.loc[qids]
    groups = np.unique(available.to_numpy())
    if len(groups) < 5:
        raise ValueError("At least five cell groups are required for nonempty 50/20/30 roles")
    rows = []
    for seed in seeds:
        shuffled = np.random.default_rng(seed).permutation(groups)
        n = len(shuffled)
        for role, selected in zip(ROLES, (shuffled[:n//2], shuffled[n//2:int(.7*n)], shuffled[int(.7*n):])):
            ids = available.index[available.isin(selected)].to_numpy()
            rows.extend(dict(split_kind="cell", seed=seed, qId=int(qid), split=role,
                             cell=str(cells.loc[qid])) for qid in ids)
    result = pd.DataFrame(rows, columns=["split_kind", "seed", "qId", "split", "cell"])
    validate_grouped_memberships(result, queries, seeds, min_exact)
    return result


def validate_grouped_memberships(frame, queries, seeds=(0, 1, 2), min_exact=10):
    """Reject missing/overlapping roles rather than repairing from outcomes."""
    seeds = _seeds(seeds)
    if not {"seed", "qId", "split", "cell"} <= set(frame):
        raise ValueError("Grouped split requires seed, qId, split and cell")
    if "split_kind" in frame and not frame.split_kind.astype(str).eq("cell").all():
        raise ValueError("Split must explicitly be grouped-cell, not ordinary query roles")
    if (frame.duplicated(["seed", "qId"]).any() or frame.cell.isna().any()
            or frame.cell.astype(str).str.strip().eq("").any()):
        raise ValueError("Grouped split has duplicate IDs or missing cell groups")
    expected = set(queries.loc[queries.truth >= min_exact, "qId"])
    for seed in seeds:
        part = frame[frame.seed == seed]
        if set(part.qId) != expected or set(part.split) != set(ROLES):
            raise ValueError("Grouped split must cover every eligible qId with three roles per seed")
        groups = [set(part.loc[part.split == role, "cell"]) for role in ROLES]
        if any(groups[i] & groups[j] for i in range(3) for j in range(i)):
            raise ValueError("Cell groups overlap between train/calibration/test")
    return frame


def _new_output(root):
    path = Path(root)/("R10-input-splits-"+datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")+"-"+uuid4().hex[:8])
    path.mkdir(parents=True, exist_ok=False)
    return path


def _historical_identity(query_path, source_paths, seeds, min_exact):
    record_path = Path(__file__).with_name("shape_split_recipe.json")
    record = json.loads(record_path.read_text())
    actual = dict(queries_sha256=prior._sha(query_path),
                  source_sha256={p.name: prior._sha(p) for p in source_paths})
    expected = record["historical_authors_r10"]
    matches = (actual["queries_sha256"] == expected["queries_sha256"]
               and actual["source_sha256"] == expected["source_sha256"]
               and seeds == expected["seeds"] and min_exact == expected["min_exact"])
    return actual, record, matches


def prepare_grouped_split(config, *, policy="auto", expected_sha256=""):
    """Return ``(input_config, audit_manifest_path)`` for ``prepare_inputs``.

    ``auto`` reuses an available valid cell split, otherwise generates the
    declared recipe. An explicitly supplied non-cell split is never discarded
    unless ``policy='generate'`` is selected. ``inherit`` forbids generation.
    Every generated cohort records identities and whether it exactly reproduces
    the known AUTHORS roles. No existing result or supplied file is overwritten.
    """
    if policy not in {"auto", "generate", "inherit"}:
        raise ValueError("split policy must be auto, generate or inherit")
    supplied = dict(config)
    cfg = shape_inputs._config(supplied)
    seeds, min_exact = _seeds(cfg["seeds"]), int(cfg["min_exact"])
    if cfg.get("preparation_dir"):
        existing = Path(cfg["preparation_dir"])/"outer_splits.csv"
        if expected_sha256 and prior._sha(existing) != expected_sha256:
            raise ValueError("Prepared grouped split SHA256 differs from expected")
        return supplied, None
    hist_dir, source_paths, _, _, query_path, inherited = prior._resolve_inputs(cfg)
    queries = prior._load_queries(query_path, cfg.get("query_metadata_file", ""))
    existing, reason = None, "no inherited grouped split found"
    if inherited is not None and policy != "generate":
        candidate = pd.read_csv(inherited)
        advertised_grouped = ("split_kind" in candidate and "cell" in set(candidate.split_kind.astype(str)))
        advertised_grouped |= "cell" in candidate and "split_kind" not in candidate
        if "seed" not in candidate and len(seeds) == 1:
            candidate["seed"] = seeds[0]
        if "split_kind" in candidate and "cell" in set(candidate.split_kind.astype(str)):
            candidate = candidate[candidate.split_kind.astype(str) == "cell"].copy()
        candidate = candidate[candidate.seed.isin(seeds)] if "seed" in candidate else candidate
        try:
            validate_grouped_memberships(candidate, queries, seeds, min_exact)
            existing = inherited
        except ValueError:
            if advertised_grouped or supplied.get("split_file") or policy == "inherit":
                raise ValueError("Supplied/inherited split is not valid grouped-cell roles; explicitly choose split policy=generate to replace it")
            reason = "inherited ordinary query split is not grouped-cell; declared recipe generated instead"
    elif policy == "inherit" and inherited is None:
        raise FileNotFoundError("No inherited grouped-cell split; supply split_file or explicitly generate it")
    actual_identity, recipe, historical_inputs = _historical_identity(query_path, source_paths, seeds, min_exact)
    destination = _new_output(cfg["output_root"])
    if existing is not None:
        split_path = existing
        frame = pd.read_csv(split_path)
        action = "reused_existing_grouped_cell_split"
    else:
        frame = grouped_memberships(queries, hist_dir, seeds, min_exact)
        split_path = destination/"outer_splits.csv"
        frame.to_csv(split_path, index=False, lineterminator="\n")
        action = "generated_declared_grouped_cell_split"
    digest = prior._sha(split_path)
    if expected_sha256 and digest != expected_sha256:
        raise ValueError("Grouped split SHA256 differs from expected; no altered memberships are accepted")
    historical_match = historical_inputs and digest == recipe["historical_authors_r10"]["outer_splits_sha256"]
    if historical_inputs and action.startswith("generated") and not historical_match:
        raise ValueError("Known AUTHORS recipe failed byte-identical historical membership reproduction")
    manifest = dict(status="complete", recipe=RECIPE, action=action, decision_reason=reason,
        policy=policy, seeds=seeds, min_exact=min_exact,
        cohort_scope="truth>=min_exact is offline evaluation eligibility only, not an online filter",
        grouped_split_path=str(split_path.resolve()), grouped_split_sha256=digest,
        source_inputs=actual_identity, queries_path=str(query_path), histogram_dir=str(hist_dir),
        total_queries=len(queries), eligible_queries=int((queries.truth >= min_exact).sum()),
        excluded_queries=int((queries.truth < min_exact).sum()),
        historical_authors_input_identity=historical_inputs,
        historical_authors_byte_identical=bool(historical_match),
        interpretation="historical AUTHORS R10 roles reproduced byte-for-byte" if historical_match else
                       "declared grouped cohort; not a reproduction claim for historical R10 measurements",
        implementation_sha256=prior._sha(__file__), recipe_provenance_sha256=prior._sha(Path(__file__).with_name("shape_split_recipe.json")),
        grouping_uses_labels=False, role_selection_uses_labels=False,
        eligibility_uses_offline_truth=True, existing_results_overwritten=False)
    manifest_path = destination/"split_manifest.json"
    prior._json(manifest_path, manifest)
    result = {**supplied, "histogram_dir": str(hist_dir), "queries_file": str(query_path),
              "split_file": str(split_path), "split_kind": "cell"}
    print("Grouped split:", split_path, "|", manifest["interpretation"], flush=True)
    return result, manifest_path
