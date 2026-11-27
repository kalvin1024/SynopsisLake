"""Validate the previous region run before selecting its disjoint follow-up workload."""

import csv
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile

import numpy as np


def _fresh_test_ids(members, previous_ids, count):
    """Draw exactly count whole-east queries after excluding the old sample."""
    from .region_budget_comparison import _select

    old = np.asarray(previous_ids)
    east = members.loc[members.region.eq("east"), "qId"].to_numpy(int)
    if (count < 1 or old.ndim != 1 or old.dtype.kind not in "iu" or len(old) != count
            or len(np.unique(old)) != count or not np.isin(old, east).all()):
        raise ValueError(f"Previous test IDs must be {count} distinct whole-east query IDs")
    available = east[~np.isin(east, old)]
    if len(available) < count:
        raise ValueError(f"Need {count} unused whole-east queries; only {len(available)} remain")
    return _select(available, count, 90127)


def _id_file(archive, member):
    with archive.open(member) as stream:
        rows = list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8")))
    if any(int(row["qId"]) != i for i, row in enumerate(rows)):
        raise ValueError(f"Non-sequential batch query IDs in {member}")
    return np.array([int(row["source_qId"]) for row in rows], dtype=int)


def _read_csv(archive, member):
    with archive.open(member) as stream:
        return list(csv.DictReader(io.TextIOWrapper(stream, encoding="utf-8")))


def load_prior(c, inputs):
    """Check identical inputs, then select fresh test IDs without using east labels.

    The source query file is fingerprinted as bytes. Only its four geometry
    columns are interpreted here. Exact test labels in the ZIP are never opened.
    """
    from . import region_budget_comparison as runner

    path = Path(c["prior_results_zip"]).expanduser().resolve()
    archive_sha = runner._sha(path)
    checked = {}
    with ZipFile(path) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        if manifest.get("status") != "complete" or manifest.get("no_target_label_tuning") is not True:
            raise ValueError("The prior ZIP must contain a complete held-out-region run")
        old_config = manifest["config"]
        for key in ("seeds", "epsilon", "delta", "merge_rate", "repeats", "n_test_queries",
                    "calibration_fraction", "n_calibration_queries", "chunk_size"):
            if c[key] != old_config[key]:
                raise ValueError(f"Matched run {key} must equal the prior setting {old_config[key]!r}")

        for item in inputs:
            name = item["dataset"]
            if name not in manifest["measured_datasets"]:
                raise ValueError(f"{name} was not measured in the prior ZIP")
            prep = json.loads(archive.read(f"{name}/preparation.json"))
            lock_bytes = archive.read(f"{name}/protocol.json")
            lock = json.loads(lock_bytes)
            result = json.loads(archive.read(f"{name}/result_manifest.json"))
            if (result["status"] != "complete" or result["exact_truth_parity"] is not True
                    or result["actual_east_query_count"] != c["n_test_queries"]
                    or result["protocol_sha256"] != hashlib.sha256(lock_bytes).hexdigest()):
                raise ValueError(f"{name}: prior evaluation or frozen protocol is incomplete")
            if (prep["strict_whole_rectangle"] is not True
                    or lock["calibration_labels"] != "west only"
                    or any(lock[key] is not False for key in (
                        "target_region_labels_used_for_training", "target_region_labels_used_for_budget",
                        "target_region_labels_used_for_calibration"))):
                raise ValueError(f"{name}: prior does not use the required west-only protocol")

            cut = prep["region_cut"]
            if item.get("region_cut") is not None and item["region_cut"] != cut:
                raise ValueError(f"{name}: region cut must match the prior cut {cut}")
            if (item["actual_rate"] != prep["actual_merge_rate"]
                    or runner._sha(item["H_path"]) != prep["original_H_sha256"]):
                raise ValueError(f"{name}: merged H differs from the prior run")
            if runner._sha(item["queries_file"]) != prep["query_file_sha256"]:
                raise ValueError(f"{name}: query file differs from the prior run; query IDs may have changed")
            source_hashes = {p.name: runner._sha(p) for p in item["sources"]}
            if source_hashes != prep["source_histogram_sha256"]:
                raise ValueError(f"{name}: source histograms differ from the prior run")

            geometry = runner._read_geometry(item["queries_file"])
            members, cut = runner.region_memberships(geometry, cut)
            old_ids = _id_file(archive, f"{name}/test_queries_geometry_only.csv.ids.csv")
            new_ids = _fresh_test_ids(members, old_ids, c["n_test_queries"])
            east_ids = members.loc[members.region.eq("east"), "qId"]
            if not np.array_equal(old_ids, runner._select(east_ids, c["n_test_queries"], 90127)):
                raise ValueError(f"{name}: prior test IDs differ from the original deterministic selection")
            for seed in c["seeds"]:
                roles = runner.source_roles(members, seed, c["calibration_fraction"])
                cal_ids = runner._select(roles.loc[roles.split.eq("calibration"), "qId"],
                                         c["n_calibration_queries"], seed+913)
                if not np.array_equal(cal_ids, _id_file(archive, f"{name}/calibration_seed_{seed}.csv.ids.csv")):
                    raise ValueError(f"{name}: prior western calibration IDs differ for seed {seed}")

            budget = c["dataset_budget_bytes"][name]
            costs = [row for row in _read_csv(archive, f"{name}/payload_costs.csv")
                     if row["method"] == "source_shape"]
            if (sorted(int(row["seed"]) for row in costs) != sorted(c["seeds"])
                    or any(int(row["used_bytes"]) != budget for row in costs)):
                raise ValueError(f"{name}: budget {budget} must match the prior Source Shape bytes")

            records = [row for row in lock["payload_files"] if row["method"] == "source_shape"]
            h_hashes = {row["sha256"] for row in records if row["component"] == "histogram_bytes"}
            if len(h_hashes) != 1:
                raise ValueError(f"{name}: prior lock must identify one serialized H")
            shape = lock["shape_metadata"]
            shape_hashes = {p: sha for p, sha in shape["artifact_sha256"].items()
                            if p == "shape.bin" or p.startswith("native/")}
            raw_hashes = {row["source_id"]: row["raw_sha256"] for row in shape["source_inputs"]}
            if ("shape.bin" not in shape_hashes or not any(p.startswith("native/") for p in shape_hashes)
                    or set(raw_hashes) != set(range(len(item["raw_paths"])))):
                raise ValueError(f"{name}: prior lock is missing Source Shape or raw input fingerprints")
            checked[name] = dict(previous_ids=old_ids, test_ids=new_ids, region_cut=cut,
                                 actual_rate=item["actual_rate"], budget=budget,
                                 h_sha256=h_hashes.pop(), shape_artifact_sha256=shape_hashes,
                                 shape_packet_budget=shape["packet_budget_bytes"],
                                 raw_sha256=raw_hashes, archive_sha256=archive_sha)
    return checked
