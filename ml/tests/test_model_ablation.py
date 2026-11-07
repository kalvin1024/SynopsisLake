"""Local correctness checks for the personal-workspace model ablation runner.

The integration fixture exports the same files as ErrorPredictionLogExp, using
weighted points generated here. No Databricks connection or team data is needed.
Run from the checkout with ``python -m unittest discover -s ml/tests -v``.
"""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import lightgbm as lgb
import numpy as np
import pandas as pd


ML_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ML_ROOT))

from features_v1 import build_table_v1  # noqa: E402
from gate_eval import split_ids  # noqa: E402
from model_ablation import (  # noqa: E402
    evaluate_scores,
    make_split_manifest,
    run_experiment,
    validate_output_root,
    variants,
)


def _integrate(rectangle, edges, counts):
    """Independent uniform-cell rectangle integral for synthetic logger exports."""
    xmin, xmax, ymin, ymax = rectangle
    widths = np.diff(edges)
    fx = np.maximum(0, np.minimum(xmax, edges[1:]) - np.maximum(xmin, edges[:-1])) / widths
    fy = np.maximum(0, np.minimum(ymax, edges[1:]) - np.maximum(ymin, edges[:-1])) / widths
    overlap = np.outer(fx, fy)
    return float(np.sum(overlap * counts)), overlap


def _write_histogram(path, rate, edges, counts):
    lines = [str(rate), ",".join(map(str, edges)), ",".join(map(str, edges))]
    lines.extend(",".join(map(str, row)) for row in counts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _tree_digests(*roots):
    return {
        str(path): hashlib.sha256(path.read_bytes()).hexdigest()
        for root in roots
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class ExportFixture:
    """160 distinct queries, two source histograms, and two merged resolutions."""

    def __init__(self, root):
        self.root = root
        self.data_root = root / "shared_data"
        self.sample_root = root / "shared_samples"
        self.dataset = "synthetic"
        self.data_dir = self.data_root / self.dataset
        self.sample_dir = self.sample_root / self.dataset
        self.output_root = root / "personal_results"
        self.data_dir.mkdir(parents=True)
        self.sample_dir.mkdir(parents=True)

        axis = np.linspace(1 / 24, 2 - 1 / 24, 24)
        xx, yy = np.meshgrid(axis, axis, indexing="ij")
        points = np.column_stack((xx.ravel(), yy.ravel()))
        source0 = np.full((24, 24), 4.0)
        source0[3:8, 3:8] = 120
        source0[15:20, 16:22] = 25
        source1 = np.full((24, 24), 3.0)
        source1[9:14, 12:18] = 100
        source1[17:22, 1:6] = 30
        weights = (source0 + source1).ravel()

        sources = []
        for index, (resolution, source) in enumerate(((2, source0), (3, source1))):
            edges = np.linspace(0, 2, resolution + 1)
            counts = np.histogram2d(points[:, 0], points[:, 1], bins=edges, weights=source.ravel())[0]
            _write_histogram(self.sample_dir / f"src_{index}.txt", 1.0, edges, counts)
            sources.append((edges, counts))

        merged = []
        for resolution, rate in ((2, 0.25), (4, 1.0)):
            edges = np.linspace(0, 2, resolution + 1)
            counts = np.histogram2d(points[:, 0], points[:, 1], bins=edges, weights=weights)[0]
            _write_histogram(self.sample_dir / f"arm_{resolution}.txt", rate, edges, counts)
            merged.append((resolution, rate, edges, counts))

        random = np.random.default_rng(741)
        queries, metadata, rows = [], [], []
        for qid in range(160):
            widths = random.uniform(0.45, 1.25, size=2)
            starts = random.uniform(0, 2 - widths)
            q = (starts[0], starts[0] + widths[0], starts[1], starts[1] + widths[1])
            inside = ((points[:, 0] >= q[0]) & (points[:, 0] <= q[1])
                      & (points[:, 1] >= q[2]) & (points[:, 1] <= q[3]))
            exact = float(weights[inside].sum())
            queries.append((*q, exact))
            metadata.append({"qId": qid, "scale": qid % 3 + 1,
                             "width": widths[0], "height": widths[1]})
            source_estimate = sum(_integrate(q, edges, counts)[0] for edges, counts in sources)
            for resolution, rate, edges, counts in merged:
                estimate, overlap = _integrate(q, edges, counts)
                bound = float(counts[overlap > 0].sum() - counts[overlap >= 1 - 1e-12].sum())
                rows.append({
                    "method": "ARM", "mergeRate": rate, "lonRes": resolution, "latRes": resolution,
                    "qId": qid, "gt": exact, "est": estimate,
                    "relErr": abs(estimate - exact) / exact,
                    "predErr": bound / max(estimate, 1.0), "betaAlpha": 1.5 + resolution,
                    "overlapSum": overlap.sum(), "cellsTouched": int((overlap > 0).sum()),
                    "srcSumEst": source_estimate,
                })
        pd.DataFrame(queries).to_csv(self.data_dir / "queries-Ans.txt", index=False, header=False)
        pd.DataFrame(metadata).to_csv(self.data_dir / "queries-meta.csv", index=False)
        pd.DataFrame(rows).to_csv(self.sample_dir / "rows.csv", index=False)

    def config(self):
        return {
            "repo_root": str(ML_ROOT.parent), "data_root": str(self.data_root),
            "sample_root": str(self.sample_root), "output_root": str(self.output_root),
            "dataset": self.dataset, "variants": "v0,v1", "seeds": "0", "epsilons": "0.25",
            "delta": "0.05", "copy_inputs": "true", "mlflow_enabled": "false",
            "generalization": "false", "max_trees": "8", "num_threads": "2",
        }


class GateEvaluationTests(unittest.TestCase):
    @staticmethod
    def frames():
        frame = pd.DataFrame({
            "qId": np.arange(8), "mergeRate": np.repeat([0.25, 1.0], 4),
            "relErr": np.tile([0.1, 0.1, 0.5, 0.4], 2),
        })
        score = np.tile([0.1, 0.1, 0.2, 0.3], 2)
        return frame, score

    def test_thresholds_use_calibration_and_test_truth_only_changes_quality(self):
        calibration, score = self.frames()
        test = calibration.copy()
        test.qId += 100
        _, original = evaluate_scores(calibration, test, {"learned": score}, {"learned": score})
        changed_test = test.copy()
        changed_test.loc[score <= 0.1, "relErr"] = 0.9
        _, changed = evaluate_scores(calibration, changed_test, {"learned": score}, {"learned": score})

        keys = ["score", "eps", "scope", "mergeRate"]
        original = original.sort_values(keys).reset_index(drop=True)
        changed = changed.sort_values(keys).reset_index(drop=True)
        pd.testing.assert_frame_equal(original[keys + ["cutoff", "accept_none", "accepted_count"]],
                                      changed[keys + ["cutoff", "accept_none", "accepted_count"]])
        self.assertTrue((original.violation_count == 0).all())
        self.assertTrue((changed.violation_count == changed.accepted_count).all())
        self.assertTrue((changed.violation == 1).all())
        self.assertTrue((~changed.quality_target_met.astype(bool)).all())

    def test_reject_all_has_no_defined_risk_or_success_claim(self):
        calibration, score = self.frames()
        calibration["relErr"] = 0.9
        test, _ = self.frames()
        _, details = evaluate_scores(calibration, test, {"learned": score}, {"learned": score})
        self.assertTrue(details.accept_none.all())
        self.assertTrue(details.cutoff.isna().all())
        self.assertTrue((details.accepted_count == 0).all())
        self.assertTrue((details.violation_count == 0).all())
        self.assertTrue(details.violation.isna().all())
        self.assertTrue((details.coverage == 0).all())
        self.assertTrue(all(pd.isna(value) or not bool(value) for value in details.quality_target_met))

    def test_calibration_never_accepts_only_good_member_of_tied_scores(self):
        calibration = pd.DataFrame({"qId": [0, 1, 2], "mergeRate": [1.0] * 3,
                                    "relErr": [0.1, 0.9, 0.9]})
        score = np.array([0.1, 0.1, 0.3])
        _, details = evaluate_scores(calibration, calibration.copy(), {"learned": score}, {"learned": score})
        self.assertTrue(details.accept_none.all())
        self.assertTrue((details.accepted_count == 0).all())

    def test_one_class_auc_does_not_prevent_gate_reporting(self):
        calibration, score = self.frames()
        test = calibration.assign(relErr=0.1)
        summary, details = evaluate_scores(calibration, test, {"learned": score}, {"learned": score})
        self.assertGreater(len(summary), 0)
        self.assertTrue((details.accepted_count > 0).all())
        self.assertTrue((details.violation == 0).all())
        for column in [name for name in summary if "auc" in name.lower()]:
            self.assertTrue(summary[column].isna().all())


class SplitAndOutputTests(unittest.TestCase):
    def test_query_split_is_disjoint_across_rates_and_records_early_stopping(self):
        frame = pd.DataFrame({"qId": np.tile(np.arange(160), 2),
                              "mergeRate": np.repeat([0.25, 1.0], 160)})
        manifest = make_split_manifest(frame, 7)
        self.assertEqual(len(manifest), 160)
        self.assertFalse(manifest.qId.duplicated().any())
        tr, ca, te = split_ids(frame.qId, 7)
        for name, expected in (("train", tr), ("calibration", ca), ("test", te)):
            self.assertEqual(set(manifest.loc[manifest.split == name, "qId"]), expected)
        held_back = manifest.loc[manifest.early_stopping, "qId"]
        self.assertEqual(set(held_back), set(sorted(tr)[::10]))
        self.assertTrue((manifest.loc[manifest.early_stopping, "split"] == "train").all())
        pd.testing.assert_frame_equal(manifest, make_split_manifest(frame.sample(frac=1, random_state=9), 7))

    def test_outputs_must_be_outside_checkout_and_input_trees(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            repo, data, sample = (root / name for name in ("checkout", "data", "samples"))
            for path in (repo, data, sample):
                path.mkdir()
            safe = root / "personal_results"
            self.assertEqual(validate_output_root(str(safe), str(repo), str(data), str(sample)), safe.resolve())
            for unsafe in (repo, repo / "results", data, data / "results", sample, sample / "results",
                           Path("/Workspace/Shared/model-ablation")):
                with self.subTest(path=unsafe), self.assertRaises(ValueError):
                    validate_output_root(str(unsafe), str(repo), str(data), str(sample))
            (root / "linked_data").symlink_to(data, target_is_directory=True)
            with self.assertRaises(ValueError):
                validate_output_root(str(root / "linked_data/results"), str(repo), str(data), str(sample))


class TrainingArtifactTests(unittest.TestCase):
    def test_notebook_object_without_py_file_does_not_block_training(self):
        import model_ablation
        original_hash = model_ablation._hash

        def hash_with_workspace_notebook(path):
            if str(path).endswith("notebooks/06_model_v1.py"):
                raise FileNotFoundError("Databricks notebook object, not a regular .py file")
            return original_hash(path)

        with tempfile.TemporaryDirectory() as temporary:
            fixture = ExportFixture(Path(temporary))
            config = fixture.config()
            config["variants"] = "v0"
            with patch.object(model_ablation, "_hash", side_effect=hash_with_workspace_notebook):
                result = run_experiment(config)
            metadata = json.loads((result / "run.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["notebook_hash_unavailable"], "FileNotFoundError")
            self.assertEqual(len(metadata["code_sha256"]), 4)

    def test_actual_exports_train_reload_and_preserve_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ExportFixture(Path(temporary))
            before = _tree_digests(fixture.data_root, fixture.sample_root)
            run_dir = Path(run_experiment(fixture.config()))
            self.assertEqual(before, _tree_digests(fixture.data_root, fixture.sample_root))
            self.assertTrue(run_dir.is_relative_to(fixture.output_root))
            for name in ("v1_ablations.csv", "gate_details.csv", "splits.csv", "input_manifest.json", "run.json"):
                self.assertTrue((run_dir / name).is_file(), name)
            run_metadata = json.loads((run_dir / "run.json").read_text())
            self.assertIn(run_metadata["status"], ("complete", "completed"))
            input_manifest = json.loads((run_dir / "input_manifest.json").read_text())
            self.assertIn("rows.csv", json.dumps(input_manifest))
            self.assertIn("queries-Ans.txt", json.dumps(input_manifest))

            expected_split = make_split_manifest(pd.DataFrame({"qId": np.arange(160)}), 0)
            saved_split = pd.read_csv(run_dir / "splits.csv")
            pd.testing.assert_frame_equal(saved_split[["seed", "qId", "split", "early_stopping"]]
                                          .sort_values("qId").reset_index(drop=True),
                                          expected_split[["seed", "qId", "split", "early_stopping"]]
                                          .sort_values("qId").reset_index(drop=True))

            features = build_table_v1(str(fixture.sample_dir), str(fixture.data_dir))
            test_ids = set(expected_split.loc[expected_split.split == "test", "qId"])
            saved_queries = []
            for variant in ("v0", "v1"):
                artifact_dir = run_dir / "models" / variant / "seed_0"
                schema = json.loads((artifact_dir / "schema.json").read_text())
                self.assertEqual(schema["features"], variants()[variant])
                booster = lgb.Booster(model_file=str(artifact_dir / "model.txt"))
                predictions = pd.read_csv(artifact_dir / "test_predictions.csv")
                self.assertEqual(set(predictions.qId), test_ids)
                self.assertFalse(predictions.duplicated(["qId", "mergeRate"]).any())
                selected = predictions[["qId", "mergeRate"]].merge(features, on=["qId", "mergeRate"],
                                                                       how="left", validate="one_to_one")
                np.testing.assert_allclose(booster.predict(selected[schema["features"]]), predictions["score"],
                                           rtol=1e-10, atol=1e-10)
                cutoffs = json.loads((artifact_dir / "cutoffs.json").read_text(),
                                     parse_constant=lambda value: self.fail(f"Non-standard JSON value: {value}"))
                self.assertTrue(cutoffs)
                saved_queries.append(predictions[["qId", "mergeRate"]].sort_values(["qId", "mergeRate"])
                                     .reset_index(drop=True))
            pd.testing.assert_frame_equal(*saved_queries)

            previous_artifacts = _tree_digests(run_dir)
            second_config = fixture.config()
            second_config["variants"] = "v0"
            second_dir = Path(run_experiment(second_config))
            self.assertNotEqual(run_dir, second_dir)
            self.assertEqual(previous_artifacts, _tree_digests(run_dir))
            self.assertEqual(before, _tree_digests(fixture.data_root, fixture.sample_root))

    def test_missing_merged_export_fails_without_completed_model(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ExportFixture(Path(temporary))
            (fixture.sample_dir / "arm_2.txt").unlink()
            with self.assertRaisesRegex((FileNotFoundError, ValueError), "arm_2"):
                run_experiment(fixture.config())
            self.assertEqual(list(fixture.output_root.rglob("model.txt")), [])


if __name__ == "__main__":
    unittest.main()
