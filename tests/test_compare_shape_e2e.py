"""Correctness checks for the shared Java comparison, not performance evidence.

The optional integration tests compile the current Java query entry points and
use a tiny raw dataset. They never train or substitute a production model.
Set COMMONS_MATH3_JAR and put a JDK (java and javac) on PATH to run them.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

import numpy as np
import pandas as pd

from ml import compare_shape_e2e as comparison


ROOT = Path(__file__).resolve().parents[1]


def _constant_model():
    # Two identical leaves provide a genuine parseable regression model. Its
    # identity score is deliberately uninformative, so calibration must reject
    # it when every calibration H is bad; it is not an ML quality experiment.
    return (
        "tree\nversion=v4\nnum_class=1\nnum_tree_per_iteration=1\nlabel_index=0\n"
        "max_feature_idx=0\nobjective=regression\nfeature_names=area_cells\n\n"
        "Tree=0\nnum_leaves=2\nnum_cat=0\nsplit_feature=0\nthreshold=0.5\n"
        "decision_type=2\nleft_child=-1\nright_child=-2\nleaf_value=0.10000000000000002 0.10000000000000002\n"
        "is_linear=0\n\nend of trees\n"
    )


def _write_histogram(path, rate, bounds, counts):
    lines = [repr(rate), ",".join(map(str, bounds)), ",".join(map(str, bounds))]
    lines += [",".join(map(str, row)) for row in counts]
    Path(path).write_text("\n".join(lines) + "\n")


def _fixture(root):
    run, raw, dataset = (root / n for n in ("run", "raw", "dataset"))
    for d in (run, raw, dataset):
        d.mkdir()
    masses = np.array([[40, 60], [80, 100]], dtype=int)
    points = np.concatenate([
        np.repeat([[i + .1, j + .1]], masses[i, j], axis=0)
        for i in range(2) for j in range(2)
    ])
    np.savetxt(raw / "0.csv", points, delimiter="\t", fmt="%.17g")
    rate, coarse_rate = .190123456789, .091234567891
    _write_histogram(run / "src_0.txt", 1., [0., 1., 2.], masses)
    _write_histogram(run / "arm_2.txt", rate, [0., 1., 2.], masses)
    _write_histogram(run / "arm_1.txt", coarse_rate, [0., 2.], [[masses.sum()]])
    queries = []
    for qid in range(40):
        i, j = (qid % 4) // 2, qid % 2
        shift = (qid // 4) * .001
        bounds = [i + .02 + shift, i + .5 + shift,
                  j + .02 + shift, j + .5 + shift]
        truth = int(((points[:, 0] >= bounds[0]) & (points[:, 0] <= bounds[1]) &
                     (points[:, 1] >= bounds[2]) & (points[:, 1] <= bounds[3])).sum())
        queries.append([*bounds, truth])
    pd.DataFrame(queries).to_csv(dataset / "queries-Ans.txt", header=False, index=False)
    # Only these logger fields are needed to inherit the eligible query IDs.
    pd.concat([pd.DataFrame({"method": "ARM", "mergeRate": r,
                            "qId": np.arange(len(queries)), "gt": np.array(queries)[:, 4]})
               for r in (rate, coarse_rate)], ignore_index=True).to_csv(run / "rows.csv", index=False)
    for name in ("v1", "v1_no_skew"):
        d = run / "models" / name
        d.mkdir(parents=True)
        (d / "model.txt").write_text(_constant_model())
        (d / "model.properties").write_text("label.transform=identity\ngate.cutoff=123\n")
        # These original cutoffs must not decide the new shared comparison.
        (d / "cutoffs.csv").write_text(
            f"mergeRate,eps,delta,cutoff\n{rate!r},0.25,0.05,123\n")
    return dict(run=run, raw=raw, dataset=dataset, points=points, rate=rate, coarse_rate=coarse_rate)


class ComparisonMathTests(unittest.TestCase):
    def test_missing_dataset_cannot_silently_select_authors(self):
        with self.assertRaisesRegex(ValueError, "dataset explicitly"):
            comparison._config({})

    def test_missing_properties_requires_matching_export_without_creating_one(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            folder = root / "models" / "v1"
            folder.mkdir(parents=True)
            (folder / "model.txt").write_text(_constant_model())
            config = dict(dataset="europe", run_dir=str(root), models_root=str(root / "models"), models=["v1"])
            with self.assertRaisesRegex(FileNotFoundError, "Check these settings.*placeholder"):
                comparison._model_dirs(config)
            self.assertFalse((folder / "model.properties").exists())

    def test_zero_truth_policy_keeps_positive_estimate_infinite(self):
        value = comparison.relative_error(np.array([0., 1., 12.]), np.array([0., 0., 10.]))
        np.testing.assert_equal(value[:2], np.array([0., np.inf]))
        self.assertAlmostEqual(value[2], .2)

    def test_calibration_respects_ties_and_can_recover_after_bad_prefix(self):
        score = np.array([0., 0., 1., 2., 3.])
        err = np.array([.3, 0., 0., 0., 0.])
        # A threshold cannot select just the good member of the score-zero tie.
        self.assertEqual(comparison.calibrated_cutoff(score[:2], err[:2], .25, .05), -np.inf)
        # A bad prefix is not a reason to stop searching the calibration data.
        self.assertEqual(comparison.calibrated_cutoff(score, err, .25, .2), 3.)

    def test_no_acceptance_is_undefined_risk(self):
        frame = pd.DataFrame({
            "qId": [0, 1], "source_qId": [10, 20],
            "answer": [40., 80.], "fromSynopsis": [False, False],
            "errorScore": [.1, .1], "cutoff": [-np.inf, -np.inf],
            "gateNs": [20, 30], "synNs": [10, 15], "rawNs": [100, 110],
        })
        result = comparison.summarize_answers(frame, np.array([10., 20.]), np.array([40., 80.]), .25)
        self.assertEqual(result["coverage"], 0.)
        self.assertTrue(np.isnan(result["violation"]))

    def test_answer_validation_catches_truth_table_substitution(self):
        frame = pd.DataFrame({
            "qId": [0, 1], "source_qId": [10, 20],
            "answer": [40., 80.], "fromSynopsis": [False, False],
            "errorScore": [.1, .1], "cutoff": [-np.inf, -np.inf],
            "gateNs": [20, 30], "synNs": [10, 15], "rawNs": [100, 110],
        })
        comparison.validate_answers(frame, np.array([10., 20.]), np.array([40., 80.]), -np.inf)
        wrong = frame.copy()
        wrong.loc[0, "answer"] = 10.
        with self.assertRaises((ValueError, AssertionError)):
            comparison.validate_answers(wrong, np.array([10., 20.]), np.array([40., 80.]), -np.inf)
        no_scan = frame.copy()
        no_scan.loc[0, "rawNs"] = 0
        with self.assertRaises((ValueError, AssertionError)):
            comparison.validate_answers(no_scan, np.array([10., 20.]), np.array([40., 80.]), -np.inf)

    def test_accepted_answers_preserve_H_and_report_their_actual_risk(self):
        frame = pd.DataFrame({
            "qId": [0, 1], "source_qId": [10, 20],
            "answer": [40., 80.], "fromSynopsis": [True, False],
            "errorScore": [.1, .4], "cutoff": [.2, .2],
            "gateNs": [20, 30], "synNs": [10, 15], "rawNs": [0, 110],
        })
        H, truth = np.array([40., 20.]), np.array([100., 80.])
        comparison.validate_answers(frame, H, truth, .2)
        result = comparison.summarize_answers(frame, H, truth, .25)
        self.assertEqual(result["coverage"], .5)
        self.assertEqual(result["violation"], 1.)
        self.assertAlmostEqual(result["mean_error"], .3)
        substituted = frame.copy()
        substituted.loc[0, "answer"] = truth[0]
        with self.assertRaises((ValueError, AssertionError)):
            comparison.validate_answers(substituted, H, truth, .2)


def _math_jar():
    supplied = os.environ.get("COMMONS_MATH3_JAR")
    if supplied and Path(supplied).is_file():
        return Path(supplied)
    m2 = Path.home() / ".m2/repository/org/apache/commons/commons-math3"
    found = sorted(m2.glob("*/commons-math3-*.jar"))
    return found[-1] if found else None


@unittest.skipUnless(importlib.util.find_spec("pyarrow"), "pyarrow is required for real CSV-to-Parquet timing")
class SharedJavaComparisonTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.java, cls.javac, cls.math = shutil.which("java"), shutil.which("javac"), _math_jar()
        if not cls.java or not cls.javac or cls.math is None:
            raise unittest.SkipTest("Need java, javac and COMMONS_MATH3_JAR (or its Maven cache)")
        cls.compilation = tempfile.TemporaryDirectory(prefix="shape-java-comparison-tests-")
        classes = Path(cls.compilation.name) / "classes"
        classes.mkdir()
        source = ROOT / "SynopsisLake/src/main/java"
        command = [cls.javac, "-cp", str(cls.math), "-sourcepath", str(source), "-d", str(classes),
                   str(source / "sigspatial25/ApproxQueryExp.java"),
                   str(source / "sigspatial25/SourceShapeReferenceExp.java")]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            cls.compilation.cleanup()
            raise RuntimeError(f"Current Java comparison code failed compilation:\n{result.stderr}")
        cls.classpath = os.pathsep.join([str(classes), str(cls.math)])

    @classmethod
    def tearDownClass(cls):
        cls.compilation.cleanup()

    def config(self, root, fixture):
        return dict(
            repo_root=str(ROOT), dataset="test_fixture", dataset_dir=str(fixture["dataset"]),
            run_dir=str(fixture["run"]), data_dir=str(fixture["raw"]),
            models="v1,v1_no_skew", seed="0", model_training_seed="0", model_manifest="",
            eps="0.25", delta="0.05",
            merge_rates=",".join(map(repr, (fixture["rate"], fixture["coarse_rate"]))),
            n_queries="all", repeats="1",
            java=self.java, classpath=self.classpath, shape_enabled="true", shape_dir="",
            shape_packet_budget="4096", chunk_size="37", output_root=str(root / "results"),
        )

    def test_real_java_all_arms_share_H_gates_and_raw_fallback(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root)
            old = root / "results" / "prior-completed-run"
            old.mkdir(parents=True)
            (old / "summary.csv").write_text("preserve this completed run\n")
            original = {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
                        for p in fixture["run"].glob("models/*/*")}
            result = Path(comparison.run_comparison(self.config(root, fixture)))
            summary = pd.read_csv(result / "summary.csv")
            self.assertEqual(set(summary.setup), {"skewness", "v1", "v1_no_skew", "source_shape", "exact"})
            gated = summary[summary.setup != "exact"]
            np.testing.assert_allclose(sorted(gated.merge_rate.unique()),
                                       sorted([fixture["rate"], fixture["coarse_rate"]]), rtol=0, atol=1e-15)
            self.assertTrue((gated.coverage == 0).all())
            self.assertTrue(gated.violation.isna().all())
            cuts = pd.read_csv(result / "frozen_cutoffs.csv")
            self.assertTrue(np.isneginf(cuts.cutoff).all())
            saved = pd.read_csv(result / "per_query.csv")
            self.assertIn("source_qId", saved)
            self.assertTrue((saved.rawNs > 0).all())
            self.assertTrue((saved.answer > 0).all())
            # At least one output must differ from the original H, making a
            # synopsis-as-fallback implementation fail this test.
            self.assertTrue((saved.answer != saved.H).any())
            self.assertTrue((saved.answer == saved.truth).all())
            self.assertEqual(original, {p: hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in original})
            self.assertEqual((old / "summary.csv").read_text(), "preserve this completed run\n")
            self.assertTrue((result / "protocol.json").is_file())
            self.assertTrue((result / "09b_comparison_results.zip").is_file())
            query_files = [result / "calibration_queries.csv", result / "test_queries.csv"]
            self.assertTrue(all(path.is_file() for path in query_files))
            for path in query_files:
                self.assertEqual(pd.read_csv(path, header=None).shape[1], 4, str(path))

    def test_poisoned_saved_truth_cannot_supply_the_exact_answer(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root)
            path = fixture["dataset"] / "queries-Ans.txt"
            q = pd.read_csv(path, header=None)
            q[4] += 7
            q.to_csv(path, header=False, index=False)
            rows = pd.read_csv(fixture["run"] / "rows.csv")
            rows["gt"] = rows.qId.map(q[4])
            rows.to_csv(fixture["run"] / "rows.csv", index=False)
            with self.assertRaisesRegex((ValueError, AssertionError), "(?i)exact|truth|raw|count"):
                comparison.run_comparison(self.config(root, fixture))

    def test_duplicate_logger_rows_rejected_before_benchmark(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root)
            path = fixture["run"] / "rows.csv"
            rows = pd.read_csv(path)
            pd.concat([rows, rows.iloc[:1]], ignore_index=True).to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "(?i)duplicate|unique"):
                comparison.run_comparison(self.config(root, fixture))

    def test_real_java_acceptance_returns_H_without_a_raw_scan(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root)
            config = self.config(root, fixture)
            # This diagnostic tolerance makes every fixture H acceptable. The
            # saved model cutoffs contain only eps=.25, so a successful eps=1
            # run also demonstrates fresh common calibration for this limit.
            config["eps"] = "1.0"
            result = Path(comparison.run_comparison(config))
            rows = pd.read_csv(result / "per_query.csv")
            learned = rows[rows.setup.isin(["v1", "v1_no_skew"])]
            self.assertTrue(learned.fromSynopsis.all())
            self.assertTrue((learned.rawNs == 0).all())
            np.testing.assert_allclose(learned.answer, learned.H, rtol=0, atol=1e-8)
            self.assertTrue((learned.answer != learned.truth).all())

    def test_model_training_seed_must_match_the_inherited_split(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            fixture = _fixture(root)
            config = self.config(root, fixture)
            config["model_training_seed"] = "1"
            with self.assertRaisesRegex(ValueError, "(?i)seed"):
                comparison.run_comparison(config)

    def test_supplied_model_manifest_cannot_name_another_split_or_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model_dir = root / "model"
            model_dir.mkdir()
            model_path = model_dir / "model.txt"
            model_path.write_text(_constant_model())
            roles = pd.DataFrame({"source_qId": [3, 1, 2]})
            ids_hash = hashlib.sha256(json.dumps([1, 2, 3], separators=(",", ":")).encode()).hexdigest()
            manifest = dict(dataset="fixture", seed=0, split_kind="query", eligible_query_ids_sha256=ids_hash,
                            models={"v1": {"model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest()}})
            manifest_path = root / "model-manifest.json"
            manifest_path.write_text(json.dumps(manifest))
            config = dict(dataset="fixture", seed=0, model_training_seed=0, model_manifest=str(manifest_path))
            verified = comparison._model_provenance(config, {"v1": model_dir}, roles)
            self.assertEqual(verified["status"], "provided_manifest_matches")
            manifest["seed"] = 1
            manifest_path.write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "(?i)manifest.*split"):
                comparison._model_provenance(config, {"v1": model_dir}, roles)
            manifest["seed"] = 0
            manifest_path.write_text(json.dumps(manifest))
            model_path.write_text(model_path.read_text() + "\nchanged\n")
            with self.assertRaisesRegex(ValueError, "(?i)hash mismatch"):
                comparison._model_provenance(config, {"v1": model_dir}, roles)


if __name__ == "__main__":
    unittest.main()
