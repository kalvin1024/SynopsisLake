"""Protocol checks, not empirical performance claims, for the shared comparison."""
import json
from pathlib import Path
import tempfile
import shutil
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from ml import local_shape_workflow as source
from ml import shape_inputs
from ml import shape_team_comparison as workflow
from ml import shape_team_model as team
from ml import sidecar_workflow as prior
from test_sidecar_workflow import fixture


class FakePredictor:
    """Deterministic non-learning test double; it has no truth or ID state."""
    def __init__(self, method):
        self.method = method

    def log_scores(self, frame):
        forbidden = {"qId", "truth", "actual_error", "label"}
        if forbidden & set(frame):
            raise AssertionError("Offline labels/IDs reached fake inference")
        return np.full(len(frame), np.log(.1 if self.method == "ML_v1" else .12))

    def predict_one(self, values):
        frame = pd.DataFrame([values])
        value = self.log_scores(frame)[0]
        return value, np.exp(value)


class CommonComparisonProtocol(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        root = Path(cls.temp.name)
        cfg, cls.queries, cls.outer = fixture(root)
        cls.histograms = Path(cfg["histogram_dir"])
        cls.prepared = shape_inputs.prepare_direct_inputs(dict(cfg, split_kind="cell"))
        cls.source = source.develop_experiment(dict(
            preparation_dir=str(cls.prepared), family="hierarchical", seeds=[0,1,2],
            extra_budgets=[49152], selection_budgets=[49152], include_ml=False,
            calibration_queries=0, test_queries=0, train_queries=0, validation_queries=0,
            output_root=str(root/"source_output"), evaluate_old_test=False,
        ))
        cls.cfg = dict(source_development_dir=str(cls.source), histogram_dir=str(cls.histograms),
                       output_root=str(root/"comparison_output"), backend="indexed", seeds=[0,1,2],
                       train_queries=0, calibration_queries=0, fresh_query_count=1000,
                       fresh_query_seed=12345, timing_queries=0, bootstrap_repeats=30)
        cls.fitted_ids = []
        def fit(table, seed, out_dir, *, augmented, model_name):
            cls.fitted_ids.append((seed, model_name, set(table.qId)))
            expected = set(cls.outer.loc[(cls.outer.seed == seed)&(cls.outer.split == "train"), "qId"])
            if set(table.qId) != expected:
                raise AssertionError("Model did not receive inherited TRAIN only")
            if not np.isfinite(table.actual_error).all():
                raise AssertionError("Unexpected target")
            destination = Path(out_dir)
            destination.mkdir(parents=True)
            for filename in team.MODEL_FILES:
                (destination/filename).write_text("{}\n")
            return SimpleNamespace(training_metadata=dict(seed=seed, method=model_name, test_double=True))
        cls.fit_patch = mock.patch.object(team, "fit_team_model", side_effect=fit)
        cls.load_patch = mock.patch.object(team, "load_team_model", side_effect=lambda p:FakePredictor(Path(p).name))
        cls.fit_patch.start()
        cls.load_patch.start()
        cls.result = workflow.develop_experiment(cls.cfg)

    @classmethod
    def tearDownClass(cls):
        cls.load_patch.stop()
        cls.fit_patch.stop()
        cls.temp.cleanup()

    def test_every_model_uses_identical_declared_train_ids(self):
        self.assertEqual(len(self.fitted_ids), 9)
        for seed in (0,1,2):
            groups = [ids for s, _, ids in self.fitted_ids if s == seed]
            self.assertEqual(groups[0], groups[1])
            self.assertEqual(groups[0], groups[2])

    def test_model_gate_scores_are_exp_of_original_log_target(self):
        geometry = self.queries[["qId", *workflow.GEOMETRY]].head(2)
        result = workflow._score_geometry(self.result, geometry, 0)
        actual = result[result.method == "ML_v1"]
        np.testing.assert_allclose(actual.score, .1)
        np.testing.assert_allclose(actual.predicted_error, .1)
        np.testing.assert_allclose(actual.raw_log_score, np.log(.1))

    def test_geometry_feature_api_rejects_truth_and_duplicate_ids(self):
        with self.assertRaisesRegex(ValueError, "without truth"):
            workflow._geometry_features(self.result, self.queries.head(2))
        geometry = self.queries[["qId", *workflow.GEOMETRY]].head(2).copy()
        geometry["qId"] = 1
        with self.assertRaisesRegex(ValueError, "unique"):
            workflow._geometry_features(self.result, geometry)

    def test_raw_labels_generated_only_after_scores_and_gates_are_frozen(self):
        real_labels = prior._indexed_labels
        observed = []
        def guarded(points, rectangles):
            runs = sorted((self.result/"confirmation").glob("team-shape-confirmation-*"))
            active = runs[-1]
            workflow.verify_confirmation(active)
            self.assertTrue((active/"score_lock.json").is_file())
            scores = pd.read_csv(active/"scores_before_labels.csv")
            self.assertNotIn("truth", scores)
            self.assertNotIn("actual_error", scores)
            observed.append(len(rectangles))
            return real_labels(points, rectangles)
        cfg = dict(self.cfg, development_dir=str(self.result))
        with mock.patch.object(prior, "_indexed_labels", side_effect=guarded):
            result = workflow.confirm_experiment(cfg)
        self.assertEqual(observed, [1000])
        audit = pd.read_csv(result/"fresh_raw_audit.csv")
        self.assertTrue(audit.matches.all())
        quality = pd.read_csv(result/"per_query_quality.csv")
        self.assertEqual(quality.qId.nunique(), 1000)
        self.assertEqual(set(quality.method), set(workflow.METHODS))
        metrics = pd.read_csv(result/"metrics.csv")
        self.assertTrue(metrics.loc[metrics.method == "SA", "error_prediction_mae_pp"].isna().all())
        self.assertTrue((result/"team_shape_results.zip").is_file())

    def test_development_freeze_covers_model_properties_and_packet(self):
        properties = self.result/"models/seed_0/ML_v1/model.properties"
        original = properties.read_bytes()
        properties.write_bytes(original+b"changed\n")
        try:
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                workflow.verify_development(self.result)
        finally:
            properties.write_bytes(original)
        workflow.verify_development(self.result)

    def test_confirmation_cannot_change_seed_after_model_freeze(self):
        cfg = dict(self.cfg, development_dir=str(self.result), fresh_query_seed=99999)
        with mock.patch.object(prior, "_indexed_labels") as labels:
            with self.assertRaisesRegex(ValueError, "declared before"):
                workflow.confirm_experiment(cfg)
            labels.assert_not_called()

    def test_payload_costs_charge_required_source_models_and_reference(self):
        rows = pd.DataFrame(workflow._payload_costs(self.result, 0)).set_index("method")
        self.assertGreater(rows.loc["ML_v1", "source_text_bytes"], 0)
        self.assertGreater(rows.loc["ML_v1", "model_schema_weights_properties_bytes"], 0)
        self.assertGreater(rows.loc["ML_v1_shape", "native_bytes"], 0)
        self.assertGreater(rows.loc["ML_v1_shape", "packet_bytes"], 0)
        self.assertGreater(rows.loc["ML_v1_fine", "fine_bytes"], 0)
        for row in rows.itertuples():
            files = [self.result/file for file in row.file_paths.split("|")]
            self.assertEqual(row.total_serialized_bytes, sum(file.stat().st_size for file in files))

    def test_wrong_source_exports_are_rejected_before_any_model_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder = Path(tmp)/"other_sources"
            shutil.copytree(self.histograms, folder)
            file = folder/"src_0.txt"
            text = file.read_text().splitlines()
            text[-1] = "7,6,6"
            file.write_text("\n".join(text)+"\n")
            with self.assertRaisesRegex(ValueError,"source identity"):
                workflow.develop_experiment(dict(self.cfg,histogram_dir=str(folder)))

    def test_executable_setup_is_explicit_even_for_indexed_backend(self):
        state = json.loads((self.result/"run.json").read_text())
        self.assertEqual(state["executable_setup"]["backend"], "indexed")
        self.assertIn("compilation_seconds", state["executable_setup"])
        self.assertTrue((self.result/"source_identity_audit.csv").is_file())

    def test_live_timing_uses_real_engine_and_frozen_epsilon_alias(self):
        # This reaches run_benchmark's frozen-cutoff validation and real raw
        # fallback, rather than mocking away the eps/epsilon integration seam.
        labeled = self.queries.tail(2).drop(columns="cell").copy()
        geometry = labeled[["qId", *workflow.GEOMETRY]]
        scored = workflow._score_geometry(self.result, geometry, 0)
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)
            scored.to_csv(output/"scores_before_labels.csv", index=False)
            timing = workflow._timing(output, self.result, geometry, labeled, self.queries,
                                      0, dict(self.cfg, timing_queries=2, timing_repeats=1,
                                              epsilon=.25, delta=.05), labeled.qId.to_numpy())
            self.assertEqual(timing.method.nunique(), len(workflow.METHODS)+1)
            self.assertTrue((timing.query_count == 2).all())
            self.assertTrue((timing.epsilon == .25).all())
            self.assertTrue(np.isfinite(timing.total_mean_ms).all())
            parity = pd.read_csv(output/"live_parity_seed_0.csv")
            self.assertTrue(parity.all_matches.all())
            actual = pd.read_csv(output/"e2e/seed_0/per_query.csv")
            rejected = actual[(actual.method == "ML_v1") & ~actual.accepted]
            self.assertGreater(len(rejected), 0)
            self.assertTrue((rejected.raw_ns > 0).all())
            np.testing.assert_array_equal(rejected.answer, rejected.truth)


class MetricSemantics(unittest.TestCase):
    def test_no_accept_risk_is_undefined_and_zero_truth_error_retained(self):
        scored = pd.DataFrame([
            dict(seed=0, method="R10", qId=0, H=1., score=.2, predicted_error=.2),
            dict(seed=0, method="R10", qId=1, H=10., score=.1, predicted_error=.1),
        ])
        quality = workflow.attach_truth(scored, pd.DataFrame(dict(qId=[0,1],truth=[0,10])))
        gate = pd.DataFrame([dict(seed=0,method="R10",cutoff=-np.inf)])
        metrics, _ = workflow.quality_tables(quality, gate, workflow._config({}))
        row = metrics[metrics.cohort == "all"].iloc[0]
        self.assertTrue(np.isinf(row.error_prediction_mae_pp))
        self.assertTrue(np.isnan(row.accepted_violation))
        self.assertEqual(row.nonfinite_actual_errors, 1)

    def test_fixed_coverage_keeps_entire_tie_group(self):
        scored = pd.DataFrame(dict(seed=[0]*4,method=["R10"]*4,qId=range(4),
            H=[10.]*4,score=[.1,.1,.1,.2],predicted_error=[.1,.1,.1,.2]))
        quality = workflow.attach_truth(scored, pd.DataFrame(dict(qId=range(4),truth=[10]*4)))
        metrics, fixed = workflow.quality_tables(quality,pd.DataFrame([dict(seed=0,method="R10",cutoff=.1)]),workflow._config({}))
        row = fixed[(fixed.cohort == "all")&(fixed.nominal_coverage == .1)].iloc[0]
        self.assertEqual(row.selected_count, 3)
        self.assertEqual(row.extra_tied_queries, 2)

    def test_paired_intervals_do_not_count_seed_appearances_independently(self):
        rows = []
        for seed in (0,1,2):
            for qid in range(3):
                for method, prediction in (("ML_v1_shape",.1),("ML_v1",.2),("ML_v1_fine",.15),("R10",.1),("fine_matched",.2)):
                    rows.append(dict(seed=seed,qId=qid,method=method,truth=10,actual_error=.1,
                        predicted_error=prediction,x0=qid/4,x1=qid/4+.1,y0=0.,y1=.1))
        histogram=(0.,np.array([0.,1.]),np.array([0.,1.]),np.array([[10.]]))
        result=workflow.paired_intervals(pd.DataFrame(rows),histogram,30)
        self.assertTrue((result.unique_queries == 3).all())
        self.assertAlmostEqual(result[result.second == "ML_v1"].delta_mae_pp.iloc[0],-10.)

    def test_truth_attachment_rejects_scores_already_containing_labels(self):
        scored=pd.DataFrame([dict(seed=0,method="R10",qId=0,H=1.,truth=1.,score=.1)])
        with self.assertRaisesRegex(ValueError,"no truth"):
            workflow.attach_truth(scored,pd.DataFrame(dict(qId=[0],truth=[1])))


if __name__ == "__main__":
    unittest.main()
