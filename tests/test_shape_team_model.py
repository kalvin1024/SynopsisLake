import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml import features
from ml import shape_team_model as team
from ml import sidecar_workflow as prior


def write_hist(path, hist):
    rate, bx, by, counts = hist
    Path(path).write_text("\n".join([repr(float(rate)), ",".join(map(str, bx)), ",".join(map(str, by)),
                                    *[",".join(map(str, row)) for row in counts]])+"\n")


def fixture(root):
    histograms = root/"histograms"
    dataset = root/"dataset"
    histograms.mkdir()
    dataset.mkdir()
    merged = (.5, np.array([-.1, .6, 1.3, 2.]), np.array([-.2, .4, 1.5]),
              np.array([[4., 6.], [10., 20.], [10., 9.]]))
    source0 = (1., np.array([0., .5, 1.]), np.array([0., .5, 1.]), np.array([[4., 6.], [10., 20.]]))
    source1 = (1., np.array([.5, 1., 1.5]), np.array([0., .75, 1.5]), np.array([[2., 3.], [5., 9.]]))
    queries = np.array([[.05, .85, .1, .95], [.4, 1.1, .2, 1.25],
                        [.2, .8, .25, .75], [1.2, 1.8, .3, .7], [3., 4., 3., 4.]])
    write_hist(histograms/"arm_3.txt", merged)
    write_hist(histograms/"src_0.txt", source0)
    write_hist(histograms/"src_1.txt", source1)
    pd.DataFrame(np.column_stack((queries, np.full(len(queries), 20)))).to_csv(
        dataset/"queries-Ans.txt", header=False, index=False)
    pd.DataFrame(dict(qId=range(len(queries)), scale=["tiny"]*len(queries))).to_csv(
        dataset/"queries-meta.csv", index=False)
    alpha = prior._alpha(merged)
    rows = []
    for qid, rectangle in enumerate(queries):
        answer = prior.histogram_count(rectangle, merged)
        rows.append(dict(method="ARM", mergeRate=.5, lonRes=3, qId=qid, gt=20., est=answer,
            relErr=abs(answer-20.)/20., predErr=prior._sa_score(rectangle, merged, alpha),
            betaAlpha=alpha, srcSumEst=sum(prior.histogram_count(rectangle, h) for h in (source0, source1))))
    pd.DataFrame(rows).to_csv(histograms/"rows.csv", index=False)
    return histograms, dataset, merged, queries


def import_v1_reference():
    path = Path(features.__file__).with_name("features_v1.py")
    spec = importlib.util.spec_from_file_location("team_v1_test_reference", path)
    module = importlib.util.module_from_spec(spec)
    # This is deliberately the unmodified team function, including its source
    # aggregation and logged-Java merge shift. No adapter implementation is used.
    with patch.dict(sys.modules, {"features": features}):
        spec.loader.exec_module(module)
    return module


class TeamFeatureParityTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.histograms, self.dataset, self.merged, self.queries = fixture(self.root)
        self.runtime = team.TeamV1Runtime(self.histograms, base_resolution=3)

    def test_live28_matches_unmodified_team_v1_table(self):
        reference = import_v1_reference()
        expected = reference.build_table_v1(self.histograms, self.dataset)
        self.assertEqual(tuple(reference.FEATURES_V1), team.TEAM_FEATURES)
        observed = self.runtime.batch_features(self.queries)
        np.testing.assert_allclose(observed.to_numpy(), expected[list(team.TEAM_FEATURES)].to_numpy(),
                                   rtol=1e-12, atol=1e-12)
        report = team.compare_logged_features(self.runtime, expected, self.queries)
        self.assertTrue(report.within_tolerance.all())
        self.assertEqual(len(report), 28)

    def test_logged_java_difference_is_reported_not_used(self):
        reference = import_v1_reference().build_table_v1(self.histograms, self.dataset)
        reference["skew_pred"] += .02
        report = team.compare_logged_features(self.runtime, reference, self.queries)
        self.assertFalse(report.set_index("feature").loc["skew_pred", "within_tolerance"])
        before = self.runtime.sa_score(self.queries[0])
        self.assertEqual(before, prior._sa_score(self.queries[0], self.merged, self.runtime.alpha))

    def test_truth_file_changes_cannot_change_live_features(self):
        before = self.runtime.batch_features(self.queries)
        (self.dataset/"queries-Ans.txt").unlink()
        (self.histograms/"rows.csv").write_text("not a valid logged feature table")
        after = team.TeamV1Runtime(self.histograms, base_resolution=3).batch_features(self.queries)
        pd.testing.assert_frame_equal(before, after)

    def test_identical_requests_recompute_features(self):
        with patch.object(features, "_query_features", wraps=features._query_features) as call:
            self.runtime.describe(self.queries[0])
            first_calls = call.call_count
            self.runtime.describe(self.queries[0])
            self.assertEqual(call.call_count, 2*first_calls)
        self.runtime.clear_query_cache()
        self.assertIsNone(self.runtime._last_features)

    def test_answer_and_sa_do_not_precompute_model_inputs(self):
        with patch.object(features, "_query_features", side_effect=AssertionError("unexpected feature computation")):
            answer = self.runtime.answer(self.queries[0])
            score = self.runtime.sa_score(self.queries[0])
        self.assertAlmostEqual(answer, prior.histogram_count(self.queries[0], self.merged))
        self.assertGreaterEqual(score, 0.)

    def test_reference_arms_add_same_two_features(self):
        fine = self.runtime.describe(self.queries[0], reference=13.)
        hierarchy = self.runtime.describe(self.queries[0], reference=15.)
        self.assertEqual(tuple(fine), team.TEAM_FEATURES+team.AUX_FEATURES)
        self.assertEqual(tuple(hierarchy), tuple(fine))
        self.assertEqual(fine[team.AUX_FEATURES[0]], np.log1p(13.))
        self.assertEqual(fine[team.AUX_FEATURES[1]], abs(self.runtime.answer(self.queries[0])-13.)/13.)
        self.assertEqual(set(fine)-set(team.TEAM_FEATURES), set(team.AUX_FEATURES))

    def test_geometry_and_reference_guards(self):
        for invalid in ([0, 1, 0, 1, 20], [0, 0, 0, 1], [0, np.nan, 0, 1]):
            with self.assertRaises(ValueError):
                self.runtime.describe(invalid)
        for invalid in (-1., np.inf, [1.]):
            with self.assertRaises(ValueError):
                self.runtime.describe(self.queries[0], reference=invalid)
        with self.assertRaises(ValueError):
            self.runtime.batch_features(np.ones((3, 5)))
        with self.assertRaises(ValueError):
            self.runtime.batch_features(self.queries, references=[1.])

    def test_serialized_merged_identity_must_match(self):
        path = self.root/"original_H.hist"
        prior.save_histogram(path, self.merged)
        runtime = team.TeamV1Runtime(self.histograms, merged_path=path)
        self.assertEqual(runtime.merge_rate, .5)
        self.assertEqual(len(runtime.input_files), 3)
        self.assertEqual(runtime.serialized_input_bytes, sum(Path(p).stat().st_size for p in runtime.input_files))
        bad = list(self.merged)
        bad[0] = .75
        with self.assertRaises(ValueError):
            team.TeamV1Runtime(self.histograms, merged_histogram=bad, merged_path=path)


class TeamLearnedArtifactTests(unittest.TestCase):
    def training_table(self):
        rng = np.random.default_rng(31)
        table = pd.DataFrame(rng.uniform(.01, 1., size=(160, 30)),
                             columns=team.TEAM_FEATURES+team.AUX_FEATURES)
        table["qId"] = np.arange(len(table))
        table["actual_error"] = np.maximum(0., table.iloc[:, 0].to_numpy()-.2)
        table["truth"] = np.arange(len(table))*1000000
        return table

    def test_original_learner_target_and_roundtrip_runtime_bytes(self):
        table = self.training_table()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"full"
            fitted = team.fit_team_model(table, 0, path)
            restored = team.load_team_model(path)
            self.assertIsNone(restored.training_metadata)
            self.assertEqual(restored.feature_names, team.TEAM_FEATURES)
            self.assertEqual(fitted.schema["parameters"]["num_leaves"], 63)
            self.assertEqual(fitted.schema["parameters"]["n_estimators"], 4000)
            self.assertEqual(fitted.schema["parameters"]["n_jobs"], 2)
            self.assertEqual(fitted.schema["label_floor"], features.LABEL_FLOOR)
            self.assertEqual(fitted.training_metadata["inner_validation_qIds"], list(range(0, 160, 10)))
            self.assertFalse(fitted.schema["refit_held_back"])
            np.testing.assert_allclose(restored.log_scores(table), fitted.log_scores(table), rtol=0, atol=0)
            np.testing.assert_allclose(restored.predict_errors(table), np.exp(restored.log_scores(table)))
            before = restored.predict_one(table.iloc[0].to_dict())
            changed = table.iloc[0].to_dict()
            changed.update(truth=-1e12, actual_error=1e12, qId=-200)
            self.assertEqual(before, restored.predict_one(changed))
            expected = sum((path/name).stat().st_size for name in team.MODEL_FILES)
            self.assertEqual(restored.model_bytes, expected)
            (path/"training.json").unlink()
            self.assertEqual(team.load_team_model(path).model_bytes, expected)

    def test_augmented_whitelist_and_artifact_identity(self):
        table = self.training_table()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"augmented"
            fitted = team.fit_team_model(table, 1, path, augmented=True, model_name="v1_plus_reference")
            self.assertEqual(fitted.feature_names, team.TEAM_FEATURES+team.AUX_FEATURES)
            schema_path = path/"model.schema.json"
            schema = json.loads(schema_path.read_text())
            schema["feature_names"][-1] = "actual_error"
            schema_path.write_text(json.dumps(schema))
            with self.assertRaisesRegex(ValueError, "unsupported"):
                team.load_team_model(path)
            schema_path.write_text(json.dumps(fitted.schema))
            (path/"model.txt").write_text((path/"model.txt").read_text()+"tampered\n")
            with self.assertRaisesRegex(ValueError, "identity"):
                team.load_team_model(path)

    def test_training_target_and_feature_inputs_fail_closed(self):
        table = self.training_table()
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)/"should_not_exist"
            for modified in (table.drop(columns=["actual_error"]),
                             table.assign(actual_error=np.inf), table.assign(actual_error=-1.),
                             table.assign(qId=-1), table.assign(split="test"),
                             table.assign(role="calibration"), table.assign(outer_split="validation")):
                with self.assertRaises(ValueError):
                    team.fit_team_model(modified, 0, path)
                self.assertFalse(path.exists())
            with self.assertRaises(ValueError):
                team.fit_team_model(table.drop(columns=[team.AUX_FEATURES[0]]), 0, path, augmented=True)


if __name__ == "__main__":
    unittest.main()
