"""Research-protocol checks for the shared unseen-region experiment.

These checks protect spatial membership, source-only calibration and the
meaning of the returned answer. They are not accuracy or latency evidence.
"""
from __future__ import annotations

import unittest
import json
from pathlib import Path
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml import region_budget_comparison as comparison


def _geometry():
    return pd.DataFrame({
        "qId": [10, 11, 12, 13, 14],
        "minLon": [-2., 0., -1., -1., 1.],
        "maxLon": [-1., 1., .2, 0., 2.],
        "minLat": [0.] * 5,
        "maxLat": [1.] * 5,
    })


def _memberships():
    # Many queries are intentional: a nonempty TRAIN/CAL split is meaningful,
    # while a pair of western queries could be emptied by a rounding policy.
    n = 100
    geom = pd.DataFrame({
        "qId": np.arange(n, dtype=int),
        "minLon": np.r_[np.full(70, -2.), np.full(25, 1.), np.full(5, -.2)],
        "maxLon": np.r_[np.full(70, -1.), np.full(25, 2.), np.full(5, .2)],
        "minLat": np.zeros(n), "maxLat": np.ones(n),
    })
    return comparison.region_memberships(geom, cut=0.)[0]


def _answers(candidate, answer, accepted, score=None):
    accepted = np.asarray(accepted, dtype=bool)
    return pd.DataFrame({
        "candidate": candidate, "answer": answer,
        "fromSynopsis": accepted,
        "errorScore": np.zeros(len(accepted)) if score is None else score,
        "gateNs": np.full(len(accepted), 100),
        "synNs": np.where(accepted, 100, 0),
        "rawNs": np.where(accepted, 0, 1000),
    })


class RegionProtocolTests(unittest.TestCase):
    def test_whole_rectangles_not_only_centers_define_the_unseen_region(self):
        roles, cut = comparison.region_memberships(_geometry(), cut=0.)
        self.assertEqual(cut, 0.)
        self.assertEqual(roles.set_index("qId").region.to_dict(), {
            10: "west", 11: "east", 12: "crossing", 13: "crossing", 14: "east",
        })
        # The rectangle ending on the cut is excluded from strict western
        # supervision; an eastern rectangle starting on the cut is test-only.
        self.assertNotEqual(roles.set_index("qId").loc[13, "region"], "west")

    def test_default_cut_reuses_the_existing_geometry_midpoint(self):
        geom = pd.DataFrame({
            "minLon": [-12., 0.], "maxLon": [-8., 10.],
            "minLat": [0., 0.], "maxLat": [1., 1.],
        }, index=[81, 97])
        roles, cut = comparison.region_memberships(geom)
        # Centers -10 and5 give -2.5; the bounding-box midpoint would be -1.
        self.assertAlmostEqual(cut, -2.5)
        self.assertEqual(set(roles.qId), {81, 97})
        self.assertEqual(set(roles.region), {"west", "east"})

    def test_invalid_rectangles_cannot_be_silently_assigned(self):
        geom = _geometry()
        geom.loc[0, "maxLon"] = geom.loc[0, "minLon"]
        with self.assertRaises(ValueError):
            comparison.region_memberships(geom, cut=0.)

    def test_reused_query_ids_cannot_receive_two_spatial_roles(self):
        geom = _geometry()
        geom.loc[1, "qId"] = geom.loc[0, "qId"]
        with self.assertRaises(ValueError):
            comparison.region_memberships(geom, cut=0.)

    def test_no_test_or_crossing_rectangle_enters_training_or_calibration(self):
        membership = _memberships()
        roles = comparison.source_roles(membership, seed=0)
        indexed = roles.set_index("qId")
        self.assertEqual(set(indexed.loc[np.arange(70, 95), "split"]), {"test"})
        self.assertEqual(set(indexed.loc[np.arange(95, 100), "split"]), {"excluded"})
        supervised = roles[roles.split.isin(["train", "calibration"])]
        self.assertEqual(set(supervised.region), {"west"})
        self.assertEqual(set(supervised.qId), set(range(70)))
        self.assertGreater((supervised.split == "train").sum(), 0)
        self.assertGreater((supervised.split == "calibration").sum(), 0)
        self.assertFalse(roles.qId.duplicated().any())

    def test_seed_changes_source_supervision_but_not_the_test_region(self):
        membership = _memberships()
        first = comparison.source_roles(membership, seed=0).set_index("qId")
        second = comparison.source_roles(membership, seed=1).set_index("qId")
        self.assertEqual(set(first[first.split == "test"].index),
                         set(second[second.split == "test"].index))
        self.assertNotEqual(set(first[first.split == "calibration"].index),
                            set(second[second.split == "calibration"].index))

    def test_membership_assignment_is_reproducible_after_input_reordering(self):
        membership = _memberships()
        a = comparison.source_roles(membership, seed=2).set_index("qId").split.sort_index()
        b = comparison.source_roles(membership.sample(frac=1., random_state=91), seed=2).set_index("qId").split.sort_index()
        pd.testing.assert_series_equal(a, b)


class SourceCalibrationTests(unittest.TestCase):
    def test_cutoff_cannot_select_the_good_half_of_a_tie(self):
        cutoff = comparison.choose_cutoff(np.array([.1, .1, .2]),
            np.array([0., .4, 0.]), epsilon=.25, delta=.05)
        self.assertEqual(cutoff, -np.inf)

    def test_a_bad_prefix_can_be_recovered_by_later_calibration_answers(self):
        # Exactly one of twenty is bad. Stopping at the first failed prefix
        # would miss the valid largest threshold; no test labels are involved.
        cutoff = comparison.choose_cutoff(np.arange(20, dtype=float),
            np.r_[.4, np.zeros(19)], epsilon=.25, delta=.05)
        self.assertEqual(cutoff, 19.)

    def test_infinite_score_is_hard_rejected_even_when_all_labels_are_good(self):
        cutoff = comparison.choose_cutoff(np.array([.1, 1., np.inf]),
            np.zeros(3), epsilon=.25, delta=.05)
        self.assertEqual(cutoff, 1.)
        self.assertTrue(np.isfinite(cutoff))

    def test_all_infinite_bound_scores_produce_no_acceptable_calibration_cutoff(self):
        cutoff = comparison.choose_cutoff(np.full(3, np.inf), np.zeros(3),
                                           epsilon=.25, delta=.05)
        self.assertEqual(cutoff, -np.inf)

    def test_zero_truth_positive_answer_is_bad_not_dropped_from_calibration(self):
        cutoff = comparison.choose_cutoff(np.array([.1, .2]),
            np.array([np.inf, 0.]), epsilon=.25, delta=.5)
        self.assertEqual(cutoff, .2)

    def test_invalid_nan_scores_or_labels_fail_instead_of_improving_risk(self):
        for scores, errors in (([.1, np.nan], [0., 0.]),
                               ([.1, .2], [0., np.nan])):
            with self.subTest(scores=scores, errors=errors):
                with self.assertRaises(ValueError):
                    comparison.choose_cutoff(np.asarray(scores), np.asarray(errors),
                                             epsilon=.25, delta=.05)


class ReturnedAnswerMeaningTests(unittest.TestCase):
    def test_fine_direct_risk_uses_its_own_answer_not_the_original_H(self):
        truth = np.array([100., 100.])
        fine = _answers([110., 150.], [110., 100.], [True, False])
        coarse = _answers([150., 110.], [150., 100.], [True, False])
        fine_result = comparison.summarize(fine, truth, epsilon=.25)
        coarse_result = comparison.summarize(coarse, truth, epsilon=.25)
        self.assertEqual(fine_result["accepted_good_count"], 1)
        self.assertEqual(fine_result["accepted_bad_count"], 0)
        self.assertEqual(coarse_result["accepted_good_count"], 0)
        self.assertEqual(coarse_result["accepted_bad_count"], 1)
        self.assertAlmostEqual(fine_result["candidate_mean_error"], .3)
        self.assertAlmostEqual(fine_result["returned_mean_error"], .05)
        self.assertAlmostEqual(coarse_result["returned_mean_error"], .25)
        self.assertEqual(fine_result["fallback_count"], 1)
        self.assertEqual(fine_result["accepted_count"], 1)
        self.assertAlmostEqual(fine_result["coverage"], .5)

    def test_zero_accepted_answers_have_undefined_risk(self):
        frame = _answers([150., 50.], [100., 100.], [False, False])
        result = comparison.summarize(frame, np.array([100., 100.]), epsilon=.25)
        self.assertEqual(result["coverage"], 0.)
        self.assertEqual(result["accepted_count"], 0)
        self.assertEqual(result["accepted_bad_count"], 0)
        self.assertEqual(result["fallback_count"], 2)
        self.assertTrue(np.isnan(result["violation"]))
        self.assertEqual(result["returned_mean_error"], 0.)

    def test_exactly_25_percent_is_good_and_strictly_over_is_bad(self):
        values = [125., 125.0001]
        frame = _answers(values, values, [True, True])
        result = comparison.summarize(frame, np.array([100., 100.]), epsilon=.25)
        self.assertEqual(result["accepted_good_count"], 1)
        self.assertEqual(result["accepted_bad_count"], 1)
        self.assertAlmostEqual(result["violation"], .5)

    def test_zero_count_queries_remain_in_the_main_report(self):
        frame = _answers([1., 0.], [0., 0.], [False, False])
        result = comparison.summarize(frame, np.array([0., 0.]), epsilon=.25)
        self.assertEqual(result["queries"], 2)
        self.assertTrue(np.isinf(result["candidate_mean_error"]))
        self.assertEqual(result["returned_mean_error"], 0.)


class LabelAndPayloadIsolationTests(unittest.TestCase):
    def test_unparseable_eastern_truth_does_not_enter_geometry_or_source_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/"queries.csv"
            path.write_text("-2,-1,0,1,42\n1,2,0,1,FORBIDDEN_EAST_LABEL\n")
            geometry=comparison._read_geometry(path)
            self.assertEqual(list(geometry.columns),["qId","minLon","maxLon","minLat","maxLat"])
            self.assertEqual(comparison._west_labels(path,[0]),{0:42.})
            with self.assertRaises(ValueError):comparison._west_labels(path,[1])

    def test_changed_payload_is_rejected_before_any_eastern_java_work(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);payload=root/"model.txt";payload.write_text("original")
            protocol={"payload_files":[dict(path=str(payload),bytes=8,sha256=comparison._sha(payload))]}
            comparison._json(root/"protocol.json",protocol)
            payload.write_text("changed!")
            with patch.object(comparison,"_java",side_effect=AssertionError("must not run test queries")):
                with self.assertRaisesRegex(RuntimeError,"Frozen runtime payload changed"):
                    comparison._evaluate_dataset({},dict(folder=root),"",[])

    def test_exact_has_no_invented_approximate_candidate_error_at_zero_truth(self):
        frame=_answers([np.nan,np.nan],[0.,5.],[False,False])
        result=comparison.summarize(frame,np.array([0.,5.]),epsilon=.25)
        self.assertTrue(np.isnan(result["candidate_mean_error"]))
        self.assertEqual(result["returned_mean_error"],0.)


if __name__ == "__main__":
    unittest.main()
