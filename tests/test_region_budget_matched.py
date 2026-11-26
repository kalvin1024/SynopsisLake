"""Holdout and method-scope checks for the SS-sized follow-up."""

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml import region_budget_comparison as comparison
from ml.region_budget_matched_inputs import _fresh_test_ids


class FreshEasternHoldoutTests(unittest.TestCase):
    def setUp(self):
        # Use enough eligible IDs to exercise actual sampling, not just take
        # every remaining row after exclusion.
        self.members = pd.DataFrame({
            "qId": np.arange(6_012),
            "region": ["west"] * 10 + ["east"] * 6_000 + ["crossing"] * 2,
        })
        self.previous_ids = np.arange(1_210, 3_210)

    def test_fresh_sample_is_deterministic_east_and_disjoint_from_prior_run(self):
        chosen = _fresh_test_ids(self.members, self.previous_ids, 2_000)
        shuffled = self.members.sample(frac=1., random_state=31)
        rerun = _fresh_test_ids(shuffled, self.previous_ids[::-1], 2_000)

        self.assertEqual(len(set(chosen)), 2_000)
        self.assertTrue(set(chosen) <= set(range(10, 6_010)))
        self.assertTrue(set(chosen).isdisjoint(self.previous_ids))
        np.testing.assert_array_equal(chosen, rerun)

    def test_incomplete_clean_holdout_fails_instead_of_shrinking_dataset(self):
        smaller = self.members[self.members.qId < 4_009]
        with self.assertRaises(ValueError):
            _fresh_test_ids(smaller, self.previous_ids, 2_000)

    def test_wrong_split_or_duplicate_old_ids_cannot_create_a_fresh_run(self):
        for invalid in (np.r_[self.previous_ids[1:], 1],
                        np.r_[self.previous_ids[1:], self.previous_ids[-1]]):
            with self.subTest(last_id=invalid[-1]), self.assertRaises(ValueError):
                _fresh_test_ids(self.members, invalid, 2_000)


class SelectedMethodsPreparationTests(unittest.TestCase):
    def test_western_calibration_preparation_needs_no_v1_features_or_training(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            histograms = root / "exports"
            histograms.mkdir()
            merged = histograms / "arm_0.txt"
            merged.write_text("0.2\n-3,3\n0,1\n42\n", encoding="utf-8")
            source = histograms / "src_0.txt"
            source.write_text("1.0\n-3,3\n0,1\n42\n", encoding="utf-8")
            queries = root / "queries-Ans.txt"
            # Malformed labels on east establish that even label parsing is
            # isolated from this phase.
            queries.write_text(
                "-2,-1,0,1,42\n" * 20 + "1,2,0,1,DO_NOT_READ_EAST_LABEL\n" * 6,
                encoding="utf-8",
            )
            c = comparison._config({
                "datasets": [{"dataset": "fixture"}],
                "seeds": [0, 1, 2],
                "n_test_queries": 2,
                "prior_results_zip": str(root / "prior.zip"),
                "dataset_budget_bytes": {"fixture": 1_000},
                "methods": ["source_shape", "fine_reference", "fine_direct"],
            })
            frozen_histogram = root / "previous_H.hist"
            comparison.prior.save_histogram(frozen_histogram,
                                            comparison.features.load_hist(merged))
            previous_ids = np.array([20, 22])
            members, _ = comparison.region_memberships(comparison._read_geometry(queries), 0.)
            fresh_ids = _fresh_test_ids(members, previous_ids, c["n_test_queries"])
            item = {
                "dataset": "fixture", "run_dir": str(histograms), "data_dir": str(root),
                "queries_file": str(queries), "sources": [source], "raw_paths": [],
                "H_path": merged, "actual_rate": 0.2, "region_cut": 0.0,
                "matched_prior": {
                    "region_cut": 0.0,
                    "previous_ids": previous_ids, "test_ids": fresh_ids,
                    "h_sha256": comparison._sha(frozen_histogram),
                },
            }

            def exact_audit(config, classpath, folder, method, query_file, **kwargs):
                self.assertEqual(method, "exact")
                n_queries = len(Path(query_file).read_text(encoding="utf-8").splitlines())
                return pd.DataFrame({"answer": [42.] * n_queries}), {}

            with (patch.object(comparison, "_java", side_effect=exact_audit) as java,
                  patch.object(comparison, "_fit_and_publish_team_model",
                               side_effect=AssertionError("unexpected model fit"))):
                info = comparison._prepare_dataset(c, item, root, "")

            self.assertEqual(java.call_count, 1)
            self.assertEqual(info["models"], {})
            np.testing.assert_array_equal(info["test_ids"], fresh_ids)
            self.assertEqual(set(info["calibration_ids"]), {0, 1, 2})
            for ids in info["calibration_ids"].values():
                self.assertGreater(len(ids), 0)
                self.assertTrue(set(ids) <= set(range(20)))
            splits = pd.read_csv(info["folder"] / "region_splits.csv")
            for seed in c["seeds"]:
                roles = splits[splits.seed.eq(seed)].set_index("qId").split
                self.assertEqual(set(roles[roles.eq("test")].index), set(fresh_ids))
                self.assertEqual(set(roles[roles.eq("previous_test")].index), set(previous_ids))
                self.assertFalse(roles.eq("train").any())
            self.assertLess(info["budget_floor"], 1_000)


if __name__ == "__main__":
    unittest.main()
