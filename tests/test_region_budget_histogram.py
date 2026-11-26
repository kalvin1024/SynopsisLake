"""Meaningful checks of budget accounting and inclusive-query count bounds."""
import gzip
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from ml import region_budget_histogram as fine


class RegionBudgetHistogramTests(unittest.TestCase):
    def test_budget_selects_largest_serializable_grid_without_padding(self):
        for n in (1, 2, 3, 17, 100):
            exact = 64 + 4 * n * n
            self.assertEqual(fine.resolution_for_budget(exact), n)
            self.assertEqual(fine.resolution_for_budget(64 + 4 * (n + 1) ** 2 - 1), n)
            self.assertEqual(fine.payload_bytes(n), exact)
        for bad in (0, 64, 67, 68.0, True):
            with self.subTest(value=bad):
                with self.assertRaises(ValueError):
                    fine.resolution_for_budget(bad)

    def test_existing_codec_compatibility_and_every_byte_is_charged(self):
        from ml.sidecar_workflow import _load_fine, _save_fine
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / "fine.counts"
            counts = np.array([[0, 3], [5, 2 ** 32 - 1]], dtype=np.uint64)
            size = fine.write_count_histogram(path, counts, (-2, 4, -1, 7), budget_bytes=83)
            self.assertEqual(size, 80)
            self.assertEqual(path.read_bytes()[:8], b"SLCNT001")
            hist = _load_fine(path)
            np.testing.assert_array_equal(hist[3], counts)
            second = Path(td) / "existing.counts"
            _save_fine(second, counts, (-2, 4, -1, 7))
            self.assertEqual(path.read_bytes(), second.read_bytes())
            np.testing.assert_array_equal(fine.load_count_histogram(second)[3], counts)
            with self.assertRaises(ValueError):
                fine.write_count_histogram(path, counts, (-2, 4, -1, 7), budget_bytes=79)
            with self.assertRaises(OverflowError):
                fine.write_count_histogram(path, np.array([[2 ** 32]], dtype=np.uint64), (0, 1, 0, 1))
            with self.assertRaises(ValueError):
                fine.write_count_histogram(path, [[1.5]], (0, 1, 0, 1))
            with self.assertRaises(ValueError):
                fine.write_count_histogram(path, [[True]], (0, 1, 0, 1))

    def test_real_file_streaming_plain_gzip_columns_and_outer_edge(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            (root / "0.csv").write_text("0 0 99\n1 1 101\n2 2 0\n")
            with gzip.open(root / "1.csv.gz", "wt") as stream:
                stream.write("0.5\t1.5\t123\n1.5\t0.5\t888\n")
            (root / "2.csv").write_text("")
            paths = [root / "0.csv", root / "1.csv.gz", root / "2.csv"]
            output = root / "out" / "fine.counts"
            result = fine.build_fine_histogram(paths, (0, 2, 0, 2), 80, output, chunk_size=2)
            self.assertEqual(result["grid_bytes"], 80)
            self.assertEqual(result["raw_rows"], 5)
            self.assertEqual(result["unused_grid_budget_bytes"], 0)
            self.assertFalse(result["selection_uses_query_labels"])
            np.testing.assert_array_equal(fine.load_count_histogram(output)[3], [[1, 1], [1, 2]])
            with self.assertRaises(ValueError):
                fine.build_fine_histogram(paths[:1] * 2, (0, 2, 0, 2), 80, output)
            (root / "outside.csv").write_text("-0.001 1\n")
            with self.assertRaisesRegex(ValueError, "outside"):
                fine.build_fine_histogram([root / "outside.csv"], (0, 2, 0, 2), 80, output)

    def test_inclusive_intervals_contain_true_counts_even_at_atoms(self):
        # Include points exactly on internal split planes and both outer edges.
        points = np.array([[0, 0], [0, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                           [2, 0], [2, 1], [2, 2], [.2, .3], [1.7, .1], [.9, 1.2]])
        counts = np.histogram2d(points[:, 0], points[:, 1], bins=2, range=((0, 2), (0, 2)))[0]
        hist = (0., np.array([0., 1., 2.]), np.array([0., 1., 2.]), counts)
        queries = [(-1, 3, -1, 3), (0, 2, 0, 2), (1, 1, 0, 2), (0, 2, 1, 1),
                   (1, 1, 1, 1), (0, 0, 0, 2), (2, 2, 0, 2), (-1, 0, -1, 0),
                   (2, 3, 2, 3), (3, 4, 3, 4), (.2, .9, .1, .7), (0, 1, 0, 1)]
        rng = np.random.default_rng(714)
        queries += [tuple([*sorted(rng.uniform(-.5, 2.5, 2)), *sorted(rng.uniform(-.5, 2.5, 2))])
                    for _ in range(200)]
        for q in queries:
            with self.subTest(query=q):
                true = int(((points[:, 0] >= q[0]) & (points[:, 0] <= q[1])
                            & (points[:, 1] >= q[2]) & (points[:, 1] <= q[3])).sum())
                answer, lower, upper, bound = fine.query_interval(hist, q)
                self.assertLessEqual(lower, true)
                self.assertLessEqual(true, upper)
                actual = abs(answer - true) / true if true else (0. if answer == 0 else math.inf)
                self.assertLessEqual(actual, bound + 1e-12)
                self.assertLessEqual(lower, answer)
                self.assertLessEqual(answer, upper)

    def test_direct_bound_zero_and_infinite_cases(self):
        axes = np.array([0., 1., 2.])
        hist = (0., axes, axes, np.array([[1, 0], [0, 3]], dtype=np.uint32))
        self.assertEqual(fine.query_interval(hist, (-1, 3, -1, 3)), (4., 4, 4, 0.))
        self.assertEqual(fine.query_interval(hist, (5, 6, 5, 6)), (0., 0, 0, 0.))
        self.assertEqual(fine.query_interval(hist, (1, 1, 1, 1)), (0., 0, 4, 1.))
        self.assertTrue(math.isinf(fine.direct_score(hist, (.1, .3, .1, .3))))
        empty = (0., axes, axes, np.zeros((2, 2), np.uint32))
        self.assertEqual(fine.query_interval(empty, (0, 0, 0, 2)), (0., 0, 0, 0.))
        for bad in ((1, 0, 0, 1), (0, 1, 0, math.inf), (0, 1, 0)):
            with self.assertRaises(ValueError):
                fine.query_interval(hist, bad)

    def test_reference_mode_scores_original_answer_not_fine_answer(self):
        self.assertEqual(fine.reference_score(20, 10), 1.)
        self.assertEqual(fine.reference_score(10, 20), .5)
        self.assertEqual(fine.reference_score(0, 0), 0.)
        for bad in ((-1, 0), (1, -1), (math.inf, 1), (1, math.nan)):
            with self.assertRaises(ValueError):
                fine.reference_score(*bad)


if __name__ == "__main__":
    unittest.main()
