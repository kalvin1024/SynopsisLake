"""Information tests: equal histogram features need not imply equal accuracy.

These synthetic counterexamples explain the representation, not measured
performance. They also preserve a limitation: finite child moments cannot
identify every range count, so R10 is not an accuracy certificate.
"""
import tempfile
from pathlib import Path
import unittest

import numpy as np

from ml import hierarchical_shape as hierarchy
from ml import local_shape as local
from ml import sidecar_workflow as prior
from ml.shape_team_model import TeamV1Runtime, TEAM_FEATURES


def example(left_x, right_x):
    """One data set with translated cells and fixed, equal count histograms."""
    left = np.column_stack((left_x, np.full(len(left_x), .25)))
    right = np.column_stack((np.asarray(right_x)+1., np.full(len(right_x), .25)))
    points = np.concatenate((left, right))
    counts = np.array([[len(left)], [len(right)]], float)
    histogram = (1., np.array([0., 1., 2.]), np.array([0., 1.]), counts)
    parent_counts, sums, _ = prior._sufficient(points, histogram)
    encoded = local.bounded.encode_moments(prior._moments(parent_counts, sums))
    quarters = local.quadrant_counts_from_points(points, histogram)
    child_counts, child_sums = hierarchy.accumulate_child_moments(points, histogram)
    children = hierarchy.encode_child_moments(child_counts, child_sums)
    payload = hierarchy.build_selection([histogram], [encoded], [quarters], [children], 4096)
    runtime = hierarchy.load_runtime([histogram], [encoded], payload)
    return points, histogram, encoded, quarters, children, runtime


def features_for(histogram, rectangles):
    """Use the actual query-time team feature adapter, without query truth."""
    with tempfile.TemporaryDirectory() as folder:
        folder = Path(folder)
        _, x, y, counts = histogram
        text = "1.0\n"+",".join(map(str, x))+"\n"+",".join(map(str, y))+"\n"
        text += "\n".join(",".join(map(str, row)) for row in counts)+"\n"
        (folder/"src_0.txt").write_text(text)
        context = TeamV1Runtime(folder, merged_histogram=histogram)
        return context.batch_features(rectangles).loc[:, list(TEAM_FEATURES)].to_numpy()


def raw_counts(points, rectangles):
    return np.asarray([np.count_nonzero((points[:, 0] >= q[0]) & (points[:, 0] <= q[1]) &
        (points[:, 1] >= q[2]) & (points[:, 1] <= q[3])) for q in rectangles])


class InformationGapTests(unittest.TestCase):
    def test_new_quadrant_information_separates_identical_existing_inputs(self):
        left = [.3125]*4+[.6875]*4
        right = [.125]+[.5]*6+[.875]
        points, hist, parent, quarters, _, runtime = example(left, right)
        queries = np.array([[0., .46875, 0., 1.], [1., 1.46875, 0., 1.]])
        observed = features_for(hist, queries)
        np.testing.assert_array_equal(observed[0], observed[1])
        np.testing.assert_array_equal(parent[0], parent[1])
        truth = raw_counts(points, queries)
        np.testing.assert_array_equal(truth, [4, 1])
        answers = np.array([prior.histogram_count(q, hist) for q in queries])
        np.testing.assert_array_equal(answers, [3.75, 3.75])
        np.testing.assert_array_equal(prior._actual_error(answers, truth) <= .25, [True, False])
        self.assertFalse(np.array_equal(quarters[0], quarters[1]))
        refs = np.array([runtime.reference_count(q) for q in queries])
        np.testing.assert_allclose(refs, truth, rtol=0, atol=1e-12)
        scores = abs(answers-refs)/np.maximum(refs, 1.)
        np.testing.assert_array_equal(scores <= .25, [True, False])

    def test_even_child_moments_do_not_identify_every_range_count(self):
        # Equal counts/moments within the left child, plus shared right mass.
        left = [.1875]*4+[.3125]*4+[.875]*8
        right = [.125]+[.25]*6+[.375]+[.875]*8
        points, hist, parent, quarters, children, runtime = example(left, right)
        queries = np.array([[0., .21875, 0., 1.], [1., 1.21875, 0., 1.]])
        observed = features_for(hist, queries)
        np.testing.assert_array_equal(observed[0], observed[1])
        np.testing.assert_array_equal(parent[0], parent[1])
        np.testing.assert_array_equal(quarters[0], quarters[1])
        np.testing.assert_array_equal(children[0], children[1])
        truth = raw_counts(points, queries)
        np.testing.assert_array_equal(truth, [4, 1])
        answers = np.array([prior.histogram_count(q, hist) for q in queries])
        np.testing.assert_array_equal(answers, [3.5, 3.5])
        np.testing.assert_array_equal(prior._actual_error(answers, truth) <= .25, [True, False])
        refs = [runtime.reference_count(q) for q in queries]
        self.assertEqual(refs[0], refs[1])


if __name__ == "__main__":
    unittest.main()
