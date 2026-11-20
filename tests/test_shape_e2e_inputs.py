"""Source-only construction and identity guards for Java E2E metadata."""
import inspect
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from ml import bounded_sidecar as bounded
from ml import hierarchical_shape
from ml import local_shape
from ml import shape_e2e_inputs
from ml import sidecar_workflow as prior


def source_fixture(root):
    run, raw = root / "histograms", root / "raw"
    run.mkdir()
    raw.mkdir()
    all_points, hists = [], []
    for sid in range(2):
        rng = np.random.default_rng(100+sid)
        points = rng.uniform(.01, 1.99, (43+sid*11, 2))
        bx = by = np.array([0., 1., 2.])
        counts = np.histogram2d(points[:, 0], points[:, 1], bins=(bx, by))[0]
        hist = (1., bx, by, counts)
        lines = ["1.0", "0.0,1.0,2.0", "0.0,1.0,2.0"]
        lines.extend(",".join(str(int(value)) for value in row) for row in counts)
        (run / f"src_{sid}.txt").write_text("\n".join(lines)+"\n")
        np.savetxt(raw / f"{sid}.csv", points, delimiter="\t", fmt="%.17g")
        all_points.append(points)
        hists.append(hist)
    # These must never be loaded by source-only construction.
    (run / "rows.csv").write_text("invalid query labels; deliberately not a table\n")
    (run / "queries-Ans.txt").write_text("invalid queries\n")
    return run, raw, all_points, hists


class SourceShapeE2EInputs(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.run, self.raw, self.points, self.hists = source_fixture(self.root)

    def tearDown(self):
        self.temp.cleanup()

    def build(self, **kwargs):
        return shape_e2e_inputs.prepare_shape_metadata(
            self.run, self.raw, self.root / "metadata", packet_budget_bytes=512,
            chunk_size=7, **kwargs)

    def test_reuses_existing_native_and_hierarchical_codecs_without_query_labels(self):
        with mock.patch.object(prior, "_load_queries", side_effect=AssertionError("No query input")):
            path = self.build()
        state = shape_e2e_inputs.verify_shape_metadata(path, self.run)
        self.assertEqual(state["raw_rows"], sum(map(len, self.points)))
        self.assertEqual(state["source_count"], 2)
        self.assertFalse(state["selection_uses_query_labels"])
        self.assertFalse(state["selection_uses_query_geometry"])
        self.assertNotIn("raw_cache", {p.name for p in path.iterdir()})
        hists, native = local_shape.load_native(path / "native")
        all_counts, child, encoded = [], [], []
        for hist, observed, points in zip(hists, native, self.points):
            counts, sums, _ = prior._sufficient(points, hist)
            np.testing.assert_array_equal(counts, hist[3].ravel())
            np.testing.assert_array_equal(observed, bounded.encode_moments(prior._moments(counts, sums)))
            quarter, quarter_sums = hierarchical_shape.accumulate_child_moments(points, hist)
            child.append(quarter)
            encoded.append(hierarchical_shape.encode_child_moments(quarter, quarter_sums))
            all_counts.append(counts)
        expected = hierarchical_shape.build_selection(hists, native, child, encoded, 512)
        self.assertEqual((path / "shape.bin").read_bytes(), expected)
        self.assertEqual(state["packet_bytes"], len(expected))
        self.assertLessEqual(state["packet_bytes"], 512)
        self.assertEqual(state["auxiliary_bytes"],
                         sum(p.stat().st_size for p in (path / "native").iterdir())+len(expected))
        signature = inspect.signature(shape_e2e_inputs.prepare_shape_metadata)
        self.assertFalse(any("quer" in name or "label" in name for name in signature.parameters))

    def test_truncated_raw_data_fails_and_keeps_failure_record(self):
        np.savetxt(self.raw / "0.csv", self.points[0][:-1], delimiter="\t")
        with self.assertRaisesRegex(ValueError, "Raw counts do not reproduce"):
            self.build()
        state = json.loads((self.root / "metadata/manifest.json").read_text())
        self.assertEqual(state["status"], "failed")
        with self.assertRaisesRegex(ValueError, "not a completed"):
            shape_e2e_inputs.verify_shape_metadata(self.root / "metadata")

    def test_weighted_mass_is_not_substituted_for_raw_row_counts(self):
        path = self.run / "src_0.txt"
        lines = path.read_text().splitlines()
        lines[3:] = [",".join(str(2*int(float(x))) for x in row.split(",")) for row in lines[3:]]
        path.write_text("\n".join(lines)+"\n")
        with self.assertRaisesRegex(ValueError, "Raw counts do not reproduce"):
            self.build()

    def test_payload_tampering_and_wrong_source_run_are_rejected(self):
        path = self.build()
        shape_file = path / "shape.bin"
        original = shape_file.read_bytes()
        shape_file.write_bytes(original+b"tamper")
        with self.assertRaisesRegex(ValueError, "artifact changed"):
            shape_e2e_inputs.verify_shape_metadata(path, self.run)
        shape_file.write_bytes(original)
        source = self.run / "src_1.txt"
        source.write_text(source.read_text()+"\n")
        with self.assertRaisesRegex(ValueError, "different source exports"):
            shape_e2e_inputs.verify_shape_metadata(path, self.run)

    def test_completed_metadata_is_never_overwritten(self):
        path = self.build()
        original = (path / "manifest.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.build()
        self.assertEqual((path / "manifest.json").read_bytes(), original)

    def test_source_raw_ids_and_regular_grids_are_checked(self):
        (self.raw / "2.csv").write_text("0.1\t0.2\n")
        with self.assertRaisesRegex(ValueError, "IDs exactly"):
            self.build()
        (self.raw / "2.csv").unlink()
        source = self.run / "src_0.txt"
        source.write_text(source.read_text().replace("0.0,1.0,2.0", "0.0,0.8,2.0", 1))
        with self.assertRaisesRegex(ValueError, "regular source grid"):
            self.build()

    def test_timestamp_column_is_not_used_as_a_point_weight(self):
        points = np.column_stack((self.points[0], np.full(len(self.points[0]), 123456789.)))
        np.savetxt(self.raw / "0.csv", points, delimiter="\t", fmt="%.17g")
        path = self.build()
        state = shape_e2e_inputs.verify_shape_metadata(path, self.run)
        self.assertEqual(state["raw_rows"], sum(map(len, self.points)))

    def test_leading_zero_ids_cannot_hide_duplicate_java_sources(self):
        alias = self.run / "src_00.txt"
        alias.write_bytes((self.run / "src_0.txt").read_bytes())
        with self.assertRaisesRegex(ValueError, "canonical, unique"):
            self.build()
        alias.unlink()
        (self.raw / "00.csv").write_bytes((self.raw / "0.csv").read_bytes())
        with self.assertRaisesRegex(ValueError, "canonical filenames"):
            self.build()


if __name__ == "__main__":
    unittest.main()
