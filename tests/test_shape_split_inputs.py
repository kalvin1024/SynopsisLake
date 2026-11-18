"""Grouped split handoff preserves identities and fails closed on changed roles."""
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np
import pandas as pd

from ml import shape_split_inputs as split
from ml import sidecar_workflow as prior


def fixture(root):
    hists, raw = root/"histograms", root/"raw"
    hists.mkdir()
    raw.mkdir()
    edges = ",".join(map(str, range(11)))
    text = "1\n"+edges+"\n0,1\n"+"10\n"*10
    (hists/"src_0.txt").write_text(text)
    (hists/"arm_130.txt").write_text(text)
    (raw/"0.csv").write_text("0.5\t0.5\n")  # Presence only; splitting must never read raw points.
    rows = [dict(qId=10+3*(2*i+j), x0=i+.05+j*.02, x1=i+.95+j*.02,
                 y0=.05, y1=.95, truth=10) for i in range(10) for j in range(2)]
    rows.append(dict(qId=999, x0=.1, x1=.9, y0=.1, y1=.9, truth=0))
    queries = pd.DataFrame(rows)
    path = root/"queries.csv"
    queries.to_csv(path, index=False)
    config = dict(histogram_dir=str(hists), raw_data_dir=str(raw), queries_file=str(path),
                  repo_root=str(root), output_root=str(root/"results"), seeds=[0, 1, 2], min_exact=10)
    return config, queries


class GroupedSplitHandoff(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.config, self.queries = fixture(self.root)

    def tearDown(self):
        self.temporary.cleanup()

    def generate(self, **overrides):
        config = {**self.config, **overrides}
        return split.prepare_grouped_split(config)

    def test_generation_preserves_original_ids_and_every_cell_has_one_role(self):
        config, audit = self.generate()
        roles = pd.read_csv(config["split_file"])
        expected = set(self.queries.loc[self.queries.truth >= 10, "qId"])
        for seed, part in roles.groupby("seed"):
            self.assertEqual(set(part.qId), expected)
            self.assertEqual(set(part.split), set(split.ROLES))
            self.assertTrue(part.groupby("cell").split.nunique().eq(1).all())
        self.assertNotIn(999, roles.qId)
        manifest = json.loads(audit.read_text())
        self.assertEqual(manifest["excluded_queries"], 1)
        self.assertFalse(manifest["historical_authors_byte_identical"])
        self.assertFalse(manifest["grouping_uses_labels"])
        self.assertFalse(manifest["role_selection_uses_labels"])

    def test_rerun_is_byte_identical_and_does_not_overwrite_prior_artifacts(self):
        first, a = self.generate()
        second, b = self.generate()
        self.assertNotEqual(a.parent, b.parent)
        self.assertEqual(Path(first["split_file"]).read_bytes(), Path(second["split_file"]).read_bytes())

    def test_source_cell_boundary_ownership_matches_original_left_search(self):
        q = self.queries.iloc[:1].copy()
        q.loc[:, ["x0", "x1"]] = [.5, 1.5]
        cells, _ = split.source_cell_ids(q, self.config["histogram_dir"])
        self.assertEqual(cells.iloc[0], "0_0_0")

    def test_raw_points_are_not_read_to_select_groups_or_roles(self):
        with mock.patch.object(prior, "iter_raw", side_effect=AssertionError("raw scan forbidden")):
            self.generate()

    def test_supplied_grouped_split_and_sha_are_reused(self):
        generated, _ = self.generate()
        source = Path(generated["split_file"])
        digest = prior._sha(source)
        config, audit = split.prepare_grouped_split({**self.config, "split_file": str(source)},
                                                   expected_sha256=digest)
        self.assertEqual(Path(config["split_file"]), source)
        self.assertEqual(json.loads(audit.read_text())["action"], "reused_existing_grouped_cell_split")
        with self.assertRaisesRegex(ValueError, "SHA256"):
            split.prepare_grouped_split({**self.config, "split_file": str(source)}, expected_sha256="0"*64)

    def test_explicit_noncell_split_never_silently_replaced(self):
        path = self.root/"query_splits.csv"
        pd.DataFrame(dict(seed=[0], qId=[10], split=["train"])).to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "not valid grouped-cell"):
            self.generate(split_file=str(path))
        config, audit = split.prepare_grouped_split({**self.config, "split_file": str(path)}, policy="generate")
        self.assertNotEqual(config["split_file"], str(path))
        self.assertEqual(json.loads(audit.read_text())["policy"], "generate")

    def test_invalid_grouped_split_overlap_is_rejected(self):
        generated, _ = self.generate()
        path = Path(generated["split_file"])
        frame = pd.read_csv(path)
        byrole = frame[frame.seed == 0].groupby("split").first()
        a, b = byrole.iloc[0], byrole.iloc[1]
        frame.loc[(frame.seed == 0) & (frame.qId == a.qId), "cell"] = b.cell
        frame.to_csv(path, index=False)
        with self.assertRaisesRegex(ValueError, "not valid grouped-cell"):
            self.generate(split_file=str(path))

    def test_inherit_policy_requires_a_supplied_or_inherited_file(self):
        with self.assertRaisesRegex(FileNotFoundError, "No inherited"):
            split.prepare_grouped_split(self.config, policy="inherit")

    def test_ordinary_inherited_query_split_gets_explicit_generated_manifest(self):
        run = self.root/"prior-run"
        run.mkdir()
        (run/"run.json").write_text(json.dumps(dict(config=self.config)))
        pd.DataFrame(dict(seed=[0], qId=[10], split=["train"], split_kind=["query"])).to_csv(run/"splits.csv", index=False)
        config, audit = self.generate(run_dir=str(run))
        manifest = json.loads(audit.read_text())
        self.assertIn("ordinary query split", manifest["decision_reason"])
        self.assertEqual(manifest["action"], "generated_declared_grouped_cell_split")
        self.assertNotEqual(config["split_file"], str(run/"splits.csv"))

    def test_known_input_generation_cannot_claim_reproduction_after_recipe_change(self):
        actual, recipe, _ = split._historical_identity(Path(self.config["queries_file"]),
            [Path(self.config["histogram_dir"])/"src_0.txt"], [0, 1, 2], 10)
        with mock.patch.object(split, "_historical_identity", return_value=(actual, recipe, True)):
            with self.assertRaisesRegex(ValueError, "byte-identical"):
                self.generate()

    def test_invalid_queries_and_seed_settings_fail(self):
        bad = self.queries.copy()
        bad.loc[0, "qId"] = bad.qId.iloc[1]
        with self.assertRaisesRegex(ValueError, "unique integer"):
            split.grouped_memberships(bad, self.config["histogram_dir"])
        bad = self.queries.copy()
        bad.loc[0, "truth"] = np.nan
        with self.assertRaisesRegex(ValueError, "nonnegative integer"):
            split.grouped_memberships(bad, self.config["histogram_dir"])
        for seeds in ([0, 0], [True], [-1]):
            with self.assertRaises(ValueError):
                split.grouped_memberships(self.queries, self.config["histogram_dir"], seeds=seeds)

    def test_generated_roles_feed_direct_raw_audited_preparation_without_old_notebooks(self):
        from ml import shape_inputs, local_shape_workflow
        from tests.test_sidecar_workflow import fixture as raw_fixture
        root = self.root/"real-file-fixture"
        root.mkdir()
        config, queries, _ = raw_fixture(root)
        config["split_file"] = ""
        config["min_exact"] = 10
        handoff, manifest = split.prepare_grouped_split(config)
        with mock.patch.object(prior, "run_experiment", side_effect=AssertionError("R09 must not run")):
            prepared = shape_inputs.prepare_inputs(handoff)
        _, state = local_shape_workflow._load_preparation(prepared)
        self.assertEqual(state["status"], "prepared")
        self.assertEqual(state["raw_rows"], 108)
        roles = pd.read_csv(prepared/"outer_splits.csv")
        self.assertEqual(set(roles.qId), set(queries.qId))
        self.assertEqual(len(roles), 3*len(queries))
        self.assertTrue(pd.read_csv(prepared/"raw_truth_parity.csv").matches.all())
        self.assertTrue(manifest.is_file())


if __name__ == "__main__":
    unittest.main()
