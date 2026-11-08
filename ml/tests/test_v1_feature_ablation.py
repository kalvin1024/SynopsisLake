"""Actual training and provenance checks for leave-one-source-feature-out runs."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import lightgbm as lgb
import numpy as np
import pandas as pd

from test_model_ablation import ExportFixture
from features_v1 import FEATURES_V1, GROUPS, build_table_v1
from gate_eval import cutoff, split_ids
from model_ablation import run_experiment as run_existing_experiment
from v1_feature_ablation import DEFAULT_VARIANTS, plan_experiment, run_experiment, variants


def _digests(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def _clone_run(source, destination):
    shutil.copytree(source, destination)
    manifest = destination / "input_manifest.json"
    rows = json.loads(manifest.read_text())
    for row in rows:
        used = row["used_path"]
        if used.startswith(str(source)):
            row["used_path"] = str(destination) + used[len(str(source)):]
    manifest.write_text(json.dumps(rows))
    return destination


class V1FeatureAblationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.fixture = ExportFixture(cls.root)
        cls.config = dict(cls.fixture.config(), variants=DEFAULT_VARIANTS,
                          seeds="0,1,2", baseline_run="none")
        cls.inputs_before = (_digests(cls.fixture.data_root), _digests(cls.fixture.sample_root))
        cls.result = run_experiment(cls.config)
        cls.previous = run_existing_experiment(dict(cls.fixture.config(), variants="v0,v1", seeds="0"))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_registry_removes_only_named_features_and_retains_shift(self):
        registry = variants()
        self.assertEqual(DEFAULT_VARIANTS.split(","), ["v1", "v1_no_src_contrast", "v1_no_src_bound"])
        self.assertEqual(registry["v1"], FEATURES_V1)
        self.assertEqual(len(registry["v1"]), 28)
        for name, removed in (("v1_no_src_contrast", {"src_edge_contrast"}),
                              ("v1_no_src_bound", {"src_bound_rel"}),
                              ("v1_no_src_bound_contrast", {"src_edge_contrast", "src_bound_rel"})):
            self.assertEqual(registry[name], [feature for feature in FEATURES_V1 if feature not in removed])
            self.assertTrue(set(GROUPS["shift"]).issubset(registry[name]))

    def test_actual_nine_fits_preserve_splits_reload_models_and_calibrate_on_calibration(self):
        registry = variants()
        table = build_table_v1(str(self.fixture.sample_dir), str(self.fixture.data_dir))
        state = json.loads((self.result / "run.json").read_text())
        self.assertEqual(state["status"], "complete")
        self.assertEqual(len(list((self.result / "models").glob("*/seed_*"))), 9)
        for seed in (0, 1, 2):
            train_ids, calibration_ids, test_ids = split_ids(table.qId, seed)
            reference = None
            for name in DEFAULT_VARIANTS.split(","):
                folder = self.result / "models" / name / f"seed_{seed}"
                schema = json.loads((folder / "schema.json").read_text())
                self.assertEqual(schema["features"], registry[name])
                cohorts = pd.read_csv(folder / "cohorts.csv")
                for group, expected in (("train", train_ids), ("calibration", calibration_ids), ("test", test_ids)):
                    self.assertEqual(set(cohorts.loc[cohorts.split == group, "qId"]), expected)
                if reference is None:
                    reference = cohorts
                else:
                    pd.testing.assert_frame_equal(reference, cohorts)
                model = lgb.Booster(model_file=str(folder / "model.txt"))
                predictions = pd.read_csv(folder / "test_predictions.csv")
                test = table[table.qId.isin(test_ids)]
                actual = pd.DataFrame({"qId": test.qId, "mergeRate": test.mergeRate,
                                       "actual": model.predict(test[registry[name]], num_threads=2)})
                checked = predictions.merge(actual, on=["qId", "mergeRate"], validate="one_to_one")
                np.testing.assert_allclose(checked.score, checked.actual)
                calibration = table[table.qId.isin(calibration_ids)]
                score = model.predict(calibration[registry[name]], num_threads=2)
                cutoffs = json.loads((folder / "cutoffs.json").read_text())
                for gate in cutoffs:
                    mask = (np.ones(len(calibration), dtype=bool) if gate["scope"] == "pooled"
                            else calibration.mergeRate.to_numpy() == gate["mergeRate"])
                    threshold = cutoff(score[mask], calibration.relErr.to_numpy()[mask], .25, .05)
                    if np.isfinite(threshold):
                        self.assertAlmostEqual(gate["cutoff"], threshold)
                    else:
                        self.assertIsNone(gate["cutoff"])
                self.assertFalse(train_ids & calibration_ids or train_ids & test_ids or calibration_ids & test_ids)

    def test_summaries_and_paired_deltas_match_saved_results(self):
        for name in ("by_seed.csv", "summary.csv", "gate_details.csv", "per_rate_summary.csv", "paired_deltas.csv"):
            self.assertTrue((self.result / name).is_file(), name)
        rows = pd.read_csv(self.result / "by_seed.csv")
        learned = rows[rows.score == "learned"]
        self.assertEqual(len(learned), 9)
        self.assertFalse(learned.duplicated(["variant", "seed", "eps"]).any())
        summary = pd.read_csv(self.result / "summary.csv")
        self.assertEqual(set(summary.variant), set(DEFAULT_VARIANTS.split(",")))
        for row in summary.itertuples():
            data = learned[learned.variant == row.variant]
            self.assertAlmostEqual(row.coverage_per_rate_mean, data.coverage_per_rate.mean())
            self.assertAlmostEqual(row.coverage_per_rate_std, data.coverage_per_rate.std())
            self.assertAlmostEqual(row.violation_per_rate_mean, data.violation_per_rate.mean())
        delta = pd.read_csv(self.result / "paired_deltas.csv")
        self.assertEqual(len(delta), 6)
        for row in delta.itertuples():
            baseline = learned[(learned.variant == "v1") & (learned.seed == row.seed)].iloc[0]
            ablated = learned[(learned.variant == row.variant) & (learned.seed == row.seed)].iloc[0]
            for metric in ("coverage_per_rate", "violation_per_rate", "auc_per_rate"):
                expected = ablated[metric] - baseline[metric]
                actual = getattr(row, "delta_" + metric)
                if pd.isna(expected):
                    self.assertTrue(pd.isna(actual))
                else:
                    self.assertAlmostEqual(actual, expected)
        details = pd.read_csv(self.result / "gate_details.csv")
        empty = details[details.accepted_count == 0]
        self.assertTrue(empty.violation.isna().all())
        self.assertTrue((empty.violation_count == 0).all())

    def test_inputs_and_existing_runs_are_preserved_and_repeat_gets_new_directory(self):
        self.assertEqual(self.inputs_before, (_digests(self.fixture.data_root), _digests(self.fixture.sample_root)))
        before = _digests(self.result)
        second = run_experiment(dict(self.config, seeds="0"))
        self.assertNotEqual(second, self.result)
        self.assertEqual(_digests(self.result), before)
        self.assertEqual(self.inputs_before, (_digests(self.fixture.data_root), _digests(self.fixture.sample_root)))

    def test_reuses_only_compatible_full_v1_seed_zero_without_modifying_prior_run(self):
        config = dict(self.config, baseline_run=str(self.previous))
        before = _digests(self.previous)
        plan = plan_experiment(config)
        self.assertEqual(plan["fit_count"], 8)
        self.assertEqual(plan["reuse_count"], 1)
        result = run_experiment(config, plan=plan)
        self.assertEqual(_digests(self.previous), before)
        original = self.previous / "models/v1/seed_0/model.txt"
        copied = result / "models/v1/seed_0/model.txt"
        self.assertEqual(original.read_bytes(), copied.read_bytes())
        state = json.loads((result / "run.json").read_text())
        reused = [row for row in state["model_sources"] if row["reused"]]
        self.assertEqual([(row["variant"], row["seed"]) for row in reused], [("v1", 0)])
        self.assertEqual(len(pd.read_csv(result / "paired_deltas.csv")), 6)

    def test_explicit_no_shift_incompatible_and_incomplete_baselines_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            clone = _clone_run(self.previous, Path(temporary) / "prior")
            config = dict(self.config, baseline_run=str(clone), output_root=str(Path(temporary)))
            self.assertEqual(plan_experiment(config)["reuse_count"], 1)
            schema = clone / "models/v1/seed_0/schema.json"
            original = schema.read_bytes()
            data = json.loads(original)
            data["features"] = [feature for feature in FEATURES_V1 if feature not in GROUPS["shift"]]
            schema.write_text(json.dumps(data))
            with self.assertRaises((ValueError, FileNotFoundError)):
                plan_experiment(config)
            schema.write_bytes(original)
            state_file = clone / "run.json"
            original = state_file.read_bytes()
            data = json.loads(original)
            data["package_versions"]["lightgbm"] = "0.0.0"
            state_file.write_text(json.dumps(data))
            with self.assertRaises((ValueError, FileNotFoundError)):
                plan_experiment(config)
            state_file.write_bytes(original)
            (clone / "models/v1/seed_0/test_predictions.csv").unlink()
            with self.assertRaises((ValueError, FileNotFoundError)):
                plan_experiment(config)

    def test_ambiguous_auto_baseline_requires_explicit_selection(self):
        duplicate = self.fixture.output_root / "synthetic-duplicate-full-v1"
        try:
            _clone_run(self.previous, duplicate)
            with self.assertRaisesRegex(ValueError, "(?i)(multiple|ambiguous)"):
                plan_experiment(dict(self.config, baseline_run="auto"))
        finally:
            shutil.rmtree(duplicate)

    def test_invalid_settings_fail_before_creating_results(self):
        before = set(self.fixture.output_root.iterdir())
        for update in ({"variants": "v1,v1_no_shift"}, {"variants": "v1,v1"}, {"seeds": "0,0"},
                       {"generalization": "true"}, {"epsilons": "nan"},
                       {"output_root": str(self.fixture.data_root / "results")}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                plan_experiment(dict(self.config, **update))
        self.assertEqual(set(self.fixture.output_root.iterdir()), before)


if __name__ == "__main__":
    unittest.main()
