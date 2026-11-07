"""Scientific and artifact-isolation checks for three-seed run comparison."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest

import numpy as np
import pandas as pd

from test_model_ablation import ExportFixture
from model_ablation import DEFAULT_VARIANTS, run_experiment
from compare_ablation_runs import find_compatible_baseline, combine_runs


def _hashes(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob("*") if p.is_file()}


def _clone_run(source, destination):
    shutil.copytree(source, destination)
    path = destination / "input_manifest.json"
    data = json.loads(path.read_text())
    for row in data:
        if row["used_path"].startswith(str(source)):
            row["used_path"] = str(destination) + row["used_path"][len(str(source)):]
    path.write_text(json.dumps(data, indent=2))
    return destination


class CompareAblationRunsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.fixture = ExportFixture(cls.root)
        cls.config = cls.fixture.config()
        cls.config["variants"] = DEFAULT_VARIANTS
        cls.config["seeds"] = "0"
        cls.baseline = run_experiment(cls.config)
        cls.next_config = dict(cls.config, seeds="1,2", baseline_run="auto")
        cls.current = run_experiment(cls.next_config)

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def test_auto_discovery_uses_compatible_seed_zero_run(self):
        self.assertEqual(find_compatible_baseline(dict(self.next_config)), self.baseline)
        config = dict(self.next_config, baseline_run=str(self.baseline))
        self.assertEqual(find_compatible_baseline(config), self.baseline)

    def test_rejects_changed_live_input_and_incompatible_parameters(self):
        path = self.fixture.data_dir / "queries-meta.csv"
        original = path.read_bytes()
        try:
            path.write_bytes(original + b"\n")
            with self.assertRaisesRegex((ValueError, FileNotFoundError), "(?i)(input|data|compatible|mismatch)"):
                find_compatible_baseline(dict(self.next_config))
        finally:
            path.write_bytes(original)
        with self.assertRaisesRegex((ValueError, FileNotFoundError), "(?i)(parameter|config|compatible|mismatch)"):
            find_compatible_baseline(dict(self.next_config, max_trees="9"))

    def test_auto_discovery_rejects_ambiguous_seed_zero_runs(self):
        duplicate = self.fixture.output_root / "synthetic-duplicate"
        try:
            _clone_run(self.baseline, duplicate)
            with self.assertRaisesRegex(ValueError, "(?i)(multiple|ambiguous)"):
                find_compatible_baseline(dict(self.next_config))
            self.assertEqual(find_compatible_baseline(dict(self.next_config, baseline_run=str(self.baseline))),
                             self.baseline)
        finally:
            if duplicate.exists():
                shutil.rmtree(duplicate)

    def test_preflight_rejects_incomplete_baseline_results_or_gate_details(self):
        for filename in ("v1_ablations.csv", "gate_details.csv"):
            with self.subTest(filename=filename), tempfile.TemporaryDirectory(dir=self.fixture.output_root) as temporary:
                baseline = _clone_run(self.baseline, Path(temporary) / "corrupt")
                path = baseline / filename
                rows = pd.read_csv(path)
                rows.iloc[1:].to_csv(path, index=False)
                config = dict(self.next_config, baseline_run=str(baseline))
                with self.assertRaisesRegex(ValueError, "(?i)(incompatible|missing|record|gate|result)"):
                    find_compatible_baseline(config)

    def test_combines_three_seeds_without_overwriting_either_run(self):
        with tempfile.TemporaryDirectory() as temporary:
            baseline = _clone_run(self.baseline, Path(temporary) / "baseline")
            current = _clone_run(self.current, Path(temporary) / "current")
            before0, before12 = _hashes(baseline), _hashes(current)
            summary, rate_summary, output = combine_runs(baseline, current)
            self.assertEqual(output, current / "comparison")
            self.assertEqual(_hashes(baseline), before0)
            for path, checksum in before12.items():
                self.assertEqual(hashlib.sha256((current / path).read_bytes()).hexdigest(), checksum)
            for name in ("combined_by_seed.csv", "summary_3_seeds.csv", "combined_gate_details.csv",
                         "per_rate_3_seeds.csv", "provenance.json"):
                self.assertTrue((output / name).is_file(), name)
            names = DEFAULT_VARIANTS.split(",")
            self.assertEqual(set(summary.variant), set(names))
            self.assertEqual(len(summary), len(names))
            self.assertTrue((summary.n_seed == 3).all())
            self.assertTrue((rate_summary.n_seed == 3).all())
            details = pd.read_csv(output / "combined_gate_details.csv")
            self.assertEqual(set(details.seed), {0, 1, 2})
            keys = ["variant", "seed", "score", "eps", "scope", "mergeRate"]
            self.assertFalse(details.duplicated(keys).any())
            rows = pd.read_csv(output / "combined_by_seed.csv")
            rows = rows[rows.score == "learned"]
            self.assertEqual(len(rows), 15)
            self.assertTrue((rows.score == "learned").all())
            for name in names:
                values = rows.loc[rows.variant == name, "coverage_per_rate"]
                actual = summary.loc[summary.variant == name, "coverage_per_rate_mean"].iloc[0]
                self.assertAlmostEqual(actual, values.mean())
            with self.assertRaises((FileExistsError, ValueError)):
                combine_runs(baseline, current)
            self.assertEqual(_hashes(baseline), before0)

    def test_rejects_seed_overlap_and_missing_variant_before_writing(self):
        with tempfile.TemporaryDirectory() as temporary:
            current = _clone_run(self.current, Path(temporary) / "current")
            meta_file = current / "run.json"
            meta = json.loads(meta_file.read_text())
            meta["config"]["seeds"] = "0,2"
            meta_file.write_text(json.dumps(meta))
            with self.assertRaisesRegex(ValueError, "(?i)(seed|overlap|duplicate)"):
                combine_runs(self.baseline, current)
            self.assertFalse((current / "comparison").exists())
        with tempfile.TemporaryDirectory() as temporary:
            current = _clone_run(self.current, Path(temporary) / "current")
            path = current / "v1_ablations.csv"
            data = pd.read_csv(path)
            data[data.variant != "v1_no_shift"].to_csv(path, index=False)
            with self.assertRaisesRegex(ValueError, "(?i)(variant|missing|incomplete|record)"):
                combine_runs(self.baseline, current)
            self.assertFalse((current / "comparison").exists())

    def test_rejects_different_input_or_training_code(self):
        for name in ("input", "code", "version"):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                current = _clone_run(self.current, Path(temporary) / "current")
                if name == "input":
                    path = current / "input_manifest.json"
                    meta = json.loads(path.read_text())
                    meta[0]["sha256"] = "0" * 64
                else:
                    path = current / "run.json"
                    meta = json.loads(path.read_text())
                    if name == "code":
                        meta["code_sha256"]["features.py"] = "0" * 64
                    else:
                        meta["package_versions"]["lightgbm"] = "0.0.0"
                path.write_text(json.dumps(meta))
                with self.assertRaisesRegex(ValueError, "(?i)(input|data|code|version|hash|mismatch|compatible)"):
                    combine_runs(self.baseline, current)
                self.assertFalse((current / "comparison").exists())

    def test_rejects_duplicate_or_missing_gate_details(self):
        for mutation in ("duplicate", "missing_rate"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as temporary:
                current = _clone_run(self.current, Path(temporary) / "current")
                path = current / "gate_details.csv"
                details = pd.read_csv(path)
                if mutation == "duplicate":
                    details = pd.concat([details, details.iloc[[0]]], ignore_index=True)
                else:
                    index = details.index[(details.variant == "v0") & (details.seed == 1) &
                                          (details.score == "learned") & (details.scope == "per_rate")][0]
                    details = details.drop(index)
                details.to_csv(path, index=False)
                with self.assertRaisesRegex(ValueError, "(?i)(duplicate|missing|rate|record|coverage)"):
                    combine_runs(self.baseline, current)
                self.assertFalse((current / "comparison").exists())

    def test_undefined_risk_remains_missing_after_combining(self):
        with tempfile.TemporaryDirectory() as temporary:
            current = _clone_run(self.current, Path(temporary) / "current")
            path = current / "v1_ablations.csv"
            rows = pd.read_csv(path)
            mask = (rows.variant == "v0") & (rows.seed == 1) & (rows.score == "learned")
            rows.loc[mask, ["violation", "violation_per_rate", "violation_pooled_per_rate_gates"]] = np.nan
            rows.loc[mask, ["coverage", "coverage_per_rate", "accepted_count", "violation_count",
                            "rates_with_acceptances", "accepted_cases_per_rate", "violating_cases_per_rate"]] = 0
            rows.to_csv(path, index=False)
            path = current / "gate_details.csv"
            details = pd.read_csv(path)
            mask = (details.variant == "v0") & (details.seed == 1) & (details.score == "learned")
            details.loc[mask, ["violation", "quality_target_met", "cutoff"]] = np.nan
            details.loc[mask, ["calibration_accepted_count", "calibration_violation_count", "accepted_count",
                               "violation_count", "coverage"]] = 0
            details.loc[mask, "accept_none"] = True
            details.to_csv(path, index=False)
            summary, rate_summary, output = combine_runs(self.baseline, current)
            combined = pd.read_csv(output / "combined_gate_details.csv")
            affected = combined[(combined.variant == "v0") & (combined.seed == 1) &
                                (combined.score == "learned")]
            self.assertTrue(affected.violation.isna().all())
            self.assertTrue((affected.accepted_count == 0).all())
            v0 = summary[summary.variant == "v0"].iloc[0]
            self.assertEqual(v0["violation_per_rate_n"], 2)
            self.assertEqual(v0.n_seed, 3)
            per_rate = rate_summary[rate_summary.variant == "v0"]
            expected = (combined[(combined.variant == "v0") & (combined.score == "learned") &
                                 (combined.scope == "per_rate")]
                        .groupby("mergeRate").violation.count())
            for row in per_rate.itertuples():
                self.assertEqual(row.defined_risk_seeds, expected[row.mergeRate])
                self.assertLessEqual(row.defined_risk_seeds, 2)


if __name__ == "__main__":
    unittest.main()
