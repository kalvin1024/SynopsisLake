"""Exercise real LightGBM export with file append disabled."""
import builtins
import errno
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml import region_budget_comparison as comparison
from ml import shape_team_model


class VolumeModelExportTests(unittest.TestCase):
    def test_export_never_appends_and_matches_legacy_model_bytes(self):
        rng = np.random.default_rng(42)
        names = shape_team_model.TEAM_FEATURES
        frame = pd.DataFrame(rng.random((512, len(names))), columns=names)
        frame["qId"] = np.arange(len(frame))
        frame["actual_error"] = .01 + .2 * frame[names[0]]
        frame["split"] = "train"

        def disable_append(open_file):
            def guarded_open(file, mode="r", *args, **kwargs):
                if "a" in mode or "+" in mode:
                    raise OSError(errno.ESPIPE, "Illegal seek")
                return open_file(file, mode, *args, **kwargs)
            return guarded_open

        with tempfile.TemporaryDirectory() as root:
            volume = Path(root).resolve()
            legacy = shape_team_model.fit_team_model(
                frame, 0, volume/"legacy", model_name="region_west_full_v1")
            with (patch("builtins.open", side_effect=disable_append(builtins.open)),
                  patch("io.open", side_effect=disable_append(io.open))):
                with self.assertRaisesRegex(OSError, "Illegal seek"):
                    legacy.booster.save_model(str(volume/"old.txt"))
                comparison.check_model_export(volume)
                fitted = comparison._fit_and_publish_team_model(frame, 0, volume/"new")

            self.assertEqual(set(p.name for p in volume.iterdir()), {"legacy", "old.txt", "new"})
            self.assertGreater(legacy.schema["best_iteration"], 1)
            for filename in shape_team_model.MODEL_FILES:
                with self.subTest(filename=filename):
                    self.assertEqual((legacy.directory/filename).read_bytes(),
                                     (fitted.directory/filename).read_bytes())
            historical = shape_team_model.load_team_model(legacy.directory)
            loaded = shape_team_model.load_team_model(fitted.directory)
            np.testing.assert_array_equal(historical.log_scores(frame), loaded.log_scores(frame))
            self.assertEqual({k: v for k, v in fitted.training_metadata.items() if k != "training_seconds"},
                             {k: v for k, v in legacy.training_metadata.items() if k != "training_seconds"})


if __name__ == '__main__':
    unittest.main()
