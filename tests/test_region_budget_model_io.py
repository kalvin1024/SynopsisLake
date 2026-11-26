"""Exercise real LightGBM export against an append-rejecting Volume stand-in."""
import builtins
import errno
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from ml import region_budget_comparison as comparison
from ml import shape_team_model


class VolumeModelExportTests(unittest.TestCase):
    def test_real_export_stages_append_locally_and_reloads_identically(self):
        rng = np.random.default_rng(42)
        frame = pd.DataFrame(rng.random((120, 28)), columns=shape_team_model.TEAM_FEATURES)
        frame['qId'] = np.arange(len(frame))
        frame['actual_error'] = rng.uniform(.01, .5, len(frame))
        frame['split'] = 'train'
        original_open = builtins.open
        with tempfile.TemporaryDirectory() as root:
            volume = Path(root).resolve()

            def no_volume_append(file, mode='r', *args, **kwargs):
                if isinstance(file, (str, bytes, Path)):
                    path = Path(file).resolve()
                    if path.is_relative_to(volume) and ('a' in mode or '+' in mode):
                        raise OSError(errno.ESPIPE, 'Illegal seek')
                return original_open(file, mode, *args, **kwargs)

            with patch('builtins.open', side_effect=no_volume_append):
                with self.assertRaisesRegex(OSError, 'Illegal seek'):
                    shape_team_model.fit_team_model(frame, 0, volume/'direct')
                fitted = comparison._fit_and_publish_team_model(frame, 0, volume/'staged')
            self.assertEqual(fitted.directory, volume/'staged')
            self.assertTrue((fitted.directory/'training.json').is_file())
            self.assertIn('pandas_categorical:', (fitted.directory/'model.txt').read_text())
            self.assertEqual(comparison._sha(fitted.directory/'model.txt'), fitted.schema['model_sha256'])
            loaded = shape_team_model.load_team_model(fitted.directory)
            np.testing.assert_array_equal(fitted.log_scores(frame), loaded.log_scores(frame))


if __name__ == '__main__':
    unittest.main()
