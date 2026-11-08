import importlib.util
from pathlib import Path
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "scripts/research_20261109_shared_feature_fit_constant.py"
spec = importlib.util.spec_from_file_location("constant_control_under_test", PATH)
control = importlib.util.module_from_spec(spec)
spec.loader.exec_module(control)


class ConstantControl(unittest.TestCase):
    def test_unselected_targets_cannot_affect_constants(self):
        y = np.asarray([[1., -2.], [10000., 10000.], [5., 6.]], np.float32)
        a2 = np.asarray([1., 0., 3.])
        result = control.selected_constants(y, a2)
        np.testing.assert_array_equal(result["drop_zero"], [0., 0.])
        np.testing.assert_array_equal(result["selected_mean"], [3., 2.])
        np.testing.assert_array_equal(result["selected_alpha2_mean"], [4., 4.])
        error = lambda bias: float(np.sum(a2[:, None] * (y.astype(np.float64) - bias) ** 2))
        self.assertLess(error(result["selected_alpha2_mean"]), error(result["selected_mean"]))
        self.assertLess(error(result["selected_alpha2_mean"]), error(result["drop_zero"]))
        with self.assertRaises(ValueError):
            control.selected_constants(y, np.zeros(3))


if __name__ == "__main__":
    unittest.main()
