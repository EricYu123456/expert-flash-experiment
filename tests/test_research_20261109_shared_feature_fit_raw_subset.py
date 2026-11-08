import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np

PATH = Path(__file__).resolve().parents[1] / "scripts/research_20261109_shared_feature_fit_raw_subset.py"
spec = importlib.util.spec_from_file_location("raw_subset_under_test", PATH)
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


class RawSubsetContract(unittest.TestCase):
    def test_ranking_uses_deployed_row_scales_and_coordinate_ties(self):
        with tempfile.TemporaryDirectory() as folder:
            prefix = str(Path(folder) / "C")
            # Equal code norms can represent very different coefficient norms
            # because row-W8 scales differ. Coordinates 0/2 intentionally tie.
            np.asarray([[4, 0, 4], [0, 1, 0]], np.int8).tofile(prefix + ".i8")
            np.asarray([1., 10.], np.float32).tofile(prefix + ".scales.f32")
            np.zeros(2, np.float32).tofile(prefix + ".bias.f32")
            ranking, importance = fit.column_ranking([{"width": 3, "prototype": None, "readout_prefix": prefix}])
            np.testing.assert_array_equal(ranking, [1, 0, 2])
            np.testing.assert_array_equal(importance, [16., 100., 16.])

    def test_gather_precedes_new_vector_A8_boundary(self):
        raw = np.asarray([[1., 1000., 3.]], np.float32)
        class Capture:
            def get(self, name):
                assert name == "x"
                return raw
        class Dataset:
            capture = Capture()
        transform = {"sources": np.zeros(2, np.uint8), "indices": np.asarray([2, 0], np.int32),
                     "center": np.asarray([1., 0.], np.float32), "inv_scale": np.asarray([.5, 2.], np.float32)}
        actual = fit.gathered_encoded(Dataset(), transform)
        # Gather+affine gives [1,2]; the removed 1000 cannot set A8 scale.
        np.testing.assert_array_equal(actual["q"], [[64, 127]])
        np.testing.assert_array_equal(actual["scale"], np.asarray([2. / 127], np.float32))


if __name__ == "__main__":
    unittest.main()
