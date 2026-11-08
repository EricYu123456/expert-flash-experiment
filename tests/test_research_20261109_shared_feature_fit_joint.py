import importlib.util
from pathlib import Path
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits


SOURCE = Path(__file__).resolve().parents[1] / "scripts/research_20261109_shared_feature_fit_joint.py"
spec = importlib.util.spec_from_file_location("joint_fit", SOURCE)
spec.loader.exec_module(joint := importlib.util.module_from_spec(spec))


class JointCalibrationTests(unittest.TestCase):
    def test_g_design_recovers_shared_residual_without_dividing_by_zero(self):
        rng = np.random.default_rng(1909)
        encoded = joint.base.encode(rng.normal(size=(600, 7)).astype(np.float32))
        r = rng.normal(size=(7, 5))
        b = rng.normal(size=5)
        g = rng.uniform(.1, 2, size=600)
        g[:40] = 0
        residual = g[:, None] * (encoded["qdq"] @ r + b)
        # Zero-g rows cannot be affected by this group and must not contaminate
        # its regression through a division-by-zero workaround.
        residual[:40] = 10000
        with threadpool_limits(limits=1):
            result = joint.fit_correction(encoded, residual, g, 1e-10)
        np.testing.assert_allclose(result["R"].T, r, atol=1e-6)
        np.testing.assert_allclose(result["bias_R"], b, atol=1e-6)
        self.assertEqual(result["nonzero_g_rows"], 560)

    def test_eta_zero_preserves_original_payload_and_nonzero_keeps_shape(self):
        rng = np.random.default_rng(1199)
        with tempfile.TemporaryDirectory() as temp:
            prefix = str(Path(temp) / "base")
            q, sw = joint.old.quantize_rows(rng.normal(size=(5, 7)).astype(np.float32))
            bias = rng.normal(size=5).astype(np.float32)
            q.tofile(prefix + ".i8")
            sw.tofile(prefix + ".scales.f32")
            bias.tofile(prefix + ".bias.f32")
            entry = {"readout_prefix": prefix, "width": 7}
            correction = {"R": rng.normal(size=(5, 7)).astype(np.float32), "bias_R": rng.normal(size=5).astype(np.float32)}
            hashes = [joint.old.sha(prefix + suffix) for suffix in (".i8", ".scales.f32", ".bias.f32")]
            zero = joint.fold(entry, correction, 0)
            self.assertNotIn("_payload", zero)
            self.assertEqual(zero["readout_prefix"], prefix)
            self.assertEqual(hashes, [joint.old.sha(prefix + suffix) for suffix in (".i8", ".scales.f32", ".bias.f32")])
            changed = joint.fold(entry, correction, .5)
            self.assertEqual(changed["_payload"]["qweight"].shape, q.shape)
            self.assertEqual(changed["_payload"]["bias"].shape, bias.shape)


if __name__ == "__main__":
    unittest.main()
