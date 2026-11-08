import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits


SOURCE = Path(__file__).resolve().parents[1] / "scripts/research_20261109_shared_feature_fit.py"
spec = importlib.util.spec_from_file_location("fit_1109", SOURCE)
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


def capture(root, name, x, z, y, selected, alpha):
    folder = root / name
    folder.mkdir()
    arrays = {"x": x, "shared_z": z, "expert_outputs": y,
              "selected_experts": selected, "routing_weights": alpha,
              "prompt_ids": np.asarray([f"{name}-{i}" for i in range(len(x))])}
    records = {}
    for key, array in arrays.items():
        p = folder / f"{key}.npy"
        np.save(p, array)
        records[key] = {"path": p.name}
    path = folder / "manifest.json"
    path.write_text(json.dumps({"expert_ids": [14, 40], "arrays": records}))
    return path


class EqualizedReadoutTests(unittest.TestCase):
    def test_routed_weighting_prioritizes_selected_teacher(self):
        rng = np.random.default_rng(1109)
        x = rng.normal(size=(500, 3)).astype(np.float32)
        encoded = fit.encode(x)
        selected = rng.random(len(x)) < .3
        a2 = selected.astype(np.float64)
        a = rng.normal(size=(3, 4))
        b = rng.normal(size=(3, 4))
        y = np.where(selected[:, None], encoded["qdq"] @ a, encoded["qdq"] @ b).astype(np.float32)
        scores = []
        with threadpool_limits(limits=1):
            for beta in [0, .99]:
                weights = fit.blend_weights(a2, beta)
                gram = fit.prepare_gram(encoded["qdq"], weights)
                mean, rhs = fit.solve_components(gram, y)
                model = fit.solve_one(gram, mean, rhs, 1e-5)
                fit.recalibrate_bias(encoded, model, y, weights)
                pred = fit.predict_encoded(encoded, model)
                scores.append(fit.weighted_nmse(y, pred, a2))
        self.assertLess(scores[1], scores[0] * .01)

    def test_screen_exports_runtime_equalization_and_opens_only_train_valid(self):
        rng = np.random.default_rng(919)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paths = {}
            for name, n in (("train", 200), ("valid", 60)):
                x = rng.normal(size=(n, 7)).astype(np.float32)
                z = np.concatenate((x ** 2, np.sin(x)), axis=1)
                z[:, 0] *= 1000
                y = np.stack((x, x ** 2), axis=1)
                selected = np.tile(np.asarray([14, 40], np.int32), (n, 1))
                alpha = np.full((n, 2), .5, np.float32)
                paths[name] = capture(root, name, x, z, y, selected, alpha)
            args = argparse.Namespace(train=str(paths["train"]), valid=str(paths["valid"]), layer=1,
                                      output=str(root / "fit"), train_target_manifest=None, valid_target_manifest=None,
                                      experts=None, bases=["raw_x_eq", "shared_z_eq", "hybrid_x_z4"],
                                      betas=[0, .99], lambdas=[.03])
            with threadpool_limits(limits=1):
                fit.fit_screen(args)
            result = json.loads((root / "fit/selection.json").read_text())
            self.assertEqual(result["status"], "VALID_screening_no_TEST_access")
            self.assertEqual(len(result["models"]), 6)
            self.assertEqual(len(result["grid"]), 12)
            model = next(m for m in result["models"] if m["basis"] == "hybrid_x_z4")
            prefix = model["transform_prefix"]
            sources = np.fromfile(prefix + ".sources.u8", np.uint8)
            self.assertEqual(sources.tolist(), [0] * 7 + [1] * 4)
            center = np.fromfile(prefix + ".center.f32", np.float32)
            gain = np.fromfile(prefix + ".inv_scale.f32", np.float32)
            self.assertEqual(len(center), 11)
            self.assertEqual(len(gain), 11)
            self.assertEqual(Path(model["readout_prefix"] + ".i8").stat().st_size, 7 * 11)
            self.assertEqual(model["cost"]["additional_GU_coefficients"], 0)
            self.assertTrue(all(len(line.split("\t")) == 6 for line in (root / "fit/registry.hybrid_x_z4.tsv").read_text().splitlines()))


if __name__ == "__main__":
    unittest.main()
