"""Shared-feature fit protocol guards, independent synthetic teacher, and exports."""
import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.special import expit
from threadpoolctl import threadpool_limits


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/research_20261108_shared_feature_fit.py"
spec = importlib.util.spec_from_file_location("shared_feature_fit", SCRIPT)
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


def write_capture(root, split, x, z, y, fs, prompt_ids):
    folder = root / split
    folder.mkdir(parents=True)
    ids = np.tile(np.asarray([14, 40, 7], np.int32), (len(x), 1))
    alpha = np.tile(np.asarray([.2, .3, .5], np.float32), (len(x), 1))
    third = np.ones_like(y[:, 0]) * np.float32(.125)
    arrays = {"x": x, "shared_z": z, "shared_output": fs, "expert_outputs": y,
              "prompt_ids": np.asarray(prompt_ids), "selected_experts": ids,
              "routing_weights": alpha, "routed_outputs": np.stack((y[:, 0], y[:, 1], third), axis=1)}
    records = {}
    for name, value in arrays.items():
        p = folder / (name + ".npy")
        np.save(p, value)
        records[name] = {"path": p.name, "dtype": str(value.dtype), "shape": list(value.shape)}
    manifest = folder / "manifest.json"
    manifest.write_text(json.dumps({"expert_ids": [14, 40], "arrays": records}))
    return manifest


class SharedFeatureFitTests(unittest.TestCase):
    def test_full_fit_freezes_before_test_target_and_recovers_shared_teacher(self):
        rng = np.random.default_rng(118)
        h, r, n = 9, 24, 1300
        x = rng.normal(size=(n, h)).astype(np.float32)
        gate, up = rng.normal(size=(h, r)), rng.normal(size=(h, r))
        z = ((x @ gate) * expit(x @ gate) * (x @ up)).astype(np.float32)
        y = np.stack([(z @ rng.normal(size=(r, h))).astype(np.float32) for _ in range(2)], axis=1)
        fs = (z @ rng.normal(size=(r, h))).astype(np.float32)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            parts = {"train": slice(0, 200), "valid": slice(200, 300), "test": slice(300, 1300)}
            manifests = {name: write_capture(root, name, x[sl], z[sl], y[sl], fs[sl],
                                              [f"{name}-{i}" for i in range(len(x[sl]))])
                         for name, sl in parts.items()}
            output = root / "fit"
            original = fit.Capture.get
            test_accesses = []

            def guarded_get(cap, name):
                if cap.path == manifests["test"].resolve() and name == "expert_outputs":
                    self.assertTrue((output / "selection.json").exists())
                    test_accesses.append(name)
                return original(cap, name)

            args = argparse.Namespace(**{k: str(v) for k, v in manifests.items()}, output=str(output),
                                      minimum_test_prompts=1000, seed=118, layer=1,
                                      lambdas=[1e-8, .01], bases=list(fit.BASES), allow_underdetermined=False)
            with patch.object(fit.Capture, "get", guarded_get), threadpool_limits(limits=1):
                fit.fit_all(args)
            self.assertTrue(test_accesses)
            selection = json.loads((output / "selection.json").read_text())
            report = json.loads((output / "test_report.json").read_text())
            self.assertEqual(report["test_prompts"], 1000)
            shared = [m for m in report["per_expert"] if m["basis"] == "shared_z" and m["precision"] == "fp"]
            self.assertTrue(all(m["nmse"] < 1e-8 for m in shared))
            raw = [m for m in report["per_expert"] if m["basis"] == "x" and m["precision"] == "fp"]
            self.assertTrue(all(m["nmse"] > .1 for m in raw))
            self.assertEqual(len((output / "registry.tsv").read_text().splitlines()), 10)
            self.assertEqual(len(selection["models"]), 20)
            deploy = output / "shared_z.expert14.deploy.i8"
            self.assertEqual(deploy.stat().st_size, h * r)
            ag = next(m for m in report["aggregates"] if m["basis"] == "shared_z" and m["precision"] == "fp")
            self.assertEqual(ag["selected_slot_fraction"], 2 / 3)
            self.assertLess(ag["routed_with_native_fallback"]["nmse"], 1e-8)
            with self.assertRaises(FileExistsError):
                fit.evaluate(output)

    def test_prompt_overlap_and_underdetermined_training_fail_closed(self):
        with self.assertRaises(ValueError):
            fit.decompose(np.ones((5, 8)))
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            x = np.zeros((2, 3), np.float32)
            y = np.zeros((2, 2, 3), np.float32)
            caps = {}
            for name in ("train", "valid", "test"):
                path = write_capture(root, name, x, x, y, x, ["same-0", "same-1"])
                caps[name] = fit.Capture(path)
            with self.assertRaisesRegex(ValueError, "prompt overlap"):
                fit.split_audit(caps, 2)

    def test_worker_local_prompt_ids_use_global_rows_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for shard in range(4):
                folder = root / f"shard{shard}" / "layer1"
                folder.mkdir(parents=True)
                x = np.zeros((2, 3), np.float32)
                x.tofile(folder / "x.f32")
                np.zeros(2, np.int64).tofile(folder / "prompt_indices.i64")
                rows = [{"prompt_id": f"global-{shard}", "position": i} for i in range(2)]
                (folder / "rows.jsonl").write_text("\n".join(map(json.dumps, rows)) + "\n")
                arrays = {"x": {"path": "layer1/x.f32", "dtype": "float32", "shape": [2, 3]},
                          "prompt_ids": {"path": "layer1/prompt_indices.i64", "dtype": "int64", "shape": [2]}}
                (folder.parent / "capture_manifest.json").write_text(json.dumps({"layers": {"1": {"expert_ids": [14, 40], "arrays": arrays}}}))
            cap = fit.Capture(root, 1)
            self.assertEqual(len(set(cap.prompts())), 4)
            self.assertEqual(cap.get("x").shape, (8, 3))

    def test_defer_test_does_not_require_or_open_future_test_capture(self):
        rng = np.random.default_rng(19)
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            manifests = {}
            for name, n in (("train", 40), ("valid", 20)):
                x = rng.normal(size=(n, 3)).astype(np.float32)
                z = np.concatenate((x, x ** 2), axis=1)
                y = np.stack((x, x ** 2), axis=1)
                manifests[name] = write_capture(root, name, x, z, y, x, [f"{name}-{i}" for i in range(n)])
            args = argparse.Namespace(**{k: str(v) for k, v in manifests.items()}, test=str(root / "future-test"),
                                      output=str(root / "fit"), minimum_test_prompts=1000, seed=19, layer=1,
                                      lambdas=[.01], bases=["shared_z"], allow_underdetermined=False, defer_test=True)
            with threadpool_limits(limits=1):
                fit.fit_all(args)
            selected = json.loads((root / "fit/selection.json").read_text())
            self.assertTrue(selected["test_capture_deferred"])
            self.assertFalse(selected["test_targets_opened_during_fit"])
            self.assertEqual(set(selected["split_audit"]), {"train", "valid"})
            self.assertFalse((root / "future-test").exists())
            self.assertFalse((root / "fit/test_report.json").exists())


if __name__ == "__main__":
    unittest.main()
