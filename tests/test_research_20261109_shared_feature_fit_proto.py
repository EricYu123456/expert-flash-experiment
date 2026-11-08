import argparse
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from threadpoolctl import threadpool_limits


SOURCE = Path(__file__).resolve().parents[1] / "scripts/research_20261109_shared_feature_fit_proto.py"
spec = importlib.util.spec_from_file_location("proto_fit", SOURCE)
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)


class PrototypeFitTests(unittest.TestCase):
    def test_weighted_moment_mixture_matches_direct_weighted_ridge(self):
        rng = np.random.default_rng(119)
        x = rng.normal(size=(90, 7)).astype(np.float32)
        y = rng.normal(size=(90, 4)).astype(np.float32)
        a2 = np.where(np.arange(90) % 4 == 0, .25, 0)
        with threadpool_limits(limits=1):
            moments = fit.all_moments(x)
            em = fit.expert_moments(moments, y, a2)
            combo = fit.combined(moments, em, .99)
            direct = fit.base.prepare_gram(x, fit.base.blend_weights(a2, .99))
            mean, rhs = fit.base.solve_components(direct, y)
            expected = fit.base.solve_one(direct, mean, rhs, .3)["coefficient"]
            actual = (combo["vectors"] @ (combo["rhs"] / (combo["values"][:, None] + .3))).T
        np.testing.assert_allclose(combo["xmean"], direct["mean"], atol=1e-12)
        np.testing.assert_allclose(actual, expected, atol=1e-7)

    def test_prompt_disjoint_ranking_excludes_self_and_formal_exports(self):
        rng = np.random.default_rng(912)
        h, width = 6, 8
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            weights = {}
            for owner in range(3):
                q, sw = fit.old.quantize_rows(rng.normal(size=(h, width)).astype(np.float32))
                prefix = root / f"owner{owner}.D"
                q.tofile(str(prefix) + ".i8")
                sw.tofile(str(prefix) + ".scales.f32")
                weights[owner] = q, sw, prefix
            paths = {}
            for split, n in [("train", 160), ("valid", 64)]:
                folder = root / split
                folder.mkdir()
                x = rng.normal(size=(n, h)).astype(np.float32)
                selected = np.tile(np.arange(4, dtype=np.int32), (n, 1))
                alpha = np.full((n, 4), .25, np.float32)
                records = {}
                for name, array in {"x": x, "selected_experts": selected, "routing_weights": alpha,
                                    "prompt_ids": np.asarray([f"{split}-{i//4}" for i in range(n)])}.items():
                    np.save(folder / f"{name}.npy", array)
                    records[name] = {"path": f"{name}.npy"}
                cap = folder / "capture.json"
                cap.write_text(json.dumps({"expert_ids": [0, 1], "arrays": records}))
                meta = {"layer": 1, "rows": n, "hidden": h, "expert_ids": list(range(4)),
                        "capture_manifest_sha256": {str(cap): fit.old.sha(cap)},
                        "input_sha256": {str(cap): fit.old.sha(cap)}, "targets": {}, "prototype_hidden": {}}
                outputs = {}
                for owner in range(3):
                    phi = rng.normal(size=(n, width)).astype(np.float32)
                    qphi, sx = fit.old.quantize_rows(phi)
                    qw, sw, prefix = weights[owner]
                    outputs[owner] = fit.old.rescale(fit.old.exact_dots(qphi, qw), sw, sx)
                    path = folder / f"prototype{owner}.f32"
                    phi.tofile(path)
                    meta["prototype_hidden"][str(owner)] = {"path": str(path), "shape": [n, width],
                                                            "dtype": "float32", "native_D_prefix": str(prefix)}
                outputs[3] = outputs[1] * np.float32(.5)
                for expert, y in outputs.items():
                    path = folder / f"expert{expert}.f32"
                    y.tofile(path)
                    meta["targets"][str(expert)] = {"path": str(path), "dtype": "float32", "shape": [n, h]}
                np.save(folder / "routed_outputs.npy", np.stack([outputs[e] for e in range(4)], axis=1))
                np.save(folder / "shared_output.npy", np.zeros((n, h), np.float32))
                records["routed_outputs"] = {"path": "routed_outputs.npy"}
                records["shared_output"] = {"path": "shared_output.npy"}
                cap.write_text(json.dumps({"expert_ids": [0, 1], "arrays": records}))
                meta["capture_manifest_sha256"] = {str(cap): fit.old.sha(cap)}
                meta["input_sha256"] = {str(cap): fit.old.sha(cap)}
                targets = folder / "targets.json"
                targets.write_text(json.dumps(meta))
                paths[split] = cap, targets
            rank_args = argparse.Namespace(train=str(paths["train"][0]), train_target_manifest=str(paths["train"][1]),
                                           layer=1, output=str(root / "rank"), rank_fit_rows=100, rank_score_rows=32,
                                           sketch_width=4, top_prototypes=2, rank_lambda=.3, seed=119, experts=None)
            with threadpool_limits(limits=1):
                fit.rank(rank_args)
            ranking = json.loads((root / "rank/ranking.json").read_text())
            self.assertFalse(set(ranking["fit_prompt_ids"]) & set(ranking["score_prompt_ids"]))
            self.assertTrue(all(int(expert) not in owners for expert, owners in ranking["choices"].items()))
            self.assertEqual(ranking["choices"]["3"][0], 1)
            fit_args = argparse.Namespace(train=str(paths["train"][0]), valid=str(paths["valid"][0]),
                                          train_target_manifest=str(paths["train"][1]), valid_target_manifest=str(paths["valid"][1]),
                                          layer=1, ranking=str(root / "rank/ranking.json"), output=str(root / "fit"),
                                          experts=None, betas=[0, .99], lambdas=[.3])
            with threadpool_limits(limits=1):
                fit.fit(fit_args)
            result = json.loads((root / "fit/selection.json").read_text())
            self.assertTrue(all(m["prototype"] != m["expert"] for m in result["families"]["best_cross_prototype"]))
            self.assertIn("raw_x_eq", result["aggregate_VALID"])
            self.assertEqual(result["aggregate_VALID"]["raw_x_eq"]["slot_coverage"], 1)
            for model in result["families"]["best_cross_prototype"]:
                pp = np.fromfile(model["transform_prefix"] + ".prototypes.i32", np.int32)
                self.assertTrue(np.all(pp == model["prototype"]))
                self.assertLess(model["TRAIN_rounding_audit"]["relative_L2_roundoff"], 1e-6)
            lines = (root / "fit/registry.raw_x_eq.excluding_owners.tsv").read_text().splitlines()
            self.assertEqual(len(lines), 1)
            self.assertEqual(int(lines[0].split("\t")[1]), 3)
            joint_path = SOURCE.with_name("research_20261109_shared_feature_fit_joint.py")
            joint_spec = importlib.util.spec_from_file_location("proto_endtoend_joint", joint_path)
            joint = importlib.util.module_from_spec(joint_spec)
            joint_spec.loader.exec_module(joint)
            joint_args = argparse.Namespace(parent_selection=str(root / "fit/selection.json"),
                                            family="raw_x_eq", exclude_owners=False,
                                            train=str(paths["train"][0]), valid=str(paths["valid"][0]),
                                            train_target_manifest=str(paths["train"][1]), valid_target_manifest=str(paths["valid"][1]),
                                            output=str(root / "joint"), ridge=.3, etas=[0, .5, 1])
            with threadpool_limits(limits=1):
                joint.joint(joint_args)
            calibrated = json.loads((root / "joint/joint_selection.json").read_text())
            self.assertLessEqual(calibrated["final_VALID"]["routed"]["squared_error"],
                                 calibrated["initial_VALID"]["routed"]["squared_error"])
            self.assertEqual(calibrated["runtime_extra_coefficients"], 0)
            self.assertEqual(len((root / "joint/registry.tsv").read_text().splitlines()), 4)


if __name__ == "__main__":
    unittest.main()
