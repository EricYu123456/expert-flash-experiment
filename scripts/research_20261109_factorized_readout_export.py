#!/usr/bin/env python3
"""Descriptors, cohort registries and CPU-canonical 7-row factor fixtures."""
import argparse
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("factor_export_tools", HERE / "research_20261109_factorized_readout_fit.py")
fit = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fit)
base, old, proto = fit.base, fit.old, fit.proto


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("selection", "valid", "valid-target-manifest", "output"):
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    selection_path = Path(args.selection).resolve()
    selection = json.loads(selection_path.read_text())
    output = Path(args.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "probes").mkdir()
    valid = proto.Dataset(args.valid, args.valid_target_manifest, 1)
    if valid.identity() != selection["valid_identity"]:
        raise ValueError("fixture VALID identity differs from factor fitting")
    records, registries = [], {}
    for model in selection["models"]:
        rank, width, hidden = model["rank"], model["input_width"], model["hidden"]
        prefix = model["factor_prefix"]
        descriptor = dict(version=1, input_width=width, width=width, H=hidden, rank=rank,
            kind="two_rowW8_with_TRAIN_latent_affine_A8", input_transform_prefix=model["input_transform_prefix"],
            payload_sha256=model["payload_sha256"], V_bias=False,
            arithmetic="input affine/A8; V intdot/F32 rescale; F32(mid-center)*inv_scale; latent vectorA8; U intdot/F32 rescale; F32 finalbias add")
        path = Path(prefix + ".factor.json")
        if path.exists():
            raise FileExistsError("factor descriptor already exists")
        for p, digest in model["payload_sha256"].items():
            assert old.sha(p) == digest
        old.dump(path, descriptor)
        a2 = base.alpha_squared(valid.capture, model["expert"])
        selected = np.flatnonzero(a2 > 0)
        if len(selected) < 7:
            raise ValueError("need seven naturally selected VALID fixture rows")
        rows = selected[np.linspace(0, len(selected) - 1, 7).astype(np.int64)]
        transform = proto.load_transform(model["input_transform_prefix"])
        features = fit.source_features(valid, transform, rows)
        V = dict(qweight=np.fromfile(prefix + ".V.i8", np.int8).reshape(rank, width),
                 wscale=np.fromfile(prefix + ".V.scales.f32", np.float32), bias=np.zeros(rank, np.float32))
        U = dict(qweight=np.fromfile(prefix + ".U.i8", np.int8).reshape(hidden, rank),
                 wscale=np.fromfile(prefix + ".U.scales.f32", np.float32), bias=np.fromfile(prefix + ".bias.f32", np.float32))
        center = np.fromfile(prefix + ".latent.center.f32", np.float32)
        gain = np.fromfile(prefix + ".latent.inv_scale.f32", np.float32)
        encoded = base.encode(features)
        mid = base.predict_encoded(encoded, V, use_bias=False)
        normalized = base.transform(mid, center, gain)
        latent_encoded = base.encode(normalized)
        prediction = base.predict_encoded(latent_encoded, U)
        fixture = output / "probes" / f"{model['family']}.expert{model['expert']}.rank{rank}.npz"
        np.savez(fixture, input_features=features, V_raw=mid, latent_normalized=normalized,
                 output=prediction, capture_rows=rows, input_A8=encoded["q"], input_scale=encoded["scale"],
                 latent_A8=latent_encoded["q"], latent_scale=latent_encoded["scale"])
        basis = "prototype_25_eq" if model["family"] == "prototype25" else "raw_x_select1408_eq"
        line = "\t".join(map(str, [1, model["expert"], basis, width, model["input_transform_prefix"], prefix]))
        registries.setdefault((model["family"], rank), {})[model["expert"]] = line
        records.append(dict(family=model["family"], expert=model["expert"], rank=rank,
            factor_prefix=prefix, descriptor=str(path), descriptor_sha256=old.sha(path),
            fixture=str(fixture), fixture_sha256=old.sha(fixture), capture_rows=rows.tolist()))
    paths = {}
    for (family, rank), by_expert in registries.items():
        order = selection["experts"]
        for count in (3, 6, 9, 12):
            if count > len(order):
                continue
            path = output / f"registry.{family}.rank{rank}.k{count}.tsv"
            path.write_text("\n".join(by_expert[e] for e in order[:count]) + "\n")
            paths[str(path)] = old.sha(path)
    receipt = dict(status="factor_descriptors_and_CPU_canonical_fixtures_complete", source_sha256=old.sha(__file__),
        selection_path=str(selection_path), selection_sha256=old.sha(selection_path), models=records,
        registry_sha256=paths, VALID_only_no_TEST=True, CPU_affinity=sorted(os.sched_getaffinity(0)))
    old.dump(output / "handoff.json", receipt)
    print(json.dumps(dict(handoff=str(output / "handoff.json"), models=len(records), registries=len(paths))), flush=True)


if __name__ == "__main__":
    with threadpool_limits(limits=2, user_api="blas"):
        main()
