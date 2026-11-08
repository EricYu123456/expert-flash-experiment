#!/usr/bin/env python3
"""Check registry gather/equalization/A8/W8 predictions against frozen C++."""
import argparse
import ctypes
import importlib.util
import json
from pathlib import Path

import numpy as np
from threadpoolctl import threadpool_limits

SOURCE = Path(__file__).with_name("research_20261109_shared_feature_fit.py")
spec = importlib.util.spec_from_file_location("check_shared_fit", SOURCE)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)
old = base.old


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registry", type=Path, required=True)
    parser.add_argument("--valid-capture", required=True)
    parser.add_argument("--valid-target-manifest", type=Path)
    parser.add_argument("--numeric-so", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    lib = ctypes.CDLL(str(args.numeric_so))
    lib.sf_set_threads(args.threads)
    ptr = lambda x: ctypes.c_void_p(x.ctypes.data)
    entries = [line.split("\t") for line in args.registry.read_text().splitlines() if line and not line.startswith("#")]
    if any(len(e) != 6 for e in entries):
        raise ValueError("six-column registry required")
    caps, target_metadata = {}, None
    if args.valid_target_manifest:
        target_metadata = json.loads(args.valid_target_manifest.read_text())
    receipt = {"status": "PASS", "scope": "VALID inputs only; no target labels opened", "CPU_only": True,
               "registry": str(args.registry), "registry_sha256": old.sha(args.registry),
               "numeric_library": str(args.numeric_so), "numeric_library_sha256": old.sha(args.numeric_so), "models": []}
    with threadpool_limits(limits=args.threads, user_api="blas"):
        for layer, expert, basis, width, tp, cp in entries:
            layer, expert, width = int(layer), int(expert), int(width)
            cap = caps.setdefault(layer, old.Capture(args.valid_capture, layer))
            n, h = cap.get("x").shape
            rows = np.asarray(sorted({0, n // 3, n // 2, n - 1}))
            x, z = np.ascontiguousarray(cap.get("x")[rows]), np.ascontiguousarray(cap.get("shared_z")[rows])
            sources = np.fromfile(tp + ".sources.u8", np.uint8)
            indices = np.fromfile(tp + ".indices.i32", np.int32)
            center = np.fromfile(tp + ".center.f32", np.float32)
            gain = np.fromfile(tp + ".inv_scale.f32", np.float32)
            if not all(len(a) == width for a in (sources, indices, center, gain)):
                raise ValueError("transform payload shape mismatch")
            # Prototype sources are remapped into one exact FP32 gather pool,
            # matching the live adapter. No extra A8 boundary is introduced.
            remapped_sources, remapped_indices = sources.copy(), indices.copy()
            if np.any(sources == 2):
                if target_metadata is None:
                    raise ValueError("prototype check needs VALID prototype metadata")
                owners = np.fromfile(tp + ".prototypes.i32", np.int32)
                pools = [z]
                offset = z.shape[1]
                for owner in sorted(set(map(int, owners[sources == 2]))):
                    rec = target_metadata["prototype_hidden"][str(owner)]
                    path = Path(rec["path"])
                    if not path.is_absolute():
                        path = args.valid_target_manifest.parent / path
                    raw = np.memmap(path, mode="r", dtype=rec.get("dtype", "float32"), shape=tuple(rec["shape"]))
                    phi = np.ascontiguousarray(raw[rows])
                    pools.append(phi)
                    mask = (sources == 2) & (owners == owner)
                    remapped_sources[mask] = 1
                    remapped_indices[mask] += offset
                    offset += phi.shape[1]
                z = np.ascontiguousarray(np.concatenate(pools, axis=1))
            gathered = np.empty((len(rows), width), np.float32)
            for source, raw in ((0, x), (1, z)):
                mask = remapped_sources == source
                if np.any(mask):
                    gathered[:, mask] = raw[:, remapped_indices[mask]]
            expected = base.transform(gathered, center, gain)
            actual = np.empty_like(expected)
            assert lib.sf_transform(len(rows), h, z.shape[1], width, ptr(x), ptr(z), ptr(remapped_sources), ptr(remapped_indices), ptr(center), ptr(gain), ptr(actual)) == 0
            assert np.array_equal(expected.view(np.uint32), actual.view(np.uint32)), "transform bits"
            encoded = base.encode(expected)
            q, sx = np.empty_like(encoded["q"]), np.empty_like(encoded["scale"])
            assert lib.sf_quantize(len(rows), width, ptr(actual), ptr(q), ptr(sx)) == 0
            assert np.array_equal(q, encoded["q"]) and np.array_equal(sx.view(np.uint32), encoded["scale"].view(np.uint32)), "A8 bits"
            model = {"qweight": np.fromfile(cp + ".i8", np.int8).reshape(h, width),
                     "wscale": np.fromfile(cp + ".scales.f32", np.float32),
                     "bias": np.fromfile(cp + ".bias.f32", np.float32)}
            prediction = base.predict_encoded(encoded, model)
            cpp = np.empty_like(prediction)
            assert lib.sf_predict(len(rows), width, h, ptr(actual), ptr(model["qweight"]), ptr(model["wscale"]), ptr(model["bias"]), ptr(cpp)) == 0
            assert np.array_equal(prediction.view(np.uint32), cpp.view(np.uint32)), "prediction bits"
            receipt["models"].append({"layer": layer, "expert": expert, "basis": basis, "width": width,
                                       "VALID_rows": rows.tolist(), "transform_floats_bitchecked": int(actual.size),
                                       "predictor_floats_bitchecked": int(cpp.size)})
    old.dump(args.output, receipt)
    print(json.dumps({"receipt": str(args.output), "models": len(receipt["models"]), "status": "PASS"}), flush=True)


if __name__ == "__main__":
    main()
