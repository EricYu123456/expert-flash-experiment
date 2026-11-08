#!/usr/bin/env python3
"""GPU SVD spectrum of actual W8 readouts: prototype and same-width raw.

Only C codes/scales are read. No TRAIN/VALID/TEST activation or target arrays
are opened. Spectral energy is a weight-space diagnostic, not output quality
or evidence that two W8 factors with a new latent A8 boundary preserve quality.
"""
import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import time

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import numpy as np
import torch


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--handoff", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--driver", choices=("gesvd", "gesvdj"), default="gesvd")
    args = parser.parse_args()
    if os.environ.get("CUDA_VISIBLE_DEVICES") != "0" or set(os.sched_getaffinity(0)) != {64, 65}:
        raise ValueError("use physical CUDA0 and CPU affinity64-65")
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    device = torch.device("cuda:0")
    properties = torch.cuda.get_device_properties(device)
    torch.cuda.set_per_process_memory_fraction(8 * 1024 ** 3 / properties.total_memory, device)
    torch.cuda.reset_peak_memory_stats(device)
    args.output.mkdir(parents=True, exist_ok=False)
    handoff = json.loads(args.handoff.read_text())
    records, rows = [], []
    hardware = dict(gpu=properties.name, device=str(device), torch=torch.__version__, CUDA=torch.version.cuda,
                    cpu_affinity=sorted(os.sched_getaffinity(0)), CPU_threads=2,
                    nvidia_smi=subprocess.run(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total,memory.used", "--format=csv,noheader"],
                                             capture_output=True, text=True, check=True).stdout.strip())
    started = time.monotonic()
    for family in ("prototype25", "raw_x_select1408_eq"):
        for entry in handoff["models"][family]:
            prefix, width, hidden = entry["readout_prefix"], entry["width"], 2048
            assert width == 1408
            input_hashes = {prefix + suffix: sha(prefix + suffix) for suffix in (".i8", ".scales.f32")}
            for path, digest in input_hashes.items():
                assert digest == entry["payload_sha256"][path]
            q = np.fromfile(prefix + ".i8", np.int8).reshape(hidden, width)
            scales = np.fromfile(prefix + ".scales.f32", np.float32)
            # The deployed dequantized C is F32(code*row_scale). Cast that exact
            # F32 matrix to FP64 for a numerically stable CUDA spectrum.
            coefficient = np.multiply(q.astype(np.float32), scales[:, None], dtype=np.float32)
            value = torch.from_numpy(coefficient).to(device=device, dtype=torch.float64)
            begin = time.monotonic()
            singular = torch.linalg.svdvals(value, driver=args.driver)
            torch.cuda.synchronize(device)
            seconds = time.monotonic() - begin
            s = singular.cpu().numpy()
            squared = s ** 2
            total = float(np.sum(coefficient.astype(np.float64) ** 2))
            ratio_error = abs(float(squared.sum()) / total - 1)
            if ratio_error > 1e-10:
                raise ValueError("SVD singular energy differs from exact input Frobenius norm")
            cumsum = np.cumsum(squared)
            path = args.output / f"{family}.expert{entry['expert']}.singular.f64"
            s.tofile(path)
            rec = dict(family=family, expert=entry["expert"], hidden=hidden, width=width,
                       driver=args.driver, seconds=seconds, arithmetic="F32 W8 dequant -> CUDA FP64 SVD",
                       input_payload_sha256=input_hashes, Frobenius_squared=total,
                       Frobenius_energy_relative_check_error=ratio_error,
                       stable_rank=total / squared[0], singular_values_path=str(path), singular_sha256=sha(path),
                       ranks_for_energy={str(f): int(np.searchsorted(cumsum, f * total) + 1) for f in (.9, .95, .99)},
                       ranks={})
            for rank in (128, 256, 512):
                energy = float(cumsum[rank - 1] / total)
                rec["ranks"][str(rank)] = dict(retained_Frobenius_energy=energy,
                    FP_optimal_weight_Frobenius_relative_error=1 - energy,
                    two_factor_coefficients=rank * (hidden + width),
                    two_W8_factors_plus_row_scales_bias_bytes=rank * (hidden + width) + 4 * rank + 8 * hidden,
                    original_W8_plus_scales_bias_bytes=hidden * width + 8 * hidden,
                    new_runtime_latent_A8_required=True)
                rows.append(dict(family=family, expert=entry["expert"], rank=rank,
                                 retained_energy=energy, discarded_energy=1 - energy,
                                 two_factor_coefficients=rank * (hidden + width),
                                 original_coefficients=hidden * width,
                                 seconds=seconds))
            for path, digest in input_hashes.items():
                assert sha(path) == digest
            records.append(rec)
            print(json.dumps({"family": family, "expert": entry["expert"], "seconds": seconds,
                              "energies": {k: v["retained_Frobenius_energy"] for k, v in rec["ranks"].items()}}), flush=True)
            del value, singular
    peak = dict(allocated_bytes=torch.cuda.max_memory_allocated(device), reserved_bytes=torch.cuda.max_memory_reserved(device))
    assert peak["reserved_bytes"] < 8 * 1024 ** 3
    result = dict(status="weight_spectrum_diagnostic_only", source_sha256=sha(__file__),
                  handoff=str(args.handoff.resolve()), handoff_sha256=sha(args.handoff), hardware=hardware,
                  records=records, GPU_peak_memory=peak, seconds=time.monotonic() - started,
                  no_activation_or_target_arrays_opened=True, runtime_implemented=False,
                  limits="No factor W8 or latent-A8 validation. Weight Frobenius energy does not bound expert output error without activation covariance; both families receive identical rank/compression options.")
    (args.output / "spectrum.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    with (args.output / "spectrum.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


if __name__ == "__main__":
    main()
