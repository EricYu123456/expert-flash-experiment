#!/usr/bin/env python3
"""Merge immutable seed/additional fits into single-GU 3/6/9/12 registries.

No fitting or target arrays are opened. Equivalent transforms are checked
byte-for-byte before sharing a single prefix within each runtime basis cache.
Cohort order/owner assignment were selected with calibration VALID; this file
does not present them as TRAIN-only or as a fresh holdout result.
"""
import argparse
import hashlib
import json
from pathlib import Path


ORDER = [4, 51, 60, 58, 5, 17, 62, 22, 1, 28, 38, 50]
SEED = set(ORDER[:3])
SUFFIXES = (".sources.u8", ".indices.i32", ".center.f32", ".inv_scale.f32", ".prototypes.i32")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", type=Path, required=True)
    args = p.parse_args()
    data = args.data_dir.resolve()
    root = data / "proto25-cohort12"
    output = root / "registries"
    output.mkdir(exist_ok=False)
    old_fit_path = data / "formal-gentle/layer1/selection.json"
    new_fit_path = root / "fit-extra9/selection.json"
    old_fit, new_fit = read(old_fit_path), read(new_fit_path)
    old_subset_path = data / "raw-subset/cohort11/layer1/selection.json"
    new_subset_path = root / "subset-extra9/selection.json"
    old_constant_path = data / "spec2-controls/train64-candidate5/controls.json"
    new_constant_path = root / "constant-extra9/controls.json"
    inputs = (old_fit_path, new_fit_path, old_subset_path, new_subset_path, old_constant_path, new_constant_path)
    old_subset, new_subset = read(old_subset_path), read(new_subset_path)
    old_constant, new_constant = read(old_constant_path), read(new_constant_path)
    families = {}
    for family, name in (("raw_x_eq", "raw_x_eq"), ("best_cross_prototype", "prototype25")):
        families[name] = [m for m in old_fit["families"][family] if m["expert"] in SEED] + new_fit["families"][family]
    families["raw_x_select1408_eq"] = [m for m in old_subset["models"] if m["expert"] in SEED] + new_subset["models"]
    for family in ("drop_zero", "selected_mean", "selected_alpha2_mean"):
        families[family] = ([m for m in old_constant["models"] if m["layer"] == 1 and m["expert"] in SEED and m["basis"] == family]
                            + [m for m in new_constant["models"] if m["layer"] == 1 and m["basis"] == family])
    result = {"status": "complete_immutable_single_prototype25_cohort_registries", "layer": 1,
              "cohort_order": ORDER, "cohort_order_uses_calibration_VALID": True,
              "widths": {"raw_x_eq": 2048, "prototype25": 1408, "raw_x_select1408_eq": 1408,
                         "drop_zero": 1, "selected_mean": 1, "selected_alpha2_mean": 1},
              "deployed_nonlinear_owner_union": [25], "prototype25_owner_stays_native": True,
              "betas_for_all_fitted_readouts": [0, .5, .9, .99], "lambdas": [1, 3, 10],
              "input_selection_sha256": {str(path): sha(path) for path in inputs},
              "source_sha256": sha(__file__), "registries": {}, "models": {}, "transforms": {},
              "no_input_or_target_tensor_opened": True,
              "cost_scope": "Charge active owner25 GU weights/scales and deduplicated GU row work; same-width raw subset adds no GU. Parent target pool metadata with eight owners does not describe active runtime GU bank."}
    for family, entries in families.items():
        lookup = {m["expert"]: m for m in entries}
        if len(entries) != len(ORDER) or set(lookup) != set(ORDER):
            raise ValueError(f"incomplete/duplicate cohort in {family}")
        entries = [lookup[e] for e in ORDER]
        tp = entries[0]["transform_prefix"]
        expected = {suffix: Path(tp + suffix).read_bytes() for suffix in SUFFIXES}
        for entry in entries:
            if entry["width"] != result["widths"][family]:
                raise ValueError("readout width contract mismatch")
            if family == "prototype25" and entry["prototype"] != 25:
                raise ValueError("an additional nonlinear owner entered single-prototype cohort")
            for suffix, payload in expected.items():
                if Path(entry["transform_prefix"] + suffix).read_bytes() != payload:
                    raise ValueError(f"basis-cache transform differs: {family} expert{entry['expert']}")
        result["transforms"][family] = {"unified_prefix": tp,
                                      "byte_identical_original_prefixes": sorted(set(m["transform_prefix"] for m in entries)),
                                      "sha256": {tp + suffix: sha(tp + suffix) for suffix in SUFFIXES}}
        result["models"][family] = entries
        result["registries"][family] = {}
        for count in (3, 6, 9, 12):
            lines = ["\t".join(map(str, [1, m["expert"], m["basis"], m["width"], tp, m["readout_prefix"]])) for m in entries[:count]]
            path = output / f"registry.{family}.k{count}.tsv"
            path.write_text("\n".join(lines) + "\n")
            result["registries"][family][str(count)] = {"path": str(path), "sha256": sha(path), "experts": ORDER[:count]}
    (root / "registry_handoff.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"handoff": str(root / "registry_handoff.json"), "cohort_order": ORDER, "families": list(families)}), flush=True)


if __name__ == "__main__":
    main()
