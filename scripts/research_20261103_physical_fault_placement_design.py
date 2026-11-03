#!/usr/bin/env python3
"""Bounded, fixed-ECC fault-placement design audit; no inference or ECC changes.

Only official QCFlash native page-read reconstruction is described. Raw errors
are never called post-ECC faults. The four-expert witness is symbolic, not a
measured NAND fault distribution or an LLM quality experiment.
"""
from __future__ import annotations

import argparse
from fractions import Fraction
import hashlib
from io import BytesIO
import itertools
import json
from pathlib import Path
import xml.etree.ElementTree as ET
import zipfile

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/research-2026-11-02/round2/physical-fault-placement-design"
ZIP = OUT / "QCFlash_artifact_v1.0.zip"
BASE = ROOT / "results/research-2026-11-02/round2/compact-coupled-solver/continuation/native/0107.npz"
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_new(path, obj):
    assert not path.exists(), path
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False) + "\n")


def protocol():
    paths = [Path(__file__), ZIP, OUT / "artifact_receipt.json", BASE,
             ROOT / "paper/aif.txt",
             ROOT / "results/research-2026-11-02/round2/physical-error-prior/review.json",
             ROOT / "results/research-2026-11-02/round2/physical-reliability/notes.json"]
    with zipfile.ZipFile(ZIP) as z:
        members = [f"QCFlash/data/vth_states_wl{wl}.xlsx" for wl in [10, 11, 12]]
        members += ["QCFlash/qcflash.py", "QCFlash/README.md", "QCFlash/DATA_LICENSE"]
        mh = {m: hashlib.sha256(z.read(m)).hexdigest() for m in members}
    return {
        "source_and_inputs_sha256": {str(p): sha(p) for p in paths},
        "zip_members_sha256": mh,
        "measured_scope": "Three released QLC WLs, native unshifted Vref reads only; no repeated reads, ECC, reprogramming, AiF be-enc, or service-time inference.",
        "measurement_processing": "Parse original XLSX XML, preserve cell index; validate programmed state/page bits, reconstruct threshold reads, exclude missing Vth in measured-only RBER and explicitly label official perfect-label fallback.",
        "window_diagnostic": "Consecutive logical 8192-bit windows are descriptive only; unknown codeword/parity/interleaver. >10 raw errors is not an observed UCE/SDC. Incomplete last windows are separate.",
        "symbolic_witness": "Enumerate all balanced two-domain placements for four experts and two equally frequent disjoint two-expert routes. Fixed independent Bernoulli domain p values are hypothetical, not fitted to QCFlash.",
        "coarse_domain_null": "Check CP0107 whole-GU expert chip touch; a one-block/chip allocation cannot decluster experts against total chip/block loss.",
        "forbidden": "No model/weight access, heldout, LLM bitflip campaign, ECC/decoder/format change, model corruption, inferred physical latency, native simulator campaign, or new hardware."
    }


def xlsx_sheets(blob):
    """Minimal dependency-free parser of the three required original sheets."""
    with zipfile.ZipFile(BytesIO(blob)) as z:
        strings = ["".join(si.itertext()) for si in ET.fromstring(z.read("xl/sharedStrings.xml"))]
        rels = ET.fromstring(z.read("xl/_rels/workbook.xml.rels"))
        targets = {r.attrib["Id"]: "xl/" + r.attrib["Target"].lstrip("/") for r in rels}
        workbook = ET.fromstring(z.read("xl/workbook.xml"))
        sheets = {}
        for s in workbook.find("m:sheets", NS):
            name = s.attrib["name"]
            if name not in ["Index_wise Vth", "Reference Positions", "shared page data"]:
                continue
            rid = s.attrib["{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id"]
            data = ET.fromstring(z.read(targets[rid])).find("m:sheetData", NS)
            rows = []
            for r in data:
                row = {}
                for c in r:
                    label = "".join(ch for ch in c.attrib["r"] if ch.isalpha())
                    v = c.find("m:v", NS)
                    if v is None:
                        value = None
                    elif c.attrib.get("t") == "s":
                        value = strings[int(v.text)]
                    else:
                        value = float(v.text)
                    row[label] = value
                rows.append(row)
            header = rows[0]
            sheets[name] = [{header[k]: r.get(k) for k in header} for r in rows[1:]]
        return sheets


def descriptive_windows(errors, valid, n=8192):
    nw = len(errors) // n
    counts = errors[:nw*n].reshape(nw, n).sum(1).astype(int)
    missing = (~valid[:nw*n]).reshape(nw, n).sum(1).astype(int)
    rem = len(errors) % n
    return {
        "full_window_bits": n, "full_windows": nw,
        "raw_error_counts": counts.tolist(), "missing_Vth_counts": missing.tolist(),
        "full_windows_observed_errors_gt10": int((counts > 10).sum()),
        "full_windows_possible_errors_gt10_if_missing_are_errors": int(((counts + missing) > 10).sum()),
        "last_incomplete_bits": rem,
        "last_incomplete_observed_errors": int(errors[nw*n:].sum()),
        "last_incomplete_missing_Vth": int((~valid[nw*n:]).sum()),
        "interpretation": "Not actual ECC codewords; no parity/error-decoder outcome available."
    }


def read_measurements():
    thresholds = {"LSB": [8], "CSB": [4, 12], "MSB": [2, 6, 10, 14], "TSB": [1, 3, 5, 7, 9, 11, 13, 15]}
    out, masks = [], {}
    with zipfile.ZipFile(ZIP) as z:
        for wl in [10, 11, 12]:
            blob = z.read(f"QCFlash/data/vth_states_wl{wl}.xlsx")
            s = xlsx_sheets(blob)
            rows = sorted(s["Index_wise Vth"], key=lambda r: r["cell_index"])
            idx = np.array([r["cell_index"] for r in rows], int)
            states = np.array([r["QLC_State"] for r in rows], int)
            vth = np.array([r["Actual_vth_V"] if r["Actual_vth_V"] is not None else np.nan for r in rows])
            assert np.array_equal(idx, np.arange(len(idx))) and len(idx) % 8 == 0
            assert np.all((states >= 0) & (states <= 15))
            refs = {r["Reference"].strip().upper(): r["Position_V"] for r in s["Reference Positions"]}
            valid = np.isfinite(vth)
            record = {"WL": wl, "cells": len(idx), "page_bytes": len(idx)//8,
                      "measured_Vth_cells": int(valid.sum()), "missing_Vth_cells": int((~valid).sum()),
                      "workbook_sha256": hashlib.sha256(blob).hexdigest(),
                      "programmed_state_sha256": hashlib.sha256(states.astype(np.uint8).tobytes()).hexdigest(),
                      "pages": {}}
            error_columns = []
            for page, ts in thresholds.items():
                expected_bytes = np.array([r[page] for r in s["shared page data"]], np.uint8)
                expected = np.unpackbits(expected_bytes)
                assert len(expected) == len(idx)
                state_bits = (1 - np.searchsorted(ts, states, side="right") % 2).astype(np.uint8)
                assert np.array_equal(state_bits, expected), (wl, page, "state to programmed data mismatch")
                voltages = [refs[f"REF{t}"] for t in ts]
                bits = (1 - (vth[:, None] >= voltages).sum(1) % 2).astype(np.uint8)
                # The artifact fills missing Vth using known programmed states;
                # that is perfect-label substitution, not an observed good read.
                bits[~valid] = expected[~valid]
                errors = bits != expected
                assert not errors[~valid].any()
                state_counts = []
                for state in range(16):
                    mask = valid & (states == state)
                    state_counts.append({"state": state, "observed_cells": int(mask.sum()), "errors": int(errors[mask].sum())})
                eidx = np.flatnonzero(errors)
                rec = {
                    "native_threshold_refs": [f"REF{t}" for t in ts],
                    "native_threshold_voltages": voltages,
                    "measured_errors": int(errors.sum()),
                    "measured_only_RBER": float(errors.sum()/valid.sum()),
                    "artifact_truth_fallback_RBER": float(errors.mean()),
                    "missing_Vth_error_rate_interval": [float(errors.sum()/len(idx)), float((errors.sum()+(~valid).sum())/len(idx))],
                    "adjacent_error_cell_pairs": int(np.sum(errors[1:] & errors[:-1])),
                    "error_indices_sha256": hashlib.sha256(eidx.astype("<i8").tobytes()).hexdigest(),
                    "state_counts": state_counts,
                    "logical_contiguous_windows": descriptive_windows(errors, valid),
                    "programmed_bytes_sha256": hashlib.sha256(expected_bytes.tobytes()).hexdigest(),
                }
                record["pages"][page] = rec
                masks[f"wl{wl}_{page}_raw_error"] = errors
                error_columns.append(errors)
            allerr = np.stack(error_columns, axis=1).astype(np.int64)
            record["same_cell_raw_error_intersections_LSB_CSB_MSB_TSB"] = (allerr.T @ allerr).tolist()
            record["same_cell_caveat"] = "Shared-cell QLC page observations; not four distinct AiF GEMV-capable reliability tiers or independent physical domains."
            masks[f"wl{wl}_measured_Vth"] = valid
            masks[f"wl{wl}_states"] = states
            out.append(record)
    return out, masks


def symbolic_witness():
    routes = [(0, 1), (2, 3)]
    ps = [Fraction(0), Fraction(1, 10**6), Fraction(1, 1000), Fraction(1, 100), Fraction(1, 10), Fraction(1, 2), Fraction(1)]
    records = []
    # A complement renames equal physical domains; keep expert0 in domain0.
    for other in [1, 2, 3]:
        d0 = {0, other}
        placement = [0 if e in d0 else 1 for e in range(4)]
        for p in ps:
            any_bad = both_bad = expected_bad = Fraction(0)
            for route in routes:
                for faults in itertools.product([0, 1], repeat=2):
                    prob = Fraction(1, 2)
                    for f in faults:
                        prob *= p if f else 1-p
                    n = sum(faults[placement[e]] for e in route)
                    any_bad += prob * (n >= 1)
                    both_bad += prob * (n >= 2)
                    expected_bad += prob * n
            colocated = placement[0] == placement[1]
            assert expected_bad == 2*p
            assert any_bad == (p if colocated else 2*p-p*p)
            assert both_bad == (p if colocated else p*p)
            records.append({"expert_to_domain": placement, "p": str(p),
                            "P_any_selected_domain_failed": str(any_bad),
                            "P_two_selected_domains_failed": str(both_bad),
                            "E_selected_experts_on_failed_domain": str(expected_bad)})
    return {
        "routes": routes, "route_probabilities": ["1/2", "1/2"],
        "expert_marginal_selection_probabilities": ["1/2"]*4,
        "identical_capacity_and_marginal_sensitivity": True,
        "placement_count_up_to_domain_rename": 3, "p_values": [str(p) for p in ps],
        "enumerated_domain_states_per_record": 4,
        "records": records,
        "warning": "p is symbolic, not measured. No LLM output corruption or expert omission is executed. Quality tail requires separately measured post-ECC SDC and input-conditioned impact; detected read loss requires exact recovery/abort, not approximate inference.",
        "fair_ties": "A marginal-only baseline can tie-break to the route-optimal map; report the best legal tied baseline, not just one unfavorable tie.",
    }


def coarse_domain_null():
    a = np.load(BASE)
    rows = a["rows"]
    gu = rows[:64, 0]
    assert rows.shape == (65, 2, 16)
    assert np.all(gu > 0) and np.all(gu.sum(1) == 1024)
    route_touches = np.any(gu[a["routes"]] > 0, axis=1)
    assert np.all(route_touches)
    return {
        "experts": 64, "chips": 16, "TRAIN_routes": len(a["routes"]),
        "all_expert_GU_touch_all_chips": True,
        "all_route_GU_touch_all_chips": True,
        "any_total_chip_loss_causes_missing_GU_rows_for_every_route": True,
        "one_block_per_chip_case": "If all local layer coefficients occupy one block, loss of that block has the same null; this does not cover finer WL/codeword errors, spare blocks, or generic replicated recovery.",
        "null_scope": "Preserve positive whole-GU row geometry; merely permuting expert rows within those chip/block domains cannot change whether a selected expert is touched. This is not a global replication impossibility."
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--freeze", action="store_true")
    args = ap.parse_args()
    p = protocol()
    proto = OUT / "measurement_protocol.frozen.json"
    if args.freeze:
        write_new(proto, p)
        print(json.dumps({"protocol_sha256": sha(proto)}, indent=2))
        return
    assert json.loads(proto.read_text()) == p, "frozen inputs/source changed"
    measurements, arrays = read_measurements()
    report = {"scope": "Bounded design evidence, not fault-aware placement gain or measured AiF error rates.",
              "protocol_sha256": sha(proto), "QCFlash_native_read_description": measurements,
              "symbolic_domain_witness": symbolic_witness(), "CP0107_coarse_domain_null": coarse_domain_null()}
    write_new(OUT / "bounded_report.json", report)
    path = OUT / "raw_native_masks.npz"
    assert not path.exists()
    np.savez_compressed(path, **arrays)
    print(json.dumps({"WL_pages": [{"WL": r["WL"], "cells": r["cells"], "missing": r["missing_Vth_cells"],
                                   "errors": {p: v["measured_errors"] for p, v in r["pages"].items()}} for r in measurements],
                      "report_sha256": sha(OUT/"bounded_report.json")}, indent=2))


if __name__ == "__main__":
    main()
