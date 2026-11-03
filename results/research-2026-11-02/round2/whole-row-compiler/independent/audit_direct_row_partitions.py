#!/usr/bin/env python3
"""Independent audit: enumerate integer row partitions, not wave-capacity DP.

All permutations of a row partition have the same cost on identical chips, so
nondecreasing positive partitions enumerate every cost-equivalence class. The
source under review is copied verbatim by SHA before it is imported/executed.
"""
import hashlib
import importlib.util
import json
import sys
from datetime import datetime, timezone
from functools import lru_cache
from math import gcd
from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
OUT = Path(__file__).resolve().parent
SOURCE = ROOT / "scripts/research_20261102_whole_row_compiler.py"
CERTIFICATES = OUT.parent / "certificates.json"


def sha(data):
    return hashlib.sha256(data).hexdigest()


def ceildiv(a, b):
    return (a + b - 1) // b


@lru_cache(None)
def partitions(rows, chips, least=1):
    if rows == 0:
        return ((),)
    if chips == 0:
        return ()
    ans = []
    for first in range(least, rows + 1):
        for tail in partitions(rows - first, chips - 1, first):
            ans.append((first,) + tail)
    return tuple(ans)


def inspect_profile(profile, total_rows, columns, cap):
    """Check a concrete emitted witness without using compiler helpers."""
    rb = columns * cap.weight_bytes
    b = cap.planes * cap.page_bytes
    errors = []
    rows = profile["rows"]
    waves = profile["waves"]
    expected_waves = [ceildiv(r * rb, b) for r in rows]
    if len(rows) != cap.chips or len(waves) != cap.chips:
        errors.append("chip vector length")
    if any(not isinstance(r, int) or r < 0 for r in rows):
        errors.append("nonnegative integer rows")
    if sum(rows) != total_rows or expected_waves != waves:
        errors.append("row coverage or independently computed wave costs")
    cursor = 0
    jobs = {j["chip"]: j for j in profile["jobs"]}
    if len(jobs) != len(profile["jobs"]):
        errors.append("duplicate chip jobs")
    for chip, r in enumerate(rows):
        if r:
            j = jobs.get(chip, {})
            expected = dict(chip=chip, row_start=cursor, rows=r,
                            column_start=0, columns=columns,
                            useful_bytes=r * rb,
                            pages=ceildiv(r * rb, cap.page_bytes),
                            waves=expected_waves[chip],
                            block_commands=ceildiv(expected_waves[chip], cap.block_wordlines))
            if j != expected:
                errors.append(f"job {chip} coordinates/costs")
        elif chip in jobs:
            errors.append(f"empty chip {chip} has job")
        cursor += r
    checks = dict(total_waves=sum(waves), max_job_waves=max(waves, default=0),
                  positive_jobs=sum(r > 0 for r in rows),
                  single_block_jobs=all(w <= cap.block_wordlines for w in waves))
    for key, value in checks.items():
        if profile[key] != value:
            errors.append(key)
    return dict(pass_=not errors, errors=errors, **checks,
                useful_pages=sum(ceildiv(r * rb, cap.page_bytes) for r in rows),
                ordinary_full_plane_read_pages=sum(waves) * cap.planes,
                physical_allocation_scope="Each positive job can start at a fresh wave-aligned block; logical certificate does not emit packed multi-job physical addresses.")


def main():
    source_bytes = SOURCE.read_bytes()
    source_sha = sha(source_bytes)
    frozen = OUT / f"compiler_source_{source_sha}.py"
    frozen.write_bytes(source_bytes)
    spec = importlib.util.spec_from_file_location("whole_row_source_under_audit", frozen)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    failures = []
    partition_classes = 0
    comparisons = 0
    lemma_cases = 0
    shapes = 0
    for b in range(1, 9):
        for c in range(1, 13):
            for r in range(1, 10):
                for m in range(1, 5):
                    shapes += 1
                    all_parts = partitions(r, m)
                    partition_classes += len(all_parts)
                    observed = []
                    for part in all_parts:
                        costs = tuple(ceildiv(n * c, b) for n in part)
                        observed.append((max(costs), sum(costs), part))
                    actual_dmin = min(x[0] for x in observed)
                    formula_dmin = ceildiv(ceildiv(r, m) * c, b)
                    if actual_dmin != formula_dmin:
                        failures.append(dict(kind="D_min", B=b, C=c, R=r, M=m,
                                             actual=actual_dmin, formula=formula_dmin))
                    if (r * c) % b == 0:
                        lemma_cases += 1
                        w = r * c // b
                        q = c // gcd(c, b)
                        exact = [x for x in observed if x[1] == w]
                        predicted = q * ceildiv(w // q, m)
                        actual = min(x[0] for x in exact)
                        if actual != predicted:
                            failures.append(dict(kind="D_exact", B=b, C=c, R=r, M=m,
                                                 actual=actual, formula=predicted))
                        for _, _, part in exact:
                            if any((n*c) % b for n in part):
                                failures.append(dict(kind="individual exact fill", B=b, C=c, R=r, M=m))
                    cap = module.Capability(chips=m, planes=1, page_bytes=b, block_wordlines=1000)
                    for depth in range(1, ceildiv(r * c, b) + 1):
                        comparisons += 1
                        feasible = [x for x in observed if x[0] <= depth]
                        expected = min((x[1] for x in feasible), default=None)
                        try:
                            result = module.minimum_work(r, c, depth, cap)
                            actual = None if result is None else result["optimal_total_waves"]
                            errors = [] if result is None else inspect_profile(result["witness"], r, c, cap)["errors"]
                            if actual != expected or errors:
                                failures.append(dict(kind="DP versus direct row partition", B=b, C=c, R=r, M=m,
                                                     depth=depth, expected=expected, actual=actual, errors=errors))
                        except Exception as error:
                            failures.append(dict(kind="DP exception", B=b, C=c, R=r, M=m,
                                                 depth=depth, error=repr(error)))
    cert_bytes = CERTIFICATES.read_bytes()
    cert = json.loads(cert_bytes)
    cap = module.Capability(**cert["capability"])
    profiles = []
    for rec in cert["records"]:
        r, c = rec["rows"], rec["columns"]
        rb, b = c * cap.weight_bytes, cap.planes * cap.page_bytes
        q = rb // gcd(rb, b)
        w = r * rb // b if r * rb % b == 0 else None
        arithmetic = rec["arithmetic"]
        checks = []
        candidates = [("equal_wave_profile", rec["equal_wave_profile"]),
                      ("balanced_wave_profile", rec["balanced_wave_profile"])]
        if "exact_work_witness" in arithmetic:
            candidates.append(("exact_work_witness", arithmetic["exact_work_witness"]))
        candidates.extend((f"depth_{x['requested_max_job_waves']}", x["witness"])
                          for x in rec["depth_constrained_certificates"])
        for name, profile in candidates:
            checks.append(dict(name=name, **inspect_profile(profile, r, c, cap)))
        expected_equal = [r // cap.chips + int(i < r % cap.chips) for i in range(cap.chips)]
        expected_dmin = ceildiv(ceildiv(r, cap.chips) * rb, b)
        expected_dexact = None if w is None else q * ceildiv(w // q, cap.chips)
        errors = []
        if rec["equal_wave_profile"]["original_equal_rows"] != expected_equal:
            errors.append("equal rows")
        if arithmetic["minimum_job_depth"] != expected_dmin:
            errors.append("minimum depth")
        if arithmetic.get("exact_work_minimum_job_depth") != expected_dexact:
            errors.append("exact-work minimum depth")
        if arithmetic["wave_period"] != q:
            errors.append("wave period")
        if not all(x["pass_"] for x in checks):
            errors.append("profile coverage/cost")
        profiles.append(dict(model=rec["model"], phase=rec["phase"], rows=r, columns=c,
                             exact_payload_waves=w, row_period=b//gcd(rb,b), wave_period=q,
                             minimum_depth=expected_dmin, exact_work_minimum_depth=expected_dexact,
                             pass_=not errors, errors=errors, profiles=checks))
    # Demonstrate the full certificate API edge case separately from the DP.
    edge_cap = module.Capability(chips=4, planes=1, page_bytes=2, block_wordlines=1000)
    try:
        edge_result = module.certificate(1, 3, edge_cap)
        edge = dict(status="certificate returned", balanced=edge_result.get("balanced_wave_profile"))
    except Exception as error:
        edge = dict(status="certificate raised", exception=repr(error))
    edge.update(R=1, C=3, B=2, M=4, valid_rows=[1,0,0,0], valid_waves=[2,0,0,0],
                cause="No globally floor/ceil-balanced wave vector is feasible at any work <= equal-row work; balanced profile should be nullable or separately scoped.")
    doc = dict(created_utc=datetime.now(timezone.utc).isoformat(),
               audited_source=str(SOURCE), executed_source_sha256=source_sha,
               frozen_executed_source=str(frozen), certificate_sha256=sha(cert_bytes),
               certificate_embedded_source_sha256=cert["source_sha256"],
               certificate_source_matches_executed=cert["source_sha256"]==source_sha,
               source_sha256_after_run=sha(SOURCE.read_bytes()),
               oracle="Exhaustive nondecreasing positive integer partitions of R into <=M jobs; direct ceil(r_i*C/B) per partition. No wave interval/capacity DP used in oracle.",
               tiny_bounds=dict(B=[1,8],C=[1,12],R=[1,9],M=[1,4],
                                depth="every integer from 1 through ceil(R*C/B)"),
               counts=dict(shapes=shapes,partition_cost_classes=partition_classes,
                           depth_comparisons=comparisons,exact_fill_lemma_cases=lemma_cases),
               minimum_work_dp_pass=not failures, failures=failures,
               generated_profiles=profiles, all_generated_profiles_pass=all(x["pass_"] for x in profiles),
               full_certificate_api_edge=edge,
               limits=["Zero-row matrices are outside the current positive-row API, although the arithmetic extends with zero work/depth.",
                       "Single-block is independently checked against a fresh wave-aligned block per job; dense packing, start offsets and reservation capacity need a physical address certificate.",
                       "Coefficient-only useful pages differ from full-plane ordinary-read pages; the latter equal waves*planes.",
                       "No claim about native latency, quantization metadata, PE output ABI, multi-block continuation or schedule optimality."])
    target = OUT / f"direct_partition_audit_{source_sha[:12]}.json"
    target.write_text(json.dumps(doc, indent=2)+"\n")
    print(json.dumps(dict(output=str(target),counts=doc["counts"],dp_pass=doc["minimum_work_dp_pass"],
                          generated_profiles_pass=doc["all_generated_profiles_pass"],
                          certificate_source_matches=doc["certificate_source_matches_executed"],
                          edge=edge),indent=2))


if __name__ == "__main__":
    main()
