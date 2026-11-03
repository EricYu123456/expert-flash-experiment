#!/usr/bin/env python3
"""Check active supplement freezes and index the completed research evidence."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/research-2026-11-02/round3-cr-ipr"
PROTOCOLS=[
    "g2-availability-full-ffn/protocol.frozen.json",
    "g2-availability-full-ffn-independent/protocol.frozen.json",
    "prospective-routes/protocol.frozen.json",
    "prospective-full-ffn/protocol.frozen.json",
    "native-group-control/v2/protocol.frozen.json",
    "second-shape-native-envelope/protocol.frozen.json",
    "eight-pair-cbc-degree4/protocol.frozen.json",
    "joint-shared-full-ffn/protocol.frozen.json",
    "joint-shared-full-ffn-independent/protocol.frozen.json",
    "joint-shared-fixed-capacity-control/protocol.frozen.json",
    "second-prospective-routes/protocol.frozen.json",
    "second-prospective-routes/comparison.protocol.frozen.json",
]
REPORTS=[
    "g2-availability-full-ffn/TRAIN512/report.json",
    "g2-availability-full-ffn-independent/report.json",
    "g2-availability-headroom/uniform-addendum/report.json",
    "g2-availability-headroom/stage-certificate/report.json",
    "g2-availability-headroom/matching-cpu-benchmark/report.json",
    "g2-availability-headroom/joint8-capacity-certificate/report.json",
    "prospective-routes/report.json", "prospective-full-ffn/report.json",
    "native-group-control/v2/report.json", "second-shape-native-envelope/report.json",
    "eight-pair-cbc-degree4/verification_report.json",
    "eight-pair-cbc-degree4/verification_manifest.json",
    "joint-shared-full-ffn/TRAIN512/report.json",
    "joint-shared-full-ffn/diagnostic-reuse/report.json",
    "joint-shared-full-ffn-independent/report.json",
    "joint-shared-full-ffn-independent/manifest.frozen.json",
    "joint-shared-fixed-capacity-control/TRAIN512/report.json",
    "joint-shared-fixed-capacity-control/exact_scalar_receipt.json",
    "second-prospective-routes/report.json",
    "second-prospective-routes/comparison/report.json",
    "second-prospective-routes/verification/report.json",
    "second-prospective-routes/manifest.frozen.json",
    "joint-figures.json", "README.md", "joint-shared-prior-notes.md",
    "joint-shared-full-ffn/README.md", "g2-availability-full-ffn/README.md",
    "prospective-full-ffn/README.md", "native-group-control/v2/README.md",
    "second-shape-native-envelope/README.md",
]


def sha(path):
    digest=hashlib.sha256()
    with open(path,"rb") as stream:
        for block in iter(lambda:stream.read(8<<20),b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    expected={}
    conflicts=[]
    protocol_hashes={}
    for relative in PROTOCOLS:
        path=BASE/relative
        document=json.loads(path.read_text())
        protocol_hashes[str(path)]=sha(path)
        for target,digest in document["inputs_sha256"].items():
            if target in expected and expected[target]!=digest:
                conflicts.append(dict(path=target,digests=[expected[target],digest],protocol=str(path)))
            expected[target]=digest
    mismatches=[]
    for target,digest in expected.items():
        actual=sha(target)
        if actual!=digest:
            mismatches.append(dict(path=target,expected=digest,actual=actual))
    # The first native driver failed before producing an event, and its source
    # copy/erratum are retained. It is not mislabeled as an active valid freeze.
    erratum=BASE/"native-group-control/erratum.json"
    first=json.loads(erratum.read_text())
    assert sha(BASE/"native-group-control/source.failed-v1.py")==first["source_v1_sha256"]
    report_hashes={str(BASE/relative):sha(BASE/relative) for relative in REPORTS}
    good=not conflicts and not mismatches
    record=dict(
        status="PASS_active_freezes" if good else "FAIL_active_freezes",
        utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),
        active_protocol_count=len(PROTOCOLS),unique_hashed_dependencies=len(expected),
        digest_conflicts=conflicts,dependency_mismatches=mismatches,
        active_protocols_sha256=protocol_hashes,research_artifacts_sha256=report_hashes,
        superseded_native_v1=dict(erratum_path=str(erratum),erratum_sha256=sha(erratum),
            source_copy_matches_original_frozen_hash=True,reason=first["reason"]),
        conclusion="Research checkpoint, not submission-ready. Standard CBC plus movable shared Down has an all-top6 matching/capacity certificate; matched-payload ablation and two authored diagnostic batches are complete. Conditional timings, manuallyinstantiated G4 shape, no nativehardware/newplacedlogit/benchmarkquality claim.",
        primary=dict(
            equal_payload_MiB=303,TRAIN_candidate_fixed_shared_mean_ns=263970.5703125,
            TRAIN_candidate_movable_mean_ns=254712,mean_reduction_pct=3.507425203324488,
            event_P99_reduction_pct=25.29695042907504,
            g4triples_reference_payload_MiB=244.5,total_improvement_vs_g4_mean_pct=9.20199910168754,
            added_payload_vs_g4_MiB=58.5,added_payload_vs_sparseg2_MiB=13.5,
            combinatorial_scope="All C(64,6)=74,974,368 selections, fixed24page full-copy family, eight2chip groups withtwo shared rowhalf objects.",
            second_prospective="16 newlyauthored prompts,16frozen profiles,768events/profile; 1 sparsefallback versus0joint. Not representative benchmark or quality evidence.",
            ordinary_read_mean_gain_vs_lower_payload_g4_pct=1.2291851876713333),
        limitations=[
            "Published/nativelyconfirmed PB copy/overwrite, ECC/PE, cold phase, command ABI and FIFO remain necessary for device claims; source/drop dependencies removed only in nativeg8 branch.",
            "CBC/matching/replication are known. No claim of a new solver, coding theorem, globalstorage/fullFFN optimum or finished reusable generic compiler.",
            "All-top6 certificate is matching/physicalcapacity, not a route-independent guarantee for every GU timing variant; RR microregressions retained.",
            "Independent event replays cover saved representatives; perqueryarrays are statistically/binding checked, not everyquery independently resimulated.",
            "First authored diagnostic was seen before joint design and is labeled reuse; only secondbatch is afterjointfreeze. Neitherbatch proves population generalization.",
        ])
    (BASE/"checkpoint.receipt.json").write_text(json.dumps(record,ensure_ascii=False,indent=2)+"\n")
    print(json.dumps(dict(status=record["status"],protocols=len(PROTOCOLS),dependencies=len(expected),conflicts=len(conflicts),mismatches=len(mismatches))))
    assert good,(conflicts,mismatches)


if __name__=="__main__":main()
