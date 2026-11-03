#!/usr/bin/env python3
"""Replay frozen full saved traces through the existing independent scalar oracle."""
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import sys
import time
import numpy as np

HERE=Path(__file__).resolve().parent
ROOT=HERE.parents[3]
AUDIT=ROOT/'scripts/research_20261102_cr_ipr_full_ffn_audit.py'
PROTOCOL=HERE/'exact_scalar_protocol.frozen.json'

def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def read(p):
    return json.loads(Path(p).read_text())

def dump(p,v):
    Path(p).write_text(json.dumps(v,indent=2)+'\n')

def freeze():
    assert not PROTOCOL.exists()
    index=read(HERE/'TRAIN512/trace_index.json')
    assert len(index)==48
    paths=[Path(__file__),AUDIT,HERE/'control.py',HERE/'protocol.frozen.json',
           HERE/'TRAIN512/report.json',HERE/'TRAIN512/trace_index.json']
    for item in index:
        p=Path(item['path'])
        paths.extend([p,p.with_name(p.stem+'.dispatch.json')])
    dump(PROTOCOL,dict(status='frozen_before_exact_replay',
        scope='All48 full TRAIN512 saved traces; existing independent scalar_execute and semantic_audit imported unchanged. No target C++/Python executor or target audit invoked.',
        inputs_sha256={str(p):sha(p) for p in paths},trace_index=index,
        acceptance='Every node start/end and makespan exactly equal saved primary C++; full existing independent semantic_audit passes; all frozen input hashes unchanged after replay.',
        no_new_routes=True,no_prospective_reads=True))
    print(json.dumps(dict(status='frozen',traces=48,protocol_sha256=sha(PROTOCOL))),flush=True)

def run():
    protocol=read(PROTOCOL)
    for p,h in protocol['inputs_sha256'].items():
        assert sha(p)==h,p
    spec=importlib.util.spec_from_file_location('fixed_shared_independent_scalar_audit',AUDIT)
    audit=importlib.util.module_from_spec(spec)
    sys.modules[spec.name]=audit
    spec.loader.exec_module(audit)
    rows=[];started=time.monotonic()
    for item in protocol['trace_index']:
        p=Path(item['path']);record=read(p)
        independent,cost=audit.scalar_execute(record['nodes'],record['service']['down_first'])
        saved=np.array([[n['start_ns'],n['end_ns']] for n in record['nodes']],dtype=np.int64)
        assert np.array_equal(independent,saved),str(p)
        assert cost==record['cost_ns'],str(p)
        semantic=audit.semantic_audit(record,independent)
        rows.append(dict(path=str(p),trace_sha256=sha(p),node_times_exact=True,makespan_exact=True,
                         cost_ns=int(cost),scalar_trace_sha256=hashlib.sha256(independent.tobytes()).hexdigest(),
                         semantic_audit=semantic))
        if len(rows)%8==0:
            print(json.dumps(dict(finished=len(rows),total=48,elapsed_seconds=time.monotonic()-started)),flush=True)
    unchanged=all(sha(p)==h for p,h in protocol['inputs_sha256'].items())
    assert unchanged
    receipt=dict(status='PASS',frozen_protocol_sha256=sha(PROTOCOL),source_sha256=sha(__file__),
        independent_oracle_source=str(AUDIT),independent_oracle_sha256=sha(AUDIT),
        traces=len(rows),nodes_compared=sum(x['semantic_audit']['nodes'] for x in rows),
        edges_checked=sum(x['semantic_audit']['edges'] for x in rows),
        all_node_times_and_makespans_exact=True,all_semantic_audits_pass=True,
        all_input_source_trace_hashes_unchanged=unchanged,
        elapsed_seconds=time.monotonic()-started,records=rows,
        scope='Independent scalar calendar reconstruction and existing semantic/continuous-drain FIFO audit of all48 saved full traces. Physical resident selection checked separately in saved_trace_checks.json. Finite PB recipe remains inherited, not independently resimulated here. No new route or prospective read.')
    dump(HERE/'exact_scalar_receipt.json',receipt)
    (HERE/'exact_scalar_receipt.md').write_text(
        '# Exact independent scalar replay\n\n'
        f"PASS: all 48 full TRAIN traces, {receipt['nodes_compared']:,} node start/end pairs and all 48 makespans exactly match the saved C++ execution. The unchanged independent `scalar_execute` reconstructs the resource calendars; its `semantic_audit` also passes for every trace, including whole-vector barriers, chip ownership, byte conservation, and continuous output FIFO occupancy.\n\n"
        f"Protocol SHA-256 `{sha(PROTOCOL)}` was saved before replay. The source, control, reports, and all traces retain their frozen hashes afterward. Oracle source SHA-256: `{sha(AUDIT)}`. Physical selection/address checks are in `saved_trace_checks.json`; this receipt adds exact independent calendar replay and does not claim a new finite-latch simulation. No new routes or prospective inputs/outcomes were read.\n")
    print(json.dumps(dict(status='PASS',traces=len(rows),nodes=receipt['nodes_compared'],receipt_sha256=sha(HERE/'exact_scalar_receipt.json'))),flush=True)

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=['freeze','run']);args=parser.parse_args()
    freeze() if args.mode=='freeze' else run()
