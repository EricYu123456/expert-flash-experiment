#!/usr/bin/env python3
"""Predeclared16-profile second authored diagnostic, with no fitting/selection.

Uses frozen graph/recipe builders and the existing sealed C++ event executor.
This is a replay driver, not an independent scheduler or a quality experiment.
"""
from collections import Counter
import argparse
import ast
import datetime
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round3-cr-ipr'
OUT = BASE / 'second-prospective-routes'
DEST = OUT / 'comparison'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    obj = importlib.util.module_from_spec(spec)
    sys.modules[name] = obj
    spec.loader.exec_module(obj)
    return obj


capture = load('second_prospective_capture_frozen', ROOT / 'scripts/research_20261102_cr_second_prospective_capture.py')
driver = load('second_prospective_joint_frozen', ROOT / 'scripts/research_20261102_cr_joint_shared_ffn.py')
f, r, a, g2, joint, native = driver.f, driver.r, driver.a, driver.g2, driver.joint, driver.native
EXISTING_BINARY = BASE / 'joint-shared-full-ffn/build' / ('graph-' + f.sha(f.CPP)[:16] + '.so')


def need(ok, message):
    if not ok:
        raise RuntimeError(message)


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def axes():
    return [(family, 'cr', plan, copies) for family in ['candidate', 'RR']
            for copies in [1, 4] for plan in ['g4triples', 'g2_fitted', 'joint8']] + [
            ('g8_fixed', array, plan, 1) for array in ['cr', 'ordinary']
            for plan in ['g4triples', 'joint8']]


def name_of(axis):
    family, array, plan, copies = axis
    return f'{family}.{array}.{plan}.input{copies}.GF'


def comparisons():
    result = []
    for family in ['candidate', 'RR']:
        for copies in [1, 4]:
            for new, old in [('g2_fitted', 'g4triples'), ('joint8', 'g4triples'), ('joint8', 'g2_fitted')]:
                result.append((name_of((family, 'cr', new, copies)), name_of((family, 'cr', old, copies))))
    for array in ['cr', 'ordinary']:
        result.append((name_of(('g8_fixed', array, 'joint8', 1)), name_of(('g8_fixed', array, 'g4triples', 1))))
    return result


def freeze():
    need(not OUT.exists(), 'new capture directory must not exist before freeze')
    old = ROOT / 'scripts/research_20261102_cr_prospective_capture.py'
    # Capture/extract/run are identical ASTs. Only OUT/PROMPTS are replaced.
    def without_choices(path):
        tree = ast.parse(path.read_text())
        tree.body = [node for node in tree.body if not (isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id in ['OUT', 'PROMPTS'] for target in node.targets))]
        return ast.dump(tree, include_attributes=False)
    need(without_choices(old) == without_choices(capture.SELF), 'capture clone altered inference/extraction logic')
    need(len(capture.PROMPTS) == 16 and Counter(task for task, _ in capture.PROMPTS) == dict(reasoning=4, code=4, zh=4, explanation=4), 'fixed prompt inventory')
    need(len(axes()) == 16 and len(set(map(name_of, axes()))) == 16, 'exact16 comparison configurations')
    need(EXISTING_BINARY.is_file(), 'sealed event executor binary missing')
    need(read(joint.OUT / 'verification_report.json')['status'] == 'PASS', 'joint builder independent verification')
    need(read(BASE / 'joint-shared-full-ffn/protocol.frozen.json')['inputs_sha256'][str(ROOT / 'scripts/research_20261102_eight_pair_cbc.py')] == f.sha(ROOT / 'scripts/research_20261102_eight_pair_cbc.py'), 'joint source frozen identity')
    capture.freeze()
    # This new authoritative receipt is also written before any model execution.
    # The unchanged capture clone's legacy evaluation paragraph is superseded.
    paths = [Path(__file__).resolve(), capture.SELF, old, capture.BINARY, capture.PLAN,
             OUT / 'protocol.frozen.json', EXISTING_BINARY,
             ROOT / 'scripts/research_20261102_cr_joint_shared_ffn.py',
             ROOT / 'scripts/research_20261102_cr_g2_availability_ffn.py',
             ROOT / 'scripts/research_20261102_eight_pair_cbc.py',
             ROOT / 'scripts/research_20261102_cr_native_group_control.py',
             ROOT / 'scripts/research_20261102_cr_ipr_full_ffn.py', f.CPP, f.PIPE,
             Path(r.__file__), Path(r.old.__file__), Path(r.old.policy.__file__), Path(a.__file__), a.SHAPER,
             g2.KERNEL, g2.OUT / 'protocol.frozen.json', g2.OUT / 'support_maps.frozen.npz',
             g2.OUT / 'resident_layouts.frozen.json', g2.OUT / 'resident_jobs.frozen.json',
             joint.OUT / 'protocol.frozen.json', joint.OUT / 'layout.frozen.json', joint.OUT / 'resident_jobs.frozen.json',
             joint.OUT / 'verification_report.json', joint.OUT / 'verification_manifest.json',
             BASE / 'joint-shared-full-ffn/protocol.frozen.json',
             r.OUT / 'Down_job_maps.frozen.json', native.OUT / 'protocol.frozen.json',
             r.BASE / 'g8-addendum/group_maps.frozen.npz',
             ROOT / 'scripts/research_20261102_cr_ipr_full_ffn_audit.py',
             ROOT / 'scripts/research_20261102_joint_shared_full_ffn_audit.py',
             ROOT / 'scripts/research_20261102_g2_availability_full_ffn_audit.py']
    cp = read(OUT / 'protocol.frozen.json')
    model_digest = capture.sha(capture.MODEL)
    need(model_digest == cp['model_sha256_inherited'], 'actual model SHA differs from inherited sealed identity')
    need(capture.MODEL.stat().st_mtime_ns == cp['model_stat']['mtime_ns'], 'model changed during identity hash')
    write(OUT / 'comparison.protocol.frozen.json', dict(
        status='frozen_before_second_model_execution', utc=f.utc(), inputs_sha256={str(p): capture.sha(p) for p in paths},
        model=dict(path=str(capture.MODEL), sha256=model_digest, **cp['model_stat']),
        prompts=cp['jobs'], prompt_choice_rule='Authored once in four categories,4each; topics changed from prior16prompts. No route/response seen, selected, filtered or replaced. Every prompt retained irrespective of outcome.',
        capture='Exact original capture/extract/run AST exceptOUT/PROMPTS.16steps,8threads/process,2batches of8prompts,1200s capture ceiling; incomplete data retained and no fullsample claim.',
        authoritative_evaluation='This protocol supersedes only the legacy evaluation paragraph in the unchanged capture clone protocol. No IDslot/seed fit or all40 matching comparison is added.',
        axes=[dict(name=name_of(x), family=x[0], array_mode=x[1], plan=x[2], input_copies=x[3], priority='GF') for x in axes()],
        configuration_count=16, event_count_per_configuration=768, route_rule='First3layers,all16prompts×16decode steps,256inputs/layer in fixed prompt/step order. No sample selection.',
        comparisons=[dict(candidate=new, reference=old) for new, old in comparisons()],
        storage_MiB=dict(g4triples=244.5, g2_fitted=289.5, joint8=303),
        contracts='Unchanged frozentrained sparseg2 supports and layout; jointdegree4alltop6 Hall certificate; standard replication. No new fitting, solver, winner, timingprofile, buffer, input splitting, context or hardware capability.',
        resources='Hostsubmit2000ns,command250ns,8channel×2B/ns,PCIe8B/ns,CPUactivation/combine0; fullGU→hostA8→privatefullC512/1024Down→combine; one jobowner/chip until compute+ONFI ends. Nativeg8selected4plane sameWL fixed0123, no drop/shape, CR coldreset explicitlycharged; ordinaryallcold28us model.',
        statistics='Report everyconfig,PBfailure/FIFO128invalid entry,cap2fallback,eventmean/P99/min/max,aligned3layersumP99,allpairedslower/equal/faster. Ratios preserve added storage cost. No post-result selection or axis changes.',
        trace_rule='Save everyprofile/layer firstquery plus first cap2fallback perprofile regardless of normalized cache hit; independently replay every saved representative with existing scalar/semantic/FIFO checker.',
        binary='Reuse this exact sealed C++ event binary; do not recompile. Capture uses exact inherited canonicalW8 helper binary andmodel.',
        limits='Second authored prospective diagnostic after layout/code freeze; not a benchmark, quality study, representative population or external-validity proof. Hall provesalltop6 matching, not empirical statistics. CanonicalW8/A8 generated routes do not establish originalGGUF or placed-logit equivalence. Conditional timing/MAC/coldphase/PB/source/drop contracts remain; no flashmeasurement. Matchingruntime excluded.'))
    print(json.dumps(dict(status='comparison_frozen_before_model', axes=16,
                          capture_protocol_sha256=f.sha(OUT / 'protocol.frozen.json'),
                          comparison_protocol_sha256=f.sha(OUT / 'comparison.protocol.frozen.json'),
                          model_sha256=model_digest)), flush=True)


def checked(require_capture=True):
    protocol = read(OUT / 'comparison.protocol.frozen.json')
    for path, expected in protocol['inputs_sha256'].items():
        need(capture.sha(path) == expected, 'frozen comparison dependency changed: ' + path)
    capture.checked()
    if require_capture:
        report = read(OUT / 'report.json')
        need(report['status'] == 'complete' and report['prompt_count'] == 16 and report['decode_sha256'] == f.sha(OUT / 'decode.routes.npz'), 'all16 complete canonical captures required')
    return protocol


def capture_run():
    checked(False)
    write(OUT / 'pre_execution.receipt.json', dict(utc=f.utc(), comparison_protocol_sha256=f.sha(OUT / 'comparison.protocol.frozen.json'),
          capture_protocol_sha256=f.sha(OUT / 'protocol.frozen.json'), all16_unchanged=True, model_not_started_by_this_driver_yet=True))
    capture.run()


def run():
    protocol = checked()
    DEST.mkdir(exist_ok=True)
    need(not (DEST / 'report.json').exists(), 'comparison report already exists')
    began = time.monotonic()
    with np.load(OUT / 'decode.routes.npz') as z:
        need(z['completed'].all() and z['routes'].shape == (16, 16, 40, 6), 'full fixed decode route inventory')
        routes = z['routes'].transpose(2, 0, 1, 3).reshape(40, 256, 6)[:3]
    with np.load(r.BASE / 'g8-addendum/group_maps.frozen.npz') as z:
        groups = z['learned'].copy()
    layouts = read(g2.OUT / 'resident_layouts.frozen.json')
    resident = [g2.resident_jobs(x) for x in layouts]
    joint_layout = read(joint.OUT / 'layout.frozen.json')
    triple_maps = read(r.OUT / 'Down_job_maps.frozen.json')
    executor = f.Executor(BASE / 'joint-shared-full-ffn')
    need(executor.receipt['binary_sha256'] == protocol['inputs_sha256'][str(EXISTING_BINARY)], 'event binary must be exact sealed build')
    write(DEST / 'run.frozen.json', dict(utc=f.utc(), comparison_protocol_sha256=f.sha(OUT / 'comparison.protocol.frozen.json'),
          capture_report_sha256=f.sha(OUT / 'report.json'), routes_sha256=f.sha(OUT / 'decode.routes.npz'), source_sha256=f.sha(__file__),
          executor=executor.receipt, axes=[name_of(x) for x in axes()], no_refitting_or_selection=True))
    arrays, records, cache, pipelines, failures = {}, [], {}, {}, []
    for ci, axis in enumerate(axes()):
        family, array, plan, copies = axis
        name = name_of(axis)
        cost = np.full((3, 256), -1, dtype=np.int64);fifo = cost.copy();capacity = cost.copy()
        aggregate = Counter();fallback_saved = False
        pkey = (family == 'g8_fixed', array)
        if pkey not in pipelines:
            pipelines[pkey] = native.Recipes(f.load_pipeline(), a.load_shaper(), array) if pkey[0] else r.Recipes(f.load_pipeline(), a.load_shaper())
        recipes = pipelines[pkey]
        for layer in range(3):
            for qi, route in enumerate(routes[layer]):
                cap = 0
                if plan == 'g4triples':
                    maps, graphplan = triple_maps, 'triple_private_shared4'
                elif plan == 'g2_fitted':
                    decision = g2.dispatch(route, layouts[layer]);cap = decision['capacity']
                    selected = [job for e, group in decision['assignment'].items() for job in resident[layer][int(e), int(group)]]
                    selected += resident[layer][64, 0]
                    graphplan = 'g2availability';maps = {family + '.' + graphplan: [selected] * 3}
                else:
                    decision = joint.assign_route(route, joint_layout);cap = decision['capacity']
                    need(decision['feasible'] and decision['group_load'] == [1] * 8, 'joint dispatch must retainallsix plussharedhalves')
                    graphplan = 'joint8';maps = {family + '.joint8': [decision['selected_jobs']] * 3}
                capacity[layer, qi] = cap
                graph = r.graph_for(route, layer, family, graphplan, f.Service(copies, 0, 0), recipes, groups, maps)
                if graph is None:
                    failures.append(dict(name=name, layer=layer, query_index=qi, route=route.tolist(), reason='finitePBkernel_infeasible'))
                    continue
                signature, data = graph.signature()
                signature = (signature, array, json.dumps(dict(graph.pipeline_totals), sort_keys=True))
                save = qi == 0 or (cap == 2 and not fallback_saved)
                if signature in cache and not save:
                    value, audit = cache[signature]
                else:
                    result = executor.run(graph, data)
                    audit = f.audit(graph, result, full=save)
                    value = result['cost_ns'];cache[signature] = value, audit
                    if save:
                        label = 'first' if qi == 0 else 'fallback'
                        f.dump_trace(DEST / f'trace.L{layer}.{name}.{label}.q{qi}.json', graph, result, audit)
                if cap == 2:
                    fallback_saved = True
                cost[layer, qi] = value;fifo[layer, qi] = audit['peak_output_fifo_bytes']
                if fifo[layer, qi] > 128:
                    failures.append(dict(name=name, layer=layer, query_index=qi, route=route.tolist(), reason='outputFIFO128_exceeded', peak_bytes=int(fifo[layer, qi])))
                aggregate.update(graph.pipeline_totals)
        arrays[name + '.cost'] = cost;arrays[name + '.fifo'] = fifo;arrays[name + '.matching_capacity'] = capacity
        valid = (cost >= 0) & (fifo >= 0) & (fifo <= 128)
        record = dict(name=name, requested_events=768, completed_events=int(np.count_nonzero(cost >= 0)),
                      FIFO128_valid_events=int(np.count_nonzero(valid)), all_primary_events_valid=bool(valid.all()),
                      cost_ns=f.stats(cost[cost >= 0]) if np.any(cost >= 0) else None,
                      three_layer_sum_ns=f.stats(cost.sum(axis=0)) if np.all(cost >= 0) else None,
                      FIFO_peak_bytes=f.stats(fifo[fifo >= 0]) if np.any(fifo >= 0) else None,
                      fallback_events=int(np.count_nonzero(capacity == 2)), payload_MiB=protocol['storage_MiB'][plan],
                      pipeline_counts_per_completed_event={k: v / max(1, np.count_nonzero(cost >= 0)) for k, v in aggregate.items()})
        records.append(record)
        write(DEST / 'progress.json', dict(finished=ci + 1, total=16, last=name, elapsed_seconds=time.monotonic() - began))
        print(json.dumps(dict(finished=ci + 1, total=16, name=name, mean_ns=record['cost_ns'], fallback_events=record['fallback_events'], FIFO128_valid=record['all_primary_events_valid'])), flush=True)
    paired = []
    for new, old in comparisons():
        nc, oc = arrays[new + '.cost'], arrays[old + '.cost']
        valid = (nc >= 0) & (oc >= 0) & (arrays[new + '.fifo'] <= 128) & (arrays[old + '.fifo'] <= 128)
        rec = dict(candidate=new, reference=old, requested_events=768, jointly_valid_events=int(np.count_nonzero(valid)),
                   all_events_jointly_valid=bool(valid.all()), candidate_payload_MiB=next(x['payload_MiB'] for x in records if x['name'] == new),
                   reference_payload_MiB=next(x['payload_MiB'] for x in records if x['name'] == old))
        if valid.any():
            nv, ov = nc[valid], oc[valid]
            rec.update(mean_difference_ns=float((nv - ov).mean()), mean_reduction_pct=100 * (1 - float(nv.mean() / ov.mean())),
                       event_P99_reduction_pct=100 * (1 - float(np.percentile(nv, 99) / np.percentile(ov, 99))),
                       slower_events=int(np.count_nonzero(nv > ov)), equal_events=int(np.count_nonzero(nv == ov)), faster_events=int(np.count_nonzero(nv < ov)))
        if valid.all():
            rec['three_layer_sum_P99_reduction_pct'] = 100 * (1 - float(np.percentile(nc.sum(axis=0), 99) / np.percentile(oc.sum(axis=0), 99)))
        paired.append(rec)
    np.savez_compressed(DEST / 'service_arrays.npz', **arrays)
    write(DEST / 'failures.json', failures)
    write(DEST / 'report.json', dict(status='complete_second_authored_diagnostic', records=records, comparisons=paired,
          failures=len(failures), source_sha256=f.sha(__file__), comparison_protocol_sha256=f.sha(OUT / 'comparison.protocol.frozen.json'),
          capture_report_sha256=f.sha(OUT / 'report.json'), route_sha256=f.sha(OUT / 'decode.routes.npz'),
          unique_normalized_graphs=len(cache), elapsed_seconds=time.monotonic() - began,
          scope=protocol['limits'], no_result_selection=True))
    print(json.dumps(dict(status='complete16configs', events=16 * 768, failures=len(failures), elapsed_seconds=time.monotonic() - began)), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['freeze', 'capture', 'run'])
    mode = parser.parse_args().mode
    {'freeze': freeze, 'capture': capture_run, 'run': run}[mode]()
