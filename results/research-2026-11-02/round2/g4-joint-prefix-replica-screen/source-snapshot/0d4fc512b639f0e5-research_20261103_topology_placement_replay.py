#!/usr/bin/env python3
"""Bounded TRAIN-only ONFI topology diagnostic; no plan/backend mutation."""
import ctypes
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2/ordinary-read-compiler'
OUT = ROOT / 'results/research-2026-11-02/round2/topology-placement'
SNAP = BASE / 'full-campaign-v6-W8-input4/source-snapshot'
CPP = SNAP / 'ef7994fa2ba37a4b-research_20261103_ordinary_read_native.cpp'
PYTHON = SNAP / '7c61f09756e24f11-research_20261103_ordinary_read_backend.py'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def load_python():
    spec = importlib.util.spec_from_file_location('topology_sealed_backend', PYTHON)
    m = importlib.util.module_from_spec(spec); sys.modules[spec.name] = m
    spec.loader.exec_module(m)
    return m


class Replay:
    def __init__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        binary = OUT / ('native-' + sha(CPP)[:16] + '.so')
        if not binary.exists():
            subprocess.run(['c++', '-std=c++17', '-O3', '-shared', '-fPIC', str(CPP), '-o', str(binary)], check=True)
        self.lib = ctypes.CDLL(str(binary)); self.call = self.lib.eval_ordinary_read
        p32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        p64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        self.call.argtypes = [p32,p32,p32,p32,ctypes.c_int,ctypes.c_int,ctypes.c_int,
                              ctypes.c_int,ctypes.c_int,p64,p64,p64,ctypes.c_void_p,p64]
        self.call.restype = None
        self.binary = binary

    def run(self, plan, routes, service, permutation=None, channels=None):
        jobs = plan['jobs'].copy()
        if permutation is not None:
            jobs[:,0] = np.asarray(permutation)[jobs[:,0]]
        config = dict(service)
        if channels is not None:
            config['channels'] = channels
        config.setdefault('input_channel_copies', 1)
        names = ['chips','channels','planes','page_bytes','block_wordlines','input_bytes',
                 'output_bytes','ordinary_read_ns','command_ns','host_submit_ns',
                 'pcie_bytes_per_second','channel_bytes_per_second','activation_ns',
                 'combine_ns','output_fringe','down_first','input_buffer_bytes','input_channel_copies']
        config = np.asarray([config[k] for k in names], np.int64)
        routes = np.ascontiguousarray(routes, np.int32)
        result = np.zeros(len(routes), np.int64); bounds = result.copy()
        nlog = np.zeros(1, np.int64)
        self.call(plan['offsets'], jobs, plan['input_columns'], routes, len(plan['offsets'])-1,
                  1, len(jobs), len(routes), routes.shape[1], config, result, bounds, None, nlog)
        return result, bounds


def main():
    native = Replay(); backend = load_python()
    from research_20261103_topology_placement import certificate
    rng = np.random.default_rng(20261103)
    permutations = [np.arange(16, dtype=np.int32)]
    # Adjacent virtual chips share a physical channel instead of opposite chips.
    permutations.append(np.asarray([x for pair in zip(range(8), range(8,16)) for x in pair], np.int32))
    permutations.extend(rng.permutation(16).astype(np.int32) for _ in range(30))
    records = []; arrays = {}; checks = []
    for campaign in ['full-campaign-v5-W8-training', 'full-campaign-v6-W8-input4']:
        d = json.loads((BASE / campaign / 'summary.json').read_text())
        for rec in d['records']:
            if rec['layer'] >= 3:
                continue
            for art in rec['artifacts']:
                with np.load(ROOT / art['arrays'], allow_pickle=False) as f:
                    plan = {k:f[k] for k in f.files}
                routes = plan['selection_routes']
                original, native_bound = native.run(plan, routes, rec['service'])
                assert np.array_equal(original, plan['cost'])
                independent_bound = certificate(plan, rec['service'])['lower']
                # Our formula adds a valid initial-submit prefix to PCIe/channel
                # work and pools channels; on these NAND-bound cases it exactly
                # matches the independently constructed native-node bound.
                assert np.array_equal(independent_bound, native_bound)
                private, _ = native.run(plan, routes, rec['service'], channels=16)
                # At 250ns the proven bound is already small: do not run a fit.
                candidates = permutations if rec['command_ns'] == 5000 else permutations[:1]
                scores = np.stack([native.run(plan, routes, rec['service'], p)[0] for p in candidates])
                best = int(np.argmin(scores.mean(1)))
                ident = f"{campaign}__{rec['model']}__cmd{rec['command_ns']}__L{rec['layer']}__{art['family']}"
                arrays[ident+'__original'] = original; arrays[ident+'__private_channels'] = private
                arrays[ident+'__candidates'] = scores
                records.append(dict(campaign=campaign, model=rec['model'], layer=rec['layer'],
                                    command_ns=rec['command_ns'], family=art['family'], queries=len(routes),
                                    original_mean_ns=float(original.mean()), private_channel_mean_ns=float(private.mean()),
                                    private_channel_relaxation_mean_reduction_pct=float(100*(1-private.mean()/original.mean())),
                                    candidate_count=len(candidates), best_candidate=best,
                                    best_permutation=candidates[best].tolist(), best_mean_ns=float(scores[best].mean()),
                                    best_mean_reduction_pct=float(100*(1-scores[best].mean()/original.mean())),
                                    best_p99_reduction_pct=float(100*(1-np.percentile(scores[best],99)/np.percentile(original,99)))))
                # Representative direct Python graph checks, preserving original
                # logical-row/job order and changing only physical chip addresses.
                if rec['layer'] == 0 and art['family'] == 'whole_rows':
                    for tag, perm, channels in [('identity',permutations[0],8),
                                                 ('private',permutations[0],16),
                                                 ('best',candidates[best],8)]:
                        p = dict(plan); p['jobs'] = plan['jobs'].copy(); p['jobs'][:,0] = perm[p['jobs'][:,0]]
                        s = dict(rec['service']); s['channels'] = channels
                        chosen = np.r_[routes[0], len(plan['offsets'])-1]
                        graph = backend.ReferenceGraph.build(p, chosen, backend.Service(**s))
                        got = graph.run(); expected = native.run(plan,routes[:1],rec['service'],perm,channels)[0][0]
                        assert got['cost'] == expected
                        waits = []
                        for i,n in enumerate(graph.nodes):
                            if 3 <= n['resource'] < 3+channels:
                                waits.append(int(got['trace'][i,2] - max((got['trace'][j,3] for j in n['deps']), default=0)))
                        checks.append(dict(campaign=campaign,model=rec['model'],command_ns=rec['command_ns'],
                                           case=tag, cost_ns=int(expected), nodes=len(graph.nodes),
                                           total_channel_queue_wait_ns=sum(waits), max_channel_queue_wait_ns=max(waits)))
    np.savez_compressed(OUT / 'small_replay.npz', **arrays)
    result = dict(scope='TRAIN64 selection queries; first3layers of G4/Deep; every disjoint/cyclic/whole_rows control; both input-copy contracts; no external access.',
                  warning='Private-channel graph is a diagnostic relaxation on the same greedy scheduler, not a proven global-schedule bound. Candidate selection uses TRAIN mean only. 32 fixed common permutations at 5us command, identity only at250ns. No placement fits, new seeds for existing labels, or frozen-plan mutation.',
                  chip_permutation_semantics='All GU/Down/shared jobs move together. Logical row and native command order retained; complete rows, per-chip work, inventory multiset and jobs unchanged. Only which pair shares each ONFI channel changes.',
                  source_cpp=dict(path=str(CPP),sha256=sha(CPP)), source_python=dict(path=str(PYTHON),sha256=sha(PYTHON)),
                  binary=dict(path=str(native.binary),sha256=sha(native.binary)),script_sha256=sha(__file__),
                  validation=dict(identity_costs_exact=72*64,
                                  independent_vs_native_node_bounds_exact=72*64,
                                  python_graph_checks=len(checks),
                                  python_graph_nodes=sum(c['nodes'] for c in checks)),
                  checks=checks,records=records)
    (OUT/'small_replay.json').write_text(json.dumps(result,indent=2)+'\n')
    for campaign in ['full-campaign-v5-W8-training','full-campaign-v6-W8-input4']:
        for model in ['granite4tiny','deepseek']:
            for cmd in [250,5000]:
                for family in ['disjoint','cyclic','whole_rows']:
                    rr=[r for r in records if (r['campaign'],r['model'],r['command_ns'],r['family'])==(campaign,model,cmd,family)]
                    orig=sum(r['original_mean_ns'] for r in rr); private=sum(r['private_channel_mean_ns'] for r in rr)
                    best=sum(r['best_mean_ns'] for r in rr)
                    print(campaign,model,cmd,family,'private_pct',round(100*(1-private/orig),6),'best32_pct',round(100*(1-best/orig),6))


if __name__ == '__main__':
    main()
