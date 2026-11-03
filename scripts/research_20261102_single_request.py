#!/usr/bin/env python3
"""Train-only, single-request ordinary-read dependency and resource study.

An event-driven feasible scheduler is compared with its own global-barrier
version. Both equal-g16 and the strongest frozen projection layouts use exactly
the same resources, priorities, command costs and host round trips. The only
new placement candidate permutes the down projection's fixed chip columns to
reduce combined gate/up+down chip work; no weights are duplicated or moved at
runtime. Relaxed workload lower bounds use *all* train routes; event schedules
use a deterministic bounded train sample, explicitly reported as such.
"""
import argparse
from collections import Counter
import heapq
import json
from pathlib import Path
import random
import time

import numpy as np

from run_corpus_traces import ROOT, MODELS, sha256

DEST = ROOT / 'results/research-2026-11-02/round2/single-request'
RESOURCE_NAMES = ['host', 'controller', 'pcie'] + [f'channel{c}' for c in range(8)] + [f'chip{c}' for c in range(16)] + ['sync']


def ceildiv(a, b):
    return (a + b - 1) // b


def profiles(rows, columns):
    rows = np.asarray(rows, dtype=np.int64)
    pages = ceildiv(rows * columns, 16384)
    waves = ceildiv(pages, 4)
    return dict(rows=rows, columns=columns, pages=pages, waves=waves, runs=ceildiv(waves, 192))


class Graph:
    def __init__(self):
        self.nodes = []

    def add(self, resource, duration, deps=(), phase=0, expert=-1, kind='', chip=-1, release_chip=-1):
        index = len(self.nodes)
        assert duration >= 0
        self.nodes.append(dict(resource=resource, duration=int(duration), deps=list(deps),
                               phase=phase, expert=expert, kind=kind, chip=chip,
                               release_chip=release_chip))
        return index

    def run(self, down_first=True, keep_trace=False):
        nodes = self.nodes
        successors = [[] for _ in nodes]
        remaining = []
        ready = [[] for _ in RESOURCE_NAMES]
        busy = [False] * len(RESOURCE_NAMES)
        blocked_chip = [False] * 16
        starts = np.full(len(nodes), -1, dtype=np.int64)
        finishes = np.full(len(nodes), -1, dtype=np.int64)
        resource_work = np.zeros(len(RESOURCE_NAMES), dtype=np.int64)
        events = []
        for i, node in enumerate(nodes):
            remaining.append(len(node['deps']))
            for dep in node['deps']:
                assert dep < i
                successors[dep].append(i)

        def enqueue(i):
            node = nodes[i]
            phase_priority = (1-node['phase']) if down_first else node['phase']
            priority = (phase_priority, node['expert'], i)
            heapq.heappush(ready[node['resource']], (priority, i))

        for i, count in enumerate(remaining):
            if not count:
                enqueue(i)
        now = completed = 0
        while completed < len(nodes):
            for resource, queue in enumerate(ready):
                chip = resource - 11
                if busy[resource] or not queue or (0 <= chip < 16 and blocked_chip[chip]):
                    continue
                _, index = heapq.heappop(queue)
                node = nodes[index]
                starts[index] = now
                busy[resource] = True
                end = now + node['duration']
                resource_work[resource] += node['duration']
                heapq.heappush(events, (end, index))
            if not events:
                raise AssertionError('Dependency/resource deadlock')
            now = events[0][0]
            while events and events[0][0] == now:
                _, index = heapq.heappop(events)
                node = nodes[index]
                finishes[index] = now
                busy[node['resource']] = False
                if node['kind'] == 'compute':
                    assert not blocked_chip[node['chip']]
                    blocked_chip[node['chip']] = True
                if node['release_chip'] >= 0:
                    assert blocked_chip[node['release_chip']]
                    blocked_chip[node['release_chip']] = False
                completed += 1
                for successor in successors[index]:
                    remaining[successor] -= 1
                    if not remaining[successor]:
                        enqueue(successor)
        # Direct interval and dependency audit of every sampled schedule.
        last_end = [0] * len(RESOURCE_NAMES)
        for index in np.argsort(starts, kind='stable'):
            node = nodes[index]
            assert finishes[index] - starts[index] == node['duration']
            assert all(finishes[d] <= starts[index] for d in node['deps'])
            if node['duration']:
                assert starts[index] >= last_end[node['resource']]
                last_end[node['resource']] = int(finishes[index])
        assert all(not x for x in blocked_chip)
        output = dict(makespan_ns=int(finishes.max()), nodes=len(nodes),
                      resource_work_ns=dict(zip(RESOURCE_NAMES, resource_work.tolist())),
                      intervals_and_dependencies_checked=True)
        if keep_trace:
            output['trace'] = [dict(index=i, start_ns=int(starts[i]), end_ns=int(finishes[i]),
                                    resource=RESOURCE_NAMES[n['resource']], kind=n['kind'],
                                    expert=n['expert'], phase=n['phase'], chip=n['chip'],
                                    deps=n['deps']) for i, n in enumerate(nodes)]
        return output


def build_graph(selected, gate, down, barrier, activation_ns=0, combine_ns=0):
    """Store-and-forward links; one output buffer per chip; no free barriers.

    PCIe is conservatively one shared bidirectional 8GB/s engine, each channel
    one shared bidirectional 2GB/s engine. A chip is held until its output has
    drained to the controller, then may begin the next ordinary GEMV. Controller
    vector buffers can hold the selected experts' outputs (no cached weights).
    """
    graph = Graph()
    submit = graph.add(0, 2000, kind='initial_host_submission')
    upload = graph.add(2, ceildiv(gate['columns'], 8), [submit], kind='common_input_pcie')
    input_nodes = {}
    active_chips = (gate['rows'][selected] > 0).any(axis=0)
    for channel in range(8):
        fanout = int(active_chips[channel]) + int(active_chips[channel+8])
        if fanout:
            node = graph.add(3+channel, ceildiv(gate['columns'] * fanout, 2), [upload], kind='common_input_channel')
            input_nodes[channel] = input_nodes[channel+8] = node
    gate_done = []
    for ei, expert in enumerate(selected):
        host_results = []
        for chip in range(16):
            rows = int(gate['rows'][expert, chip])
            if not rows:
                continue
            waves = int(gate['waves'][expert, chip])
            assert waves <= 192
            command = graph.add(1, 250, [submit], expert=ei, kind='matrix_command', chip=chip)
            compute = graph.add(11+chip, waves*28000, [command, input_nodes[chip]], expert=ei,
                                kind='compute', chip=chip)
            channel_out = graph.add(3+chip%8, ceildiv(rows*4, 2), [compute], expert=ei,
                                    kind='output_channel', chip=chip, release_chip=chip)
            host_out = graph.add(2, ceildiv(rows*4, 8), [channel_out], expert=ei,
                                 kind='output_pcie', chip=chip)
            host_results.append(host_out)
        gate_done.append(graph.add(27, 0, host_results, expert=ei, kind='expert_gate_ready'))
    global_gate = graph.add(27, 0, gate_done, kind='global_gate_barrier') if barrier else None
    if barrier:
        activations = [graph.add(0, activation_ns, [global_gate], phase=1, expert=ei,
                                  kind='host_activation') for ei in range(len(selected))]
        batch_down_submit = graph.add(0, 2000, activations, phase=1, kind='batched_down_host_submission')
    terminal_outputs = []
    for ei, expert in enumerate(selected):
        if barrier:
            down_submit = batch_down_submit
        else:
            activation = graph.add(0, activation_ns, [gate_done[ei]], phase=1, expert=ei,
                                   kind='host_activation')
            down_submit = graph.add(0, 2000, [activation], phase=1, expert=ei, kind='down_host_submission')
        down_upload = graph.add(2, ceildiv(down['columns'], 8), [down_submit], phase=1,
                                expert=ei, kind='expert_input_pcie')
        chip_input = {}
        for channel in range(8):
            fanout = int(down['rows'][expert, channel] > 0) + int(down['rows'][expert, channel+8] > 0)
            if fanout:
                node = graph.add(3+channel, ceildiv(down['columns']*fanout, 2), [down_upload],
                                 phase=1, expert=ei, kind='expert_input_channel')
                chip_input[channel] = chip_input[channel+8] = node
        for chip in range(16):
            rows = int(down['rows'][expert, chip])
            if not rows:
                continue
            waves = int(down['waves'][expert, chip])
            assert waves <= 192
            command = graph.add(1, 250, [down_submit], phase=1, expert=ei, kind='matrix_command', chip=chip)
            compute = graph.add(11+chip, waves*28000, [command, chip_input[chip]], phase=1,
                                expert=ei, kind='compute', chip=chip)
            channel_out = graph.add(3+chip%8, ceildiv(rows*4, 2), [compute], phase=1,
                                    expert=ei, kind='output_channel', chip=chip, release_chip=chip)
            host_out = graph.add(2, ceildiv(rows*4, 8), [channel_out], phase=1,
                                 expert=ei, kind='output_pcie', chip=chip)
            terminal_outputs.append(host_out)
    graph.add(0, combine_ns, terminal_outputs, phase=1, kind='host_combine')
    return graph


def optimize_chip_permutation(gate_work, down_work, weights, trials, seed):
    """Static down-column permutation, accepted against exact weighted work LB."""
    permutation = np.arange(16)
    joined = gate_work + down_work
    initial = current = int(joined.max(axis=1) @ weights)
    rng = random.Random(seed)
    accepted = 0
    if np.all(gate_work == gate_work[:, :1]):
        return permutation, dict(attempted=0, accepted=0, initial_peak_weighted=initial, final_peak_weighted=current)
    for _ in range(trials):
        a, b = rng.sample(range(16), 2)
        old_a, old_b = joined[:, a].copy(), joined[:, b].copy()
        joined[:, a] = gate_work[:, a] + down_work[:, permutation[b]]
        joined[:, b] = gate_work[:, b] + down_work[:, permutation[a]]
        proposed = int(joined.max(axis=1) @ weights)
        if proposed < current:
            permutation[a], permutation[b] = permutation[b], permutation[a]
            current = proposed
            accepted += 1
        else:
            joined[:, a], joined[:, b] = old_a, old_b
    return permutation, dict(attempted=trials, accepted=accepted,
                             initial_peak_weighted=initial, final_peak_weighted=current)


def analyze(name, args):
    source_path = ROOT/'results/research-2026-11-02/projection-study'/f'{name}.frozen_plan.json'
    frozen = json.loads(source_path.read_text())
    assert Path(frozen['train_path']).name == 'train_routes.npz'
    assert sha256(Path(frozen['train_path'])) == frozen['train_sha256']
    with np.load(frozen['train_path']) as cached:
        routes = cached['routes']
        assert cached['prompt_ids'].tolist() == frozen['train_ids']
        assert all(x.startswith('train_') for x in cached['prompt_ids'].tolist())
        assert cached['layers'].tolist() == frozen['layers']
    active = frozen['active']
    indices = np.linspace(0, routes.shape[0]*routes.shape[1]-1, args.samples, dtype=int)
    totals = Counter()
    resource_totals = {}
    layer_results = []
    traces = {}
    started = time.time()
    for li, layer in enumerate(frozen['layers']):
        phase_spec = frozen['phases'][2*li:2*li+2]
        gate = profiles(phase_spec[0]['chosen_rows'], phase_spec[0]['columns'])
        down = profiles(phase_spec[1]['chosen_rows'], phase_spec[1]['columns'])
        flat_routes = routes[:, :, li].reshape(-1, active)
        unique, weights = np.unique(np.sort(flat_routes, axis=1), axis=0, return_counts=True)
        gu_work = gate['waves'][unique].sum(axis=1)
        d_work = down['waves'][unique].sum(axis=1)
        separate = gu_work.max(axis=1) + d_work.max(axis=1)
        combined = (gu_work + d_work).max(axis=1)
        permutation, stats = optimize_chip_permutation(gu_work, d_work, weights, args.permutation_trials,
                                                       args.seed + layer*131)
        permuted = profiles(down['rows'][:, permutation], down['columns'])
        # Stored capacity must remain equal on every chip, not merely in total.
        for key in ('rows','pages','waves'):
            np.testing.assert_array_equal(down[key].sum(axis=0), permuted[key].sum(axis=0))
        combined_permuted = (gu_work + d_work[:, permutation]).max(axis=1)
        bounds = dict(barrier_nand_ms=float(separate @ weights / weights.sum() * .028),
                      joined_chip_work_lower_bound_ms=float(combined @ weights / weights.sum() * .028),
                      permuted_joined_chip_work_lower_bound_ms=float(combined_permuted @ weights / weights.sum() * .028),
                      pooled_stored_wave_work_lower_bound_ms=float((gu_work+d_work).mean(axis=1) @ weights / weights.sum() * .028))
        for key, value in bounds.items():
            totals[key] += value
        # Row counts reconstruct the immutable complete matrices for all experts.
        equal_gate = profiles(phase_spec[0]['baseline_rows'], gate['columns'])
        equal_down = profiles(phase_spec[1]['baseline_rows'], down['columns'])
        layer_costs = Counter()
        for selected in flat_routes[indices]:
            for placement, g, d in (('equal_g16', equal_gate, equal_down),
                                    ('projection', gate, down), ('permuted_projection', gate, permuted)):
                for barrier in (True, False):
                    schedule = build_graph(selected, g, d, barrier, args.activation_ns, args.combine_ns)
                    key = placement + ('__barrier' if barrier else '__ready')
                    keep_trace = li == 0 and np.array_equal(selected, flat_routes[indices[0]]) and key not in traces
                    result = schedule.run(down_first=not args.gate_first, keep_trace=keep_trace)
                    layer_costs[key] += result['makespan_ns'] / len(indices) / 1e6
                    if key not in resource_totals:
                        resource_totals[key] = Counter()
                    for resource, work in result['resource_work_ns'].items():
                        resource_totals[key][resource] += work / len(indices)
                    if keep_trace:
                        traces[key] = dict(layer=layer, selected=selected.tolist(), **result)
        totals.update(layer_costs)
        layer_results.append(dict(layer=layer, all_train_unique_routes=len(unique), bounds=bounds,
                                  sample_schedule_means_ms=dict(layer_costs),
                                  down_chip_permutation=permutation.tolist(), permutation_optimization=stats,
                                  gate_rows=gate['rows'].tolist(), down_rows=permuted['rows'].tolist()))
        if (li+1) % 8 == 0:
            print(f'{name}: {li+1}/{len(frozen["layers"])} layers; {time.time()-started:.1f}s', flush=True)
    result = dict(model=name, source_projection_plan=str(source_path), source_projection_sha256=sha256(source_path),
                  train_cache_path=frozen['train_path'], train_cache_sha256=frozen['train_sha256'],
                  train_ids=frozen['train_ids'], train_shape=list(routes.shape),
                  sample_flat_train_indices=indices.tolist(), all_train_route_bounds=True,
                  sample_schedule_means_ms=dict(totals), schedule_resource_work_ns={k:dict(v) for k,v in resource_totals.items()},
                  layer_results=layer_results, elapsed_seconds=time.time()-started)
    for name_key, trace in traces.items():
        (args.output/f'{name}.{name_key}.trace.json').write_text(json.dumps(trace, indent=2)+'\n')
    print(json.dumps(dict(model=name, totals=dict(totals), seconds=time.time()-started)), flush=True)
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models', nargs='+', choices=MODELS, default=list(MODELS))
    parser.add_argument('--samples', type=int, default=32)
    parser.add_argument('--permutation-trials', type=int, default=256)
    parser.add_argument('--activation-ns', type=int, default=0)
    parser.add_argument('--combine-ns', type=int, default=0)
    parser.add_argument('--gate-first', action='store_true')
    parser.add_argument('--seed', type=int, default=20261102)
    parser.add_argument('--output', type=Path, default=DEST)
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    results=[]
    for name in args.models:
        results.append(analyze(name,args))
        document=dict(study='Single-request dependency-aware no-CR train-only study',
                      source_sha256=sha256(Path(__file__)), activation_ns=args.activation_ns,
                      combine_ns=args.combine_ns, priority='gate-first' if args.gate_first else 'down-first',
                      samples_per_layer=args.samples, permutation_trials=args.permutation_trials,
                      hardware=dict(chips=16, planes_per_chip=4, ordinary_wave_ns=28000, page_bytes=16384,
                                    pcie_bytes_per_ns=8, channel_bytes_per_ns=2, channels=8,
                                    matrix_block_command_ns=250, initial_and_each_down_host_submission_ns=2000,
                                    global_barrier_down_submission='one batch host submission after all host activations',
                                    output_element_bytes=4, input_element_bytes=1,
                                    controller_buffers='selected activation/output vectors only; no weight cache',
                                    chip_output_buffer='one shard output; chip cannot begin next GEMV before channel drain'),
                      formulas=dict(barrier_nand='28000*(max_chip sum_selected gate_waves + max_chip sum_selected down_waves)',
                                    joined_chip_work_lower_bound='28000*max_chip sum_selected(gate_waves+down_waves); ignores dependencies and transfers',
                                    pooled_stored_wave_lower_bound='28000/16*sum_chip sum_selected(gate_waves+down_waves)',
                                    schedule='All commands, host submissions, activation, chip services, channel transfers and PCIe transfers are dependency/resource events; data transfers store and forward.'),
                      limitations=['Feasible event schedules are not proven globally optimal schedules.',
                                   'Only train routes are opened. All-train bounds are exact for these immutable layouts; schedule means use deterministic train samples.',
                                   'Host activation/reassembly/combine compute are exposed parameters with zero default, not measured free operations.',
                                   'Compared with old barrier estimates, event model changes command/transfer overlap. Global-barrier down uses one 2us batch host submission; ready mode pays 2us for each expert down submission.',
                                   'Same scheduler, buffers, priority and host costs apply to equal-g16 and projection placements.',
                                   'New placement candidate only permutes fixed down chip ownership; per-chip stored rows/pages/waves and full weight coverage remain conserved.',
                                   'Shared experts, dense layers, attention, cross-token/request overlap and energy are excluded.'],
                      test_routes_opened=False, models=results)
        (args.output/'summary.json').write_text(json.dumps(document, indent=2)+'\n')


if __name__=='__main__':
    main()
