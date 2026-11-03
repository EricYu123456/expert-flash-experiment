#!/usr/bin/env python3
"""Train-only shared-GU NAND bound and equal-byte routed-copy controls.

This is an all-ready GU-stage calculation, not an end-to-end DAG simulator.
No existing result, layout, source weight, or route is changed.
"""
from datetime import datetime, timezone
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2'
SOURCE = BASE / 'geometry-study/granite4tiny.frozen.json'
OUT = BASE / 'shared-oracle'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stat(x):
    return dict(mean_ns=float(x.mean()), p99_ns=float(np.percentile(x, 99)),
                min_ns=int(x.min()), max_ns=int(x.max()))


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source_hash = sha(SOURCE)
    d = json.loads(SOURCE.read_text())
    route_path = Path(d['training_routes'])
    route_hash = sha(route_path)
    assert route_hash == d['training_routes_sha256']
    with np.load(route_path) as f:
        routes = f['routes'][:32, :64].copy()
        layers = f['layers'].tolist()
        prompt_ids = f['prompt_ids'][:32].tolist()
    assert routes.shape == (32, 64, 40, 6)
    assert all(x.startswith('train_') for x in prompt_ids)
    records = []
    totals = {}
    greedy_state = []
    pairs = list(itertools.combinations(range(64), 2))
    for li, layer in enumerate(layers):
        phase = next(p for p in d['phases'] if p['layer'] == layer and p['name'] == 'gate_up')
        rows = np.asarray(phase['rows'], dtype=np.int64)
        assert rows.shape == (64, 16)
        assert np.all(rows.sum(axis=1) == 1024)
        home_a = rows[:, 0] > 0
        expected = np.where(home_a[:, None] == (np.arange(16)[None, :] < 8), 128, 0)
        assert np.array_equal(rows, expected)
        selected = routes[:, :, li].reshape(-1, 6)
        assert np.all(np.diff(np.sort(selected, axis=1), axis=1) > 0)
        incidence = np.zeros((2048, 64), dtype=np.int8)
        incidence[np.arange(2048)[:, None], selected] = 1
        na = home_a[selected].sum(axis=1)
        nb = 6 - na
        majority_a = na > nb
        m = np.maximum(na, nb)
        demand = m - 3
        useful = incidence * (home_a[None, :] == majority_a[:, None]) * (demand[:, None] > 0)

        def cost(copies):
            a = incidence[:, copies][:, home_a[copies]].sum(axis=1) if len(copies) else np.zeros(2048, dtype=int)
            b = incidence[:, copies][:, ~home_a[copies]].sum(axis=1) if len(copies) else np.zeros(2048, dtype=int)
            # Every routed expert still executes exactly eight 3-wave jobs.
            return 84000 * (1 + np.maximum(3, np.maximum(na-a, nb-b)))

        base = 84000 * (m + 1)
        shared_full = 84000 * np.maximum(4, m)
        # Eight shared tile copies can move just one originally-stored half.
        one_sided = [84000 * np.where(na > nb, np.maximum(4, m), m+1),
                     84000 * np.where(nb > na, np.maximum(4, m), m+1)]
        side = min(range(2), key=lambda i: (int(one_sided[i].sum()), i))
        counts = incidence.sum(axis=0)
        hot = sorted(range(64), key=lambda e: (-int(counts[e]), e))
        single = min(range(64), key=lambda e: (int(cost([e]).sum()), e))
        best_pair = min(pairs, key=lambda es: (int(cost(list(es)).sum()), es))

        def hybrid(e, direction):
            # direction0 stores original A shared tiles also at B; direction1
            # stores original B tiles also at A. Original opposite tiles remain.
            flipped_na = na + incidence[:,e] * (-1 if home_a[e] else 1)
            choices=[]
            for do_flip, a in enumerate((na, flipped_na)):
                for move_shared in (0,1):
                    sa=1 if not move_shared else (0 if direction==0 else 2)
                    peak=np.maximum(a+sa,6-a+2-sa)
                    choices.append((peak,do_flip,sa))
            values=np.stack([x[0] for x in choices])
            ix=values.argmin(axis=0)
            best=values.min(axis=0)
            return best*84000, np.asarray([choices[j][1] for j in ix],dtype=np.uint8), np.asarray([choices[j][2] for j in ix],dtype=np.uint8)

        hd, he = min(itertools.product(range(2), range(64)),
                     key=lambda de: (int(hybrid(de[1],de[0])[0].sum()),de))
        hybrid_cost, hybrid_flip, hybrid_shared_a = hybrid(he,hd)
        series = dict(onecopy=base, paired_shared=shared_full,
                      shared_half_budget=one_sided[side], hot1=cost(hot[:1]),
                      optimal1=cost([single]), hot2=cost(hot[:2]), optimal2=cost(list(best_pair)),
                      optimal_hybrid_halfshared1routed=hybrid_cost,
                      adaptive_routed_copy_placement_16tiles_oracle=84000*np.maximum(4, m-1))
        for k, v in series.items():
            totals.setdefault(k, []).append(v)
        # Exhaustively verify the integer two-half assignment formula, and
        # preserve a concrete assignment for the selected pair on every route.
        assignment = np.zeros((2048, 6), dtype=np.uint8)
        for i, route in enumerate(selected):
            variable = [j for j, e in enumerate(route) if e in best_pair]
            optimal = int(series['optimal2'][i] // 84000 - 1)
            found = False
            for flips in itertools.product((0, 1), repeat=len(variable)):
                owner_a = home_a[route].copy()
                for j, flip in zip(variable, flips):
                    owner_a[j] ^= bool(flip)
                if max(int(owner_a.sum()), 6-int(owner_a.sum())) == optimal:
                    assignment[i] = owner_a
                    found = True
                    break
            assert found
        np.savez_compressed(OUT / f'layer{layer}.npz', n_a=na, **series,
                            optimal2_owner_a=assignment,hybrid_flip_routed_expert=hybrid_flip,
                            hybrid_shared_tiles_at_A_per_pair=hybrid_shared_a)
        records.append(dict(layer=layer, n_a_histogram=np.bincount(na, minlength=7).tolist(),
            hot1=hot[:1], hot2=hot[:2], optimal1=[single], optimal2=list(best_pair),
            optimal_hybrid=dict(routed_expert=he,copy_original_shared_half='A' if hd==0 else 'B',
                shared_extra_tiles=8,routed_extra_tiles=8,extra_useful_bytes=3145728,
                extra_pages=192,extra_stored_waves=48,
                policy='Joint per-route minimum over copied routedexpert original/oppositehalf and shared128+128 or legal256+0 rows; original shared tiles reused.'),
            shared_half_budget_copies_original_half='A' if side == 0 else 'B',
            stats={k: stat(v) for k, v in series.items()},
            route_assignment_file=f'layer{layer}.npz',
            optimal2_copy_rows=[dict(expert=int(e), source_half='A' if home_a[e] else 'B',
                destination_half='B' if home_a[e] else 'A', rows=1024, columns=1536,
                tiles=8, rows_per_tile=128, extra_useful_bytes=1572864,
                extra_pages=96, extra_waves_stored=24) for e in best_pair]))
        greedy_state.append(dict(useful=useful, demand=demand,
            cover=np.zeros(2048, dtype=np.int8), selected=[]))

    # Same *model-wide* 120 MiB, without artificially forcing two copies/layer.
    # This is a feasible greedy baseline, not a globally optimal budget solver.
    greedy_steps = []
    for step in range(80):
        options = []
        for li, state in enumerate(greedy_state):
            marginal = state['useful'][state['cover'] < state['demand']].sum(axis=0)
            marginal[state['selected']] = -1
            e = int(marginal.argmax())
            options.append((int(marginal[e]), -li, -e))
        benefit, nli, ne = max(options)
        li, e = -nli, -ne
        state = greedy_state[li]
        state['selected'].append(e)
        state['cover'] += state['useful'][:, e]
        greedy_steps.append(dict(step=step+1, layer=layers[li], expert=e,
                                 train_saved_3wave_units=benefit))
    global_costs = []
    for rec, state in zip(records, greedy_state):
        saving = np.minimum(state['cover'], state['demand'])
        v = totals['onecopy'][len(global_costs)] - saving*84000
        global_costs.append(v)
        rec['global_budget_greedy_copies'] = state['selected']
        rec['global_budget_greedy_stats'] = stat(v)
    totals['global_budget_greedy80'] = global_costs
    aggregate = {k: stat(np.stack(v).sum(axis=0)) for k, v in totals.items()}
    b = aggregate['onecopy']['mean_ns']
    for k, val in aggregate.items():
        val['mean_reduction_vs_onecopy_pct'] = 100*(1-val['mean_ns']/b)
    out = dict(status='complete train-only GU-stage NAND bounds; no full-DAG speedup claim',
        created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
        frozen_source=str(SOURCE), frozen_source_sha256=source_hash,
        training_routes=str(route_path), training_routes_sha256=route_hash,
        train_slice='32 prompts x 64 steps x all40 layers', training_prompt_ids=prompt_ids,
        formulas=dict(onecopy='84000*(1+max(nA,nB))',
            paired_shared='84000*max(4,nA,nB)',
            fixed_routed_copy_set='84000*(1+max(3,nA-selectedCopiedA,nB-selectedCopiedB))',
            hybrid='Enumerate shared sourcehalf A/B and routedexpert64 offline. Online enumerate copiedexpert original/oppositehalf and sharedfixed128+128/legal256+0, then minimize exactGU peak.',
            independent_routed_oracle='84000*max(4,max(nA,nB)-1) with per-route16tile placement',
            why_84000='Each128 whole-output-row tile has1536 I8 columns:12 pages,3 ordinary28us waves.',
            budget='Pairedshared extra16tiles=3MiB/layer equals two routedexpert GU copies,8tiles each.'),
        capacity=dict(extra_shared_full_model_bytes=125829120, extra_shared_pages=7680,
            extra_routed_expert_bytes=1572864, global_budget_expert_copies=80,
            full_model_routed_ffn_bytes=6039797760, full_model_shared_ffn_bytes=188743680,
            extra_pct_full_routed_plus_shared_ffn=100*125829120/(6039797760+188743680),
            extra_shared_per_chip_full_model_bytes=7864320,
            reservations='Payloads are whole128row tiles aligned to full64KiB waves; page and useful-byte budgets match. Block reservations/allocator overhead require physical materialization, not inferred from payload.'),
        native_GU_work=dict(onecopy_jobs=64, paired_shared_jobs_balanced=64,
            paired_shared_jobs_unbalanced=56, routed_copy_jobs=64,
            onecopy_ONFI_input_bytes=98304, paired_shared_ONFI_input_bytes_unbalanced=86016,
            all_policies_output_bytes=32768, PCIe_common_input_bytes=1536,
            command_ns=250, host_initial_submission_ns=2000,
            note='These mandatory command/input/output costs are exposed, not included in NAND-only bounds. DAG/host activation/private down/channel contention must be replayed before any latency claim.'),
        totals=aggregate, layers=records, global_greedy_steps=greedy_steps,
        limitations=['Train selection and train scoring use the same routes; no validation/generalization claim.',
            'optimal2 is exact among two complete routed-GU opposite-half expert copies, not arbitrary partial tiles or all replica layouts.',
            'optimal_hybrid_halfshared1routed is exact among128 halfshared-plus-onewhole-routed choices, not all possible16tile shared/routed budgets.',
            'global_budget_greedy80 is feasible with same total useful/page budget but not a global optimum; per-chip reservation capacity must be checked.',
            'Stage bounds require all GU jobs ready together; joined GU/down DAG may change the ranking. No summing stage oracles as measured end-to-end latency.',
            'Same existing abstract raw-I8/A8/output4B timing contract. The original GGUF numeric model and native AiF output/scales are not established by this script.',
            'Shared replication and dynamic expert dispatch have direct prior art; these controls do not establish novelty.'],
        primary_prior=[dict(title='DeepSeek-V3 technical report,3.4.2', url='https://arxiv.org/html/2412.19437v2',
            relevance='Decoding explicitly treats always-selected shared expert as heavy-loaded routed expert and deploys redundant/shared experts.'),
            dict(title='Official DeepSeek EPLB', url='https://github.com/deepseek-ai/EPLB',
                 relevance='Duplicates heavy-loaded experts and packs replicas to balance execution load.'),
            dict(title='Official Megatron shared expert implementation',url='https://github.com/NVIDIA/Megatron-LM/blob/main/megatron/core/transformer/moe/shared_experts.py',
                 relevance='Explicit shared-expert/dispatcher overlap state machine; overlap alone is prior art.')])
    assert sha(SOURCE) == source_hash and sha(route_path) == route_hash
    (OUT/'screen.json').write_text(json.dumps(out, indent=2)+'\n')
    manifest = {str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.glob('*')) if p.is_file() and p.name!='completion.json'}
    manifest[str(Path(__file__).relative_to(ROOT))] = sha(__file__)
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete', artifacts_sha256=manifest,
        inputs_unchanged=True), indent=2)+'\n')
    print(json.dumps(aggregate, indent=2))


if __name__ == '__main__':
    main()
