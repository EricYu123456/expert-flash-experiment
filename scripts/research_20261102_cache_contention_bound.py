#!/usr/bin/env python3
"""Train-only cache-residual placement screen and row-safe preemption bounds.

This is a structural headroom study, not a cache policy or new placement method.
Host execution and cache setup are excluded equally from flash residual costs;
those costs must be added before any end-to-end performance claim.
"""
from pathlib import Path
import hashlib
import json
import math

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research-2026-11-02/round2/cache-contention-bound'
RESEARCH = ROOT / 'results/research-2026-11-02'


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def costs(selected, miss, rows, columns, common):
    rows = np.asarray(rows, dtype=np.int64)
    pages = (rows * columns + 16383) // 16384
    waves = (pages + 3) // 4
    runs = (waves + 191) // 192
    chips = (rows[selected] * miss[:, :, None]).sum(axis=1)
    wave_load = (waves[selected] * miss[:, :, None]).sum(axis=1)
    page_load = (pages[selected] * miss[:, :, None]).sum(axis=1)
    run_load = (runs[selected] * miss[:, :, None]).sum(axis=1)
    present = (rows[selected] > 0) & miss[:, :, None]
    inputs = (present.any(axis=1) if common else present.sum(axis=1)) * columns
    outputs = chips * 4
    M = miss.sum(axis=1)
    host_input = columns * ((M > 0) if common else M)
    nand = wave_load.max(axis=1) * 28000
    flash = np.maximum(nand, page_load.max(axis=1) * 2560)
    control = run_load.sum(axis=1) * 250
    host = (M > 0) * 2000
    iv = np.maximum((host_input + 7) // 8, ((inputs[:, :8] + inputs[:, 8:]).max(axis=1) + 1) // 2)
    ov = np.maximum((outputs.sum(axis=1) + 7) // 8, ((outputs[:, :8] + outputs[:, 8:]).max(axis=1) + 1) // 2)
    return dict(total=flash + control + host + iv + ov, nand=nand,
                commands=control, vector_io=iv + ov)


def relaxation(M, R, C, common):
    # Every selected uncached expert still requires its entire unchanged matrix.
    # Relax fixed owners, whole-row job grain, and chip-fanout into pooled work.
    per_expert_waves = (R * C + 65535) // 65536
    nand = ((M * per_expert_waves + 15) // 16) * 28000
    # >=1 native GEMV per expert, >=1 host phase command if nonempty, plus
    # unavoidable host input/output transfer. ONFI fanout and host cache work
    # are relaxed, so this is an optimistic lower bound, not an executable plan.
    bound = nand + M * 250 + (M > 0) * 2000
    host_input = C * ((M > 0) if common else M)
    bound += (host_input + 7) // 8 + (M * R * 4 + 7) // 8
    return nand, bound


def chunk_profile(R, C, qrows):
    pieces = [min(qrows, R - first) for first in range(0, R, qrows)]
    pages = [math.ceil(r * C / 16384) for r in pieces]
    waves = [math.ceil(p / 4) for p in pages]
    # One matrix stream on each of16 chips; no input retained across commands.
    # Host phase command is shared; every native run reloads its complete input.
    return dict(rows_per_chip=R, columns=C, rows_per_native_run=qrows,
        runs_per_chip=len(pieces), pages_per_chip=sum(pages), waves_per_chip=sum(waves),
        max_nonpreemptive_nand_ns=max(waves) * 28000,
        phase_nand_ns=sum(waves) * 28000,
        control_ns=len(pieces) * 16 * 250,
        input_channel_ns=len(pieces) * C,  # 2chips/channel xC /2GB/s
        input_pcie_ns_once_per_host_phase=(C + 7) // 8,
        output_pcie_ns=(R * 16 * 4 + 7) // 8,
        max_nonpreemptive_run_service_ns=max(waves) * 28000 + 16 * 250 + C +
            (min(qrows, R) * 16 * 4 + 7) // 8,
        caveat='All16 chips stream identical geometry; no extra host command per controller-internal run. Host output concat only.')


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    inputs = {}
    cases = []
    plans = {}
    for model in ('granite', 'olmoe', 'deepseek', 'qwen3', 'mixtral', 'granite4tiny'):
        path = RESEARCH / ('round2/geometry-study/granite4tiny.frozen.json' if model == 'granite4tiny'
                           else f'projection-study/{model}.frozen_plan.json')
        inputs[str(path)] = sha(path)
        plan = json.loads(path.read_text())
        plans[model] = plan
        route_path = Path(plan.get('train_path', plan.get('training_routes')))
        inputs[str(route_path)] = sha(route_path)
        with np.load(route_path) as z:
            routes = z['routes'][:32, :64].copy()
            layers = list(z['layers'])
            ids = z['prompt_ids'][:32].tolist()
        assert len(routes) == 32
        E = plan['experts']
        stronger = {}
        strong_path = RESEARCH / ('round2/unbalanced-group/granite.full.strong.frozen.json' if model == 'granite'
            else f'round2/unbalanced-group/{model}.screen.strong.frozen.json')
        if strong_path.exists():
            inputs[str(strong_path)] = sha(strong_path)
            strong = json.loads(strong_path.read_text())
            stronger = {(p['layer'], p['name']): p for p in strong['phases']}
        for fraction in (0., .25, .5):
            totals = {k: np.zeros(16 * 64, np.int64) for k in
                ('chosen', 'g16', 'menu', 'nand_menu', 'nand_bound', 'total_bound')}
            phase_records = []
            cached_bytes = 0
            all_misses = []
            for phase in plan['phases']:
                li = layers.index(phase['layer'])
                fitting = routes[:16, :, li].reshape(-1)
                frequency = np.bincount(fitting, minlength=E)
                order = np.lexsort((np.arange(E), -frequency))
                pinned = np.zeros(E, bool)
                pinned[order[:int(E * fraction)]] = True
                selected = routes[16:, :, li].reshape(-1, plan['active'])
                miss = ~pinned[selected]
                M = miss.sum(axis=1)
                all_misses.append(M)
                R, C = phase['total_rows'], phase['columns']
                cached_bytes += int(pinned.sum()) * R * C
                chosen_rows = phase.get('chosen_rows', phase.get('rows'))
                menu = {'frozen_chosen': chosen_rows}
                for field in ('baseline_rows', 'legacy_rows', 'row_only_rows', 'group_rows'):
                    if field in phase:
                        menu[field] = phase[field]
                if model == 'granite4tiny':
                    for candidate in phase['candidates']:
                        if 'rows' in candidate:
                            menu[candidate['label']] = candidate['rows']
                if (phase['layer'], phase['name']) in stronger:
                    menu['strong_same_g'] = stronger[(phase['layer'], phase['name'])]['chosen_rows']
                full = np.tile([R // 16 + (c < R % 16) for c in range(16)], (E, 1))
                menu['g16'] = full
                scores = {name: costs(selected, miss, rows, C, phase['common']) for name, rows in menu.items()}
                best = min(scores, key=lambda name: scores[name]['total'].sum())
                nand_lb, total_lb = relaxation(M, R, C, phase['common'])
                assert np.all(scores[best]['nand'] >= nand_lb)
                assert np.all(scores[best]['total'] >= total_lb)
                totals['chosen'] += scores['frozen_chosen']['total']
                totals['g16'] += scores['g16']['total']
                totals['menu'] += scores[best]['total']
                totals['nand_menu'] += scores[best]['nand']
                totals['nand_bound'] += nand_lb
                totals['total_bound'] += total_lb
                phase_records.append(dict(layer=phase['layer'], phase=phase['name'],
                    cached_experts=np.flatnonzero(pinned).tolist(), best_existing_menu=best,
                    mean_misses=float(M.mean()), zero_miss_fraction=float((M == 0).mean()),
                    mean_menu_ns=float(scores[best]['total'].mean()), mean_pooled_bound_ns=float(total_lb.mean()),
                    menu_nand_gap_ns=float((scores[best]['nand'] - nand_lb).mean())))
            misses = np.concatenate(all_misses)
            means = {k: float(v.mean()) for k, v in totals.items()}
            cases.append(dict(model=model, pinned_fraction=fraction, cached_I8_payload_bytes=cached_bytes,
                calibration_prompt_ids=ids[:16], screen_prompt_ids=ids[16:], train_steps=64,
                mean_misses_per_phase=float(misses.mean()), all_cached_phase_fraction=float((misses == 0).mean()),
                means_ns=means,
                old_chosen_to_existing_menu_gain_pct=100 * (1 - means['menu'] / means['chosen']) if means['chosen'] else 0,
                remaining_service_relaxation_gap_pct=100 * (1 - means['total_bound'] / means['menu']) if means['menu'] else 0,
                remaining_nand_gap_pct=100 * (1 - means['nand_bound'] / means['nand_menu']) if means['nand_menu'] else 0,
                phases=phase_records))

    # Mixtral fixed g16 native geometry; no new operator or input context.
    quantum = []
    for name, R, C in (('gate_up', 1792, 4096), ('down', 256, 14336)):
        alignment_rows = 65536 // math.gcd(C, 65536)
        for qrows in sorted({R, alignment_rows, 4, 8, 16, 32, 64, 128}):
            if qrows <= R:
                quantum.append(dict(phase=name, dense_wave_aligned_rows=alignment_rows,
                                    **chunk_profile(R, C, qrows)))
    small = next(c for c in cases if c['model'] == 'granite' and c['pinned_fraction'] == 0.)['means_ns']['menu']
    profile_map = {(p['phase'], p['rows_per_native_run']): p for p in quantum}
    q8 = profile_map['down', 8]
    q4 = profile_map['down', 4]
    q32 = profile_map['down', 32]
    preemption = dict(dense_no_work_inflation=q32, strong_small_page_aligned_chunk=q8,
        padded_one_wave_chunk=q4,
        reservation_scheduler='Once a high-priority token starts, reserve required resources across its GU/host/down dependencies. Lower-priority work can block only its initial arrival, not each phase. This can waste actual host gaps; those are not calibrated here.',
        max_initial_blocking_improvement_over_q8_ns=q8['max_nonpreemptive_run_service_ns'] - q4['max_nonpreemptive_run_service_ns'],
        max_relative_to_Granite_flash_residual_service_pct=100 * (q8['max_nonpreemptive_run_service_ns'] - q4['max_nonpreemptive_run_service_ns']) / small,
        note='q8 and q4 have the same64 NAND waves/256rows; q4 doubles native commands and input reload versusq8. q32 uses56waves but has a larger quantum.32-row/7-wave is minimum wave-aligned zero-work-inflation, not the minimum legal software split.',
        qualification='Offset units and partial-plane starting offsets are not fully specified by paper API. Separate row chunks stored at wave-aligned starts are a conservative existing-dense-GEMV construction. q8 needs7pages/2waves, q4 needs4pages/1wave. Existing controller run segmentation is a software capability assumption, not a new PE/context.')
    report = dict(protocol='First16 original-train prompts calibrate static per-layer LFU pin sets; next16 original-train prompts x64steps screen. No test or inference. No new layout fitting.',
        scope='Flash-residual demand only; fixed complete expert weights in host cache; unchanged routing and whole output rows. INT8 architectural costs, not matched numerical AiF execution.',
        cases=cases, mixtral_preemption_profiles=quantum, preemption_bound=preemption,
        caveats=['Cache fraction is a research scenario, not a user memory budget.',
            'Static pinning requires an initial cache-fill cost and host copies, reported as useful I8 bytes only; scales/padding/host execution are additional.',
            'Host cache GEMV/combine costs are zeroed equally as an optimistic residual-flash screen; reported percentages are not end-to-end speedups.',
            'Existing menu winner is selected on the same16 screening training prompts, favoring a strong cache-aware control; this is an optimistic control, not heldout validation.',
            'Pooled relaxation permits route-adaptive fragment placement and ignores fanout/job-grain; a positive gap does not establish an executable method.',
            'Cache changes residual queries; optimizing those with a hypergraph is not a newly identified mechanism.',
            'Token-reservation scheduler bounds only initial lower-priority blocking by one native-run quantum; host gaps must be measured before claiming idle-capacity benefits.',
            'No suspend/abort/preempt-in-flight GEMV is assumed; switching only between complete native dense-GEMV runs.'],
        formulas=dict(ordinary='maxchip(sumwaves)*28us, all ECC/page work retained',
            pooled='ceil(missing_experts*ceil(R*C/64KiB)/16)*28us',
            full_cost='flash +250ns/native_run +2us/nonempty_host_phase +max(PCIe,paired-ONFI)inputs +max(PCIe,paired-ONFI)outputs',
            preemption='minimum zero-padding row boundary at65536/gcd(C,65536) rows; short matrix chunks can reduce atomic quantum by paying extra page/wave padding, full input reload and commands'),
        source_sha256=sha(Path(__file__)), input_sha256=inputs)
    (OUT / 'screen.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps([{k: v for k, v in c.items() if k not in ('phases', 'calibration_prompt_ids', 'screen_prompt_ids')} for c in cases], indent=2))


if __name__ == '__main__':
    main()
