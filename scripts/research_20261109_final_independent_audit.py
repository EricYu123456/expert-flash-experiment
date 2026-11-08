#!/usr/bin/env python3
"""Independent, streaming audit of the frozen five-arm 1000-article experiment.

The completion barrier is checked before opening corpus records or any TEST
summary, position, or routing file. This program deliberately does not import
the experiment's quality-analysis helpers or rerun inference/model selection.
"""
from collections import Counter
from datetime import datetime, timezone
import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np

DATA = Path('/home/nfs_client/ericyu/expert_flash_experiment/research-2026-11-09-shared-features')
RECEIPT_SHA256 = '0714103fab74c6a488c72f0313b967230ea9ec4644611d17fa29b679ad576e7d'
H, I, IS, K, L, VOCAB = 2048, 1408, 2816, 6, 26, 102400
NATIVE_MAC = 3 * H * I
EXPECTED_ARMS = {
    'native', 'prototype25_dense_k12', 'prototype25_rank256_k12',
    'raw_x_select1408_eq_rank256_k12', 'selected_alpha2_mean_k12',
}
EXPECTED_EXPERTS = {1, 4, 5, 17, 22, 28, 38, 50, 51, 58, 60, 62}
PHASES = ('NLL_prefix63', 'last_prefix_for_first_generated',
          'decode_causal', 'decode_last_unused')
SUMMARY_COUNTERS = (
    'replaced_expert_token_events', 'native_shared_z_floats_bitchecked',
    'prototype_GU_row_pairs_computed', 'prototype_owner_selected_phi_floats_bitchecked',
    'prototype_cross_replaced_events', 'basis_A8_unique_rows', 'private_latent_A8_rows',
    'factorized_latent_affine_values', 'factorized_V_coefficient_MAC',
    'factorized_U_coefficient_MAC',
)


def require(condition, message):
    if not condition:
        raise AssertionError(message)


def read(path):
    return json.loads(Path(path).read_text())


def json_lines(path):
    with Path(path).open() as stream:
        for number, line in enumerate(stream, 1):
            require(bool(line.strip()), f'{path}:{number}: empty record')
            yield json.loads(line)


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def check_int(value, lo, hi, description):
    require(type(value) is int and lo <= value <= hi, description)


def completion_barrier(plan):
    """Only plan and root execution metadata may be opened before this returns."""
    require(plan['stage'] == 'test' and plan['prompts'] == 1000, 'Wrong stage/sample size')
    require(plan['max_tokens'] == 64 and plan['steps'] == 16, 'Wrong prefix/generation length')
    require(plan['shards_per_arm'] == 16 and plan['threads_per_worker'] == 4, 'Wrong scheduling plan')
    names = [arm['name'] for arm in plan['arms']]
    require(len(names) == 5 and set(names) == EXPECTED_ARMS, 'Wrong five-arm design')
    execution_path = Path(plan['output']) / 'execution.json'
    require(execution_path.is_file(), 'Final execution is not complete; TEST results have not been opened')
    execution = read(execution_path)
    require(execution['status'] == 'complete' and execution.get('all_input_hashes_rechecked') is True,
            'Final execution/posthashguard incomplete; TEST results have not been opened')
    batches = execution['batches']
    require(len(batches) == 80, 'Expected exactly 80 completed shards')
    expected = {(name, shard) for name in names for shard in range(16)}
    actual = [(batch['arm'], batch['shard']) for batch in batches]
    require(len(set(actual)) == 80 and set(actual) == expected, 'Missing/duplicate final shards')
    for batch in batches:
        done = batch.get('completion', {})
        require(batch['exit_code'] == 0 and done.get('status') == 'complete', 'Nonzero/incomplete shard')
        require(done['gpu_layers'] == 0 and done['max_tokens'] == 64 and done['decode_steps'] == 16,
                'Unexpected execution backend/shape')
        require(done['jobs'] == (63 if batch['shard'] < 8 else 62), 'Unexpected shard job count')
    return execution


class Integrity:
    def __init__(self):
        self.checked = {}

    def check(self, path, expected):
        path = str(Path(path).resolve())
        if path not in self.checked:
            self.checked[path] = sha(path)
        require(self.checked[path] == expected, f'Hash changed: {path}')


def frozen_integrity(plan_path, plan, integrity):
    receipt_path = Path(plan['selection_receipt'])
    integrity.check(receipt_path, RECEIPT_SHA256)
    receipt = read(receipt_path)
    require(receipt['status'] == 'frozen_before_new_TEST', 'Selection was not frozen')
    require(receipt['TEST_results_used_for_selection'] is False and receipt['TEST_inference_not_started'] is True,
            'Invalid pre-TEST selection declaration')
    require(len(receipt['files_sha256']) == 343, 'Unexpected frozen-input count')
    require(len(receipt['source_snapshots_sha256']) == 11, 'Unexpected source-snapshot count')
    for mapping in ('files_sha256', 'source_snapshots_sha256'):
        for path, expected in receipt[mapping].items():
            integrity.check(path, expected)
    frozen = receipt['files_sha256']
    for path in (str(plan_path.resolve()), str(Path(plan['binary']).resolve()), *plan['source_files']):
        require(path in frozen, f'Execution source/plan not covered by receipt: {path}')
    # In addition to independent source hashes, verify that each named snapshot
    # equals its frozen executed-source counterpart where that counterpart exists.
    for snapshot, expected in receipt['source_snapshots_sha256'].items():
        originals = [path for path in frozen if Path(path).name == Path(snapshot).name
                     and str(Path(path).parent) != str(Path(snapshot).parent)]
        require(len(originals) == 1 and frozen[originals[0]] == expected,
                f'Source snapshot/original mismatch: {snapshot}')
    protocol = read(Path(plan['output']) / 'protocol.frozen.json')
    require(protocol['plan'] == plan, 'Executed protocol does not match final plan')
    require(protocol['CUDA_VISIBLE_DEVICES'] == '' and protocol['gpu_layers'] == 0,
            'GPU was enabled in executed protocol')
    require(protocol['threads_per_worker'] == 4, 'Unexpected executed thread count')
    for path, expected in frozen.items():
        require(protocol['inputs_sha256'].get(path) == expected, f'Input absent from posthashguard: {path}')
    for path, expected in protocol['inputs_sha256'].items():
        integrity.check(path, expected)
    require(protocol['model_identity'] == receipt['model_identity'], 'Executed model identity drift')
    integrity.check(receipt['model_identity']['path'], receipt['model_identity']['sha256'])
    quality = receipt['quality']
    require(quality['primary_arm'] == plan['primary_arm'] == 'prototype25_rank256_k12', 'Primary contrast drift')
    require(quality['bootstrap_replicates'] == 10000 and quality['bootstrap_seed'] == 20261109,
            'Bootstrap specification drift')
    require(quality['bootstrap_unit'] == 'article' and quality['primary_margin_relative_PPL'] == .02,
            'Inference specification drift')
    return receipt, protocol


def read_registry(path, frozen):
    """Parse transform ownership and factor dimensions without qa.registry/load_arm."""
    if path == '-':
        return {}, {}, set()
    files = {str(Path(path).resolve())}
    heads, bases = {}, {}
    for line in Path(path).read_text().splitlines():
        if not line or line.startswith('#'):
            continue
        cols = line.split('\t')
        require(len(cols) == 6, f'Invalid registry row: {line}')
        layer, expert, basis, width, transform, head = cols
        layer, expert, width = int(layer), int(expert), int(width)
        require(1 <= layer <= L and 0 <= expert < 64 and 0 < width <= IS, 'Invalid head dimensions/ID')
        require((layer, expert) not in heads, 'Duplicate head identity')
        src = np.fromfile(transform + '.sources.u8', dtype=np.uint8)
        ix = np.fromfile(transform + '.indices.i32', dtype='<i4')
        center = np.fromfile(transform + '.center.f32', dtype='<f4')
        gain = np.fromfile(transform + '.inv_scale.f32', dtype='<f4')
        proto = Path(transform + '.prototypes.i32')
        owner = np.fromfile(proto, dtype='<i4') if proto.exists() else np.full(width, -1)
        require(all(len(v) == width for v in (src, ix, center, gain, owner)), 'Transform width mismatch')
        require(np.all(src <= 2) and np.all(ix >= 0), 'Invalid transform sources/indices')
        require(np.isfinite(center).all() and np.isfinite(gain).all() and np.all(gain > 0), 'Invalid affine')
        for source, bound in ((0, H), (1, IS), (2, I)):
            mask = src == source
            require(np.all(ix[mask] < bound), 'Transform source index out of bounds')
            require(np.all((owner[mask] >= 0) & (owner[mask] < 64)) if source == 2
                    else np.all(owner[mask] == -1), 'Invalid prototype owner vector')
        owners = frozenset(map(int, owner[src == 2]))
        identity = (layer, basis)
        meta = dict(width=width, transform=transform, owners=owners)
        require(identity not in bases or bases[identity] == meta, 'Shared basis metadata differ')
        bases[identity] = meta
        files.update(transform + suffix for suffix in
                     ('.sources.u8', '.indices.i32', '.center.f32', '.inv_scale.f32'))
        if proto.exists():
            files.add(str(proto))
        factor = Path(head + '.factor.json')
        rank = 0
        if factor.exists():
            cfg = read(factor)
            rank = int(cfg['rank'])
            require(cfg['version'] == 1 and cfg['H'] == H and cfg.get('input_width', cfg.get('width')) == width,
                    'Factor descriptor dimensions differ')
            require(cfg.get('width', width) == width and rank == 256 and not cfg.get('V_bias', False),
                    'Unexpected rank/width/V bias')
            require(cfg.get('input_transform_prefix', transform) == transform, 'Factor transform mismatch')
            suffix_sizes = {'.V.i8': rank * width, '.V.scales.f32': 4 * rank,
                            '.latent.center.f32': 4 * rank, '.latent.inv_scale.f32': 4 * rank,
                            '.U.i8': H * rank, '.U.scales.f32': 4 * H, '.bias.f32': 4 * H}
            require(set(cfg['payload_sha256']) == {head + s for s in suffix_sizes}, 'Factor payload set differs')
            for payload, expected in cfg['payload_sha256'].items():
                require(frozen.get(payload) == expected, 'Factor payload hash absent/different in receipt')
            files.add(str(factor))
        else:
            suffix_sizes = {'.i8': H * width, '.scales.f32': 4 * H, '.bias.f32': 4 * H}
        for suffix, size in suffix_sizes.items():
            payload = head + suffix
            require(Path(payload).stat().st_size == size, f'Head payload size mismatch: {payload}')
            files.add(payload)
        heads[layer, expert] = dict(basis=basis, width=width, owners=owners, rank=rank,
                                   mac=rank * (H + width) if rank else H * width)
    require(set(heads) == {(1, expert) for expert in EXPECTED_EXPERTS}, 'Final eligible expert set changed')
    owners_by_layer = {}
    for (layer, _), meta in bases.items():
        owners_by_layer.setdefault(layer, set()).update(meta['owners'])
    require(not any(expert in owners_by_layer.get(layer, set()) for layer, expert in heads),
            'A resident GU owner was substituted')
    require(files <= set(frozen), 'Registry/transform/head file absent from frozen receipt')
    return heads, bases, files


def route_counters(route, heads):
    layer, ids = route['layer'], route['expert_ids']
    selected_ids = [expert for expert in ids if (layer, expert) in heads]
    selected = [heads[layer, expert] for expert in selected_ids]
    factor_ids = [expert for expert in selected_ids if heads[layer, expert]['rank']]
    required = set().union(*(head['owners'] for head in selected)) if selected else set()
    require(route['eligible_replacement_expert_ids'] == selected_ids, 'Selected replacement IDs differ')
    require(route['factorized_replacement_expert_ids'] == factor_ids, 'Selected factor IDs differ')
    require(route['required_prototype_ids'] == sorted(required), 'Required GU owner IDs differ')
    unique_bases = {head['basis']: head for head in selected}
    factor = [head for head in selected if head['rank']]
    counter = Counter(route_rows=1, routed_expert_events=K, native_coefficient_MAC=K * NATIVE_MAC,
                      replaced_expert_token_events=len(selected),
                      prototype_GU_row_pairs_computed=len(required),
                      prototype_owner_selected_phi_floats_bitchecked=len(required.intersection(ids)) * I,
                      prototype_cross_replaced_events=sum(bool(head['owners']) for head in selected),
                      basis_A8_unique_rows=len(unique_bases),
                      transformed_feature_values=sum(head['width'] for head in unique_bases.values()),
                      private_latent_A8_rows=len(factor),
                      factorized_latent_affine_values=sum(head['rank'] for head in factor),
                      factorized_V_coefficient_MAC=sum(head['rank'] * head['width'] for head in factor),
                      factorized_U_coefficient_MAC=sum(head['rank'] * H for head in factor),
                      readout_coefficient_MAC=sum(head['mac'] for head in selected),
                      incremental_prototype_GU_MAC=len(required.difference(ids)) * 2 * H * I)
    counter['coefficient_MAC_after_replacement_and_incremental_GU'] = (
        (K - len(selected)) * NATIVE_MAC + counter['readout_coefficient_MAC']
        + counter['incremental_prototype_GU_MAC'])
    return counter


def audit_prompt(folder, record, prompt_index, heads, native, native_prefill):
    key = record['id']
    summary = read(folder / (key + '.summary.json'))
    require(summary['prompt_id'] == key and summary['prompt_index'] == prompt_index, 'Summary prompt identity/order drift')
    require(summary['input_tokens'] == 64 and summary['scored_tokens'] == 63 and summary['decode_steps'] == 16,
            'Summary shape differs')
    tokens, generated = summary['token_ids'], summary['generated_token_ids']
    require(len(tokens) == 64 and 1 <= len(generated) <= 16, 'Token sequence length invalid')
    for token in tokens + generated:
        check_int(token, 0, VOCAB - 1, 'Invalid token ID')
    require(type(summary['stopped_eog']) is bool, 'Invalid EOG flag')
    require(len(generated) == summary['actual_generated_length'], 'Generated length differs')
    require(summary['stopped_eog'] or len(generated) == 16, 'Non-EOG generation stopped early')
    decoded = summary['decoded_generated_inputs']
    require(decoded == len(generated) - int(summary['stopped_eog']), 'EOG token was fed or last decode missing')
    pp = list(json_lines(folder / (key + '.positions.jsonl')))
    require(len(pp) == 63 and [row['position'] for row in pp] == list(range(63)), 'Missing/duplicate NLL positions')
    require([row['input_token_id'] for row in pp] == tokens[:-1]
            and [row['target_token_id'] for row in pp] == tokens[1:], 'Teacher-forced token alignment differs')
    for row in pp:
        require(math.isfinite(row['nll_F64']) and row['nll_F64'] >= -1e-10, 'Invalid NLL')
        check_int(row['top1_token_id'], 0, VOCAB - 1, 'Invalid top1 token ID')
    total = math.fsum(row['nll_F64'] for row in pp)
    require(abs(total - summary['sum_nll_F64']) < 1e-10, 'NLL sum does not match 63 stored targets')
    require(abs(summary['sum_nll_F64'] / 63 - summary['mean_nll_F64']) < 1e-13, 'Mean NLL differs')
    require(math.isfinite(summary['seconds']) and summary['seconds'] >= 0, 'Invalid inference duration')
    if native is not None:
        require(tokens == native['tokens'], 'Candidate input token sequence differs from native')
    seen = set()
    phases = {name: Counter() for name in (*PHASES, 'BOS', 'nonBOS62')}
    prefill = np.empty((64, L, K), dtype=np.uint8) if native is None else None
    route_agreement = 0
    for route in json_lines(folder / (key + '.routing.jsonl')):
        phase, step, token, layer = (route[name] for name in ('phase', 'step', 'token_in_batch', 'layer'))
        check_int(layer, 1, L, 'Invalid routed layer')
        ids = route['expert_ids']
        require(len(ids) == len(set(ids)) == K, 'Router did not select six distinct experts')
        for expert in ids:
            check_int(expert, 0, 63, 'Invalid routed expert')
        identity = phase, step, token, layer
        require(identity not in seen, 'Duplicate routing row')
        seen.add(identity)
        if phase == 'prefill':
            require(step == -1 and type(token) is int and 0 <= token < 64, 'Invalid prefill position')
            bucket = 'NLL_prefix63' if token < 63 else 'last_prefix_for_first_generated'
            if native is None:
                prefill[token, layer - 1] = ids
            else:
                same = bool(np.array_equal(native_prefill[token, layer - 1], ids))
                route_agreement += same
                if layer == 1:
                    require(same, 'Prefill router changed before the first substituted FFN')
        else:
            require(phase == 'decode' and token == 0 and type(step) is int and 0 <= step < decoded,
                    'Invalid decode step/position')
            bucket = 'decode_causal' if step + 1 < len(generated) else 'decode_last_unused'
        counts = route_counters(route, heads)
        phases[bucket].update(counts)
        if phase == 'prefill' and token < 63:
            phases['BOS' if token == 0 else 'nonBOS62'].update(counts)
    require(len(seen) == L * (64 + decoded), 'Missing/excess routing rows')
    require(phases['NLL_prefix63']['route_rows'] == 63 * L
            and phases['BOS']['route_rows'] == L and phases['nonBOS62']['route_rows'] == 62 * L,
            'NLL/BOS/nonBOS routing cardinality differs')
    require(phases['last_prefix_for_first_generated']['route_rows'] == L, 'First-generated causal prefix missing')
    require(phases['decode_causal']['route_rows'] == (len(generated) - 1) * L, 'Causal decode count differs')
    require(phases['decode_last_unused']['route_rows'] == (0 if summary['stopped_eog'] else L),
            'Last unused decode was counted as causal')
    all_counts = Counter()
    for phase in PHASES:
        all_counts.update(phases[phase])
    all_counts['native_shared_z_floats_bitchecked'] = len({layer for layer, expert in heads}) * IS * (64 + decoded)
    for name in SUMMARY_COUNTERS:
        require(summary[name] == all_counts[name], f'{key}: routing-derived summary counter differs: {name}')
    stored = dict(tokens=tokens, generated=generated, nll=[row['nll_F64'] for row in pp],
                  top1=[row['top1_token_id'] for row in pp], mean_nll=summary['mean_nll_F64'],
                  stopped_eog=summary['stopped_eog'], decoded=decoded,
                  route_agreement=route_agreement)
    return stored, prefill, phases, all_counts


def audit_arm(arm, plan, records, batch_results, frozen, baseline=None, baseline_prefill=None):
    heads, bases, files = read_registry(arm['registry'], frozen)
    name, root = arm['name'], Path(plan['output']) / arm['name']
    require({path.name for path in root.iterdir() if path.is_dir()} == {f'shard{i}' for i in range(16)},
            'Unexpected/missing shard directories')
    require((not heads) == (name == 'native'), 'Native/replacement registry mismatch')
    if 'rank256' in name:
        require(all(head['rank'] == 256 for head in heads.values()), 'Missing rank256 head')
    else:
        require(all(head['rank'] == 0 for head in heads.values()), 'Unexpected factorized head')
    if name.startswith('prototype25'):
        require(all(head['owners'] == {25} for head in heads.values()), 'Resident prototype changed')
    else:
        require(all(not head['owners'] for head in heads.values()), 'Unexpected resident prototype')
    groups = {tuple(group) for group in plan['cpu_groups']}
    require(groups == {tuple(range(start, start + 4)) for start in range(0, 64, 4)}, 'Unexpected CPU groups')
    summaries, all_phases, arm_counts = {}, {phase: Counter() for phase in (*PHASES, 'BOS', 'nonBOS62')}, Counter()
    prefill_by_id = {} if baseline is None else None
    for shard in range(16):
        folder = root / f'shard{shard}'
        jobs = records[shard::16]
        command = read(folder / 'command.json')
        argv = command['argv']
        require(command['CPU_only'] is True and len(argv) == 12 and argv[:2] == ['taskset', '-c'], 'Unexpected command/backend')
        require(tuple(map(int, argv[2].split(','))) in groups, 'Worker affinity outside frozen groups')
        require(argv[3] == plan['binary'] and argv[4] == str(DATA.parent / 'models/deepseek-v2-lite-chat-q8_0.gguf'),
                'Wrong binary/model command')
        require(argv[5:] == [str(folder / 'jobs.tsv'), str(folder / 'data'), '64', '4', '16', arm['registry'], 'w8'],
                'Wrong executed quality arguments')
        require((folder / 'jobs.tsv').read_text() == ''.join(f"{row['path']}\t{row['id']}\n" for row in jobs),
                'Shard job IDs/input order differ')
        execution = read(folder / 'execution.json')
        require(execution == batch_results[name, shard], 'Shard/root execution receipt differs')
        done = read(folder / 'data/completion.json')
        require(done == execution['completion'], 'Shard/root completion differs')
        require(done['readouts'] == len(heads) and done['transform_bases'] == len(bases), 'Loaded head/basis count differs')
        require(done['prototype_owners_kept_native'] is True and done['harness_recomputes_native_FFN'] is True,
                'Native-owner/harness contract differs')
        for suffix in ('.summary.json', '.positions.jsonl', '.routing.jsonl'):
            actual = {path.name[:-len(suffix)] for path in (folder / 'data').glob('*' + suffix)}
            require(actual == {row['id'] for row in jobs}, 'Extra/missing prompt artifacts')
        shard_counts = Counter()
        for index, record in enumerate(jobs):
            key = record['id']
            require(key not in summaries, 'Duplicate prompt across shards')
            row, prefill, phases, counts = audit_prompt(folder / 'data', record, index, heads,
                                                       baseline.get(key) if baseline is not None else None,
                                                       baseline_prefill.get(key) if baseline is not None else None)
            summaries[key] = row
            if prefill_by_id is not None:
                prefill_by_id[key] = prefill
            for phase in all_phases:
                all_phases[phase].update(phases[phase])
            shard_counts.update(counts)
        for key in SUMMARY_COUNTERS:
            completion_key = 'new_basis_A8_rows' if key == 'basis_A8_unique_rows' else key
            require(done[completion_key] == shard_counts[key], f'{name}/shard{shard}: completion counter differs: {key}')
        require(done['transformed_feature_values'] == shard_counts['transformed_feature_values'], 'Transform values differ')
        arm_counts.update(shard_counts)
        if shard % 4 == 3:
            print(json.dumps(dict(event='independent_audit', arm=name, completed_shards=shard + 1)), flush=True)
    require(len(summaries) == len({tuple(row['tokens']) for row in summaries.values()}) == 1000,
            'Expected 1000 unique complete input token sequences')
    require(set(summaries) == {row['id'] for row in records}, 'Arm/corpus prompt IDs differ')
    details = dict(prompts=1000, unique_input_prefixes=1000, articles=1000, scored_targets=63000,
                   input_tokens_match_native=True if baseline is not None else None,
                   early_EOG_prompts=sum(row['stopped_eog'] for row in summaries.values()),
                   generated_tokens=sum(len(row['generated']) for row in summaries.values()),
                   generated_length_range=[min(len(row['generated']) for row in summaries.values()),
                                           max(len(row['generated']) for row in summaries.values())],
                   canonical_readouts=len(heads), transform_bases=len(bases), registry_files=len(files),
                   summary_counter_totals={key: arm_counts[key] for key in SUMMARY_COUNTERS},
                   phases={phase: dict(counts) for phase, counts in all_phases.items()},
                   mean_nll=float(np.mean([row['mean_nll'] for row in summaries.values()])))
    return summaries, prefill_by_id, details


def contrast(candidate, native, primary):
    keys = sorted(native)
    delta = np.asarray([candidate[key]['mean_nll'] - native[key]['mean_nll'] for key in keys])
    nll_change = float(delta.mean())
    result = dict(mean_delta_nll=nll_change, relative_subset_PPL_change=math.expm1(nll_change),
                  prefill_top1_agreement=sum(sum(x == y for x, y in zip(candidate[key]['top1'], native[key]['top1']))
                                             for key in keys) / 63000,
                  prefill_routing_slot_order_agreement=sum(candidate[key]['route_agreement'] for key in keys) / (1000 * 64 * L),
                  generation_exact_prompt_fraction=sum(candidate[key]['generated'] == native[key]['generated'] for key in keys) / 1000,
                  generation_is_task_accuracy=False,
                  BOS_mean_delta_nll=float(np.mean([candidate[key]['nll'][0] - native[key]['nll'][0] for key in keys])),
                  nonBOS62_mean_delta_nll=float(np.mean([np.mean(candidate[key]['nll'][1:]) - np.mean(native[key]['nll'][1:])
                                                        for key in keys])))
    if primary:
        generator = np.random.default_rng(20261109)
        draws = np.empty(10000)
        for start in range(0, 10000, 100):
            samples = generator.integers(1000, size=(100, 1000))
            draws[start:start + 100] = delta[samples].mean(axis=1)
        lo, hi = map(float, np.quantile(draws, [.025, .975]))
        margin = math.log(1.02)
        result.update(ci95_delta_nll=[lo, hi], ci95_relative_subset_PPL_change=[math.expm1(lo), math.expm1(hi)],
                      verdict='pass' if hi <= margin else ('fail' if lo > margin else 'inconclusive'),
                      margin_log_PPL=margin,
                      bootstrap=dict(seed=20261109, replicates=10000, articles=1000,
                                     unit='article', one_prompt_per_article=True))
    else:
        result['verdict'] = 'descriptive_secondary_comparison'
    return result


def run(args):
    plan_path = Path(args.plan).resolve()
    plan = read(plan_path)
    execution = completion_barrier(plan)  # Do not move TEST reads above this barrier.
    integrity = Integrity()
    receipt, protocol = frozen_integrity(plan_path, plan, integrity)
    corpus_path = DATA / 'corpus103/manifest.json'
    corpus = read(corpus_path)
    records = [row for row in corpus['records'] if row['split'] == 'test']
    ids, articles = [row['id'] for row in records], [row['article_id'] for row in records]
    require(len(records) == len(set(ids)) == len(set(articles)) == 1000, 'TEST must contain 1000 unique prompt/article IDs')
    validation_articles = {row['article_id'] for row in corpus['records'] if row['split'] == 'quality_validation'}
    require(not validation_articles.intersection(articles), 'Fresh quality validation/TEST article overlap')
    for record in records:
        integrity.check(record['path'], record['prompt_sha256'])
        require(protocol['inputs_sha256'].get(record['path']) == record['prompt_sha256'], 'Prompt missing from frozen protocol')
    batch_results = {(batch['arm'], batch['shard']): batch for batch in execution['batches']}
    arms = {arm['name']: arm for arm in plan['arms']}
    native, native_prefill, native_report = audit_arm(arms['native'], plan, records, batch_results,
                                                     receipt['files_sha256'])
    result = dict(status='PASS', at_utc=datetime.now(timezone.utc).isoformat(),
                  source=str(Path(__file__).resolve()), source_sha256=sha(__file__),
                  plan=str(plan_path), plan_sha256=sha(plan_path),
                  selection_receipt=plan['selection_receipt'], selection_receipt_sha256=RECEIPT_SHA256,
                  independent_of_qa_load_arm=True, completion_barrier_passed_before_TEST_reads=True,
                  completed_shards=80, prompts_per_arm=1000, unique_articles=1000,
                  corpus_id_article_sha256=canonical_sha(sorted(zip(ids, articles))),
                  integrity=dict(frozen_inputs=343, source_snapshots=11,
                                 model_full_sha256=receipt['model_identity']['sha256'],
                                 posthashguard_covers_all_frozen_inputs=True),
                  CPU_only=dict(CUDA_VISIBLE_DEVICES='', gpu_layers=0, threads_per_worker=4,
                                worker_groups=plan['cpu_groups'], static_binary_sha256=sha(plan['binary'])),
                  primary_arm=plan['primary_arm'], native=native_report, arms={},
                  limitations=[
                      'Length/EOG/decode causality is independently reconstructed; discarded vocabulary logits prevent an independent generation argmax/NLL recomputation.',
                      'Successful native-owner/shared-z bitchecks are reconstructed from routes and frozen runtime completion counters; raw activation tensors were not retained by quality runs.',
                      'Subset PPL and greedy agreement do not measure task accuracy or standard concatenated perplexity.',
                      'Coefficient MAC counts are deployment models; this verification harness recomputes native FFNs and does not measure compressed-system speed or physical flash IO.',
                  ])
    for name, arm in arms.items():
        if name == 'native':
            continue
        value, _, report = audit_arm(arm, plan, records, batch_results, receipt['files_sha256'], native, native_prefill)
        report['vs_native'] = contrast(value, native, name == plan['primary_arm'])
        result['arms'][name] = report
    # Optional single comparison to the preexisting final-summary implementation,
    # after all independent data/counter/inference checks have completed.
    if args.reference_summary:
        reference = read(args.reference_summary)
        require(reference['primary_arm'] == result['primary_arm'], 'Reference primary differs')
        require(reference['native']['unique_prompts'] == 1000, 'Reference native count differs')
        for name, arm in result['arms'].items():
            ref = reference['arms'][name]
            require(ref['audit']['unique_prompts'] == 1000 and ref['vs_native']['prompts'] == 1000, 'Reference arm count differs')
            require(abs(ref['vs_native']['mean_delta_nll'] - arm['vs_native']['mean_delta_nll']) < 1e-12,
                    'Reference mean delta differs')
            for key, count in arm['summary_counter_totals'].items():
                if key in ref['audit']['summary_counter_totals']:
                    require(ref['audit']['summary_counter_totals'][key] == count, 'Reference counter total differs')
            if name == result['primary_arm']:
                require(np.allclose(ref['vs_native']['ci95_delta_nll'], arm['vs_native']['ci95_delta_nll'], rtol=0, atol=1e-12),
                        'Reference primary bootstrap interval differs')
                require(ref['vs_native']['verdict'] == arm['vs_native']['verdict'], 'Reference primary verdict differs')
        result['reference_summary_comparison'] = dict(status='PASS', path=args.reference_summary,
                                                       sha256=sha(args.reference_summary))
    result['integrity']['distinct_files_rehashed'] = len(integrity.checked)
    output = Path(args.output).resolve()
    require(str(output) not in receipt['files_sha256'] and str(output) not in receipt['source_snapshots_sha256'],
            'Refusing to overwrite a frozen input')
    require(not output.exists(), 'Audit output already exists; choose a new independent-audit revision')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + '.partial')
    temporary.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n')
    temporary.replace(output)
    print(json.dumps(dict(status='PASS', output=str(output), primary=result['arms'][result['primary_arm']]['vs_native'])), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', default=str(DATA / 'final1000-plan.json'))
    parser.add_argument('--output', required=True)
    parser.add_argument('--reference-summary')
    run(parser.parse_args())
