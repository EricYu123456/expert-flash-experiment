#!/usr/bin/env python3
"""Independent integer/type/physical-row audit; never imports the generator."""
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import itertools
import json
import math
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
SOURCE = ROOT / 'scripts/research_20261102_compound_replica_theorem.py'
WAVE = 65536


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def brute_assignment(experts, forbidden):
    """Enumerate color choices directly; independent of the source's DP matcher."""
    paired_targets = {tuple(2 if c in (a, b) else 1 for c in range(4))
                      for a in (0, 2) for b in (1, 3)}
    allowed = [[c for c in range(4) if forbidden[e] != c] for e in experts]
    for colors in itertools.product(*allowed):
        counts = tuple(colors.count(c) for c in range(4))
        if counts in paired_targets:
            return dict(zip(experts, colors)), counts
    raise AssertionError(('No integral paired matching', experts))


def main():
    theorem_path = HERE / 'theorem.json'
    before = {str(p): sha(p) for p in (SOURCE, theorem_path)}
    data = json.loads(theorem_path.read_text())
    assert data['source_sha256'] == sha(SOURCE)
    default, forbidden, tiles = (data[k] for k in
        ('default_omitted_group', 'forbidden_omitted_group', 'tiles'))
    assert len(tiles) == 64 * 12
    by_expert, chip_tiles, singleton = defaultdict(list), Counter(), Counter()
    locations, storage = {}, []
    for ti, t in enumerate(tiles):
        e, start, group, lane, owners = (t[k] for k in
            ('expert', 'row_start', 'primary_group', 'lane', 'locations'))
        assert t['rows'] == 128 and start % 128 == 0
        assert group != default[e] and owners[0] == group * 4 + lane
        assert len(owners) in (1, 2) and len(owners) == len(set(owners))
        if len(owners) == 1:
            assert group == forbidden[e]
            singleton[owners[0]] += 1
        else:
            assert owners[1] == default[e] * 4 + lane
            assert group != forbidden[e]
        by_expert[e].append(t)
        for copy, chip in enumerate(owners):
            block, wl = divmod(chip_tiles[chip], 192)
            storage.append([ti, e, start, chip, copy, block, wl])
            locations[ti, chip] = block, wl
            chip_tiles[chip] += 1
    assert sum(chip_tiles.values()) == 1456
    assert [singleton[c] for c in range(16)] == [5] * 16
    native_mappings = 0
    for e in range(64):
        assert sorted(t['row_start'] for t in by_expert[e]) == list(range(0, 1536, 128))
        assert forbidden[e] != default[e]
        for omit in range(4):
            if omit == forbidden[e]:
                continue
            chips, rows = [], []
            for t in by_expert[e]:
                chip = t['locations'][0]
                if omit != default[e] and t['primary_group'] == omit:
                    chip = 4 * default[e] + t['lane']
                assert chip in t['locations'] and chip // 4 != omit
                chips.append(chip); rows.append(t['row_start'])
            assert len(set(chips)) == 12
            assert set(chips) == set(range(16)) - set(range(4 * omit, 4 * omit + 4))
            assert sorted(rows) == list(range(0, 1536, 128))
            native_mappings += 1
    assert native_mappings == 236
    types = [[e for e in range(64) if forbidden[e] == f] for f in (-1, 0, 1, 2, 3)]
    assert list(map(len, types)) == [44, 5, 5, 5, 5]
    tested, weighted = [], 0
    # Independent Cartesian count enumeration, rather than source compositions().
    for counts in itertools.product(range(7), repeat=5):
        if sum(counts) != 6 or any(n > len(pool) for n, pool in zip(counts, types)):
            continue
        query = [e for n, pool in zip(counts, types) for e in pool[:n]]
        omission, target = brute_assignment(query, forbidden)
        # Direct Hall check for every color subset, with target color capacities.
        for mask in range(1, 16):
            color_set = [c for c in range(4) if mask >> c & 1]
            neighbors = sum(any(forbidden[e] != c for c in color_set) for e in query)
            assert sum(target[c] for c in color_set) <= neighbors
        jobs, experts_per_chip, rows_seen = [0] * 16, [set() for _ in range(16)], set()
        for e in query:
            for t in by_expert[e]:
                owner = t['locations'][0]
                if omission[e] != default[e] and t['primary_group'] == omission[e]:
                    owner = 4 * default[e] + t['lane']
                assert owner in t['locations']
                assert (e, t['row_start']) not in rows_seen
                rows_seen.add((e, t['row_start'])); jobs[owner] += 1
                experts_per_chip[owner].add(e)
        assert len(rows_seen) == sum(jobs) == 72
        assert max(jobs) == 5 and [jobs[c] + jobs[c + 8] for c in range(8)] == [9] * 8
        assert all(len(es) == n for es, n in zip(experts_per_chip, jobs))
        input_bytes = [512 * len(es) for es in experts_per_chip]
        output_bytes = [128 * 4 * n for n in jobs]
        costs = dict(flash_ns=max(jobs) * 28000, control_ns=72 * 250, host_ns=2000,
            input_ns=max(math.ceil(6 * 512 / 8),
                max(math.ceil((input_bytes[c] + input_bytes[c + 8]) / 2) for c in range(8))),
            output_ns=max(math.ceil(sum(output_bytes) / 8),
                max(math.ceil((output_bytes[c] + output_bytes[c + 8]) / 2) for c in range(8))))
        assert sum(costs.values()) == 166912
        weight = math.prod(math.comb(len(pool), n) for n, pool in zip(counts, types))
        weighted += weight
        tested.append(dict(type_counts=counts, route_count=weight, omitted=omission,
                           target_counts=target, components_ns=costs))
    assert len(tested) == 206 and weighted == math.comb(64, 6) == 74974368
    assert before == {str(p): sha(p) for p in (SOURCE, theorem_path)}
    ledger_path = HERE / 'diagnostic_storage_ledger.json'
    ledger = dict(columns=['tile_index', 'expert', 'row_start', 'chip', 'replica_index',
                           'block_per_plane', 'wordline_all_four_planes'], records=storage,
                  pages_per_record=4, page_bytes=16384,
                  logical_bytes=768 * WAVE, copied_bytes=688 * WAVE,
                  physical_page_bytes=1456 * WAVE,
                  per_chip_stored_tiles=[chip_tiles[c] for c in range(16)],
                  per_chip_block_stripes=[math.ceil(chip_tiles[c] / 192) for c in range(16)],
                  covered_plane_page_slots_are_disjoint=True,
                  virtual_addresses_only_no_programmed_weights=True)
    ledger_path.write_text(json.dumps(ledger, indent=2) + '\n')
    reread = json.loads(ledger_path.read_text())
    occupied_pages = {(record[3], plane, record[5], record[6])
                      for record in reread['records'] for plane in range(4)}
    assert len(occupied_pages) == len(reread['records']) * 4 == 5824
    # A native dense block-diagonal zero embedding refutes the inference that
    # one input-vector buffer implies one expert context per ordinary wave.
    import numpy as np
    rng = np.random.default_rng(20261102)
    wa = rng.integers(-128, 128, size=(32, 512), dtype=np.int16).astype(np.int8)
    wb = rng.integers(-128, 128, size=(32, 512), dtype=np.int16).astype(np.int8)
    xa = rng.integers(-128, 128, size=512, dtype=np.int16).astype(np.int8)
    xb = rng.integers(-128, 128, size=512, dtype=np.int16).astype(np.int8)
    blockdiag = np.zeros((64, 1024), dtype=np.int8)
    blockdiag[:32, :512], blockdiag[32:, 512:] = wa, wb
    reference = np.concatenate([wa.astype(np.int64) @ xa.astype(np.int64),
                                wb.astype(np.int64) @ xb.astype(np.int64)])
    combined = blockdiag.astype(np.int64) @ np.concatenate([xa, xb]).astype(np.int64)
    assert np.array_equal(reference, combined)
    assert blockdiag.nbytes == WAVE and wa.nbytes == wb.nbytes == 16384
    audit = dict(status='PASS_conditional_restricted_storage_theorem',
        created_utc=datetime.now(timezone.utc).isoformat(), unchanged_source_sha256=before,
        auditor_source_sha256=sha(Path(__file__)),
        type_compositions=206, represented_routes=weighted, expert_omit_row_maps=native_mappings,
        primary_payload_mib=48, copied_payload_mib=43, stored_payload_mib=91,
        stored_pages=5824, storage_ratio=1456 / 768,
        minimum_extra_canonical_copies=688, source_counterexample_found=False,
        components_ns=tested[0]['components_ns'], phase_model_ns=166912,
        native_dataflow='Every run reads one full 128x512 I8 matrix, returns128 complete '
            'rows and takes one input vector. Per expert/chip there is exactly one '
            'executed run. Host scatters rows; no partial sums, coding, quantization '
            'or new accumulator type. GU->host activation->down dependency is retained.',
        software_required='Controller stores/follows tile owner and row-offset maps, '
            'matches currently selected six experts to four omission choices and '
            'dispatches existing block/offset/page-count GEMVs. The published host '
            'aif_post does not directly expose these addresses. No new die PE is used.',
        numeric_scope='236 structural expert/omit maps independently checked. The '
            'generator also ran synthetic INT64 reference GEMVs. Neither demonstrates '
            'a physical AiF result datatype or real model inference; full-row copies '
            'preserve the original per-row arithmetic instead of introducing a new type.',
        capacity='The43MiB extra are useful/physical-page I8 payload. In this standalone '
            'phase all90-92 atoms/chip fit one192-wave block stripe; block reservation '
            'is192MiB including idle tails. Other model streams may share those tails. '
            'Do not equate page storage optimality with reserved-stripe optimality. '
            'Scale arrays, FTL/ECC/spare and other tensors are excluded.',
        timing_limit='166912ns is an arithmetic model, not measured latency: it omits '
            'route-matcher/controller lookup work and host reassembly/activation compute. '
            'The72-command and transfer costs are explicit; tR28us/control250ns/host2us '
            'remain uncalibrated assumptions. No background/erase, q-replay or request SLO claim.',
        byte_bound_extension=dict(status='valid_under_private_input_single_context_assumptions',
            not_an_inherent_property_of_existing_hardware=True,
            proof='Let u_ec be logical weight bytes of expert e that have exactly one '
            'stored copy, on chip c. For every6-expert route S, native distinct-input '
            'GEMVs require at least sum_e_in_S ceil(u_ec/65536) waves on c. Thus at most '
            'five experts can have u_ec>0. A route containing all of those experts plus '
            'arbitrary others gives sum_e ceil(u_ec/65536)<=5, hence sum_e u_ec<=5W. '
            'Across16chips U<=80W. With logical bytes B=768W and no cross-item dedup or '
            'coding, stored useful bytes >=U+2(B-U)=2B-U, so extra>=688W. Construction '
            'attains that byte count without row fragmentation or padding.',
            requirements=['down experts have independent input contexts per ordinary NAND wave',
                'no read-once/PB replay across those contexts or independent per-plane contexts',
                'no dense block-diagonal zero embedding with concatenated expert inputs',
                'all necessary weights stored on device; no host weight computation/cache shortcut',
                'no coding or physical dedup across distinct logical weight items'],
            exclusions='GU common-input row packing is not covered by sum-per-expert '
                'ceil. Neither are block-diagonal multi-expert dense matrices, altered '
                'cache/compute interfaces or cross-item representations. The byte bound '
                'is about copies, not erase-block tails. Do not claim unrestricted '
                'arbitrary-row optimality from this conditional lemma.',
            existing_hardware_counterexample=dict(
                matrices=['32x512 I8 expert A fringe', '32x512 I8 expert B fringe'],
                embedding='diag(W_A,W_B) has64rows x1024columns with32768bytes useful '
                    'weights and32768bytes structural zeros; concatenate x_A and x_B.',
                separate_normal_read_waves=2, combined_normal_read_waves=1,
                combined_physical_bytes=65536, combined_physical_pages=4,
                no_new_PE_or_quantization=True, exact_synthetic_I8_dot_identity_checked=True,
                violates='sum_e ceil(u_ec/W) as an unrestricted hardware lower bound',
                canonical_scope='Two whole128x512 canonical tiles instead produce '
                    'a256x1024 block diagonal matrix,256KiB/fourwaves versus original '
                    'two full waves. This fringe counterexample does not refute the '
                    'audited canonical tile construction.')),
        novelty_assessment='Not established. The construction uses standard replicated '
            'declustering plus a mandatory singleton-byte counting lower bound and a '
            'four-color Hall matching. The exact compound-query parameter point is '
            'validated, but these ingredients look like a short corollary of existing '
            'replicated/batch-code theory. A full primary literature check is needed '
            'before claiming a new theorem, and this alone is not an architecture breakthrough.',
        prior_primary_urls=[
            'https://repository.bilkent.edu.tr/bitstreams/938fed97-cc40-48c5-bbd8-01da42e95ae2/download',
            'https://faculty.cse.louisville.edu/nihat/papers/TOS2013.pdf'],
        no_routing_or_model_weights_opened=True, no_parent_artifact_modified=True,
        storage_ledger=dict(path=str(ledger_path), sha256=sha(ledger_path)),
        type_witnesses=tested)
    (HERE / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n')
    print(json.dumps({k: audit[k] for k in ('status', 'type_compositions', 'represented_routes',
        'minimum_extra_canonical_copies', 'stored_payload_mib', 'phase_model_ns')}))


if __name__ == '__main__':
    main()
