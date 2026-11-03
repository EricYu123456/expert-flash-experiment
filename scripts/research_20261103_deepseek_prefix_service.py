#!/usr/bin/env python3
"""Fixed dense raw0 service/footprint, separate from26routed DeepSeek layers.

The numerical reference includes raw0. Add this identical, fixed cost to both
placement comparisons when reporting complete DeepSeek FFN service. No router,
attention, full-model time or hardware calibration is supplied here.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np

from research_20261103_ordinary_read_backend import (
    ROOT, NativeReplay, ReferenceGraph, Service, lower_rows, physical_addresses, sha,
)

BASE = ROOT / 'results/research-2026-11-02/round2'
DEST = BASE / 'ordinary-read-compiler/deepseek-fixed-dense-prefix'


def main():
    DEST.mkdir(parents=True, exist_ok=True)
    source = BASE / 'deepseek-full-ffn-integer-reference/frozen_equal_projection_rows.json'
    metadata = json.loads(source.read_text())
    # Verified actual dimensions in the sealed full-FFN reference; retain its
    # whole inventory hash rather than reinterpreting shared2816 as dense10944.
    inventory = {r['weight']:r for r in metadata}
    for projection, rows, columns, per_chip in [('gate', 10944, 2048, 684),
                                                ('up', 10944, 2048, 684),
                                                ('down', 2048, 10944, 128)]:
        entry = inventory[f'blk.0.ffn_{projection}.weight']
        assert (entry['rows'], entry['columns'], entry['experts']) == (rows, columns, 1)
        assert entry['equal_rows_per_chip'] == [per_chip] * 16
    native = NativeReplay(DEST / 'build'); records = []
    for command in (250, 5000, 10000):
        s = Service(command_ns=command, output_fringe=1, input_buffer_bytes=32768)
        plan = lower_rows(np.full((1, 16), 1368), np.full((1, 16), 128), 2048, 10944, s)
        got = native.run(plan, np.asarray([[0]], dtype=np.int32), 1, 0, s, trace=True)
        reference = ReferenceGraph.build(plan, [0], s).run()
        assert np.array_equal(got['trace'], reference['trace'])
        address = physical_addresses(plan, s)
        label = f'cmd{command}'
        np.savez_compressed(DEST / (label + '.npz'), **{k:plan[k] for k in ('offsets', 'jobs', 'input_columns', 'total_rows')}, trace=got['trace'])
        (DEST / (label + '.physical.json')).write_text(json.dumps(dict(raw_model_layer=0,
             projection='dense gate_up21888x2048 anddown2048x10944',
             service=asdict(s), logical=plan['logical'], **address), indent=2) + '\n')
        records.append(dict(command_ns=command, cost_ns=int(got['cost'][0]),
                            lower_bound_ns=int(got['lower_bound'][0]),
                            useful_bytes=address['useful_bytes'],
                            programmed_page_bytes=address['programmed_page_bytes'],
                            reserved_wave_bytes=address['reserved_wave_bytes'],
                            standalone_allocated_lsb_block_bytes=address['allocated_lsb_block_bytes'],
                            chip_blocks=address['chip_blocks']))
    (DEST / 'summary.json').write_text(json.dumps(dict(status='complete fixed dense prefix reference service',
         created_utc=datetime.now(timezone.utc).isoformat(), source_sha256=sha(__file__),
         source_inventory=str(source), source_inventory_sha256=sha(source),
         backend_sha256=sha(Path(__file__).with_name('research_20261103_ordinary_read_backend.py')),
         native_build=native.provenance, records=records,
         scope='Same fixed16-chip equal complete rows in both layout families. Add exactly once per token, before26MoE layers. Not measured full model; attention/router still excluded. Standalone allocated blocks may merge with existing holes in a later joint physical pack.'), indent=2) + '\n')
    print(json.dumps(records, indent=2), flush=True)


if __name__ == '__main__':
    main()
