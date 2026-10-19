#!/usr/bin/env python3
"""Trace-driven INT8 AiF chip demand, page rounding and cold-read timing.

This is an explicit architectural model, not a measurement of hardware MAC lanes.
GGUF quantization is used for correct routing only. Modeled weights are INT8.
"""
import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
import struct
import math

CHIPS, PLANES, PAGE = 16, 4, 16384
CHIP_BW = 6_400_000_000
COLD_NS, CR_NS = 28000, 9700


def ceildiv(a, b):
    return (a + b - 1) // b


@dataclass
class Demand:
    useful: int = 0
    pages: int = 0
    rounds: int = 0
    cold: int = 0

    @property
    def service_ns(self):
        return max(self.rounds * CR_NS + self.cold * (COLD_NS - CR_NS),
                   ceildiv(self.pages * PAGE * 5, 32))

    def add_stream(self, nbytes, block_pages, cold_per_block):
        if not nbytes:
            return
        pages = ceildiv(nbytes, PAGE)
        rounds = ceildiv(pages, PLANES)
        full, remainder = divmod(rounds, block_pages)
        self.useful += nbytes
        self.pages += pages
        self.rounds += rounds
        self.cold += full * min(cold_per_block, block_pages) + min(cold_per_block, remainder)


def phase_demand(experts, projections, group_chips, block_pages=192, cold_per_block=1):
    """Each expert has a fixed chip group; preserve complete output rows.

    Whole row shards are block-contained sequential page streams; pages use four planes.
    All ready experts in a phase can run concurrently. A chip serializes its
    streams. Gate/up are separate phases unless caller explicitly combines them.
    """
    if group_chips not in (1, 2, 4, 8, 16):
        raise ValueError("group_chips must divide 16")
    demands = [Demand() for _ in range(CHIPS)]
    for expert in experts:
        first_chip = (expert % (CHIPS // group_chips)) * group_chips
        for rows, columns in projections:
            for shard in range(group_chips):
                row_count = rows // group_chips + (shard < rows % group_chips)
                demands[first_chip + shard].add_stream(row_count * columns, block_pages, cold_per_block)
    return demands


def encode_command(demands):
    return struct.pack('<IIII', 0x31464941, 1, CHIPS, 0) + b''.join(
        struct.pack('<QQII', d.useful, d.pages, d.rounds, d.cold) for d in demands)


def tensor_dimensions(metadata):
    dimensions = {}
    for tensor in metadata['tensors']:
        if 'projection' in tensor:
            shape = tensor['shape_ggml']
            dimensions.setdefault(tensor['layer'], {})[tensor['projection']] = (shape[1], shape[0])
    if not dimensions:
        raise ValueError('No routed expert tensors in metadata')
    for layer, values in dimensions.items():
        if set(values) != {'gate', 'up', 'down'}:
            raise ValueError(f'Layer {layer} missing SwiGLU expert projection: {values}')
        gate, up, down = (values[key] for key in ('gate', 'up', 'down'))
        if gate != up or down != (up[1], up[0]):
            raise ValueError(f'Unexpected SwiGLU dimensions in layer {layer}')
    return dimensions


def load_events(paths):
    events = []
    for path in paths:
        records = [json.loads(line) for line in path.read_text().splitlines() if line]
        seen = set()
        for record in records:
            key = record['token_index'], record['layer']
            if key in seen or len(set(record['expert_ids'])) != len(record['expert_ids']):
                raise ValueError(f'Duplicate route or experts in {path}: {key}')
            seen.add(key)
            record['prompt'] = path.stem.replace('.routing', '')
        events.extend(records)
    return events


def evaluate(events, dimensions, group_chips, *, block_pages=192, cold_per_block=1,
             serial_experts=False, fuse_gate_up=False, stack_gate_up=False, intermediate_override=None,
             binary_path=None):
    totals = dict(useful_bytes=0, physical_pages=0, read_rounds=0, cold_reads=0,
                  model_ns=0, active_chip_commands=0, commands=0, chip_service_ns=0)
    chip_bytes, chip_service = [0]*CHIPS, [0]*CHIPS
    token_keys, prompt_keys = set(), set()
    token_ns = {}
    binary = binary_path.open('wb') if binary_path else None
    try:
        for event in events:
            token_keys.add((event['prompt'], event['token_index']))
            prompt_keys.add(event['prompt'])
            dims = dimensions[event['layer']]
            if intermediate_override is not None:
                hidden = dims['up'][1]
                intermediate = intermediate_override
                dims = dict(gate=(intermediate, hidden), up=(intermediate, hidden),
                            down=(hidden, intermediate))
            phases = [[dims['gate']], [dims['up']], [dims['down']]]
            if fuse_gate_up:
                # The two streams remain separate, including their cold starts.
                phases = [[dims['gate'], dims['up']], [dims['down']]]
            if stack_gate_up:
                # A real layout change: vertically concatenate matrices sharing x.
                # One contiguous 2I x H matrix, followed by the dependent down GEMV.
                phases = [[(dims['gate'][0]+dims['up'][0], dims['up'][1])], [dims['down']]]
            expert_sets = [[e] for e in event['expert_ids']] if serial_experts else [event['expert_ids']]
            for phase in phases:
                for experts in expert_sets:
                    demands = phase_demand(experts, phase, group_chips, block_pages, cold_per_block)
                    duration = max(d.service_ns for d in demands)
                    totals['model_ns'] += duration
                    key = event['prompt'], event['token_index']
                    token_ns[key] = token_ns.get(key, 0) + duration
                    totals['commands'] += 1
                    for chip, d in enumerate(demands):
                        chip_bytes[chip] += d.useful
                        chip_service[chip] += d.service_ns
                        totals['useful_bytes'] += d.useful
                        totals['physical_pages'] += d.pages
                        totals['read_rounds'] += d.rounds
                        totals['cold_reads'] += d.cold
                        totals['chip_service_ns'] += d.service_ns
                        totals['active_chip_commands'] += d.useful > 0
                    if binary:
                        binary.write(encode_command(demands))
    finally:
        if binary:
            binary.close()
    if not totals['commands']:
        raise ValueError('No routing events')
    latencies = sorted(token_ns.values())
    totals.update(ffn_p50_ms=latencies[math.ceil(0.50*len(latencies))-1]/1e6,
                  ffn_p95_ms=latencies[math.ceil(0.95*len(latencies))-1]/1e6,
                  token_model_ns=latencies, group_chips=group_chips, block_pages=block_pages,
                  cold_per_block=cold_per_block, serial_experts=serial_experts,
                  fuse_gate_up=fuse_gate_up, stack_gate_up=stack_gate_up, intermediate_override=intermediate_override,
                  prompts=len(prompt_keys), decode_tokens=len(token_keys), routing_records=len(events),
                  ffn_ms_per_token=totals['model_ns']/len(token_keys)/1e6,
                  active_chip_fraction=totals['active_chip_commands']/(CHIPS*totals['commands']),
                  service_balance=totals['chip_service_ns']/(CHIPS*totals['model_ns']),
                  useful_bandwidth_fraction=totals['useful_bytes']*1e9/(CHIPS*CHIP_BW*totals['model_ns']),
                  physical_page_fill=totals['useful_bytes']/(totals['physical_pages']*PAGE),
                  plane_slot_fill=totals['useful_bytes']/(totals['read_rounds']*PLANES*PAGE),
                  chip_bytes=chip_bytes, chip_service_ns_by_chip=chip_service,
                  weight_format='int8_1_byte_per_weight', hardware_mac_lane_utilization='not_measured')
    assert 0 < totals['useful_bandwidth_fraction'] <= 1
    assert 0 < totals['active_chip_fraction'] <= 1
    assert 0 < totals['plane_slot_fill'] <= totals['physical_page_fill'] <= 1
    return totals


def write_csv(path, records):
    scalars = [{k:v for k,v in r.items() if not isinstance(v, list)} for r in records]
    with path.open('w', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=list(scalars[0]))
        writer.writeheader()
        writer.writerows(scalars)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--metadata', type=Path, required=True)
    parser.add_argument('--trace', type=Path, action='append', required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = json.loads(args.metadata.read_text())
    dims = tensor_dimensions(metadata)
    events = load_events(args.trace)
    architecture = metadata['metadata']['general.architecture']
    expert_count = metadata['metadata'][architecture + '.expert_count']
    for event in events:
        if any(e < 0 or e >= expert_count for e in event['expert_ids']):
            raise ValueError('Expert ID out of bounds')
    results = []
    for g in (1,2,4,8,16):
        for serial in (False,True):
            tag = f'g{g}_' + ('serial' if serial else 'ready')
            r = evaluate(events,dims,g,serial_experts=serial,
                         binary_path=args.output_dir/(tag+'.bin'))
            r['label'] = tag
            results.append(r)
            (args.output_dir/(tag+'.expected.json')).write_text(json.dumps(r,indent=2)+'\n')
    write_csv(args.output_dir/'placement.csv',results)
    stacked=[]
    for g in (1,2,4,8,16):
        tag=f'g{g}_stacked'
        r=evaluate(events,dims,g,stack_gate_up=True,binary_path=args.output_dir/(tag+'.bin'))
        r['label']=tag
        stacked.append(r)
        (args.output_dir/(tag+'.expected.json')).write_text(json.dumps(r,indent=2)+'\n')
    write_csv(args.output_dir/'stacked_placement.csv',stacked)
    sensitivity=[]
    for g in (1,2,4,8,16):
        for intermediate in (128,256,512,768,1024,1408,2048,4096):
            for cold in (1,2):
                r=evaluate(events,dims,g,intermediate_override=intermediate,cold_per_block=cold)
                r['label']='dimension_sweep_same_real_routes'
                sensitivity.append(r)
    write_csv(args.output_dir/'dimension_sensitivity.csv',sensitivity)
    other=[]
    for block in (192,256):
        for fuse in (False,True):
            for g in (1,2,4,8,16):
                r=evaluate(events,dims,g,block_pages=block,fuse_gate_up=fuse)
                r['label']='block_and_gateup_schedule_sensitivity'
                other.append(r)
    write_csv(args.output_dir/'scheduling_sensitivity.csv',other)
    manifest=dict(metadata=str(args.metadata.resolve()), traces=[str(p.resolve()) for p in args.trace],
                  trace_source='real_llama_cpp_decode', model_sha256=metadata['sha256'],
                  dimensions=dims, parameters=dict(chips=CHIPS,planes=PLANES,page_bytes=PAGE,
                    chip_bandwidth_bytes_per_second=CHIP_BW,cold_ns=COLD_NS,cr_ns=CR_NS,
                    default_block_lsb_pages=192,default_cold_reads_per_block=1),
                  limitations=['No MAC lane count available; no measured hardware PE duty cycle.',
                    'A chip processes one matrix stream at a time; cross-expert plane concurrency is not implemented.',
                    'Ready mode uses global projection barriers; it is not an optimal per-expert DAG scheduler.',
                    'Short streams are packed wholly within a block; long streams start aligned and split. Virtual plans verify offsets separately.',
                    'Aggregate chip bandwidth bound assumes homogeneous equal-sized projection streams in each phase.',
                    'Timing excludes command dispatch, input/output vector transfer, host activation, routing and attention.',
                    'No DRAM expert cache, GC, flash programming, quantization hardware or cross-layer overlap.',
                    'No claim of real AiF hardware or full end-to-end inference speed.',
                    'Dimension sensitivity reuses real expert IDs with hypothetical tensor dimensions.'])
    (args.output_dir/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps(results,indent=2))


if __name__ == '__main__':
    main()
