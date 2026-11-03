#!/usr/bin/env python3
"""Independent scalar arithmetic and artifact audit for the v2 study.

The reference reconstructs every selected expert/chip/bin from its slot and
does not call the new vector cost function or the old aggregate scheduler.
"""
import argparse
import json
from pathlib import Path

import numpy as np

import aif_model as legacy
import joint_plane_search as joint
from run_corpus_traces import ROOT, sha256, write_json

KEYS = ('flash_ns','control_ns','host_ns','input_ns','output_ns','model_ns',
        'useful_bytes','physical_pages','read_waves','cold_reads','matrix_commands',
        'nand_read_commands','plane_read_commands','ordinary_read_epochs')


def ceildiv(value, divisor):
    return (value + divisor - 1) // divisor


def scalar_event(experts, slots, dimensions, g, width, use_cr):
    totals = {key:0 for key in KEYS}
    group_count, bins = 16 // g, 4 // width
    gate = dimensions['gate']; up = dimensions['up']; down = dimensions['down']
    for rows, columns, common in ((gate[0]+up[0],up[1],True),(*down,False)):
        chips = [dict(nand=[0]*bins,waves=[0]*bins,cold=[0]*bins,bytes=0,pages=0,
                      inputs=0,outputs=0,commands=0,nand_commands=0) for _ in range(16)]
        for expert in experts:
            slot = int(slots[expert]); group = slot % group_count; lane = (slot//group_count) % bins
            for shard in range(g):
                nrows = rows//g + int(shard < rows%g)
                if nrows == 0:
                    continue
                state = chips[group*g+shard]
                nbytes = nrows*columns; pages = ceildiv(nbytes,16384)
                waves = ceildiv(pages,width); blocks = ceildiv(waves,192)
                state['nand'][lane] += waves*9700+blocks*18300 if use_cr else waves*28000
                state['waves'][lane] += waves; state['cold'][lane] += blocks
                state['bytes'] += nbytes; state['pages'] += pages
                state['inputs'] = columns if common else state['inputs']+columns
                state['outputs'] += nrows*4; state['commands'] += blocks
                state['nand_commands'] += waves
        flash = max(max(max(c['nand']),c['pages']*2560) for c in chips)
        host_in = columns*(1 if common else len(experts))
        channel_in = [chips[c]['inputs']+chips[c+8]['inputs'] for c in range(8)]
        channel_out = [chips[c]['outputs']+chips[c+8]['outputs'] for c in range(8)]
        totals['flash_ns'] += flash
        totals['input_ns'] += max(ceildiv(host_in,8),ceildiv(max(channel_in),2))
        totals['output_ns'] += max(ceildiv(sum(c['outputs'] for c in chips),8),ceildiv(max(channel_out),2))
        totals['control_ns'] += 250*sum(c['commands'] for c in chips)
        totals['host_ns'] += 2000
        totals['useful_bytes'] += sum(c['bytes'] for c in chips)
        totals['physical_pages'] += sum(c['pages'] for c in chips)
        totals['read_waves'] += sum(max(c['waves']) for c in chips)
        totals['cold_reads'] += sum(c['commands'] for c in chips)
        totals['matrix_commands'] += sum(c['commands'] for c in chips)
        totals['nand_read_commands'] += sum(c['nand_commands'] for c in chips)
        totals['plane_read_commands'] += sum(c['pages'] for c in chips)
        totals['ordinary_read_epochs'] += sum(max(c['cold'] if use_cr else c['waves']) for c in chips)
    totals['model_ns'] = sum(totals[k] for k in KEYS[:5])
    return totals


def arithmetic_audit():
    rng = np.random.default_rng(20261101)
    dimensions = [dict(gate=(257,513),up=(257,513),down=(513,257)),
                  dict(gate=(1408,2048),up=(1408,2048),down=(2048,1408)),
                  dict(gate=(14336,4096),up=(14336,4096),down=(4096,14336))]
    checks = 0
    for dims in dimensions:
        slots = rng.permutation(16).tolist()
        routes = np.stack([rng.choice(16,6,replace=False) for _ in range(13)])
        for g in (1,2,4,8,16):
            for width in (4,2,1):
                counts = joint.counts_for(routes,slots,g,width)
                both = []
                for use_cr in (True,False):
                    profiles = joint.phase_profiles(dims,g,width,use_cr)
                    fast = joint.costs_from_counts(counts,profiles,g,width,6)
                    for i,experts in enumerate(routes):
                        expected = scalar_event(experts,slots,dims,g,width,use_cr)
                        for key in KEYS:
                            assert int(fast[key][i]) == expected[key],(g,width,use_cr,i,key,fast[key][i],expected[key])
                        checks += 1
                    full = joint.model_ns_from_counts(counts,profiles,g,width,6)
                    np.testing.assert_array_equal(full,fast['model_ns'])
                    both.append(fast)
                for key in ('control_ns','host_ns','input_ns','output_ns','useful_bytes',
                            'physical_pages','read_waves','cold_reads','matrix_commands',
                            'nand_read_commands','plane_read_commands'):
                    np.testing.assert_array_equal(both[0][key],both[1][key])
                assert np.all(both[1]['flash_ns'] >= both[0]['flash_ns'])
    # With CR on and width four, the new NAND formula is the legacy one.
    for pages in (1,2,4,5,768,769,2048):
        demand = legacy.Demand(); demand.add_stream(pages*16384,192,1)
        waves=ceildiv(pages,4); cold=ceildiv(waves,192)
        assert demand.service_ns == max(waves*9700+cold*18300,pages*2560)
    return dict(independent_event_cases=checks,components=list(KEYS),
                fast_objective_equal_to_full_costs=True,
                read_mode_changes_only_nand_and_derived_latency=True)


def artifact_audit(results):
    """Check saved hashes, capacities and training-selected winners, if present."""
    rows=[]
    for folder in ('joint-cr','no-cr'):
        location=results/folder
        if not location.exists():
            continue
        for path in sorted(location.glob('*.frozen_plan.json')):
            plan=json.loads(path.read_text())
            rows.append(dict(path=str(path),sha256=sha256(path),keys=list(plan)))
    return rows


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results/plane-wave-v2')
    args=parser.parse_args();args.results.mkdir(exist_ok=True)
    result=dict(arithmetic=arithmetic_audit(),artifacts=artifact_audit(args.results),
                source_sha256={n:sha256(ROOT/'scripts'/n) for n in ('audit_joint_no_cr.py','joint_plane_search.py')})
    write_json(args.results/'arithmetic_audit.json',result)
    print(json.dumps(result['arithmetic'],indent=2))


if __name__ == '__main__':
    main()
