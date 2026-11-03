#!/usr/bin/env python3
"""Read-only checks of v2 freezes, train selection, saved costs and NAND slots."""
import argparse
from collections import Counter
from pathlib import Path
import json

import numpy as np

from audit_joint_no_cr import scalar_event
import aif_model as flash
import joint_plane_search as joint
import plane_wave_model as plane
from run_corpus_traces import DATA, MODELS, ROOT, geometry, sha256, write_json


def selected_specs(plan, cr):
    out={}
    for policy, spec in plan['winners'].items():
        g,w=spec['group_chips'],spec['plane_width']
        if cr:
            maps=plan['mappings'][spec['family']][f'g{g}'][f'w{w}']
        elif spec['key']=='baseline':
            first=next(iter(plan['mappings'].values()))
            maps={layer:list(range(len(values))) for layer,values in first.items()}
        else:
            maps=plan['mappings'][spec['key']]
        out[policy]=(g,w,maps)
    return out


def training_population(policy, candidates, cr):
    """Independently state each reported policy's eligible training cells."""
    if cr:
        families={
            'baseline':'modulo','modulo':'modulo','group_only':'warm_start',
            'historical_staged':'warm_start','historical_forced_plane':'warm_start',
            'staged_equal':'staged_equal','staged_equal_forced_plane':'staged_equal',
        }
        family=families.get(policy,'joint')
        population=[c for c in candidates if c['family']==family]
        if policy=='baseline':
            return [c for c in population if c['group_chips']==16 and c['plane_width']==4]
        if policy=='group_only' or policy=='joint_width4':
            return [c for c in population if c['plane_width']==4]
        if policy in ('joint_width1','joint_width2'):
            return [c for c in population if c['plane_width']==int(policy[-1])]
        if policy=='plane_only':
            return [c for c in population if c['group_chips']==16 and c['plane_width']<4]
        if policy.endswith('forced_plane'):
            return [c for c in population if c['plane_width']<4]
        return population
    if policy=='baseline':
        return []
    family=policy.removesuffix('_forced_plane')
    if policy.startswith('plane_only_g16'):
        family='joint_group_plane'
    population=[c for c in candidates if c['family']==family]
    if policy.startswith('plane_only_g16'):
        population=[c for c in population if c['group_chips']==16]
    if policy.endswith('forced_plane'):
        population=[c for c in population if c['plane_width']<4]
    return population


def capacity_and_intervals(dimensions, maps, g, width, expert_count):
    """One monotone allocator per physical chip/lane for the ENTIRE model."""
    chip_bytes=[{} for _ in range(16)];chip_masks=[{} for _ in range(16)]
    expected=0;logical_shards=0
    for li,layer in enumerate(sorted(dimensions)):
        groups,bins=joint.decode_slots(maps[str(layer)],g,width)
        dims=dimensions[layer]
        projections=[(dims['gate'][0]+dims['up'][0],dims['up'][1]),dims['down']]
        for pi,(rows,cols) in enumerate(projections):
            expected+=expert_count*rows*cols
            assert sum(rows//g+int(s<rows%g) for s in range(g))==rows
            for expert in range(expert_count):
                key=(li*2+pi)*expert_count+expert
                mask=tuple(range(int(bins[expert])*width,(int(bins[expert])+1)*width))
                for shard in range(g):
                    count=rows//g+int(shard<rows%g)
                    if not count: continue
                    chip=int(groups[expert])*g+shard
                    assert key not in chip_bytes[chip]
                    chip_bytes[chip][key]=count*cols
                    chip_masks[chip][key]=mask
                    logical_shards+=1
    totals={}; interval_count=0
    for chip in range(16):
        layout=plane.build_layout(chip_bytes[chip],policy={4:'baseline',2:'pair',1:'single'}[width],masks=chip_masks[chip])
        summary=layout.summary()
        intervals=[[] for _ in range(4)]
        for shard in layout.shards.values():
            assert len(shard.segments)==(shard.waves+191)//192
            for segment in shard.segments:
                for offset,physical_plane in enumerate(shard.mask):
                    count=(segment.pages+width-1-offset)//width
                    if count:
                        start=segment.block*192+segment.wordline
                        intervals[physical_plane].append((start,start+count))
        checked=0
        for entries in intervals:
            entries.sort()
            assert all(a[1]<=b[0] for a,b in zip(entries,entries[1:])),chip
            checked+=sum(end-start for start,end in entries)
            interval_count+=len(entries)
        assert checked==summary['physical_pages']
        for key in ('useful_bytes','physical_pages','physical_page_bytes','reserved_lsb_bytes',
                    'intra_page_padding_bytes','stream_tail_slots','boundary_padding_slots','final_block_tail_slots'):
            totals[key]=totals.get(key,0)+summary[key]
    assert totals['useful_bytes']==expected
    return dict(totals=totals,logical_whole_row_shards=logical_shards,intervals=interval_count,
                all_physical_pages_nonoverlapping=True,whole_rows_exactly_once=True,
                weight_replication=False,scope='Virtual metadata only; independent plane addressing required for width<4.')


def all_mappings(plan,cr):
    if cr:
        for family,groups in plan['mappings'].items():
            for gtag,widths in groups.items():
                for wtag,maps in widths.items():
                    yield family,int(gtag[1:]),int(wtag[1:]),maps
    else:
        for spec in plan['training_candidates']:
            yield spec['family'],spec['group_chips'],spec['plane_width'],plan['mappings'][spec['key']]


def audit_model(folder,name,cr):
    plan_path=folder/f'{name}.frozen_plan.json'
    plan=json.loads(plan_path.read_text())
    validation=json.loads((folder/f'{name}.validation.json').read_text())
    assert validation['frozen_plan_sha256']==sha256(plan_path)
    assert not plan['test_routes_loaded_before_freeze'] and not plan['test_routes_used_for_selection']
    for source,h in plan.get('software_sha256',plan.get('source_sha256',{})).items():
        assert sha256(ROOT/'scripts'/source)==h,(folder,name,source)
    analysis=DATA/'runs/corpus-300-v1'/name/'analysis'
    with np.load(analysis/'test_routes.npz',allow_pickle=False) as cache:
        routes,ids,layers=cache['routes'],cache['prompt_ids'].tolist(),cache['layers'].tolist()
    assert set(ids).isdisjoint(plan['training_prompt_ids'])
    assert len(ids)==100 and len(plan['training_prompt_ids'])==200
    assert sha256(analysis/'train_routes.npz')==plan['training_cache']['sha256']
    assert sha256(analysis/'test_routes.npz')==validation['test_cache']['sha256']
    metadata=json.loads((DATA/'models'/f'{name}-metadata.json').read_text())
    dims=flash.tensor_dimensions(metadata);_,experts,_=geometry(metadata)
    capacity_checks=0
    for family,g,w,maps in all_mappings(plan,cr):
        template_groups,template_bins=joint.decode_slots(list(range(experts)),g,w)
        template=Counter(zip(template_groups.tolist(),template_bins.tolist()))
        for layer,slots in maps.items():
            assert sorted(slots)==list(range(experts)),(name,family,g,w,layer)
            groups,bins=joint.decode_slots(slots,g,w)
            assert Counter(zip(groups.tolist(),bins.tolist()))==template
            capacity_checks+=1
    chosen=selected_specs(plan,cr)
    summary=[r for r in json.loads((folder/'summary.json').read_text()) if r['model']==name]
    token=np.load(folder/f'{name}.token_costs.npz',allow_pickle=False)
    assert token['prompt_ids'].tolist()==ids
    scalar_checks=0
    arrays_checked=0
    for policy,(g,w,maps) in chosen.items():
        # Directly reconstruct full-token baseline from paper constants.
        selected_events=[(0,0),(1,31),(99,127)] if policy!='baseline' else [(0,0),(0,127),(99,127)]
        for pi,ti in selected_events:
            expected={key:0 for key in ('flash_ns','control_ns','host_ns','input_ns','output_ns','model_ns')}
            for li,layer in enumerate(layers):
                values=scalar_event(routes[pi,ti,li],maps[str(layer)],dims[layer],g,w,cr)
                for key in expected:expected[key]+=values[key]
            for key,value in expected.items():
                assert token[f'{policy}__{key}'][pi,ti]==value,(name,policy,key,pi,ti)
            scalar_checks+=1
        for row in (r for r in summary if r['policy']==policy):
            window=row['window'];value=token[f'{policy}__model_ns'][:,:window].mean()/1e6
            assert abs(value-row['mean_ms_per_token'])<1e-10,(name,policy,window)
        components=sum(token[f'{policy}__{key}'] for key in
                       ('flash_ns','control_ns','host_ns','input_ns','output_ns'))
        np.testing.assert_array_equal(components,token[f'{policy}__model_ns'])
        np.testing.assert_array_equal(token[f'{policy}__useful_bytes'],token['baseline__useful_bytes'])
        if policy=='baseline':
            for key in ('flash_ns','control_ns','host_ns','input_ns','output_ns','model_ns'):
                assert (token[f'baseline__{key}']==token[f'baseline__{key}'][0,0]).all(),(name,key)
        arrays_checked+=1
    # Check the selected slot map came from training-only winner candidates.
    for policy,winner in plan['winners'].items():
        if not cr and winner['key']=='baseline':continue
        matches=[c for c in plan['training_candidates'] if c['group_chips']==winner['group_chips']
                 and c['plane_width']==winner['plane_width'] and c['family']==winner['family']]
        assert len(matches)==1 and abs(matches[0]['mean_ms_per_token']-winner['mean_ms_per_token'])<1e-12
        population=training_population(policy,plan['training_candidates'],cr)
        minimum=min(population,key=lambda c:(c['mean_ms_per_token'],-c['plane_width'],c['group_chips']))
        assert (minimum['family'],minimum['group_chips'],minimum['plane_width'])==(
            winner['family'],winner['group_chips'],winner['plane_width']),(name,policy)
    physical={}
    physical_policies=('joint','joint_forced_plane','staged_equal') if cr else ('joint_group_plane','joint_group_plane_forced_plane','staged_group_then_plane')
    for policy in physical_policies:
        g,w,maps=chosen[policy]
        physical[policy]=dict(g=g,width=w,**capacity_and_intervals(dims,maps,g,w,experts))
    token.close()
    return dict(model=name,use_cr=cr,frozen_hash=sha256(plan_path),
                source_and_cache_hashes_match=True,training_test_disjoint=True,
                fixed_group_bin_capacity_checks=capacity_checks,scalar_selected_token_cases=scalar_checks,
                selected_cost_arrays_checked=arrays_checked,summary_matches_saved_token_costs=True,
                selected_maps_are_training_frozen=True,training_minima_recomputed=True,
                all_saved_cost_components_sum_exactly=True,baseline_all_tokens_equal_scalar=True,
                physical_layouts=physical)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--results',type=Path,default=ROOT/'results/plane-wave-v2')
    args=parser.parse_args()
    rows=[]
    for subfolder,cr in (('joint-cr',True),('no-cr',False)):
        for name in MODELS:
            rows.append(audit_model(args.results/subfolder,name,cr))
            print(subfolder,name,'artifact, scalar and whole-model address audit passed',flush=True)
    write_json(args.results/'artifact_audit.json',dict(models=rows,
        source_sha256={name:sha256(ROOT/'scripts'/name) for name in
                       ('audit_joint_study_artifacts.py','audit_joint_no_cr.py','joint_plane_search.py','plane_wave_model.py')}))


if __name__ == '__main__':
    main()
