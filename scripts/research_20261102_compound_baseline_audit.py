#!/usr/bin/env python3
"""Independently enumerate whole-profile choices; retain one-copy controls."""
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from research_20261102_compound_baselines import DEST,MAIN,THM,ORACLE_COST,python_profile,summary
from research_20261102_projection_study import evaluate,scalar_cost
from run_corpus_traces import ROOT,sha256


def main():
    completion=DEST/'completion.json';receipt=sha256(completion);doc=json.loads(completion.read_text())
    main_plan=json.loads(MAIN.read_text());theorem=json.loads(THM.read_text())
    assert all(sha256(Path(p))==digest for p,digest in doc['input_hashes'].items())
    route_path=Path(doc['training_routes']);assert sha256(route_path)==doc['training_routes_sha256']
    with np.load(route_path) as f:
        routes=f['routes'][:32,:64];ids=f['prompt_ids'][:32].tolist();layers=f['layers'].tolist()
    assert ids==doc['train_ids']
    costpath=Path(doc['cost_archive']);assert sha256(costpath)==doc['cost_archive_sha256']
    costs=np.load(costpath);controls=[];series={k:np.zeros(32*64,dtype=np.int64) for k in ['one_copy_group','no_replica_g12','whole_copy','compound','generic_flow']}
    strong_path=ROOT/'results/research-2026-11-02/round2/unbalanced-group/granite4tiny.screen.strong.frozen.json'
    strong=json.loads(strong_path.read_text());gu=np.zeros(32*64,dtype=np.int64);enumerations=0
    for result in doc['results']:
        layer=result['layer'];li=layers.index(layer);selected=np.sort(routes[:,:,li].reshape(-1,6),axis=1)
        phase=next(p for p in main_plan['phases'] if p['layer']==layer and p['name']=='down')
        group=np.asarray(phase['group_rows']);rows=np.asarray(phase['rows'])
        one_copy=evaluate(selected,group,512,False)['model_ns'];no_rep=evaluate(selected,rows,512,False)['model_ns']
        np.testing.assert_array_equal(no_rep,costs['layer'+str(layer)+'_no_replica'])
        base=np.asarray(result['base_omitted_color']);alt=np.asarray(result['secondary_omitted_color'])
        assert np.count_nonzero(alt>=0)==57 and np.all(alt[alt>=0]!=base[alt>=0])
        unique,inverse=np.unique(selected,axis=0,return_inverse=True)
        independent=np.asarray([python_profile(q,base,alt) for q in unique],dtype=np.int64)[inverse]
        whole=costs['layer'+str(layer)+'_whole'];np.testing.assert_array_equal(independent,whole);enumerations+=len(unique)
        generic=costs['layer'+str(layer)+'_generic_flow'];assert np.all(generic==ORACLE_COST)
        for sample in result['flow_assignment_samples']:
            seen=set();load=np.zeros(16,dtype=int);coverage={e:set() for e in sample['query']}
            for e,t,c in sample['jobs']:
                assert c in theorem['tiles'][e*12+t]['locations']
                assert (e,c) not in seen;seen.add((e,c));load[c]+=1;coverage[e].add(t)
            assert all(t==set(range(12)) for t in coverage.values())
            assert load.max()==5 and np.all(load[:8]+load[8:]==9)
            assert load.tolist()==sample['chip_loads']
        for index in [0,1024,2047]:
            assert scalar_cost(selected[index],group,512,False)['model_ns']==one_copy[index]
            assert scalar_cost(selected[index],rows,512,False)['model_ns']==no_rep[index]
        gu_phase=next(p for p in strong['phases'] if p['layer']==layer and p['name']=='gate_up')
        gu+=evaluate(selected,np.asarray(gu_phase['chosen_rows']),gu_phase['columns'],True)['model_ns']
        for name,array in [('one_copy_group',one_copy),('no_replica_g12',no_rep),('whole_copy',whole),('compound',generic),('generic_flow',generic)]:series[name]+=array
        controls.append(dict(layer=layer,one_copy_group=summary(one_copy),no_replica_g12=summary(no_rep),
            whole_copy=summary(whole),compound=summary(generic),generic_RDA_flow=summary(generic),
            unique_queries_independently_enumerated=len(unique)))
    aggregate={name:summary(array+gu) for name,array in series.items()}
    mean_gap=aggregate['whole_copy']['mean_ns']-aggregate['compound']['mean_ns']
    assert mean_gap==.875 and enumerations==4221
    assert sha256(completion)==receipt
    output=dict(status='passed',audit_utc=datetime.now(timezone.utc).isoformat(),
        completion_sha256=receipt,cost_archive_sha256=sha256(costpath),layers=[0,1,2],
        controls=controls,three_layer_routed_FFN_including_identical_unreplicated_GU=aggregate,
        whole_copy_minus_compound=dict(mean_ns=mean_gap,p99_ns=aggregate['whole_copy']['p99_ns']-aggregate['compound']['p99_ns']),
        all4221_unique_whole_profile_choices_independently_enumerated=True,
        generic_flow_all4221_unique_queries_checked_in_collector=True,
        generic_flow_sample_row_coverage_location_and_native_resource_checks_passed=True,
        one_copy_controls_independent_scalar_samples_passed=True,
        capacities=dict(extra_down_tiles_budget=688,whole_copy_extra_tiles_used=684,whole_copy_remainder_tiles=4,
            generic_and_compound_extra_tiles_used=688,GU_copies=0,
            down_payload_overhead_pct=100*688/768,whole_routed_FFN_payload_overhead_pct=100*688/(768*3)),
        additional_input_hashes={str(strong_path):sha256(strong_path)},source_sha256=sha256(Path(__file__)),
        decision='No mean or p99 performance headroom versus generic replicated declustering; whole-copy baseline already matches p99 and has only0.875ns aggregate mean gap on this screen. No expansion or holdout is justified for a performance claim.',
        limitations=['Three layers and training routes only; does not certify whole-copy worst-case behavior or unseen queries.',
            'Full-copy membership57hottest is not globally optimized; a stronger whole-copy search can only reduce its already negligible gap.',
            'The compound combinatorial storage certificate can remain mathematically valid without creating a new performance method relative to its generic flow superset.',
            'Service model only; selector runtime, hardware PE/NAND, full-model execution and quality were not measured.'])
    path=DEST/'audit.json';path.write_text(json.dumps(output,indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__':main()
