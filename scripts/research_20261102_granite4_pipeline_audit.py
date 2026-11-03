#!/usr/bin/env python3
"""Independent dependency/native-context audit for Granite4 pipeline traces."""
from datetime import datetime, timezone
import json
from pathlib import Path

import numpy as np

from research_20261102_granite4_pipeline import DEST, SOURCE, EXPECTED
from run_corpus_traces import sha256


def main():
    screen=DEST/'screen.json';d=json.loads(screen.read_text());assert sha256(SOURCE)==EXPECTED
    doc=json.loads(SOURCE.read_text());files=sorted(DEST.glob('layer*.trace.json'));assert len(files)==24
    audits=[]
    for path in files:
        tr=json.loads(path.read_text());nodes=tr['trace'];selected=tr['selected']
        layer=int(path.name.split('.')[0][5:]);placement='frozen_g12' if 'frozen_g12' in path.name else 'frozen_phase_group'
        key='rows' if placement=='frozen_g12' else 'group_rows'
        phases=[p for p in doc['phases'] if p['layer']==layer]
        end_by_resource={};chip_intervals={c:[] for c in range(16)};jobs={}
        for n in sorted(nodes,key=lambda n:(n['start_ns'],n['index'])):
            assert all(nodes[i]['end_ns']<=n['start_ns'] for i in n['deps'])
            if n['end_ns']>n['start_ns']:
                assert n['start_ns']>=end_by_resource.get(n['resource'],0)
                end_by_resource[n['resource']]=n['end_ns']
            if n['job']>=0:jobs.setdefault(n['job'],{})[n['kind']]=n
        for job in jobs.values():
            cmd=job['matrix_command'];inp=job['input_channel_reload'];comp=job['nonpreemptive_native_compute'];out=job['output_channel']
            c=cmd['chip'];ei=cmd['expert'];phase=cmd['phase'];p=phases[phase];r=p[key][selected[ei]][c]
            assert r>0
            wave=((r*p['columns']+16383)//16384+3)//4
            assert cmd['end_ns']-cmd['start_ns']==250
            assert inp['end_ns']-inp['start_ns']==(p['columns']+1)//2
            assert comp['end_ns']-comp['start_ns']==28000*wave and wave<=192
            assert out['end_ns']-out['start_ns']==r*2
            assert cmd['end_ns']<=inp['start_ns']<=inp['end_ns']<=comp['start_ns']<=comp['end_ns']<=out['start_ns']
            chip_intervals[c].append((cmd['start_ns'],out['end_ns']))
            if phase==1:
                ready=next(n for n in nodes if n['kind']=='whole_expert_gu_host_ready' and n['expert']==ei)
                act=next(n for n in nodes if n['kind']=='host_activation' and n['expert']==ei)
                gu_outputs=[n for n in nodes if n['phase']==0 and n['expert']==ei and n['kind']=='output_pcie']
                assert max(n['end_ns'] for n in gu_outputs)<=ready['end_ns']<=act['start_ns']
                assert act['end_ns']<=cmd['start_ns']
                submits=[n for n in nodes if n['kind'] in ['expert_down_host_submission','batch_down_host_submission'] and (n['expert']==ei or n['expert']==-1)]
                assert len(submits)==1 and submits[0]['end_ns']<=cmd['start_ns']
                assert submits[0]['end_ns']-submits[0]['start_ns']==2000
        for intervals in chip_intervals.values():
            intervals.sort();assert all(b<=c for (_,b),(c,_) in zip(intervals,intervals[1:]))
        audits.append(dict(trace=path.name,sha256=sha256(path),nodes=len(nodes),native_jobs=len(jobs),
            intervals_dependencies_single_context_reload_whole_expert_ready='PASS'))
    contrasts={}
    for priority in ['down_first','gate_first']:
        t=d['total_schedules'];gb=t['frozen_phase_group__barrier__'+priority]['mean_ns'];gr=t['frozen_phase_group__ready__'+priority]['mean_ns']
        nb=t['frozen_g12__barrier__'+priority]['mean_ns'];nr=t['frozen_g12__ready__'+priority]['mean_ns']
        contrasts[priority]=dict(placement_gain_barrier_pct=100*(1-nb/gb),placement_gain_ready_pct=100*(1-nr/gr),
            group_scheduler_gain_ns=gb-gr,g12_scheduler_gain_ns=nb-nr,
            group_scheduler_gain_pct=100*(1-gr/gb),g12_scheduler_gain_pct=100*(1-nr/nb),
            differential_pipeline_saving_ns=(nb-nr)-(gb-gr),differential_pipeline_saving_over_group_barrier_pct=100*((nb-nr)-(gb-gr))/gb)
    native_bounds=[]
    route=Path(d['training_routes']);assert sha256(route)==d['training_routes_sha256']
    with np.load(route) as f:routes=f['routes'][:32,:64];layers=f['layers'].tolist()
    for lr in d['layer_results']:
        layer=lr['layer'];selected=routes[:,:,layers.index(layer)].reshape(-1,doc['active'])
        ps=[p for p in doc['phases'] if p['layer']==layer]
        for placement,key in [('frozen_phase_group','group_rows'),('frozen_g12','rows')]:
            work=np.zeros((len(selected),16),dtype=np.int64)
            for p in ps:
                rows=np.asarray(p[key]);waves=(((rows*p['columns']+16383)//16384)+3)//4
                per_job=np.where(rows>0,waves*28000+250+(p['columns']+1)//2+rows*2,0)
                work+=per_job[selected].sum(axis=1)
            lb=2000+work.max(axis=1)
            native_bounds.append(dict(layer=layer,placement=placement,mean_native_chip_occupancy_lower_bound_ns=float(lb.mean()),
                definition='2us initial launch + max chip sum(native job command250ns + input reload + NAND compute + channel output). Ignores dependency, resource contention and final PCIe drain; only a lower bound.'))
    out=dict(status='PASS',created_utc=datetime.now(timezone.utc).isoformat(),screen_sha256=sha256(screen),
        frozen_plan_sha256=EXPECTED,representative_trace_audits=audits,
        all_schedule_runtime_audit='Every evaluated schedule also checks all resource intervals, dependencies and complete job-ownership intervals inside Graph.run.',
        attribution=contrasts,native_chip_occupancy_lower_bounds=native_bounds,
        limitations=['Trace files cover first scheduled route of each case/layer; internal runtime audit covers all schedules.',
            'No physical timing or global scheduling optimality claimed; activation/combine are exposed parameters.',
            'Pooled14waves relaxes fixed chip ownership and is not a reachable-placement certificate.'],source_sha256=sha256(Path(__file__)))
    target=DEST/'audit.json';target.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(contrasts,indent=2))


if __name__=='__main__':main()
