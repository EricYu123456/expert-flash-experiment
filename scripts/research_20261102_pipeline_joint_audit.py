#!/usr/bin/env python3
"""Seal combined replay, paired vectors and independent native trace checks."""
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from research_20261102_granite4_pipeline import SOURCE,EXPECTED
from research_20261102_pipeline_joint_replay import DEST,JOINT,SEALED,rows_from_labels,paired
from run_corpus_traces import ROOT,sha256


def trace_audit(folder,doc,joint):
    records=[]
    for path in sorted(folder.glob('layer*.trace.json')):
        d=json.loads(path.read_text());nodes=d['trace'];selected=d['selected']
        layer=int(path.name.split('.')[0][5:]);variant=path.name.split('.')[1].split('__')[0]
        ps=[p for p in doc['phases'] if p['layer']==layer];q=next(q for q in joint['results'] if q['layer']==layer)
        gu=np.asarray(ps[0]['rows']);down=np.asarray(ps[1]['rows'])
        if variant in ['matched_phase_best','joint_best']:down=rows_from_labels(q[variant]['labels'],ps[1]['total_rows'])
        elif variant=='g8_opposite':
            down=np.zeros_like(down)
            for e,r in enumerate(gu):down[e,8:]=ps[1]['total_rows']//8 if r[0]>0 else 0;down[e,:8]=ps[1]['total_rows']//8 if r[0]==0 else 0
        elif variant=='gu_wave_rotations':gu=np.asarray(next(c for c in ps[0]['candidates'] if c['label']=='wave_rotations')['rows'])
        assert np.all(gu.sum(axis=1)==ps[0]['total_rows']) and np.all(down.sum(axis=1)==ps[1]['total_rows'])
        resources={};jobs={};chip_intervals=[[] for _ in range(16)]
        for n in sorted(nodes,key=lambda n:(n['start_ns'],n['index'])):
            assert all(nodes[i]['end_ns']<=n['start_ns'] for i in n['deps'])
            if n['end_ns']>n['start_ns']:
                assert n['start_ns']>=resources.get(n['resource'],0);resources[n['resource']]=n['end_ns']
            if n['job']>=0:jobs.setdefault(n['job'],{})[n['kind']]=n
        for job in jobs.values():
            cmd=job['matrix_command'];inp=job['input_channel_reload'];comp=job['nonpreemptive_native_compute'];out=job['output_channel']
            chip=cmd['chip'];ei=cmd['expert'];phase=cmd['phase'];rows=[gu,down][phase];C=ps[phase]['columns'];r=int(rows[selected[ei],chip])
            wave=(((r*C+16383)//16384)+3)//4
            assert cmd['end_ns']-cmd['start_ns']==250
            assert inp['end_ns']-inp['start_ns']==(C+1)//2
            assert comp['end_ns']-comp['start_ns']==wave*28000 and 0<wave<=192
            assert out['end_ns']-out['start_ns']==r*2
            assert cmd['end_ns']<=inp['start_ns']<=inp['end_ns']<=comp['start_ns']<=comp['end_ns']<=out['start_ns']
            chip_intervals[chip].append((cmd['start_ns'],out['end_ns']))
            if phase:
                ready=next(n for n in nodes if n['kind']=='whole_expert_gu_host_ready' and n['expert']==ei)
                activation=next(n for n in nodes if n['kind']=='host_activation' and n['expert']==ei)
                gu_outputs=[n['end_ns'] for n in nodes if n['phase']==0 and n['expert']==ei and n['kind']=='output_pcie']
                assert max(gu_outputs)<=ready['end_ns']<=activation['start_ns']<=activation['end_ns']<=cmd['start_ns']
                submit=next(n for n in nodes if n['kind'] in ['expert_down_host_submission','batch_down_host_submission'] and n['expert'] in [ei,-1])
                assert submit['end_ns']-submit['start_ns']==2000 and submit['end_ns']<=cmd['start_ns']
        for intervals in chip_intervals:
            intervals.sort();assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
        records.append(dict(path=str(path.relative_to(ROOT)),sha256=sha256(path),native_jobs=len(jobs),nodes=len(nodes),status='PASS'))
    return records


def main():
    assert sha256(SOURCE)==EXPECTED;doc=json.loads(SOURCE.read_text());joint=json.loads(JOINT.read_text())
    studies={'zero':DEST/'zero','one-us':DEST/'one-us','opposite-full':DEST/'opposite-full',
        'opposite-one-us':DEST/'opposite-one-us','wave-full':ROOT/'results/research-2026-11-02/round2/pipeline-gu-screen/full',
        'wave-one-us':ROOT/'results/research-2026-11-02/round2/pipeline-gu-screen/one-us'}
    records={};vectors={};documents={};traces=[];artifacts={}
    for key,folder in studies.items():
        path=folder/'screen.json';d=json.loads(path.read_text());documents[key]=d
        assert d.get('source_plan_sha256',d.get('frozen_plan_sha256'))==EXPECTED
        with np.load(folder/'schedule_costs.npz') as f:
            np.testing.assert_array_equal(f['indices'],np.arange(2048));vectors[key]={k:f[k].copy() for k in f.files if k!='indices'}
        traces+=trace_audit(folder,doc,joint)
        records[key]=dict(screen_sha256=sha256(path),schedule_costs_sha256=sha256(folder/'schedule_costs.npz'))
        artifacts[str(path.relative_to(ROOT))]=sha256(path)
    tables={};deltas={};comparisons={};inventory={}
    for sensitivity,core,opp,wave in [('zero','zero','opposite-full','wave-full'),('one-us','one-us','opposite-one-us','wave-one-us')]:
        merged=dict(vectors[core]);merged.update({k:v for k,v in vectors[opp].items() if k.startswith('g8_opposite')})
        merged.update({k:v for k,v in vectors[wave].items() if k.startswith('gu_wave_rotations')})
        for other in [opp,wave]:
            for name,v in vectors[other].items():
                if name.startswith('source'):np.testing.assert_array_equal(v,vectors[core][name])
        table={};comp={}
        for suffix in ['__barrier__gate_first','__ready__gate_first']:
            totals={variant:merged[variant+suffix].sum(axis=0) for variant in ['source','matched_phase_best','joint_best','gu_wave_rotations','g8_opposite']}
            table[suffix]={v:dict(mean_ns=float(x.mean()),p95_ns=float(np.percentile(x,95)),p99_ns=float(np.percentile(x,99))) for v,x in totals.items()}
            simple=min(['source','matched_phase_best','gu_wave_rotations','g8_opposite'],key=lambda v:totals[v].mean())
            for variant in totals:
                if variant=='source':continue
                name=variant+'_vs_source'+suffix;comp[name]=paired(totals[variant],totals['source'])
                deltas[sensitivity+'__'+name]=totals[variant]-totals['source']
                deltas[sensitivity+'__layer__'+name]=merged[variant+suffix]-merged['source'+suffix]
            name='joint_vs_best_simple'+suffix;comp[name]=dict(best_simple_family=simple,**paired(totals['joint_best'],totals[simple]))
            deltas[sensitivity+'__'+name]=totals['joint_best']-totals[simple]
            name='joint_vs_matched_phase_best'+suffix;comp[name]=paired(totals['joint_best'],totals['matched_phase_best'])
            deltas[sensitivity+'__'+name]=totals['joint_best']-totals['matched_phase_best']
        tables[sensitivity]=table;comparisons[sensitivity]=comp
    for study in ['zero','opposite-full','wave-full']:
        for variant,value in documents[study]['physical_inventory'].items():
            if variant=='source' and variant in inventory:continue
            chips=value['chips'];inventory[variant]=dict(total_useful_bytes=value['total_useful_bytes'],total_physical_pages=value['total_physical_pages'],
                total_stored_read_waves=value['total_stored_read_waves'],minimum_useful_bytes_per_chip=min(c['useful_int8_bytes'] for c in chips),
                maximum_useful_bytes_per_chip=max(c['useful_int8_bytes'] for c in chips),
                minimum_required_lsb_bytes_per_chip=min(c['reserved_lsb_bytes'] for c in chips),
                maximum_required_lsb_bytes_per_chip=max(c['reserved_lsb_bytes'] for c in chips),
                required_lsb_bytes_per_chip=[c['reserved_lsb_bytes'] for c in chips])
    delta_path=DEST/'paired_deltas.npz';np.savez_compressed(delta_path,**deltas)
    result=dict(status='complete train-only combined replay; no heldout',created_utc=datetime.now(timezone.utc).isoformat(),
        frozen_plan_sha256=EXPECTED,root_joint_source_sha256=sha256(JOINT),tables=tables,paired_comparisons=comparisons,
        physical_inventory=inventory,study_hashes=records,representative_native_trace_audits=traces,
        paired_deltas_sha256=sha256(delta_path),source_sha256=sha256(Path(__file__)),
        interpretation=['All cases use the sealed strict scheduler, full input reload and one native context per chip. Representative traces independently checked; every new schedule has internal resource/dependency/job-ownership audit.',
            'Rows cover each matrix once; no replicated weights, changed routing, new input context or new operators. Wave-rotation padding increases physical pages and wave work and is listed explicitly.',
            'Per-chip physical LSB inventory is the deterministic whole-model WaveAllocator realization, not a certified storage-minimum allocator. No invented capacity cap is imposed; ECC/spares/unrelated tensors remain additional.',
            'Numbers are first3layer routed-FFN service simulations. Necessary NAND bounds are not reported as achieved latency. Activation/combine are synthetic sensitivity values, not hardware calibration.',
            'Best-simple family is one static whole-layout family selected by training mean, not a per-query switch between physical plans.',
            'Paired delta arrays are indexed by original train prompt-major64steps; layer arrays preserve first3layers. Negative delta means candidate faster.'])
    target=DEST/'combined_audit.json';target.write_text(json.dumps(result,indent=2)+'\n')
    for path in [target,delta_path,Path(__file__),Path(__file__).with_name('research_20261102_pipeline_joint_replay.py'),Path(__file__).with_name('research_20261102_pipeline_gu_screen.py')]:artifacts[str(path.relative_to(ROOT))]=sha256(path)
    seal=dict(status='sealed exploratory train-only replays; no heldout',created_utc=datetime.now(timezone.utc).isoformat(),artifacts=artifacts,
        frozen_plan_sha256=EXPECTED,root_joint_source_sha256=sha256(JOINT),old_strict_runner_sha256=sha256(ROOT/'scripts/research_20261102_granite4_pipeline.py'))
    (DEST/'seal.json').write_text(json.dumps(seal,indent=2)+'\n')
    print(json.dumps(dict(tables=tables,paired_comparisons=comparisons,physical_inventory=inventory),indent=2))


if __name__=='__main__':main()
