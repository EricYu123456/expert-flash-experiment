#!/usr/bin/env python3
"""Independent full-profile trace audit and seven-family paired-data seal."""
from datetime import datetime,timezone
import json
from pathlib import Path

import numpy as np

from research_20261102_granite4_pipeline import SOURCE,EXPECTED
from research_20261102_joint_depth_full import INPUT,DEST,SEALED
from research_20261102_pipeline_joint_replay import paired
from run_corpus_traces import ROOT,sha256


def main():
    assert sha256(SOURCE)==EXPECTED;root=json.loads(INPUT.read_text());oldseal=json.loads((SEALED/'seal.json').read_text())
    for rel,h in oldseal['artifacts'].items():assert sha256(ROOT/rel)==h
    tables={};comparisons={};trace_records=[];delta_arrays={};artifacts={}
    for case in ['full-zero','full-one-us']:
        folder=DEST/case;path=folder/'screen.json';d=json.loads(path.read_text())
        assert d['root_source_sha256']==sha256(INPUT)
        with np.load(folder/'schedule_costs.npz') as f:
            np.testing.assert_array_equal(f['indices'],np.arange(2048));vectors={k:f[k].copy() for k in f.files if k!='indices'}
        tables[case]=d['total_schedules'];comparisons[case]=d['paired_comparisons']
        for suffix in ['__barrier__gate_first','__ready__gate_first']:
            v=vectors['joined_fit'+suffix].sum(axis=0)
            for ref in ['phase_fit','source','matched_g12','joined_g12','wave_rotations','g8_opposite']:
                delta_arrays[case+'__joined_fit_vs_'+ref+suffix]=v-vectors[ref+suffix].sum(axis=0)
        files=sorted(folder.glob('layer*.trace.json'));assert len(files)==12
        for trace in files:
            t=json.loads(trace.read_text());nodes=t['trace'];selected=t['selected'];layer=int(trace.name.split('.')[0][5:])
            variant=trace.name.split('.')[1].split('__')[0];key='barrier' if variant=='phase_fit' else 'joined'
            q=next(q for q in root['results'] if q['layer']==layer)[key]
            rows=[np.asarray(q['gu_rows']),np.asarray(q['down_rows'])]
            assert np.all(rows[0].sum(axis=1)==1024) and np.all(rows[1].sum(axis=1)==1536)
            resource_end={};jobs={};chip_intervals=[[] for _ in range(16)]
            for n in sorted(nodes,key=lambda n:(n['start_ns'],n['index'])):
                assert all(nodes[i]['end_ns']<=n['start_ns'] for i in n['deps'])
                if n['end_ns']>n['start_ns']:
                    assert n['start_ns']>=resource_end.get(n['resource'],0);resource_end[n['resource']]=n['end_ns']
                if n['job']>=0:jobs.setdefault(n['job'],{})[n['kind']]=n
                if n['kind']=='host_activation':assert n['end_ns']-n['start_ns']==d['activation_per_expert_ns']
                if n['kind']=='host_combine':assert n['end_ns']-n['start_ns']==6*d['combine_per_expert_ns']
            for job in jobs.values():
                cmd=job['matrix_command'];inp=job['input_channel_reload'];comp=job['nonpreemptive_native_compute'];out=job['output_channel']
                c=cmd['chip'];ei=cmd['expert'];phase=cmd['phase'];r=int(rows[phase][selected[ei],c]);C=[1536,512][phase]
                wave=(((r*C+16383)//16384)+3)//4
                assert cmd['end_ns']-cmd['start_ns']==250 and inp['end_ns']-inp['start_ns']==C//2
                assert comp['end_ns']-comp['start_ns']==wave*28000 and out['end_ns']-out['start_ns']==r*2
                assert cmd['end_ns']<=inp['start_ns']<=inp['end_ns']<=comp['start_ns']<=comp['end_ns']<=out['start_ns']
                chip_intervals[c].append((cmd['start_ns'],out['end_ns']))
                if phase:
                    ready=next(n for n in nodes if n['kind']=='whole_expert_gu_host_ready' and n['expert']==ei)
                    activation=next(n for n in nodes if n['kind']=='host_activation' and n['expert']==ei)
                    guout=[n['end_ns'] for n in nodes if n['kind']=='output_pcie' and n['expert']==ei and n['phase']==0]
                    assert max(guout)<=ready['end_ns']<=activation['start_ns']<=activation['end_ns']<=cmd['start_ns']
                    submit=next(n for n in nodes if n['kind'] in ['expert_down_host_submission','batch_down_host_submission'] and n['expert'] in [ei,-1])
                    assert submit['end_ns']-submit['start_ns']==2000 and submit['end_ns']<=cmd['start_ns']
            for intervals in chip_intervals:
                intervals.sort();assert all(a[1]<=b[0] for a,b in zip(intervals,intervals[1:]))
            trace_records.append(dict(path=str(trace.relative_to(ROOT)),sha256=sha256(trace),native_jobs=len(jobs),status='PASS'))
        for p in [path,folder/'schedule_costs.npz',folder/'completion.json']:artifacts[str(p.relative_to(ROOT))]=sha256(p)
    npz=DEST/'full_paired_deltas.npz';np.savez_compressed(npz,**delta_arrays)
    result=dict(status='PASS complete train-only strict-family audit',created_utc=datetime.now(timezone.utc).isoformat(),
        root_source_sha256=sha256(INPUT),frozen_plan_sha256=EXPECTED,tables=tables,paired_comparisons=comparisons,
        representative_trace_audits=trace_records,full_paired_deltas_sha256=sha256(npz),
        runtime_checks='Every newly evaluated schedule also checks all dependency/resource and native-job ownership intervals inside sealed Graph.run.',
        qualifiers=['The0case depth-joined family improves mean by only0.303%over the strongest prior mean family, while P99 improves3.257%over the strongest prior tail family. Different reference families are explicitly retained.',
            'The newmatched phase-fit comparison yields1.143%mean improvement; it is not a comparison to the strongest prior mean family.',
            'Always-active sharedFFN/SSM/attention remain excluded. Do not add shared latency serially or infer full-model role balance.',
            'Only first3layers and original train32x64; no heldout or physical timing measurement.',
            'The25wave capacity1059 certificate is for all16active chips; allowing13active chips can provide1062rows. Both suffice; the24wave arbitrary-active maximum1020 still proves that depth<=2requires at least25waves.'],
        source_sha256=sha256(Path(__file__)))
    target=DEST/'full_audit.json';target.write_text(json.dumps(result,indent=2)+'\n')
    for p in [target,npz,Path(__file__),Path(__file__).with_name('research_20261102_joint_depth_full.py'),INPUT]:artifacts[str(p.relative_to(ROOT))]=sha256(p)
    (DEST/'full_seal.json').write_text(json.dumps(dict(status='sealed; wait for sharedDAG before broader claims',created_utc=datetime.now(timezone.utc).isoformat(),
        artifacts=artifacts,frozen_plan_sha256=EXPECTED,strict_source_sha256=sha256(ROOT/'scripts/research_20261102_granite4_pipeline.py')),indent=2)+'\n')
    print(json.dumps(dict(audit_sha256=sha256(target),paired_sha256=sha256(npz),traces=len(trace_records)),indent=2))


if __name__=='__main__':main()
