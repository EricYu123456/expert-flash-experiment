#!/usr/bin/env python3
"""Format-matched routing replay after a full-FFN integer CPU reference.

No new fitting: apply the older frozen physical plans to the actual W8A8
decode routes. All forty layers and the same one-copy shared FFN are counted.
"""
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

from research_20261102_shared_pipeline import build_jobs

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
REFERENCE=BASE/'full-ffn-integer-reference'
PLAN=BASE/'geometry-study/granite4tiny.frozen.json'
EXPECTED='a55f17010b7b48c6c396dee92bb1d127a541e28a93b94aee2e42aac9f3dc2538'
DEST=REFERENCE/'frozen-placement-replay'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def worker(task):
    layer,case,route,output=task;target=Path(output);doc=json.loads(PLAN.read_text())
    phases={p['name']:p for p in doc['phases'] if p['layer']==layer}
    key='rows' if case=='frozen_whole_rows' else 'group_rows'
    gu=np.asarray(phases['gate_up'][key]);down=np.asarray(phases['down'][key]);values=[];bound=[];trace=None
    for selected in route:
        items=[dict(gu_rows=gu[e].tolist(),down_rows=down[e].tolist(),down_columns=512) for e in selected]
        # Exact same fixed shared physical partition as the CPU reference.
        items.append(dict(gu_rows=[128]*16,down_rows=[96]*16,down_columns=1024))
        result=build_jobs(items).run(False,trace=trace is None);load=np.zeros(16,dtype=np.int64)
        for item in items:
            for name,c in (('gu_rows',1536),('down_rows',item['down_columns'])):
                r=np.asarray(item[name],dtype=np.int64);w=(r*c+65535)//65536
                load+=np.where(r>0,w*28000+250+(c+1)//2+2*r,0)
        lb=2000+int(load.max());assert result['makespan_ns']>=lb
        values.append(result['makespan_ns']);bound.append(lb)
        if trace is None:trace=dict(layer=layer,case=case,selected=selected.tolist(),items=items,**result)
    label=f'L{layer}.{case}';np.savez_compressed(target/(label+'.npz'),cost=np.asarray(values,dtype=np.int64),native_chip_bound=np.asarray(bound,dtype=np.int64))
    (target/(label+'.trace.json')).write_text(json.dumps(trace,indent=2)+'\n')
    return dict(layer=layer,case=case,label=label,samples=len(values),mean_ns=float(np.mean(values)),p99_ns=float(np.percentile(values,99)),artifacts_sha256={ext:sha(target/(label+'.'+ext)) for ext in ('npz','trace.json')})


def main():
    DEST.mkdir(parents=True,exist_ok=True);assert sha(PLAN)==EXPECTED
    paths=sorted((REFERENCE/'main/traces').glob('*.placed.routing.jsonl'));assert len(paths)==2
    routes=[];promptids=[]
    for path in paths:
        rows=[json.loads(line) for line in path.open() if json.loads(line)['phase']=='decode']
        a=np.full((16,40,6),-1,dtype=np.int32)
        for r in rows:
            assert np.all(a[r['step'],r['layer']]==-1);a[r['step'],r['layer']]=r['expert_ids']
        assert np.all(a>=0);routes.append(a);promptids.append(path.name.split('.placed')[0])
    routes=np.stack(routes);np.savez_compressed(DEST/'routes.npz',routes=routes,prompt_ids=np.asarray(promptids))
    inputs=[PLAN,*paths,REFERENCE/'main/numeric_report.json',REFERENCE/'main/full_token_coverage_audit.json',REFERENCE/'main/count_definitions.json']
    hashes={str(p):sha(p) for p in inputs};sources={str(p):sha(p) for p in (Path(__file__),ROOT/'scripts/research_20261102_shared_pipeline.py',ROOT/'scripts/research_20261102_granite4_pipeline.py')}
    protocol=dict(created_utc=datetime.now(timezone.utc).isoformat(),inputs_sha256=hashes,source_sha256=sources,
                  fit='No label/geometry/shared fitting; use old frozen full40layer plan. Two earlier training prompts, actual fullFFN W8A8 decode routes, not fresh test.',
                  shared='Same numeric-reference onecopy16-chip shared GU128rows/chip anddown96rows/chip. No sharedcopyorpartialsum.')
    (DEST/'protocol.frozen.json').write_text(json.dumps(protocol,indent=2)+'\n');records=[]
    with ProcessPoolExecutor(max_workers=8) as pool:
        for f in as_completed([pool.submit(worker,(l,c,routes[:,:,l].reshape(-1,6),str(DEST))) for l in range(40) for c in ('frozen_phase_group','frozen_whole_rows')]):records.append(f.result())
    totals=[]
    for case in ('frozen_phase_group','frozen_whole_rows'):
        costs=np.stack([np.load(DEST/(r['label']+'.npz'))['cost'] for r in records if r['case']==case]).sum(axis=0)
        assert len(costs)==32
        totals.append(dict(case=case,mean_ns=float(costs.mean()),p99_ns=float(np.percentile(costs,99)),per_prompt_mean_ns=costs.reshape(2,16).mean(axis=1).tolist()))
        np.save(DEST/(case+'.sum40.npy'),costs)
    assert all(sha(p)==h for p,h in hashes.items()) and all(sha(p)==h for p,h in sources.items())
    report=dict(status='complete format-matched full40layer frozen placement replay',created_utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(DEST/'protocol.frozen.json'),records=records,totals=totals,
                mean_reduction=1-totals[1]['mean_ns']/totals[0]['mean_ns'],
                limits=['Uncalibrated software service model; ordinary28us reads,250ns/chip command,2us hostsubmit,4Boutput;0activation/combine.',
                        'Only FFN service including shared. Attention/router/SSM remain sourceGGUF host operations and their time is omitted.',
                        'W8A8 numerical reference differs from the sourceGGUF checkpoint; source-control token mismatches are disclosed, not a quality validation.',
                        'Only two earlier training prompts,32decodes; no heldout/performance-generalization/new-method claim.',
                        'The whole-row geometry and complementary striping are existing-method controls.'])
    (DEST/'summary.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(dict(totals=totals,mean_reduction=report['mean_reduction']),indent=2),flush=True)


if __name__=='__main__':main()
