#!/usr/bin/env python3
"""Independent replay of the three largest aggregate greedy anomalies."""
from datetime import datetime, timezone
import json
from pathlib import Path
import numpy as np
from audit_fringe import ROOT,BASE,RUN,OUT,OLD,FROZEN,DATA,templates,items_from_plan,fringe_nodes,read,sha,schedule


def main():
    with np.load(OUT/'paired-48layer-costs.npz') as a:
        original=a['mode0__rect_min_io4'];new=a['mode1__rect_min_io4']
    worst=np.argsort(new-original)[:3]
    with np.load(DATA) as a:routes=a['routes'].copy();req=a['req_ids'].copy();pos=a['token_pos'].copy()
    inv=templates();freeze=read(FROZEN)
    plans={r['layer']:r for r in freeze['records'] if r['case']=='rect_min_io4'}
    output=[]
    for ix in worst:
        layers=[]
        for layer in range(48):
            rec=plans[layer];plan=read(ROOT/rec['plan'])
            item=items_from_plan(plan,inv,routes[ix,layer]);n0=fringe_nodes(item,0);n1=fringe_nodes(item,1)
            assert len(n0)==len(n1)
            assert all(b['duration']>=a['duration'] for a,b in zip(n0,n1))
            s0,e0=schedule(n0,False);s1,e1=schedule(n1,False)
            with np.load(OLD/(rec['label']+'.npz')) as a:v0=int(a['cost'][ix])
            with np.load(RUN/(rec['label']+'.mode1.npz')) as a:v1=int(a['cost'][ix])
            with np.load(RUN/(rec['label']+'.mode2.npz')) as a:assert int(a['cost'][ix])==v1
            assert int(e0.max())==v0 and int(e1.max())==v1
            if v1<v0:
                resources=sorted({n['resource'] for n in n0})
                changed_order=[]
                for resource in resources:
                    ids=[i for i,n in enumerate(n0) if n['resource']==resource]
                    before=sorted(ids,key=lambda i:(s0[i],i));after=sorted(ids,key=lambda i:(s1[i],i))
                    if before!=after:changed_order.append(resource)
                assert changed_order
            else:changed_order=[]
            layers.append(dict(layer=layer,original_ns=v0,fringe_ns=v1,signed_change_ns=v1-v0,
                all_node_durations_nondecreasing=True,changed_resource_orders=changed_order))
        assert sum(r['original_ns'] for r in layers)==int(original[ix])
        assert sum(r['fringe_ns'] for r in layers)==int(new[ix])
        output.append(dict(query_index=int(ix),req_id=int(req[ix]),token_pos=int(pos[ix]),
            summed48_signed_change_ns=int(new[ix]-original[ix]),layers=layers))
    result=dict(status='PASS independent greedy-anomaly controls',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),spec_adapter_sha256=sha(Path(__file__).with_name('audit_fringe.py')),
        independently_replayed_schedules=3*48*2,records=output,
        interpretation='Increasing selected node durations can alter greedy resource order. The lower summed cost is reproduced by the separate set-based scheduler; no cheaper node or reduced NAND work is introduced. This is not a general optimal-schedule speedup from more outputs.')
    p=OUT/'audit-negative-cases.json';assert not p.exists();p.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(path=str(p),sha256=sha(p),queries=[{k:r[k] for k in ['query_index','req_id','token_pos','summed48_signed_change_ns']} for r in output]),indent=2))


if __name__=='__main__':main()
