"""Independent union/ownership/capacity and native-DAG audit. No fitting."""
import ctypes
import hashlib
import json
import sys
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

ROOT=Path('/home/ericyu/expert_flash_experiment');BASE=Path(__file__).resolve().parent.parent
WORK=BASE/'work-screen-v1';DAG=BASE/'native-dag-v1';OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(ROOT/'scripts'))
from research_20261103_topology_placement_replay import Replay,load_python


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def union_views(ranges,selected,chip,merge):
    # A positive gap costs >=28064ns/wave, exceeding a saved command+input
    # (at most9096ns). Therefore the exact work-DP optimum is precisely the
    # connected union of overlapping/touching rounded intervals in each block.
    intervals=[]
    for ordinal,e in enumerate(selected):
        lo,hi=map(int,ranges[e,chip])
        while lo<hi:
            block=lo//6144;end=min(hi,(block+1)*6144)
            intervals.append(dict(block=block,start=(lo-block*6144)//32,end=(end-block*6144+31)//32,owners={ordinal},owned=[(int(e),lo,end)]))
            lo=end
    intervals.sort(key=lambda v:(v['block'],v['start'],v['end']))
    views=[]
    for v in intervals:
        if merge and views and views[-1]['block']==v['block'] and v['start']<=views[-1]['end']:
            views[-1]['end']=max(views[-1]['end'],v['end']);views[-1]['owners']|=v['owners'];views[-1]['owned']+=v['owned']
        else:views.append({**v,'owners':set(v['owners']),'owned':list(v['owned'])})
    expected=sorted((e,lo,hi) for v in intervals for e,lo,hi in v['owned'])
    got=sorted((e,lo,hi) for v in views for e,lo,hi in v['owned'])
    assert got==expected # Each needed segment has exactly one scatter owner.
    for v in views:
        assert 0<=v['start']<v['end']<=192
        assert all(v['block']*6144+v['start']*32<=lo<hi<=v['block']*6144+v['end']*32 for e,lo,hi in v['owned'])
    return views


def graph(plan,ranges,selected,s,mode):
    if mode==0:return backend.ReferenceGraph.build(plan,selected,s)
    g=backend.ReferenceGraph(s);submit=g.add(0,s.host_submit_ns)
    upload=g.add(2,s.transfer(2048,s.pcie_bytes_per_second),[submit])
    allviews=[];outputs=[[] for _ in selected]
    for chip in range(16):
        for v in union_views(ranges,selected,chip,mode==2):allviews.append({**v,'chip':chip})
    allviews.sort(key=lambda v:(min(v['owners']),v['chip'],v['block'],v['start'],v['end']-v['start']))
    for v in allviews:
        w=v['end']-v['start'];done=g.matrix(min(v['owners']),0,[v['chip'],w*32,2048,w*4,w],submit,upload)
        for owner in sorted(v['owners']):outputs[owner].append(done)
    ready=[g.add(s.resources-1,0,deps,expert=i) for i,deps in enumerate(outputs)]
    complete=[]
    for i,e in enumerate(selected):
        act=g.add(0,s.activation_ns,[ready[i]],1,i);submit=g.add(0,s.host_submit_ns,[act],1,i)
        upload=g.add(2,s.transfer(int(plan['input_columns'][e,1]),s.pcie_bytes_per_second),[submit],1,i)
        lo,hi=plan['offsets'][e,1];complete.extend(g.matrix(i,1,d,submit,upload) for d in plan['jobs'][lo:hi])
    g.add(0,s.combine_ns,complete,1)
    return g,allviews


work=json.loads((WORK/'screen.json').read_text());wp=json.loads((WORK/'protocol.frozen.json').read_text())
dag=json.loads((DAG/'screen.json').read_text());dp=json.loads((DAG/'protocol.frozen.json').read_text())
for p in [wp,dp]:
    for item in p['inputs']:assert sha(item['path'])==item['sha256']
assert work['protocol_sha256']==sha(WORK/'protocol.frozen.json') and dag['protocol_sha256']==sha(DAG/'protocol.frozen.json')
backend=load_python();native=Replay();layout_checks=[];union_cases=0
byid={r['id']:r for r in work['records']};profilebyid={r['id']:r for r in wp['profiles']}
for profile in wp['profiles']:
    rows=np.asarray(profile['rows']);copies=profile['copies']
    assert np.all(rows[:64,0].sum(1)==2816) and rows[64,0].sum()==5632
    for name in wp['ordering_layouts']:
        with np.load(WORK/(profile['id']+'.'+name+'.layout.npz')) as z:ranges=z['ranges'];rr=z['rows']
        assert np.array_equal(rr,rows)
        assert np.array_equal(ranges[:,:,1]-ranges[:,:,0],rows[:,0])
        for chip in range(16):
            segments=sorted((int(lo),int(hi)) for lo,hi in ranges[:,chip] if hi>lo)
            assert all(a[1]<=b[0] for a,b in zip(segments,segments[1:]))
        # Logical coefficients on chips remain the same row partition. Physical
        # interval order changes neither full expert row sums nor row identities.
        logical=np.cumsum(rows[:,0],axis=1)
        assert np.all(logical[:64,-1]==2816) and logical[64,-1]==5632
        ident=profile['id']+'.'+name
        with np.load(WORK/(ident+'.cmd250.npz')) as z:routes=z['routes'];gu=z['gu_controls']
        expected=np.zeros_like(gu);max_layout_rows=max_layout_owners=0
        for q in range(512):
            selected=np.r_[routes[q],64]
            for chip in range(16):
                for mode,at in [(False,0),(True,4)]:
                    views=union_views(ranges,selected,chip,mode)
                    if mode and views:
                        max_layout_rows=max(max_layout_rows,max((v['end']-v['start'])*32 for v in views))
                        max_layout_owners=max(max_layout_owners,max(len(v['owners']) for v in views))
                    waves=sum(v['end']-v['start'] for v in views);commands=len(views);out=32*waves
                    expected[q,chip,at:at+4]=[waves*28064+commands*(250+1024*copies),waves,commands,out]
                union_cases+=1
        assert np.array_equal(expected,gu)
        with np.load(WORK/(ident+'.cmd5000.npz')) as z:gu5=z['gu_controls']
        for at in [0,4]:expected[:,:,at]+=4750*expected[:,:,at+2]
        assert np.array_equal(expected,gu5)
        r=byid[ident+'.cmd250'];capacity=r['capacity']
        for chip,used in enumerate(capacity['gu_protected_wordlines_by_block']):
            # Construct actual safe Down tail allocations, not only a bin count.
            pos=list(used);residual=[192-u for u in used];occupied=[]
            for e in range(65):
                n=int(rows[e,1,chip]);C=1408 if e<64 else 2816
                if not n:continue
                w=(n*C+65535)//65536;assert w<=192
                where=next((i for i,space in enumerate(residual) if space>=w),None)
                if where is None:where=len(residual);residual.append(192);pos.append(0)
                start=pos[where];pos[where]+=w;residual[where]-=w
                assert start+w<=192 and start>=(used[where] if where<len(used) else 0)
                occupied.append((where,start,start+w,e))
            assert len(residual)==capacity['tailfilled_blocks_by_chip'][chip]
        layout_checks.append(dict(profile=profile['id'],ordering=name,capacity=capacity,max_fused_rows_all512=max_layout_rows,max_fused_owners_all512=max_layout_owners))

lib=ctypes.CDLL(str(DAG/'tile_dag.so'));fn=lib.eval_gu_tile_dag
i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
fn.argtypes=[i32,i32,i32,i32,i32,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,i64,i64,i64,ctypes.c_void_p,i64]
trace_checks=[];fringes=[];bridge=0;maxrows=maxowners=0
for r in dag['records']:
    with np.load(r['arrays']) as z:rr=z['rows'];ranges=z['ranges'];routes=z['routes'];values=z['cost'];bounds=z['lower_bound']
    assert values.shape==(512,) and np.all(values>=bounds)
    assert r['mean_ns']==float(values.mean()) and r['p99_ns']==float(np.percentile(values,99))
    config={**dp['services'][str(r['copies'])],'down_first':r['priority']};s=backend.Service(**config)
    plan=backend.lower_rows(rr[:,0],rr[:,1],2048,[1408]*64+[2816],s)
    if r['mode']==0:
        old,_=native.run(plan,routes,config);assert np.array_equal(old,values);bridge+=512
    for q in [0,511]:
        result=graph(plan,ranges,np.r_[routes[q],64],s,r['mode'])
        if r['mode']==0:g=result;views=[]
        else:g,views=result
        got=g.run();assert got['cost']==values[q]
        log=np.zeros((20000,8),np.int64);n=np.zeros(1,np.int64);v=np.zeros(1,np.int64);b=v.copy()
        fn(plan['offsets'],plan['jobs'],plan['input_columns'],ranges,np.ascontiguousarray(routes[q:q+1],np.int32),64,len(plan['jobs']),1,6,r['mode'],s.array(),v,b,ctypes.c_void_p(log.ctypes.data),n)
        assert np.array_equal(log[:n[0]],got['trace'])
        trace_checks.append(dict(id=r['id'],query=q,nodes=int(n[0])))
        if views:
            maxrows=max(maxrows,max((v['end']-v['start'])*32 for v in views));maxowners=max(maxowners,max(len(v['owners']) for v in views))
    if r['mode']==1:
        base=next(x for x in dag['records'] if x['candidate']==r['candidate'] and x['priority']==r['priority'] and x['mode']==0)
        with np.load(base['arrays']) as z:old=z['cost']
        fringes.append(dict(id=r['id'],mean_change_vs_original_ns=float((values-old).mean()),min_change_ns=int((values-old).min()),max_change_ns=int((values-old).max())))

paired=[];arrays={}
for c in dag['comparisons']:
    winner=c['fused_best'];assert not c['expand_gate']
    with np.load(winner['arrays']) as z:fused=z['cost']
    for label,base in [('strongest_original',c['original_best']),('same_layout_singleton',c['same_layout_singleton'])]:
        with np.load(base['arrays']) as z:old=z['cost']
        delta=old-fused;key=f'input{c["copies"]}.{label}';arrays[key]=delta
        paired.append(dict(copies=c['copies'],baseline=label,mean_gain_pct=100*(1-fused.mean()/old.mean()),p99_gain_pct=100*(1-np.percentile(fused,99)/np.percentile(old,99)),improved=int((delta>0).sum()),equal=int((delta==0).sum()),worse=int((delta<0).sum()),delta_ns_quantiles=np.percentile(delta,[0,1,50,95,99,100]).tolist()))
matching=json.loads((WORK/'ordering.fit.json').read_text());assert matching['matching_unperturbed_edge_weight']==658
eq=next(r for r in work['records'] if 'equal_gu176_on_v2free' in r['labels'] and r['copies']==1 and r['command_ns']==250 and r['ordering']=='matching_blockslots')
assert eq['coalesced_GU_waves_mean']==752-16*658/512
np.savez_compressed(OUT/'paired_deltas.npz',**arrays)
native_layouts=[x for x in layout_checks if any(c['profile']==x['profile'] and c['ordering']==x['ordering'] for c in dp['candidates'])]
summary=dict(status='passed_independent_audit_no_expansion',utc=datetime.now(timezone.utc).isoformat(),work_protocol_sha256=sha(WORK/'protocol.frozen.json'),dag_protocol_sha256=sha(DAG/'protocol.frozen.json'),validation=dict(layouts=len(layout_checks),all_query_chip_union_cases=union_cases,independent_work_controls=union_cases*2,original_bridge_queries=bridge,independent_python_cpp_traces=len(trace_checks),independent_python_cpp_nodes=sum(x['nodes'] for x in trace_checks),max_fullwave_GU_rows_in_checked_queries=maxrows,max_fused_owners_in_checked_queries=maxowners,max_fullwave_GU_rows_all_native_candidate_queries=max(x['max_fused_rows_all512'] for x in native_layouts),max_fused_owners_all_native_candidate_queries=max(x['max_fused_owners_all512'] for x in native_layouts)),matching_certificate=dict(observed_matched_pairs_per_query=658/512,uniform_top6_expected_pairs_per_query=6*5/(2*63),observed_GU_waves_saved_per_query=16*658/512,scope='Uniform176 block-aligned32perfectpairs only; no claim of globally optimal all-layout order'),paired=paired,singleton_fringe_and_blocksplit_costs=fringes,layout_checks=layout_checks,trace_checks=trace_checks,limits=['Command5000ns sensitivity is synthetic,notdevice timing','Default250ns work bound excludes >=2% improvement in this screened set','No all-layout/global optimizer impossibility claim','No native FIFO/maxpagecount/outputscale validation;architecturecontract remainsconditional','Whole-view PCIe completion and full-expertGU A8 scale barrier retained'])
(OUT/'audit.json').write_text(json.dumps(summary,indent=2)+'\n')
print(json.dumps(dict(validation=summary['validation'],paired=paired),indent=2))
