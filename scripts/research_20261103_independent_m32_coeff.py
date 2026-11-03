#!/usr/bin/env python3
"""Bounded independent M32 row/profile/owned-service audit; no solver or infer."""
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path
import argparse
import hashlib
import importlib.util
import json
import shutil
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2/chip-count-regime-screen'
STAGE=BASE/'stage-b'
OUT=BASE/'independent-m32-coeff'

def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x): Path(p).write_text(json.dumps(x,indent=2)+'\n')
def load(p):
    with np.load(p) as z: return {k:z[k] for k in z.files}
def up(x,d): return (int(x)+int(d)-1)//int(d)

def freeze():
    OUT.mkdir(exist_ok=False);(OUT/'source-snapshot').mkdir()
    stage=json.loads((STAGE/'protocol.frozen.json').read_text())
    deps={
      'backend':next(Path(x['snapshot']) for x in stage['inputs'] if x['path'].endswith('ordinary_read_backend.py')),
      'compiler':next(Path(x['snapshot']) for x in stage['inputs'] if x['path'].endswith('whole_row_compiler.py')),
      'cpp':next(Path(x['snapshot']) for x in stage['inputs'] if x['path'].endswith('ordinary_read_native.cpp')),
      'stage_protocol':STAGE/'protocol.frozen.json',
      'plan':STAGE/'granite4tiny.M32.minimum_depth.plan.npz',
      'native_trace_archive':STAGE/'cost_arrays.npz',
      'producer':Path(__file__),
    }
    records={}
    for name,path in deps.items():
      item={'path':str(path),'sha256':sha(path)}
      if name in ('backend','compiler','cpp','producer'):
        filename={'backend':'research_20261103_ordinary_read_backend.py','compiler':'research_20261102_whole_row_compiler.py',
          'cpp':'ordinary_read_native.cpp','producer':Path(__file__).name}[name]
        dst=OUT/'source-snapshot'/filename;shutil.copy2(path,dst);item['snapshot']=str(dst)
      records[name]=item
    dump(OUT/'protocol.frozen.json',{
      'status':'frozen_before_audit_execution','utc':datetime.now(timezone.utc).isoformat(),
      'inputs':records,'copies':[1,4],'maps':['stageB_minimum_depth','seed20261103','seed20261104'],
      'query_scope':'All512ordered actualW8TRAIN L0 routes; no holdout, solver, fitting, inference or newnative cost campaign.',
      'checks':'Independent integer service/rows/pages/waves, same shared rows, one-copy logical ranges, single-block address witness, backend table and native frozen q0 trace bridge.',
      'claims':'Standard compact cardinality/packing control; no first mapping/solver/CP claim. Owned service is work, not lock queue dwell or native optimum.',
    })
    print(json.dumps({'protocol_sha256':sha(OUT/'protocol.frozen.json'),'source_sha256':sha(__file__)}))

def run():
    p=json.loads((OUT/'protocol.frozen.json').read_text())
    for x in p['inputs'].values():
      assert sha(x.get('snapshot',x['path']))==x['sha256']
    assert sha(__file__)==p['inputs']['producer']['sha256']
    sys.path.insert(0,str(OUT/'source-snapshot'))
    f=p['inputs']['backend']['snapshot'];spec=importlib.util.spec_from_file_location('m32_independent_backend',f)
    b=importlib.util.module_from_spec(spec);sys.modules[spec.name]=b;spec.loader.exec_module(b)
    stage=json.loads(Path(p['inputs']['stage_protocol']['path']).read_text())
    z=load(p['inputs']['plan']['path']);original=z['rows'];routes=z['routes']
    assert routes.shape==(512,6) and all(len(set(row.tolist()))==6 for row in routes)
    assert np.all((routes>=0)&(routes<64))
    maps={'stageB_minimum_depth':original.copy()}
    for seed in [20261103,20261104]:
      rng=np.random.default_rng(seed);r=np.zeros_like(original);r[64]=original[64]
      for e in range(64):
        perm=rng.permutation(32);X=perm[:5];O=perm[5:12]
        gu=np.setdiff1d(np.arange(32),O)
        r[e,0,gu]=41;r[e,0,gu.min()]=40;r[e,1,np.r_[X,O]]=128
      maps[f'seed{seed}']=r
    base=b.Service(**dict(stage['service_base'],chips=32,channels=8))

    def service_one(n,c,s):
      if n==0:return 0,0,0,0
      pages=up(n*c,s.page_bytes);waves=up(pages,s.planes)
      emitted=pages*s.page_bytes//c
      inp=up(c*s.input_bytes*s.input_channel_copies*10**9,s.channel_bytes_per_second)
      out=up(emitted*s.output_bytes*10**9,s.channel_bytes_per_second)
      return s.command_ns+inp+waves*s.ordinary_read_ns+out,pages,waves,emitted

    coefficient=[];results=[];saved={};inventory_reference=None;native_bridge=None
    for copies in [1,4]:
      s=replace(base,input_channel_copies=copies)
      gu=service_one(41,1536,s)[0];assert gu==service_one(40,1536,s)[0]
      down=service_one(128,512,s)[0]
      record={'input_copies':copies,'base_ns':gu,'X_ns':down,'O_ns':down-gu,
        'GU_only_ns':gu,'Down_only_ns':down,'both_ns':gu+down,
        'per_expert_total_owned_service_ns':25*gu+12*down,
        'GU40_or41':dict(zip(['service_ns','pages','waves','emitted_rows'],service_one(41,1536,s))),
        'Down128':dict(zip(['service_ns','pages','waves','emitted_rows'],service_one(128,512,s)))}
      coefficient.append(record)
      assert (gu,down,down-gu)==((29102,28762,-340) if copies==1 else (31406,29530,-1876))
      for name,rows in maps.items():
        A=rows[:64,0]>0;D=rows[:64,1]>0;X=A&D;O=~A
        assert np.all(A.sum(1)==25) and np.all(D.sum(1)==12)
        assert np.all(X.sum(1)==5) and np.all(O.sum(1)==7)
        assert not np.any(X&O) and np.all(A|D)
        assert np.all(np.sort(rows[:64,0],axis=1)==np.array([0]*7+[40]+[41]*24))
        assert np.all(np.sort(rows[:64,1],axis=1)==np.array([0]*20+[128]*12))
        assert np.array_equal(rows[64],original[64])
        plan=b.lower_rows(rows[:,0],rows[:,1],1536,[512]*64+[1024],s)
        b.validate_plan(plan,65,s)
        table=np.zeros((65,2,32),dtype=np.int64);addresses=[];cursor=np.zeros(32,dtype=np.int64)
        total={'rows':0,'useful_bytes':0,'pages':0,'waves':0,'jobs':0}
        for e in range(65):
          for phase in range(2):
            c=1536 if phase==0 else (512 if e<64 else 1024)
            R=(1024 if phase==0 else 1536) if e<64 else (2048 if phase==0 else 1536)
            coverage=np.zeros(R,np.uint8);logical_cursor=0
            lo,hi=map(int,plan['offsets'][e,phase]);band=[]
            for j in range(lo,hi):
              chip,n,C,pages,waves=map(int,plan['jobs'][j]);assert C==c and n==int(rows[e,phase,chip])
              value,P,W,emitted=service_one(n,c,s);assert (P,W)==(pages,waves)
              assert W<=192 and logical_cursor+n<=R
              coverage[logical_cursor:logical_cursor+n]+=1
              logical=plan['logical'][j]
              assert logical['row_start']==logical_cursor and logical['rows']==n
              start=int(cursor[chip]);end=start+W;assert end<=192
              addresses.append([j,e,phase,chip,0,start,W,pages,0,logical_cursor,n,c])
              cursor[chip]=end;logical_cursor+=n;table[e,phase,chip]+=value
              total['rows']+=n;total['useful_bytes']+=n*c;total['pages']+=P;total['waves']+=W;total['jobs']+=1
              band.append((chip,n,P,W))
            assert logical_cursor==R and np.all(coverage==1)
        assert np.array_equal(table,b.occupancy_table(plan,s)['chip_ns'])
        formula=gu+down*X+(down-gu)*O
        assert np.array_equal(table[:64].sum(1),formula)
        actual=table[routes].sum((1,2))+table[64].sum(0)
        compact=formula[routes].sum(1)+table[64].sum(0)
        assert np.array_equal(actual,compact)
        if inventory_reference is None: inventory_reference=total
        assert total==inventory_reference
        assert total=={'rows':64*(1024+1536)+2048+1536,'useful_bytes':int(148.5*2**20),'pages':9764,'waves':2441,'jobs':2424}
        trace_checks=[]
        for q in [0,255,511]:
          graph=b.ReferenceGraph.build(plan,np.r_[routes[q],64],s)
          from_nodes=np.zeros(32,np.int64)
          for node in graph.nodes:
            if node['job']>=0:from_nodes[node['chip']]+=node['duration']
          assert np.array_equal(from_nodes,compact[q])
          trace_checks.append({'q':q,'nodes':len(graph.nodes),'owned_services_exact':True})
          if name=='stageB_minimum_depth' and copies==1 and q==0:
            got=graph.run();archive=load(p['inputs']['native_trace_archive']['path'])
            key='granite4tiny.M32.minimum_depth.GF.q0trace'
            assert np.array_equal(got['trace'],archive[key])
            native_bridge={'q':0,'nodes':len(got['trace']),'cost_ns':got['cost'],'all_trace_fields_exact':True}
        if copies==1:
          saved[name+'.rows']=rows;saved[name+'.addresses']=np.asarray(addresses,np.int64)
          saved[name+'.jobs']=plan['jobs'];saved[name+'.offsets']=plan['offsets']
        results.append({'map':name,'input_copies':copies,'all512route_chip_services_exact':True,
          'category_counts':{'X':5,'O':7,'GU_only':20},'shared_rows_unchanged':True,
          'one_copy_full_row_coverage':True,'inventory':total,'chip_wave_inventory':cursor.tolist(),
          'single_block_each_chip':True,'physical_allocated_blocks':32,
          'representative_graph_node_checks':trace_checks})
    saved['routes']=routes;np.savez_compressed(OUT/'audit_arrays.npz',**saved)
    report={'status':'PASS_INDEPENDENT_M32_COMPACT_COEFFICIENT_AND_METADATA_CONTRACT',
      'utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(OUT/'protocol.frozen.json'),
      'coefficients':coefficient,'results':results,'native_frozen_trace_bridge':native_bridge,
      'address_fields':['job','expert','phase','chip','block','start_WL','reserved_WL','pages','first_plane','logical_row_start','rows','full_C'],
      'universal_capacity_bound':'Each expert contributes at most oneGU+oneDown wave to any chip;64*2+sharedGUmax2+sharedDownmax1=131<192. Thus any feasibleX5/O7map admits this single-layer one-block/chip witness.',
      'ledger_scope':'All64 routed experts and shared e64 GU2048x1536/Down1536x1024 included once. No extra coefficient copies; pages/waves/jobs equal stageB. Scale metadata remains host-associated per complete logical row.',
      'claim_limits':['Owned service sums four operation durations, excluding lock waiting/queueing; not native makespan.','CP/hypergraph/cardinality optimization is a standard strong control, not first/new solver or joint mapping.','Beating route-blind is a need-fit diagnostic; no all-IFP/general hardware breakthrough claim.','M32 is an equal-per-chip AiF-capability scale model, not a commercial32-chip AiF implementation.','No MAC/output ABI or numerical hardware proof added; whole-row opacity assumptions remain.','OnlyTRAIN L0, three feasible maps; no solver output, heldout, or global optimizer audit.']}
    dump(OUT/'audit.json',report)
    files=[f for f in OUT.rglob('*') if f.is_file() and f.name!='completion.json']
    dump(OUT/'completion.json',{'status':'SEALED_PASS','utc':datetime.now(timezone.utc).isoformat(),
      'files':{str(f.relative_to(OUT)):sha(f) for f in sorted(files)}})
    print(json.dumps({'status':report['status'],'audit_sha256':sha(OUT/'audit.json'),
      'completion_sha256':sha(OUT/'completion.json'),'coefficients':coefficient},indent=2))

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);a=ap.parse_args()
    freeze() if a.mode=='freeze' else run()
