#!/usr/bin/env python3
"""Frozen row-prefix construction, reusing existing W8 bytes without conversion."""
import collections, hashlib, json, math, os, pathlib, subprocess, sys, time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
OUT=BASE/'routed-boundary-view-numeric'
OLD=BASE/'shared-materialize'
PLAN=BASE/'ordinary-read-compiler/full-campaign-v4-deduplicated/granite4tiny.L0.cmd250/whole_rows.plan.json'
TRAIN=BASE/'w8-format-training/granite4tiny'
PAGE=16384; WAVE=4*PAGE; BLOCK=192*WAVE
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def bh(x):return hashlib.sha256(x).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def fnv(bs):
 h=14695981039346656037
 for b in bs:h=((h^b)*1099511628211)&((1<<64)-1)
 return str(h)
def quant(x):
 x=np.asarray(x,dtype=np.float32);m=np.max(np.abs(x));s=np.float32(1) if m==0 else np.float32(m/np.float32(127))
 v=(x/s).astype(np.float32);q=np.copysign(np.floor(np.abs(v.astype(np.float64))+.5),v)
 return np.clip(q,-127,127).astype(np.int8),s
def gemv(w,sc,x):
 q,s=quant(x);i=w.astype(np.int64)@q.astype(np.int64)
 return (i.astype(np.float32)*(sc*s).astype(np.float32)).astype(np.float32),i
def activate(y):
 g,u=np.split(y,2);sig=(np.float32(1)/(np.float32(1)+np.exp(-g).astype(np.float32))).astype(np.float32)
 return ((g*sig).astype(np.float32)*u).astype(np.float32)
def main():
 began=time.monotonic();OUT.mkdir(exist_ok=False)
 man=json.loads((OLD/'onecopy/manifest.json').read_text());sm=json.loads((OLD/'logical_scales.json').read_text());sb=(OLD/'logical_scales.f32').read_bytes()
 plan=json.loads(PLAN.read_text());routes=TRAIN/'decode.routes.npz'
 with np.load(routes) as r:
  experts=r['routes'][0,0,0].tolist();pid=str(r['prompt_ids'][0]);assert pid.startswith('train_') and len(set(experts))==6
 keys=[f'routed.{e}' for e in experts]+['shared'];weights={};scales={};sources={};inputs=[PLAN,routes,OLD/'onecopy/manifest.json',OLD/'logical_scales.json',OLD/'logical_scales.f32',OLD/'fixtures.npz']
 prior={};prior_paths=sorted(TRAIN.glob('batch*/weight_cache_ledger.jsonl'));inputs+=prior_paths
 for p in prior_paths:
  for r in map(json.loads,p.read_text().splitlines()):
   k=r['weight'],r['expert'];fp=r['canonical_bytes_fnv1a64'],r['row_scales_fnv1a64']
   assert k not in prior or prior[k][:2]==fp;prior[k]=(*fp,r)
 fds={}
 for chip in man['chips']:
  p=OLD/chip['path'];assert sha(p)==chip['sha256'];fds[chip['chip']]=os.open(p,os.O_RDONLY);inputs.append(p)
 for key in keys:
  e=0 if key=='shared' else int(key.split('.')[1]);suffix='shexp' if key=='shared' else 'exps';I=1024 if key=='shared' else 512
  for phase,R,C in [('gu',2*I,1536),('down',1536,I)]:
   k=key+'.'+phase;w=np.empty((R,C),dtype=np.int8);cover=np.zeros(R,dtype=np.uint8)
   vs=[v for a in man['shared_gu_legal_views'].values() for v in a['1']] if k=='shared.gu' else man['execution_views'][k]
   for v in vs:
    start,n=v['logical_row_start'],v['rows'];raw=os.pread(fds[v['chip']],v['weight_bytes'],v['linear_byte_offset']);w[start:start+n]=np.frombuffer(raw,dtype=np.int8).reshape(n,C);cover[start:start+n]+=1
   assert np.all(cover==1);s=np.frombuffer(sb,dtype='<f4',count=R,offset=sm[k]['offset']).copy()
   checks=[]
   for proj,start,n in [('gate',0,I),('up',I,I)] if phase=='gu' else [('down',0,R)]:
    a,b,record=prior[f'blk.0.ffn_{proj}_{suffix}.weight',e]
    assert fnv(w[start:start+n].tobytes())==a and fnv(s[start:start+n].tobytes())==b
    checks.append(dict(projection=proj,canonical_bytes_fnv1a64=a,row_scales_fnv1a64=b,TRAIN_record=record))
   weights[k]=w;scales[k]=s;sources[k]=dict(rows=R,columns=C,weight_sha256=bh(w.tobytes()),scale_sha256=bh(s.tobytes()),row_sha256=[bh(row.tobytes()) for row in w],TRAIN_fingerprint_checks=checks)
 for fd in fds.values():os.close(fd)
 # Preserve logical scale identity, independent of physical row order.
 scale_meta={};offset=0
 with (OUT/'logical_scales.f32').open('wb') as f:
  for k in sorted(scales):
   raw=scales[k].astype('<f4').tobytes();f.write(raw);scale_meta[k]=dict(offset=offset,rows=len(scales[k]),sha256=bh(raw));offset+=len(raw)
 dump(OUT/'logical_scales.json',scale_meta);dump(OUT/'source_rows.json',sources)
 files=[(OUT/f'chip{c:02d}.bin').open('w+b') for c in range(16)];cursor=[0]*16;alloc=[];views={};decisions=[]
 def store(k,c,rowids,role):
  C=weights[k].shape[1];n=len(rowids);assert n>0;waves=math.ceil(n*C/WAVE);at=cursor[c]
  if at%192+waves>192:at=math.ceil(at/192)*192
  raw=weights[k][rowids].tobytes();files[c].seek(at*WAVE);files[c].write(raw)
  a=dict(id=len(alloc),matrix=k,chip=c,role=role,rowids=rowids,columns=C,weight_bytes=len(raw),pages=math.ceil(len(raw)/PAGE),reserved_waves=waves,block=at//192,start_wordline=at%192,byte_offset=at*WAVE,payload_sha256=bh(raw))
  alloc.append(a);cursor[c]=at+waves;return a
 def view(a,rowids):
  n=len(rowids);C=a['columns'];pages=math.ceil(n*C/PAGE);complete=pages*PAGE//C;partial=pages*PAGE%C;stored=a['rowids']
  assert stored[:n]==rowids
  return dict(allocation=a['id'],chip=a['chip'],matrix=a['matrix'],rowids=rowids,rows=n,columns=C,block=a['block'],start_wordline=a['start_wordline'],start_plane=0,byte_in_plane_page=0,byte_offset=a['byte_offset'],requested_pages=pages,requested_waves=math.ceil(pages/4),useful_bytes=n*C,scale_logical_rows=rowids,drain_drop=dict(page_bytes=pages*PAGE,useful_complete_rows=n,extra_complete_rows=complete-n,extra_complete_rowids=stored[n:min(complete,len(stored))],extra_zero_padding_complete_rows=max(0,complete-len(stored)),partial_row_bytes=partial,partial_row_logical_id=stored[complete] if complete<len(stored) else None,scope='Software computes only declared complete useful rows. Extra complete rows require drain/drop; partial-row flush/reset and unused-plane behavior remain native ABI unknowns.'))
 for key in keys:
  e=64 if key=='shared' else int(key.split('.')[1])
  for phase in [0,1]:
   k=key+('.gu' if phase==0 else '.down');bands=[b for b in plan['logical'] if b['expert']==e and b['phase']==phase];bands.sort(key=lambda b:b['chip']);assert len({b['chip'] for b in bands})==len(bands)
   original={b['chip']:list(range(b['row_start'],b['row_start']+b['rows'])) for b in bands}
   assert sorted(r for rs in original.values() for r in rs)==list(range(weights[k].shape[0]))
   if phase==0 and e!=64:
    A=[len(original.get(c,[])) for c in range(16)];s=next(c for c in range(16) if A[c] in [82,83] and A[c^8]==40);t=s^8;delta=A[s]-A[t]
    packet=original[s][-delta:];core={c:list(rs) for c,rs in original.items()};core[s]=core[s][:-delta]
    va=[];vb=[]
    for c,rs in sorted(core.items()):
     physical=rs+packet if c in [s,t] else rs;a=store(k,c,physical,'common_core_plus_boundary' if c in [s,t] else 'common_core')
     va.append(view(a,rs+packet if c==s else rs));vb.append(view(a,rs+packet if c==t else rs))
    B=A.copy();B[s],B[t]=B[t],B[s];assert [v['rows'] for v in va]==A and [v['rows'] for v in vb]==B
    views[k]=dict(A=va,B=vb);decisions.append(dict(matrix=k,A_rowcounts=A,B_rowcounts=B,source=s,destination=t,delta_rows=delta,boundary_rowids=packet,duplicate_useful_bytes=delta*1536))
   else:
    vv=[view(store(k,c,rs,'fixed_original_geometry'),rs) for c,rs in sorted(original.items())];views[k]=dict(A=vv,B=vv)
 chips=[]
 for c,f in enumerate(files):
  # Reserve only addressable complete LSB blocks; holes contain zeros.
  size=math.ceil(cursor[c]/192)*BLOCK;f.truncate(size);f.close();p=OUT/f'chip{c:02d}.bin';chips.append(dict(chip=c,path=p.name,sha256=sha(p),used_wave_slots=cursor[c],reserved_blocks=size//BLOCK,reserved_bytes=size))
 inventory=dict(base_useful_bytes=sum(w.nbytes for w in weights.values()),stored_useful_bytes=sum(a['weight_bytes'] for a in alloc),duplicated_boundary_rows=sum(d['delta_rows'] for d in decisions),duplicate_useful_bytes=sum(d['duplicate_useful_bytes'] for d in decisions),stored_payload_pages=sum(a['pages'] for a in alloc),max_view_reserved_waves=sum(a['reserved_waves'] for a in alloc),allocated_LSB_block_bytes=sum(c['reserved_bytes'] for c in chips),allocation_count=len(alloc),one_block_each_allocation=all(a['start_wordline']+a['reserved_waves']<=192 for a in alloc))
 assert inventory['stored_useful_bytes']-inventory['base_useful_bytes']==inventory['duplicate_useful_bytes']
 provenance={str(p):sha(p) for p in inputs};dump(OUT/'manifest.json',dict(chips=chips,allocations=alloc,views=views,decisions=decisions,inventory=inventory,scale_meta=scale_meta,address_formula='((block*192+wordline)*4+plane)*16384; all jobs start plane0 byte0',source_provenance_sha256=provenance,construction='v4 L0 balanced GU row counts; only chosen source/destination counts swapped; original v4 Down and shared geometry fixed',interpretation='Row-prefix materialization/numeric construction proof; not v5/v6 selected winner, heldout/fullmodel inference, native AiF ABI or performance claim.'))
 # Reuse existing saved input vectors; router alphas are explicit fixtures.
 fx=np.load(OLD/'fixtures.npz');saved={};records=[]
 for fi in [0,1]:
  x=fx[f'x{fi}'].copy();alpha=fx[f'alpha{fi}'].copy();assert len(alpha)==6;saved[f'x{fi}']=x;saved[f'alpha{fi}']=alpha;mixed=np.zeros(1536,dtype=np.float32)
  for j,key in enumerate(keys):
   gu,gi=gemv(weights[key+'.gu'],scales[key+'.gu'],x);act=activate(gu);down,di=gemv(weights[key+'.down'],scales[key+'.down'],act);aq,ascl=quant(act)
   for label,value in [('gu',gu),('gu_i64',gi),('activation',act),('down',down),('down_i64',di),('down_input_a8',aq),('down_input_scale',np.asarray(ascl))]:saved[f'f{fi}.{key}.{label}']=value
   mixed=(mixed+(down*alpha[j]).astype(np.float32)).astype(np.float32) if j<6 else (mixed+down).astype(np.float32)
  saved[f'output{fi}']=mixed;records.append(dict(fixture=fi,output_sha256=bh(mixed.tobytes())))
 np.savez(OUT/'fixtures_and_canonical_outputs.npz',**saved)
 summary=dict(status='MATERIALIZED',experts=experts,shared=True,layer=0,TRAIN_route=dict(archive=str(routes),sha256=sha(routes),prompt=pid,prompt_index=0,decode_step=0,layer=0),fixture_input_source=str(OLD/'fixtures.npz'),fixture_input_sha256=sha(OLD/'fixtures.npz'),fixtures=records,inventory=inventory,source_sha256=sha(__file__),numeric_contract='Existing GGUF-to_float fullrow W8 and logical F32 scales copied byte-exact and matched TRAIN FNV. Whole-vector dynamic A8[-127,127], F32 maxabs/127 and ties-away, exact I64 dot -> F32 once; F32(rowScale*inputScale) once, separate multiply. Existing fixture host F32 SwiGLU; no new inference/quality.',weight_quantization_performed=False,original_GGUF_opened=False,scope='Construction proof only; original v4 geometry is not W8 TRAIN v5/v6 L0 winner.',materialize_seconds=time.monotonic()-began)
 dump(OUT/'summary.json',summary);print(json.dumps(summary,indent=2))
 subprocess.run([sys.executable,str(ROOT/'scripts/research_20261102_routed_boundary_replay.py')],check=True)
if __name__=='__main__':main()
