#!/usr/bin/env python3
"""Independent fresh process, page pread only; no canonical weight matrices."""
import collections,hashlib,json,math,os,pathlib,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'results/research-2026-11-02/round2/routed-boundary-view-numeric'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def bh(x):return hashlib.sha256(x).hexdigest()
def quant(x):
 m=np.max(np.abs(x));s=np.float32(1) if m==0 else np.float32(m/np.float32(127));v=(x/s).astype(np.float32)
 return np.clip(np.copysign(np.floor(np.abs(v.astype(np.float64))+.5),v),-127,127).astype(np.int8),s
def activate(y):
 n=len(y)//2;g=y[:n];u=y[n:];denom=(np.float32(1)+np.exp(-g).astype(np.float32)).astype(np.float32)
 sig=(np.float32(1)/denom).astype(np.float32);return ((g*sig).astype(np.float32)*u).astype(np.float32)
def same(a,b):assert a.dtype==b.dtype and a.shape==b.shape and a.tobytes()==b.tobytes()
def main():
 began=time.monotonic();s=json.loads((OUT/'summary.json').read_text());m=json.loads((OUT/'manifest.json').read_text());sources=json.loads((OUT/'source_rows.json').read_text());sc=np.fromfile(OUT/'logical_scales.f32',dtype='<f4');fx=np.load(OUT/'fixtures_and_canonical_outputs.npz');fds={};inputs=[OUT/p for p in ['manifest.json','summary.json','source_rows.json','logical_scales.f32','logical_scales.json','fixtures_and_canonical_outputs.npz']]
 for chip in m['chips']:
  p=OUT/chip['path'];assert sha(p)==chip['sha256'];fds[chip['chip']]=os.open(p,os.O_RDONLY);inputs.append(p)
 hashes={str(p):sha(p) for p in inputs}
 def pages(a,count):
  bs=[]
  for page in range(count):
   wl=a['start_wordline']+page//4;plane=page%4;assert wl<192
   at=((a['block']*192+wl)*4+plane)*16384;b=os.pread(fds[a['chip']],16384,at);assert len(b)==16384;bs.append(b)
  return b''.join(bs)
 coverage={k:np.zeros(v['rows'],dtype=np.int16) for k,v in sources.items()};ranges=collections.defaultdict(list);audit_rows=0
 for a in m['allocations']:
  assert a['byte_offset']==(a['block']*192+a['start_wordline'])*65536
  assert a['start_wordline']+a['reserved_waves']<=192
  assert a['pages']==math.ceil(a['weight_bytes']/16384) and a['reserved_waves']==math.ceil(a['weight_bytes']/65536)
  payload=pages(a,a['pages'])[:a['weight_bytes']];assert bh(payload)==a['payload_sha256'];C=a['columns'];assert len(payload)==len(a['rowids'])*C
  for slot,r in enumerate(a['rowids']):
   assert bh(payload[slot*C:(slot+1)*C])==sources[a['matrix']]['row_sha256'][r];coverage[a['matrix']][r]+=1;audit_rows+=1
  ranges[a['chip']].append((a['byte_offset'],a['byte_offset']+a['reserved_waves']*65536))
 for rs in ranges.values():
  rs.sort();assert all(b<=c for (_,b),(c,_) in zip(rs,rs[1:]))
 decision={x['matrix']:x for x in m['decisions']};duplicate=0
 for k,cov in coverage.items():
  expected=np.ones(len(cov),dtype=np.int16)
  if k in decision:expected[decision[k]['boundary_rowids']]+=1;duplicate+=len(decision[k]['boundary_rowids'])
  same(cov,expected)
  sm=m['scale_meta'][k];assert bh(sc[sm['offset']//4:sm['offset']//4+sm['rows']].tobytes())==sources[k]['scale_sha256']
 view_fringe=[]
 for k,variants in m['views'].items():
  for label,vs in variants.items():
   seen=np.zeros(sources[k]['rows'],dtype=np.uint8);assert len(set(v['chip'] for v in vs))==len(vs)
   for v in vs:
    a=m['allocations'][v['allocation']];assert v['rowids']==a['rowids'][:v['rows']];assert v['scale_logical_rows']==v['rowids'];seen[v['rowids']]+=1
    assert v['requested_pages']==math.ceil(v['rows']*v['columns']/16384)
    actual=pages(v,v['requested_pages']);d=v['drain_drop'];full=len(actual)//v['columns'];partial=len(actual)%v['columns']
    assert d['extra_complete_rows']==full-v['rows'] and d['partial_row_bytes']==partial
    assert d['extra_complete_rowids']==a['rowids'][v['rows']:min(full,len(a['rowids']))]
    view_fringe.append(dict(matrix=k,view=label,chip=v['chip'],**d))
   assert np.all(seen==1)
  if k in decision:
   d=decision[k];a=[v['rows'] for v in variants['A']];b=[v['rows'] for v in variants['B']];assert a==d['A_rowcounts'] and b==d['B_rowcounts'];want=a.copy();want[d['source']],want[d['destination']]=want[d['destination']],want[d['source']];assert b==want
 def matvec(k,x,label):
  q,sx=quant(x);R=sources[k]['rows'];out=np.empty(R,dtype=np.float32);integer=np.empty(R,dtype=np.int64);seen=np.zeros(R,dtype=np.uint8);sm=m['scale_meta'][k]
  for v in m['views'][k][label]:
   # Requested pages include fringe; software explicitly drops fringe bytes.
   raw=pages(v,v['requested_pages']);w=np.frombuffer(raw[:v['useful_bytes']],dtype=np.int8).reshape(v['rows'],v['columns']);ids=np.asarray(v['rowids']);z=w.astype(np.int64)@q.astype(np.int64);sw=sc[sm['offset']//4+ids];factor=(sw*sx).astype(np.float32);out[ids]=(z.astype(np.float32)*factor).astype(np.float32);integer[ids]=z;seen[ids]+=1
  assert np.all(seen==1);return out,integer
 records=[];scalars=0;integer_rows=0;matrix_phases=0
 keys=[f'routed.{e}' for e in s['experts']]+['shared']
 # Every independently selected expert A/B mask, for both saved input fixtures.
 for fi in [0,1]:
  x=fx[f'x{fi}'];alpha=fx[f'alpha{fi}']
  for mask in range(64):
   mixed=np.zeros(1536,dtype=np.float32)
   for j,k in enumerate(keys):
    label='B' if j<6 and mask>>j&1 else 'A';gu,gi=matvec(k+'.gu',x,label);same(gu,fx[f'f{fi}.{k}.gu']);same(gi,fx[f'f{fi}.{k}.gu_i64']);act=activate(gu);same(act,fx[f'f{fi}.{k}.activation']);q,qs=quant(act);same(q,fx[f'f{fi}.{k}.down_input_a8']);same(np.asarray(qs),fx[f'f{fi}.{k}.down_input_scale']);down,di=matvec(k+'.down',act,'A');same(down,fx[f'f{fi}.{k}.down']);same(di,fx[f'f{fi}.{k}.down_i64']);matrix_phases+=2;integer_rows+=len(gi)+len(di);scalars+=len(gu)+len(act)+len(down)
    mixed=(mixed+(down*alpha[j]).astype(np.float32)).astype(np.float32) if j<6 else (mixed+down).astype(np.float32)
   same(mixed,fx[f'output{fi}']);scalars+=len(mixed);records.append(dict(fixture=fi,mask=mask,output_sha256=bh(mixed.tobytes())))
 for fd in fds.values():os.close(fd)
 assert hashes=={str(p):sha(p) for p in inputs}
 inv=m['inventory'];assert duplicate==inv['duplicated_boundary_rows'];assert inv['stored_useful_bytes']==sum(a['weight_bytes'] for a in m['allocations']);assert inv['stored_useful_bytes']-inv['base_useful_bytes']==duplicate*1536
 out=dict(status='PASS',source_sha256=sha(__file__),fresh_process=True,materializer_imported=False,canonical_weight_matrices_loaded=False,model_inference=False,weight_quantization=False,cases=len(records),all_64_masks_each_fixture=True,fixtures=2,matrix_phases=matrix_phases,compared_F32_scalars=scalars,compared_I64_rows=integer_rows,bit_mismatches=0,audited_physical_rows=audit_rows,storage_row_coverage_and_copies_exact=True,logical_scale_identity=True,views_each_cover_useful_rows_once=True,each_chip_each_view_single_contiguous_fullrow_GEMV=True,physical_reserved_ranges_disjoint=True,single_block_per_allocation=True,inventory=inv,input_sha256=hashes,wall_seconds=time.monotonic()-began,limitations=['CPU complete useful-row numeric/address proof only; no AiF valid-row/reset/output ABI or timing claim.','Page-rounded short prefixes include boundary-copy complete/partial rows; explicit drain/drop ledger does not establish hardware support.','v4 L0 construction geometry is not v5/v6 W8 TRAIN selected winner. Router coefficients and hidden inputs are existing explicit fixtures, not new model inference or quality.'],results=records)
 p=OUT/'independent_replay.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');(OUT/'drain_drop_ledger.json').write_text(json.dumps(view_fringe,indent=2)+'\n');print(json.dumps({k:out[k] for k in ['status','cases','matrix_phases','compared_F32_scalars','compared_I64_rows','bit_mismatches','inventory','wall_seconds']},indent=2))
if __name__=='__main__':main()
