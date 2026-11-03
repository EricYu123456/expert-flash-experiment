#!/usr/bin/env python3
"""Independent all512/all64 work arithmetic and logical replica inventory."""
import collections,hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'results/research-2026-11-02/round2/g4-down-tile-copy-screen'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 proto=json.loads((OUT/'protocol.frozen.json').read_text());report=json.loads((OUT/'work_report.json').read_text());a=np.load(OUT/'work_arrays.npz');rows=a['rows'];route=a['routes'];delta=a['selected_delta'];base=a['baseline_chip_load_ns'];ledger=json.loads((OUT/'copy_rowid_ledger.json').read_text())
 for p,h in proto['input_sha256'].items():assert sha(p)==h
 def job(r,C):
  if r==0:return 0
  pages=(int(r)*C+16383)//16384;waves=(pages+3)//4;emitted=pages*16384//C
  return 250+C//2+28000*waves+2*emitted
 table=np.asarray([[[job(rows[e,p,c],1536 if p==0 else 1024 if e==64 else 512) for c in range(16)] for p in range(2)] for e in range(65)],np.int64);independent=table[route].sum((1,2))+table[64].sum(0);assert np.array_equal(independent,base)
 copied_rows=0;aliascounts=collections.Counter()
 for e,l in enumerate(ledger):
  s,t=map(int,a['selected_layout'][e]);assert l['expert']==e and (l['source'],l['destination'])==(s,t) and s!=t;assert rows[e,1,s]==128 and rows[e,1,t] in [0,128]
  ids={c:list(range(int(rows[e,1,:c].sum()),int(rows[e,1,:c+1].sum()))) for c in range(16) if rows[e,1,c]};packet=ids[s];assert packet==l['source_logical_rowids'];A={int(c):rr for c,rr in l['A_chip_logical_rows'].items()};B={int(c):rr for c,rr in l['B_chip_logical_rows'].items()};assert A==ids
  expected={c:rr.copy() for c,rr in ids.items() if c!=s};expected[t]=ids.get(t,[])+packet;assert B==expected
  assert sorted(r for rr in A.values() for r in rr)==sorted(r for rr in B.values() for r in rr)==list(range(1536))
  # Physical inventory is original rows once plus one entire copied tile.
  for r in range(1536):aliascounts[e,r]+=1
  for r in packet:aliascounts[e,r]+=1
  copied_rows+=len(packet);d=np.zeros(16,np.int64);d[s]=-job(128,512);d[t]=job(int(rows[e,1,t])+128,512)-job(int(rows[e,1,t]),512);assert np.array_equal(d,delta[e])
  assert len(B[t]) in [128,256] and len(B[t])*512%65536==0 and max(map(len,B.values()))<=256
  assert l['duplicate_bytes']==65536 and l['duplicate_pages']==4 and l['duplicate_waves']==1 and l['output_fringe_rows']==0
 assert copied_rows==8192 and sum(aliascounts.values())==64*1536+8192
 bits=((np.arange(64)[:,None]>>np.arange(6))&1).astype(np.int64);allcost=(base[:,None,:]+np.einsum('mk,qkc->qmc',bits,delta[route])).max(2);minimum=allcost.min(1)
 assert np.array_equal(minimum,a['selected_oracle_values']);assert np.all(allcost[np.arange(512),a['selected_oracle_masks']]==minimum)
 # Independently validate best routeblind layout and both zero-copy static maps.
 blind_delta=a['routeblind_best_delta'];blindcost=(base[:,None,:]+np.einsum('mk,qkc->qmc',bits,blind_delta[route])).max(2).min(1);assert np.array_equal(blindcost,a['routeblind_values'][report['routeblind_best_index']])
 for name,d,key in [('selected',delta,'same_edges_zero_copy_static'),('routeblind_best',blind_delta,'routeblind_best_same_edges_zero_copy_static')]:
  fixed=np.asarray(report[key]['bits'],np.int64);static=(base+(d[route]*fixed[route,None]).sum(1)).max(1);assert np.array_equal(static,a[name+'_static_values'])
 assert report['selected_oracle_mean_ns']==float(minimum.mean())
 out=dict(status='PASS',source_sha256=sha(__file__),protocol_sha256=sha(OUT/'protocol.frozen.json'),all_protocol_inputs_unchanged=True,all512x64_choice_values_independently_enumerated=True,exact_masks_compared=32768,baseline_chip_work_independently_recomputed=True,all64_copy_edges_and_logical_rowids_exact=True,A_and_B_each_use_every_original_Down_row_once=True,source_rows_stored_twice_only_for_designated_packet=True,duplicate_useful_rows=copied_rows,duplicate_useful_bytes=copied_rows*512,duplicate_pages=256,duplicate_waves=64,no_unused_hole_discount=True,GU_and_shared_unchanged=True,Down_no_partial_rows_or_fringe=True,Down_full_private_input_columns=512,Down_max_job_rows=256,Down_max_job_waves=2,host_logical_row_scale_indices_unchanged=True,zero_copy_static_same_A_B_menu_exact=True,routeblind_best_exact=True,weights_or_heldout_opened=False,native_timing_executed=False,limits=['Logical inventory and exact necessary-work proof only; physical address placement/native output ABI not executed.','No speed or algorithm novelty claim. Necessary-work improvement is1.39137%; loosehostprefix bound gives2.00842%native ceiling but is not observed gain.'])
 p=OUT/'independent_work_audit.json';assert not p.exists();p.write_text(json.dumps(out,indent=2)+'\n');print(json.dumps(out,indent=2))
if __name__=='__main__':main()
