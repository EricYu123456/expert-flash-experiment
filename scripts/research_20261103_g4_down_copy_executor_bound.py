#!/usr/bin/env python3
"""Fixed-executor first-dispatch bound and three representative full traces."""
import hashlib,json,pathlib,sys,time
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'results/research-2026-11-02/round2/g4-down-tile-copy-screen';SELF=pathlib.Path(__file__).resolve()
sys.path.insert(0,str(ROOT/'scripts'))
import research_20261103_ordinary_read_backend as backend
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 p=OUT/'executor_bound_protocol.frozen.json';assert not p.exists();proto=json.loads((OUT/'protocol.frozen.json').read_text());inputs=[SELF,OUT/'protocol.frozen.json',OUT/'work_report.json',OUT/'work_arrays.npz',ROOT/'scripts/research_20261103_ordinary_read_native.cpp',ROOT/'scripts/research_20261103_ordinary_read_backend.py'];hashes={str(f):sha(f) for f in inputs};scope='Only the frozen current priority executor, not arbitrary schedulers. Firstselected expert has16positiveGUjobs in canonical chiporder;16commands*250ns=4us finishes before earliest28usGUcomputecompletion, so chipc firstjob command starts host_submit+c*250ns. Ownership cannot finish before this start plus Wc.'
 dump(p,dict(input_sha256=hashes,scope=scope,operation='Recompute all512x64 minimum of host_submit+maxc(Wc+c*250ns), then3representative native/ReferenceGraph traces. No furtherlayoutfit or heavy controls.'))
 a=np.load(OUT/'work_arrays.npz');rows=a['rows'];route=a['routes'];delta=a['selected_delta'];base=a['baseline_chip_load_ns'];s=backend.Service(**proto['service']);assert s.command_ns==250 and s.host_submit_ns==2000 and s.ordinary_read_ns==28000 and np.all(rows[:64,0]>0)
 bits=((np.arange(64)[:,None]>>np.arange(6))&1).astype(np.int64);loads=base[:,None,:]+np.einsum('mk,qkc->qmc',bits,delta[route]);perchoice=s.host_submit_ns+(loads+np.arange(16)*250).max(2);minimum=perchoice.min(1);masks=perchoice.argmin(1).astype(np.int32);baselinebound=s.host_submit_ns+(base+np.arange(16)*250).max(1);assert np.all(baselinebound<=a['baseline_native_ns'])
 # Duplicate only descriptor views, not resident GU/Down weights. Both variants
 # refer to the same logical original rows plus the one copied Down tile.
 views=np.concatenate([rows[:64],rows[:64],rows[64:]],axis=0)
 for e,(source,dest) in enumerate(a['selected_layout']):views[64+e,1,source]=0;views[64+e,1,dest]+=128
 plan=backend.lower_rows(views[:,0],views[:,1],1536,[512]*128+[1024],s);native=backend.NativeReplay(OUT/'executor-bound-build');selected=[0,int(np.flatnonzero(masks)[0]),int(np.argmax([int(m).bit_count() for m in masks]))];selected=list(dict.fromkeys(selected));assert len(selected)==3;traces={};checks=[]
 for q in selected:
  routed=(route[q]+64*bits[masks[q]]).astype(np.int32)[None];got=native.run(plan,routed,128,1,s,trace=True);g=backend.ReferenceGraph.build(plan,np.r_[routed[0],128],s);ref=g.run();assert ref['cost']==int(got['cost'][0]);tr=got['trace'];cmd=tr[(tr[:,0]==1)&(tr[:,4]==0)&(tr[:,5]==0)];assert cmd.shape==(16,8) and cmd[:,6].tolist()==list(range(16));assert np.array_equal(cmd[:,2],2000+np.arange(16)*250);assert np.all(cmd[:,1]==250)
  allcmd=tr[tr[:,0]==1]
  for c in range(16):assert int(allcmd[allcmd[:,6]==c,2].min())==2000+c*250
  assert int(minimum[q])<=int(got['cost'][0]);traces[f'q{q}.native']=tr;traces[f'q{q}.ReferenceGraph']=ref['trace'];checks.append(dict(query=q,chosen_LB_mask=int(masks[q]),B_experts=int(masks[q]).bit_count(),first_GU_command_starts_ns=cmd[:,2].tolist(),native_ns=int(got['cost'][0]),ReferenceGraph_ns=int(ref['cost']),fixed_executor_LB_ns=int(minimum[q]),nodes=len(tr)))
 np.savez_compressed(OUT/'executor_bound_arrays.npz',all64_LB=perchoice,minimum_LB=minimum,minimum_masks=masks,baseline_fixed_executor_LB=baselinebound,**traces)
 saving=float(100*(1-minimum.mean()/a['baseline_native_ns'].mean()));assert hashes=={f:sha(f) for f in hashes};result=dict(status='PASS',source_sha256=sha(SELF),protocol_sha256=sha(p),scope=scope,all512x64_masks_enumerated=True,baseline_native_mean_ns=float(a['baseline_native_ns'].mean()),minimum_fixed_executor_LB_mean_ns=float(minimum.mean()),remaining_native_mean_improvement_upper_bound_pct=saving,below_2pct_finite_layout_gate=bool(saving<2),representative_fulltrace_checks=checks,GU_positive16chips=True,earliest_GU_compute_duration_ns=28000,first16_commands_complete_by_ns=6000,no_selector_cost_added=True,metadata_views_do_not_duplicate_resident_GU=True,limitations=['Bound applies only to current fixed executor and selected fixed copy layout.','Coordinate search is finite, not a global copy-placement optimum or impossibility proof.','Three trace costs validate dispatch assumptions; they are not all512 native performance estimates.','No heavy same-budget controls, extra fitting or heldout reads performed.']);dump(OUT/'executor_bound_report.json',result);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
