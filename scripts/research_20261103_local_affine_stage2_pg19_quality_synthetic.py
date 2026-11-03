#!/usr/bin/env python3
"""Generated fixtures only: fixed bootstrap, report overflow, exact interfaces."""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parent))
import research_20261103_local_affine_stage2_pg19_quality_core as q
from research_20261103_local_affine_stage2_pg19_quality_worker import QualityRegistry
np,r=q.np,q.r
def rejects(fn):
    try:fn()
    except (AssertionError,RuntimeError):return
    raise AssertionError('malformed fixture accepted')
def whole_summary_fixture(folder,books):
    folder.mkdir();jobs,contextmap=q.context_plan(books);paths=[];exposures=[];data=[];events=[];values=np.zeros((2,16,4))
    for mi,name in enumerate(q.MODELS):
        model={'status':'complete','jobs':[]};worker=[];pilots={1:[14,40],13:[10,56]} if mi==0 else {0:[6,17],8:[3,9]}
        for bi,book in enumerate(books):
            runs=[];source=2+bi*.01;v=[source,source+.01,source+.03+(bi-7.5)*(mi+1)*.001,source+.04-bi*.0005];joint=list(range(512));values[mi,bi]=v
            for ai,arm in enumerate(q.ARMS):
                ctx=jobs[bi]['contexts'][['source','B','C','D'][ai]];exposure={str(l):{str(e):{z:1 if arm==0 else 2 for z in q.ZONES} for e in es} for l,es in pilots.items()};runs.append({'model':mi+1,'reported_arm':arm,'internal_hook_mode':1 if arm==0 else arm,'run':100+mi,'context':ctx,'fresh_KV':True,'four_batches_128':True,'unused511_fed':True,'nll_targets':[{'input_position':p,'target_position':p+1,'target_id':joint[p+1],'nll':v[ai]} for p in range(255,511)],'pilot_exposure':exposure})
                if arm:
                    for l,es in pilots.items():
                        for e in es:
                            for z in q.ZONES:worker.append({'model':mi+1,'book_id':book,'context':ctx,'arm':arm,'layer':l,'expert':e,'zone':z,'returned_rows':2,'canonical_fallback_rows':int(arm==5 and z==q.ZONES[1])})
            controls={k:True for k in ['prospectively_fixed_first_book','source_nohook_vs_observer_fullvocab_memcmp','source_observer_vs_copyoriginal_trace_memcmp','B_vs_copycanonical_fullvocab_trace_memcmp','fresh512GT_KV_each']} if bi==0 else None;model['jobs'].append({'book_id':book,'status':'complete','immutable_joint_ids':joint,'runs':runs,'prospective_firstbook_controls':controls,'zero_prior_replacements_C_B_but_prior_logits_differ':False})
        l=min(pilots);worker.append({'model':mi+1,'book_id':books[0],'context':5,'arm':4,'layer':l,'expert':pilots[l][0],'zone':q.ZONES[2],'returned_rows':1,'canonical_fallback_rows':0});mp=folder/(name+'.json');wp=folder/(name+'.exposure.jsonl');r.save(mp,model);wp.write_text(''.join(json.dumps(v)+'\n' for v in worker));paths.append(mp);exposures.append(wp);data.append(model);events.append(worker)
    return paths,exposures,data,events,values
def main(output):
    out=Path(output);folder=out.parent/'generated_quality';folder.mkdir();books=list(range(100,116));paths,exposures,data,events,values=whole_summary_fixture(folder/'screen',books);summary=q.summarize_outputs(paths,exposures,books);assert not summary['primary_family_inconclusive'];assert len(summary['per_book'])==32 and len(summary['exposure'])==2*16*4*4*3
    # Independent scalar-loop construction of every paired-book resample.
    draws=np.random.default_rng(2026110302).integers(0,16,size=(10000,16));d=values[:,:,2]-values[:,:,1];hat=d.mean(axis=1);boot=np.array([[float(np.mean(d[j,row])) for row in draws] for j in range(2)]);sd=np.std(boot,axis=1,ddof=1);cut=float(np.quantile(np.max(np.abs((boot-hat[:,None])/sd[:,None]),axis=0),.95,method='linear'));assert abs(summary['cutoff']-cut)<1e-12
    for j,name in enumerate(q.MODELS):assert np.allclose(summary['models'][name]['primary_C_minus_B']['simultaneous95_NLL_band'],[hat[j]-cut*sd[j],hat[j]+cut*sd[j]],rtol=1e-12,atol=1e-12)
    zero=np.zeros((2,16,4));assert q.paired_summary(zero,books)['primary_family_inconclusive'];missing=values.copy();missing[0,3,2]=np.nan;assert q.paired_summary(missing,books)['primary_family_inconclusive'];missing[0,3,2]=np.inf;assert q.paired_summary(missing,books)['primary_family_inconclusive']
    large=values+800;large[:,:,2]+=800;large_summary=q.paired_summary(large,books);assert not large_summary['primary_family_inconclusive'] and large_summary['models']['deepseek']['arms']['0']['custom_subset_tokenPPL'] is None and large_summary['models']['deepseek']['arms']['0']['custom_subset_tokenPPL_overflow'];assert large_summary['models']['deepseek']['primary_C_minus_B']['simultaneous95_relative_PPL_overflow']==[True,True];json.dumps(large_summary,allow_nan=False)
    rejects(lambda:q.summarize_outputs(paths[:1],exposures,books));bad=copy.deepcopy(data[0]);bad['jobs'][0]['runs'][0]['nll_targets'][0]['target_position']=255;bp=folder/'wrong_index.json';r.save(bp,bad);rejects(lambda:q.summarize_outputs([bp,paths[1]],exposures,books));bad=copy.deepcopy(data[0]);bad['jobs'][1]['runs'][0]['context']=bad['jobs'][0]['runs'][0]['context'];r.save(bp,bad);rejects(lambda:q.summarize_outputs([bp,paths[1]],exposures,books));bad=copy.deepcopy(events[0]);bad[0]['canonical_fallback_rows']=3;bw=folder/'bad_counts.jsonl';bw.write_text(''.join(json.dumps(v)+'\n' for v in bad));rejects(lambda:q.summarize_outputs(paths,[bw,exposures[1]],books));bad=copy.deepcopy(data[0]);bad['jobs'][0]['runs'][0]['pilot_exposure']['1']['14'][q.ZONES[0]]=-1;r.save(bp,bad);rejects(lambda:q.summarize_outputs([bp,paths[1]],exposures,books))
    # Actual worker metadata extension without changing or mocking predictors.
    reg=QualityRegistry.__new__(QualityRegistry);reg.model=1;reg.K=6;jobs,contexts=q.context_plan(books);reg.start_exposure(folder/'worker_generated.jsonl',{'run':12,'contexts':contexts});h=r.w.header(mode=5,arm=5,run=12,context=jobs[0]['contexts']['C'],subrun=5,micro=0,model=1,layer=1,T=128,K=6,position=128,token_phase=0);reg.observe(h,np.zeros((128,2048),np.float32),np.zeros((128,6),np.int32));reg.record_returned([(126,0,14,0,None),(127,0,14,1,None)],5,1);reg.finish_exposure();ev=[json.loads(v) for v in (folder/'worker_generated.jsonl').read_text().splitlines()];assert {e['zone']:e['canonical_fallback_rows'] for e in ev}=={q.ZONES[0]:0,q.ZONES[1]:1}
    file=folder/'dependency.bytes';file.write_bytes(b'generated_model_dependency');fit={'files':{str(file):r.sha(file)}};before=q.fit_dependency_identities(fit);assert q.fit_dependency_identities(fit)==before;file.write_bytes(b'changed');rejects(lambda:q.fit_dependency_identities(fit))
    result={'status':'passed','generated_only':True,'no_actual_FIT_or_PGvalidation_payload':True,'bootstrap_shared10000bookdraws_independent_reference':True,'two_endpoint_family_zero_missing_nonfinite_preserved':True,'largefinite_logNLL_PPL_overflow_JSON_safe':True,'whole_two_model_summary_zone_returned_vs_scattered':True,'negative_count_fallback_context_targetindex_guards':True,'source0_internalObserver1_context_plan67':True,'worker_flag_zones_without_predictor_change':True,'prepost_dependency_hash_change_detection':True};r.save(out,result);print(json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);a=p.parse_args();main(a.output)
