"""Frozen PG quality summary/phase helpers. Imports read source/libs only."""
import json
from pathlib import Path
import sys
import math
ROOT=Path('/home/ericyu/expert_flash_experiment');ROUND=ROOT/'results/research-2026-11-03-no-cr/round02-local-affine';REFRESH=ROUND/'stage2-refresh';FREF=REFRESH/'freezes/rev002';OUT=ROUND/'stage2-pg19-quality';PROTOCOL=ROUND/'stage2_pg19_quality.protocol.frozen.json';EXPECTED='3f95838e8af920ad6fe477ee0d11153621680c10488452cd7585fc1b4273ac01';REFRESH_FREEZE='8f89fc18a3583e28e22ff0129c5ac09eb3d23a06fdcf09ec6aaddcf333e9206b'
sys.path.insert(0,str(FREF))
import research_20261103_local_affine_stage2_refresh_core as r
assert Path(r.__file__).resolve()==FREF/'research_20261103_local_affine_stage2_refresh_core.py'
np,w,old,core=r.np,r.w,r.old,r.core
MODELS=['deepseek','olmoe'];ARMS=[0,3,5,6];ZONES=['unscored_inputs_0_254','scored_predecessors_255_510','unused_input_511']
def safe_exponential(value,minus_one=False):
    value=float(value);assert math.isfinite(value)
    try:return (math.expm1(value) if minus_one else math.exp(value)),False
    except OverflowError:return None,True
def zone(pos):
    assert 0<=pos<512
    return ZONES[0 if pos<255 else 1 if pos<511 else 2]
def context_plan(book_ids):
    assert len(book_ids)==len(set(book_ids))==16;counter=0;jobs=[];contexts={}
    for bi,book in enumerate(book_ids):
        modes=[('source_nohook',0),('source',0),('copyoriginal',2),('B',3),('copyB',4),('C',5),('D',6)] if bi==0 else [('source',0),('B',3),('C',5),('D',6)];job={'book_id':book,'contexts':{}}
        for key,mode in modes:counter+=1;job['contexts'][key]=counter;contexts[str(counter)]={'book_id':book,'mode':mode}
        jobs.append(job)
    assert counter==67;return jobs,contexts
def authorize(freeze,review):
    fp=Path(freeze);f=r.read(fp);a=r.read(review);assert a['status']=='approved' and a['phase']=='PG19_QUALITY'
    assert a['protocol_sha256']==f['protocol_sha256']==EXPECTED and r.sha(PROTOCOL)==EXPECTED
    assert a['source_revision']==f['revision'] and a['source_freeze_sha256']==r.sha(fp)
    assert a['refresh_source_freeze_sha256']==f['refresh_source_freeze_sha256']==REFRESH_FREEZE
    for p,h in f['files'].items():assert r.sha(p)==h,p
    fit=r.read(REFRESH/'FIT_REFRESH.complete.json');assert a['FIT_REFRESH_complete_sha256']==r.sha(REFRESH/'FIT_REFRESH.complete.json') and fit['status']=='complete_refreshed_artifacts_frozen_before_PGvalidation' and fit['source_freeze_sha256']==REFRESH_FREEZE
    registry=Path(fit['registry_path']);assert a['registry_sha256']==fit['registry_sha256']==r.sha(registry)
    assert r.read(registry)['refresh_protocol_sha256']==r.EXPECTED
    return f,fit,registry
def fit_dependency_identities(fit):
    identities={}
    for p,h in fit['files'].items():
        before=Path(p).stat();actual=r.sha(p);after=Path(p).stat();assert actual==h and all(getattr(before,k)==getattr(after,k) for k in ['st_size','st_mtime_ns','st_dev','st_ino']),p
        identities[p]={'sha256':actual,'size':after.st_size,'mtime_ns':after.st_mtime_ns,'dev':after.st_dev,'inode':after.st_ino}
    return identities
def paired_summary(nll,book_ids,repeats=10000,seed=2026110302):
    """Exact prespecified two-model centered-max-|t| shared-book bootstrap."""
    assert repeats==10000 and seed==2026110302;values=np.asarray(nll,np.float64);assert values.shape==(2,16,4) and len(book_ids)==16 and len(set(book_ids))==16
    result={'scope':'paired16book custom-subset tokenPPL, not official full PG19 wordPPL or cross-tokenizer comparison','book_ids':list(book_ids),'model_order':MODELS,'arm_order':ARMS,'repeats':repeats,'seed':seed,'margin_NLL':math.log(1.02),'relative_PPL_budget':.02,'classification':'Root+critic only; no automatic success/failure','zeroSD_policy':'no literal shortcut; any zero/nonfiniteSD conservatively makes full primaryfamily inconclusive','models':{}}
    if not np.isfinite(values).all():result.update(primary_family_inconclusive=True,reason='missing book or nonfinite meanNLL, no dropping or redrawing',cutoff=None);return result
    rng=np.random.default_rng(seed);draws=rng.integers(0,16,size=(repeats,16));primary=values[:,:,2]-values[:,:,1];hat=primary.mean(axis=1);boot=primary[:,draws].mean(axis=2);sd=np.std(boot,axis=1,ddof=1);undefined=not (np.isfinite(sd).all() and np.all(sd>0));cutoff=None if undefined else float(np.quantile(np.max(np.abs((boot-hat[:,None])/sd[:,None]),axis=0),.95,method='linear'));result.update(primary_family_inconclusive=undefined,cutoff=cutoff,reason='zero/nonfinite bootstrapSD; both endpoints preserved, no redraw/drop' if undefined else None,shared_draws_array_hash=old.array_hash(draws),bootstrap_SD_ddof=1,quantile_method='linear')
    comparisons={'B_minus_source':(1,0),'C_minus_B':(2,1),'C_minus_source':(2,0),'D_minus_B':(3,1),'D_minus_source':(3,0),'C_minus_D':(2,3)}
    for mi,name in enumerate(MODELS):
        arms={};parts={}
        for ai,arm in enumerate(ARMS):
            mean=float(values[mi,:,ai].mean());ppl,overflow=safe_exponential(mean);per=[safe_exponential(v) for v in values[mi,:,ai]];arms[str(arm)]={'meanNLL':mean,'custom_subset_tokenPPL':ppl,'custom_subset_tokenPPL_overflow':overflow,'per_book_meanNLL':values[mi,:,ai].tolist(),'per_book_custom_subset_tokenPPL':[p[0] for p in per],'per_book_custom_subset_tokenPPL_overflow':[p[1] for p in per]}
        for label,(a,b) in comparisons.items():
            delta=values[mi,:,a]-values[mi,:,b];bs=delta[draws].mean(axis=1);relative,overflow=safe_exponential(delta.mean(),True);parts[label]={'mean_delta_NLL':float(delta.mean()),'relative_custom_subset_PPL_change':relative,'relative_custom_subset_PPL_overflow':overflow,'per_book_delta_NLL':delta.tolist(),'descriptive_paired_book_percentile95':np.quantile(bs,[.025,.975],method='linear').tolist(),'primary_decision_interval':False}
        band=None if undefined else [float(hat[mi]-cutoff*sd[mi]),float(hat[mi]+cutoff*sd[mi])];relative=None if band is None else [safe_exponential(v,True) for v in band];result['models'][name]={'arms':arms,'comparisons':parts,'primary_C_minus_B':{'Delta_hat':float(hat[mi]),'bootstrapSD_ddof1':float(sd[mi]),'simultaneous95_NLL_band':band,'simultaneous95_relative_PPL_band':None if relative is None else [p[0] for p in relative],'simultaneous95_relative_PPL_overflow':None if relative is None else [p[1] for p in relative],'per_book_delta_NLL':primary[mi].tolist()},'source_decomposition_identity_max_error':float(np.max(np.abs((values[mi,:,2]-values[mi,:,0])-((values[mi,:,2]-values[mi,:,1])+(values[mi,:,1]-values[mi,:,0])))))}
    return result
def summarize_outputs(paths,worker_exposures,books):
    assert len(paths)==len(worker_exposures)==2 and len(books)==len(set(books))==16
    values=np.full((2,16,4),np.nan);perbook=[];exposure=[]
    for mi,(name,path,wp) in enumerate(zip(MODELS,paths,worker_exposures)):
        d=r.read(path);assert [j['book_id'] for j in d['jobs']]==books and d['status'] in ['complete','complete_with_insufficient_books'];events=[json.loads(v) for v in Path(wp).read_text().splitlines()];lookup={}
        for event in events:
            assert event['model']==mi+1 and event['arm'] in [3,4,5,6] and event['zone'] in ZONES and type(event['returned_rows']) is type(event['canonical_fallback_rows']) is int and 0<=event['canonical_fallback_rows']<=event['returned_rows']
            allowed={1:[14,40],13:[10,56]} if mi==0 else {0:[6,17],8:[3,9]};assert event['book_id'] in books and event['layer'] in allowed and event['expert'] in allowed[event['layer']]
            if event['arm'] in [3,4]:assert event['canonical_fallback_rows']==0
            if event['arm']==4:assert event['book_id']==books[0] and event['context']==5
            key=(event['context'],event['layer'],event['expert'],event['zone']);assert key not in lookup;lookup[key]=event
        used=set();contexts=set();runid=None
        for bi,job in enumerate(d['jobs']):
            if job['status']=='insufficient_window':perbook.append({'model':name,'book_id':job['book_id'],'status':'insufficient_window','native_tokens':job['native_tokens']});continue
            assert job['status']=='complete' and len(job['immutable_joint_ids'])==512 and [v['reported_arm'] for v in job['runs']]==ARMS
            if bi==0:
                control=job['prospective_firstbook_controls'];assert all(control.get(k) is True for k in ['prospectively_fixed_first_book','source_nohook_vs_observer_fullvocab_memcmp','source_observer_vs_copyoriginal_trace_memcmp','B_vs_copycanonical_fullvocab_trace_memcmp','fresh512GT_KV_each'])
            else:assert job['prospective_firstbook_controls'] is None
            for ai,run in enumerate(job['runs']):
                assert run['reported_arm']==ARMS[ai] and run['model']==mi+1 and run['internal_hook_mode']==(1 if ai==0 else ARMS[ai]) and isinstance(run['context'],int) and run['context']>0 and run['context'] not in contexts
                contexts.add(run['context']);runid=run['run'] if runid is None else runid;assert run['run']==runid and runid>0
                scores=run['nll_targets'];assert len(scores)==256 and [v['input_position'] for v in scores]==list(range(255,511)) and [v['target_position'] for v in scores]==list(range(256,512)) and [v['target_id'] for v in scores]==job['immutable_joint_ids'][256:];assert run['four_batches_128'] and run['unused511_fed'] and run['fresh_KV'];a=np.asarray([v['nll'] for v in scores],np.float64);assert np.isfinite(a).all();values[mi,bi,ai]=a.mean()
                for layer,experts in ({1:[14,40],13:[10,56]} if mi==0 else {0:[6,17],8:[3,9]}).items():
                    for expert in experts:
                        for z in ZONES:
                            cppcount=run['pilot_exposure'][str(layer)][str(expert)][z];assert type(cppcount) is int and cppcount>=0;row={'model':name,'book_id':job['book_id'],'arm':ARMS[ai],'layer':layer,'expert':expert,'zone':z}
                            if ai==0:row.update(source_potential_live_pilot_hits=cppcount,actual_scattered_rows=0,canonical_fallback_rows=0,approximation_rows=0)
                            else:
                                key=(run['context'],layer,expert,z);event=lookup.get(key);count=0 if event is None else event['returned_rows'];fallback=0 if event is None else event['canonical_fallback_rows'];assert type(count) is type(fallback) is int and 0<=fallback<=count==cppcount,'invalid counts or returned rows versus actual successful C++scatter mismatch'
                                approx=count-fallback if ARMS[ai] in [5,6] else 0
                                if event is not None:used.add(key);assert event['book_id']==job['book_id'] and event['arm']==ARMS[ai] and event['model']==mi+1
                                row.update(source_potential_live_pilot_hits=None,actual_scattered_rows=cppcount,canonical_fallback_rows=fallback,approximation_rows=approx,canonical_B_rows=count if ARMS[ai]==3 else 0)
                            exposure.append(row)
            assert not job['zero_prior_replacements_C_B_but_prior_logits_differ'],'instrumentation issue: both B/C zero replacements over all causal positions0..510 but earlier logits differ'
            perbook.append({'model':name,'book_id':job['book_id'],'status':'complete','meanNLL_by_arm':{str(a):float(values[mi,bi,k]) for k,a in enumerate(ARMS)},'controls':job.get('prospective_firstbook_controls'),'zero_exposure_sanity':True})
        # Firstbook copycanonical control events are retained but outside primary
        # arms; they are explicitly labelled and never merged into quality hits.
        assert all(k in used or lookup[k]['arm']==4 for k in lookup),'unmatched noncontrol worker exposure'
    summary=paired_summary(values,books);summary.update(per_book=perbook,exposure=exposure,exposure_causality='Positions0..510 collectively can influence scored outputs through live state/KV; unused511 is extra diagnostic only, never a causal explanation for these256 target losses.',no_PGvalidation_selection=True,TEST_opened=False);return summary
