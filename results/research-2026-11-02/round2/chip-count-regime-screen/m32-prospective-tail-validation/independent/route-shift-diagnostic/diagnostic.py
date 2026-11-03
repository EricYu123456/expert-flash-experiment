#!/usr/bin/env python3
"""Descriptive route shift only. Analyze new routes only after capture is sealed."""
from pathlib import Path
from datetime import datetime,timezone
from itertools import combinations
import argparse,hashlib,json,math
import numpy as np

HERE=Path(__file__).resolve().parent
BASE=HERE.parents[2]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def dump(p,x):Path(p).write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')

def validate_routes(a):
 a=np.asarray(a)
 assert a.ndim==2 and a.shape[1]==6 and np.issubdtype(a.dtype,np.integer)
 assert np.all((a>=0)&(a<64)) and all(len(set(r.tolist()))==6 for r in a)
 return np.sort(a.astype(np.int32),axis=1)

def jsd_ln(p,q):
 p=np.asarray(p,np.float64);q=np.asarray(q,np.float64);m=(p+q)/2
 assert np.all(p>=0) and np.all(q>=0) and np.isclose(p.sum(),1) and np.isclose(q.sum(),1)
 pm=p>0;qm=q>0
 value=.5*float(np.sum(p[pm]*np.log(p[pm]/m[pm])))+.5*float(np.sum(q[qm]*np.log(q[qm]/m[qm])))
 assert -1e-14<=value<=math.log(2)+1e-14
 return max(0.,value)

def features(train,new):
 t=validate_routes(train);n=validate_routes(new)
 seen={tuple(row) for row in t.tolist()}
 unique=[set(row) for row in seen]
 pairs={pair for row in seen for pair in combinations(row,2)}
 out=[]
 for row in n.tolist():
  ss=set(row);exact=tuple(row) in seen
  maximum=max(len(ss&old) for old in unique)
  unseen=sum(pair not in pairs for pair in combinations(row,2))
  assert exact==(maximum==6) and 0<=unseen<=15
  out.append({'sorted_top6':row,'exact_TRAIN_set_seen':exact,'max_TRAIN_set_intersection':maximum,'TRAIN_unseen_pair_count':unseen})
 return out

def summarize(train,new,feat):
 t=validate_routes(train);n=validate_routes(new);assert len(n)==len(feat)
 def bins(key,end):
  return [{'bin':i,'queries':sum(f[key]==i for f in feat),
    'fraction':sum(f[key]==i for f in feat)/len(feat) if feat else None} for i in range(end+1)]
 p=np.bincount(t.ravel(),minlength=64).astype(np.float64)/(len(t)*6)
 counts=np.bincount(n.ravel(),minlength=64).astype(np.int64)
 q=counts.astype(np.float64)/(len(n)*6) if len(n) else None
 l1=float(np.abs(p-q).sum()) if q is not None else None
 return {'queries':len(n),'exact_set_coverage_count':sum(f['exact_TRAIN_set_seen'] for f in feat),
   'exact_set_coverage_fraction':sum(f['exact_TRAIN_set_seen'] for f in feat)/len(feat) if feat else None,
   'max_intersection_bins_0_to_6':bins('max_TRAIN_set_intersection',6),
   'unseen_pair_bins_0_to_15':bins('TRAIN_unseen_pair_count',15),
   'unseen_pair_occurrences':sum(f['TRAIN_unseen_pair_count'] for f in feat),
   'unseen_pair_occurrence_fraction':sum(f['TRAIN_unseen_pair_count'] for f in feat)/(len(feat)*15) if feat else None,
   'expert_counts':counts.tolist(),'TRAIN_selected_slot_distribution':p.tolist(),
   'new_selected_slot_distribution':q.tolist() if q is not None else None,
   'selected_slot_marginal_L1':l1,
   'selection_frequency_L1':l1*6 if l1 is not None else None,
   'JSD_natural_log':jsd_ln(p,q) if q is not None else None}

def freeze():
 assert not (HERE/'protocol.frozen.json').exists()
 path=BASE/'stage-b/granite4tiny.M32.minimum_depth.plan.npz'
 with np.load(path) as z:train=z['routes'].copy()
 train=validate_routes(train);assert train.shape==(512,6)
 for source in ['g4-M32-coupled-cp/native/incumbent4.plan.npz','g4-M32-cyclic-rotation-cp/native/incumbent0.plan.npz',
   'g4-M32-independent-rotation-cp/native/incumbent2.plan.npz']:
  with np.load(BASE/source) as z:assert np.array_equal(train,np.sort(z['routes'],axis=1))
 np.save(HERE/'TRAIN_sets.frozen.npy',train)
 protocol={'status':'frozen_before_new_capture_routes_read','utc':datetime.now(timezone.utc).isoformat(),
  'source_sha256':sha(__file__),'TRAIN_source':str(path),'TRAIN_source_sha256':sha(path),
  'TRAIN_sets_path':str(HERE/'TRAIN_sets.frozen.npy'),'TRAIN_sets_sha256':sha(HERE/'TRAIN_sets.frozen.npy'),
  'TRAIN_queries':512,'expert_universe':list(range(64)),'topk':6,
  'identity':'Unordered sorted six-unique-expert tuple. TRAIN duplicates remain in frequencies; unique tuples only for membership/intersection.',
  'exact_set_coverage':'Fraction of all new queries whose sorted top6 tuple appears at least once in TRAIN512.',
  'max_intersection':'For each new tuple, maximum cardinality intersection with any TRAIN tuple. Report all bins0..6, including empty bins.',
  'unseen_pairs':'For each new tuple, among its15 unordered pairs count those absent from the union of all TRAIN pairs. Report every bin0..15, including empty bins.',
  'marginal_definition':'Selected-slot distribution p_e=count(e)/(6*N). Report L1=sum_e|p_e-q_e|. Also report unnormalized per-query selection-frequency L1=6*L1, clearly labeled.',
  'JSD_definition':'Natural-log Jensen-Shannon=.5*sum p*ln(p/m)+.5*sum q*ln(q/m), m=(p+q)/2; zero numerator contributes0, both-zero contributes0, no pseudocount/smoothing. Range[0,ln2]. Empty groups return null.',
  'groups':'Report all512 queries and each of the two frozen source groups independently versus the same full TRAIN512. Source order comes from frozen acquisition metadata, not any outcome.',
  'all_queries':'Keep all32x16 queries. No warmup/first-step/tail/route-novelty filters; all per-query records and all condition bins included.',
  'input_contract':'New complete route archive shape32x16x40x6; rawlayer0 only. Prompt source labels length32, exactly16 per source. Bind acquisition/capture protocols and completed capture manifest hashes after root confirms sealed capture.',
  'excluded_from_this_addendum':'No costs, per-bin performance winner, model fit, plan selection, primary metric/bootstrap change, hypothesis test or causal explanation. Descriptive coverage is not independent performance evidence.',
  'activation':'Only freeze/selfcheck may run now. analyze(binding) is deferred until explicit root confirmation of completed sealed capture.'}
 dump(HERE/'protocol.frozen.json',protocol)
 # Meaningful boundary checks using existing TRAIN and tiny analytical supports.
 fs=features(train,train);sm=summarize(train,train,fs)
 assert sm['exact_set_coverage_count']==512 and sm['selected_slot_marginal_L1']==0 and sm['JSD_natural_log']==0
 toy=np.asarray([[0,1,2,3,4,5]],np.int32);disjoint=np.asarray([[58,59,60,61,62,63]],np.int32)
 tf=features(toy,disjoint);ts=summarize(toy,disjoint,tf)
 assert tf[0]['max_TRAIN_set_intersection']==0 and tf[0]['TRAIN_unseen_pair_count']==15
 assert ts['selected_slot_marginal_L1']==2 and abs(ts['JSD_natural_log']-math.log(2))<1e-14
 empty=summarize(train,np.empty((0,6),np.int32),[]);assert empty['JSD_natural_log'] is None and empty['selected_slot_marginal_L1'] is None
 dump(HERE/'selfcheck.TRAIN_only.json',{'status':'PASS','TRAIN_self_coverage':1.,'TRAIN_self_JSD':0.,
   'disjoint_support_L1':2.,'disjoint_support_JSD_ln':ts['JSD_natural_log'],'empty_group_null':True,
   'new_routes_opened':False,'no_cost_or_inference':True})
 files=[f for f in HERE.rglob('*') if f.is_file() and f.name!='completion.preparation.json']
 dump(HERE/'completion.preparation.json',{'status':'SEALED_RULE_AND_IMPLEMENTATION_PREPARATION_ONLY',
  'files':{str(f.relative_to(HERE)):sha(f) for f in sorted(files)}})
 print(json.dumps({'protocol_sha256':sha(HERE/'protocol.frozen.json'),'source_sha256':sha(__file__),
  'completion_sha256':sha(HERE/'completion.preparation.json')},indent=2))

def analyze(binding_path):
 protocol=json.loads((HERE/'protocol.frozen.json').read_text());assert sha(__file__)==protocol['source_sha256']
 binding=json.loads(Path(binding_path).read_text())
 assert binding['capture_complete'] is True and binding['root_confirmed_sealed_capture'] is True
 for key in ['routes','acquisition_protocol','capture_protocol','capture_manifest']:
  assert sha(binding[key]['path'])==binding[key]['sha256']
 assert sha(protocol['TRAIN_sets_path'])==protocol['TRAIN_sets_sha256']
 train=np.load(protocol['TRAIN_sets_path']);source_keys=binding['source_keys'];sources=binding['prompt_sources']
 assert len(source_keys)==len(set(source_keys))==2 and len(sources)==32
 assert all(sources.count(k)==16 for k in source_keys)
 with np.load(binding['routes']['path']) as z:routes=z[binding['routes']['key']].copy()
 assert routes.shape==(32,16,40,6)
 # Full archive integrity before extracting L0; no query is dropped.
 validate_routes(routes.reshape(-1,6))
 new=routes[:,:,0,:].reshape(512,6);feat=features(train,new)
 for i,f in enumerate(feat):f.update(query_index=i,prompt_index=i//16,decode_step=i%16,source_key=sources[i//16])
 groups={'all':summarize(train,new,feat)}
 for key in source_keys:
  inds=[i for i in range(512) if sources[i//16]==key]
  groups[key]=summarize(train,new[inds],[feat[i] for i in inds])
 out=HERE/'sealed_capture_diagnostic';out.mkdir(exist_ok=False)
 dump(out/'report.json',{'status':'DESCRIPTIVE_ROUTE_SHIFT_ONLY','protocol_sha256':sha(HERE/'protocol.frozen.json'),
  'binding_sha256':sha(binding_path),'groups':groups,'per_query':feat,
  'scope':'All queries retained. No costs, primary/bootstrap modification, winner selection, refit or independent performance evidence.'})
 dump(out/'completion.json',{'status':'SEALED_DESCRIPTIVE_ONLY','files':{'report.json':sha(out/'report.json')}})
 print(json.dumps({'queries':len(feat),'report_sha256':sha(out/'report.json')}))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','analyze']);ap.add_argument('--binding');a=ap.parse_args()
 if a.mode=='freeze':freeze()
 else:assert a.binding;analyze(a.binding)
