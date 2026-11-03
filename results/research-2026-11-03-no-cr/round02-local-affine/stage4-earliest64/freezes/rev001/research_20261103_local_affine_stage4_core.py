"""Stage4 helpers. Import opens source/libraries only; payload entry points are explicit."""
from collections import Counter
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import sys

ROOT=Path('/home/ericyu/expert_flash_experiment')
ROUND=ROOT/'results/research-2026-11-03-no-cr/round02-local-affine'
OUT=ROUND/'stage4-earliest64'
PROTOCOL=ROUND/'stage4_earliest64.protocol.frozen.json'
EXPECTED='32213b7b76cba5c18254f8bb13c7cfb8091c1db6c800468a541c4020e66c6b33'
APPROVAL=ROUND/'Root_stage4_SOURCE_BUILD_GENERATED.approved.json'
APPROVAL_SHA='a76d20aa1c7be08c3fa9f354b861b6b25bebfdcd6a7e2ed8807336f545e5a801'
F3=ROUND/'stage3-control-cost/freezes/rev006'
REF=ROUND/'stage2-refresh'
sys.path.insert(0,str(F3))
import research_20261103_local_affine_stage3_core as upstream
assert Path(upstream.__file__).resolve()==F3/'research_20261103_local_affine_stage3_core.py'
np,old,core,pipe,r,w=upstream.np,upstream.old,upstream.core,upstream.pipe,upstream.r,upstream.w
read,sha,now=upstream.read,upstream.sha,upstream.now
PHASES=['BIND_EXISTING64','AUDIT_PREFLIGHT','FIT_C64','ACQUIRE_FRESH16_TOKENIZE','QUALITY_C64']
B64,COPY_B64,C64,COPY_C64=9,10,11,12
ZONES=['unscored_inputs_0_254','scored_predecessors_255_510','unused_input_511']

def spec():
    assert sha(PROTOCOL)==EXPECTED
    return read(PROTOCOL)

def config(name):return next(m for m in spec()['models'] if m['name']==name)

def atomic_json(path,value):
    path=Path(path);assert not path.exists(),str(path)
    partial=path.with_name(path.name+'.pending');assert not partial.exists(),str(partial)
    with partial.open('x') as f:
        json.dump(value,f,indent=2,allow_nan=False);f.write('\n');f.flush();os.fsync(f.fileno())
    os.replace(partial,path)
    fd=os.open(path.parent,os.O_RDONLY|os.O_DIRECTORY)
    try:os.fsync(fd)
    finally:os.close(fd)

def stat(path):
    s=Path(path).stat();return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,dev=s.st_dev,inode=s.st_ino)

def identities(files):
    out={}
    for path,h in files.items():
        a=stat(path);actual=sha(path);b=stat(path);assert a==b and actual==h,str(path)
        out[str(path)]={**b,'sha256':actual}
    return out

def verify_unit(path,expected=None):
    d=read(path);assert d['status']=='complete'
    if expected is not None:
        for k,v in expected.items():assert d[k]==v,(str(path),k)
    identities(d['files']);return d

def old_checks():
    a=read(APPROVAL);assert sha(APPROVAL)==APPROVAL_SHA
    for path,h in a['old_sources_before_sha256'].items():assert sha(path)==h,path
    return a['old_sources_before_sha256']

def authorize(freeze,review,phase,verify_prior_payload=True):
    f,a=read(freeze),read(review);assert phase in PHASES and a['status']=='approved' and a['phase']==phase
    assert a['protocol_sha256']==f['protocol_sha256']==EXPECTED==sha(PROTOCOL)
    assert a['source_revision']==f['revision'] and a['source_freeze_sha256']==sha(freeze)
    assert a['parent_receipts']==f['parent_receipts']
    identities(f['files']);identities(f['parent_receipts']);old_checks()
    for prior in PHASES[:PHASES.index(phase)]:
        p=OUT/(prior+'.complete.json');assert a[prior+'_complete_sha256']==sha(p)
        prior_receipt=read(p);assert prior_receipt['status']=='complete'
        if verify_prior_payload:
            identities(prior_receipt['files'])
            identities(prior_receipt.get('runtime_payload_files',{}))
    if phase in ['ACQUIRE_FRESH16_TOKENIZE','QUALITY_C64']:
        assert a['registry_sha256']==sha(OUT/'C64_registry.frozen.json')
    if phase=='FIT_C64':
        existing=list((OUT/'fit').glob('*/*/*.complete.json'))
        if existing:
            c=a.get('continuation');assert c and c['approved'] is True
            assert c['completed_units']=={str(p):sha(p) for p in existing}
            assert c['additional_budget_seconds']>0 and c['cumulative_budget_seconds']>=c['additional_budget_seconds']
    return f,a

def finish(phase,freeze,paths,extra=None):
    value={'status':'complete','phase':phase,'at_utc':now(),'protocol_sha256':EXPECTED,'source_freeze_sha256':sha(freeze),'files':{str(p):sha(p) for p in paths},'TEST_opened':False,'automatic_scientific_classification':False,**(extra or {})}
    atomic_json(OUT/(phase+'.complete.json'),value);return value

def disk_check(extra=0):
    used=sum(p.stat().st_size for p in OUT.rglob('*') if p.is_file())
    free=shutil.disk_usage(OUT).free
    assert extra>=0 and used+extra<=64000000000 and free-extra>=32000000000,('disk cap/reserve',used,free,extra)
    return {'at_utc':now(),'stage4_file_bytes':used,'free_bytes':free,'prospective_extra_bytes':extra}

def model_source(name):
    m=next(m for m in read(ROUND/'stage0_protocol.frozen.json')['models'] if m['name']==name)
    ident=next(i for i in read(ROUND/'stage0/content_identities.frozen.json')['models'] if i['path']==m['gguf'])
    assert stat(m['gguf'])=={k:ident[k] for k in stat(m['gguf'])}
    return m,ident

def provenance(name):
    m=config(name);path=REF/'fitted'/name/f'layer{m["layer"]}'/'union_provenance.jsonl'
    rows=[json.loads(v) for v in path.read_text().splitlines()]
    assert len(rows)==m['union_rows']
    for i,p in enumerate(rows):assert p['row']==i and p['model']==name and p['layer']==m['layer'] and p['origin'] in ['oldFIT','pg19train']
    return rows

def preflight_selection(name,rows):
    m=config(name);strata=[('oldFIT',128),('PGprefix',64),('Bdecode',22),('Cdecode',21),('Ddecode',21)]
    def admit(p,s):
        if s=='oldFIT':return p['origin']=='oldFIT'
        if p['origin']!='pg19train':return False
        if s=='PGprefix':return p['phase']=='prefix' and p['arm']==3
        return p['phase']=='decode' and p['arm']=={'Bdecode':3,'Cdecode':5,'Ddecode':6}[s]
    selected=[]
    for st,quota in strata:
        prefix=b'stage4-preflight-row-v1\0'+name.encode()+b'\0'+str(m['layer']).encode()+b'\0'+st.encode()+b'\0'
        pool=[]
        for i,p in enumerate(rows):
            assert p['row']==i and p['model']==name and p['layer']==m['layer']
            if admit(p,st):
                b=json.dumps(p,sort_keys=True,separators=(',',':'),ensure_ascii=True,allow_nan=False).encode()
                pool.append((hashlib.sha256(prefix+b).digest(),i,p))
        assert len(pool)>=quota,st
        selected += [{'stratum':st,'row':i,'digest':h.hex(),'original_record':p} for h,i,p in sorted(pool,key=lambda a:(a[0],a[1]))[:quota]]
    assert len(selected)==len({p['row'] for p in selected})==256
    four=[next(p['row'] for p in selected if p['stratum']==s) for s in ['oldFIT','PGprefix','Bdecode','Ddecode']]
    expert=min(m['new_experts'],key=lambda e:(hashlib.sha256(f'stage4-preflight-v1/{name}/{m["layer"]}/{e}'.encode()).digest(),e))
    return {'model':name,'layer':m['layer'],'expert':expert,'rows':selected,'four_audit_rows':four,'selection_before_values':True}

def load_original_union(name):
    m=config(name);layer=m['layer'];fitids=[v['id'] for v in read(core.STAGE0/'content_identities.frozen.json')['records'] if v['split']=='FIT']
    assert len(fitids)==120
    ds=pipe.dataset(core.STAGE0/'capture'/name/'FIT',layer,fitids)
    nf=REF/'collected'/name/f'layer{layer}'
    x=np.concatenate([ds['x'],np.load(nf/'inputs.npy',allow_pickle=False)])
    ids=np.concatenate([ds['ids'],np.load(nf/'ids.npy',allow_pickle=False)])
    alpha=np.concatenate([ds['alpha'],np.load(nf/'alpha.npy',allow_pickle=False)])
    identity=read(REF/'fitted'/name/f'layer{layer}'/'union_input_identity.json')
    assert x.shape==(m['union_rows'],2048) and x.dtype==np.float32 and np.isfinite(x).all()
    for a,key in [(x,'input_array_hash'),(ids,'live_ids_array_hash'),(alpha,'live_alpha_array_hash'),(np.ones(len(x),np.bool_),'include_allrows_array_hash')]:assert old.array_hash(a)==identity[key]
    rows=provenance(name);assert len(rows)==len(x)
    return x,ids,alpha,rows,identity

def load_bound_union(name):
    m=config(name);folder=OUT/'bound'/name
    arrays=[np.load(folder/(n+'.npy'),mmap_mode='r',allow_pickle=False) for n in ['inputs','ids','alpha']]
    receipt=read(folder/'union.json')
    for a,k in zip(arrays,['input_array_hash','live_ids_array_hash','live_alpha_array_hash']):assert old.array_hash(a)==receipt['union_identity'][k]
    rows=[json.loads(v) for v in (folder/'union_provenance.jsonl').read_text().splitlines()]
    assert len(rows)==len(arrays[0])==m['union_rows']
    return *arrays,rows,receipt

def local_hash(m):return old.array_hash(m['qmaps'],m['scales'],m['bias'],np.asarray(m['unsupported'],np.int32))

def validate_local(m):
    assert m['qmaps'].dtype==np.int8 and m['qmaps'].shape==(32,2048,128)
    assert m['scales'].dtype==m['bias'].dtype==np.float32 and m['scales'].shape==m['bias'].shape==(32,2048)
    assert np.isfinite(m['scales']).all() and np.all(m['scales']>0) and np.isfinite(m['bias']).all()
    assert len(m['unsupported'])==len(set(m['unsupported'])) and all(0<=c<32 for c in m['unsupported'])

def bits(a,b):
    assert a.shape==b.shape and a.dtype==b.dtype==np.float32 and np.array_equal(a.view(np.uint32),b.view(np.uint32))

class Meter:
    """Disjoint invoked primitive counts; preserves the pinned numeric call order."""
    def __init__(self,num):self.base=num;self.label='unlabelled';self.counts={}
    def __getattr__(self,k):return getattr(self.base,k)
    @contextmanager
    def category(self,label):
        previous=self.label;self.label=label
        try:yield self
        finally:self.label=previous
    def add(self,k,v):self.counts.setdefault(self.label,Counter())[k]+=int(v)
    def quantize(self,x):self.add('quantized_elements',x.size);return self.base.quantize(x)
    def dots(self,q,weight,audit=True):
        before=self.base.dot_audits;z=self.base.dots(q,weight,audit=audit)
        self.add('integer_MAC',len(q)*len(weight)*q.shape[1]);self.add('internal_sampled_I64_MAC',(self.base.dot_audits-before)*q.shape[1]);return z
    def rescale(self,d,sw,sx):self.add('rescaled_F32_elements',d.size);return self.base.rescale(d,sw,sx)
    def linear(self,q,sx,weight,sw):return self.rescale(self.dots(q,weight),sw,sx)
    def swiglu(self,g,u):self.add('whole_I_product_elements',g.size);self.add('pinned_SiLU_and_fused_audit_elements_each',g.size);return self.base.swiglu(g,u)
    def add_bias(self,y,b):self.add('F32_bias_add_elements',y.size);return self.base.add_bias(y,b)
    def bias(self,y,p,weight):self.add('sequential_F64_weighted_residual_elements',y.size);return self.base.bias(y,p,weight)
    def project(self,x,P):self.add('P_FP64_MAC',len(x)*x.shape[1]*len(P));return self.base.project(x,P)
    def tree_project(self,x,n):self.add('tree_FP64_MAC',x.size);return self.base.tree_project(x,n)
    def receipt(self):return {k:dict(v) for k,v in self.counts.items()}

def context_plan(books):
    jobs=[];contexts={};n=0
    for bi,b in enumerate(books):
        modes=[('source_nohook',0),('source',1),('copyoriginal',2),('B64',9),('copyB64',10),('C64',11),('copyC64',12)] if bi==0 else [('source',1),('B64',9),('C64',11)]
        job={'book_id':b,'contexts':{}}
        for key,mode in modes:n+=1;job['contexts'][key]=n;contexts[str(n)]={'book_id':b,'mode':mode}
        jobs.append(job)
    assert len(books)==len(set(books))==16 and n==52
    return jobs,contexts

def paired_summary(values,books):
    v=np.asarray(values,np.float64);assert v.shape==(2,16,3) and len(books)==len(set(books))==16
    result={'book_ids':list(books),'primary_family':'two model C64-minus-B64 paired-book meanNLL endpoints','primary_family_inconclusive':True,'seed':2026110304,'repeats':10000,'ddof':1,'quantile_method':'linear','margin_NLL':math.log(1.02),'automatic_scientific_classification':False,'models':{}}
    if not np.isfinite(v).all():result['reason']='missing/nonfinite book or endpoint; no drop/redraw';return result
    delta=v[:,:,2]-v[:,:,1];hat=delta.mean(axis=1);draws=np.random.default_rng(2026110304).integers(0,16,(10000,16));boot=delta[:,draws].mean(axis=2);sd=boot.std(axis=1,ddof=1);valid=np.isfinite(sd).all() and (sd>0).all()
    cutoff=float(np.quantile(np.max(np.abs((boot-hat[:,None])/sd[:,None]),axis=0),.95,method='linear')) if valid else None
    both=bool(valid and np.all(hat+cutoff*sd<math.log(1.02)))
    any_harm=bool(valid and np.any(hat-cutoff*sd>math.log(1.02)))
    result.update(bootstrap_defined=bool(valid),both_upper_strictly_below_margin=both,any_lower_strictly_above_margin=any_harm,primary_family_inconclusive=not bool(valid) or not (both or any_harm),budget_disposition='Root+critic decide; interval relation only',interval_relation='both_upper_below' if both else 'any_lower_above' if any_harm else 'crossing_boundary_or_undefined',reason=None if valid else 'zero/nonfinite bootstrapSD; wholefamily retained',cutoff=cutoff,shared_draws_array_hash=old.array_hash(draws))
    def expm(x):
        try:return {'relative_PPL_change':math.expm1(float(x)),'overflow':False}
        except OverflowError:return {'relative_PPL_change':None,'overflow':True}
    for i,name in enumerate(['deepseek','olmoe']):
        band=[float(hat[i]-cutoff*sd[i]),float(hat[i]+cutoff*sd[i])] if valid else None
        result['models'][name]={'meanNLL_source_B64_C64':v[i].mean(axis=0).tolist(),'per_book_meanNLL_source_B64_C64':v[i].tolist(),'primary_C64_minus_B64':{'Delta_hat':float(hat[i]),'bootstrapSD':float(sd[i]),'simultaneous95_NLL_band':band,'simultaneous95_relative_PPL_band':None if band is None else [expm(t) for t in band],'per_book_delta':delta[i].tolist()},'descriptive_B64_minus_source':(v[i,:,1]-v[i,:,0]).tolist(),'descriptive_C64_minus_source':(v[i,:,2]-v[i,:,0]).tolist()}
    return result
