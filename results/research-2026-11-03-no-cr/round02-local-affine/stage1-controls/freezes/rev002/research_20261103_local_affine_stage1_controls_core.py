"""Same-Round2 controls. Imports read source/libraries only; no candidate data."""
import os
for _k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS','OMP_THREAD_LIMIT']:
    os.environ[_k]='8'
os.environ['OMP_DYNAMIC']='FALSE'
import ctypes as ct
import hashlib
import json
from pathlib import Path
import sys
import time
import numpy as np
import scipy.linalg as sla

ROOT=Path('/home/ericyu/expert_flash_experiment')
ROUND=ROOT/'results/research-2026-11-03-no-cr/round02-local-affine'
STAGE0=ROUND/'stage0'
FROZEN0=STAGE0/'freezes/rev003'
sys.path.insert(0,str(FROZEN0))
import research_20261103_local_affine_stage0_core as old
import research_20261103_local_affine_stage0_pipeline as pipe
assert Path(old.__file__).resolve()==FROZEN0/'research_20261103_local_affine_stage0_core.py'
assert Path(pipe.__file__).resolve()==FROZEN0/'research_20261103_local_affine_stage0_pipeline.py'

EXPECTED='42483d1490187fba7500a23bd9944c394a9e437ff00df365d13e32aa08e3269a'
FORMATS=[(128,'F32','host_F32'),(128,'W8','host_floatlatent'),(128,'W8','native_A8latent'),
         (512,'W8','host_floatlatent'),(512,'W8','native_A8latent')]

class FMA:
    def __init__(self,path):
        self.lib=ct.CDLL(str(path))
        assert self.lib.lac_environment()==8,'all eight FMA workers require RN/FTZ0/DAZ0'
    def dot(self,x,w,reference=False):
        x=np.ascontiguousarray(x,np.float32);w=np.ascontiguousarray(w)
        assert x.ndim==w.ndim==2 and x.shape[1]==w.shape[1]
        assert w.dtype in [np.dtype('float32'),np.dtype('int8')]
        assert np.isfinite(x).all() and np.isfinite(w).all()
        out=np.empty((len(x),len(w)),np.float32)
        fn=self.lib.lac_reference if reference else self.lib.lac_i8 if w.dtype==np.int8 else self.lib.lac_f32
        assert not reference or w.dtype==np.float32
        assert fn(old.Numeric.ptr(x),old.Numeric.ptr(w),len(x),x.shape[1],len(w),old.Numeric.ptr(out))==0
        return out

def subset(d,ix):
    return {k:(v[ix] if k!='P' and isinstance(v,np.ndarray) else v) for k,v in d.items()}

def input_statistics(d,prompts,nfit,local):
    ix=np.flatnonzero(prompts<nfit);f=subset(d,ix);r=f['v'].shape[1]
    pcov=np.zeros((r,r),np.float64);leaves=[]
    for c in range(32):
        rows=np.flatnonzero(f['leaf']==c);a=f['v'][rows];sw=len(rows)
        mean=np.mean(a,axis=0) if sw else np.zeros(r)
        dv=a-mean;ww=np.ones(sw)
        cov=(dv.T*ww)@dv/sw if sw else np.zeros((r,r))
        pcov+=cov*(sw/len(ix))
        leaves.append({'ix':rows,'mean_v':mean,'dv':dv,'sum_w':sw,
                       'cov':cov if local else None,'factor':old.eigen(cov) if local else None})
    return {'d':f,'ix':ix,'leaves':leaves,'pcov':pcov,'pfactor':old.eigen(pcov),'local':local}

def fitted_slopes(inputs,y,eta,lam):
    y=np.asarray(y[inputs['ix']],np.float32);r=inputs['pcov'].shape[0];H=y.shape[1]
    pcross=np.zeros((r,H));stats=[]
    for st in inputs['leaves']:
        rows=st['ix'];sw=st['sum_w'];yy=y[rows].astype(np.float64)
        my=yy.mean(axis=0) if sw else np.zeros(H)
        cross=(st['dv'].T*np.ones(sw))@(yy-my)/sw if sw else np.zeros((r,H))
        pcross+=cross*(sw/len(y));stats.append({**st,'cross':cross,'mean_y':my})
    prior,_=old.solve(inputs['pcov'],pcross,eta,factor=inputs['pfactor'])
    if inputs['local']:
        As=[old.solve(s['cov'],s['cross'],lam,prior,factor=s['factor'])[0] for s in stats]
    else:As=[prior]
    return As,prior,pcross,stats,y

def target_matrix(As,inputs,pcross,stats):
    H=As[0].shape[1];M=np.zeros((H,H));total=len(inputs['ix'])
    if not inputs['local']:
        A=As[0];M=pcross.T@A+A.T@pcross-A.T@inputs['pcov']@A
    else:
        for A,s in zip(As,stats):
            M+=(s['sum_w']/total)*(s['cross'].T@A+A.T@s['cross']-A.T@s['cov']@A)
    return (M+M.T)*.5

def output_basis(M,maxq=512):
    vals,U=sla.eigh(M,driver='evd',check_finite=True)
    order=np.argsort(-vals,kind='stable');vals,U=vals[order],U[:,order]
    scale=float(np.max(np.abs(vals),initial=0));tol=1e-10*scale
    keep=np.flatnonzero(vals>tol)[:maxq];out=np.zeros((len(M),maxq))
    for j,k in enumerate(keep):out[:,j]=old.signfix(U[:,k])
    return out,{'solver':'symmetric_FP64_evd_full_descending_stable','positive_relative_cut':1e-10,
        'positive_rank':len(keep),'requested_maxq':maxq,'zero_padding':maxq-len(keep),
        'min_eigenvalue':float(vals.min(initial=0)),'max_eigenvalue':float(vals.max(initial=0)),
        'negative_beyond_roundoff':int(np.sum(vals < -tol)),'eigenvalues':vals.tolist(),
        'objective':'normalized teacher-SSE projection of fixed centered ridge predictions; no quantized-fit optimum'}

def rms_gauge(U,B,d):
    ss=np.zeros(U.shape[1]);n=0
    for c in range(32):
        rows=np.flatnonzero(d['leaf']==c)
        h=d['v'][rows]@B[0 if len(B)==1 else c].T
        ss+=np.sum(h*h,axis=0);n+=len(rows)
    rho=np.sqrt(ss/max(n,1));mx=float(rho.max(initial=0))
    scale=np.maximum(rho,mx/16) if mx else np.ones(len(rho))
    return U*scale[None,:],B/scale[None,:,None],scale

def lower_factors(num,U,B,d,fmt,path,gauge):
    if gauge=='FITrmsfloor_maxRMS_div16':U,B,scale=rms_gauge(U,B,d)
    else:scale=np.ones(U.shape[1])
    model={'U':U.astype(np.float32),'B':B.astype(np.float32),'Us':np.ones(U.shape[0],np.float32),
           'Bs':np.ones(B.shape[:2],np.float32),'bias':np.zeros((32,U.shape[0]),np.float32),
           'unsupported':np.array([c for c in range(32) if not np.any(d['leaf']==c)],np.int32),'gauge_scale':scale,
           'meta':{'format':fmt,'path':path,'gauge':gauge,'q':U.shape[1],'input_rank':B.shape[2],
                   'family':'G_RRR' if len(B)==1 else 'PE_RRR'}}
    if fmt=='W8':
        model['U'],model['Us']=num.quantize(model['U'])
        pairs=[num.quantize(a) for a in model['B']]
        model['B']=np.stack([p[0] for p in pairs]);model['Bs']=np.stack([p[1] for p in pairs])
    return model

def predict(num,fma,model,d,bias=True,batch=256):
    q=model['U'].shape[1];H=model['U'].shape[0];latent=np.empty((len(d['q']),q),np.float32)
    fmt,path=model['meta']['format'],model['meta']['path']
    for c in range(32):
        ix=np.flatnonzero(d['leaf']==c);j=0 if len(model['B'])==1 else c
        for start in range(0,len(ix),batch):
            rows=ix[start:start+batch]
            if fmt=='F32':
                z=fma.dot(d['q'][rows].astype(np.float32),model['B'][j])
                latent[rows]=np.multiply(z,d['sx'][rows,None],dtype=np.float32)
            else:latent[rows]=num.linear(d['q'][rows],d['sx'][rows],model['B'][j],model['Bs'][j])
    out=np.empty((len(latent),H),np.float32)
    for start in range(0,len(out),batch):
        sl=slice(start,start+batch);h=latent[sl]
        if path=='native_A8latent':
            hq,hs=num.quantize(h);out[sl]=num.linear(hq,hs,model['U'],model['Us'])
        else:
            p=fma.dot(h,model['U'])
            out[sl]=np.multiply(p,model['Us'][None,:],dtype=np.float32) if fmt=='W8' else p
        if bias:
            # Existing exact separate F32 addition, one per selected leaf.
            for c in np.unique(d['leaf'][sl]):
                rows=np.flatnonzero(d['leaf'][sl]==c)
                out[start+rows]=num.add_bias(out[start+rows],model['bias'][c])
    return out

def calibrate(num,fma,model,d,y):
    pred=predict(num,fma,model,d,bias=False)
    for c in range(32):
        ix=np.flatnonzero(d['leaf']==c)
        if len(ix):model['bias'][c]=num.bias(y[ix],pred[ix],np.ones(len(ix)))
    return model

def archive(folder,model):
    meta=json.dumps(model['meta'],sort_keys=True,separators=(',',':')).encode()
    arrays={k:v for k,v in model.items() if k!='meta'}
    arrays['meta_json']=np.frombuffer(meta,np.uint8).copy()
    key=old.array_hash(*(arrays[k] for k in sorted(arrays)))
    p=Path(folder)/(key+'.npz')
    if not p.exists():np.savez(p,**arrays)
    else:assert model_hash(load(p))==key
    return str(p),key

def model_hash(model):
    arrays={k:v for k,v in model.items() if k!='meta'}
    arrays['meta_json']=np.frombuffer(json.dumps(model['meta'],sort_keys=True,separators=(',',':')).encode(),np.uint8)
    return old.array_hash(*(arrays[k] for k in sorted(arrays)))

def load(path):
    with np.load(path,allow_pickle=False) as z:
        m={k:z[k].copy() for k in z.files if k!='meta_json'}
        m['meta']=json.loads(z['meta_json'].tobytes())
    H,q=m['U'].shape;nB,qb,r=m['B'].shape
    assert qb==q and nB in [1,32] and m['bias'].shape==(32,H)
    assert m['Us'].shape==(H,) and m['Bs'].shape==(nB,q)
    assert m['bias'].dtype==m['Us'].dtype==m['Bs'].dtype==np.float32
    want=np.float32 if m['meta']['format']=='F32' else np.int8
    assert m['U'].dtype==m['B'].dtype==want
    assert all(np.isfinite(m[k]).all() for k in ['U','B','Us','Bs','bias','gauge_scale'])
    assert np.all(m['Us']>0) and np.all(m['Bs']>0)
    assert m['unsupported'].dtype==np.int32 and np.all((m['unsupported']>=0)&(m['unsupported']<32))
    assert m['meta']['input_rank']==r and m['meta']['q']==q
    return m

def span_floor(U,bias,y,leaf,natural):
    # A real-valued fixed stored-matrix span oracle. Finite FMA rounding may
    # leave this span: not a universal lower bound for rounded instructions.
    Q,s,_=sla.svd(np.asarray(U,np.float64),full_matrices=False,lapack_driver='gesdd',check_finite=True)
    rank=int(np.sum(s>s[0]*1e-10)) if len(s) and s[0]>0 else 0;Q=Q[:,:rank]
    ss=energy=0.
    for c in range(32):
        ix=np.flatnonzero(natural&(leaf==c));a=y[ix].astype(np.float64)-bias[c].astype(np.float64)
        residual=a-(a@Q)@Q.T;ss+=float(np.sum(residual**2));energy+=float(np.sum(y[ix].astype(np.float64)**2))
    return {'relative_SSE':ss/energy if energy else None,'rank':rank,'SSE':ss,'energy':energy,
            'scope':'fixed bias and real lift using this stored U; not all-U/finite-rounding/NLL bound'}

def resources(model,intermediate,natural,leaf):
    meta=model['meta'];H,q=model['U'].shape;r=model['B'].shape[2];fmt=meta['format'];native=meta['path']=='native_A8latent'
    supported=natural&~np.isin(leaf,model['unsupported']);fallback=natural&~supported
    Ubytes=model['U'].nbytes;Bbytes=model['B'].nbytes;scales=model['Us'].nbytes+model['Bs'].nbytes if fmt=='W8' else 0
    activeB=q*r
    return {'retained_coefficients_bytes':Ubytes+Bbytes,'row_scale_bytes':scales,'leaf_bias_bytes':model['bias'].nbytes,
        'keys_bytes':8*(len(model['B'])+1),'active_U_bytes':Ubytes,'active_B_bytes':model['B'][0].nbytes,
        'natural_rows':int(natural.sum()),'factor_rows':int(supported.sum()),'original_fallback_rows':int(fallback.sum()),
        'factor_MACs_per_covered_row':H*q+activeB,'original_fallback_MACs_per_row':3*H*intermediate,
        'total_MACs':int(supported.sum())*(H*q+activeB)+int(fallback.sum())*3*H*intermediate,
        'host_shared_P_MACs_per_layer_input':H*128 if meta['family']=='PE_RRR' else 0,
        'single_operator_lazy_host_projection_MACs':int(supported.sum())*H*128 if meta['family']=='PE_RRR' else 0,
        'host_P_cost_is_shared_not_additive_over_experts':True,'U_ordinary_read_pages_per_covered_row':(Ubytes+16383)//16384 if native else 0,
        'complete_output_rows':H,'U_complete_row_bytes':q*(4 if fmt=='F32' else 1),
        'possible_chip_groups':[2,4,8,16],'actual_group_FIFO_command_latency_measured':False,
        'native_latent_A8_payload_per_covered_row':q+4 if native else 0,'native_output_F32_payload':H*4 if native else 0,
        'host_U_i8_to_F32_conversions_per_covered_row':H*q if fmt=='W8' and not native else 0,
        'target_latency_measured':False,'all64_or_whole_layer_compiled':False,'zero_NAND_does_not_imply_zero_host_latency':True}
