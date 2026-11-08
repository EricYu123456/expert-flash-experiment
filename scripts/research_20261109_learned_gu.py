#!/usr/bin/env python3
"""CPU-only selected-route distillation of a shared GU with fixed W8 readouts.

Torch is a TRAIN-only STE surrogate. Every reported VALID score is recomputed
with integer-exact W8/A8, C++ F32 rescale and pinned native SwiGLU. No TEST
interface exists. An updated GU is a NEW basis: it cannot reuse a selected
native owner's GU, and its weights/compute must be charged separately.
"""
import argparse
import importlib.util
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits
import torch

HERE = Path(__file__).resolve().parent


def module(name, filename):
    spec=importlib.util.spec_from_file_location(name,HERE/filename)
    result=importlib.util.module_from_spec(spec);spec.loader.exec_module(result)
    return result


fit=module('prototype_fit_learned','research_20261109_shared_feature_fit_proto.py')
targets=module('target_numeric_learned','research_20261109_shared_feature_targets.py')
base,old=fit.base,fit.old


def ste_qdq(x):
    # F32 quotient then FP64 half offset: std::round ties away from zero.
    scale=x.detach().abs().amax(dim=1,keepdim=True)/127
    scale=torch.where(scale==0,torch.ones_like(scale),scale)
    ratio=x.detach()/scale
    q=ratio.sign()*(ratio.abs().double()+.5).floor().float()
    dequantized=q.clamp(-127,127)*scale
    return x+(dequantized-x).detach()


def exact_features(numeric,x,gate,up,transform,native_weights=None):
    if native_weights is None:
        gq,gs=old.quantize_rows(gate);uq,us=old.quantize_rows(up)
    else:
        gq,gs=native_weights['gate'];uq,us=native_weights['up']
    result=[]
    for start in range(0,len(x),1024):
        xx=np.asarray(x[start:start+1024],np.float32)
        g=numeric.matvecs(xx,gq,gs);u=numeric.matvecs(xx,uq,us)
        phi=np.empty(g.shape,np.float32)
        assert numeric.lib.sf_swiglu(*phi.shape,g,u,phi)==0
        result.append(base.transform(phi,transform['center'],transform['inv_scale']))
    return base.encode(np.concatenate(result))


def evaluate(numeric,x,pair_rows,pair_experts,pair_targets,pair_a2,models,gate,up,transform,native_weights=None):
    unique,inverse=np.unique(pair_rows,return_inverse=True)
    features=exact_features(numeric,x[unique],gate,up,transform,native_weights)
    error,energy=0.,0.
    records=[]
    for expert,model in models.items():
        selected=np.flatnonzero(pair_experts==expert)
        enc={k:v[inverse[selected]] for k,v in features.items()}
        pred=base.predict_encoded(enc,model)
        yy=pair_targets[selected].astype(np.float64)
        ee=np.sum((yy-pred.astype(np.float64))**2,axis=1)
        nn=np.sum(yy**2,axis=1);a2=pair_a2[selected]
        sse=float(a2@ee);target_energy=float(a2@nn)
        records.append(dict(expert=expert,selected_rows=len(selected),weighted_sse=sse,
                            weighted_target_energy=target_energy,alpha2_nmse=sse/target_energy))
        error+=sse;energy+=target_energy
    return dict(alpha2_nmse=error/energy,weighted_sse=error,weighted_target_energy=energy,
                experts=records,canonical_W8A8=True,GU_unique_input_rows=len(unique))


def pairs(dataset,experts):
    rows,ids,ys,weights=[],[],[],[]
    for expert in experts:
        a2=base.alpha_squared(dataset.capture,expert)
        selected=np.flatnonzero(a2>0)
        rows.append(selected);ids.append(np.full(len(selected),expert,np.int32))
        ys.append(np.asarray(dataset.targets.get(expert)[selected],np.float32))
        weights.append(a2[selected])
    return np.concatenate(rows),np.concatenate(ids),np.concatenate(ys),np.concatenate(weights)


def run(args):
    torch.set_num_threads(args.threads);torch.set_num_interop_threads(1)
    torch.manual_seed(args.seed);rng=np.random.default_rng(args.seed)
    out=Path(args.output).resolve();out.mkdir(parents=True,exist_ok=False)
    selection_path=Path(args.selection).resolve();selection=json.loads(selection_path.read_text())
    layer=selection['layer'];owners=set(selection['prototype_ids'])
    # Only experts for which this native prototype is the chosen best cross
    # basis. Native owners stay exact in this local and eventual live study.
    records=[m for m in selection['families']['best_cross_prototype']
             if m['prototype']==args.prototype and m['expert'] not in owners]
    assert records,'No eligible cross-expert group for prototype'
    experts=sorted(m['expert'] for m in records)
    train=fit.Dataset(args.train,args.train_target_manifest,layer)
    valid=fit.Dataset(args.valid,args.valid_target_manifest,layer)
    assert selection['train_identity']==train.identity() and selection['valid_identity']==valid.identity()
    transform=fit.load_transform(records[0]['transform_prefix'])
    assert all(m['transform_prefix']==records[0]['transform_prefix'] for m in records)
    assert len(transform['indices'])==1408 and np.all(transform['sources']==2)
    assert np.all(transform['prototypes']==args.prototype)
    assert np.array_equal(transform['indices'],np.arange(1408,dtype=np.int32))
    models={}
    for m in records:
        prefix=m['readout_prefix']
        models[m['expert']]={
            'qweight':np.fromfile(prefix+'.i8',np.int8).reshape(2048,m['width']),
            'wscale':np.fromfile(prefix+'.scales.f32',np.float32),
            'bias':np.fromfile(prefix+'.bias.f32',np.float32)}
    weights_meta=json.loads((Path(args.weights)/'weights.json').read_text())
    assert weights_meta['layer']==layer and str(args.prototype) in weights_meta['experts']
    native=targets.load_weights(Path(args.weights),args.prototype)
    gate0=(native['gate'][0].astype(np.float32)*native['gate'][1][:,None]).astype(np.float32)
    up0=(native['up'][0].astype(np.float32)*native['up'][1][:,None]).astype(np.float32)
    gate=torch.nn.Parameter(torch.from_numpy(gate0.copy()))
    up=torch.nn.Parameter(torch.from_numpy(up0.copy()))
    assert gate.device.type==up.device.type=='cpu'
    optimizer=torch.optim.Adam([gate,up],lr=args.learning_rate)
    tr=pairs(train,experts);va=pairs(valid,experts)
    x=train.capture.get('x');vx=valid.capture.get('x')
    q,sx=old.quantize_rows(x)
    xqdq=torch.from_numpy((q.astype(np.float32)*sx[:,None]).astype(np.float32))
    y=torch.from_numpy(tr[2]);a2=torch.from_numpy(tr[3].astype(np.float32))
    torch_heads={e:(torch.from_numpy(m['qweight'].astype(np.float32)*m['wscale'][:,None]),
                    torch.from_numpy(m['bias'])) for e,m in models.items()}
    center=torch.from_numpy(transform['center']);inv=torch.from_numpy(transform['inv_scale'])
    # Fixed global TRAIN normalization; no VALID-derived loss scale.
    normalization=float(np.sum(tr[3]*np.sum(tr[2].astype(np.float64)**2,axis=1))/len(tr[0]))
    numeric=targets.Numeric(args.threads)
    checkpoints=[];started=time.monotonic();best_score=float('inf');best=None
    for step in range(args.steps+1):
        if step in args.checkpoints or step==args.steps:
            g=gate.detach().numpy().copy();u=up.detach().numpy().copy()
            metric=evaluate(numeric,vx,*va,models,g,u,transform,native if step==0 else None)
            if step==0:
                expected_error=sum(m['weighted_squared_error'] for m in records)
                expected_energy=sum(m['selected_VALID_weighted_target_energy'] for m in records)
                assert abs(metric['alpha2_nmse']-expected_error/expected_energy)<1e-9
            rec=dict(step=step,seconds=time.monotonic()-started,VALID=metric)
            checkpoints.append(rec)
            if metric['alpha2_nmse']<best_score:
                best_score=metric['alpha2_nmse'];best=(step,g,u)
            old.dump(out/'progress.json',dict(checkpoints=checkpoints))
            print(json.dumps(rec),flush=True)
        if step==args.steps:break
        ix=rng.integers(len(tr[0]),size=args.batch)
        optimizer.zero_grad(set_to_none=True)
        xx=xqdq[tr[0][ix]]
        phi=torch.nn.functional.silu(xx@ste_qdq(gate).T)*(xx@ste_qdq(up).T)
        feature=ste_qdq((phi-center)*inv)
        loss=torch.zeros((),device='cpu')
        for expert in np.unique(tr[1][ix]):
            loc=np.flatnonzero(tr[1][ix]==expert)
            cw,cb=torch_heads[int(expert)]
            pred=feature[loc]@cw.T+cb
            loss=loss+((pred-y[ix[loc]]).square().sum(dim=1)*a2[ix[loc]]).sum()
        loss=loss/(args.batch*normalization)
        # Small relative displacement regularizer, independent of VALID.
        penalty=((gate-torch.from_numpy(gate0)).square().mean()/max(float(np.mean(gate0**2)),1e-20)
                 +(up-torch.from_numpy(up0)).square().mean()/max(float(np.mean(up0**2)),1e-20))
        objective=loss+args.regularization*penalty
        assert torch.isfinite(objective)
        objective.backward();torch.nn.utils.clip_grad_norm_([gate,up],10.)
        optimizer.step()
    best_step,g,u=best
    payload={}
    for name,value in [('gate',g),('up',u)]:
        w,scale=native[name] if best_step==0 else old.quantize_rows(value)
        for suffix,array in [('.i8',w),('.scales.f32',scale)]:
            path=out/(name+suffix);array.tofile(path);payload[str(path)]=old.sha(path)
    result=dict(status='TRAIN_STE_VALID_exact_no_TEST',layer=layer,native_initial_prototype=args.prototype,
                experts=experts,best_step=best_step,checkpoints=checkpoints,
                learning_rate=args.learning_rate,regularization=args.regularization,
                train_pair_count=len(tr[0]),valid_pair_count=len(va[0]),
                selection_path=str(selection_path),selection_sha256=old.sha(selection_path),
                source_sha256=old.sha(__file__),numeric_sha256=numeric.binary_sha256,
                payload_sha256=payload,CPU_only=True,threads=args.threads,
                training_arithmetic='FP32 Torch STE surrogate, CPU; not canonical integer runtime',
                reported_VALID_arithmetic='canonical integer-exact W8A8 + frozen native SwiGLU',
                native_owner_GU_reuse_allowed=False,
                deployment='New GU weights require separate resident bytes and compute; original owners remain native.',
                total_seconds=time.monotonic()-started)
    old.dump(out/'selection.json',result)
    print(json.dumps(dict(output=str(out),best_step=best_step,best_VALID_nmse=best_score)),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('selection','train','valid','train-target-manifest','valid-target-manifest','weights','output'):
        p.add_argument('--'+key,required=True)
    p.add_argument('--prototype',type=int,required=True)
    p.add_argument('--steps',type=int,default=64)
    p.add_argument('--checkpoints',nargs='+',type=int,default=[0,16,32,64])
    p.add_argument('--learning-rate',type=float,default=.0003)
    p.add_argument('--regularization',type=float,default=.001)
    p.add_argument('--batch',type=int,default=128)
    p.add_argument('--threads',type=int,default=8)
    p.add_argument('--seed',type=int,default=20261109)
    a=p.parse_args()
    assert a.learning_rate>0 and a.regularization>=0 and a.steps>0 and a.batch>0
    with threadpool_limits(limits=a.threads,user_api='blas'):
        run(a)
