#!/usr/bin/env python3
"""Crosscheck the new fused two-stage C++ kernel against independent fixtures."""
import argparse
import ctypes
import json
from pathlib import Path

import numpy as np

import research_20261109_shared_feature_fit as base
from research_20261109_shared_feature_targets import DATA, ROOT, Numeric, dump, sha


class Kernel:
    def __init__(self):
        self.path=DATA/'build/factorized-numeric-v1.so'
        self.lib=ctypes.CDLL(str(self.path))
        f=np.ctypeslib.ndpointer(dtype=np.float32,flags='C_CONTIGUOUS')
        q=np.ctypeslib.ndpointer(dtype=np.int8,flags='C_CONTIGUOUS')
        self.lib.sf_factorized_set_threads.argtypes=[ctypes.c_int]
        self.lib.sf_factorized_set_threads(2)
        self.lib.sf_factorized_predict_q.argtypes=[ctypes.c_int]*4+[q,f,q,f,f,f,q,f,f,f,f,f,q,f]

    def run(self,qx,sx,V,sv,center,gain,U,su,bias):
        N,C=qx.shape;Q=V.shape[0];R=U.shape[0]
        out=np.empty((N,R),np.float32);raw=np.empty((N,Q),np.float32);affine=np.empty_like(raw)
        codes=np.empty((N,Q),np.int8);scales=np.empty(N,np.float32)
        code=self.lib.sf_factorized_predict_q(N,C,Q,R,qx,sx,V,sv,center,gain,U,su,bias,out,raw,affine,codes,scales)
        assert code==0,code
        return {'output':out,'V_raw':raw,'latent_normalized':affine,'latent_A8':codes,'latent_scale':scales}


def identical(actual,expected,label):
    assert actual.dtype==expected.dtype and actual.shape==expected.shape,label
    a=actual.view(np.uint32) if actual.dtype==np.float32 else actual
    b=np.ascontiguousarray(expected).view(np.uint32) if expected.dtype==np.float32 else expected
    assert np.array_equal(a,b),label


def selftest(kernel):
    rng=np.random.default_rng(20261109)
    cases=[]
    for name,N,C,Q,R in [('random',5,37,19,23),('zero',2,17,13,11),('half_ties',2,16,16,16)]:
        x=rng.normal(size=(N,C)).astype(np.float32)
        if name=='zero':x[:]=0
        qx,sx=base.old.quantize_rows(x)
        V,sv=base.old.quantize_rows(rng.normal(size=(Q,C)).astype(np.float32))
        U,su=base.old.quantize_rows(rng.normal(size=(R,Q)).astype(np.float32))
        center=rng.normal(size=Q).astype(np.float32);gain=np.exp(rng.normal(size=Q)).astype(np.float32)
        bias=rng.normal(size=R).astype(np.float32)
        if name=='zero':center[:]=0
        if name=='half_ties':
            V[:]=0;sv[:]=1
            center=-np.array([-127,-126.5,-2.5000002,-2.5,-2.4999998,-1.5,-.5,-.49999997,0,.49999997,.5,1.5,2.4999998,2.5,126.5,127],np.float32)
            gain[:]=1
        # Independent I64 arithmetic and explicit F32 stage boundaries.
        d=qx.astype(np.int64)@V.astype(np.int64).T
        raw=base.old.rescale(d,sv,sx)
        affine=base.transform(raw,center,gain)
        mq,ms=base.old.quantize_rows(affine)
        ud=mq.astype(np.int64)@U.astype(np.int64).T
        expected={'V_raw':raw,'latent_normalized':affine,'latent_A8':mq,'latent_scale':ms,
                  'output':np.add(base.old.rescale(ud,su,ms),bias,dtype=np.float32)}
        actual=kernel.run(qx,sx,V,sv,center,gain,U,su,bias)
        for key in expected:identical(actual[key],expected[key],name+'/'+key)
        cases.append({'case':name,'status':'passed','float_output_bits_equal':True,'latent_A8_codes_and_scales_equal':True})
    return cases


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--handoff',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args()
    kernel=Kernel();frozen=Numeric(2)
    data=json.loads(a.handoff.read_text())
    checks=[]
    for entry in data['models']:
        assert sha(entry['fixture'])==entry['fixture_sha256'] and sha(entry['descriptor'])==entry['descriptor_sha256']
        cfg=json.loads(Path(entry['descriptor']).read_text())
        C=cfg.get('input_width',cfg.get('width'));Q=cfg['rank'];R=cfg['H']
        assert cfg['version']==1 and ('width' not in cfg or cfg['width']==C)
        for path,identity in cfg['payload_sha256'].items():assert sha(path)==identity
        prefix=entry['factor_prefix']
        V=np.fromfile(prefix+'.V.i8',np.int8).reshape(Q,C);sv=np.fromfile(prefix+'.V.scales.f32',np.float32)
        center=np.fromfile(prefix+'.latent.center.f32',np.float32);gain=np.fromfile(prefix+'.latent.inv_scale.f32',np.float32)
        U=np.fromfile(prefix+'.U.i8',np.int8).reshape(R,Q);su=np.fromfile(prefix+'.U.scales.f32',np.float32);bias=np.fromfile(prefix+'.bias.f32',np.float32)
        fixture=np.load(entry['fixture'])
        qx,sx=frozen.quantize(fixture['input_features'])
        identical(qx,fixture['input_A8'],'input codes');identical(sx,fixture['input_scale'],'input scales')
        actual=kernel.run(qx,sx,V,sv,center,gain,U,su,bias)
        for key in actual:identical(actual[key],fixture[key],prefix+'/'+key)
        checks.append({'family':entry['family'],'expert':entry['expert'],'rank':Q,'input_width':C,
                       'fixture_sha256':entry['fixture_sha256'],'descriptor_sha256':entry['descriptor_sha256'],
                       'V_and_affine_floats_bitchecked':2*actual['V_raw'].size,
                       'output_floats_bitchecked':actual['output'].size,
                       'latent_A8_codes_bitchecked':actual['latent_A8'].size,'status':'passed'})
    report={'status':'passed','handoff_path':str(a.handoff),'handoff_sha256':sha(a.handoff),
            'kernel_binary_sha256':sha(kernel.path),'frozen_numeric_binary_sha256':frozen.binary_sha256,
            'source_sha256':sha(Path(__file__)),'header_sha256':sha(ROOT/'scripts/research_20261109_factorized_readout.hpp'),
            'CPU_affinity':sorted(__import__('os').sched_getaffinity(0)),'threads':2,
            'selftests':selftest(kernel),'models':checks,
            'model_count':len(checks),'output_floats_bitchecked':sum(r['output_floats_bitchecked'] for r in checks),
            'scope':'TRAIN/VALID numerical fixtures only, no new TEST or live-quality claim.'}
    dump(a.output,report)
    print(json.dumps({'status':'passed','models':len(checks),'output_floats_bitchecked':report['output_floats_bitchecked']}))


if __name__=='__main__':main()
