#!/usr/bin/env python3
"""Only generated arrays/temp archives. Never reads candidate dataset bodies."""
import argparse
import copy
import tempfile
import time
from pathlib import Path
from research_20261103_local_affine_stage1_controls_core import *
import research_20261103_local_affine_stage1_controls as run

def check(lib,output):
    begin=time.monotonic();rng=np.random.default_rng(20261103);num=old.Numeric(FROZEN0/'numeric.so');fma=FMA(lib);checks=[]
    fma.lib.lac_get_mxcsr.restype=ct.c_uint32
    fma.lib.lac_set_mxcsr.argtypes=[ct.c_uint32];fma.lib.lac_set_mxcsr.restype=None
    original=fma.lib.lac_get_mxcsr();assert fma.lib.lac_x87_nearest()==1
    tx=np.ones((1,8),np.float32);tw=np.ones((1,8),np.float32);tout=np.empty((1,1),np.float32)
    try:
        for mode in [0x2000,0x4000,0x6000]:
            fma.lib.lac_set_mxcsr((original&~0x6000)|mode)
            assert fma.lib.lac_get_mxcsr()&0x6000==mode and fma.lib.lac_x87_nearest()==1
            assert fma.lib.lac_environment()!=8
            assert fma.lib.lac_f32(old.Numeric.ptr(tx),old.Numeric.ptr(tw),1,8,1,old.Numeric.ptr(tout))!=0
    finally:fma.lib.lac_set_mxcsr(original)
    assert fma.lib.lac_get_mxcsr()==original and fma.lib.lac_environment()==8
    checks.append('MXCSR_only_all_three_nonRN_modes_rejected_with_x87_RN_and_environment_restored')
    for K in [1,7,8,13,128,512,2048]:
        x=rng.normal(size=(5,K)).astype(np.float32);w=rng.normal(size=(17,K)).astype(np.float32)
        a=fma.dot(x,w);b=fma.dot(x,w,reference=True)
        assert np.array_equal(a.view('u4'),b.view('u4'))
        assert np.array_equal(a.view('u4'),np.concatenate([fma.dot(z[None],w) for z in x]).view('u4'))
        wi=np.clip(np.floor(w*24),-127,127).astype(np.int8)
        assert np.array_equal(fma.dot(x,wi).view('u4'),fma.dot(x,wi.astype(np.float32),reference=True).view('u4'))
    checks.append('AVX2_FMA_vs_scalar_fmaf_and_B1_batch_bitwise_F32_and_i8_K1_7_8_13_128_512_2048')
    # Adversarial F32 cancellation and subnormals; workers must not flush.
    x=np.array([[2**-140,2**-140,1.,1.,2**20,-2**20,1.,1.]],np.float32)
    w=np.ones((1,8),np.float32)
    assert np.array_equal(fma.dot(x,w).view('u4'),fma.dot(x,w,reference=True).view('u4'))
    assert float(fma.dot(np.full((1,8),2**-140,np.float32),w)[0,0])>0
    checks.append('eight_workers_FE_NEAREST_FTZ_DAZ_off_and_subnormal_retention')
    N=120*32;r=5;H=9
    q,sx=num.quantize(rng.normal(size=(N,r)).astype(np.float32))
    d={'q':q,'sx':sx,'v':q.astype(np.float64)*sx[:,None].astype(np.float64),
       'leaf':np.tile(np.arange(32,dtype=np.int32),120),'P':np.eye(r,dtype=np.float32),'distance':np.zeros(N)}
    prompts=np.repeat(np.arange(120),32)
    y=(d['v']@rng.normal(size=(r,H))+rng.normal(size=(32,H))[d['leaf']]+rng.normal(size=(N,H))*.2).astype(np.float32)
    inp=input_statistics(d,prompts,80,True);As,prior,cross,stats,yn=fitted_slopes(inp,y,0.,.1)
    M=target_matrix(As,inp,cross,stats);direct=np.zeros((H,H))
    for A,st in zip(As,stats):
        yc=yn[st['ix']].astype(np.float64)-st['mean_y'];p=st['dv']@A
        direct+=yc.T@p+p.T@yc-p.T@p
    direct/=len(yn)
    assert np.allclose(M,direct,rtol=2e-12,atol=2e-12)
    U,info=output_basis(M,8);assert info['positive_rank']<=8
    yy=y.copy();yy[prompts>=80]+=rng.normal(size=yy[prompts>=80].shape).astype(np.float32)*100
    As2,_,cross2,stats2,_=fitted_slopes(inp,yy,0.,.1)
    assert all(np.array_equal(a,b) for a,b in zip(As,As2))
    assert np.array_equal(M,target_matrix(As2,inp,cross2,stats2))
    inp120=input_statistics(d,prompts,120,True)
    As120,_,c120,s120,_=fitted_slopes(inp120,yy,0.,.1)
    assert not np.allclose(M,target_matrix(As120,inp120,c120,s120))
    checks.append('within_leaf_actual_A8_centering_Mtarget_matches_explicit_Yhat_and_N80_supervised_isolation')
    gi=input_statistics(d,prompts,80,False);ga,gp,gc,gs,gy=fitted_slopes(gi,y,.01,.1)
    GM=target_matrix(ga,gi,gc,gs);gd=np.zeros((H,H))
    for st in gs:
        yc=gy[st['ix']].astype(np.float64)-st['mean_y'];p=st['dv']@ga[0]
        gd+=yc.T@p+p.T@yc-p.T@p
    assert np.allclose(GM,gd/len(gy),rtol=2e-12,atol=2e-12)
    correction,_=old.solve(inp['pcov'],cross,.1,prior,factor=inp['pfactor'])
    assert np.linalg.norm(correction-prior)<1e-10
    checks.append('G_global_pooled_centered_M_and_eta0_two_stage_closure')
    ranktest,inf=output_basis(np.diag([5.,3.,-2.,1e-12]),3)
    assert inf['positive_rank']==2 and np.all(ranktest[:,2]==0)
    zero,inf=output_basis(np.zeros((4,4)),3);assert not np.any(zero) and inf['positive_rank']==0
    checks.append('positive_rank_prefix_sign_zero_padding_and_material_negative_diagnostic')
    small=subset(inp['d'],np.arange(96));ys=yn[:96];Uq=U[:,:3];B=np.stack([(a@Uq).T for a in As])
    bu,bb,scale=rms_gauge(Uq,B,small)
    for c in range(32):
        rows=np.flatnonzero(small['leaf']==c)
        assert np.allclose((small['v'][rows]@B[c].T)@Uq.T,(small['v'][rows]@bb[c].T)@bu.T,rtol=1e-12,atol=1e-12)
    _,_,zs=rms_gauge(np.zeros_like(Uq),np.zeros_like(B),small);assert np.array_equal(zs,np.ones(3))
    checks.append('FIT_only_RMS_max_coordinate_floor_and_real_map_gauge_identity')
    with tempfile.TemporaryDirectory(prefix='stage1-synthetic-') as temp:
        tmp=Path(temp);models=[]
        for fmt,path in [('F32','host_F32'),('W8','host_floatlatent'),('W8','native_A8latent')]:
            for family in ['PE_RRR','G_RRR']:
                bx=B if family=='PE_RRR' else B[:1]
                model=calibrate(num,fma,lower_factors(num,Uq,bx,small,fmt,path,'identity'),small,ys)
                p=predict(num,fma,model,small);p1=np.concatenate([predict(num,fma,model,subset(small,np.array([j]))) for j in range(len(ys))])
                assert np.array_equal(p.view('u4'),p1.view('u4'))
                name,key=archive(tmp,model);loaded=load(name);assert model_hash(loaded)==key
                assert np.array_equal(p.view('u4'),predict(num,fma,loaded,small).view('u4'))
                a=resources(loaded,13,np.ones(len(ys),bool),small['leaf'])
                assert a['retained_coefficients_bytes']==model['U'].nbytes+model['B'].nbytes
                assert a['factor_MACs_per_covered_row']==H*3+r*3
                modified=copy.deepcopy(model);modified['unsupported']=np.array([0],np.int32)
                name2,key2=archive(tmp,modified);assert key!=key2
                assert resources(load(name2),13,np.ones(len(ys),bool),small['leaf'])['original_fallback_rows']==3
                assert model_hash(load(name))==key
                models.append((fmt,path,family,name,key))
                # Actual bias uses deployed prediction and original row order.
                raw=predict(num,fma,model,small,bias=False)
                for c in range(32):
                    ix=np.flatnonzero(small['leaf']==c);ref=np.zeros(H,np.float64)
                    for row in ix:ref+=ys[row].astype(np.float64)-raw[row].astype(np.float64)
                    ref=(ref/len(ix)).astype(np.float32)
                    assert np.array_equal(ref.view('u4'),model['bias'][c].view('u4'))
                us=model['U'].astype(np.float64)*(model['Us'][:,None].astype(np.float64) if fmt=='W8' else 1)
                floor=span_floor(us,model['bias'],ys,small['leaf'],np.ones(len(ys),bool))
                pinv=np.linalg.pinv(us,rcond=1e-10);res=ys.astype(np.float64)-model['bias'][small['leaf']]
                residual=res-(res@pinv.T)@us.T
                assert np.isclose(floor['SSE'],np.sum(residual**2),rtol=1e-10,atol=1e-10)
        checks.append('all_six_factor_paths_actual_bias_B1_batch_bits_archive_reload_cost_unsupported_identity_actual_span')
        # Pure evaluator used by the real entrypoint, on generated 40-ID data.
        # No file under Stage0/capture or fitted is read here.
        ids=['synthetic-'+str(i) for i in range(40)];rows=80;dd=subset(small,np.arange(rows))
        ds={'x':np.zeros((rows,r),np.float32),'prompt':np.repeat(np.arange(40),2),'phase':np.tile([0,1],40),
            'ids':np.tile(np.array([0,1],np.int32),(rows,1)),'alpha':np.full((rows,2),.5,np.float32),'jobs':ids}
        datasets={};designs={};targets={};base={};points=[];oldout=run.OUT;run.OUT=tmp
        for name,layers in [('deepseek',[1,13]),('olmoe',[0,8])]:
            for layer in layers:
                datasets[(name,layer)]=ds
                for family in ['PE_RRR','G_RRR']:designs[(name,layer,family)]=dd
                for expert in [0,1]:
                    op=(name,layer,expert);target=ys[:rows].copy()+np.float32(expert*.1);targets[op]=target
                    baseline={'qmaps':np.zeros((32,H,r),np.int8),'scales':np.ones((32,H),np.float32),
                        'bias':rng.normal(size=(32,H)).astype(np.float32),'unsupported':[],'map_hashes':['zero']*32}
                    bp,bh=old.archive_model(tmp,baseline);bid=f'{name}-{layer}-{expert}-baseline';base[bid]={'model_path':bp,'id':bid}
                    for fmt,path,family,mp,mh in models:
                        points.append({'id':f'{name}-{layer}-{expert}-{family}-{path}','model':name,'layer':layer,'expert':expert,
                            'nfit':120,'family':family,'q':3,'format':fmt,'path':path,'model_path':mp,'model_hash':mh,'baseline_id':bid})
        run.comparison_freeze(points,ids);freeze=json.loads((tmp/'comparisons.before_evaluation.frozen.json').read_text())
        boot=run.evaluate_inputs(num,fma,freeze,base,datasets,designs,targets,tmp/'evaluation')
        assert len(boot['comparisons'])==len(freeze['comparisons'])==168
        assert len(json.loads((tmp/'evaluation/prompt_metrics.json').read_text()))==56
        run.comparison_freeze(points[:-1],ids)
        missing=json.loads((tmp/'comparisons.before_evaluation.frozen.json').read_text())
        assert missing['incomplete_composite_families']
        run.OUT=oldout
        checks.append('generated_8operators_40IDs_archive_to_real_evaluator_all_168_frozen_comparisons_no_real_capture')
    # Shape-only current-server kernel observation, not deployment evidence.
    x=rng.normal(size=(1,512)).astype(np.float32);w=rng.normal(size=(2048,512)).astype(np.float32)
    ts=time.monotonic();fma.dot(x,w);shape_seconds=time.monotonic()-ts
    num.close()
    result={'status':'pass','at_utc':run.now(),'checks':checks,'seconds':time.monotonic()-begin,
        'eight_workers_environment':fma.lib.lac_environment(),'integer_dot_audits':num.dot_audits,
        'shape_only_host_FMA_1x512_by2048_seconds':shape_seconds,'timing_not_target_or_competitive_latency_claim':True,
        'all_inputs_generated_no_candidate_target_activation_model_prompt_reads':True}
    old.save(output,result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--lib',required=True);ap.add_argument('--output',required=True);a=ap.parse_args();check(a.lib,a.output)
