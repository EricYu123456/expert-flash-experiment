#!/usr/bin/env python3
"""Compact same-Round2 controls. prepare is synthetic-only; real stages gated."""
import argparse
import datetime
import hashlib
import json
import os
import re
from pathlib import Path
import resource
import shutil
import subprocess
import sys
import time
from research_20261103_local_affine_stage1_controls_core import *

OUT=ROUND/'stage1-controls'
PROTOCOL=ROUND/'stage1_controls.protocol.frozen.json'
PREFIX='research_20261103_local_affine_stage1_controls'
SOURCES=[ROOT/'scripts'/(PREFIX+s) for s in ['.py','_core.py','_synthetic.py','_fma.cpp']]
FLAGS=['-std=c++17','-O3','-fPIC','-shared','-fopenmp','-mavx2','-mfma','-fno-fast-math','-ffp-contract=off']
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()

def manifest():
    return {'stage':'same Round2 compact controls','protocol_sha256':EXPECTED,'H':2048,'C':32,'input_PE':128,
        'N':[80,120],'weight':'unweighted','guard':100,'families':['PE_RRR','G_RRR'],
        'representations':FORMATS,'gauges_W8':['identity','FITrmsfloor_maxRMS_div16'],
        'FP32_gauge':'identity only','selection':'lowest actual VALID natural-route SSE/energy per operator,N,family,q,format,path; identity wins exact tie; no rank/path/format selection',
        'fit':'same-N frozen Stage0 eta/lambda; same full120 unsupervised tree/P/meanX; within-leaf centered exact-real dequantized A8 design; free bias recalibrated on actual deployed same-N FIT predictions',
        'matrix':'sum leaf mass*(cross.T@A+A.T@cross-A.T@cov@A); G uses pooled equivalents; symmetric F64; normalized by FIT rows',
        'eigen':'one full symmetric F64 scipy.linalg.eigh driver evd; stable descending; largest-abs coordinate positive; retain values>1e-10*max(abs(all eigenvalues)); diagnose negative values below minus this tolerance; zero-pad to512; q128 prefix, no inclusion of nonpositive directions',
        'gauge':'F64 real-fit B@actual-dequant-v RMS over same-N FIT rows, including actual nonzero leaf means; floor=maxRMS/16, allzero=>ones; absorb d into U and inverse d into B',
        'F32_kernel':'AVX2 explicit FMA8 independent lanes; increasing k in each lane; tail uses lane k mod8; pair reduction ((0+1)+(2+3))+((4+5)+(6+7)); fixed per-sample path B1/batch; eight OMP workers RN/FTZ0/DAZ0 checked each call',
        'FP32_path':'round factors toF32; B dot exactF32 A8codes then separate F32 commonsx; U dot F32 latent; final separate F32 bias',
        'W8_host_path':'canonical exact integer B/A8 and F32 scale order; F32 latent retained; convert U_i8 lanes toF32, same FMA reduction, separate F32 row scale, separate F32 bias',
        'W8_native_path':'canonical B/A8, full-q latent canonical A8 tiesaway, canonical U/A8, separate F32 bias; ordinary U reads, posts and two-factor dependence paid',
        'oracle':'SVD gesdd actual stored F32/dequantizedW8 U, relative singular cut1e-10; fixed actual F32 bias; idealF64 U also reported. This is a mathematical real-lift fixed-span floor: finite FMA rounding can leave that span; not a universal rounded-kernel or all-U/NLL bound.',
        'archive':'dtype/shape/finite/positive scale checked; path,gauge,family,q,input rank and unsupported behavior included in hash; reload executes same factors, never dense reconstructed map',
        'comparisons':'before development SCORE: every selected gauge endpoint versus same-N fixed localbank, both phases; peroperator plus permodel/both equal-operator and pooled-energy composites; all predeclared paths/precisions retained, no automatic science classification',
        'evaluation':'old SCORE only under separate SCORE_DEVELOPMENT approval; same40 prompt IDs maxT10000 seed20261103 frozen Stage0 formula; exploratory, TEST never opened',
        'timing':'structural quality/development and shape-only SIMD checks; no target-host latency benchmark, cache-residency or zeroIO speed claim; shared placement host/AiF allowed symmetrically',
        'compile_flags':FLAGS,'no_stage0_source_or_output_mutation':True}

def prepare():
    assert old.sha(PROTOCOL)==EXPECTED
    OUT.mkdir(exist_ok=True)
    freezes=OUT/'freezes';freezes.mkdir(exist_ok=True)
    rev=1+max([int(p.name[3:]) for p in freezes.glob('rev[0-9][0-9][0-9]')],default=0)
    rd=freezes/f'rev{rev:03d}';rd.mkdir()
    for p in SOURCES:shutil.copy2(p,rd/p.name)
    old.save(rd/'algorithm_manifest.frozen.json',manifest())
    cmd=['g++',*FLAGS,str(rd/(PREFIX+'_fma.cpp')),'-o',str(rd/'fma.so')]
    build=subprocess.run(cmd,capture_output=True,text=True)
    (rd/'build.log').write_text(build.stdout+build.stderr)
    old.save(rd/'build_receipt.json',{'argv':cmd,'returncode':build.returncode,'compiler':subprocess.run(['g++','--version'],capture_output=True,text=True).stdout})
    assert build.returncode==0,'preserved failed build; repair implementation in a new revision'
    assembly=subprocess.run(['objdump','-d',str(rd/'fma.so')],capture_output=True,text=True,check=True).stdout
    (rd/'fma.disassembly.txt').write_text(assembly)
    assert 'vfmadd' in assembly
    env=os.environ.copy();env['PYTHONDONTWRITEBYTECODE']='1'
    test=subprocess.run([sys.executable,str(rd/(PREFIX+'_synthetic.py')),'--lib',str(rd/'fma.so'),'--output',str(rd/'synthetic.json')],cwd=ROOT,env=env,capture_output=True,text=True)
    (rd/'synthetic.log').write_text(test.stdout+test.stderr)
    assert test.returncode==0,'preserved failed synthetic; no real data opened'
    upstream=json.loads((FROZEN0/'freeze_receipt.json').read_text())
    dependencies={str(p):old.sha(p) for p in SOURCES}
    for name,want in upstream['files'].items():
        assert old.sha(name)==want
        dependencies[name]=want
    for p in rd.iterdir():
        if p.is_file():dependencies[str(p)]=old.sha(p)
    dependencies[str(PROTOCOL)]=EXPECTED
    dependencies[str(FROZEN0/'freeze_receipt.json')]=old.sha(FROZEN0/'freeze_receipt.json')
    linked=subprocess.run(['ldd',str(rd/'fma.so')],capture_output=True,text=True,check=True).stdout
    (rd/'linked_libraries.txt').write_text(linked)
    for name in re.findall(r'(/[^\s()]+)',linked):
        p=Path(name)
        if p.is_file():dependencies[str(p.resolve())]=old.sha(p)
    dependencies[str(rd/'linked_libraries.txt')]=old.sha(rd/'linked_libraries.txt')
    dependencies[str(Path(sys.executable).resolve())]=old.sha(sys.executable)
    receipt={'status':'READY_implementation_build_synthetic_only_pending_Root_source_review','at_utc':now(),
        'revision':rev,'protocol_sha256':EXPECTED,'files':dependencies,'stage1_model_prompt_activation_target_bodies_read':False,
        'stage0_upstream_revision':3,'manifest':manifest(),'real_stages_require_explicit_review':True}
    old.save(rd/'freeze_receipt.json',receipt)
    old.save(OUT/'CURRENT_FREEZE.json',{'revision':rev,'freeze_receipt':str(rd/'freeze_receipt.json'),'sha256':old.sha(rd/'freeze_receipt.json')})
    print(json.dumps({'READY':str(rd/'freeze_receipt.json'),'sha256':old.sha(rd/'freeze_receipt.json'),'synthetic':json.loads((rd/'synthetic.json').read_text())}),flush=True)

def verify(review,phase='FITVAL',evaluation_review=None):
    assert old.sha(PROTOCOL)==EXPECTED
    pointer=json.loads((OUT/'CURRENT_FREEZE.json').read_text());rd=Path(pointer['freeze_receipt']).parent
    assert old.sha(rd/'freeze_receipt.json')==pointer['sha256']
    receipt=json.loads((rd/'freeze_receipt.json').read_text());approved=json.loads(Path(review).read_text())
    def check(a,p):
        assert a['status']=='approved' and a['phase']==p
        assert a['protocol_sha256']==EXPECTED and a['source_revision']==receipt['revision']
        assert a['source_freeze_sha256']==pointer['sha256']
    check(approved,'FITVAL')
    for p,h in receipt['files'].items():assert old.sha(p)==h,p
    if phase=='SCORE_DEVELOPMENT':
        a=json.loads(Path(evaluation_review).read_text());check(a,phase)
        assert a['FITVAL_complete_sha256']==old.sha(OUT/'FITVAL.complete.json')
        assert a['comparisons_sha256']==old.sha(OUT/'comparisons.before_evaluation.frozen.json')
    return rd,receipt

def frozen_points(pr):
    assert old.sha(STAGE0/'comparisons.before_SCORE.frozen.json')==pr['stage0_comparisons_sha256']
    allpoints=json.loads((STAGE0/'comparisons.before_SCORE.frozen.json').read_text())['points'];byid={p['id']:p for p in allpoints}
    sets={120:pr['primary_points'],80:pr['learning_curve']['N80_points']}
    for n,points in sets.items():
        for p in points:
            a=byid[p['id']];b=byid[p['full_H_global_point']]
            assert all(a[k]==v for k,v in {'family':'local','C':32,'rank':128,'format':'normal','weight':'unweighted','nfit':n,'guard':100,'endpoint':'selected'}.items())
            assert a['model_hash']==p['model_hash'] and a['eta']==p['eta'] and a['lambda']==p['lambda']
            assert b['rank']==2048 and b['format']=='identity' and b['C']==32 and b['nfit']==n and b['guard']==100
            assert b['eta']==p['full_H_global_eta']
    return sets,byid

def full_model_ledger(points):
    source=json.loads((STAGE0/'resource_ledger.json').read_text());result={'scope':'analytic extrapolation; eight operators do not compile full model or measure target timing','models':{},'budgets':[8000000000,8589934592],
        'unknowns':source['unknowns'],'cache':'pilot fits current server LLC; target cache/cold/warm and runtime workspace unresolved',
        'physical':'ordinary native U pages only; actual groups/FIFO/commands/collisions and legal shared/host overlap DAG required; no zero-NAND implies latency dominance'}
    for name,L in [('deepseek',26),('olmoe',16)]:
        base=source['models'][name]['configs'][0];rows=[];E=64;H=2048;C=32
        for family in ['PE_RRR','G_RRR']:
            for q,fmt,path in FORMATS:
                r=128 if family=='PE_RRR' else H;nB=C if family=='PE_RRR' else 1;nb=4 if fmt=='F32' else 1
                coeff=L*E*(H*q+nB*q*r)*nb
                scales=L*E*(H+nB*q)*4 if fmt=='W8' else 0
                bias=L*E*C*H*4;keys=L*E*(nB+1)*8
                P=L*H*128*4 if family=='PE_RRR' else 0
                tree=L*(31*H*4+31*8);means=L*C*H*4
                common=sum(base[k] for k in ['original_nonFFN_encoded_tensor_bytes_host','pilot_context256_CPU_KV_allocation_bytes_upper_from_rounded_log','retained_original_W8_FFN_row_scales_bytes'])
                need=coeff+scales+bias+keys+P+tree+means+common
                shared=L*3*H*2816 if name=='deepseek' else 0
                rows.append({'family':family,'q':q,'format':fmt,'path':path,'coefficient_bank_bytes':coeff,
                    'row_scale_bytes':scales,'leaf_bias_bytes':bias,'keys_bytes':keys,'P_bytes':P,'tree_bytes':tree,'meanX_bytes':means,
                    'nonFFN_KV_originalscales_known_bytes':common,'shared_placement_options':{
                        'AiF':{'host_known_floor_bytes':need},'host':{'host_known_floor_bytes':need+shared,'added_shared_coefficients_bytes':shared}},
                    'both8e9_8GiB_floor_fit_shared_AiF':[need<=b for b in result['budgets']],
                    'both8e9_8GiB_floor_fit_shared_host':[need+shared<=b for b in result['budgets']],
                    'affirmative_complete_host_fit_claim':False,'retained_original_raw_W8_FFN_fallback_bytes':source['models'][name]['original_raw_W8_FFN_fallback_bytes'],
                    'native_U_pages':(H*q*nb+16383)//16384 if path=='native_A8latent' else 0})
        result['models'][name]=rows
    old.save(OUT/'resource_ledger.json',result)

def comparison_freeze(points,ids):
    comparisons=[];groups={}
    for p in points:
        for phase in ['prefill','decode']:
            key=(p['nfit'],p['family'],p['q'],p['format'],p['path'],phase)
            c={'id':p['id']+':'+phase,'candidate':p['id']+':'+phase,'control':p['baseline_id']+':'+phase,
               'model':p['model'],'layer':p['layer'],'expert':p['expert'],'phase':phase}
            comparisons.append(c);groups.setdefault(key,[]).append(c)
    for key,parts in groups.items():
        for scope in ['deepseek','olmoe','both']:
            own=[p for p in parts if scope=='both' or p['model']==scope]
            if len(own)!=(8 if scope=='both' else 4):continue
            for agg in ['equal_operator','pooled_energy']:
                tag=hashlib.sha256(json.dumps([key,scope,agg]).encode()).hexdigest()
                comparisons.append({'id':tag,'candidate':tag+':candidate','control':tag+':control','components':own,
                    'aggregation':agg,'model_scope':scope,'config':key})
    result={'status':'frozen_before_development_SCORE','same_round':2,'scope':'development/exploratory; no confirmatory or wholemodel quality claim',
        'points':points,'comparisons':comparisons,'score_prompt_ids':ids,'selection':'gauge only by VALID; all fixed representations/runtime paths and N retained',
        'bootstrap':{'repeats':10000,'seed':20261103,'same40IDs':True,'undefined_or_nonliteral_zero_SD':'whole family inconclusive; never drop/redraw'}}
    old.save(OUT/'comparisons.before_evaluation.frozen.json',result)

def fitval(review):
    rd,receipt=verify(review);assert not (OUT/'FITVAL.started.json').exists()
    old.save(OUT/'FITVAL.started.json',{'at_utc':now(),'source_revision':receipt['revision'],'phase':'FITVAL_only','no_new_teacher_queries':True})
    pr=json.loads(PROTOCOL.read_text());sets,byid=frozen_points(pr)
    complete=json.loads((STAGE0/'FITVAL.complete.json').read_text())
    for p,h in complete['files'].items():assert old.sha(p)==h
    identity=json.loads((STAGE0/'content_identities.frozen.json').read_text());ids={s:[r['id'] for r in identity['records'] if r['split']==s] for s in ['FIT','VALID','SCORE']}
    num=old.Numeric(FROZEN0/'numeric.so');fma=FMA(rd/'fma.so');points=[];baselines={};inconclusive=[];fitroot=OUT/'fitted';fitroot.mkdir()
    for name,layers in [('deepseek',[1,13]),('olmoe',[0,8])]:
        for layer in layers:
            train=pipe.dataset(STAGE0/'capture'/name/'FIT',layer,ids['FIT']);valid=pipe.dataset(STAGE0/'capture'/name/'VALID',layer,ids['VALID'])
            tf=STAGE0/'fitted'/name/f'layer{layer}';tree,P,sigma,_=pipe.tree_load(tf)
            ds={fam:(old.design(num,train['x'],tree,32,P,sigma,128 if fam=='PE_RRR' else 2048,'normal' if fam=='PE_RRR' else 'identity'),
                     old.design(num,valid['x'],tree,32,P,sigma,128 if fam=='PE_RRR' else 2048,'normal' if fam=='PE_RRR' else 'identity')) for fam in ['PE_RRR','G_RRR']}
            inputs={(n,fam):input_statistics(ds[fam][0],train['prompt'],n,fam=='PE_RRR') for n in [80,120] for fam in ds}
            for n in [80,120]:
                for op in [p for p in sets[n] if p['model']==name and p['layer']==layer]:
                    started=time.monotonic();expert=op['expert'];folder=fitroot/f'{name}.L{layer}.E{expert}.N{n}';folder.mkdir();bank=folder/'models';bank.mkdir()
                    y=np.load(tf/f'expert{expert}.FIT.npy',mmap_mode='r',allow_pickle=False);yv=np.load(tf/f'expert{expert}.VALID.npy',mmap_mode='r',allow_pickle=False)
                    natural,_=pipe.route(valid,expert);localpoint=byid[op['id']];baselines[op['id']]=localpoint
                    raw=[]
                    for family in ['PE_RRR','G_RRR']:
                        inp=inputs[(n,family)];d=inp['d'];dv=ds[family][1]
                        As,prior,cross,stats,yn=fitted_slopes(inp,y,op['eta'] if family=='PE_RRR' else op['full_H_global_eta'],op['lambda'])
                        M=target_matrix(As,inp,cross,stats);U,info=output_basis(M)
                        np.savez(folder/(family+'.float_fit.npz'),slopes=np.stack(As),Mtarget=M,U512=U)
                        old.save(folder/(family+'.eigen.json'),info)
                        if family=='PE_RRR':
                            corrected,_=old.solve(inp['pcov'],cross,op['lambda'],prior,factor=inp['pfactor'])
                            a=old.lower(num,[prior],d,yn,np.ones(len(yn)),32,pooled=True)
                            b=old.lower(num,[corrected],d,yn,np.ones(len(yn)),32,pooled=True)
                            rebuilt=old.lower(num,As,d,yn,np.ones(len(yn)),32)
                            # Source identity comparison only, never retune on a mismatch.
                            rebuilt_hash=old.array_hash(rebuilt['qmaps'],rebuilt['scales'],rebuilt['bias'],np.asarray(rebuilt['unsupported'],np.int32))
                            old.save(folder/'pooled_closure.json',{'eta':op['eta'],'lambda':op['lambda'],
                                'relative_float_correction':float(np.linalg.norm(corrected-prior)/max(np.linalg.norm(prior),1e-300)),
                                'lowered_before_after_identical':all(np.array_equal(a[k],b[k]) for k in ['qmaps','scales','bias']),
                                'rebuilt_local_hash':rebuilt_hash,'frozen_local_hash':op['model_hash'],'rebuilt_local_identical':rebuilt_hash==op['model_hash']})
                        for q,fmt,path in FORMATS:
                            Uq=U[:,:q];B=np.stack([(A@Uq).T for A in As]);options=[]
                            for gauge in ['identity'] if fmt=='F32' else ['identity','FITrmsfloor_maxRMS_div16']:
                                model=calibrate(num,fma,lower_factors(num,Uq,B,d,fmt,path,gauge),d,yn)
                                pred=predict(num,fma,model,dv);covered=~np.isin(dv['leaf'],model['unsupported'])
                                loss=old.loss(pred,yv,natural,covered);mp,mh=archive(bank,model)
                                params={'model':name,'layer':layer,'expert':expert,'nfit':n,'family':family,'q':q,'format':fmt,'path':path,'gauge':gauge,
                                    'input_rank':d['q'].shape[1],'C':32,'guard':100,'model_path':mp,'model_hash':mh,'baseline_id':op['id'],
                                    'VALID_loss':loss,'unsupported_leaves':model['unsupported'].tolist(),'diagnostic_only':q==128 and fmt=='W8'}
                                params['id']=hashlib.sha256(json.dumps(params,sort_keys=True).encode()).hexdigest()
                                raw.append(params);options.append((params,model))
                            eligible=[x for x in options if x[0]['VALID_loss'] is not None]
                            if not eligible:
                                inconclusive.append({'operator':op['id'],'family':family,'q':q,'format':fmt,'path':path,'reason':'VALID_no_natural_hits_or_zero_energy'});continue
                            selected,model=min(eligible,key=lambda x:(x[0]['VALID_loss'],0 if x[0]['gauge']=='identity' else 1))
                            selected={**selected,'VALID_resources':resources(model,1408 if name=='deepseek' else 1024,natural,dv['leaf'])}
                            Ustored=model['U'].astype(np.float64)*(model['Us'].astype(np.float64)[:,None] if fmt=='W8' else 1)
                            selected['VALID_span_diagnostics']={'ideal_real_U':span_floor(Uq,model['bias'],yv,dv['leaf'],natural),
                                'actual_stored_U':span_floor(Ustored,model['bias'],yv,dv['leaf'],natural)}
                            points.append(selected)
                    old.save(folder/'VALID_all_predeclared_lowerings.json',raw)
                    old.save(folder/'fit_report.json',{'seconds':time.monotonic()-started,'N':n,'operator':op['id'],'raw_lowerings':len(raw),'all_data_development_only':True})
                    print(f'FITVAL completed {name} L{layer} E{expert} N{n}',flush=True)
            del inputs,ds,train,valid
    old.save(OUT/'baselines.frozen.json',baselines);old.save(OUT/'inconclusive_configurations.json',inconclusive)
    full_model_ledger(points);comparison_freeze(points,ids['SCORE'])
    deps=[x for x in fitroot.rglob('*') if x.is_file()]+[OUT/x for x in ['baselines.frozen.json','inconclusive_configurations.json','resource_ledger.json','comparisons.before_evaluation.frozen.json']]
    old.save(OUT/'FITVAL.complete.json',{'status':'complete_pending_development_SCORE_review','at_utc':now(),'source_revision':receipt['revision'],
        'points':len(points),'files':{str(x):old.sha(x) for x in deps},'SCORE_evaluated':False,'TEST_opened':False,
        'offline_parent_maxRSS_KiB':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'numeric_dot_audits':num.dot_audits,'no_new_teacher_queries':True})
    num.close()

def evaluate_inputs(num,fma,freeze,base,datasets,designs,targets,result):
    """Same real evaluator exercised by synthetic archive/interface smoke."""
    metrics={};stats={};result=Path(result);result.mkdir()
    for point in freeze['points']:
        name,layer,expert=point['model'],point['layer'],point['expert'];key=(name,layer)
        ds=datasets[key];op=key+(expert,)
        y=targets[op];natural,alpha=pipe.route(ds,expert);d=designs[key+(point['family'],)];model=load(point['model_path']);assert model_hash(model)==point['model_hash']
        pred=predict(num,fma,model,d);covered=~np.isin(d['leaf'],model['unsupported']);m=pipe.score_point(pred,y,ds,natural,alpha,covered)
        metrics[point['id']]={'point':point,'metrics':m,'resources_by_phase':{phase:resources(model,1408 if name=='deepseek' else 1024,natural&(ds['phase']==ph),d['leaf']) for phase,ph in [('prefill',0),('decode',1)]}}
        for phase in ['prefill','decode']:stats[point['id']+':'+phase]=m[phase]['natural']
        bid=point['baseline_id']
        if bid not in metrics:
            b=base[bid];bm=old.load_model(b['model_path']);bd=designs[key+('PE_RRR',)]
            bp=old.predict(num,bm,bd);bcov=~np.isin(bd['leaf'],bm['unsupported']);mb=pipe.score_point(bp,y,ds,natural,alpha,bcov)
            metrics[bid]={'point':b,'metrics':mb}
            for phase in ['prefill','decode']:stats[bid+':'+phase]=mb[phase]['natural']
    for c in freeze['comparisons']:
        if 'components' not in c:continue
        for endpoint in ['candidate','control']:
            parts=[stats[p[endpoint]] for p in c['components']];a=np.stack([p['sse'] for p in parts]);b=np.stack([p['energy'] for p in parts])
            if c['aggregation']=='pooled_energy':a,b=a.sum(axis=0),b.sum(axis=0)
            stats[c[endpoint]]={'sse':a.tolist(),'energy':b.tolist()}
    old.save(result/'prompt_metrics.json',metrics)
    # No factors are asserted literal equal to a direct bank merely because
    # their finite prompt errors coincide. A nonliteral zero-SD stays undefined.
    boot=old.bootstrap(freeze['comparisons'],stats,freeze['score_prompt_ids'],set())
    old.save(result/'paired_bootstrap.json',boot)
    return boot

def evaluate(review,evaluation_review):
    rd,_=verify(review,'SCORE_DEVELOPMENT',evaluation_review)
    assert not (OUT/'EVALUATION.started.json').exists()
    complete=json.loads((OUT/'FITVAL.complete.json').read_text())
    for p,h in complete['files'].items():assert old.sha(p)==h
    old.save(OUT/'EVALUATION.started.json',{'at_utc':now(),'phase':'SCORE_DEVELOPMENT','TEST_opened':False})
    freeze=json.loads((OUT/'comparisons.before_evaluation.frozen.json').read_text());base=json.loads((OUT/'baselines.frozen.json').read_text())
    num=old.Numeric(FROZEN0/'numeric.so');fma=FMA(rd/'fma.so');datasets={};designs={};targets={}
    for point in freeze['points']:
        name,layer,expert=point['model'],point['layer'],point['expert'];key=(name,layer)
        if key not in datasets:
            ds=pipe.dataset(STAGE0/'capture'/name/'SCORE',layer,freeze['score_prompt_ids']);datasets[key]=ds
            tree,P,sigma,_=pipe.tree_load(STAGE0/'fitted'/name/f'layer{layer}')
            for family in ['PE_RRR','G_RRR']:
                designs[key+(family,)]=old.design(num,ds['x'],tree,32,P,sigma,128 if family=='PE_RRR' else 2048,'normal' if family=='PE_RRR' else 'identity')
        op=key+(expert,)
        if op not in targets:targets[op]=np.load(STAGE0/'fitted'/name/f'layer{layer}'/f'expert{expert}.SCORE.npy',mmap_mode='r',allow_pickle=False)
    boot=evaluate_inputs(num,fma,freeze,base,datasets,designs,targets,OUT/'evaluation')
    num.close()
    old.save(OUT/'EVALUATION.complete.json',{'status':'complete_development_only_Root_critic_review_required','at_utc':now(),'points':len(freeze['points']),
        'comparisons':len(freeze['comparisons']),'family_inconclusive':boot['family_inconclusive'],'TEST_opened':False,'no_automatic_scientific_classification':True})

if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('action',choices=['prepare','fitval','evaluate']);ap.add_argument('--source-review');ap.add_argument('--evaluation-review');a=ap.parse_args()
    if a.action=='prepare':prepare()
    elif a.action=='fitval':fitval(a.source_review)
    else:evaluate(a.source_review,a.evaluation_review)
