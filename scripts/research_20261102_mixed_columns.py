#!/usr/bin/env python3
"""Matched train-only audit of fixed and mixed asymmetric column rectangles.

Composes dense GEMVs and counts all partial outputs; native result contract and
host reduction remain unresolved. This is a generic coalesced-2D baseline audit,
not a new-method or hardware-legality claim. No heldout paths are opened.
"""
import argparse
import ctypes
from datetime import datetime, timezone
import json
from pathlib import Path
import subprocess

import numpy as np

from research_20261102_projection_study import evaluate
from run_corpus_traces import ROOT, sha256

DEST=ROOT/'results/research-2026-11-02/round2/adversarial-design/mixed-columns'
FAMILIES={'whole_rows_03':[0,3],'fixed_asymmetric_12':[1,2],'mixed_0123':[0,1,2,3]}


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def components(selected,g,R,C,common):
    chosen=g[selected];S=chosen.sum(axis=1);Z=((chosen>0)&(chosen<3)).sum(axis=1)
    result=dict(flash_ns=np.maximum(S,3*selected.shape[1]-S)*28000,
        control_ns=8*(selected.shape[1]+Z)*250,
        host_ns=np.full(len(selected),2000,dtype=np.int64),
        input_ns=np.full(len(selected),selected.shape[1]*C//2,dtype=np.int64),
        output_ns=R//2*(selected.shape[1]+Z))
    result['model_ns']=sum(result.values())
    result['interior_selected']=Z
    return result


def scalar(selected,g,R,C,common):
    """Independent literal chip/rectangle accounting for a route."""
    wave=[0]*16;pages=[0]*16;inp=[0]*16;out=[0]*16;cmd=0
    for e in selected:
        for c in range(16):
            width=(int(g[e]) if c<8 else 3-int(g[e]))*(C//3)
            if not width:continue
            size=(R//8)*width
            assert size%65536==0
            pages[c]+=size//16384;wave[c]+=size//65536
            inp[c]+=width;out[c]+=(R//8)*4;cmd+=1
    host_input=C*(1 if common else len(selected))
    result=dict(flash_ns=max(max(wave)*28000,max(pages)*2560),control_ns=cmd*250,
        host_ns=2000,input_ns=max((host_input+7)//8,max((inp[c]+inp[c+8]+1)//2 for c in range(8))),
        output_ns=max((sum(out)+7)//8,max((out[c]+out[c+8]+1)//2 for c in range(8))))
    result['model_ns']=sum(result.values())
    return result


def summary(v,w):
    expanded=np.repeat(v,w)
    return dict(mean_ns=float(v@w/w.sum()),p95_ns=float(np.percentile(expanded,95)),
        p99_ns=float(np.percentile(expanded,99)),min_ns=int(v.min()),max_ns=int(v.max()))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--iterations',type=int,default=12000)
    ap.add_argument('--sweeps',type=int,default=100);args=ap.parse_args()
    DEST.mkdir(parents=True,exist_ok=True)
    cpp=Path(__file__).with_suffix('.cpp');so=DEST/('fit-'+sha256(cpp)+'.so')
    if not so.exists():subprocess.run(['g++','-std=c++17','-O3','-march=native','-shared','-fPIC',str(cpp),'-o',str(so)],check=True)
    lib=ctypes.CDLL(str(so));p32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');p64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    lib.fit_columns.argtypes=[p32,p32,ctypes.c_int,p32,p64]+[ctypes.c_int]*7+[ctypes.c_uint64,ctypes.c_double,p64]
    lib.fit_columns.restype=ctypes.c_double
    models=[]
    for model,target_phase in [('granite4tiny','gate_up'),('qwen3','down')]:
        path=ROOT/'results/research-2026-11-02/round2/unbalanced-group'/(model+'.screen.strong.frozen.json')
        d=json.loads(path.read_text());plan_sha=sha256(path);route=Path(d['training_routes'])
        assert sha256(route)==d['training_routes_sha256']
        with np.load(route) as f:
            routes=f['routes'][:32,:64].copy();layers=f['layers'].tolist();ids=f['prompt_ids'][:32].tolist()
        assert len(ids)==32 and all(s.startswith('train_') for s in ids)
        phases=[];K=d['active'];E=d['experts']
        for p in d['phases']:
            if p['name']!=target_phase:continue
            R,C=p['total_rows'],p['columns'];assert R%8==C%3==0 and (R//8)*(C//3)==65536
            selected,weights=np.unique(np.sort(routes[:,:,layers.index(p['layer'])].reshape(-1,K),axis=1),axis=0,return_counts=True)
            selected=np.ascontiguousarray(selected,dtype=np.int32);weights=np.ascontiguousarray(weights,dtype=np.int64)
            oldrows=np.asarray(p['chosen_rows']);assert np.all(np.count_nonzero(oldrows,axis=1)==8)
            color=(oldrows[:,0]>0).astype(np.int32)
            original=evaluate(selected,oldrows,C,p['common'])
            freq=np.bincount(selected.ravel(),weights=np.repeat(weights,K),minlength=E)
            fcolor=np.zeros(E,dtype=np.int32);load=[0.,0.]
            for e in np.argsort(-freq):
                c=int(load[1]<load[0]);fcolor[e]=c;load[c]+=freq[e]
            binary=[color,1-color,fcolor,1-fcolor]
            for seed in range(4):binary.append(np.random.default_rng(20261102+seed).integers(0,2,E,dtype=np.int32))
            # Preserve good colors discovered in the initial raw-ns exploratory
            # schedule. This is a shared seed pool, available to all families.
            # Its binary objective is affine-equivalent for both fixed families.
            pool=[];pool_best=None;pool_allowed=np.asarray([0,3],dtype=np.int32)
            for schedule,start_count in [('raw_ns',8),('normalized_union',16)]:
                stage_best=None
                for j in range(start_count+2):
                    g=3*binary[j%8] if j<start_count else np.asarray(stage_best['g'],dtype=np.int32)
                    temp=[0.,500.,3000.,10000.][j%4] if j<start_count else [500.,3000.][j-start_count]
                    if schedule=='normalized_union' and (j<8 or j>=start_count):temp*=3
                    rng_index=j if schedule=='raw_ns' else (j%8 if j<16 else j-8)
                    g=np.ascontiguousarray(g.copy(),dtype=np.int32);stats=np.zeros(4,dtype=np.int64)
                    score=lib.fit_columns(g,pool_allowed,2,selected,weights,E,len(selected),K,R,C,args.iterations,args.sweeps,
                        20261102+p['layer']*1009+rng_index*31,temp,stats)
                    record=dict(schedule=schedule,mean_ns=score,g=g.tolist(),temperature_ns=temp)
                    pool.append(record)
                    if stage_best is None or score<stage_best['mean_ns']:stage_best=record
                    if pool_best is None or score<pool_best['mean_ns']:pool_best=record
            pooled_color=np.asarray(pool_best['g'],dtype=np.int32)//3
            winners={};all_runs={}
            for family,allowed_list in FAMILIES.items():
                allowed=np.ascontiguousarray(allowed_list,dtype=np.int32);runs=[];best=None
                seeds=[]
                for j in range(16):
                    b=binary[j%8]
                    if family=='whole_rows_03':g=3*b
                    elif family=='fixed_asymmetric_12':g=1+b
                    else:g=3*b if j%2==0 else 1+b
                    # Include both formerly tried raw-ns schedules in the
                    # common normalized pool; do not discard a known good seed
                    # merely to impose identical trajectories afterwards.
                    temp=[0.,500.,3000.,10000.][j%4]/(3 if j>=8 else 1)
                    seeds.append((f'matched_binary_{j}',np.ascontiguousarray(g,dtype=np.int32),temp))
                for j,b in enumerate([pooled_color,1-pooled_color]):
                    g=3*b if family=='whole_rows_03' or (family=='mixed_0123' and j==0) else 1+b
                    seeds.append((f'shared_pool_{j}',np.ascontiguousarray(g,dtype=np.int32),0.))
                # Identical two additional full budgets per family. The mixed
                # family starts these at the two already-fitted nested controls.
                for j in range(20):
                    if j<18:label,g,temp=seeds[j]
                    elif family=='mixed_0123':
                        key=['whole_rows_03','fixed_asymmetric_12'][j-18]
                        label='nested_control_'+key;g=np.asarray(winners[key]['g'],dtype=np.int32);temp=[500.,3000.][j-18]
                    else:
                        label='own_best_restart_'+str(j-18);g=np.asarray(best['g'],dtype=np.int32);temp=[500.,3000.][j-18]
                    # Binary fixed-family deltas differ by exactly3x. Scale
                    # temperature so their paired trajectories cannot differ
                    # solely because the same colors use different wave slopes.
                    if family=='whole_rows_03':temp*=3
                    g=np.ascontiguousarray(g.copy(),dtype=np.int32);stats=np.zeros(4,dtype=np.int64)
                    score=lib.fit_columns(g,allowed,len(allowed),selected,weights,E,len(selected),K,R,C,args.iterations,args.sweeps,
                        20261102+p['layer']*1009+(j%8 if j<16 else j-10)*31,temp,stats)
                    v=components(selected,g,R,C,p['common']);reference=float(v['model_ns']@weights/weights.sum())
                    assert abs(score-reference)<1e-6
                    for event in sorted({0,len(selected)//2,len(selected)-1}):
                        ss=scalar(selected[event],g,R,C,p['common'])
                        for key,value in ss.items():assert value==v[key][event],(key,value,v[key][event])
                    record=dict(start=label,temperature_ns=temp,mean_ns=score,g=g.tolist(),
                        valid_moves=int(stats[0]),anneal_accepts=int(stats[1]),greedy_moves=int(stats[2]),coordinate_converged=bool(stats[3]))
                    runs.append(record)
                    if best is None or score<best['mean_ns']:best=record
                g=np.asarray(best['g'],dtype=np.int32);v=components(selected,g,R,C,p['common'])
                # Every route is independently checked, not only sampled rows.
                for event in range(len(selected)):
                    for key,value in scalar(selected[event],g,R,C,p['common']).items():assert value==v[key][event]
                avgz=float(v['interior_selected']@weights/weights.sum())
                winner=dict(g=g.tolist(),g_counts=np.bincount(g,minlength=4).tolist(),cost=summary(v['model_ns'],weights),
                    component_mean_ns={key:float(v[key]@weights/weights.sum()) for key in ['flash_ns','control_ns','host_ns','input_ns','output_ns']},
                    mean_interior_selected=avgz,mean_host_sum_scalar_adds=avgz*R,mean_minimum_three_array_pass_bytes=12*R*avgz,
                    useful_bytes=E*R*C,physical_pages=E*R*C//16384,
                    per_chip_useful_bytes=[int((g if c<8 else 3-g).sum())*(R//8)*(C//3) for c in range(16)],
                    coordinate_converged_all_runs=all(r['coordinate_converged'] for r in runs))
                winners[family]=winner;all_runs[family]=runs
            # Any binary color found by any family is a valid color in both
            # fixed families. Share it afterwards too: a better binary local
            # minimum must not be reported as a mixed-shape contribution.
            colors=[np.asarray(winners['whole_rows_03']['g'])//3,
                np.asarray(winners['fixed_asymmetric_12']['g'])-1,pooled_color]
            mixed_g=np.asarray(winners['mixed_0123']['g'])
            if np.all((mixed_g==0)|(mixed_g==3)):colors.append(mixed_g//3)
            if np.all((mixed_g==1)|(mixed_g==2)):colors.append(mixed_g-1)
            shared_color=min(colors,key=lambda b:float(components(selected,3*b,R,C,p['common'])['model_ns']@weights))
            for family in ['whole_rows_03','fixed_asymmetric_12']:
                g=3*shared_color if family=='whole_rows_03' else 1+shared_color
                v=components(selected,g,R,C,p['common']);avgz=float(v['interior_selected']@weights/weights.sum())
                winners[family].update(g=g.tolist(),g_counts=np.bincount(g,minlength=4).tolist(),cost=summary(v['model_ns'],weights),
                    component_mean_ns={key:float(v[key]@weights/weights.sum()) for key in ['flash_ns','control_ns','host_ns','input_ns','output_ns']},
                    mean_interior_selected=avgz,mean_host_sum_scalar_adds=avgz*R,mean_minimum_three_array_pass_bytes=12*R*avgz,
                    per_chip_useful_bytes=[int((g if c<8 else 3-g).sum())*(R//8)*(C//3) for c in range(16)],
                    post_search_binary_color_sharing=True)
            nested=min(['whole_rows_03','fixed_asymmetric_12'],key=lambda k:winners[k]['cost']['mean_ns'])
            if winners[nested]['cost']['mean_ns']<winners['mixed_0123']['cost']['mean_ns']:
                winners['mixed_0123']=dict(winners[nested],nested_fixed_fallback=nested)
            np.testing.assert_array_equal(np.asarray(winners['whole_rows_03']['g'])//3,np.asarray(winners['fixed_asymmetric_12']['g'])-1)
            fixed=min(['whole_rows_03','fixed_asymmetric_12'],key=lambda k:winners[k]['cost']['mean_ns'])
            mix=winners['mixed_0123'];fixed_win=winners[fixed]
            old=summary(original['model_ns'],weights)
            phase=dict(layer=p['layer'],name=p['name'],rows=R,columns=C,active=K,common=p['common'],unique_routes=len(selected),
                legacy_whole_row=old,winners=winners,best_fixed_family=fixed,
                mixed_gain_vs_best_fixed_ns=fixed_win['cost']['mean_ns']-mix['cost']['mean_ns'],
                mixed_gain_vs_legacy_ns=old['mean_ns']-mix['cost']['mean_ns'],
                mixed_host_sum_break_even_vs_legacy_ns=old['mean_ns']-mix['cost']['mean_ns'],
                mixed_vs_best_fixed_hostsum_condition='mixed host_sum_ns minus fixed host_sum_ns must be below mixed_gain_vs_best_fixed_ns',
                native_requirements=['Read-only dense GEMV on contiguous partial-column matrices; same x segments for GU.',
                    'Existing output datatype/dequantization must allow host partial-output summation without a new hardware or ABI requirement.',
                    'All commands reload their own input segment, including common-input GU. Host reduction and shape-dependent PE throughput not calibrated.'],
                search=all_runs,shared_binary_pool=pool)
            assert mix['cost']['mean_ns']<=min(v['cost']['mean_ns'] for v in winners.values())+1e-6
            phases.append(phase)
            print(model,p['layer'],[(k,round(v['cost']['mean_ns'],3),v['g_counts']) for k,v in winners.items()],
                'mixed-extra',phase['mixed_gain_vs_best_fixed_ns'],'hostsum-budget',phase['mixed_host_sum_break_even_vs_legacy_ns'],flush=True)
            write(DEST/(model+'.partial.json'),phases)
        unaffected=sum(p['chosen_ns'] for p in d['phases'] if p['name']!=target_phase)
        legacy_total=unaffected+sum(p['legacy_whole_row']['mean_ns'] for p in phases)
        fixed_total=unaffected+sum(p['winners'][p['best_fixed_family']]['cost']['mean_ns'] for p in phases)
        mixed_total=unaffected+sum(p['winners']['mixed_0123']['cost']['mean_ns'] for p in phases)
        models.append(dict(model=model,phases=phases,training_routes=str(route),training_routes_sha256=sha256(route),
            source_plan=str(path),source_plan_sha256=plan_sha,training_slice='[:32,:64]',prompt_ids=ids,
            routed_ffn_screen_total_ns=dict(legacy=legacy_total,best_fixed=fixed_total,mixed=mixed_total),
            whole_ffn_mixed_gain_beyond_fixed_pct=100*(1-mixed_total/fixed_total),
            whole_ffn_mixed_gain_vs_legacy_excluding_host_sum_pct=100*(1-mixed_total/legacy_total),
            aggregate_mean_host_sum_budget_vs_legacy_ns=legacy_total-mixed_total))
        assert sha256(path)==plan_sha
    output=dict(status='training-only conditional cost screen; no novelty, native-legality or end-to-end certificate',
        frozen_utc=datetime.now(timezone.utc).isoformat(),models=models,arguments=vars(args),
        objective='28000*max(S,3*K-S)+(2000+R/2)*(K+Z)+2000+K*C/2 ns; S=sum selected g, Z=count selected g in {1,2}',
        matched_budget='A28-run binary seed pool preserves both earlier raw-ns(10runs) and normalized-union(18runs) schedules, and is shared by all families. Each family then has eight common binary seeds at each of two normalized temperature schedules, two shared-pool seeds, and two restarts(20runs), with equal iterations/sweeps and RNG schedules. Whole-row03 temperatures are3x asymmetric12 because their binary move objective deltas differ by exactly3x. Every binary color discovered by any family is finally shared with both fixed controls; paired winning colors are asserted equal, preventing a better color local minimum from masquerading as a mixed-shape gain. Mixed last two starts use the fitted nested controls; fixed families use their own fitted winner. No global optimality claim.',
        physical_inventory='One copy of each weight; 8 row blocks per expert; each block columns split g:(3-g). All partial rectangle bytes are multiples of64KiB. Same useful INT8 bytes and physical pages for all g. Per-chip inventory is allowed to differ;20GiB chip capacity is nonbinding in this small screen.',
        limitations=['FP32 output transfer inherited from existing service model; native output/dequantization contract unverified.',
            'Host summation is excluded and must fit the reported margins. No native arithmetic throughput calibrated.',
            'No heldout read, routing change, new hardware, new quantization, replication, or frozen-main-plan modification.',
            'Fractional expert sharding and coactivation-aware placement have direct HDA-MoE prior. Uniform2D page tiling and output reduction have Cambricon-LLM prior. This is a stronger generic baseline, not a new2D-method claim.'],
        source_sha256={p.name:sha256(p) for p in [Path(__file__),cpp]})
    write(DEST/'screen.json',output)
    print(json.dumps({m['model']:{k:m[k] for k in ['routed_ffn_screen_total_ns','whole_ffn_mixed_gain_beyond_fixed_pct','aggregate_mean_host_sum_budget_vs_legacy_ns']} for m in models},indent=2))


if __name__=='__main__':main()
