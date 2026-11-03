#!/usr/bin/env python3
"""Extend the heavy-chip bound to independent GU/Down masks; no native."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2/chip-count-regime-screen'
MAIN=BASE/'free-position-bound'
OUT=MAIN/'addendum-phase-independent'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(name,x):
    p=OUT/name;assert not p.exists(),p
    p.write_text(json.dumps(x,indent=2)+'\n')
def cd(a,b):return (int(a)+int(b)-1)//int(b)


def freeze():
    OUT.mkdir(exist_ok=False)
    paths=[Path(__file__),MAIN/'protocol.frozen.json',MAIN/'report.json',MAIN/'arrays.npz',MAIN/'completion.json',
           BASE/'stage-b/granite4tiny.M32.minimum_depth.plan.npz',BASE/'g4-M32-coupled-cp/protocol.frozen.json']
    save('protocol.frozen.json',{'status':'frozen_before_independentGUDown_enumeration','utc':datetime.now(timezone.utc).isoformat(),
                                'inputs_sha256':{str(p):sha(p) for p in paths},
                                'scope':'Same32chips,unchanged25GUpositivebands40/41rowsC1536 and12Down128rowsC512 perexpert,K6,fixedshared,input1/4. GU/Down masks independent; overlapmayexceed5 andunionmaynotcover32.',
                                'enumeration':'Independent g,d in0..6 on eachchip;heavyifg+d+sharedwaves>=10; compute full command/input/read/output service directly, no native or fit.',
                                'claim':'Sameownedworkfloor andfirst-acquire-ranknecessaryLB extend toanyfixedrowmask inthisgeometry. Coupledmainwitnessisafeasiblesubfamily,sofreepositionownedoptimumunchanged. No claimforothergeometry/sharedfit/newcontexts.'})
    print(sha(OUT/'protocol.frozen.json'))


def run():
    proto=json.loads((OUT/'protocol.frozen.json').read_text())
    for p,h in proto['inputs_sha256'].items():assert sha(p)==h,p
    source=json.loads((BASE/'g4-M32-coupled-cp/protocol.frozen.json').read_text())['service']
    with np.load(BASE/'stage-b/granite4tiny.M32.minimum_depth.plan.npz') as a:shared=a['rows'][64].copy()
    with np.load(MAIN/'arrays.npz') as a:main={k:a[k].copy() for k in a}
    records=[];saved={}
    for copies in [1,4]:
        s=dict(source,input_channel_copies=copies)
        def op(R,C):
            if not R:return 0,0
            P=cd(R*C,s['page_bytes']);W=cd(P,s['planes']);E=P*s['page_bytes']//C
            inp=cd(C*s['input_bytes']*copies*10**9,s['channel_bytes_per_second'])
            out=cd(E*s['output_bytes']*10**9,s['channel_bytes_per_second'])
            return s['command_ns']+inp+W*s['ordinary_read_ns']+out,W
        G=op(41,1536)[0];D=op(128,512)[0];assert G==op(40,1536)[0]
        SW=np.array([op(int(shared[0,c]),1536)[0]+op(int(shared[1,c]),1024)[0] for c in range(32)],np.int64)
        WW=np.array([op(int(shared[0,c]),1536)[1]+op(int(shared[1,c]),1024)[1] for c in range(32)],np.int64)
        assert np.array_equal(SW,main[f'input{copies}.shared_work'])
        assert np.array_equal(WW,main[f'input{copies}.shared_waves'])
        states=[];heavy=[]
        for chip in range(32):
            for g in range(7):
                for d in range(7):
                    waves=g+d+int(WW[chip]);work=G*g+D*d+int(SW[chip])
                    row=[chip,g,d,waves,work]
                    states.append(row)
                    if waves>=10:
                        heavy.append(row)
                        x,o=g+d-6,6-g
                        assert x>=0 and o>=0 and x+o<=6
                        assert work==6*G+int(SW[chip])+D*x+(D-G)*o
        states=np.array(states,np.int64);heavy=np.array(heavy,np.int64)
        minimum=int(heavy[:,4].min())
        expected=287752 if copies==1 else 298504
        assert minimum==expected
        total=6*25+6*12+int(WW.sum());assert total==295
        minheavies=total-32*9;assert minheavies==7
        rankcase=s['host_submit_ns']+6*s['command_ns']+minimum
        readcase=s['host_submit_ns']+11*s['ordinary_read_ns']
        bound=min(rankcase,readcase)
        assert bound==(291252 if copies==1 else 302004)
        X,O=main[f'input{copies}.X'],main[f'input{copies}.O']
        GU=~O;Down=X|O
        g,d=GU.sum(0),Down.sum(0)
        assert np.all(GU.sum(1)==25) and np.all(Down.sum(1)==12)
        witness=G*g+D*d+SW
        assert witness.max()==minimum and np.array_equal(witness,main[f'input{copies}.chip_work'])
        saved[f'input{copies}.all_chip_g_d_wave_work_states']=states
        saved[f'input{copies}.heavy_states']=heavy
        records.append({'input_copies':copies,'independent_g_d_states':len(states),'heavy_states':len(heavy),
                        'min_heavy_owned_ns':minimum,'minimum_states_chip_g_d_wave_work':heavy[heavy[:,4]==minimum].tolist(),
                        'all_heavy_states_representable_in_previous_x_o_enumeration':True,
                        'initial_host_plus_rank_native_LB_ns':bound,'initial_host_plus_any11read_LB_ns':readcase,
                        'allquery_readwaves':295,'minimum_10wave_owners_if_max10':7,
                        'coupled_witness_feasible_for_independent_family':True,'independent_freeposition_ownedwork_optimum_ns':minimum,
                        'subfamily_witness_chip_GU_Down_counts':np.stack([g,d],axis=1).tolist()})
    save('report.json',{'status':'PASS phase-independent fixedgeometry scope extension','protocol_sha256':sha(OUT/'protocol.frozen.json'),
                        'results':records,'proof':'Anyheavychiphasg+d>=10-maxsharedwaves=7. Setx=g+d-6,o=6-g;thenx,o>=0andx+o=d<=6,exactworkmatchesoriginalstate. Global295waves/rankargumentusesnominimumoverlapcondition. Thecoupledconstructionattainsthesamelowerownedworkwithinthisbroaderfamily.',
                        'limits':'Only25GU40/41rowjobs+12Down128rowjobs,fullinput/read costs andunchangedfixedshared/Service;nototherrowmultisets,sharedfit,newcontexts,commandpreemption,chipparalleljobs,freePB or newnativeABI. No native/fit/weights/heldout.'})
    p=OUT/'arrays.npz';assert not p.exists();np.savez_compressed(p,**saved)
    print(json.dumps(records,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);a=ap.parse_args()
    freeze() if a.mode=='freeze' else run()
