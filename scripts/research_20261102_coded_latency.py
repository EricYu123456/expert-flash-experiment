#!/usr/bin/env python3
"""Persistent-background screen for exact integer local-parity output repair.

Uses the strongest frozen projection rows, never substitutes weak g16 data
placement for coding. Busy intervals exist on one absolute device timeline;
foreground completion never clears an erase. Decisions see only the currently
busy chip and its known busy_until. Mid-phase new erases are counted offline
as boundary uncertainty, not exposed to the decision maker.
"""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import numpy as np

from run_corpus_traces import ROOT, sha256

DEST=ROOT/'results/research-2026-11-02/round2/coded-latency'
FIELDS=('waves','pages','useful_bytes','commands','inputs','outputs')


def ceildiv(a,b):return (a+b-1)//b


class Background:
    def __init__(self,length_ns,duty,seed=20261102,shuffle=False):
        self.length_ns=int(length_ns);self.duty=float(duty)
        self.period=self.length_ns/self.duty if duty else float('inf')
        self.offset=self.period*.37 if duty else float('inf')
        self.chips=np.arange(16)
        if shuffle:np.random.default_rng(seed).shuffle(self.chips)

    def current(self,now):
        if not self.duty or now<self.offset:return None
        event=int((now-self.offset)//self.period)
        start=self.offset+event*self.period
        if now<start+self.length_ns:
            return int(self.chips[event%16]),start+self.length_ns,event
        return None

    def future_start_crossed(self,start,end):
        # Audit only. This value must never affect foreground decisions.
        if not self.duty or end<=self.offset:return False
        event=max(0,int(np.floor((start-self.offset)/self.period))+1)
        if start<self.offset:event=0
        return self.offset+event*self.period<end


def basic_ledger(selected,rows,columns,common):
    counts=np.bincount(selected,minlength=len(rows))
    useful=rows*columns;pages=ceildiv(useful,16384);waves=ceildiv(pages,4)
    return dict(waves=counts@waves,pages=counts@pages,useful_bytes=counts@useful,
                commands=counts@ceildiv(waves,192),
                inputs=((rows[selected]>0).any(axis=0) if common else (rows[selected]>0).sum(axis=0))*columns,
                outputs=(counts@rows)*4,host_input=columns*(1 if common else len(selected)))


def add_service(ledger,mac_gweights_s):
    ledger['service_ns']=np.maximum(ledger['waves']*28000,
                                    np.maximum(ledger['pages']*2560,ledger['useful_bytes']/mac_gweights_s))
    return ledger


def phase_duration(ledger,now,busy=None,decode_ns=0,cap_ns=None):
    inputs=ledger['inputs'];outputs=ledger['outputs'];service=ledger['service_ns']
    prefix=2000+250*ledger['commands'].sum()+max(ceildiv(ledger['host_input'],8),ceildiv((inputs[:8]+inputs[8:]).max(),2))
    finish=now+prefix+service.max()
    suspended=False;debt=0.
    if busy is not None:
        chip,busy_until,_=busy
        if service[chip]>0:
            ready=busy_until if cap_ns is None else min(busy_until,now+cap_ns)
            finish=now+prefix+max(service.max(),max(0,ready-now-prefix)+service[chip])
            suspended=cap_ns is not None and now+cap_ns<busy_until
            if suspended:debt=float(service[chip])
    output=max(ceildiv(outputs.sum(),8),ceildiv((outputs[:8]+outputs[8:]).max(),2))
    return float(finish-now+output+decode_ns),suspended,debt


def parity_plan(rows,slots,columns):
    extra=np.zeros((16,),dtype=np.int64);parts=[]
    for expert in range(len(rows)):
        for half in (0,1):
            padded=int(rows[expert,half*8:half*8+8].max(initial=0))
            if not padded:continue
            owner0=(1-half)*8+int(slots[expert])%8
            owner1=(1-half)*8+(int(slots[expert])+1)%8
            assert owner0!=owner1
            for owner in (owner0,owner1):extra[owner]+=padded*columns
            parts.append((expert,half,padded,owner0,owner1))
    data=int(rows.sum()*columns)
    pages=sum(2*ceildiv(part[2]*columns,16384) for part in parts)
    return parts,dict(data_weight_bytes=data,parity_weight_bytes=int(extra.sum()),
                      extra_weight_fraction=float(extra.sum()/data),parity_bytes_per_chip=extra.tolist(),
                      parity_physical_pages=int(pages),data_physical_pages=int(ceildiv(rows*columns,16384).sum()),
                      parity_storage_peak_over_mean=float(extra.max()/extra.mean()))


def repair_ledger(selected,base,rows,columns,parts,erased,mac_rate):
    coded={key:base[key].copy() if isinstance(base[key],np.ndarray) else base[key] for key in FIELDS+('host_input',)}
    for key in FIELDS:coded[key][erased]=0
    added=Counter();needed=False
    selected_set=set(int(e) for e in selected)
    half=erased//8
    for expert,h,padded,owner0,owner1 in parts:
        if h!=half or expert not in selected_set or rows[expert,erased]==0:continue
        needed=True;useful=padded*columns;pages=ceildiv(useful,16384);waves=ceildiv(pages,4)
        for owner in (owner0,owner1):
            assert owner!=erased
            coded['waves'][owner]+=waves;coded['pages'][owner]+=pages;coded['useful_bytes'][owner]+=useful
            coded['commands'][owner]+=ceildiv(waves,192)
            # Conservatively re-stage parity operands even if a data GEMV on
            # that owner had the same vector; host logical vector fanout once.
            coded['inputs'][owner]+=columns;coded['outputs'][owner]+=padded*4
            added['parity_gemvs']+=1;added['parity_weight_bytes']+=useful;added['parity_pages']+=pages
            added['parity_commands']+=ceildiv(waves,192);added['parity_input_chip_bytes']+=columns
            added['parity_output_bytes']+=padded*4
    assert coded['waves'][erased]==0
    add_service(coded,mac_rate)
    return coded,needed,dict(added)


def compile_model(name,args):
    frozen_path=ROOT/'results/research-2026-11-02/projection-study'/f'{name}.frozen_plan.json'
    frozen=json.loads(frozen_path.read_text());legacy_path=Path(frozen['legacy_path']);legacy=json.loads(legacy_path.read_text())
    assert sha256(legacy_path)==frozen['legacy_sha256']
    if args.external:
        route_path=ROOT/f'results/research-2026-11-02/round2/public-validation/{name}/external_routes.npz'
    else:route_path=Path(frozen['train_path'])
    with np.load(route_path) as z:
        routes=z['routes'];layers=z['layers'].tolist() if 'layers' in z else frozen['layers']
    assert layers==frozen['layers']
    if not args.external:assert sha256(route_path)==frozen['train_sha256']
    indices=np.linspace(0,routes.shape[0]*routes.shape[1]-1,args.tokens,dtype=int)
    selected=routes.reshape(-1,len(layers),frozen['active'])[indices]
    phase_specs=[];capacity=[]
    for phase in frozen['phases']:
        rows=np.asarray(phase['chosen_rows'],dtype=np.int64)
        label=phase['chosen_label']
        slots=legacy['mappings'][label][str(phase['layer'])] if label.startswith('group_only') else np.arange(frozen['experts'])
        parts,cap=parity_plan(rows,slots,phase['columns']);capacity.append(dict(layer=phase['layer'],phase=phase['name'],**cap))
        phase_specs.append((phase,rows,parts))
    data=[];equal=[];coded=[];needed=[];extra=[]
    for token in selected:
        for pi,(phase,rows,parts) in enumerate(phase_specs):
            route=token[pi//2]
            base=add_service(basic_ledger(route,rows,phase['columns'],phase['common']),args.mac_rate)
            baseline=add_service(basic_ledger(route,np.asarray(phase['baseline_rows']),phase['columns'],phase['common']),args.mac_rate)
            repairs=[];required=[];extras=[]
            for erased in range(16):
                repair,need,add=repair_ledger(route,base,rows,phase['columns'],parts,erased,args.mac_rate)
                repairs.append(repair);required.append(need);extras.append(add)
            data.append(base);equal.append(baseline);coded.append(repairs);needed.append(required);extra.append(extras)
    def stack(records):
        return {key:np.asarray([record[key] for record in records]) for key in FIELDS+('host_input','service_ns')}
    data_stack=stack(data);equal_stack=stack(equal)
    code_stack={key:np.asarray([[record[key] for record in records] for records in coded]) for key in FIELDS+('host_input','service_ns')}
    export={**{'data_'+k:v for k,v in data_stack.items()},**{'g16_'+k:v for k,v in equal_stack.items()},
            **{'coded_'+k:v for k,v in code_stack.items()},'needed':np.asarray(needed),'selected':selected,'sample_indices':indices,
            'phase_layer':np.array([p['layer'] for p,_,_ in phase_specs]),'phase_name':np.array([p['name'] for p,_,_ in phase_specs])}
    phase_dir=args.output/'phase_ledgers';phase_dir.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(phase_dir/f'{name}.npz',**export)
    return frozen,route_path,indices,data,equal,coded,needed,extra,capacity


def simulate(data,equal,coded,needed,extra,layers,background,kind,decode_ns,cap_ns):
    now=0.;token_start=0.;latencies=[];stats=Counter()
    for index,base in enumerate(equal if kind=='g16_wait' else data):
        busy=background.current(now);wait,suspended,debt=phase_duration(base,now,busy,0,cap_ns)
        duration=wait;used_code=False
        if busy is not None:
            chip,end,event=busy;stats['phase_start_busy']+=1
            stats['phase_start_busy_and_data_active']+=int(base['service_ns'][chip]>0)
            if kind=='coded' and needed[index][chip]:
                phase_decode=float(decode_ns[index%(layers*2)]) if isinstance(decode_ns,(list,np.ndarray)) else decode_ns
                repair,_,_=phase_duration(coded[index][chip],now,busy,phase_decode,None)
                if repair<wait:
                    duration=repair;used_code=True;suspended=False;debt=0
                    stats['coded_repair_phases']+=1
                    for key,value in extra[index][chip].items():stats[key]+=value
        stats['suspend_bound_phases']+=int(suspended);stats['foreground_erase_service_debt_lower_bound_ns']+=debt
        stats['new_erase_start_inside_phase_uncertainty']+=int(background.future_start_crossed(now,now+duration))
        now+=duration
        if (index+1)%(layers*2)==0:
            latencies.append((now-token_start)/1e6);token_start=now
    values=np.asarray(latencies)
    return dict(mean_ms=float(values.mean()),p50_ms=float(np.quantile(values,.5)),p95_ms=float(np.quantile(values,.95)),
                p99_ms=float(np.quantile(values,.99)),total_elapsed_ms=now/1e6,per_token_ms=latencies,stats=dict(stats))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',default=['granite','deepseek','qwen3'])
    parser.add_argument('--tokens',type=int,default=32)
    parser.add_argument('--external',action='store_true')
    parser.add_argument('--mac-rate',type=float,default=6.4)
    parser.add_argument('--output',type=Path,default=DEST)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);models=[]
    for name in args.models:
        started=time.time()
        frozen,path,indices,data,equal,coded,needed,extra,capacity=compile_model(name,args)
        measured_path=ROOT/'results/research-2026-11-02/round2/integer-parity/host_decode_measurements.json'
        measured=json.loads(measured_path.read_text())
        measured_costs={key:[] for key in ('measured_median','measured_p95')}
        measured_shapes=[]
        for phase in frozen['phases']:
            maximum_rows=int(np.max(phase['chosen_rows']))
            match=[m for m in measured['measurements'] if m['known_shards']==7 and m['rows']==maximum_rows and m['experts']==frozen['active']]
            assert len(match)==1,(name,phase['name'],maximum_rows,frozen['active'])
            measured_costs['measured_median'].append(match[0]['median_ns']);measured_costs['measured_p95'].append(match[0]['p95_ns'])
            measured_shapes.append(dict(layer=phase['layer'],phase=phase['name'],rows=maximum_rows,experts=frozen['active'],known_shards=7))
        scenarios=[]
        for length_ms in (3,5):
            for duty in (0,.02,.1,.25):
                background=Background(length_ms*1000000,duty)
                policies={}
                for cap in (None,0,10000,100000,1000000):
                    cap_label='none' if cap is None else str(cap)
                    for kind in ('chosen_wait','g16_wait','coded'):
                        for decode in ((0,2000,10000,'measured_median','measured_p95') if kind=='coded' else (0,)):
                            key=f'{kind}_cap{cap_label}_decode{decode}'
                            policies[key]=simulate(data,equal,coded,needed,extra,len(frozen['layers']),background,kind,measured_costs[decode] if isinstance(decode,str) else decode,cap)
                # With instantaneous suspension, the coded min must never lose.
                for decode in (0,2000,10000):
                    assert policies[f'coded_cap0_decode{decode}']['mean_ms']<=policies['chosen_wait_cap0_decode0']['mean_ms']+1e-7
                scenarios.append(dict(erase_ms=length_ms,duty=duty,period_ms=background.period/1e6,offset_ms=background.offset/1e6,
                                      chip_cycle=background.chips.tolist(),policies=policies))
        total_data=sum(c['data_weight_bytes'] for c in capacity);total_parity=sum(c['parity_weight_bytes'] for c in capacity)
        result=dict(model=name,route_path=str(path),route_sha256=sha256(path),sample_indices=indices.tolist(),tokens=len(indices),
                    frozen_projection_sha256=sha256(ROOT/f'results/research-2026-11-02/projection-study/{name}.frozen_plan.json'),
                    capacity=dict(data_weight_bytes=total_data,parity_weight_bytes=total_parity,extra_weight_fraction=total_parity/total_data,
                                  physical_page_extra_fraction=sum(c['parity_physical_pages'] for c in capacity)/sum(c['data_physical_pages'] for c in capacity),phases=capacity),
                    measured_decode=dict(path=str(measured_path),sha256=sha256(measured_path),shapes=measured_shapes,costs_ns=measured_costs,
                                         interpretation='Measured warm host-memory full-K and maximum-row workload per phase. For grouped g8 phases fewer than K experts may require repair; charging full K is conservative. Includes no DMA, dispatch, interrupts or cold-cache effects.'),
                    scenarios=scenarios,seconds=time.time()-started)
        models.append(result)
        print(json.dumps(dict(model=name,extra_weight_fraction=total_parity/total_data,seconds=time.time()-started)),flush=True)
        doc=dict(study='Persistent background, local integer-parity output repair latency screen',source_sha256=sha256(Path(__file__)),
                 split='external independent routes, no fitting' if args.external else 'old train sample, no new fitting',mac_gweights_s=args.mac_rate,
                 assumptions=dict(background='global at most one erase, fixed absolute intervals, cycle0..15, initial offset=.37*period; duty is global, not per chip',
                                  decision='Only current busy chip and its known busy_until; min(wait_cap, coded) never inspects future starts',
                                  phase='2us host + 250ns per matrix-block command + input transfer + max chip service with current release + output transfer + optional host I64 decode',
                                  service='max(28us*ordinary waves,physical pages*2560ns,useful INT8 weights/R_MAC); no CR',
                                  code='Per expert/source half, pad eight data shard rows to group maximum, sum weights, store two signed-I8 radix256 parts on opposite-half owners slot%8 and (slot+1)%8',
                                  capacity='All parity parts stored persistently; original chosen data rows unchanged. Native W8A8 raw I32 outputs and host I64 decoding required.',
                                  parity_traffic='All healthy original selected shards still execute. Each required parity part is a complete extra GEMV; input is conservatively re-staged, output includes full padded rows.',
                                  suspension='cap is a lower-bound read-availability deadline requested at phase start; original absolute calendar remains fixed. Foreground chip service counted as displaced erase-work debt, not an implemented resume schedule.'),
                 limitations=['Phase-start busy-known variant: a background interval beginning inside a phase is not applied retroactively. Counts flag this boundary uncertainty; no NAND wave abort is claimed.',
                              'Suspension caps ignore actual erase progress loss, resume and future event displacement. They are strong latency bounds, not free maintenance progress.',
                              'The 0/10us suspension caps are hypothetical strong hardware bounds, not demonstrated AiF capabilities. This screen rejects a universal large-tail-gain claim, not all coding opportunities on restricted nonpreemptive devices.',
                              'Only 32 sampled routed-FFN tokens by default; p95/p99 are descriptive with limited tail resolution.',
                              'Current GGUF Q8 group scales and FP32/BF16 arithmetic are not covered by native integer parity exactness.',
                              'Host decode is a parameter sweep, pending native measurements. No physical NAND/PE/driver calibration or energy claim.',
                              'Raw parity algebra is established prior art; comparing with uncoded same-capacity replication and erase suspension is required before novelty or performance claims.'],
                 models=models)
        (args.output/'summary.json').write_text(json.dumps(doc,indent=2)+'\n')


if __name__=='__main__':main()
