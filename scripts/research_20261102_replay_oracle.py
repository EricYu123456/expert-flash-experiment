#!/usr/bin/env python3
"""Ready-cohort raw-page-buffer replay screen, with finite vector SRAM.

SLIM already fetches one NAND weight tile for multiple inputs. Therefore q>1
versus q=1 is prior-mechanism headroom, while every placement comparison uses
the *same* q, MAC rate, timing and SRAM/input-buffer policy on both sides.
Only 64 frozen train prompts x prefix16 steps are opened; no test/inference/VM.
"""
import argparse
from collections import Counter
import csv
import gzip
import json
from pathlib import Path
import time

import numpy as np

import research_20261102_projection_study as projection
from run_corpus_traces import ROOT, MODELS, sha256

DEST = ROOT/'results/research-2026-11-02/round2/replay-oracle'
SRAM = 65536
COMPONENTS = ('service_ns','control_ns','host_ns','input_ns','output_ns','model_ns')


def ceildiv(a,b):
    return (a+b-1)//b


def phase_profile(rows, columns, q, mac_gweights_s):
    rows=np.asarray(rows,dtype=np.int64)
    nbytes=rows*columns
    pages=ceildiv(nbytes,16384)
    waves=ceildiv(pages,4)
    commands=ceildiv(waves,192)
    # A wave can intersect ceil(64KiB / columns)+1 rows. Keep one extra
    # partial-row accumulator; completed output rows stream out every wave.
    accumulator_rows=np.minimum(rows,ceildiv(65536,columns)+1)
    vector_sram=q*(columns+accumulator_rows*4)
    valid=bool(np.all(vector_sram[rows>0]<=SRAM))
    whole_cohort_accumulator=accumulator_rows*4*q
    full=(np.maximum(nbytes,1)-1)//65536
    tail=np.where(nbytes>0,nbytes-full*65536,0)
    tail_pages=ceildiv(tail,16384)
    # Every input replays raw PB bytes through ECC; there is no corrected
    # weight SRAM. ECC bytes count physical pages, MAC counts useful weights.
    full_processing=max(65536/6.4,65536/mac_gweights_s)
    per_vector_processing=full*full_processing+np.maximum(tail_pages*2560,tail/mac_gweights_s)
    per_vector_processing=np.where(rows>0,per_vector_processing,0)
    return dict(rows=rows,columns=columns,pages=pages,waves=waves,commands=commands,
                bytes=nbytes,mac_ns=nbytes/mac_gweights_s,ecc_ns=pages*2560,
                processing_ns=per_vector_processing,mask=(rows>0).astype(np.int64),
                sram_q_vectors_and_wave_accumulators=vector_sram,
                wave_accumulator_bytes=whole_cohort_accumulator,
                feasible=valid,peak_sram_bytes=int(vector_sram[rows>0].max(initial=0)))


def evaluate(counts, request_routes, p, q, batch, timing, common, cached_input_activity=None):
    if not p['feasible']:
        return None
    # These small integer totals are below 2**53. Float64 GEMMs preserve them
    # exactly and avoid NumPy's much slower non-BLAS integer matrix products.
    counts_f=counts.astype(np.float64)
    reads=ceildiv(counts,q).astype(np.float64)
    nand=reads@p['waves'].astype(np.float64)*28000
    ecc=counts_f@p['ecc_ns'].astype(np.float64)
    mac=counts_f@p['mac_ns']
    if timing=='ideal_overlap':
        chip_service=np.maximum(nand,np.maximum(ecc,mac))
    elif timing=='conservative_raw_pb':
        chip_service=nand+counts_f@p['processing_ns']
    else:
        raise ValueError(timing)
    commands=reads@p['commands'].astype(np.float64)
    outputs=(counts_f@p['rows'].astype(np.float64))*4
    if common:
        # The baseline may retain all B common gate/up vectors whenever those
        # vectors plus q wave accumulators fit. Do not force needless restages.
        cache_fits=(batch*p['columns']+p['wave_accumulator_bytes'].max(axis=0))<=SRAM
        used=(p['mask'][request_routes]>0).any(axis=2).sum(axis=1) if cached_input_activity is None else cached_input_activity
        once=used*p['columns']
        restaged=(counts_f@p['mask'].astype(np.float64))*p['columns']
        inputs=np.where(cache_fits[None,:],once,restaged)
        # Primary conservatively assumes no extra controller input cache.
        # If any chip cannot retain the cohort, host restages expert inputs too.
        host_in=np.full(len(counts),batch*p['columns'],dtype=np.int64)
        if not np.all(cache_fits):
            host_in=counts.sum(axis=1)*p['columns']
    else:
        cache_fits=np.zeros(16,dtype=bool)
        inputs=(counts_f@p['mask'].astype(np.float64))*p['columns']
        host_in=counts.sum(axis=1)*p['columns']
    channel_in=(inputs[:,:8]+inputs[:,8:]).max(axis=1)
    channel_out=(outputs[:,:8]+outputs[:,8:]).max(axis=1)
    result=dict(service_ns=np.ceil(chip_service.max(axis=1)).astype(np.int64),
                control_ns=commands.sum(axis=1)*250,
                host_ns=np.full(len(counts),2000,dtype=np.int64),
                input_ns=np.maximum(ceildiv(host_in,8),ceildiv(channel_in,2)),
                output_ns=np.maximum(ceildiv(outputs.sum(axis=1),8),ceildiv(channel_out,2)))
    result['model_ns']=sum(result.values())
    result['weight_read_pages']=reads@p['pages'].sum(axis=1).astype(np.float64)
    result['ecc_replay_pages']=counts_f@p['pages'].sum(axis=1).astype(np.float64)
    result['weight_mac_ops']=counts_f@p['bytes'].sum(axis=1).astype(np.float64)
    result['input_chip_bytes']=inputs.sum(axis=1)
    result['input_host_bytes']=host_in
    result['output_bytes']=outputs.sum(axis=1)
    result['peak_sram_bytes']=p['peak_sram_bytes']
    result['common_input_cohort_cached_chips']=int(cache_fits.sum()) if common else 0
    return result


def scalar_reference(counts, request_routes, p, q, batch, timing, common, mac_gweights_s):
    """Independent selected-expert, chip and physical-wave accounting."""
    service=[];inputs=[];outputs=[];commands=0;read_pages=ecc_pages=mac_ops=0
    cohort_cache=[]
    for chip in range(16):
        nand=ecc=mac=processing=0.;nin=nout=0
        for expert,count in enumerate(counts):
            count=int(count);r=int(p['rows'][expert,chip]);cols=p['columns']
            if not count or not r:continue
            useful=r*cols;npages=(useful+16383)//16384;nwaves=(npages+3)//4;nread=(count+q-1)//q
            nand+=nread*nwaves*28000;ecc+=count*npages*2560;mac+=count*useful/mac_gweights_s
            left=useful
            while left:
                wb=min(65536,left);physical=((wb+16383)//16384)*16384
                processing+=count*max(physical/6.4,wb/mac_gweights_s);left-=wb
            commands+=nread*((nwaves+191)//192);nout+=count*r*4;nin+=count*cols
            read_pages+=nread*npages;ecc_pages+=count*npages;mac_ops+=count*useful
        fit=batch*p['columns']+int(p['wave_accumulator_bytes'][:,chip].max())<=SRAM
        cohort_cache.append(fit)
        if common and fit:
            nin=sum(any(p['rows'][e,chip]>0 for e in route) for route in request_routes)*p['columns']
        service.append(max(nand,ecc,mac) if timing=='ideal_overlap' else nand+processing)
        inputs.append(nin);outputs.append(nout)
    host_input=(batch if common and all(cohort_cache) else int(sum(counts)))*p['columns']
    result=dict(service_ns=int(np.ceil(max(service))),control_ns=commands*250,host_ns=2000,
                input_ns=max((host_input+7)//8,max((inputs[c]+inputs[c+8]+1)//2 for c in range(8))),
                output_ns=max((sum(outputs)+7)//8,max((outputs[c]+outputs[c+8]+1)//2 for c in range(8))))
    result['model_ns']=sum(result.values())
    result.update(weight_read_pages=read_pages,ecc_replay_pages=ecc_pages,weight_mac_ops=mac_ops,
                  input_chip_bytes=sum(inputs),input_host_bytes=host_input,output_bytes=sum(outputs))
    return result


def cohort_data(routes, batch, experts):
    # Adjacent frozen train prompt IDs form immutable independent-request
    # cohorts; positions are synchronous decode indices, never future tokens.
    nrequest,steps,active=routes.shape
    selected=routes.reshape(nrequest//batch,batch,steps,active).transpose(0,2,1,3).reshape(-1,batch,active)
    counts=np.zeros((len(selected),experts),dtype=np.int64)
    np.add.at(counts,(np.repeat(np.arange(len(selected)),batch*active),selected.reshape(-1)),1)
    assert np.all(counts.sum(axis=1)==batch*active)
    return counts,selected


def analyze(name,args):
    started=time.time()
    frozen_path=ROOT/'results/research-2026-11-02/projection-study'/f'{name}.frozen_plan.json'
    frozen=json.loads(frozen_path.read_text());legacy_path=Path(frozen['legacy_path']);legacy=json.loads(legacy_path.read_text())
    assert sha256(legacy_path)==frozen['legacy_sha256']
    train_path=Path(frozen['train_path']);assert train_path.name=='train_routes.npz' and sha256(train_path)==frozen['train_sha256']
    with np.load(train_path) as z:
        routes=z['routes'][:64,:16];ids=z['prompt_ids'][:64].tolist();assert ids==frozen['train_ids'][:64]
        assert all(x.startswith('train_') for x in ids)
    cells={};audits=0
    for batch in (1,4,16):
        for q in (1,2,4):
            for rate in (1.6,3.2,6.4):
                for timing in ('ideal_overlap','conservative_raw_pb'):
                    key=f'B{batch}_q{q}_mac{rate}_{timing}'
                    cells[key]=dict(batch=batch,q=q,mac_gweights_s=rate,timing=timing,
                                    policies={label:Counter() for label in ('equal_g16','projection','strong_prior','best_fixed_group','best_fixed_phase_candidate')},
                                    chosen_group_counts=Counter(),chosen_candidate_counts=Counter(),per_layer_phase=[])
    for li,layer in enumerate(frozen['layers']):
        for phase in frozen['phases'][2*li:2*li+2]:
            rows=phase['total_rows'];columns=phase['columns'];common=phase['common']
            candidates=[('equal_g16',np.asarray(phase['baseline_rows'])),('projection',np.asarray(phase['chosen_rows']))]
            for g in (1,2,4,8,16):
                key=f'group_only_learned__g{g}__w4'
                candidates.append((f'g{g}',projection.equal_rows(rows,frozen['experts'],g,legacy['mappings'][key][str(layer)])))
            assert all(np.all(row.sum(axis=1)==rows) for _,row in candidates)
            datasets={batch:cohort_data(routes[:,:,li,:],batch,frozen['experts']) for batch in (1,4,16)}
            input_activity={(batch,label):(row[selected]>0).any(axis=2).sum(axis=1)
                            for batch,(_,selected) in datasets.items() for label,row in candidates} if common else {}
            for q in (1,2,4):
                for rate in (1.6,3.2,6.4):
                    profiled=[(label,phase_profile(row,columns,q,rate)) for label,row in candidates]
                    for batch,(counts,selected) in datasets.items():
                        for timing in ('ideal_overlap','conservative_raw_pb'):
                            key=f'B{batch}_q{q}_mac{rate}_{timing}';cell=cells[key];values={}
                            for label,p in profiled:
                                value=evaluate(counts,selected,p,q,batch,timing,common,input_activity.get((batch,label)))
                                if value is None:continue
                                means={k:float(v.mean()) for k,v in value.items() if isinstance(v,np.ndarray)}
                                means['peak_sram_bytes']=value['peak_sram_bytes'];means['common_input_cohort_cached_chips']=value['common_input_cohort_cached_chips']
                                values[label]=means
                                if li==0 and label in ('equal_g16','projection','g1'):
                                    ref=scalar_reference(counts[0],selected[0],p,q,batch,timing,common,rate)
                                    for component,number in ref.items():assert abs(float(value[component][0])-number)<1.01,(name,key,label,component,value[component][0],number)
                                    audits+=1
                            fixed=min((label for label in values if label.startswith('g')),key=lambda label:values[label]['model_ns'])
                            winner=min(values,key=lambda label:values[label]['model_ns'])
                            strong_prior=min(('equal_g16','projection'),key=lambda label:values[label]['model_ns'])
                            for policy,label in (('equal_g16','equal_g16'),('projection','projection'),('strong_prior',strong_prior),('best_fixed_group',fixed),('best_fixed_phase_candidate',winner)):
                                previous_peak=cell['policies'][policy].get('peak_sram_bytes',0)
                                cell['policies'][policy].update({k:v for k,v in values[label].items() if k!='peak_sram_bytes'})
                                cell['policies'][policy]['peak_sram_bytes']=max(previous_peak,values[label]['peak_sram_bytes'])
                            cell['chosen_group_counts'][fixed]+=1;cell['chosen_candidate_counts'][winner]+=1
                            cell['per_layer_phase'].append(dict(layer=layer,phase=phase['name'],best_group=fixed,best_candidate=winner,
                                                                candidate_means_ns=values, cohorts=len(counts)))
        if (li+1)%8==0:print(f'{name}: {li+1}/{len(frozen["layers"])} layers; {time.time()-started:.1f}s',flush=True)
    summaries=[]
    for key,cell in cells.items():
        cell['policies']={k:dict(v) for k,v in cell['policies'].items()}
        cell['chosen_group_counts']=dict(cell['chosen_group_counts']);cell['chosen_candidate_counts']=dict(cell['chosen_candidate_counts'])
        base=cell['policies']['equal_g16']['model_ns'];project=cell['policies']['projection']['model_ns'];best=cell['policies']['best_fixed_phase_candidate']['model_ns']
        cell['placement_gain_vs_same_q_g16']=1-best/base
        cell['additional_placement_gain_vs_same_q_projection']=1-best/project
        cell['additional_placement_gain_vs_same_q_strong_prior']=1-best/cell['policies']['strong_prior']['model_ns']
        q1=cells[f'B{cell["batch"]}_q1_mac{cell["mac_gweights_s"]}_{cell["timing"]}']['policies']['equal_g16']['model_ns']
        cell['prior_replay_gain_vs_q1_g16']=1-base/q1
        summaries.append(dict(model=name,key=key,**{k:cell[k] for k in ('batch','q','mac_gweights_s','timing','placement_gain_vs_same_q_g16','additional_placement_gain_vs_same_q_projection','additional_placement_gain_vs_same_q_strong_prior','prior_replay_gain_vs_q1_g16')},
                              g16_ms_per_cohort=base/1e6,projection_ms_per_cohort=project/1e6,best_ms_per_cohort=best/1e6))
    best=max(summaries,key=lambda c:c['additional_placement_gain_vs_same_q_strong_prior'])
    print(json.dumps(dict(model=name,best_additional_vs_projection=best,scalar_cases=audits,seconds=time.time()-started)),flush=True)
    detail_path=args.output/f'{name}.details.json.gz'
    detail={key:cell.pop('per_layer_phase') for key,cell in cells.items()}
    with gzip.GzipFile(filename=str(detail_path),mode='wb',mtime=0) as stream:
        stream.write(json.dumps(detail,separators=(',',':')).encode())
    return dict(model=name,source_projection_sha256=sha256(frozen_path),source_legacy_sha256=sha256(legacy_path),train_cache_sha256=sha256(train_path),
                train_prompt_ids=ids,train_shape=list(routes.shape),scalar_cases=audits,summary=summaries,cells=cells,
                details_path=str(detail_path),details_sha256=sha256(detail_path),seconds=time.time()-started)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--models',nargs='+',choices=MODELS,default=['granite','deepseek','qwen3'])
    parser.add_argument('--output',type=Path,default=DEST)
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=True);results=[]
    for name in args.models:
        results.append(analyze(name,args))
        doc=dict(study='Finite vector SRAM, raw PB replay, all-ready cohort placement oracle',source_sha256=sha256(Path(__file__)),
                 dependencies_sha256={p:sha256(ROOT/'scripts'/p) for p in ('research_20261102_projection_study.py','run_corpus_traces.py')},
                 prior=dict(url='https://arxiv.org/html/2507.09201v1',section='5.4, output-stationary PEs and input SRAM',
                            attribution='Fetching weights once for B inputs is existing SLIM mechanism, not placement novelty.'),
                 hardware=dict(chips=16,planes=4,page_bytes=16384,wave_ns=28000,cr_read=False,vector_sram_bytes_per_chip=65536,
                               sram_formula='q*(input_columns + 4*min(shard_rows,ceil(65536/input_columns)+1))',
                               raw_pb_bytes_per_chip=65536,corrected_weight_sram_bytes=0,ecc_gbytes_s=6.4,mac_gweights_s=[1.6,3.2,6.4],
                               pcie_bytes_ns=8,channel_bytes_ns=2,block_command_ns=250,host_phase_ns=2000),
                 formulas=dict(read_repetitions='R_e=ceil(multiplicity_e/q)',
                               ideal='max_chip(max(28000*sum_e R_e*waves_ec, sum_e count_e*physical_pages_ec*2560, sum_e count_e*useful_weights_ec/R_MAC))',
                               conservative='max_chip(sum_e R_e*waves_ec*28000 + sum_e count_e*sum_wave max(physical_wave_bytes/6.4,useful_wave_weights/R_MAC))',
                               phase_cost='chip_service + 250*sum matrix-block commands + 2000 + max(host_input/8,max_channel_input/2) + max(host_output/8,max_channel_output/2)',
                               dependency='cohort gate_up phase -> host activation/reassembly -> cohort down phase; full two-phase barrier retained',
                               input='If all B gate/up inputs + q wave accumulators fit in 64KiB, cache once per input/chip; otherwise re-stage per expert-vector. Down always stages each expert-vector once per owning chip. No extra controller input cache assumed.'),
                 limitations=['All requests in each fixed cohort are already ready. No arrival delay, next-token overlap, request latency/SLO or end-to-end tokens/s claim.',
                              'MAC rate is an independent exposed assumption; AiF paper does not provide lane count. None of the MAC/SRAM/replay numbers is physical calibration.',
                              'Raw PB must support repeated ECC reads and q vector/accumulator contexts; this is a conditional hardware variant, not proven AiF capability.',
                              'Repeated ECC per vector is a specified conservative raw-PB variant, not universal: streaming a corrected weight through a small holding register for q MAC uses could amortize ECC without a full corrected-wave SRAM, requiring a different dataflow/backpressure design.',
                              'Ideal overlap relaxes raw-PB refill/replay conflicts. Conservative raw-PB formula serializes read then repeated ECC/MAC per wave.',
                              'Commands and aggregate input restaging/output/link costs are charged, but this is a resource-cost oracle, not a constructed limited-SRAM multi-chip/queue schedule. Positive gains require later event scheduling verification.',
                              'Wave accumulators stream completed FP32 rows out. ECC handles full physical pages for every vector, MAC counts useful INT8 weights; no corrected-weight SRAM is granted.',
                              'SRAM formula excludes metadata and replay-control state, so fitting is an upper bound on capacity feasibility.',
                              'Best fixed phase candidate chooses one immutable layout per layer/phase for each operating point using only these train cohorts. Different operating points are alternative stored layouts, never free dynamic remapping.',
                              'Strong prior selects the better of existing equal-g16 and frozen projection placement independently per layer/phase, preventing recovery from an old projection regression from being counted as new placement.',
                              'Host activation/reassembly/combine compute unmeasured, as before. Shared experts, dense layers, attention and FTL excluded.'],
                 test_routes_opened=False,models=results)
        (args.output/'summary.json').write_text(json.dumps(doc,indent=2)+'\n')
    rows=[r for m in results for r in m['summary']]
    with (args.output/'summary.csv').open('w') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
