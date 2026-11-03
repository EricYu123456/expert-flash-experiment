#!/usr/bin/env python3
"""Freeze row-scale/address ledgers and a bounded host gather fallback probe."""
import argparse
from datetime import datetime,timezone
import hashlib
import json
from pathlib import Path
import subprocess

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
DEST=BASE/'host-row-gather'
METADATA=Path('/home/nfs_client/ericyu/expert_flash_experiment/models')


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def write(name,value):
    p=DEST/name;p.write_text(json.dumps(value,indent=2)+'\n');return p
def ceildiv(a,b):return (a+b-1)//b


def prepare():
    DEST.mkdir(parents=True,exist_ok=True)
    configs={
      'granite3':('precision-scope/ibm-granite__granite-3.1-1b-a400m-instruct.config.json',1024,512,32,8,24,0,'granite'),
      'olmoe':('precision-scope/allenai__OLMoE-1B-7B-0924.config.json',2048,1024,64,8,16,0,'olmoe'),
      'deepseek':('precision-scope/deepseek-ai__DeepSeek-V2-Lite-Chat.config.json',2048,1408,64,6,26,2816,'deepseek'),
      'qwen3':('precision-scope/Qwen__Qwen3-30B-A3B-Instruct-2507.config.json',2048,768,128,8,48,0,'qwen3'),
      'mixtral':('precision-scope/mistralai__Mixtral-8x7B-Instruct-v0.1.config.json',4096,14336,8,2,32,0,'mixtral'),
      'granite4tiny':('precision-scope/ibm-granite__granite-4.0-h-tiny.config.json',1536,512,64,6,40,1024,'granite4tiny'),
      'qwen38':('qwen38-placement/official_config.json',2560,640,512,10,48,640,None)}
    inputs={};records=[]
    for name,(file,h,i,e,k,l,si,meta_name) in configs.items():
        p=BASE/file;inputs[str(p)]=sha(p);raw=json.loads(p.read_text());cfg=raw.get('text_config',raw)
        assert cfg['hidden_size']==h and cfg['num_hidden_layers']==(l+1 if name=='deepseek' else l)
        assert cfg.get('moe_intermediate_size',cfg.get('intermediate_size'))==i
        assert cfg.get('num_experts',cfg.get('num_local_experts',cfg.get('n_routed_experts')))==e
        assert cfg['num_experts_per_tok']==k
        receipt=[]
        if meta_name:
            mp=METADATA/f'{meta_name}-metadata.json';inputs[str(mp)]=sha(mp);m=json.loads(mp.read_text())
            routed=[t for t in m['tensors'] if t.get('projection') in ('gate','up','down')]
            layers=set(t['layer'] for t in routed);assert len(layers)==l
            for t in routed:
                expected=[i,h,e] if t['projection']=='down' else [h,i,e]
                assert t['shape_ggml']==expected
            shared=[t for t in m['tensors'] if t.get('shared_projection') in ('gate','up','down')]
            assert len(shared)==(3*l if si else 0)
            for t in shared:assert t['shape_ggml']==([si,h] if t['shared_projection']=='down' else [h,si])
            receipt=[dict(name=t['name'],shape_ggml=t['shape_ggml']) for t in routed[:3]+shared[:3]]
        else:assert cfg['shared_expert_intermediate_size']==si
        row=2*i+h;shared_row=2*si+h if si else 0;shared_objects=int(si>0)
        resident_routed_rows=l*e*row;resident_shared_rows=l*shared_row
        token_routed_rows=l*k*row;token_shared_rows=l*shared_row
        resident_objects=l*(e+shared_objects);token_objects=l*(k+shared_objects)
        scale_page_per_expert=ceildiv(8*i,16384)+ceildiv(4*h,16384)
        shared_pages=(ceildiv(8*si,16384)+ceildiv(4*h,16384)) if si else 0
        rec=dict(model=name,hidden=h,routed_intermediate=i,routed_experts=e,top_k=k,moe_layers=l,
                 shared_combined_intermediate=si,shared_matrix_objects_per_layer=shared_objects,
                 rows_per_routed_expert=row,rows_per_shared_ffn=shared_row,
                 resident_row_scales=dict(routed_scalars=resident_routed_rows,shared_scalars=resident_shared_rows,
                   routed_FP32_bytes=4*resident_routed_rows,shared_FP32_bytes=4*resident_shared_rows,
                   total_FP32_bytes=4*(resident_routed_rows+resident_shared_rows)),
                 per_token=dict(routed_row_scales=token_routed_rows,shared_row_scales=token_shared_rows,
                   scale_read_bytes_if_each_selected_scale_used_once=4*(token_routed_rows+token_shared_rows),
                   useful_opaque_4B_output_bytes=4*(token_routed_rows+token_shared_rows),
                   dense_pooled_metadata_pages_lower_bound=ceildiv(4*(token_routed_rows+token_shared_rows),16384),
                   projection_page_aligned_metadata_fetch_pages=l*(k*scale_page_per_expert+shared_pages)),
                 address_ledger=dict(logical_fused_projection_handles_resident=2*resident_objects,
                   logical_fused_projection_handles_per_token=2*token_objects,
                   g16_physical_row_band_entries_resident=32*resident_objects,
                   g16_physical_row_band_entries_per_token=32*token_objects,
                   g16_resident_band_descriptor_bytes_at32B=32*32*resident_objects,
                   g16_per_token_band_descriptor_bytes_at32B=32*32*token_objects,
                   g16_per_token_gather_descriptor_bytes_at16B=16*32*token_objects,
                   scale_per_row_address_entries_required=0),shape_receipt=receipt)
        if name=='deepseek':
            dense_rows=2*cfg['intermediate_size']+h
            rec['excluded_initial_dense_layer']=dict(layers=1,intermediate=cfg['intermediate_size'],
                row_scales=dense_rows,FP32_scale_bytes=4*dense_rows,useful_4B_output_bytes=4*dense_rows)
        if name=='qwen38':rec['excluded_MTP_layers']=cfg.get('mtp_num_hidden_layers',0)
        records.append(rec)
    ledger=write('ledger.json',dict(status='derived nominal per-output-row W8 scale/address ledger',records=records,
      assumptions=['One FP32 weight scale per logical output row; nominal rowwise-W8 contract, not actual GGUF/NVFP4/groupwise metadata or a demonstrated AiF ABI.',
       'Scales live in ordinary existing host DRAM and remain in original row order. INT8 coefficient row bytes and wave geometry are unchanged.',
       'Per-token scale read is a logical payload if host rescale/dequantization uses each selected row scale once. Opaque F32 copy does not itself require those reads.',
       'Two logical projections mean fused vertical gate/up plus down. Unfused gate/up/down uses three handles; physical row scale count is unchanged.',
       '32B band entry is an explicit host-format sizing assumption: uint64 flash address plus uint32 row_start,rows,columns,pages,chip,flags. No published AiF descriptor size is asserted.',
       '16B gather entry is the actual benchmark struct: uint32 chip,source_offset,destination_offset,rows. Per-row pointers are unnecessary with dense scale vectors and row bands.',
       'Flash-resident metadata is a separate ordinary read, not free MAC coefficient access. Page-aligned per-projection fetch is a stated alternative; pooled ceil is only a payload lower bound and ignores layer boundaries/route locality.',
       'Routed/shared FFN only; routers, biases, activation input scales, attention, KV, dense initial layer and auxiliary MTP are excluded or separately identified. No alpha/output merging assumed.']))
    workloads=[]
    def add(name,model,layers,all_layer_items,provenance):
        source=[0]*16;bands=[];cursor=0
        for items in all_layer_items:
            for phase in ('gate_up','down'):
                for item in items:
                    jobs=item[phase];r=max(j['row_start']+j['rows'] for j in jobs);cover=np.zeros(r,dtype=np.uint8)
                    for j in jobs:
                        lo=j['row_start'];n=j['rows'];cover[lo:lo+n]+=1
                        assert j['column_start']==0
                        chip=j['chip'];emitted=ceildiv(n*j['columns'],16384)*16384//j['columns'];assert emitted>=n
                        bands.append((chip,source[chip],cursor+lo,n));source[chip]+=emitted
                    assert np.all(cover==1);cursor+=r
        workloads.append(dict(name=name,model=model,layers=layers,output_rows=cursor,source_rows=source,bands=bands,provenance=provenance))
    for case in ('group_g16','overlap_best','whole_row'):
        items=[]
        for layer in range(48):
            p=BASE/f'qwen38-placement/parameter-sensitivity/L{layer}.{case}.command_250.trace.json';inputs[str(p)]=sha(p)
            d=json.loads(p.read_text());assert d['layer']==layer and d['case']==case
            items.append(d['items'])
        add('qwen38_'+case,'qwen38',48,items,'actual_frozen48_first_saved_route')
    gp=BASE/'geometry-study/granite4tiny.frozen.json';inputs[str(gp)]=sha(gp);g=json.loads(gp.read_text())
    rp=Path(g['training_routes']);inputs[str(rp)]=sha(rp);assert inputs[str(rp)]==g['training_routes_sha256']
    with np.load(rp) as z:routes=z['routes'][0,0].copy()
    assert routes.shape==(40,6)
    cp=BASE/'whole-row-compiler/certificates.json';inputs[str(cp)]=sha(cp);cert=json.loads(cp.read_text())
    balanced={r['phase']:r['balanced_wave_profile']['rows'] for r in cert['records'] if r['model']=='granite4tiny'}
    def jobs(rows,columns,rotation=0):
        out=[];cursor=0
        for chip,n in enumerate(rows):
            if n:out.append(dict(chip=(chip+rotation)%16,row_start=cursor,rows=int(n),column_start=0,columns=columns))
            cursor+=int(n)
        return out
    for label in ('equal_g16','actual_phase_group','actual_geometry_chosen','generic_balanced_wave'):
        layer_items=[]
        for layer in range(40):
            phases={p['name']:p for p in g['phases'] if p['layer']==layer};items=[]
            for expert in routes[layer]:
                item={}
                for phase in ('gate_up','down'):
                    p=phases[phase]
                    if label=='equal_g16':rows=[p['total_rows']//16]*16
                    elif label=='generic_balanced_wave':rows=balanced[phase]
                    else:rows=p['group_rows' if label=='actual_phase_group' else 'rows'][expert]
                    item[phase]=jobs(rows,p['columns'],int(expert)%16 if label=='generic_balanced_wave' else 0)
                items.append(item)
            # The sealed G4 routed study excludes shared. Attach the same
            # verified-shape equal16 shared row bands to every gather control.
            items.append(dict(gate_up=jobs([128]*16,1536),down=jobs([96]*16,1024)))
            layer_items.append(items)
        add('granite4tiny_'+label,'granite4tiny',40,layer_items,'frozen_routed40_train0_plus_identical_shared_equal16')
    path=DEST/'workloads.txt'
    with path.open('w') as f:
        f.write(str(len(workloads))+'\n')
        for w in workloads:
            f.write(f"{w['name']} {w['model']} {w['layers']} {w['output_rows']} {len(w['bands'])} {w['provenance']}\n")
            f.write(' '.join(map(str,w['source_rows']))+'\n')
            for b in w['bands']:f.write(' '.join(map(str,b))+'\n')
    write('workload_inventory.json',[{k:v for k,v in w.items() if k!='bands'}|dict(bands=len(w['bands']),useful_output_bytes=4*w['output_rows'],staging_bytes_including_fringe=4*sum(w['source_rows'])) for w in workloads])
    cpp=Path(__file__).with_suffix('.cpp');compiler=subprocess.check_output(['g++','--version'],text=True)
    sysinfo=dict(lscpu=subprocess.check_output(['lscpu'],text=True),
                 loadavg=Path('/proc/loadavg').read_text().strip(),
                 pinned_cpu=31,pinned_cpu_L3_size=Path('/sys/devices/system/cpu/cpu31/cache/index3/size').read_text().strip(),
                 pinned_cpu_L3_sharers=Path('/sys/devices/system/cpu/cpu31/cache/index3/shared_cpu_list').read_text().strip())
    inputs.update({str(p):sha(p) for p in (Path(__file__),cpp,ledger,path,DEST/'workload_inventory.json')})
    protocol=write('protocol.frozen.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),source_inputs_sha256=inputs,
       host=sysinfo,compiler=compiler,build_command=['g++','-O3','-march=native','-std=c++17',str(cpp),'-o',str(DEST/'host_row_gather')],
       benchmark_arguments=dict(cpu=31,samples=256,stream_target_MiB=128,threads=1),
       controls=['Seven real-shape gather workloads. Q38 uses actual first saved route items in all48 frozen layers for equal16, width13 overlap and balanced unequal rows. No new routes or refit.',
        'G4 uses all40 frozen row allocations and training request0/step0 routes; common equal16 shared I1024 geometry appended identically because original routed-only study excluded shared.',
        '16 separately64B-aligned chip staging buffers; each native job reserves all complete outputs in requested16KiB pages. Copy only useful complete row bands to contiguous logical per-expert/projection output views.',
        'Primary opaque4B/F32 copies use memcpy, no column reductions. Conditional I32-to-F32 row-scale multiplication is separately labeled and does not assert AiF rawI32 support.',
        'Warm: one full-FFN token. Stream: rotating replicas totaling>=128MiB, exceeding pinned CPU CCD32MiB L3, with first touch after affinity pinning. No cache-flush/DMA or overlap with native computation.',
        'Bitwise every copied scalar verified outside timing; poisoned fringe ignored. Per-operation256 samples, median/p99 raw durations retained. One thread; no unsupported concurrency or edge calibration.'],
       limits=['EPYC9354 host software fallback timing only, not AiF/edge/device/DMA timing. Background contention and CPU frequency uncontrolled and reported distributions are descriptive.',
        'Optional fallback: a backend documenting direct DMA/scatter into logical row offsets could eliminate this copy. No such interface is assumed verified here.',
        'I32 rescale uses a selected dense logical scale vector per replica, not a full360MiB model-scale random-access benchmark; scale residency is independently costed in ledger.',
        'Full-FFN token copy loops sum work across layers/phases after bytes are available, without measuring runtime synchronization, kernel activation, dispatch or DMA.']))
    print('protocol_frozen',sha(protocol),'workloads',len(workloads),flush=True)


def run():
    p=DEST/'protocol.frozen.json';protocol=json.loads(p.read_text());assert all(sha(f)==h for f,h in protocol['source_inputs_sha256'].items())
    subprocess.run(protocol['build_command'],check=True)
    binary=DEST/'host_row_gather';receipt=write('build_receipt.json',dict(protocol_sha256=sha(p),binary_sha256=sha(binary),built_utc=datetime.now(timezone.utc).isoformat()))
    a=protocol['benchmark_arguments'];cmd=[str(binary),str(DEST/'workloads.txt'),str(DEST/'measurements.json'),str(a['cpu']),str(a['samples']),str(a['stream_target_MiB'])]
    subprocess.run(cmd,check=True)
    assert all(sha(f)==h for f,h in protocol['source_inputs_sha256'].items())
    measurements=json.loads((DEST/'measurements.json').read_text());summary=[]
    for r in measurements['records']:
        summary.append({k:v for k,v in r.items() if k!='durations_ns'})
    write('summary.json',dict(protocol_sha256=sha(p),build_receipt_sha256=sha(receipt),measurement_sha256=sha(DEST/'measurements.json'),records=summary))
    print(json.dumps(summary,indent=2))


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--prepare',action='store_true');ap.add_argument('--run',action='store_true');a=ap.parse_args()
    if a.prepare:prepare()
    if a.run:run()
