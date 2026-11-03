#!/usr/bin/env python3
"""Real GGUF rows -> functional W8 virtual NAND files -> descriptor GEMVs.

Isolated Granite4 Tiny layer0 numerical/address proof. No inference, no device
execution, no native output ABI or unchanged-GGUF quantization claim.
"""
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
OUT=BASE/'shared-materialize'
META=Path('/home/nfs_client/ericyu/expert_flash_experiment/models/granite4tiny-metadata.json')
PLAN=BASE/'geometry-study/granite4tiny.frozen.json'
GGUF_PY=Path('/home/nfs_client/ericyu/expert_flash_experiment/sources/llama.cpp-927dada6c/gguf-py')
sys.path.insert(0,str(GGUF_PY))
import gguf.quants as gguf_quants
from gguf import GGMLQuantizationType
from gguf.quants import dequantize

PAGE=16384
PLANES=4
WAVE=PAGE*PLANES
WORDLINES=192
BLOCK=WAVE*WORDLINES


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(4*1024*1024):h.update(b)
    return h.hexdigest()


def bh(a):return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()
def file_identity(p):
    s=Path(p).stat()
    return dict(size=s.st_size,mtime_ns=s.st_mtime_ns,ctime_ns=s.st_ctime_ns,inode=s.st_ino)


def quantize(values):
    """Same F32 maxabs/127, F32 division, ties-away contract as raw smoke."""
    values=np.asarray(values,dtype=np.float32)
    a=np.max(np.abs(values),axis=-1)
    scale=np.where(a==0,np.float32(1),a/np.float32(127)).astype(np.float32)
    assert np.all(np.isfinite(values)) and np.all(scale>0)
    normalized=(values/np.expand_dims(scale,-1)).astype(np.float32)
    rounded=np.sign(normalized)*np.floor(np.abs(normalized.astype(np.float64))+.5)
    return np.clip(rounded,-127,127).astype(np.int8),scale


def gemv(weights,scales,input_q,input_scale):
    # 1536*127^2=24774144: full dot is safe in I32, no overflow.
    integer=weights.astype(np.int32)@input_q.astype(np.int32)
    assert np.max(np.abs(integer.astype(np.int64)))<=weights.shape[1]*127*127
    factor=(scales*input_scale).astype(np.float32)
    return (integer.astype(np.float32)*factor).astype(np.float32),integer


def swiglu(gu):
    n=len(gu)//2;g=gu[:n];u=gu[n:]
    sig=(np.float32(1)/(np.float32(1)+np.exp(-g).astype(np.float32))).astype(np.float32)
    return ((g*sig).astype(np.float32)*u).astype(np.float32)


class Layout:
    def __init__(self,name,replica,weights,scales_meta,gu_rows,down_rows):
        self.name=name;self.replica=replica;self.weights=weights
        self.scales_meta=scales_meta;self.cursor=[0]*16
        self.allocations=[];self.views=defaultdict(list);self.shared_views={}
        self.directory=OUT/name;self.directory.mkdir(parents=True,exist_ok=True)
        self.files=[]
        for c in range(16):
            p=self.directory/f'chip{c:02d}.bin'
            f=p.open('w+b');self.files.append(f)
        for e in range(64):
            for phase,inventory in [('gu',gu_rows),('down',down_rows)]:
                key=f'routed.{e}.{phase}';start=0
                for c,n in enumerate(inventory[e]):
                    if n:
                        allocation=self.allocate(key,c,start,int(n))
                        self.views[key].append(self.view(allocation,start,int(n)))
                        start+=int(n)
                assert start==weights[key].shape[0]
        # Stored duplicate chunks are256 contiguous rows. The onecopy files
        # store only the original two128-row chunks, each at its own owner.
        for p in range(8):
            first=256*p;key='shared.gu'
            if replica:
                aa=self.allocate(key,p,first,256)
                bb=self.allocate(key,p+8,first,256)
                self.shared_views[str(p)]={
                    '0':[self.view(bb,first,256)],
                    '1':[self.view(aa,first,128),self.view(bb,first+128,128)],
                    '2':[self.view(aa,first,256)]}
            else:
                aa=self.allocate(key,p,first,128)
                bb=self.allocate(key,p+8,first+128,128)
                self.shared_views[str(p)]={'1':[self.view(aa,first,128),self.view(bb,first+128,128)]}
        start=0
        for c in range(16):
            count=128 if c<8 else 64
            a=self.allocate('shared.down',c,start,count)
            self.views['shared.down'].append(self.view(a,start,count));start+=count
        assert start==1536
        self.chip_records=[]
        for c,f in enumerate(self.files):
            size=((self.cursor[c]+WORDLINES-1)//WORDLINES)*BLOCK
            f.truncate(size);f.flush();os.fsync(f.fileno());f.close()
            path=self.directory/f'chip{c:02d}.bin'
            self.chip_records.append(dict(chip=c,path=str(path.relative_to(OUT)),
                useful_bytes=sum(a['weight_bytes'] for a in self.allocations if a['chip']==c),
                used_wave_slots=self.cursor[c],reserved_blocks=size//BLOCK,
                reserved_bytes=size,sha256=sha(path)))
        self.fds=[os.open(self.directory/f'chip{c:02d}.bin',os.O_RDONLY) for c in range(16)]

    def allocate(self,key,chip,start,count):
        w=self.weights[key];cols=w.shape[1];size=count*cols
        assert size%WAVE==0
        waves=size//WAVE;assert waves<=WORDLINES
        at=self.cursor[chip]
        if at%WORDLINES+waves>WORDLINES:at=((at+WORDLINES-1)//WORDLINES)*WORDLINES
        raw=w[start:start+count].tobytes()
        f=self.files[chip];f.seek(at*WAVE);assert f.write(raw)==size
        a=dict(allocation=len(self.allocations),logical_matrix=key,chip=chip,row_start=start,
            rows=count,columns=cols,weight_bytes=size,nand_waves=waves,pages=size//PAGE,
            block=at//WORDLINES,start_wordline=at%WORDLINES,linear_byte_offset=at*WAVE,
            payload_sha256=hashlib.sha256(raw).hexdigest())
        self.allocations.append(a);self.cursor[chip]=at+waves
        return a

    def view(self,a,start,count):
        cols=a['columns'];offset=a['linear_byte_offset']+(start-a['row_start'])*cols
        assert a['row_start']<=start and start+count<=a['row_start']+a['rows']
        assert offset%WAVE==0 and count*cols%WAVE==0
        return dict(logical_matrix=a['logical_matrix'],storage_allocation=a['allocation'],
            chip=a['chip'],logical_row_start=start,rows=count,columns=cols,
            scale_file_byte_offset=self.scales_meta[a['logical_matrix']]['offset']+start*4,
            block=offset//BLOCK,start_wordline=offset%BLOCK//WAVE,start_plane=0,
            byte_in_plane_page=0,linear_byte_offset=offset,
            weight_bytes=count*cols,nand_waves=count*cols//WAVE,pages=count*cols//PAGE)

    def matrix_views(self,key,choices):
        if key!='shared.gu':return self.views[key]
        return [v for p,a in enumerate(choices) for v in self.shared_views[str(p)][str(a)]]

    def read_view(self,v):
        # Compute linear position from the physical geometry fields; do not
        # trust the redundant linear byte field as the execution address.
        at=((v['block']*WORDLINES+v['start_wordline'])*PLANES+v['start_plane'])*PAGE+v['byte_in_plane_page']
        assert at==v['linear_byte_offset']
        assert v['start_wordline']+v['nand_waves']<=WORDLINES
        raw=os.pread(self.fds[v['chip']],v['weight_bytes'],at)
        assert len(raw)==v['weight_bytes']
        return np.frombuffer(raw,dtype=np.int8).reshape(v['rows'],v['columns'])

    def compute(self,key,q,qs,choices,scale_fd):
        size=self.weights[key].shape[0]
        output=np.zeros(size,dtype=np.float32);integer=np.zeros(size,dtype=np.int32)
        coverage=np.zeros(size,dtype=np.uint8);executed=[]
        for v in self.matrix_views(key,choices):
            start,n=v['logical_row_start'],v['rows']
            w=self.read_view(v)
            raw=os.pread(scale_fd,n*4,v['scale_file_byte_offset'])
            assert len(raw)==n*4
            s=np.frombuffer(raw,dtype='<f4')
            y,z=gemv(w,s,q,qs)
            output[start:start+n]=y;integer[start:start+n]=z;coverage[start:start+n]+=1
            executed.append(v)
        assert np.all(coverage==1)
        return output,integer,executed

    def audit_storage(self):
        covers={k:np.zeros(len(w),dtype=np.uint8) for k,w in self.weights.items()}
        perchip=defaultdict(list)
        for a in self.allocations:
            raw=os.pread(self.fds[a['chip']],a['weight_bytes'],a['linear_byte_offset'])
            assert hashlib.sha256(raw).hexdigest()==a['payload_sha256']
            key=a['logical_matrix'];start=a['row_start'];n=a['rows']
            assert raw==self.weights[key][start:start+n].tobytes()
            covers[key][start:start+n]+=1
            perchip[a['chip']].append((a['linear_byte_offset'],a['linear_byte_offset']+a['weight_bytes']))
        for ts in perchip.values():
            ts.sort();assert all(b<=c for (_,b),(c,_) in zip(ts,ts[1:]))
        for key,cover in covers.items():
            assert np.all(cover==(2 if self.replica and key=='shared.gu' else 1))
        return dict(allocation_payloads_bitexact=True,physical_ranges_disjoint=True,
            complete_logical_row_coverage=True,shared_gu_copy_count=2 if self.replica else 1,
            stored_useful_bytes=sum(a['weight_bytes'] for a in self.allocations),
            stored_pages=sum(a['pages'] for a in self.allocations),
            reserved_bytes=sum(c['reserved_bytes'] for c in self.chip_records))

    def write_manifest(self):
        d=dict(name=self.name,paired_shared_replica=self.replica,chips=self.chip_records,
            storage_allocations=self.allocations,execution_views=dict(self.views),
            shared_gu_legal_views=self.shared_views,scales_file='logical_scales.f32',
            address_formula='((block*192+wordline)*4+plane)*16384+byte_in_plane_page',
            address_scope='Virtual linearized NAND plane/WL coordinates, not a verified aif_post native address ABI.',
            storage_audit=self.audit_storage())
        p=OUT/self.name/'manifest.json';p.write_text(json.dumps(d,indent=2)+'\n')
        return d['storage_audit']


def main():
    OUT.mkdir(parents=True,exist_ok=True);began=time.monotonic()
    metadata=json.loads(META.read_text());plan=json.loads(PLAN.read_text())
    original=Path(metadata['model_path']);identity=file_identity(original)
    route_path=Path(plan['training_routes']);route_sha=sha(route_path)
    assert route_sha==plan['training_routes_sha256']
    old_inputs={str(p):sha(p) for p in [META,PLAN,Path(gguf_quants.__file__)]}
    weights={};scales={};source_ranges=[]
    lookup={t['name']:t for t in metadata['tensors']}
    for shared in (False,True):
        prefix='shared' if shared else 'routed'
        for e in ([0] if shared else range(64)):
            matrices=[];matrix_scales=[]
            for proj in ('gate','up','down'):
                name=f'blk.0.ffn_{proj}_'+('shexp' if shared else 'exps')+'.weight'
                t=lookup[name];c,r=t['shape_ggml'][:2]
                count=t['n_bytes']//(1 if shared else 64)
                offset=t['data_offset']+e*count
                packed=np.memmap(original,mode='r',dtype=np.uint8,offset=offset,shape=(count,))
                dtype=getattr(GGMLQuantizationType,t['ggml_type'])
                decoded=dequantize(np.asarray(packed),dtype).reshape(r,c).astype(np.float32)
                q,s=quantize(decoded);assert q.shape==(r,c)
                key='shared' if shared else f'routed.{e}'
                source_ranges.append(dict(tensor=name,expert=e if not shared else None,
                    columns=c,rows=r,source_type=t['ggml_type'],byte_offset=offset,bytes=count,
                    original_packed_sha256=hashlib.sha256(packed).hexdigest(),
                    functional_i8_sha256=bh(q),functional_row_scales_sha256=bh(s)))
                if proj=='down':weights[key+'.down']=q;scales[key+'.down']=s
                else:matrices.append(q);matrix_scales.append(s)
            weights[key+'.gu']=np.concatenate(matrices,axis=0)
            scales[key+'.gu']=np.concatenate(matrix_scales,axis=0)
    quant_seconds=time.monotonic()-began
    assert sum(w.nbytes for w in weights.values())==155713536
    scale_meta={};offset=0
    with (OUT/'logical_scales.f32').open('wb') as f:
        for key in sorted(scales):
            raw=scales[key].astype('<f4').tobytes();f.write(raw)
            scale_meta[key]=dict(offset=offset,rows=len(scales[key]),bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest())
            offset+=len(raw)
    (OUT/'logical_scales.json').write_text(json.dumps(scale_meta,indent=2)+'\n')
    (OUT/'source_ranges.json').write_text(json.dumps(source_ranges,indent=2)+'\n')
    ps={p['name']:p for p in plan['phases'] if p['layer']==0}
    gu_rows=np.asarray(ps['gate_up']['rows'],dtype=np.int64)
    down_rows=np.asarray(ps['down']['rows'],dtype=np.int64)
    layouts=[Layout('onecopy',False,weights,scale_meta,gu_rows,down_rows),
             Layout('paired_shared',True,weights,scale_meta,gu_rows,down_rows)]
    storage={x.name:x.write_manifest() for x in layouts}
    assert storage['paired_shared']['stored_useful_bytes']-storage['onecopy']['stored_useful_bytes']==3145728
    materialize_seconds=time.monotonic()-began-quant_seconds
    with np.load(route_path) as f:
        routes=f['routes'][:32,:64,0].reshape(2048,6).copy()
        pids=f['prompt_ids'][:32].tolist()
    assert all(x.startswith('train_') for x in pids)
    home=gu_rows[:,0]>0;na=home[routes].sum(axis=1)
    indices=[int(np.flatnonzero(na==a)[0]) for a in sorted(set(na.tolist()))]
    rng=np.random.Generator(np.random.PCG64(20261102))
    alphas=(np.arange(1,7,dtype=np.float32)/np.float32(21)).astype(np.float32)
    records=[];all_executed=[];fixture_data={};phase_outputs={}
    scale_fd=os.open(OUT/'logical_scales.f32',os.O_RDONLY)
    variants=[('onecopy',[1]*8),('paired_shared',[0]*8),('paired_shared',[1]*8),
              ('paired_shared',[2]*8),('paired_shared',[0,1,2,0,1,2,0,2])]
    for fi,idx in enumerate(indices):
        route=routes[idx].tolist();x=rng.standard_normal(1536,dtype=np.float32)
        q,qs=quantize(x);fixture_data[f'x{fi}']=x;fixture_data[f'alpha{fi}']=alphas
        keys=[f'routed.{e}' for e in route]+['shared'];reference={}
        for key in keys:
            gu,gi=gemv(weights[key+'.gu'],scales[key+'.gu'],q,qs)
            z=swiglu(gu);zq,zs=quantize(z)
            down,di=gemv(weights[key+'.down'],scales[key+'.down'],zq,zs)
            reference[key]=dict(gu=gu,gu_integer=gi,z=z,zq=zq,zs=zs,down=down,down_integer=di)
        full=np.zeros(1536,dtype=np.float32)
        for alpha,key in zip(alphas,keys[:6]):full=(full+(reference[key]['down']*alpha).astype(np.float32)).astype(np.float32)
        full=(full+reference['shared']['down']).astype(np.float32)
        fixture_data[f'reference_output{fi}']=full
        for vi,(name,choice) in enumerate(variants):
            layout=next(l for l in layouts if l.name==name);executed=[];actual={};compared=0
            for key in keys:
                gu,gi,jobs=layout.compute(key+'.gu',q,qs,choice,scale_fd);executed+=jobs
                assert np.array_equal(gi,reference[key]['gu_integer'])
                assert np.array_equal(gu.view(np.uint32),reference[key]['gu'].view(np.uint32))
                z=swiglu(gu);zq,zs=quantize(z)
                assert np.array_equal(z.view(np.uint32),reference[key]['z'].view(np.uint32))
                assert np.array_equal(zq,reference[key]['zq']) and zs.view(np.uint32)==reference[key]['zs'].view(np.uint32)
                down,di,jobs=layout.compute(key+'.down',zq,zs,choice,scale_fd);executed+=jobs
                assert np.array_equal(di,reference[key]['down_integer'])
                assert np.array_equal(down.view(np.uint32),reference[key]['down'].view(np.uint32))
                compared+=len(gu)+len(down)+len(z)
                actual[key]=down
            mixed=np.zeros(1536,dtype=np.float32)
            for alpha,key in zip(alphas,keys[:6]):mixed=(mixed+(actual[key]*alpha).astype(np.float32)).astype(np.float32)
            mixed=(mixed+actual['shared']).astype(np.float32)
            assert np.array_equal(mixed.view(np.uint32),full.view(np.uint32))
            label=f'fixture{fi}.{name}.a'+''.join(map(str,choice))
            phase_outputs[label]=mixed
            chipwaves=[sum(v['nand_waves'] for v in executed if v['chip']==c) for c in range(16)]
            rec=dict(label=label,route_flat_index=idx,prompt=pids[idx//64],step=idx%64,
                experts=route,n_a=int(na[idx]),shared_tiles_A=choice,
                compared_scalar_outputs=compared+1536,integer_and_f32_bits_equal=True,
                output_sha256=bh(mixed),native_jobs=len(executed),
                executed_weight_bytes=sum(v['weight_bytes'] for v in executed),
                total_chip_waves=sum(chipwaves),per_chip_waves=chipwaves,
                onfi_input_bytes=sum(v['columns'] for v in executed),
                onfi_output_bytes=sum(v['rows']*4 for v in executed),
                controller_command_ns=len(executed)*250,
                activation='GU complete and logically reassembled before whole SwiGLU; quantize entire private down vector once; then down descriptors.',
                mixture='In saved selected expert order, multiply down by fixed alpha in F32, add sequentially in F32, then add shared down.')
            records.append(rec)
            all_executed.append(dict(label=label,jobs=executed))
    os.close(scale_fd)
    np.savez_compressed(OUT/'fixtures.npz',**fixture_data)
    np.savez_compressed(OUT/'actual_outputs.npz',**phase_outputs)
    (OUT/'executed_descriptors.json').write_text(json.dumps(all_executed,indent=2)+'\n')
    for layout in layouts:
        assert all(sha(OUT/r['path'])==r['sha256'] for r in layout.chip_records)
        for fd in layout.fds:os.close(fd)
    assert file_identity(original)==identity and sha(route_path)==route_sha
    assert old_inputs=={p:sha(p) for p in old_inputs}
    out=dict(status='PASS virtual-address functional W8A8 CPU reference',created_utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),numpy_version=np.__version__,input_sha256=old_inputs,
        original_GGUF=dict(path=str(original),previously_recorded_sha256=metadata['sha256'],
            full_payload_rehashed_this_run=False,file_identity_unchanged=True,identity=identity,
            selected_raw_byte_ranges_and_sha256='source_ranges.json',opened_readonly=True),
        routes=dict(path=str(route_path),sha256=route_sha,layer=0,scope='first32trainingprompts x64steps',
            selected_indices=indices,selection='First train route for each observed nA count, not selected by output error or speed.'),
        fixture=dict(input='Explicit fixture, not real captured activation: NumPy PCG64 seed20261102, standard_normal F32 vector1536 per case.',
            alpha='Explicit fixture, not router scores: F32[1,2,3,4,5,6]/21 in original selected expert order.',
            file='fixtures.npz',inputs_fixed_across_all_layouts=True),
        quantization=dict(source='Original GGUF dequantized logical row; each row fixed across layouts.',
            signed_range=[-127,127],scale='F32 maxabs/127, zero-vector scale1',rounding='F32 division; nearest with half ties away; clip',
            activation='Full GU1536 input quantized once and reused; each entire512/1024 SwiGLU private down vector quantized once.',
            output='I32 exact CPU dot (max bound24774144), convert once F32 then multiply F32(rowScale*inputScale).',
            scale_storage='Host logical_scales.f32, recomposed by saved logical row offsets; no scalar-free I8 assumption.'),
        geometry=dict(page_bytes=PAGE,planes=PLANES,wordlines_per_block=WORDLINES,wave_bytes=WAVE,block_bytes=BLOCK,
            physical='Linear virtualchip files map every four16KiB plane pages to a wordline; each native run starts plane0 and never crosses block.',
            routed_gu='Onecopy source g8 rows128; canonical logical order gate512 then up512.',
            routed_down='Frozen source g12 rows128;512columns,12waves/expert.',
            shared_gu='Logical gate1024 then up1024; adjacent128row tiles2p,2p+1 stored at paired chips; replica coalesces256rows.',
            shared_down='Highhalf0:8x128rows/2waves plus8x64rows/1wave;private1024column input.'),
        storage=storage,cases=len(records),fixtures=len(indices),results=records,
        compared_scalar_outputs=sum(r['compared_scalar_outputs'] for r in records),bit_mismatches=0,
        setup_and_audit_wall_seconds=dict(quantization=quant_seconds,materialization_and_full_storage_readback=materialize_seconds,
            total=time.monotonic()-began,note='Software preparation/audit timing only; not NAND or inference latency.'),
        limitations=['Isolated FFN layer with fixed fixture inputs/scores and real old train routes, not end-to-end inference or quality/routing revalidation.',
            'Functional W8 re-quantization changes original GGUF values; both compared layouts use identical fixed W8 rows and host scales.',
            'Uses ordinary dense whole-row GEMV composition and host concatenation; no new MAC or multiple chip contexts. CPU I32 is a reference, not proof nativeAiF returns I32/FP32 raw exact results.',
            'Virtual block/page descriptors and file pread are executable address evidence, not verified native driver/controller ABI or NAND execution.',
            'No hardware latency claim: host activation/scaling/reassembly, controller dispatch, ECC/MAC and DMA need calibration. Address-read order is numeric execution, not concurrent scheduler validation.',
            'Additional replicas occupy3MiB useful bytes, but both layer layouts reserve one12MiB block/chip in this virtual allocator; report both capacity measures.',
            'Shared expert replication/dynamic owner choice has direct prior; this artifact validates mapping and arithmetic only.'])
    (OUT/'summary.json').write_text(json.dumps(out,indent=2)+'\n')
    manifest={str(p.relative_to(ROOT)):sha(p) for p in sorted(OUT.rglob('*')) if p.is_file() and p.name!='completion.json'}
    manifest[str(Path(__file__).relative_to(ROOT))]=sha(__file__)
    (OUT/'completion.json').write_text(json.dumps(dict(status='complete',artifacts_sha256=manifest,
        original_source_inputs_unchanged=True),indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['status','storage','cases','fixtures','compared_scalar_outputs','bit_mismatches','setup_and_audit_wall_seconds']},indent=2))


if __name__=='__main__':main()
