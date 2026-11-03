"""Read bounded Q8_0 physical-row samples; test independent exact byte codecs."""
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import zstandard as zstd

ROOT=Path('/home/ericyu/expert_flash_experiment')
DATA=Path('/home/nfs_client/ericyu/expert_flash_experiment')
DEST=Path(__file__).resolve().parent
compressor=zstd.ZstdCompressor(level=3)
decompressor=zstd.ZstdDecompressor()


def bitshuffle(value):
    return np.packbits(np.unpackbits(value).reshape(-1,8).T,axis=1).reshape(-1)


def unshuffle(value):
    return np.packbits(np.unpackbits(value).reshape(8,-1).T,axis=1).reshape(-1)


def encode(value,mode):
    if mode=='raw':return value
    if mode=='bitshuffle':return bitshuffle(value)
    signed=value.view(np.int8).astype(np.int16)
    zigzag=((signed<<1)^(signed>>7)).astype(np.uint8)
    return bitshuffle(zigzag)


def decode(value,mode):
    if mode=='raw':return value
    out=unshuffle(value)
    if mode=='bitshuffle':return out
    u=out.astype(np.int16)
    return ((u>>1)^-(u&1)).astype(np.int8).view(np.uint8)


def read_rows(mapping,tensor,expert,start,count):
    if not count:return np.empty((0,34),dtype=np.uint8)
    columns=tensor['shape_ggml'][0];assert columns%32==0
    first=tensor['data_offset']+expert*tensor['bytes_per_expert']+start*(columns//32)*34
    length=count*(columns//32)*34
    return np.asarray(mapping[first:first+length]).copy().reshape(-1,34)


records=[];started=time.time();total_payload=0;checks=0
for name in ('deepseek','qwen3','mixtral'):
    metadata_path=DATA/'models'/f'{name}-metadata.json'
    metadata=json.loads(metadata_path.read_text())
    tensors={(t['layer'],t['projection']):t for t in metadata['tensors'] if 'projection' in t}
    frozen=json.loads((ROOT/f'results/research-2026-11-02/projection-study/{name}.frozen_plan.json').read_text())
    mapping=np.memmap(metadata['model_path'],dtype=np.uint8,mode='r')
    layers=frozen['layers'];selected_layers=[layers[0],layers[len(layers)//2],layers[-1]]
    experts=[0,frozen['experts']//2,frozen['experts']-1]
    for phase in frozen['phases']:
        layer=phase['layer']
        if layer not in selected_layers:continue
        rows=np.array(phase['chosen_rows'])
        for expert in experts:
            for chip in (0,8):
                count=int(rows[expert,chip]);start=int(rows[expert,:chip].sum())
                if not count:continue
                if phase['name']=='gate_up':
                    intermediate=tensors[(layer,'gate')]['shape_ggml'][1]
                    gate_count=max(0,min(start+count,intermediate)-start)
                    up_start=max(0,start-intermediate);up_count=count-gate_count
                    raw=np.concatenate((read_rows(mapping,tensors[(layer,'gate')],expert,start,gate_count),
                                        read_rows(mapping,tensors[(layer,'up')],expert,up_start,up_count)))
                else:raw=read_rows(mapping,tensors[(layer,'down')],expert,start,count)
                payload=raw[:,2:].copy().reshape(-1);scales=raw[:,:2].copy().reshape(-1)
                total_payload+=len(payload)
                histogram=np.bincount(payload,minlength=256);prob=histogram[histogram>0]/len(payload)
                entropy=float(-(prob*np.log2(prob)).sum())
                chunks={}
                for tile_size in (16384,65536):
                    sizes={mode:8 for mode in ('raw','bitshuffle','zigzag_bitshuffle')};best=8
                    scale_packed=0;ntiles=0;fallbacks=0
                    for offset in range(0,len(payload),tile_size):
                        tile=payload[offset:offset+tile_size]
                        scale_tile=scales[(offset//32)*2:((offset+len(tile))//32)*2]
                        packed_scales=compressor.compress(scale_tile.tobytes())
                        assert decompressor.decompress(packed_scales)==scale_tile.tobytes()
                        scale_packed+=min(len(scale_tile),len(packed_scales))+5
                        options=[]
                        for mode in sizes:
                            transformed=encode(tile,mode)
                            packed=compressor.compress(transformed.tobytes())
                            restored=decode(np.frombuffer(decompressor.decompress(packed),dtype=np.uint8),mode)
                            np.testing.assert_array_equal(restored,tile);checks+=1
                            stored=min(len(tile),len(packed))+5
                            sizes[mode]+=stored;options.append(stored)
                        winner=min(options);best+=winner
                        fallbacks+=int(winner==len(tile)+5);ntiles+=1
                    # Whole-shard raw fallback carries no compression metadata.
                    best=min(best,len(payload))
                    native_pages=(best+16383)//16384;raw_pages=(len(payload)+16383)//16384
                    native_waves=(native_pages+3)//4;raw_waves=(raw_pages+3)//4
                    chunks[str(tile_size)]=dict(independent_tiles=ntiles,codec_payload_bytes=sizes,best_payload_bytes=best,
                        native_i8_payload_fraction=best/len(payload),scales_compressed_bytes=scale_packed,
                        exact_q8_including_scales_fraction=(best+scale_packed)/(len(payload)+len(scales)),
                        native_payload_pages=raw_pages,compressed_native_pages=native_pages,
                        native_payload_waves=raw_waves,compressed_native_waves=native_waves,
                        ideal_native_nand_gain=1-native_waves/raw_waves,
                        decoder_min_output_gbytes_s=len(payload)/(native_waves*28000),
                        raw_fallback_tiles=fallbacks)
                records.append(dict(model=name,layer=layer,phase=phase['name'],expert=expert,chip=chip,row_start=start,rows=count,
                                    columns=phase['columns'],payload_bytes=len(payload),q8_scale_bytes=len(scales),entropy_bits_per_i8=entropy,
                                    payload_sha256=hashlib.sha256(payload).hexdigest(),chunks=chunks))
    print(name,'records',sum(r['model']==name for r in records),'seconds',time.time()-started,flush=True)

summary=[]
for name in ('deepseek','qwen3','mixtral'):
    selected=[r for r in records if r['model']==name];n=sum(r['payload_bytes'] for r in selected)
    for tile in ('16384','65536'):
        packed=sum(r['chunks'][tile]['best_payload_bytes'] for r in selected)
        old_waves=sum(r['chunks'][tile]['native_payload_waves'] for r in selected)
        new_waves=sum(r['chunks'][tile]['compressed_native_waves'] for r in selected)
        summary.append(dict(model=name,tile_bytes=int(tile),sampled_shards=len(selected),payload_bytes=n,
                            weighted_entropy_bits=float(sum(r['entropy_bits_per_i8']*r['payload_bytes'] for r in selected)/n),
                            best_payload_bytes_fraction=packed/n,summed_sampled_shard_nand_gain=1-new_waves/old_waves,
                            shards_with_fewer_waves=sum(r['chunks'][tile]['compressed_native_waves']<r['chunks'][tile]['native_payload_waves'] for r in selected)))
value=dict(source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),records=records,summary=summary,
           total_payload_bytes=total_payload,exact_tile_roundtrip_checks=checks,seconds=time.time()-started,
           scope='Three Q8 models, first/middle/last routed layer, experts0/E2/E-1, physical chips0/8 of frozen strongest projection. Samples, not full-model traffic estimates.',
           codecs='Zstd3 independent frames, no dictionary or history across tiles/experts; raw/bitshuffle/zigzag-bitshuffle tried. 5 bytes mode+offset per tile, 8 byte stream header, raw fallback.',
           limitations=['Native-I8 payload metric and exact-Q8 including per32 FP16 scales reported separately. No quantization changes or inference run.',
                        'The sampled-shard sum is not a full-model critical path estimate; unsampled chips/layers and channel/control costs remain.',
                        'Packing concatenates independent compressed frames within one selected expert shard; no cross-expert overfetch/dependency granted.',
                        'Decoder threshold assumes ideal overlap, not a hardware implementation or area estimate; MAC/ECC/decoded output bandwidth remain required.'])
(DEST/'weight_probe.json').write_text(json.dumps(value,indent=2)+'\n')
print(json.dumps(value['summary'],indent=2));print('roundtrip_checks',checks,'seconds',time.time()-started)
