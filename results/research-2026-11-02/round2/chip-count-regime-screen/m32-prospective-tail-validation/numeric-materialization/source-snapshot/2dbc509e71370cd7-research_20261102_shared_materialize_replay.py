#!/usr/bin/env python3
"""Fresh-process replay using only saved virtualchip manifests and fixtures.

Does not import the materializer, load canonical in-memory weights, or invoke
llama. Uses I64 CPU dot as an independent check of materializer's I32 dot.
"""
from collections import defaultdict
import hashlib
import json
import os
from pathlib import Path
import re

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/'results/research-2026-11-02/round2/shared-materialize'


def sha(p):
    h=hashlib.sha256()
    with Path(p).open('rb') as f:
        while b:=f.read(4194304):h.update(b)
    return h.hexdigest()


def qvector(x):
    m=np.max(np.abs(x));s=np.float32(1) if m==0 else np.float32(m/np.float32(127))
    normalized=(x/s).astype(np.float32)
    q=np.copysign(np.floor(np.abs(normalized.astype(np.float64))+.5),normalized)
    return np.clip(q,-127,127).astype(np.int8),s


def activate(y):
    half=len(y)//2
    gate=y[:half];up=y[half:]
    sigmoid=np.reciprocal((np.float32(1)+np.exp(-gate).astype(np.float32)).astype(np.float32))
    return ((gate*sigmoid).astype(np.float32)*up).astype(np.float32)


def main():
    summary=json.loads((OUT/'summary.json').read_text())
    smeta=json.loads((OUT/'logical_scales.json').read_text())
    files=[OUT/'summary.json',OUT/'fixtures.npz',OUT/'actual_outputs.npz',OUT/'logical_scales.f32',OUT/'logical_scales.json']
    manifests={name:json.loads((OUT/name/'manifest.json').read_text()) for name in ('onecopy','paired_shared')}
    fds={}
    for name,man in manifests.items():
        files.append(OUT/name/'manifest.json')
        for chip in man['chips']:
            p=OUT/chip['path'];assert sha(p)==chip['sha256'];files.append(p)
            fds[name,chip['chip']]=os.open(p,os.O_RDONLY)
    hashes={str(p):sha(p) for p in files}
    scales=np.fromfile(OUT/'logical_scales.f32',dtype='<f4')
    fixtures=np.load(OUT/'fixtures.npz');outputs=np.load(OUT/'actual_outputs.npz')

    def decode_views(name,key,choices):
        man=manifests[name]
        if key=='shared.gu':
            return [v for p,a in enumerate(choices) for v in man['shared_gu_legal_views'][str(p)][str(a)]]
        return man['execution_views'][key]

    def matvec(name,key,x,choices):
        q,s=qvector(x);r=smeta[key]['rows'];out=np.empty(r,dtype=np.float32)
        integer=np.empty(r,dtype=np.int64);coverage=np.zeros(r,dtype=np.uint8)
        for v in decode_views(name,key,choices):
            row,n,c=v['logical_row_start'],v['rows'],v['columns']
            # Read each16KiB plane page by separately derived coordinates.
            # This differs from the materializer's contiguous-range pread.
            packed=[]
            for page in range(v['pages']):
                wl=v['start_wordline']+page//4;plane=page%4
                assert wl<192
                address=((v['block']*192+wl)*4+plane)*16384
                b=os.pread(fds[name,v['chip']],16384,address)
                assert len(b)==16384;packed.append(b)
            w=np.frombuffer(b''.join(packed),dtype=np.int8).reshape(n,c)
            z=w.astype(np.int64)@q.astype(np.int64)
            so=v['scale_file_byte_offset']//4
            assert so==smeta[key]['offset']//4+row
            factor=(scales[so:so+n]*s).astype(np.float32)
            out[row:row+n]=(z.astype(np.float32)*factor).astype(np.float32)
            integer[row:row+n]=z;coverage[row:row+n]+=1
        assert np.all(coverage==1)
        return out,integer

    records=[];phase_count=scalar_count=0
    for rec in summary['results']:
        match=re.fullmatch(r'fixture(\d+)\.(onecopy|paired_shared)\.a([012]{8})',rec['label'])
        assert match
        fi,name,chosen=int(match[1]),match[2],list(map(int,match[3]))
        x=fixtures[f'x{fi}'];alpha=fixtures[f'alpha{fi}']
        keys=[f'routed.{e}' for e in rec['experts']]+['shared']
        mixed=np.zeros(1536,dtype=np.float32)
        for j,key in enumerate(keys):
            y,yi=matvec(name,key+'.gu',x,chosen)
            reference,ri=matvec('onecopy',key+'.gu',x,[1]*8)
            assert np.array_equal(yi,ri) and np.array_equal(y.view(np.uint32),reference.view(np.uint32))
            z=activate(y);rz=activate(reference)
            assert np.array_equal(z.view(np.uint32),rz.view(np.uint32))
            down,di=matvec(name,key+'.down',z,chosen)
            rd,rdi=matvec('onecopy',key+'.down',rz,[1]*8)
            assert np.array_equal(di,rdi) and np.array_equal(down.view(np.uint32),rd.view(np.uint32))
            phase_count+=2;scalar_count+=len(y)+len(z)+len(down)
            mixed=(mixed+(down*alpha[j]).astype(np.float32)).astype(np.float32) if j<6 else (mixed+down).astype(np.float32)
        assert np.array_equal(mixed.view(np.uint32),fixtures[f'reference_output{fi}'].view(np.uint32))
        assert np.array_equal(mixed.view(np.uint32),outputs[rec['label']].view(np.uint32))
        scalar_count+=len(mixed)
        assert hashlib.sha256(mixed.tobytes()).hexdigest()==rec['output_sha256']
        records.append(dict(label=rec['label'],integer_and_f32_bits_exact=True))
    for fd in fds.values():os.close(fd)
    assert hashes=={str(p):sha(p) for p in files}
    out=dict(status='PASS',source_sha256=sha(__file__),fresh_process=True,
        author_materializer_imported=False,canonical_memory_weights_used=False,
        saved_source_files_unchanged=True,input_sha256=hashes,
        accumulation='Independent I64 CPU GEMV; author used safe I32. Same F32 final scaling contract.',
        addressing='Every16KiB plane page separately pread from derived block/WL/plane; author used contiguous descriptor-range reads.',
        scales='Host logical scales reassembled by saved row byte offsets, checked against logical matrix scale index.',
        cases=len(records),matrix_phases=phase_count,compared_scalars=scalar_count,bit_mismatches=0,results=records,
        limitations=['Software W8 reference/address proof only. No AiF native output-format, real device execution, timing, end-to-end model quality, or routing invariance claim.',
            'Inputs and mixture coefficients are explicit fixtures; selected expert IDs come from real old GGUF training traces.'])
    (OUT/'independent_replay.json').write_text(json.dumps(out,indent=2)+'\n')
    print(json.dumps({k:out[k] for k in ['status','fresh_process','cases','matrix_phases','compared_scalars','bit_mismatches']},indent=2))


if __name__=='__main__':main()
