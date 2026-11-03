#!/usr/bin/env python3
"""Repack saved W8 bytes into three M32 banks and prove full-row composition."""
import argparse
import gzip
import hashlib
import importlib.util
import json
import os
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ROUND=ROOT/'results/research-2026-11-02/round2'
SOURCE=ROUND/'shared-materialize'
PARENT=ROUND/'chip-count-regime-screen/m32-prospective-tail-validation'
OUT=PARENT/'numeric-materialization'
HELPER=ROOT/'scripts/research_20261102_shared_materialize_replay.py'
PAGE=16384;WAVE=65536;BLOCK=192*WAVE
NAMES=('free','common_cyclic','phase_independent')


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as f:
        while chunk:=f.read(4194304):h.update(chunk)
    return h.hexdigest()


def bh(array):return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def load(path):
    with np.load(path,allow_pickle=False) as z:return {k:z[k] for k in z.files}


def freeze():
    OUT.mkdir(exist_ok=False);(OUT/'source-snapshot').mkdir()
    man=json.loads((SOURCE/'onecopy/manifest.json').read_text())
    sources=[Path(__file__),HELPER,SOURCE/'onecopy/manifest.json',SOURCE/'logical_scales.f32',
             SOURCE/'logical_scales.json',SOURCE/'source_ranges.json',SOURCE/'fixtures.npz',
             PARENT/'protocol.frozen.json',PARENT/'completion.frozen.json',
             ROUND/'w8-format-training/granite4tiny/decode.routes.npz']
    sources += [SOURCE/x['path'] for x in man['chips']]
    sources += [PARENT/'plans'/(name+'.frozen.npz') for name in NAMES]
    parent=json.loads((PARENT/'protocol.frozen.json').read_text())
    sources += [Path(x['path']) for x in parent['references'] if x['path'].endswith(('.cpp','.py')) and ('backend' in x['path'] or 'ordinary_read_native' in x['path'])]
    references=[]
    for path in sources:
        item=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
        if path.suffix in ('.py','.cpp'):
            snap=OUT/'source-snapshot'/(item['sha256'][:16]+'-'+path.name);shutil.copy2(path,snap);item['snapshot']=str(snap)
        references.append(item)
    for chip in man['chips']:assert sha(SOURCE/chip['path'])==chip['sha256']
    with np.load(ROUND/'w8-format-training/granite4tiny/decode.routes.npz') as z:
        routes=z['routes'][:,:,0].reshape(512,6)
        chosen=routes[[0,511]].copy()
    assert all(len(set(x))==6 for x in chosen)
    np.savez_compressed(OUT/'train_route_fixtures.frozen.npz',query_indices=np.asarray([0,511]),routes=chosen)
    protocol=dict(status='frozen_before_materialization_numeric_run',utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),references=references,parent_plan_protocol_sha256=sha(PARENT/'protocol.frozen.json'),
        route_fixture_sha256=sha(OUT/'train_route_fixtures.frozen.npz'),
        source='Existing16chip onecopy savedW8payloads only. Reconstruct130matrices and originalhostrow-scales; noGGUFread, download, requantization or originalmaterializer.main.',
        checks='Sourcebankfullsha/1312allocationpayloads/rowcoverage,195source_range functionalgate/up/down W8/scales hashes. Thenall3frozenplans completeonecopy/fullrow/address/nonoverlap; readbackallpayloads/zero-pagepadding.',
        geometry=dict(chips=32,planes=4,page_bytes=PAGE,wordlines_per_block=192,blocks_per_chip=1),
        cases='Cartesianproduct existingfixture x0/alpha0 andx1/alpha1 with fixedactualW8 TRAINquery0/511 top6orderedexperts; fourcases. Oldfixture reference_outputs deliberatelyunused.',
        numeric='I64exactdot thenoneF32convert andF32(originalW8scale*A8scale) multiplication. Pinnedpurehelpers qvector/activate only. CompleteexpertGU assembled beforefullvectorA8 maxabs/127, sameprivateDownI64/F32, orderedalphamixthenaddshared.',
        fringe='Writeusefulbytes thenzero-pad allprogrammedpages; CPUreadcompletepages andcompute everyemittedwholepaddingrow,drain/dropit. Read/drop partialtrailingrowbytes. Applyoriginalscales onlytovalidlogicalrows; neverinventfringescales orprovephysicalpartialflush.',
        comparisons='EveryvalidGU/DownI64 andF32bitpattern, SwiGLUF32, fullA8bytes/scale, orderedmix/outputbits; allthreeplaced vscanonicalreconstructedW8. Explicitstage scalarcounts andhashes.',
        scope='L0softwarephysical-materialization/fullrownumericproof only; no newheldout, newfullmodelpairedproof, nativeABI/device/timing/quality claim. ExistingW8canonicalmodelroute provenance inherited, not rerun.',
        execution='Target<90s compute plus boundedI/O, retainfullverification ratherthantruncate ifslower. Source/plan/protocol hashes recheckedafterrun.')
    dump(OUT/'protocol.frozen.json',protocol)
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__)),indent=2))


def reconstruct():
    manifest=json.loads((SOURCE/'onecopy/manifest.json').read_text())
    smeta=json.loads((SOURCE/'logical_scales.json').read_text())
    scale_bytes=(SOURCE/'logical_scales.f32').read_bytes()
    matrices={};scales={};coverage={};columns={}
    for allocation in manifest['storage_allocations']:
        key=allocation['logical_matrix'];c=allocation['columns']
        assert key not in columns or columns[key]==c
        columns[key]=c
    for key,m in smeta.items():
        raw=scale_bytes[m['offset']:m['offset']+m['bytes']]
        assert len(raw)==m['rows']*4 and hashlib.sha256(raw).hexdigest()==m['sha256']
        scales[key]=np.frombuffer(raw,dtype='<f4').copy()
        assert np.all(np.isfinite(scales[key])) and np.all(scales[key]>0)
        matrices[key]=np.empty((m['rows'],columns[key]),np.int8);coverage[key]=np.zeros(m['rows'],np.uint8)
    intervals={};fd={x['chip']:os.open(SOURCE/x['path'],os.O_RDONLY) for x in manifest['chips']}
    for a in manifest['storage_allocations']:
        key=a['logical_matrix'];row,n,c=a['row_start'],a['rows'],a['columns']
        at=((a['block']*192+a['start_wordline'])*4)*PAGE
        assert at==a['linear_byte_offset'] and a['weight_bytes']==n*c
        raw=os.pread(fd[a['chip']],n*c,at)
        assert len(raw)==n*c and hashlib.sha256(raw).hexdigest()==a['payload_sha256']
        matrices[key][row:row+n]=np.frombuffer(raw,np.int8).reshape(n,c)
        coverage[key][row:row+n]+=1
        intervals.setdefault(a['chip'],[]).append((at,at+n*c))
    for f in fd.values():os.close(f)
    for key,x in coverage.items():assert np.all(x==1)
    for spans in intervals.values():
        spans.sort();assert all(b<=c for (_,b),(c,_) in zip(spans,spans[1:]))
    ranges=json.loads((SOURCE/'source_ranges.json').read_text())
    for entry in ranges:
        prefix='shared' if entry['expert'] is None else 'routed.'+str(entry['expert'])
        if '_down_' in entry['tensor']:
            key=prefix+'.down';w=matrices[key];s=scales[key]
        else:
            key=prefix+'.gu';half=len(matrices[key])//2;start=0 if '_gate_' in entry['tensor'] else half
            w=matrices[key][start:start+half];s=scales[key][start:start+half]
        assert w.shape==(entry['rows'],entry['columns'])
        assert bh(w)==entry['functional_i8_sha256'] and bh(s)==entry['functional_row_scales_sha256']
    assert len(matrices)==130 and len(ranges)==195 and sum(w.nbytes for w in matrices.values())==155713536
    audit=dict(storage_allocations=len(manifest['storage_allocations']),logical_matrices=len(matrices),source_projection_ranges=len(ranges),
               bytes=sum(w.nbytes for w in matrices.values()),allocation_payloads_exact=True,source_functional_W8_and_scales_exact=True,
               logical_rows_once=True,packed_original_GGUF_reread=False,
               matrices={key:dict(shape=list(w.shape),weight_sha256=bh(w),scale_sha256=bh(scales[key])) for key,w in matrices.items()})
    return matrices,scales,smeta,audit


def materialize(name,weights,smeta):
    plan=load(PARENT/'plans'/(name+'.frozen.npz'))
    directory=OUT/'banks'/name;directory.mkdir(parents=True)
    handles=[(directory/f'chip{chip:02d}.bin').open('x+b') for chip in range(32)]
    for f in handles:f.truncate(BLOCK)
    views={};allocations=[];occupied={};cover={key:np.zeros(len(w),np.uint8) for key,w in weights.items()}
    for e in range(65):
        for phase in range(2):
            key=('shared' if e==64 else 'routed.'+str(e))+('.gu' if phase==0 else '.down');start=0
            begin,end=map(int,plan['offsets'][e,phase])
            for job in range(begin,end):
                chip,n,c,pages,waves=map(int,plan['jobs'][job]);ac,block,wl,span,ap,plane=map(int,plan['addresses'][job])
                assert (chip,pages,waves)==(ac,ap,span) and block==plane==0 and wl+waves<=192
                assert c==weights[key].shape[1] and pages==(n*c+PAGE-1)//PAGE
                at=((block*192+wl)*4+plane)*PAGE
                raw=weights[key][start:start+n].tobytes();full=raw+bytes(pages*PAGE-len(raw))
                f=handles[chip];f.seek(at);assert f.write(full)==len(full)
                cover[key][start:start+n]+=1
                occupied.setdefault(chip,[]).append((wl,wl+waves))
                a=dict(job=job,logical_matrix=key,chip=chip,block=block,wordline=wl,plane=plane,
                       row_start=start,rows=n,columns=c,pages=pages,waves=waves,physical_byte_offset=at,
                       scale_byte_offset=smeta[key]['offset']+4*start,weight_bytes=len(raw),
                       payload_sha256=hashlib.sha256(raw).hexdigest(),programmed_sha256=hashlib.sha256(full).hexdigest(),
                       emitted_rows=pages*PAGE//c,partial_tail_bytes=pages*PAGE%c)
                allocations.append(a);views.setdefault(key,[]).append(a);start+=n
            assert start==len(weights[key])
    for x in cover.values():assert np.all(x==1)
    for spans in occupied.values():
        spans.sort();assert all(b<=c for (_,b),(c,_) in zip(spans,spans[1:]))
    for f in handles:f.flush();os.fsync(f.fileno());f.close()
    fds=[os.open(directory/f'chip{chip:02d}.bin',os.O_RDONLY) for chip in range(32)]
    for a in allocations:
        raw=os.pread(fds[a['chip']],a['pages']*PAGE,a['physical_byte_offset'])
        assert hashlib.sha256(raw).hexdigest()==a['programmed_sha256']
        assert raw[:a['weight_bytes']]==weights[a['logical_matrix']][a['row_start']:a['row_start']+a['rows']].tobytes()
        assert not any(raw[a['weight_bytes']:])
    chip_files=[dict(chip=c,path=str(directory/f'chip{c:02d}.bin'),sha256=sha(directory/f'chip{c:02d}.bin'),
                     size=BLOCK,filesystem_allocated_bytes=(directory/f'chip{c:02d}.bin').stat().st_blocks*512) for c in range(32)]
    audit=dict(name=name,copies=0,complete_logical_row_coverage=True,all_payloads_and_zero_padding_readback=True,
               nonoverlap_single_block_perchip=True,useful_bytes=sum(a['weight_bytes'] for a in allocations),
               pages=sum(a['pages'] for a in allocations),waves=sum(a['waves'] for a in allocations),
               allocated_lsb_bytes=32*BLOCK,allocations=allocations,chips=chip_files)
    assert audit['useful_bytes']==155713536 and audit['pages']==9764 and audit['waves']==2441
    with gzip.open(directory/'manifest.json.gz','wt') as f:json.dump(audit,f)
    return fds,views,{key:v for key,v in audit.items() if key!='allocations'}


def run():
    p=json.loads((OUT/'protocol.frozen.json').read_text());assert sha(__file__)==p['source_sha256']
    for item in p['references']:assert sha(item['path'])==item['sha256']
    began=time.monotonic();weights,scales,smeta,source_audit=reconstruct();rebuild_seconds=time.monotonic()-began
    spec=importlib.util.spec_from_file_location('pinned_pure_saved_numeric_helpers',HELPER)
    helper=importlib.util.module_from_spec(spec);spec.loader.exec_module(helper)
    assert helper.__name__!='__main__'
    layouts={};storage={};t=time.monotonic()
    for name in NAMES:
        fds,views,audit=materialize(name,weights,smeta);layouts[name]=(fds,views);storage[name]=audit
    materialize_seconds=time.monotonic()-t
    def gemv(key,q,qs,name=None):
        count=len(weights[key]);out=np.empty(count,np.float32);integer=np.empty(count,np.int64)
        if name is None:
            z=weights[key].astype(np.int64)@q.astype(np.int64)
            factor=(scales[key]*qs).astype(np.float32)
            return (z.astype(np.float32)*factor).astype(np.float32),z,dict(computed_rows=count,emitted_rows=count,pages=0,partial_tail_bytes=0)
        fds,views=layouts[name];coverage=np.zeros(count,np.uint8);ledger=dict(computed_rows=0,emitted_rows=0,pages=0,partial_tail_bytes=0)
        for a in views[key]:
            # Independent per-plane-page physical address reads, not payload-only reads.
            packed=[]
            for page in range(a['pages']):
                wl=a['wordline']+page//4;plane=page%4
                at=((a['block']*192+wl)*4+plane)*PAGE
                raw=os.pread(fds[a['chip']],PAGE,at);assert len(raw)==PAGE;packed.append(raw)
            raw=b''.join(packed);emitted=a['emitted_rows'];c=a['columns'];n=a['rows'];start=a['row_start']
            w=np.frombuffer(raw[:emitted*c],np.int8).reshape(emitted,c)
            z=w.astype(np.int64)@q.astype(np.int64)
            assert not np.any(z[n:]) and not any(raw[emitted*c:])
            assert a['scale_byte_offset']==smeta[key]['offset']+start*4
            factor=(scales[key][start:start+n]*qs).astype(np.float32)
            out[start:start+n]=(z[:n].astype(np.float32)*factor).astype(np.float32)
            integer[start:start+n]=z[:n];coverage[start:start+n]+=1
            ledger['computed_rows']+=emitted;ledger['emitted_rows']+=emitted;ledger['pages']+=a['pages'];ledger['partial_tail_bytes']+=a['partial_tail_bytes']
        assert np.all(coverage==1)
        return out,integer,ledger
    fixtures=load(SOURCE/'fixtures.npz');route_fixtures=load(OUT/'train_route_fixtures.frozen.npz')
    saved={};results=[];compared={};matrix_phases=0;numeric_start=time.monotonic()
    def calculate(fi,route,name=None):
        q,qs=helper.qvector(fixtures[f'x{fi}']);alpha=fixtures[f'alpha{fi}'];mixed=np.zeros(1536,np.float32);stages={};ledgers={}
        for j,e in enumerate(list(map(int,route))+[64]):
            key='shared' if e==64 else 'routed.'+str(e)
            gu,gi,gl=gemv(key+'.gu',q,qs,name);activation=helper.activate(gu)
            dq,ds=helper.qvector(activation);down,di,dl=gemv(key+'.down',dq,ds,name)
            stages[key]=dict(gu_i64=gi,gu_f32=gu,activation_f32=activation,down_a8=dq,
                             down_a8_scale_f32=np.asarray(ds),down_i64=di,down_f32=down)
            ledgers[key]=dict(gu=gl,down=dl)
            mixed=(mixed+(down*alpha[j]).astype(np.float32)).astype(np.float32) if j<6 else (mixed+down).astype(np.float32)
        return stages,mixed,ledgers
    for fi in (0,1):
        for qi,route in zip(route_fixtures['query_indices'],route_fixtures['routes']):
            label=f'fixture{fi}.trainq{qi}';reference,mixed,_=calculate(fi,route)
            saved[label+'.canonical.output']=mixed
            for key,stages in reference.items():
                for stage,value in stages.items():saved[label+'.canonical.'+key+'.'+stage]=value
            for name in NAMES:
                actual,output,ledgers=calculate(fi,route,name);hashes={}
                for key,stages in actual.items():
                    for stage,value in stages.items():
                        target=reference[key][stage]
                        assert value.dtype==target.dtype and value.shape==target.shape
                        assert value.tobytes()==target.tobytes(),(label,name,key,stage)
                        compared[stage]=compared.get(stage,0)+value.size
                        saved[label+'.'+name+'.'+key+'.'+stage]=value
                        hashes[key+'.'+stage]=bh(value)
                    matrix_phases+=2
                assert output.tobytes()==mixed.tobytes()
                compared['ordered_mixed_output_f32']=compared.get('ordered_mixed_output_f32',0)+output.size
                saved[label+'.'+name+'.output']=output
                results.append(dict(case=label,layout=name,route=list(map(int,route)),every_stage_bitexact=True,
                                    output_sha256=bh(output),stages_sha256=hashes,actual_page_emission_ledgers=ledgers))
    for fds,_ in layouts.values():
        for fd in fds:os.close(fd)
    numeric_seconds=time.monotonic()-numeric_start
    np.savez_compressed(OUT/'numeric_arrays.npz',**saved)
    dump(OUT/'source_reconstruction.json',source_audit)
    for item in p['references']:assert sha(item['path'])==item['sha256']
    report=dict(status='PASS',utc=datetime.now(timezone.utc).isoformat(),protocol_sha256=sha(OUT/'protocol.frozen.json'),
        source_sha256=sha(__file__),source_files_unchanged=True,source_audit='source_reconstruction.json',storage=storage,
        fixtures=4,placed_cases=len(results),matrix_phases=matrix_phases,compared_scalars_by_stage=compared,
        total_compared_scalars=sum(compared.values()),bit_mismatches=0,results=results,
        seconds=dict(reconstruct=rebuild_seconds,materialize_and_full_readback=materialize_seconds,numeric=numeric_seconds,total=time.monotonic()-began),
        limitations=p['scope'],fringe_scope=p['fringe'],original_materializer_main_executed=False,
        pure_helpers_imported_only=['qvector','activate'],old_fixture_reference_outputs_used=False,
        new_capture_or_test_payload_read=False)
    dump(OUT/'report.json',report)
    files=[x for x in OUT.rglob('*') if x.is_file() and x.suffix!='.bin' and not x.name.endswith('.log')]
    dump(OUT/'completion.json',dict(status='sealed_PASS',utc=datetime.now(timezone.utc).isoformat(),
        files=[dict(path=str(x),sha256=sha(x),bytes=x.stat().st_size) for x in sorted(files)],
        binary_banks='All96bankhashes/sizes in report storage and gzmanifests; sourceunchangedverified beforeandafter.'))
    print(json.dumps({k:report[k] for k in ('status','placed_cases','matrix_phases','total_compared_scalars','bit_mismatches','seconds')},indent=2))
    print('completion_sha256',sha(OUT/'completion.json'))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','run'));args=parser.parse_args()
    {'freeze':freeze,'run':run}[args.mode]()
