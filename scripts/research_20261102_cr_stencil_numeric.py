#!/usr/bin/env python3
"""Byte-ownership and full-size fixed-W8/A8 numerical audit of G4 GU CR+IPR.

Only host software artifacts are produced. The hardware premises are conditional.
Run: python scripts/research_20261102_cr_stencil_numeric.py
"""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import subprocess
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/research-2026-11-02/round3-cr-ipr/numeric'
OLD = ROOT / 'results/research-2026-11-02/round2/shared-materialize'
SELF = Path(__file__).resolve()
CPP = SELF.with_suffix('.cpp')
H, PAGE, WL, E, CH, RR, SR = 1536, 16384, 192, 64, 16, 1024, 2048
NROWS = E * RR + SR
ORDERS = ['0123', '2301', '2013', '0132', '3201', '2013']


def sha(p):
    h = hashlib.sha256()
    with Path(p).open('rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def write(p, x):
    Path(p).write_text(json.dumps(x, indent=2, ensure_ascii=False) + '\n')


def check(condition, message):
    if not condition:
        raise RuntimeError(message)


def quantize(x):
    m = np.max(np.abs(x))
    s = np.float32(1) if m == 0 else np.float32(m / np.float32(127))
    z = (x / s).astype(np.float32)
    q = np.copysign(np.floor(np.abs(z.astype(np.float64)) + .5), z)
    return np.clip(q, -127, 127).astype(np.int8), s


def addresses():
    # Independent direct coordinate formula; bank layout remains WL-major.
    for c in range(CH):
        for e in range(E):
            for lp, (plane, wl) in enumerate([(0, 2*e), (1, 2*e), (2, e),
                                               (0, 2*e+1), (1, 2*e+1), (3, e)]):
                start = (e*RR+c*64)*H+lp*PAGE
                yield c, plane, wl, 'routed', e, lp, start
        for lp, (plane, wl) in enumerate([(3,64),(2,64),(3,65),(2,65),(3,66),(2,66),
                                         (3,67),(2,67),(3,68),(2,68),(3,69),(2,69)]):
            start = (E*RR+c*128)*H+lp*PAGE
            yield c, plane, wl, 'shared', -1, lp, start


def pack(directory, weights, scales, routes, inputs, input_scales):
    directory.mkdir(parents=True, exist_ok=True)
    weights.tofile(directory/'canonical.w8')
    scales.tofile(directory/'canonical.scales.f32')
    routes.astype('<i4').tofile(directory/'routes.i32')
    inputs.tofile(directory/'inputs.a8')
    input_scales.astype('<f4').tofile(directory/'inputs.scales.f32')
    flat = weights.reshape(-1)
    coverage = np.zeros(flat.size, dtype=np.uint8)
    occupied, counts = set(), np.zeros((CH, 4), dtype=np.int32)
    banks = [open(directory/f'chip{c:02d}.bin', 'w+b') for c in range(CH)]
    for f in banks:
        f.truncate(4*WL*PAGE)
    ledger = open(directory/'page_ownership.jsonl', 'w')
    n = 0
    for c, plane, wl, kind, e, lp, start in addresses():
        key = c, plane, 0, wl
        check(key not in occupied and 0 <= wl < WL, 'physical overlap / block overflow')
        occupied.add(key)
        coverage[start:start+PAGE] += 1
        counts[c, plane] += 1
        f = banks[c]
        off = (wl*4+plane)*PAGE
        f.seek(off)
        payload = flat[start:start+PAGE].tobytes()
        f.write(payload)
        f.flush()
        f.seek(off)
        check(f.read(PAGE) == payload, 'byte readback differs')
        ledger.write(json.dumps(dict(chip=c, plane=plane, block=0, wordline=wl,
                                    bank_byte=off, kind=kind, expert=e, logical_page=lp,
                                    canonical_byte=start, bytes=PAGE,
                                    payload_sha256=hashlib.sha256(payload).hexdigest()))+'\n')
        n += 1
    ledger.close()
    for f in banks:
        f.close()
    check(np.all(coverage == 1), 'logical byte missing / replicated')
    check(np.all(counts == np.array([128, 128, 70, 70])), 'resident plane counts')
    # Fresh mmap readback checks every owned byte, separately from writes.
    bychip = [np.memmap(directory/f'chip{c:02d}.bin', dtype=np.int8, mode='r') for c in range(CH)]
    for c, plane, wl, kind, e, lp, start in addresses():
        off = (wl*4+plane)*PAGE
        check(np.array_equal(bychip[c][off:off+PAGE], flat[start:start+PAGE]), 'fresh byte audit')
    result = dict(resident_pages=n, resident_weight_bytes=int(flat.size),
                  verified_owner_bytes=int(coverage.size), logical_byte_coverage_exactly_once=True,
                  physical_page_ownership_unique=True, replicas=0, padding_bytes=0,
                  resident_pages_per_chip=396, plane_pages_per_chip=[128,128,70,70],
                  reserved_bytes=CH*4*WL*PAGE,
                  host_row_scales_bytes=int(scales.nbytes),
                  banks=[dict(path=str(directory/f'chip{c:02d}.bin'),
                              sha256=sha(directory/f'chip{c:02d}.bin')) for c in range(CH)])
    write(directory/'storage_audit.json', result)
    return result


def synthetic():
    rng = np.random.Generator(np.random.PCG64(2026110203))
    w = rng.integers(-127, 128, size=(NROWS,H), dtype=np.int8)
    for e in range(E):
        w[e*RR] = 127
        w[e*RR+1] = -127
        w[e*RR+2] = 0
    s = ((np.arange(NROWS, dtype=np.float32) % np.float32(997)+np.float32(1))*np.float32(.0001)).astype('<f4')
    boundary = [list(range(6)), list(range(58,64)), [0,63,1,62,2,61],
                [31,32,30,33,29,34], [7,8,15,16,31,32], [63,0,32,31,1,62]]
    boundary += [list(reversed(r)) for r in boundary]
    boundary += [[(b+j)%64 for j in range(6)] for b in (0,1,7,8,15,16,31,32,55,56,57,58,59,60,61,62,63)]
    routes = np.array(boundary+[rng.choice(E,6,replace=False).tolist() for _ in range(1000)], dtype='<i4')
    x = rng.integers(-127,128,size=(len(routes),H),dtype=np.int8)
    x[0] = 0
    x[1] = 127
    x[2] = -127
    x[3] = np.tile(np.array([127,-127],dtype=np.int8), H//2)
    for i,k in enumerate((0,511,512,1023,1024,1535), start=4):
        x[i] = 0
        x[i,k] = 127
    sx = ((np.arange(len(routes),dtype=np.float32)%17+1)*np.float32(.003)).astype('<f4')
    return w,s,routes,x,sx,dict(seed=2026110203, generator='NumPy PCG64', random_top6_cases=1000,
                              boundary_cases=len(boundary), input_extremes_and_phase_impulses=True)


def real_weights():
    manpath = OLD/'onecopy/manifest.json'
    man = json.loads(manpath.read_text())
    scale_meta = json.loads((OLD/'logical_scales.json').read_text())
    scales = np.fromfile(OLD/'logical_scales.f32',dtype='<f4')
    summary = json.loads((OLD/'summary.json').read_text())
    paths = [manpath,OLD/'logical_scales.json',OLD/'logical_scales.f32',OLD/'summary.json',OLD/'fixtures.npz',OLD/'source_ranges.json']
    fds = {}
    for chip in man['chips']:
        p = OLD/chip['path']
        check(sha(p)==chip['sha256'], 'source bank SHA changed')
        paths.append(p)
        fds[chip['chip']] = os.open(p,os.O_RDONLY)
    w = np.empty((NROWS,H),dtype=np.int8)
    sw = np.empty(NROWS,dtype='<f4')
    coverage = np.zeros(NROWS,dtype=np.uint8)
    for e in range(E+1):
        key = f'routed.{e}.gu' if e<E else 'shared.gu'
        views = man['execution_views'][key] if e<E else [v for p in range(8) for v in man['shared_gu_legal_views'][str(p)]['1']]
        base = e*RR if e<E else E*RR
        for v in views:
            r,n = v['logical_row_start'],v['rows']
            payload = []
            for page in range(v['pages']):
                wl,plane = v['start_wordline']+page//4,page%4
                off = ((v['block']*WL+wl)*4+plane)*PAGE
                payload.append(os.pread(fds[v['chip']],PAGE,off))
            raw = b''.join(payload)
            check(len(raw)==n*H and v['columns']==H, 'source view geometry')
            w[base+r:base+r+n] = np.frombuffer(raw,dtype=np.int8).reshape(n,H)
            so = v['scale_file_byte_offset']//4
            check(so==scale_meta[key]['offset']//4+r,'source scale index')
            sw[base+r:base+r+n] = scales[so:so+n]
            coverage[base+r:base+r+n] += 1
        check(sha_bytes(sw[base:base+(RR if e<E else SR)].tobytes())==scale_meta[key]['sha256'],'logical scales hash')
    for fd in fds.values():
        os.close(fd)
    check(np.all(coverage==1),'real canonical reconstruction coverage')
    fixtures = np.load(OLD/'fixtures.npz')
    cases = [next(r for r in summary['results'] if r['label']==f'fixture{i}.onecopy.a11111111') for i in range(summary['fixtures'])]
    routes = np.array([r['experts'] for r in cases],dtype='<i4')
    q,sx = zip(*(quantize(fixtures[f'x{i}']) for i in range(len(cases))))
    provenance = dict(source_input_sha256={str(p):sha(p) for p in paths},
                      checkpoint=summary['original_GGUF'], original_source_ranges='source_ranges.json',
                      numerical_contract=summary['quantization'], fixtures=summary['fixture'],
                      no_requantization=True, source_banks_reconstructed_readonly=True,
                      source_gu_layer=0, real_checkpoint_weights=True, real_captured_input=False,
                      route_cases=[dict(fixture=i,experts=r['experts'],train_prompt=r['prompt'],train_step=r['step']) for i,r in enumerate(cases)])
    return w,sw,routes,np.array(q),np.array(sx,dtype='<f4'),provenance


def sha_bytes(b):
    return hashlib.sha256(b).hexdigest()


def schedule_example(routes):
    example=[]
    route=list(map(int,routes[0]))
    for chip in range(CH):
        for group in range(2):
            macros=[]
            for label,which,half in [('A1',0,0),('D',-1,0),('A2',1,0),('B1',0,1),
                                     ('B2',1,1),('C',-2,0),('A3',2,0),('B3',2,1)]:
                sh=which<0
                start=chip*128+group*64+(32 if which==-2 else 0) if sh else chip*64+half*32
                slot=6 if sh else group*3+which
                macros.append(dict(label=label,kind='shared' if sh else 'routed',
                                   expert=-1 if sh else route[slot],route_output_slot=slot,
                                   projection='gate' if start<(1024 if sh else 512) else 'up',
                                   logical_GU_row_start=start,rows=32,input_phase_start=0,
                                   output_fifo_row_start=group*256+len(macros)*32,
                                   row_scale_start=(E*RR if sh else route[slot]*RR)+start))
            example.append(dict(chip=chip,group=group,macros=macros,
                                page_start_input_phases=[i*PAGE%H for i in range(24)],
                                wave_start_input_phases=[i*4*PAGE%H for i in range(6)]))
    return dict(route=route,description='Positional host FIFO scatter descriptors; serialized JSON is not a hardware ABI.',
                scale_storage='One host F32 scale per original logical row.',chips_and_groups=example)


def numpy_verify(directory, count):
    w = np.memmap(directory/'canonical.w8',dtype=np.int8,mode='r',shape=(NROWS,H))
    sw = np.fromfile(directory/'canonical.scales.f32',dtype='<f4')
    ids = np.fromfile(directory/'routes.i32',dtype='<i4').reshape(-1,6)
    x = np.fromfile(directory/'inputs.a8',dtype=np.int8).reshape(-1,H)
    sx = np.fromfile(directory/'inputs.scales.f32',dtype='<f4')
    checked = 0
    with open(directory/'run.outputs.bin','rb') as f:
        for t in range(count):
            integer = np.frombuffer(f.read(8192*8),dtype='<i8')
            floats = np.frombuffer(f.read(8192*4),dtype='<f4')
            ix = np.concatenate([np.arange(e*RR,(e+1)*RR) for e in ids[t]]+[np.arange(E*RR,NROWS)])
            ref = w[ix].astype(np.int64) @ x[t].astype(np.int64)
            rf = (ref.astype(np.float32)*(sw[ix]*sx[t]).astype(np.float32)).astype(np.float32)
            check(np.array_equal(integer,ref),'independent NumPy integer mismatch')
            check(np.array_equal(floats.view(np.uint32),rf.view(np.uint32)),'independent NumPy F32 mismatch')
            checked += len(ref)
    return dict(cases=count,integer_rows=checked,scaled_f32_rows=checked,mismatches=0,
                implementation='Independent NumPy I64 logical-row matvec; no page schedule imported')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out',type=Path,default=OUT)
    args = ap.parse_args()
    out = args.out.resolve()
    check(out==OUT.resolve(), 'writes are restricted to the assigned numeric directory')
    out.mkdir(parents=True,exist_ok=True)
    start = time.monotonic()
    argv = ['c++','-O3','-march=native','-fno-fast-math','-ffp-contract=off','-std=c++17','-fopenmp-simd',str(CPP),'-o',str(out/'stencil-interpreter')]
    subprocess.run(argv,check=True)
    write(out/'build_command.json',dict(argv=argv,source_sha256=sha(CPP),binary_sha256=sha(out/'stencil-interpreter'),compiler=subprocess.check_output(['c++','--version'],text=True).splitlines()[0]))
    report = dict(status='running',utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  source_sha256={str(p):sha(p) for p in (SELF,CPP)},numpy_version=np.__version__,
                  python_version=platform.python_version(),geometry=dict(H=H,I=512,E=E,K=6,shared_I=1024,W=8,q=4,g=16,page_bytes=PAGE,wordlines_per_block=WL),
                  assumptions=['User-premised skip-WL CR plus independent-plane read.',
                               'Per-wave PB source permutation remains conditional.',
                               'Existing single row accumulation may carry its state across page and wave boundaries; no new accumulator is introduced.',
                               'Common full A8 input1536 is loaded once and cycled in its original order.',
                               'Row scales and positional FIFO scatter are host metadata; their capacity/time is separately disclosed, not a native AiF ABI claim.'],
                  prohibitions=dict(input_fragments=False,host_column_partials=False,new_accumulator=False,weight_replicas=False),
                  schedule=dict(macros=['A1','D','A2','B1','B2','C','A3','B3'],consume_orders=ORDERS,groups=2,waves_per_chip=12,pages_per_chip=48,emitted_rows_per_chip=512,macro_rows=32))
    report['accumulation']=dict(cpu_reference='I64 exact whole-row sum',worst_case_abs_sum=H*127*127,
                                fits_signed_I32=True,new_hardware_I64_accumulator_assumed=False,
                                page_stream_state='One current input column and one current row sum, carried continuously across pages/waves.')
    write(out/'protocol.json',report)
    modes = {}
    for name, factory in [('synthetic',synthetic),('checkpoint',real_weights)]:
        print('MATERIALIZE',name,flush=True)
        w,sw,routes,x,sx,provenance = factory()
        directory = out/name
        storage = pack(directory,w,sw,routes,x,sx)
        write(directory/'schedule_example.json',schedule_example(routes))
        check(w.shape==(NROWS,H) and sw.shape==(NROWS,), 'canonical geometry')
        write(directory/'provenance.json',provenance)
        input_hashes = {str(directory/p):sha(directory/p) for p in ['canonical.w8','canonical.scales.f32','routes.i32','inputs.a8','inputs.scales.f32']}
        del w,sw,x,sx
        cmd = [str(out/'stencil-interpreter'),str(directory),str(len(routes)),str(directory/'run')]
        with open(directory/'stdout.log','w') as log, open(directory/'stderr.log','w') as err:
            run = subprocess.run(cmd,stdout=log,stderr=err)
        write(directory/'run_command.json',dict(argv=cmd,exit_code=run.returncode))
        check(run.returncode==0,f'{name} interpreter failed; see stderr.log')
        stats = json.loads((directory/'run.json').read_text())
        check(stats['row_comparisons']==len(routes)*8192 and stats['physical_pages_read']==len(routes)*48*CH,'numeric counters')
        independent = numpy_verify(directory,len(routes) if name=='checkpoint' else 29)
        check(input_hashes=={p:sha(p) for p in input_hashes},'fixed operands changed')
        if name=='checkpoint':
            check(all(sha(p)==h for p,h in provenance['source_input_sha256'].items()),'real source changed')
        modes[name] = dict(storage=storage,numeric=stats,independent_numpy=independent,
                           fixed_operand_sha256=input_hashes,output_sha256=sha(directory/'run.outputs.bin'),
                           provenance=str(directory/'provenance.json'))
        print('COMPLETE',name,stats,flush=True)
    mutations=[]
    for fault in ['weight','phase','permutation','unselected','scatter','scale']:
        directory=out/'synthetic'
        # Exact geometry uses the complete input file. Failures occur in the
        # first case, except scale corruption in the second (nonzero) case.
        cmd=[str(out/'stencil-interpreter'),str(directory),str(modes['synthetic']['numeric']['cases']),str(out/f'mutation-{fault}'),fault]
        run=subprocess.run(cmd,capture_output=True,text=True)
        check(run.returncode!=0,f'mutation {fault} was not detected')
        mutations.append(dict(fault=fault,argv=cmd,exit_code=run.returncode,diagnostic=run.stderr.strip()))
    report.update(status='pass',modes=modes,mutation_controls=mutations,
                  failures=0,elapsed_seconds=time.monotonic()-start,
                  claims=dict(cpu_address_and_fixed_format_numerical_equivalence=True,
                              flash_execution=False,flash_timing=False,task_quality=False,
                              original_GGUF_numerical_equivalence=False,native_AiF_ABI=False,
                              publication_ready_hardware_validation=False))
    write(out/'report.json',report)
    syn=modes['synthetic']['numeric'];real=modes['checkpoint']['numeric']
    text=f'''The G4 one-copy common-input GU stencil passes this CPU packed-byte and numerical audit. This is a conditional address construction and fixed-format arithmetic result, not a flash measurement, task-quality result, original-GGUF equivalence, or native AiF ABI.

The fixed geometry is H1536, routed I512/E64/K6, shared I1024, W8, q4, g16, 16KiB pages. All 64 experts and shared GU are resident once: 6,336 pages / 103,809,024 weight bytes, with per-chip plane occupancies (128,128,70,70), in one 192-WL block per plane. Every owned physical byte was read back; every logical byte appears exactly once. Host row scales occupy 270,336 bytes and are not silently charged to flash payload.

Each group consumes A1,D,A2,B1,B2,C,A3,B3, with three pages / 32 complete H rows per macro. The six plane orders are 0123/2301/2013/0132/3201/2013. Two groups read 48 pages / 12 waves per chip, producing 512 rows per chip and 8,192 GU outputs overall. No unselected expert is read. All page-start phases and complete-row scatter positions are checked. One stream row accumulator carries row state across page/wave boundaries; only complete rows are rescaled and scattered. No input fragment, host column partial sum, replica, or extra accumulator is introduced.

Synthetic: {syn['cases']} cases = 29 boundary/route-order cases plus 1,000 deterministic random top6 routes, all across 16 chips, with {syn['row_comparisons']:,} exact canonical I64 and same-count F32-bit row comparisons, zero mismatches. Fixed A8 vectors include zeros, ±127, alternating signs and single impulses at phase boundaries. The first 29 cases also pass independent NumPy logical-row arithmetic.

Checkpoint audit: {real['cases']} pre-existing TRAIN-route fixtures reuse read-only L0 raw W8 from all 64 Granite4Tiny expert GU tensors and shared GU, including their original saved row scales. Their inputs are the older explicit synthetic fixtures, not captured GU activations. No requantization or model execution occurs. All {real['row_comparisons']:,} row results pass C++ packed-vs-canonical checks and an independent NumPy I64/F32-bit audit. Before/after source and operand SHA256 checks pass.

Six deliberately corrupted interpreter runs (weight contribution, phase reset, page permutation, unselected read, scatter and row scale) must fail and their diagnostics are in report.json. These are detection controls, not observed candidate failures.

Reproduce: `python {SELF}`. build_command.json and per-mode run_command.json preserve the exact commands. Source, binary, bank, fixed-operand and output SHA256 digests are retained. page_ownership.jsonl gives every physical page and its contiguous logical-byte owner; the formula plus exact byte readback covers every byte.

Limits: skip-WL CR plus IPR are user-premised; PB source permutation remains conditional. The interpreter assumes the ordinary whole-row input phase and accumulator can continue across ordered page/wave drains. Positional output metadata and host row scales are modeled and audited, but no hardware ABI, finite FIFO implementation, ECC/CR circuit, DMA, latency or energy is validated. No downstream activation/Down or full-model task-quality result is claimed. Passing checks does not establish publication readiness.
'''
    (out/'report.md').write_text(text)
    print(json.dumps(dict(status=report['status'],synthetic=syn,checkpoint=real,elapsed_seconds=report['elapsed_seconds']),indent=2),flush=True)


if __name__=='__main__':
    main()
