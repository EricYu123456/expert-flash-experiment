#!/usr/bin/env python3
"""Round2 stages. prepare is model/prompt-body-free; every real stage is gated."""
import argparse
import datetime as dt
import json
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
sys.dont_write_bytecode = True
import time
from research_20261103_local_affine_stage0_core import *
from research_20261103_local_affine_stage0_pipeline import *

REPO = Path('/home/ericyu/expert_flash_experiment')
ROOT = REPO / 'results/research-2026-11-03-no-cr/round02-local-affine'
OUT = ROOT / 'stage0'
PROTOCOL = ROOT / 'stage0_protocol.frozen.json'
EXPECTED = '9254e7fdf71f965b8a8716a35838522fb378ca2421c6779d9650e385aad62dc8'
SOURCES = sorted((REPO / 'scripts').glob('research_20261103_local_affine_stage0*'))
UP = Path('/home/nfs_client/ericyu/expert_flash_experiment/sources/llama.cpp-927dada6c')
BLD = Path('/home/nfs_client/ericyu/expert_flash_experiment/build/llama-route-trace/llama')
LIBS = [BLD / s for s in ['src/libllama.a', 'ggml/src/libggml.a', 'ggml/src/libggml-cpu.a', 'ggml/src/libggml-base.a']]


def now():
    return dt.datetime.now(dt.timezone.utc).isoformat()


def protocol():
    assert sha(PROTOCOL) == EXPECTED
    return json.loads(PROTOCOL.read_text())


def stat(p):
    s = Path(p).stat()
    return {'path': str(Path(p).resolve()), 'size': s.st_size, 'mtime_ns': s.st_mtime_ns, 'dev': s.st_dev, 'inode': s.st_ino}


def freeze_splits(pr):
    # Filename IDs only. No prompt-body opens here; TEST names need not be listed.
    paths = sorted(Path(pr['data']['source']).glob('train_*.txt'))
    assert len(paths) == 200 and len({p.stem for p in paths}) == 200
    paths.sort(key=lambda p: (hashlib.sha256(('20261103-local-affine-v1:' + p.stem).encode()).hexdigest(), p.stem))
    records = [{'id': p.stem, **stat(p), 'rank_sha256': hashlib.sha256(('20261103-local-affine-v1:' + p.stem).encode()).hexdigest(),
                'split': 'FIT' if i < 120 else 'VALID' if i < 160 else 'SCORE', 'split_index': i if i < 120 else i - 120 if i < 160 else i - 160}
               for i, p in enumerate(paths)]
    target = OUT / 'split_ids.frozen.json'
    content = {'status': 'filename_ids_stat_frozen_no_bodies_read', 'records': records, 'TEST_opened': False}
    if target.exists():
        assert json.loads(target.read_text()) == content
    else:
        save(target, content)
    return content


def algorithm_manifest():
    return {'status': 'algorithm_choices_before_candidate_data', 'tree': {
        'leading_solver': 'scipy eigsh LinearOperator centered covariance, k1 LA, fixed normal v0 seed20261103+layer*1000+node, tol1e-10 maxiter4096 ncvmin(H,20); residual<=1e-8; explicit evr symmetric covariance fallback',
        'sign': 'largest-absolute coordinate positive; argmax chooses smallest coordinate tie',
        'normal_storage': 'F32; recompute split projected values using runtime C++ sequential FP64 products/adds, no contraction',
        'threshold': 'choose distinct-value boundary minimizing |2left-N|; lower boundary tie; FP64 midpoint or lower endpoint if midpoint rounds up',
        'ties': 'no ID/rank split; identical x always same runtime result; actual FIT occupancy audited; missing depth is inconclusive',
        'leaf_centers': 'unweighted FP64 FIT mean, stored F32, each C prefix separately'},
        'projection': {'solver': 'full centered-FIT FP64 covariance scipy symmetric eigh evd, descending stable eigenvalue order, same sign convention; nested basis',
                       'normal_and_balanced': 'both complete configurations separately refit; balanced stored F32 P_j divided by max(sigma_j,sigma1/16), identity scaling if sigma1=0',
                       'runtime': 'F32 x-minus-mean, stored F32 P promoted to FP64, pure-FP64 BLAS DGEMM, cast once F32, then C++ dynamicA8',
                       'families': 'r128 C1/4/16/32 primary+ablation; r64 C16/32 secondary; pooled R2 r512 normal/balanced and H2048 identity at each C'},
        'ridge': {'grid': GRID, 'normalization': 'local normalized by sum local weights; pooled within-leaf demean then sum weighted statistics normalized by total weight',
                  'eigen': 'symmetric evd, clamp only roundoff negatives, relative cutoff1e-10 of largest eigenvalue; lambda0 minimum-norm correction around own-basis optimal pooled prior',
                  'infinity': 'pooled eta infinity=zero; local lambda infinity=own-basis pooled slope; zero trace=no correction',
                  'selection': 'natural VALID total deployed error for each operator/C/rank/format/weight/guard/N; independent pooled eta before local lambda; preserve selected and best finite',
                  'learning': '80 first deterministic FIT IDs versus all120 coefficients; same full120 unsupervised tree/P disclosed; no SCORE sample selection'},
        'numeric': {'quantization': 'C++ std::round ties away on F32 quotient; zero amax scale1 q0; no numpy rint',
                    'integer_dots': 'pure FP64 ndarray DGEMM integer operands alpha1 beta0; width127^2<2^53 proof plus signed/random I64 audits',
                    'teacher': 'canonical G/U full rows then F32 sw*sx, dot castF32, multiply; original pinned full-I ggml SiLU/product bit comparison via persistent static-linked binary worker (original libraries are non-PIC); whole-I A8; canonical Down',
                    'bias': 'C++ sequential original row order FP64 weighted residual accumulation per output, divide sum weights, castF32; actual prediction then separate F32+bias, no FMA',
                    'FP32_diagnostic': 'selected FP32 map coefficients, same exact-real A8 design, FP64 BLAS accumulation and F32 cast, independently calibrated same biases; VALID quantization diagnostic only'},
        'guard': 'natural VALID Euclidean distance<=np.quantile(method=linear) at50/70/85; 100 unguarded; original exact fallback has error0 and paid3HI/two epochs; missing cutoff/hits=inconclusive',
        'bootstrap': 'same40prompt multinomial draws from default_rng(20261103),10000; SD ddof1; max-|t| centered Delta_b-Delta_hat; linear95quantile; any undefined draw/nonliteral SD0 makes family inference inconclusive; no redraw/drop',
        'comparison_selection': 'both formats separate; finite local selected and bestfinite versus best VALID eligible same-format same-rank pooled R1 or same-leaf R0; cost eligibility includes actual leaf zero maps, guard fraction, host P and retained metadata; peroperator, permodel, equaloperator and pooled aggregates frozen before SCORE',
        'SCORE_gate': 'separate explicit Root score-review reference after comparisons frozen; no SCORE capture/teacher/fit in prepare or fitval',
        'activation_worker_IPC': 'same x86 little-endian host; int32[N,I], then contiguous F32 gate[N,I], Up[N,I]; return int32 status0 then F32 product[N,I]. Exact byte loops; stdout binary only, stderr diagnostics; close stdin and require exit0. N1..256,I1..10944; all8 worker environments checked; fixed original static GGML/ISA. Calls/bytes/walltime retained as offline teacher costs.',
        'resource_claim': 'logical complete-row pages/work only; separate actual CPU walltime. No NAND latency, PE rate, one-phase whole-layer or whole-model quality claim. Full rawbank, scales/bias/keys/P/tree/centers/nonFFN/KV ledgers audited under8e9 and8GiB.'}


def prepare():
    pr = protocol()
    OUT.mkdir(exist_ok=True)
    freeze_splits(pr)
    manifest = OUT / 'algorithm_manifest.frozen.json'
    if manifest.exists():
        assert json.loads(manifest.read_text()) == algorithm_manifest()
    else:
        save(manifest, algorithm_manifest())
    fd = OUT / 'freezes'; fd.mkdir(exist_ok=True)
    revision = 1 + max([int(p.name[3:]) for p in fd.glob('rev*')] or [0])
    rd = fd / f'rev{revision:03d}'; rd.mkdir()
    numeric = REPO / 'scripts/research_20261103_local_affine_stage0_numeric.cpp'
    capture = REPO / 'scripts/research_20261103_local_affine_stage0_capture.cpp'
    flags = ['c++', '-O3', '-march=native', '-fno-fast-math', '-ffp-contract=off', '-std=c++17', '-fopenmp', '-pthread', '-I' + str(UP / 'include'), '-I' + str(UP / 'ggml/include')]
    link = [str(LIBS[0]), str(LIBS[1]), '-ldl', str(LIBS[2]), str(LIBS[3]), '-lm']
    commands = [flags + [str(numeric), '-o', str(rd / 'numeric-self-test')] + link,
                flags + ['-DLOCAL_AFFINE_LIBRARY', '-DLOCAL_AFFINE_NO_GGML', '-shared', '-fPIC', str(numeric), '-o', str(rd / 'numeric.so'), '-lm'],
                flags + ['-DLOCAL_AFFINE_LIBRARY', str(capture), str(numeric), '-o', str(rd / 'capture')] + link]
    save(rd / 'build_commands.json', commands)
    for i, command in enumerate(commands):
        built = subprocess.run(command, capture_output=True, text=True)
        (rd / f'build{i}.stdout').write_text(built.stdout); (rd / f'build{i}.stderr').write_text(built.stderr)
        assert built.returncode == 0, built.stderr[-5000:]
    libs = []
    for module in [np, __import__('scipy')]:
        libs += list(Path(module.__file__).parent.parent.glob(module.__name__ + '.libs/*'))
    files = [p for p in SOURCES if p.is_file()] + [PROTOCOL, manifest, OUT / 'split_ids.frozen.json', *LIBS,
        UP / 'ggml/src/ggml-cpu/vec.cpp', UP / 'src/llama-graph.cpp', UP / 'src/models/deepseek2.cpp', UP / 'src/models/olmoe.cpp',
        rd / 'numeric-self-test', rd / 'numeric.so', rd / 'capture', rd / 'build_commands.json'] + [p for p in libs if p.is_file()]
    receipt = {'status': 'frozen_before_round2_candidate_data', 'at_utc': now(), 'revision': revision,
               'protocol_sha256': EXPECTED, 'files': {str(p): sha(p) for p in files},
               'model_stat': [stat(m['gguf']) for m in pr['models']],
               'compiler': subprocess.check_output(['c++', '--version'], text=True),
               'lscpu': subprocess.check_output(['lscpu'], text=True), 'python': sys.version,
               'numpy': np.__version__, 'scipy': __import__('scipy').__version__,
               'model_weights_activations_prompt_bodies_read': False,
               'authorization': 'implementation/build/synthetic only; real data requires separate Root source review'}
    for p in SOURCES:
        if p.is_file(): shutil.copy2(p, rd / p.name)
    save(rd / 'freeze_receipt.json', receipt)
    cpp = subprocess.run([str(rd / 'numeric-self-test')], capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS='8', OMP_THREAD_LIMIT='8'))
    (rd / 'synthetic_cpp.stdout').write_text(cpp.stdout); (rd / 'synthetic_cpp.stderr').write_text(cpp.stderr)
    assert cpp.returncode == 0, cpp.stderr
    test = subprocess.run([sys.executable, str(REPO / 'scripts/research_20261103_local_affine_stage0_synthetic.py'), str(rd / 'numeric.so')], capture_output=True, text=True)
    (rd / 'synthetic_python.stdout').write_text(test.stdout); (rd / 'synthetic_python.stderr').write_text(test.stderr)
    assert test.returncode == 0, test.stderr[-5000:]
    save(rd / 'synthetic_receipt.json', {'at_utc': now(), 'status': 'passed', 'cpp': cpp.stdout, 'python': test.stdout,
                                       'source_snapshot_precedes_synthetic': True})
    save(OUT / 'latest_ready.json', {'revision': revision, 'receipt': str(rd / 'freeze_receipt.json'), 'folder': str(rd)})
    print(json.dumps({'ready_for_source_review': str(rd), 'source_hashes': {str(p): sha(p) for p in SOURCES if p.is_file()}, 'synthetic': test.stdout}, indent=2), flush=True)


def verify_freeze(review):
    assert review and Path(review).is_file(), 'explicit immutable Root review receipt required'
    latest = json.loads((OUT / 'latest_ready.json').read_text())
    receipt = json.loads(Path(latest['receipt']).read_text())
    approved=json.loads(Path(review).read_text())
    assert approved['status']=='approved' and approved['phase']=='FITVAL'
    assert approved['protocol_sha256']==EXPECTED and approved['source_revision']==receipt['revision']
    assert approved['source_freeze_sha256']==sha(latest['receipt'])
    for p, h in receipt['files'].items(): assert sha(p) == h, p
    for m, ident in zip(protocol()['models'], receipt['model_stat']): assert stat(m['gguf']) == ident
    return Path(latest['folder']), receipt


def identities(pr):
    entries = json.loads((OUT / 'split_ids.frozen.json').read_text())['records']
    path = OUT / 'content_identities.frozen.json'
    if path.exists(): return json.loads(path.read_text())
    records = [{**r, 'sha256': sha(r['path'])} for r in entries]
    previous = REPO / 'results/research-2026-11-03-no-cr/round01-stage0/model_identities.frozen.json'
    prior = json.loads(previous.read_text())['models'] if previous.exists() else []
    models = []
    for m in pr['models']:
        ident = stat(m['gguf']); cached = [r for r in prior if str(Path(r['path']).resolve()) == ident['path'] and r['size'] == ident['size'] and r['mtime_ns'] == ident['mtime_ns'] and r['device']==ident['dev'] and r['inode']==ident['inode']]
        models.append({**ident, 'sha256': cached[0]['sha256'] if cached else sha(m['gguf']), 'reused_unchanged_round1_hash': bool(cached)})
        assert stat(m['gguf'])==ident,'model mutated during hashing'
    result = {'records': records, 'models': models, 'created_at_utc': now(), 'TEST_opened': False,
              'SCORE_hash_only_no_capture_or_semantic_read': True}
    save(path, result)
    return result


def capture_splits(rd, pr, identity, splits):
    for split in splits:
        jobs = [r for r in identity['records'] if r['split'] == split]
        for r in jobs:
            assert stat(r['path'])=={k:r[k] for k in stat(r['path'])}
            assert sha(r['path'])==r['sha256'],'prompt changed before capture'
        tsv = OUT / (split + '.frozen.tsv')
        tsv.write_text(''.join(r['path'] + '\t' + r['id'] + '\n' for r in jobs))
        children = []
        for model in pr['models']:
            ident=next(r for r in identity['models'] if r['path']==str(Path(model['gguf']).resolve()))
            assert stat(model['gguf'])=={k:ident[k] for k in stat(model['gguf'])}
            folder = OUT / 'capture' / model['name'] / split
            assert not folder.exists(), 'preserve capture attempts; use explicit fresh attempt before retry'
            folder.mkdir(parents=True)
            log = open(folder / 'capture.log', 'w')
            mode = 'capture-control' if split == 'FIT' else 'capture'
            command = [str(rd / 'capture'), mode, model['name'], model['gguf'], str(tsv), str(folder)]
            p = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            children.append((p, log, command))
        codes = []
        for p, log, command in children:
            codes.append(p.wait()); log.close()
        save(OUT / (split + '.capture_receipt.json'), {'at_utc': now(), 'commands': [c for _, _, c in children], 'returncodes': codes})
        assert codes == [0, 0], 'invalid capture; repair same round, retain all logs'


def resource_ledger(pr):
    result = {'scope': 'Analytic full-model banks and pilot geometry; not a measured schedule or native delay',
              'target_host_budgets': [8000000000, 8 * 2 ** 30], 'be_encoded_coefficient_nominal_budget_decimal': 10 ** 12 / 3,
              'hardware': {'chips': 16, 'planes_per_chip': 4, 'page_bytes': 16384, 'be_LSB_us': 28, 'CR': False},
              'unknowns': ['physical usable/ECC/spare', 'actual PE throughput', 'input-post ABI/retention', 'FIFO/command/block fragmentation', 'group/channel collision schedule', 'measured CPU cache/all-host time', 'whole-layer fallback DAG'], 'models': {}}
    for m in pr['models']:
        H,E,L,I = m['hidden'],m['experts'],m['routed_layer_count'],m['intermediate']
        original = L * E * 3 * H * I + L * 3 * H * m['native_shared_intermediate']
        if m['name'] == 'deepseek': original += 3 * H * 10944
        capture_report=json.loads((OUT/'capture'/m['name']/'FIT'/'capture_report.json').read_text())
        nonffn=capture_report['source_nonFFN_tensor_bytes']
        log=(OUT/'capture'/m['name']/'FIT'/'capture.log').read_text(errors='replace')
        kv=[float(x) for x in re.findall(r'CPU KV buffer size\s*=\s*([0-9.]+) MiB',log)]
        assert kv,'missing actual CPU KV allocation accounting'
        kv_upper=math.ceil((max(kv)+.005)*2**20)
        configs = []
        families=[(r,C,'local') for r in [64,128] for C in ([16,32] if r==64 else [1,4,16,32])]
        families += [(r,C,'pooled') for r in [64,128,512,2048] for C in ([16,32] if r==64 else [1,4,16,32])]
        for r,C,family in families:
            multiplicity=C if family=='local' else 1
            g={64:2,128:4,512:16,2048:16}[r]
            raw = L * E * multiplicity * H * r
            aux = {'map_row_scales_F32': L * E * multiplicity * H * 4, 'map_leaf_bias_F32': L * E * C * H * 4,
                   'region_map_keys_U64': L * E * C * 8, 'P_F32': 0 if r==2048 else L * r * H * 4,
                   'tree_normals_thresholds': L*((C-1)*H*4+(C-1)*8), 'leaf_means': L*C*H*4}
            lower = raw + sum(aux.values())
            host_aux=sum(aux.values())+nonffn+kv_upper
            configs.append({'rank': r, 'C':C, 'family':family, 'raw_map_bank_bytes':raw, 'auxiliary_bytes':aux,
                'subtotal_excluding_nonFFN_KV_cache':lower,'8e9_subtotal_fits':lower<=8000000000,'8GiB_subtotal_fits':lower<=8*2**30,
                'original_nonFFN_encoded_tensor_bytes_host':nonffn,'pilot_context256_CPU_KV_allocation_bytes_upper_from_rounded_log':kv_upper,
                'cache_allowance_bytes':0,'allhost_known_required_inclusive_bytes':raw+host_aux,'NANDmaps_known_host_aux_inclusive_bytes':host_aux,
                'allhost_8e9_ruled_out_by_known_bytes':raw+host_aux>8000000000,'allhost_8GiB_ruled_out_by_known_bytes':raw+host_aux>8*2**30,
                'nonFFN_KV_cache_inclusive_host_fit':'Known tensor/aux/KV byte floor only; framework/runtime buffers, group contexts, cache and replicas must be added before any affirmative host-fit claim. Offline teacher is allowed above target.',
                'complete_output_rows':H,'complete_row_bytes':r,'pages_per_map':H*r//16384,'g':g,
                'ideal_pages_per_chip':H*r//16384//g,'per_plane_pages_if_even':H*r//16384//g//4,
                'coefficient_bytes_including_retained_fallback':original+raw,
                'separate_physical_conditions':'requires actual groups/collisions/FIFO/posts/commands/pages; ideal one sensing wave not latency'})
        result['models'][m['name']]={'original_raw_W8_FFN_fallback_bytes':original,'original_fallback_retained':True,'configs':configs,
            'r0_NAND_map_bytes':0,'infinity_shared_map_dedup':'one lowered map per expert if identical; never bill C duplicated maps',
            'native_shared_branch_unchanged':m['native_shared_intermediate']>0,'pilot_compiles_all64_or_all_layers':False}
    save(OUT/'resource_ledger.json',result)


def fitval(review):
    rd, receipt = verify_freeze(review); pr = protocol()
    assert not (OUT / 'FITVAL.started.json').exists(), 'preserve prior attempt'
    save(OUT / 'FITVAL.started.json', {'at_utc':now(),'review':review,'source_revision':receipt['revision']})
    identity = identities(pr)
    capture_splits(rd, pr, identity, ['FIT', 'VALID'])
    split_ids = {s:[r['id'] for r in identity['records'] if r['split']==s] for s in ['FIT','VALID','SCORE']}
    operators=[]
    for m in pr['models']:
        for op in select_experts(OUT/'capture'/m['name']/'FIT',m['pilot_layers'],split_ids['FIT']):
            operators.append({**op,'model':m['name'],'intermediate':m['intermediate']})
    assert len(operators)==8
    save(OUT/'operators.FIT_only.frozen.json',{'at_utc':now(),'operators':operators,'VALID_SCORE_targets_not_queried':True})
    num=Numeric(rd/'numeric.so'); points=[]; inconclusive=[]
    for m in pr['models']:
        for layer in m['pilot_layers']:
            train=dataset(OUT/'capture'/m['name']/'FIT',layer,split_ids['FIT'])
            valid=dataset(OUT/'capture'/m['name']/'VALID',layer,split_ids['VALID'])
            tree=fit_tree(num,train['x'],layer); basis,sigma=fit_projection(train['x'])
            tf=OUT/'fitted'/m['name']/f'layer{layer}'; tf.parent.mkdir(parents=True,exist_ok=True)
            tree_archive(tf,tree,basis,sigma,num,train['x'])
            for op in [p for p in operators if p['model']==m['name'] and p['layer']==layer]:
                of=tf/f'expert{op["expert"]}'; wf=tf/f'weights{op["expert"]}'
                command=[str(rd/'capture'),'weights',m['gguf'],str(layer),str(op['expert']),str(wf)]
                subprocess.run(command,check=True)
                weights=load_weights(wf,m['intermediate'])
                yf=query_targets(num,train,weights,tf/f'expert{op["expert"]}.FIT.npy',op,'FIT')
                yv=query_targets(num,valid,weights,tf/f'expert{op["expert"]}.VALID.npy',op,'VALID')
                p,ic=fit_operator(num,of,op,train,valid,yf,yv,tree,basis,sigma,m['intermediate'])
                points+=p; inconclusive += [{**q,'operator':[m['name'],layer,op['expert']]} for q in ic]
    resource_ledger(pr)
    freeze_comparisons(points,OUT/'comparisons.before_SCORE.frozen.json',split_ids['SCORE'],inconclusive)
    selected_files=[OUT/'comparisons.before_SCORE.frozen.json',OUT/'operators.FIT_only.frozen.json',OUT/'content_identities.frozen.json',OUT/'resource_ledger.json']
    selected_files += sorted(p for p in (OUT/'fitted').rglob('*') if p.is_file())
    num.close()
    save(OUT/'FITVAL.complete.json',{'status':'complete_pending_SCORE_release','at_utc':now(),'source_revision':receipt['revision'],
        'files':{str(p):sha(p) for p in selected_files},'points':len(points),'inconclusive_configurations':len(inconclusive),'SCORE_capture_or_targets_opened':False,'numeric_dot_audits':num.dot_audits,
        'activation_worker':{'calls':num.activation_calls,'IPC_bytes':num.activation_ipc_bytes,'seconds':num.activation_seconds}})
    print('FITVAL complete; SCORE remains sealed',flush=True)


def score(review, score_review):
    rd,source_receipt=verify_freeze(review); assert score_review and Path(score_review).is_file(),'explicit Root SCORE release required'
    approved=json.loads(Path(score_review).read_text())
    assert approved['status']=='approved' and approved['phase']=='SCORE' and approved['protocol_sha256']==EXPECTED
    assert approved['source_revision']==source_receipt['revision'] and approved['source_freeze_sha256']==sha(OUT/'freezes'/f'rev{source_receipt["revision"]:03d}'/'freeze_receipt.json')
    assert approved['FITVAL_complete_sha256']==sha(OUT/'FITVAL.complete.json')
    assert approved['comparisons_sha256']==sha(OUT/'comparisons.before_SCORE.frozen.json')
    complete=json.loads((OUT/'FITVAL.complete.json').read_text())
    assert complete['status']=='complete_pending_SCORE_release'
    for p,h in complete['files'].items(): assert sha(p)==h
    assert not (OUT/'SCORE.started.json').exists(),'SCORE single-open stage; preserve attempt and mark any reuse development'
    save(OUT/'SCORE.started.json',{'at_utc':now(),'Root_release':score_review,'comparisons_sha256':sha(OUT/'comparisons.before_SCORE.frozen.json')})
    pr=protocol();identity=json.loads((OUT/'content_identities.frozen.json').read_text())
    capture_splits(rd,pr,identity,['SCORE'])
    ids=[r['id'] for r in identity['records'] if r['split']=='SCORE'];num=Numeric(rd/'numeric.so')
    operators=json.loads((OUT/'operators.FIT_only.frozen.json').read_text())['operators'];datasets={};targets={};trees={}
    for m in pr['models']:
        for layer in m['pilot_layers']:
            ds=dataset(OUT/'capture'/m['name']/'SCORE',layer,ids);datasets[(m['name'],layer)]=ds
            tf=OUT/'fitted'/m['name']/f'layer{layer}';trees[(m['name'],layer)]=tree_load(tf)
            for op in [p for p in operators if p['model']==m['name'] and p['layer']==layer]:
                weights=load_weights(tf/f'weights{op["expert"]}',m['intermediate'])
                targets[(m['name'],layer,op['expert'])]=query_targets(num,ds,weights,tf/f'expert{op["expert"]}.SCORE.npy',op,'SCORE')
    freeze=json.loads((OUT/'comparisons.before_SCORE.frozen.json').read_text())
    score_stage(num,freeze,datasets,targets,trees,OUT/'score_results')
    num.close()
    save(OUT/'SCORE.complete.json',{'at_utc':now(),'status':'complete_Root_critic_classification_required','TEST_opened':False,'no_new_inference_quality_claim':True})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','fitval','score']);p.add_argument('--source-review',default='');p.add_argument('--score-review',default='');a=p.parse_args()
    if a.action=='prepare': prepare()
    elif a.action=='fitval': fitval(a.source_review)
    else: score(a.source_review,a.score_review)
