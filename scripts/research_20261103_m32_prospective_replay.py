#!/usr/bin/env python3
"""Sealed-plan routing bind/replay; prospective execution requires explicit release."""
import argparse
import ctypes
import hashlib
import importlib.util
import json
import shutil
import subprocess
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
ROUND=ROOT/'results/research-2026-11-02/round2'
PARENT=ROUND/'chip-count-regime-screen/m32-prospective-tail-validation'
OUT=PARENT/'replay-adapter'
ANALYSIS=ROOT/'scripts/research_20261103_m32_prospective_tail_validation.py'
SNAP=ROUND/'ordinary-read-compiler/full-campaign-v6-W8-input4/source-snapshot'
CPP=SNAP/'ef7994fa2ba37a4b-research_20261103_ordinary_read_native.cpp'
PYTHON=SNAP/'7c61f09756e24f11-research_20261103_ordinary_read_backend.py'
TRAIN=ROUND/'w8-format-training/granite4tiny/decode.routes.npz'
NAMES=('free','common_cyclic','phase_independent')
ORIGINS={'free':('g4-M32-coupled-cp','incumbent4'),
         'common_cyclic':('g4-M32-cyclic-rotation-cp','incumbent0'),
         'phase_independent':('g4-M32-independent-rotation-cp','incumbent2')}
TRACE_QUERIES=(0,255,511)


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path,value):Path(path).write_text(json.dumps(value,indent=2)+'\n')
def load(path):
    with np.load(path,allow_pickle=False) as z:return {key:z[key] for key in z.files}


def freeze():
    OUT.mkdir(exist_ok=False);(OUT/'source-snapshot').mkdir()
    parent=json.loads((PARENT/'protocol.frozen.json').read_text())
    files=[Path(__file__),ANALYSIS,PARENT/'protocol.frozen.json',PARENT/'completion.frozen.json',CPP,PYTHON,
           ROOT/'scripts/research_20261102_whole_row_compiler.py',TRAIN]
    files += [Path(parent['plans'][name]['path']) for name in NAMES]
    files += [ROUND/'chip-count-regime-screen'/folder/'native_arrays.npz' for folder,ident in ORIGINS.values()]
    references=[]
    for path in files:
        item=dict(path=str(path),sha256=sha(path),bytes=path.stat().st_size)
        if path.suffix in ('.py','.cpp'):
            snap=OUT/'source-snapshot'/(item['sha256'][:16]+'-'+path.name);shutil.copy2(path,snap);item['snapshot']=str(snap)
        references.append(item)
    protocol=dict(status='frozen_before_new_capture_payload_read',utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__),references=references,parent_analysis_protocol_sha256=sha(PARENT/'protocol.frozen.json'),
        release='run requires explicit--confirm-root-sealed, expectedcaptureprotocol/completionSHA andrecordedrootconfirmationnote. Operatorwaitsforrootconfirmation; noauto-detection/release.',
        input_schema=dict(routes='int32[32,16,40,6]',prompt_ids='unique[32]',source_ids='datasetitemID[32]',
                          source_keys='twodatasetgroups[32],16each',layers='int32[40]exact0..39',completed='bool[32]alltrue',
                          token_positions='int32[32,16]nonnegative,consecutivewithineachprompt',token_ids='int32[32,16]nonnegative',
                          representation='fixed_full_FFN_W8A8_canonical_only',split='prospective_blind'),
        metadata_schema='manifest.json.items ordered32 withid/source_key/source_id/prompt_sha256/dataset_revision; capture_summary.json statuscomplete/completed_prompts32/complete_queries512/all40layersfinite true/manifest_sha256/protocol_sha256/decode_archive_sha256.',
        normalization='Neverreorder/filter/dropwarmup. Selectrawlayer0onlyafterall40layercoveragevalidation. Preserveprompt/tokenorder. source_keys becomes costarchive source_ids for frozenanalysis grouping; originalsource_ids retainedasdataset_item_ids.',
        immutable='ThreeGFplans,allrowvectors/jobs/offsets/addresses/inputcolumns unchanged. Cmd250,input1primary; input4sameplans sensitivity. No onlinewinner/refit/relabel/seedchange.',
        native=dict(source_sha256=sha(CPP),queries=3072,plans=3,input_copies=[1,4],priority='GF',
                    traces='q0/255/511 per3plansx2profiles=18; fullfrozenPythonReferenceGraph allnode compare.',
                    context='Originalsinglecontext/fullGUglobalA8/privateDown graph unchanged.'),
        preflight='OnlyexistingTRAIN/schema: all32x16x40x6 coveragechecks,18CPP/Pythonfulltraces, savedTRAIN scalarbridge (input1allplans,input4free). No newcapturepayload access.',
        binding='Recordexpectedrootseals, fullcapture/acquisitionprotocol+manifest+summary+NPZ+completion hashes. Completemanifest providestransitiveprovenance; adapterdoesnotreadrawprompttext orentirelargecapturelogs.',
        analysis='Directlyexec existinghash-pinned frozenanalyze withcostarchive; primaryfreevspredeclaredphase_independent, nosecondarywinnerchoice.',
        scope='ProspectiveL0syntheticservicegeneralization only; no newpairedfullmodel/device/quality proof. Nominal32PEhardware scale sensitivity.')
    dump(OUT/'protocol.frozen.json',protocol)
    print(json.dumps(dict(protocol_sha256=sha(OUT/'protocol.frozen.json'),source_sha256=sha(__file__)),indent=2))


def checked_inputs():
    p=json.loads((OUT/'protocol.frozen.json').read_text())
    assert sha(__file__)==p['source_sha256']
    for item in p['references']:assert sha(item['path'])==item['sha256'],item['path']
    return p,json.loads((PARENT/'protocol.frozen.json').read_text())


def check_routes(z,prospective):
    routes=z['routes'];assert routes.shape==(32,16,40,6) and routes.dtype==np.int32
    assert np.all((routes>=0)&(routes<64)) and np.all(np.diff(np.sort(routes,axis=-1),axis=-1)>0)
    assert np.array_equal(z['layers'],np.arange(40))
    assert z['completed'].dtype==np.bool_ and z['completed'].shape==(32,) and z['completed'].all()
    ids=z['prompt_ids'].astype(str);assert ids.shape==(32,) and len(set(ids))==32
    if prospective:
        assert str(z['representation'])=='fixed_full_FFN_W8A8_canonical_only'
        assert str(z['split'])=='prospective_blind'
        for key in ('source_ids','source_keys'):assert z[key].shape==(32,)
        assert sorted(Counter(z['source_keys'].astype(str)).values())==[16,16]
        positions=z['token_positions'];tokens=z['token_ids']
        assert positions.shape==tokens.shape==(32,16) and positions.dtype==tokens.dtype==np.int32
        assert np.all(positions>=0) and np.all(np.diff(positions,axis=1)==1) and np.all(tokens>=0)
    else:assert str(z['split'])=='TRAIN'
    return np.ascontiguousarray(routes[:,:,0].reshape(512,6),np.int32)


class Native:
    def __init__(self,directory):
        directory.mkdir(parents=True,exist_ok=True)
        self.binary=directory/'native.so'
        subprocess.run(['c++','-std=c++17','-O3','-shared','-fPIC',str(CPP),'-o',str(self.binary)],check=True)
        spec=importlib.util.spec_from_file_location('m32_prospective_sealed_backend',PYTHON)
        self.backend=importlib.util.module_from_spec(spec);sys.modules[spec.name]=self.backend;spec.loader.exec_module(self.backend)
        self.library=ctypes.CDLL(str(self.binary));self.fn=self.library.eval_ordinary_read
        i32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS');i64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
        self.fn.argtypes=[i32,i32,i32,i32,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,ctypes.c_int,i64,i64,i64,ctypes.c_void_p,i64]
        self.fn.restype=None

    def evaluate(self,plan,routes,service,trace=False):
        self.backend.validate_plan(plan,65,service)
        routes=np.ascontiguousarray(routes,np.int32);cost=np.zeros(len(routes),np.int64);bounds=cost.copy();nlog=np.zeros(1,np.int64)
        log=np.zeros((6000 if trace else 1,8),np.int64)
        self.fn(plan['offsets'],plan['jobs'],plan['input_columns'],routes,64,1,len(plan['jobs']),len(routes),6,service.array(),
                cost,bounds,ctypes.c_void_p(log.ctypes.data) if trace else None,nlog)
        assert np.all(cost>=bounds)
        if trace:assert nlog[0]<=len(log)
        return cost,bounds,log[:nlog[0]].copy() if trace else None


def replay(routes,parent,directory,full):
    native=Native(directory/'build');arrays={};checks=[]
    selected=np.arange(512) if full else np.asarray(TRACE_QUERIES)
    for name in NAMES:
        plan=load(parent['plans'][name]['path'])
        assert 'routes' not in plan
        for copies in (1,4):
            config=dict(parent['hardware_primary']);config.update(input_channel_copies=copies,down_first=0)
            service=native.backend.Service(**config)
            cost,bounds,_=native.evaluate(plan,routes[selected],service)
            key=f'input{copies}.{name}'
            arrays[key+'.cost']=cost.reshape(32,16) if full else cost
            arrays[key+'.bound']=bounds.reshape(32,16) if full else bounds
            for query in TRACE_QUERIES:
                c,b,tr=native.evaluate(plan,routes[query:query+1],service,True)
                py=native.backend.ReferenceGraph.build(plan,np.r_[routes[query],64],service).run()
                assert int(c[0])==py['cost'] and np.array_equal(tr,py['trace'])
                assert c[0]==cost[query] if full else c[0]==cost[list(selected).index(query)]
                arrays[key+f'.q{query}.trace']=tr
                checks.append(dict(plan=name,input_copies=copies,query=query,cost_ns=int(c[0]),nodes=len(tr),all_nodes_exact=True))
    return arrays,checks,dict(source=str(CPP),source_sha256=sha(CPP),binary=str(native.binary),binary_sha256=sha(native.binary))


def preflight():
    p,parent=checked_inputs();directory=OUT/'preflight';directory.mkdir(exist_ok=False)
    train=load(TRAIN);routes=check_routes(train,False)
    arrays,checks,build=replay(routes,parent,directory,False)
    for name,(folder,ident) in ORIGINS.items():
        prior=load(ROUND/'chip-count-regime-screen'/folder/'native_arrays.npz')
        assert np.array_equal(arrays['input1.'+name+'.cost'],prior[ident+'.GF.cost'][list(TRACE_QUERIES)])
        if name=='free':assert np.array_equal(arrays['input4.free.cost'],prior['selected_input4.GF.cost'][list(TRACE_QUERIES)])
    np.savez_compressed(directory/'arrays.npz',**arrays)
    dump(directory/'report.json',dict(status='PASS_TRAIN_only',protocol_sha256=sha(OUT/'protocol.frozen.json'),
         route_shape=list(train['routes'].shape),full40layer_validation=True,checks=checks,cpp=build,
         primary_scalar_bridges=9,input4_free_bridges=3,new_capture_payload_read=False,
         no_fits_or_plan_changes=True,analysis_source_unchanged=sha(ANALYSIS)))
    dump(directory/'completion.json',dict(status='sealed_preflight_PASS',
         files=[dict(path=str(x),sha256=sha(x),bytes=x.stat().st_size) for x in sorted(directory.rglob('*')) if x.is_file()]))
    print(json.dumps(dict(preflight_sha256=sha(directory/'completion.json'),trace_checks=len(checks),no_new_data=True),indent=2))


def bind_run(args):
    p,parent=checked_inputs()
    assert args.confirm_root_sealed and args.root_confirmation_note and args.capture_completion_sha256 and args.capture_protocol_sha256
    assert (OUT/'preflight/completion.json').exists()
    # Allpath/hash release arguments are supplied only after root explicitly confirms the capture seal.
    directory=OUT/'prospective';assert not directory.exists()
    capture=Path(args.capture_root).resolve();cp=Path(args.capture_protocol).resolve();cc=Path(args.capture_completion).resolve()
    assert sha(cp)==args.capture_protocol_sha256 and sha(cc)==args.capture_completion_sha256
    capture_protocol=json.loads(cp.read_text())
    assert p['parent_analysis_protocol_sha256'] in json.dumps(capture_protocol)
    manifest_path=capture/'manifest.json';summary_path=capture/'capture_summary.json';route_path=capture/'granite4tiny/decode.routes.npz'
    summary=json.loads(summary_path.read_text());manifest=json.loads(manifest_path.read_text())
    assert summary['status']=='complete' and summary['completed_prompts']==32 and summary['complete_queries']==512 and summary['all40layersfinite'] is True
    assert summary['manifest_sha256']==sha(manifest_path) and summary['protocol_sha256']==sha(cp)
    assert summary['decode_archive_sha256']==sha(route_path)
    z=load(route_path);routes=check_routes(z,True)
    items=manifest['items'];assert len(items)==32
    for index,item in enumerate(items):
        assert str(item['id'])==str(z['prompt_ids'][index])
        assert str(item['source_key'])==str(z['source_keys'][index]) and str(item['source_id'])==str(z['source_ids'][index])
        assert len(item['prompt_sha256'])==64
        assert str(item['dataset_revision']) not in ('','main','master','HEAD','latest')
    assert len(set(item['prompt_sha256'] for item in items))==32
    directory.mkdir()
    binding=dict(status='bound_after_explicit_root_capture_release',utc=datetime.now(timezone.utc).isoformat(),
        adapter_protocol_sha256=sha(OUT/'protocol.frozen.json'),parent_analysis_protocol_sha256=p['parent_analysis_protocol_sha256'],
        root_confirmation_note=args.root_confirmation_note,
        files=[dict(path=str(x),sha256=sha(x),bytes=x.stat().st_size) for x in (cp,cc,manifest_path,summary_path,route_path)],
        capture_provenance='Root-confirmedcapturecompletion/protocol seals provide fulltransitiveprovenance. Rawprompttext/answers/capturelogsnotopened byadapter.',
        coverage='All32x16x40x6validatedbeforeL0select;finitefullFFNcoverageassertedandboundviacapturesummary,rawroutesIDs/uniqueTop6 independentlychecked.',
        query_order='Originalpromptorder then16decodepositions; nofilter/sort/warmupdrop. source_keys maps toanalysis source_ids groups; datasetitemIDs retainedseparately.',
        fixed_maps={name:dict(path=parent['plans'][name]['path'],sha256=parent['plans'][name]['sha256']) for name in NAMES})
    dump(directory/'binding.frozen.json',binding)
    np.savez_compressed(directory/'L0.routes.npz',routes=routes.reshape(32,16,6),prompt_ids=z['prompt_ids'],source_keys=z['source_keys'],dataset_item_ids=z['source_ids'],token_positions=z['token_positions'],token_ids=z['token_ids'])
    arrays,checks,build=replay(routes,parent,directory,True)
    arrays.update(prompt_ids=z['prompt_ids'],source_ids=z['source_keys'],dataset_item_ids=z['source_ids'],token_positions=z['token_positions'],token_ids=z['token_ids'])
    costs=directory/'costs.npz';np.savez_compressed(costs,**arrays)
    dump(directory/'replay_report.json',dict(status='PASS_fixed_plan_prospective_replay',binding_sha256=sha(directory/'binding.frozen.json'),
        cost_archive_sha256=sha(costs),native_costs=3072,priority='GF',trace_checks=checks,cpp=build,
        analysis='Frozenexistinganalysis is run directly; no choice amongmaps/baselines/priorities.',no_fit=True))
    assert sha(ANALYSIS)==parent['source_sha256']
    subprocess.run([sys.executable,str(ANALYSIS),'analyze','--costs-path',str(costs)],check=True)
    checked_inputs()
    files=[x for x in directory.rglob('*') if x.is_file()]+[PARENT/'analysis/report.json']
    dump(directory/'completion.json',dict(status='sealed_prospective_replay_and_frozen_analysis',utc=datetime.now(timezone.utc).isoformat(),
        files=[dict(path=str(x),sha256=sha(x),bytes=x.stat().st_size) for x in sorted(files)],
        scope='L0syntheticservicegeneralization, notdevice/quality/fullmodelperformance. Allnegativecases retained.'))
    print(json.dumps(dict(completion=str(directory/'completion.json'),sha256=sha(directory/'completion.json')),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','preflight','bind-run'))
    parser.add_argument('--capture-root');parser.add_argument('--capture-protocol');parser.add_argument('--capture-completion')
    parser.add_argument('--capture-protocol-sha256');parser.add_argument('--capture-completion-sha256')
    parser.add_argument('--confirm-root-sealed',action='store_true');parser.add_argument('--root-confirmation-note')
    args=parser.parse_args();{'freeze':freeze,'preflight':preflight,'bind-run':lambda:bind_run(args)}[args.mode]()
