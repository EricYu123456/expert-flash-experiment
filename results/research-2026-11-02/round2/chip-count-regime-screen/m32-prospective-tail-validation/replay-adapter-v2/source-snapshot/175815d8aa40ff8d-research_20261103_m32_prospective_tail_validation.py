#!/usr/bin/env python3
"""Freeze prospective M32 maps/statistics before any new workload is opened."""
import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ROUND = ROOT / 'results/research-2026-11-02/round2'
BASE = ROUND / 'chip-count-regime-screen'
OUT = BASE / 'm32-prospective-tail-validation'
ORIGINS = {
    'free': ('g4-M32-coupled-cp', 'incumbent4'),
    'common_cyclic': ('g4-M32-cyclic-rotation-cp', 'incumbent0'),
    'phase_independent': ('g4-M32-independent-rotation-cp', 'incumbent2'),
}
PRIMARY = 'phase_independent'
SEED = 20261115
DRAWS = 2000


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load(path):
    with np.load(path, allow_pickle=False) as z:
        return {key: z[key] for key in z.files}


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def stat(values):
    return dict(mean_ns=float(np.mean(values)), p99_ns=float(np.quantile(values, .99, method='linear')))


def freeze():
    OUT.mkdir(exist_ok=False)
    (OUT / 'plans').mkdir()
    (OUT / 'source-snapshot').mkdir()
    sources = [Path(__file__)]
    plans = {}
    shared = None
    for name, (folder, ident) in ORIGINS.items():
        directory = BASE / folder
        original = directory / 'native' / (ident + '.plan.npz')
        z = load(original)
        assert z['rows'].shape == (65, 2, 32)
        assert np.all(np.sort(z['rows'][:64, 0], axis=1) == [0] * 7 + [40] + [41] * 24)
        assert np.all(np.sort(z['rows'][:64, 1], axis=1) == [0] * 20 + [128] * 12)
        assert np.all(z['rows'][:64, 0].sum(1) == 1024) and np.all(z['rows'][:64, 1].sum(1) == 1536)
        assert np.array_equal(z['rows'][64].sum(1), [2048, 1536])
        if shared is None:
            shared = z['rows'][64].copy()
        assert np.array_equal(shared, z['rows'][64])
        target = OUT / 'plans' / (name + '.frozen.npz')
        # Only the static plan is copied. Prior TRAIN routes are deliberately omitted.
        np.savez_compressed(target, **{key: value for key, value in z.items() if key != 'routes'})
        r = json.loads((directory / 'native_report.json').read_text())
        record = next(x for x in r['records'] if x['id'] == ident and x['priority'] == 'GF' and x.get('input_copies', 1) == 1)
        plans[name] = dict(path=str(target), sha256=sha(target), original_path=str(original), original_sha256=sha(original),
                           original_report_sha256=sha(directory / 'native_report.json'), id=ident,
                           selection='PreviouslysealedTRAINmean-selected, GF fixed before newworkload. Never chooseamongmaps online.',
                           training_cost=record['cost'])
        sources += [original, directory / 'protocol.frozen.json', directory / 'native_report.json', directory / 'native_arrays.npz']
    frozen_backend = ROUND / 'ordinary-read-compiler/full-campaign-v6-W8-input4/source-snapshot'
    cpp = frozen_backend / 'ef7994fa2ba37a4b-research_20261103_ordinary_read_native.cpp'
    python = frozen_backend / '7c61f09756e24f11-research_20261103_ordinary_read_backend.py'
    numeric = ROUND / 'full-ffn-integer-reference/frozen_protocol.json'
    training = ROUND / 'w8-format-training/protocol.frozen.json'
    training_json = json.loads(training.read_text())
    binary = ROUND / 'w8-format-training/build/granite4tiny-canonical-capture'
    capture_source = ROOT / 'scripts/research_20261103_w8_training_capture_granite4tiny.cpp'
    assert sha(binary) == '8f5171680fdab9c0284c11af11ac214998a2087314d2302e0af869f5a8476b80'
    assert sha(numeric) == '3163341827c25b210c42126ae7465c1fe8be9827aab93d314a89ef8a54975f40'
    checkpoint = Path('/home/nfs_client/ericyu/expert_flash_experiment/models/granite4tiny-q4_k_m.gguf')
    checkpoint_hash = training_json['inputs_sha256'][str(checkpoint)]
    assert checkpoint_hash == '5a38b08c441ae1adbafb1d2b8a7167e0d48734d83af68b268cefea1eec553dcd'
    sources += [cpp, python, numeric, training, binary, capture_source,
                ROUND / 'w8-format-training/selected_train_prompts.json']
    references = []
    for path in sources:
        item = dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)
        if path.suffix in ('.py', '.cpp'):
            snap = OUT / 'source-snapshot' / (item['sha256'][:16] + '-' + path.name)
            shutil.copy2(path, snap)
            item['snapshot'] = str(snap)
        references.append(item)
    service = dict(json.loads((BASE / 'g4-M32-independent-rotation-cp/protocol.frozen.json').read_text())['service'])
    assert service['chips'] == 32 and service['channels'] == 8 and service['command_ns'] == 250
    assert service['input_channel_copies'] == 1 and service['down_first'] == 0
    protocol = dict(status='frozen_before_new_prompt_or_capture_payload', utc=datetime.now(timezone.utc).isoformat(),
                    source_sha256=sha(__file__), references=references, plans=plans,
                    model=dict(checkpoint=str(checkpoint), checkpoint_sha256=checkpoint_hash,
                               checkpoint_hash_provenance='InheritedverifiedfrozenW8numeric/trainingcontract, checkpointnotreread forthisfreeze.',
                               numeric_protocol_sha256=sha(numeric), capture_binary_sha256=sha(binary),
                               H=1536, routed_I=512, shared_I=1024, experts=64, topk=6, full_layers=40, evaluated_raw_layer=0),
                    hardware_primary=service,
                    sensitivity=dict(input_channel_copies=4, same_static_plans=True, same_GF=True,
                                     rule='Predeclaredsecondarysensitivity, no newfit/reselection. Costprofile changeonly.'),
                    primary_comparison=dict(candidate='free', baseline=PRIMARY, priority='GF',
                                            rule='Compare all512newqueries; never selectbestbaseline/map onvalidation. Commoncyclicreportedseparately.'),
                    acquisition=dict(owner='Separateauthorizedcaptureagent', prompts=32, sources=2, prompts_per_source=16,
                                     decode_per_prompt=16, total_decode_queries=512,
                                     required='Acquisitionprotocol pinsdatasetrevision/IDs/rawtextSHA and excludes allpreviouslycapturedpromptIDs/textSHA; freezeafterthisplanprotocol butbeforetargetpayload. Rawanswers notinprompts. No output/modelbasedselection.',
                                     data_access='Thisproducerhasnotdownloaded/openednewprompt/test/capturepayload. Noacquisition/inference bythisscript.'),
                    capture_contract=dict(binary='ExistingcanonicalG4W8capturebinary referencedabove',
                                          numeric='All40layer fullFFN fixed-rowW8[-127,127], full-vector dynamicA8maxabs/127, I64dot/oneF32rescale; attention/router retainGGUF. FullGU/A8barrier andprivateDown unchanged.',
                                          required_routes_shape=[32,16,40,6], experts_unique_per_topk=True, expert_id_range=[0,63],
                                          route_scope='Useonlydecode rawlayer0inexactcaptureprompt/tokenorder. Prefillseparate/excluded; no firststep/warmup/slowquerydrop. Includeall16fixeddecodesteps.',
                                          validity='Requireall32promptscomplete/all40layers finitecoverage/noinvalidrouteID/duplicateorholes. Onfailurearchivepartial, noresample/truncate/replaceusingoutputs; do notpublishfull512primarymetrics.'),
                    analysis=dict(quantile='numpyquantile methodlinear', primary_metrics=['mean_ns','p99_ns'],
                                  percent_gain='100*(1-candidate_metric/baseline_metric),positive=faster',
                                  paired='Perprompt16-querymean, P99 andmax; queryimprove/tie/regress andmaximumregression; all32promptrecords retained.',
                                  source_breakdown='Both16-promptsourcegroups descriptive, no selection.',
                                  cluster_bootstrap=dict(draws=DRAWS, seed=SEED, rng='numpy.default_rng.PCG64',
                                                         cluster='Prompt,32clusters sampledwithreplacement; retainall16steps andsameindicesforallplans.',
                                                         intervals='Percentile2.5/97.5 forpairedmean/P99difference andrelativegain; descriptiveonly,not512IIDobservations orsignificanceproof.'),
                                  fixed_decision='Reportpositive/negativegeneralization asobserved. No fitting/relabeling/changingpriority/primarybaseline/threshold afterpayload.'),
                    replay_validation='Bindnewcapturemanifest/protocol/routeshash onlyafterrootconfirmscapturecomplete. UnmodifiedfrozenCPP; q0/255/511 fullPythontrace perplan/profile. Noonlinemapchoice.',
                    numeric_scope_bridge='Eachplancontains everylogicalGU1024/down1536rowonce perroutedexpert andsharedGU2048/down1536. Preserve originalrowweightscales; concatenatewhole-row results bylogicalrowmap beforefull-vectorA8. This isalgebraic fullrowcontractbridge, not anewpairedM32modelrun, deviceABIproof orqualityclaim.',
                    limitations='Prospectivegeneralizationpilot, notallmodelservice/quality/device/end-to-end orpublicationproof.32PEisexplicit hardwarescale sensitivity. NominalnativeI32/outputscaleABI remainsconditional aspriorstudies.')
    dump(OUT / 'protocol.frozen.json', protocol)
    dump(OUT / 'completion.frozen.json', dict(status='plan_and_analysis_frozen_before_workload', utc=datetime.now(timezone.utc).isoformat(),
         protocol_sha256=sha(OUT / 'protocol.frozen.json'), source_sha256=sha(__file__),
         files=[dict(path=str(x),sha256=sha(x),bytes=x.stat().st_size) for x in sorted(OUT.rglob('*')) if x.is_file()],
         no_new_payload_read=True))
    print(json.dumps(dict(path=str(OUT / 'protocol.frozen.json'), protocol_sha256=sha(OUT / 'protocol.frozen.json'),
                          completion_sha256=sha(OUT / 'completion.frozen.json'), plans=plans), indent=2))


def analyze(path):
    """Only called later on an independently bound cost archive; never reads text/routes."""
    p = json.loads((OUT / 'protocol.frozen.json').read_text())
    assert sha(__file__) == p['source_sha256']
    z = load(path)
    ids, source_ids = z['prompt_ids'].astype(str), z['source_ids'].astype(str)
    assert ids.shape == source_ids.shape == (32,) and len(set(ids)) == 32
    assert sorted(Counter(source_ids).values()) == [16,16]
    draw = np.random.default_rng(SEED).integers(0,32,size=(DRAWS,32))
    report = dict(status='prospective_frozen_analysis', protocol_sha256=sha(OUT / 'protocol.frozen.json'),
                  costs_archive_sha256=sha(path), profiles={})
    for profile in ('input1','input4'):
        keys = {name:profile+'.'+name+'.cost' for name in ORIGINS}
        if not all(key in z for key in keys.values()):
            assert profile == 'input4'
            continue
        values = {name:np.asarray(z[key],np.int64).reshape(32,16) for name,key in keys.items()}
        assert all(np.all(x>0) for x in values.values())
        baseline = values[PRIMARY]
        base_boot = baseline[draw].reshape(DRAWS,512)
        base_mean, base_p99 = base_boot.mean(1), np.quantile(base_boot,.99,axis=1,method='linear')
        result = dict(fixed_primary_baseline=PRIMARY, plans={})
        for name,cost in values.items():
            difference = baseline-cost
            boot = cost[draw].reshape(DRAWS,512)
            boot_mean,boot_p99 = boot.mean(1),np.quantile(boot,.99,axis=1,method='linear')
            ci = lambda x:np.quantile(x,[.025,.975],method='linear').tolist()
            perprompt = [dict(prompt_id=str(ids[i]),source_id=str(source_ids[i]),**stat(cost[i]),
                              max_ns=int(cost[i].max()),paired_mean_gain_ns=float(difference[i].mean()),
                              paired_p99_gain_ns=float(np.quantile(baseline[i],.99,method='linear')-np.quantile(cost[i],.99,method='linear')))
                         for i in range(32)]
            result['plans'][name] = dict(**stat(cost), mean_gain_pct=100*(1-cost.mean()/baseline.mean()),
                p99_gain_pct=100*(1-np.quantile(cost,.99,method='linear')/np.quantile(baseline,.99,method='linear')),
                improved_queries=int((difference>0).sum()),tied_queries=int((difference==0).sum()),regressed_queries=int((difference<0).sum()),
                max_regression_ns=int(max(0,-difference.min())),perprompt=perprompt,
                by_source={source:stat(cost[source_ids==source]) for source in sorted(set(source_ids))},
                descriptive_cluster_CI95=dict(mean_gain_ns=ci(base_mean-boot_mean),p99_gain_ns=ci(base_p99-boot_p99),
                                             mean_gain_pct=ci(100*(1-boot_mean/base_mean)),p99_gain_pct=ci(100*(1-boot_p99/base_p99))))
        report['profiles'][profile]=result
    assert 'input1' in report['profiles']
    target=OUT/'analysis';target.mkdir(exist_ok=False)
    dump(target/'report.json',report)
    print(json.dumps(dict(report=str(target/'report.json'),sha256=sha(target/'report.json'))))


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=('freeze','analyze'))
    parser.add_argument('--costs-path')
    args=parser.parse_args()
    if args.mode=='freeze':freeze()
    else:
        assert args.costs_path
        analyze(args.costs_path)
