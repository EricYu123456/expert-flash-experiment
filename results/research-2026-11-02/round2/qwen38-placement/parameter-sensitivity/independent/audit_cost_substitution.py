#!/usr/bin/env python3
"""Small independent duration/count/trace audit; never reruns a campaign."""
import ctypes
import hashlib
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
DEST = OUT.parent
BASE = DEST.parent
ROOT = BASE.parents[3]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ceildiv(a, b):
    return (a+b-1)//b


def audit_trace(doc):
    profile = doc['profile']
    trace = doc['trace']
    nodes = {n['index']: n for n in trace}
    jobs = [j for phase in ('gate_up', 'down') for item in doc['items'] for j in item[phase]]
    command_for_job = {n['job']: n for n in trace if n['acquire']}
    output_for_job = {n['job']: n for n in trace if n['release']}
    assert len(jobs) == len(command_for_job) == len(output_for_job)
    work = Counter()
    intervals = defaultdict(list)
    by_kind = Counter()
    for n in trace:
        kind = n['kind']
        got = n['end_ns']-n['start_ns']
        if kind in ('initial_host_submission', 'expert_down_host_submission'):
            expected = profile['host_ns']
        elif kind == 'matrix_command':
            expected = profile['command_ns']
        elif kind in ('input_channel_reload', 'nonpreemptive_native_compute', 'output_channel'):
            job = jobs[n['job']]
            r, c = job['rows'], job['columns']
            pages = ceildiv(r*c, 16384)
            emitted = pages*16384//c
            expected = {'input_channel_reload': ceildiv(c,2),
                        'nonpreemptive_native_compute': ceildiv(r*c,65536)*profile['read_ns'],
                        'output_channel': emitted*2}[kind]
        elif kind == 'output_pcie':
            # Its only predecessor is the output-channel node carrying job ID.
            job = jobs[nodes[n['deps'][0]]['job']]
            emitted = ceildiv(job['rows']*job['columns'],16384)*16384//job['columns']
            expected = ceildiv(emitted*4,8)
        else:
            expected = {'common_input_to_controller':320,
                        'private_input_to_controller':80,
                        'host_activation':0,'host_combine':0,
                        'whole_expert_gu_host_ready':0,
                        'gu_partial_reconstruction':1000,
                        'down_partial_reconstruction':1000}[kind]
        assert got == expected, (kind, got, expected)
        assert all(nodes[d]['end_ns'] <= n['start_ns'] for d in n['deps'])
        work[n['resource']] += got
        by_kind[kind] += 1
        if got:
            intervals[n['resource']].append((n['start_ns'],n['end_ns']))
    for resource, seq in intervals.items():
        ordered = sorted(seq)
        assert all(a[1] <= b[0] for a,b in zip(ordered,ordered[1:])),resource
    locks = defaultdict(list)
    for job, command in command_for_job.items():
        release = output_for_job[job]
        assert command['chip'] == release['chip']
        locks[command['chip']].append((command['start_ns'],release['end_ns']))
    for chip, seq in locks.items():
        ordered = sorted(seq)
        assert all(a[1] <= b[0] for a,b in zip(ordered,ordered[1:])),chip
    assert doc['makespan_ns'] == max(n['end_ns'] for n in trace)
    commands = sorted(command_for_job.values(), key=lambda n:(n['start_ns'],n['index']))
    idle = sum(b['start_ns']-a['end_ns'] for a,b in zip(commands,commands[1:]))
    return dict(command_count=len(commands),
                routed_commands=sum(n['expert'] < 10 for n in commands),
                shared_commands=sum(n['expert'] == 10 for n in commands),
                controller_work_ns=work['controller'],controller_internal_idle_ns=idle,
                controller_first_start_ns=commands[0]['start_ns'],
                controller_last_end_ns=commands[-1]['end_ns'],
                tail_after_last_command_ns=doc['makespan_ns']-commands[-1]['end_ns'],
                makespan_ns=doc['makespan_ns'],kinds=dict(by_kind))


def main():
    protocol_path = DEST/'protocol.frozen.json'
    summary_path = DEST/'summary.json'
    protocol = json.loads(protocol_path.read_text())
    summary = json.loads(summary_path.read_text())
    assert sha(protocol_path) == summary['protocol_sha256']
    inputs = [{'path':p,'expected':h,'actual':sha(p)} for p,h in protocol['source_inputs_sha256'].items()]
    assert all(x['expected']==x['actual'] for x in inputs)
    frozen_path = BASE/'native-full48/plans.frozen.json'
    frozen = json.loads(frozen_path.read_text())
    assert len(frozen['records']) == 240
    assert all(sha(r['plan']) == r['plan_sha256'] for r in frozen['records'])
    templates = json.loads((BASE/'phase-fit3/templates.json').read_text())
    inv = {phase:{x['label']:x for x in candidates} for phase,candidates in templates.items()}
    plan_counts = []
    for rec in frozen['records']:
        if rec['case'] not in ('whole_row','overlap_best'):
            continue
        plan = json.loads(Path(rec['plan']).read_text())
        expert_counts = np.zeros(512,dtype=int)
        shared = 0
        for phase in ('gate_up','down'):
            p = plan['plans'][phase]
            assert len(p['labels']) == 512
            v = inv[phase][p['label']]['variants']
            expert_counts += [sum(j['rows']>0 for j in v[color]) for color in p['labels']]
            s = plan['fixed_shared'][phase]
            shared += sum(j['rows']>0 for j in inv[phase][s['label']]['variants'][s['color']])
        assert expert_counts.min() == expert_counts.max()
        expected_per_expert = 32 if rec['case']=='whole_row' else 26
        assert int(expert_counts[0]) == expected_per_expert and shared == 32
        plan_counts.append(dict(layer=rec['layer'],case=rec['case'],routed_commands_per_expert=int(expert_counts[0]),
                                shared_commands=shared,top10_total=10*int(expert_counts[0])+shared))
    assert len(plan_counts)==96
    lib = ctypes.CDLL(str(DEST/'native_parameters.so'))
    p32=np.ctypeslib.ndpointer(dtype=np.int32,flags='C_CONTIGUOUS')
    p64=np.ctypeslib.ndpointer(dtype=np.int64,flags='C_CONTIGUOUS')
    lib.eval_native_ffn_parameters.argtypes=[p32,p32,p32]+[ctypes.c_int]*3+[ctypes.c_int64]*2+[ctypes.c_int]*2+[ctypes.c_int64]*3+[p64]
    lib.eval_native_ffn_parameters.restype=None
    traces = []
    structural = {}
    profiles_to_check=['command_250','command_10000','host_10000','read_56000']
    selections=[(0,c,p) for c in ('whole_row','overlap_best') for p in profiles_to_check]
    selections += [(layer,c,'command_10000') for layer in (23,47) for c in ('whole_row','overlap_best')]
    for layer,case,profile_name in selections:
        path=DEST/f'L{layer}.{case}.{profile_name}.trace.json'
        doc=json.loads(path.read_text())
        checked=audit_trace(doc)
        # Reconstruct only these exact physical descriptors from saved trace
        # items, independent of the original encode()/route-loading path.
        descriptors=np.zeros((513,2,16,4),dtype=np.int32)
        partial=np.zeros((513,2),dtype=np.int32)
        for ei,item in enumerate(doc['items']):
            logical=ei if ei<10 else 512
            for phase,name in enumerate(('gate_up','down')):
                partial[logical,phase]=item['gu_partial' if phase==0 else 'down_partial']
                for slot,j in enumerate(item[name]):
                    descriptors[logical,phase,slot]=(j['chip'],j['rows'],j['columns'],ceildiv(j['rows']*j['columns'],65536))
        route=np.arange(10,dtype=np.int32).reshape(1,10)
        result=np.zeros(1,dtype=np.int64)
        p=doc['profile']
        lib.eval_native_ffn_parameters(descriptors,partial,route,512,1,10,1000,0,0,1,
                                      p['command_ns'],p['host_ns'],p['read_ns'],result)
        assert int(result[0])==doc['makespan_ns']
        identity=[{k:v for k,v in n.items() if k not in ('start_ns','end_ns')} for n in doc['trace']]
        if layer==0:
            value=dict(selected=doc['selected'],items=doc['items'],nodes=identity)
            if case in structural:
                assert value==structural[case]
            else:
                structural[case]=value
        traces.append(dict(file=str(path),sha256=sha(path),compiled_one_event_matches=True,**checked))
    closed_form=[]
    for case,per_layer_commands in [('whole_row',352),('overlap_best',292)]:
        full_values=[]
        for layer in range(48):
            with np.load(DEST/f'L{layer}.{case}.npz') as data:
                profiles=data['profiles'].tolist()
                assert profiles==[p['label'] for p in protocol['profiles']]
                assert np.array_equal(data['external_indices'],protocol['external_indices'])
                v=data['cost'][profiles.index('command_10000')]
                assert v.shape==(256,) and np.all(v==per_layer_commands*10000+30575)
                full_values.append(v)
        total=np.stack(full_values).sum(axis=0)
        assert np.all(total==per_layer_commands*48*10000+48*30575)
        s=next(x for x in summary['totals'] if x['case']==case and x['label']=='command_10000')
        assert s['mean_ns']==int(total[0])==s['p99_ns']
        closed_form.append(dict(case=case,commands_per_layer=per_layer_commands,
                                commands_48_layers=per_layer_commands*48,
                                controller_floor_48_layers_ns=per_layer_commands*48*10000,
                                per_layer_noncontroller_tail_ns=30575,
                                full_tail_ns=48*30575,observed_full_ns=int(total[0]),
                                all_48x256_saved_values_equal=True))
    for t in traces:
        if '.command_10000.' in t['file']:
            assert t['controller_internal_idle_ns']==0
            assert t['controller_first_start_ns']==2000
            assert t['tail_after_last_command_ns']==28575
    gains=[]
    for name in ('command_250','command_4000','command_5000','command_10000'):
        s={x['case']:x for x in summary['totals'] if x['label']==name}
        gains.append(dict(profile=name,whole_ms=s['whole_row']['mean_ns']/1e6,
                          overlap_ms=s['overlap_best']['mean_ns']/1e6,
                          whole_relative_latency_reduction_vs_overlap=1-s['whole_row']['mean_ns']/s['overlap_best']['mean_ns']))
    report=dict(status='PASS small independent frozen-plan parameter substitution/cost sanity audit',
                created_utc=datetime.now(timezone.utc).isoformat(),
                protocol_sha256=sha(protocol_path),summary_sha256=sha(summary_path),
                verified_protocol_inputs=inputs,all_240_plan_hashes_match=True,
                command_counts_all_96_whole_overlap_plans=plan_counts,
                traces_checked=traces,closed_controller_forms=closed_form,conditional_curve=gains,
                profile_substitution_read_audit={
                  'command_ns':'Only resource1 acquiring matrix_command duration; every matrix job charged equally. Function receives no case label.',
                  'host_ns':'Only initial host submission and each expert_down_host_submission; same initial plus11 down submissions in each checked FFN. No activation or reconstruction duration changed.',
                  'read_ns':'Only each nonpreemptive native compute duration=waves*read_ns. Wave/job geometry unchanged.',
                  'unchanged':'Same descriptors/routes, 1us conditional partial reconstructions, zero activation/combine, requested-page fringe mode1, rawI8/A8 input transfer, assumed4B output transfer, channel/chip assignment, priorities, chip lock through output-channel drain.',
                  'python_guard':'Python duration==2000 host substitution is safe for this campaign because activation/combine=0 and reconstruction=1000; a generic graph containing another2000ns host kind would require kind-based replacement.'},
                tail_derivation='Each layer: initial host2000 + continuous serialized command_count*10000 + last shared down input320 + one read28000 +102 emitted outputs*2ns +PCIe ceil(102*4/8)=51. Thus noncontroller overhead2000+320+28000+204+51=30575ns, and48 layers give1467600ns.',
                scope='Read code, hash240 plans, count96 relevant layouts, inspect12 saved traces, execute12 one-event C++ checks, and reduce saved10us vectors. No refit, new route opening, or full campaign rerun.',
                interpretation='250ns command timing is uncalibrated. Whole-row configuration benefits at low modeled dispatch cost but issues2880 more matrix commands per48-layer FFN. At10us, those commands explain exactly28.8ms loss. This is a conditional software cost result, not a universal hardware speedup or native device timing measurement.')
    target=OUT/'cost_sanity.json'
    target.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(output=str(target),sha256=sha(target),traces_checked=len(traces),
                          closed_controller_forms=closed_form,conditional_curve=gains),indent=2))


if __name__=='__main__':main()
