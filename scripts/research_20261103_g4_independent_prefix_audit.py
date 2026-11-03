#!/usr/bin/env python3
"""Independent contract, LP-certificate, ownership and native-event audit.

No imports of the producer, compiler backend, or native replay implementation.
"""
import hashlib
import heapq
import json
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/'results/research-2026-11-02/round2/g4-elastic-prefix-screen/work-v3-independent-prefix'
NATIVE = OUT/'native-v1'
AUDIT = OUT/'independent'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stats(x):
    return dict(mean_ns=float(np.mean(x)), p99_ns=float(np.percentile(x, 99)))


def job(n, c=1536):
    if not n:
        return (0, 0, 0, 0)
    pages = (int(n)*int(c)+16383)//16384
    waves = (pages+3)//4
    emit = pages*16384//int(c)
    return (250+(int(c)+1)//2+28000*waves+2*emit, pages, waves, emit)


def runtime_rows(original, entry, u):
    rr = original.copy()
    for i, c in enumerate(entry['sources']):
        rr[c] = 128 if u[i] else original[c]
    for i, c in enumerate(entry['low_destinations']):
        rr[c] = 0 if u[i] else 40
    hd = entry['high_destination']
    rr[hd] = original[hd] if not u[7] else entry['high_prefix_rows'] if not all(u) else 0
    return rr


def independent_graph(plan, selected):
    """Generic ready-list scheduler, deliberately not the C++ command queues."""
    nodes = []
    next_job = 0
    def add(r, duration, deps=(), phase=0, expert=-1, chip=-1, jid=-1, acq=False, rel=False):
        nodes.append((r, int(duration), tuple(map(int, deps)), phase, expert, chip, jid, acq, rel))
        return len(nodes)-1
    def matrix(d, expert, phase, submit, upload):
        nonlocal next_job
        c, nr, nc, pages, waves = map(int, d)
        jid = next_job
        next_job += 1
        emit = pages*16384//nc
        cmd = add(1,250,[submit],phase,expert,c,jid,True)
        inp = add(3+c%8,(nc+1)//2,[cmd,upload],phase,expert,c,jid)
        comp = add(11+c,waves*28000,[inp],phase,expert,c,jid)
        drain = add(3+c%8,emit*2,[comp],phase,expert,c,jid,False,True)
        return add(2,(emit*4+7)//8,[drain],phase,expert)
    submit = add(0,2000)
    upload = add(2,192,[submit])
    ready = []
    for ordinal, e in enumerate(selected):
        out = [matrix(d,ordinal,0,submit,upload) for d in plan['jobs'][slice(*plan['offsets'][e,0])]]
        ready.append(add(27,0,out,expert=ordinal))
    outputs = []
    for ordinal, e in enumerate(selected):
        act = add(0,0,[ready[ordinal]],1,ordinal)
        ds = add(0,2000,[act],1,ordinal)
        nc = int(plan['input_columns'][e,1])
        inp = add(2,(nc+7)//8,[ds],1,ordinal)
        outputs.extend(matrix(d,ordinal,1,ds,inp) for d in plan['jobs'][slice(*plan['offsets'][e,1])])
    add(0,0,outputs,1)
    remaining = [len(n[2]) for n in nodes]
    followers = [[] for n in nodes]
    pending = [[] for r in range(28)]
    critical = np.zeros(len(nodes),np.int64)
    resources, chipwork = np.zeros(28,np.int64), np.zeros(16,np.int64)
    for i,n in enumerate(nodes):
        for dep in n[2]:
            followers[dep].append(i)
        if not n[2]:
            pending[n[0]].append(i)
        critical[i] = n[1]+max((critical[d] for d in n[2]),default=0)
        resources[n[0]] += n[1]
        if n[6] >= 0:
            chipwork[n[5]] += n[1]
    busy, owner = [False]*28, [-1]*16
    begin, end = np.zeros(len(nodes),np.int64), np.zeros(len(nodes),np.int64)
    events = []
    now = done = 0
    while done < len(nodes):
        for r in range(28):
            if busy[r]:
                continue
            eligible = [i for i in pending[r] if nodes[i][6]<0 or
                        (owner[nodes[i][5]]<0 if nodes[i][7] else owner[nodes[i][5]]==nodes[i][6])]
            if not eligible:
                continue
            i = min(eligible,key=lambda i:(nodes[i][3],nodes[i][4],i))
            pending[r].remove(i)
            n = nodes[i]
            if n[7]:
                owner[n[5]] = n[6]
            begin[i] = now
            busy[r] = True
            heapq.heappush(events,(now+n[1],i))
        assert events
        now = events[0][0]
        while events and events[0][0] == now:
            _,i = heapq.heappop(events)
            n = nodes[i]
            end[i] = now
            busy[n[0]] = False
            if n[8]:
                assert owner[n[5]] == n[6]
                owner[n[5]] = -1
            done += 1
            for j in followers[i]:
                remaining[j] -= 1
                if not remaining[j]:
                    pending[nodes[j][0]].append(j)
    assert owner == [-1]*16
    trace = np.asarray([[n[0],n[1],begin[i],end[i],n[3],n[4],n[5],n[6]] for i,n in enumerate(nodes)],np.int64)
    bound = max(int(resources.max()),int(critical.max()),2000+int(chipwork.max()),2000+int(resources[1]))
    return now,bound,trace


def main():
    AUDIT.mkdir(exist_ok=False)
    for pp in [OUT/'protocol.frozen.json',NATIVE/'protocol.frozen.json']:
        p = json.loads(pp.read_text())
        for item in p['inputs']:
            assert sha(item['path']) == item['sha256']
            if 'snapshot' in item:
                assert sha(item['snapshot']) == item['sha256']
    def read_arrays(path):
        with np.load(path) as z:
            return {k:z[k] for k in z.files}
    # Decode compressed arrays once, rather than decompressing an entire array
    # at each inner-loop element. This changes audit I/O only, not any check.
    a = read_arrays(OUT/'contract_arrays.npz')
    lp = read_arrays(OUT/'LP_arrays.npz')
    greedy = read_arrays(OUT/'greedy_arrays.npz')
    native = read_arrays(NATIVE/'arrays.npz')
    owners = read_arrays(NATIVE/'query_row_owners.npz')
    ledger = json.loads((OUT/'rowid_ledgers.json').read_text())
    rows,routes = a['rows'],a['routes']
    v1 = read_arrays(OUT.parent/'work-v1/work_arrays.npz')
    expected_table = np.zeros_like(a['table'])
    for e in range(65):
        for phase in range(2):
            nc = 1536 if phase==0 else 1024 if e==64 else 512
            for chip,nr in enumerate(rows[e,phase]):
                expected_table[e,phase,chip] = job(int(nr),nc)[0]
    assert np.array_equal(expected_table,a['table'])
    assert np.array_equal(expected_table[routes].sum((1,2))+expected_table[64].sum(0),a['base'])
    assert np.array_equal(routes,owners['routes']) and np.array_equal(owners['logical_row_ids'],np.arange(1024))
    D = np.zeros_like(a['D'])
    Z = np.zeros_like(a['Z'])
    view_checks = 0
    for layout in range(4):
        for e,entry in enumerate(ledger[layout]['experts']):
            ss,ll,hd = entry['sources'],entry['low_destinations'],entry['high_destination']
            assert len(set(ss))==8 and len(set(ll))==7 and set(ss).isdisjoint(ll)
            assert sorted(ss+ll+[hd])==list(range(16))
            high = np.flatnonzero(rows[e,0]>40)
            assert hd==high[v1['selected_layout'][e]]
            assert all(rows[e,0,c]>40 for c in ss+[hd]) and all(rows[e,0,c]==40 for c in ll)
            original = {c:list(range(int(rows[e,0,:c].sum()),int(rows[e,0,:c+1].sum()))) for c in range(16)}
            assert {int(c):rr for c,rr in entry['B_original_rows'].items()} == original
            image = {int(c):rr for c,rr in entry['resident_prefix_image'].items()}
            pn = int(rows[e,0,hd]+rows[e,0,ss[7]]-128)
            assert pn==entry['high_prefix_rows'] and pn in (36,37,38)
            at = 0
            for i,c in enumerate(ss[:7]):
                n = 128-int(rows[e,0,c])-40
                piece = original[hd][at:at+n]
                at += n
                assert piece==entry['prefix_piece_rows'][i]
                assert image[c]==original[c]+original[ll[i]]+piece
                D[layout,e,i,c] = 28086
                D[layout,e,i,ll[i]] = -29102
            assert at==pn and image[ss[7]]==original[ss[7]]+original[hd][pn:]
            assert all(image[c]==original[c] for c in ll+[hd])
            assert all(len(image[c])==a['resident_rows'][e,0,c] for c in range(16))
            donors = [rr for c in ll+[hd] for rr in original[c]]
            assert Counter(rr for band in image.values() for rr in band)==Counter({rr:2 if rr in donors else 1 for rr in range(1024)})
            D[layout,e,7,ss[7]] = 28086
            D[layout,e,7,hd] = -28086
            Z[e,hd] = -29102
            for mask in range(256):
                u = [(mask>>i)&1 for i in range(8)]
                rr = runtime_rows(rows[e,0],entry,u)
                computed = [r for c in range(16) for r in image[c][:int(rr[c])]]
                assert set(computed)==set(range(1024))
                counts = [0,0,0,int(rr.sum()),0,1024]
                work = np.zeros(16,np.int64)
                for c,n in enumerate(rr):
                    if n:
                        w,pg,wa,em = job(int(n))
                        work[c] = w
                        counts[0]+=1;counts[1]+=pg;counts[2]+=wa;counts[4]+=em
                expected = a['table'][e,0]+np.einsum('i,ic->c',u,D[layout,e])+int(all(u))*Z[e]
                assert np.array_equal(work,expected)
                assert np.array_equal(counts,a['view_counts'][layout,e,mask])
                view_checks += 1
    assert np.array_equal(D,a['D']) and np.array_equal(Z,a['Z'])

    # Independent LP box-dual validation. This does not trust optimality/status
    # or the primal objective: the saved dual alone proves a necessary bound.
    dual_checks = 0
    for layout in range(4):
        for q,selected in enumerate(routes):
            M = np.zeros((70,55))
            rhs = np.zeros(70)
            M[:16,:48] = D[layout,selected].reshape(48,16).T/28000
            M[:16,48:54] = Z[selected].T/28000
            M[:16,-1] = -1
            rhs[:16] = -a['base'][q]/28000
            for e in range(6):
                for i in range(8):
                    M[16+8*e+i,8*e+i] = -1
                    M[16+8*e+i,48+e] = 1
                M[64+e,8*e:8*e+8] = 1
                M[64+e,48+e] = -1
                rhs[64+e] = 7
            y = lp['dual_multipliers'][layout,q]
            assert np.all(y<=0) and -y[:16].sum()<=1
            c = np.zeros(55);c[-1]=1
            residual = c-M.T@y
            assert residual[-1]>=0
            bound = (y@rhs+np.minimum(residual[:54],0).sum())*28000
            assert int(np.floor(bound-1e-4))==lp['necessary_work_LB'][layout,q]
            assert -1e-4<=lp['primal_work'][layout,q]-bound<0.1
            dual_checks += 1

    # Every greedy endpoint can be accounted from actual row jobs, rather than
    # assuming unique rows==computed rows or importing the delta implementation.
    for layout in range(4):
        for start in range(2):
            for q,selected in enumerate(routes):
                loads = a['table'][64].sum(0)+a['table'][selected,1].sum(0)
                for ordinal,e in enumerate(selected):
                    rr = runtime_rows(rows[e,0],ledger[layout]['experts'][e],greedy['bits'][layout,start,q,ordinal])
                    loads += [job(int(n))[0] for n in rr]
                assert loads.max()==greedy['values'][layout,start,q]
    assert np.array_equal(greedy['feasible_work'],greedy['values'].min(1))
    assert np.all(lp['necessary_work_LB']<=greedy['feasible_work'])
    assert int(np.argmin(greedy['feasible_work'].mean(1)))==3

    # Full-query consumed ownership and emitted-row billing.
    counters = np.zeros((512,2,7),np.int64)
    logical_checks = 0
    for q,selected in enumerate(routes):
        for ordinal,e in enumerate(np.r_[selected,64]):
            if e<64:
                entry = ledger[3]['experts'][e]
                rr = runtime_rows(rows[e,0],entry,greedy['feasible_bits'][3,q,ordinal])
                image = {int(c):bb for c,bb in entry['resident_prefix_image'].items()}
                for row_id,(c,position) in enumerate(owners['owner_map'][q,ordinal]):
                    assert 0<=position<rr[c] and image[int(c)][int(position)]==row_id
                logical_checks += 1
            else:
                rr = rows[e,0]
            for phase,band in [(0,rr),(1,rows[e,1])]:
                nc = 1536 if phase==0 else 1024 if e==64 else 512
                for n in band:
                    if n:
                        w,pg,wa,em = job(int(n),nc)
                        counters[q,phase] += [1,int(n)*nc,pg,wa,nc,em*4,(nc+1)//2+em*2]
    assert np.array_equal(counters,native['counters'])
    assert np.array_equal(native['greedy_work'],greedy['feasible_work'][3])
    assert np.array_equal(native['baseline'],a['baseline_native'])
    trace_checks = []
    for q in (0,255,511):
        for tag in ('B','allA','greedy'):
            plan = np.load(NATIVE/f'{tag}_q{q}.npz')
            assert np.array_equal(plan['total_rows'],plan['rows'].sum(2))
            now,lb,trace = independent_graph(plan,np.r_[routes[q],64])
            assert now==plan['cost'][0] and lb==plan['lower_bound'][0]
            assert np.array_equal(trace,plan['trace'])
            for desc,alias in zip(plan['jobs'],plan['aliases']):
                c,nr,nc,pg,wa = map(int,desc)
                ac,block,wl,reserve,storedpages,plane = map(int,alias)
                assert ac==c and plane==0 and wa<=reserve and pg<=storedpages and wl+wa<=192
            trace_checks.append(dict(case=tag,query=q,nodes=len(trace),cost_ns=now))

    original,candidate = native['baseline'],native['greedy']
    gain = dict(mean_pct=100*(1-candidate.mean()/original.mean()),
                p99_pct=100*(1-np.percentile(candidate,99)/np.percentile(original,99)))
    assert gain['mean_pct']<2 and gain['p99_pct']<5
    audit = dict(status='PASS_bounded_native_gate_failed',
                 source_sha256=sha(__file__), work_protocol_sha256=sha(OUT/'protocol.frozen.json'),
                 native_protocol_sha256=sha(NATIVE/'protocol.frozen.json'),
                 checks=dict(exhaustive_expert_views=view_checks, LP_box_dual_certificates=dual_checks,
                             actual_row_greedy_endpoints=4*2*512, unique_original_row_maps=logical_checks,
                             unique_row_assignments=logical_checks*1024,
                             independent_native_schedules=trace_checks),
                 baseline=stats(original), greedy=stats(candidate), improvement=gain,
                 allA=stats(native['allA']),
                 stored_copy_rows=23202, useful_copy_bytes=35638272,
                 extra_waves_per_layer=512, extra_programmed_bytes_per_layer=33554432,
                 computed_mean_GU_rows_per_query=float(counters[:,0,1].mean()/1536),
                 limits=['This checks row identities/counters/event costs, not real numerical output ABI or modelquality.',
                         'Native timing is synthetic; selector/hostscatter/extra runtime software cost is zero optimistic sensitivity.',
                         'Gate failed for selected fixedpairing and tested greedy selector, not all integer masks or layouts.',
                         'LP still permits about4.9% mean opportunity, so the broader independentview family is not ruled out.',
                         'No selectorbench, samebudgetcopy controls, heldout or all-layer expansion performed.'])
    dump = lambda path,value:Path(path).write_text(json.dumps(value,indent=2)+'\n')
    dump(AUDIT/'audit.json',audit)
    report = f'''The independent-prefix TRAIN screen and one authorized native sensitivity are sealed. Each expert retains the same eight source chips and 33-wave resident image as v1; seven source suffixes contain low40 plus a high-prefix piece and the eighth contains the high suffix. Its 256 valid read views are not extra coefficient replicas. Every selected expert still waits for all GU jobs and the full-vector A8 quantization barrier before private Down.

All 65,536 expert-view cases preserve the original 1,024 unique GU rows. Meaningful computed rows can reach 1,062 and page-emitted rows 1,066 because high-prefix duplicates and fringe are truly read/drained. The lowerer declares actual computed total_rows and leaves the original backend guard intact; a separate ownership array maps exactly 1,024 consumed row IDs. Independent checks reproduced 2,048 LP box-dual certificates, all 4,096 greedy endpoint row-job costs, 3,072 selected expert ownership maps and nine full native schedules (B/allA/greedy at q0/255/511, every node/start/end and bound exact).

| Native sensitivity | Mean (ms) | Empirical P99 (ms) |
|---|---:|---:|
| Frozen B incumbent | {original.mean()/1e6:.9f} | {np.percentile(original,99)/1e6:.9f} |
| Selected layout3 greedy bits | {candidate.mean()/1e6:.9f} | {np.percentile(candidate,99)/1e6:.9f} |
| All-A endpoint | {native['allA'].mean()/1e6:.9f} | {np.percentile(native['allA'],99)/1e6:.9f} |

The candidate improves modeled mean {gain['mean_pct']:.6f}% and P99 {gain['p99_pct']:.6f}% with zero selector/scatter/software overhead. It misses the 2% mean / 5% tail expansion gate, so selector benchmarks and strong same-budget replica controls were not launched. Per-query comparisons: 491 improved, four equal, 17 worsened; maximum regression 2,450 ns. The all-A endpoint loses substantially, demonstrating that lower aggregate work does not remove readiness/depth costs.

Continuous LP lower bounds permit about 4.67–4.99% mean opportunity; the selected layout3 LP permits 4.914%. The tested two-start/two-pass greedy feasible work corresponds to only 2.353% optimistic host+work headroom before full native scheduling; the replay realizes 1.725%. Therefore this is a finite selector/layout no-go, not a proof that every independent prefix-view selector fails. No small integer oracle was run after native gate failure. Source/row pairing remains fixed offline and no heldout was read.

The copies remain 23,202 rows / 35,638,272 useful bytes / 512 extra resident waves (32 MiB programmed) per layer. Prefix storage savings are separate from adaptive-view latency; a full duplicate image with identical read descriptors has the same modeled service. Ordinary-read page-fringe output and host scatter remain assumed contracts rather than device or quality measurements.
'''
    (AUDIT/'report.md').write_text(report)
    files = [OUT/'protocol.frozen.json',OUT/'contract.json',OUT/'contract_arrays.npz',
             OUT/'rowid_ledgers.json',OUT/'LP_report.json',OUT/'LP_arrays.npz',
             OUT/'greedy_report.json',OUT/'greedy_arrays.npz',NATIVE/'protocol.frozen.json',
             NATIVE/'report.json',NATIVE/'arrays.npz',NATIVE/'query_row_owners.npz',
             AUDIT/'audit.json',AUDIT/'report.md',Path(__file__)]
    dump(AUDIT/'completion.json',dict(status='sealed_bounded_native_gate_failed',files=[dict(path=str(f),sha256=sha(f)) for f in files]))
    print(json.dumps(dict(status=audit['status'],improvement=gain,
                         completion_sha256=sha(AUDIT/'completion.json')),indent=2))


if __name__=='__main__':
    main()
