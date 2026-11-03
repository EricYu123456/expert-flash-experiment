#!/usr/bin/env python3
"""Train-only DeepSeek shared-GU fixed overlapping-range replication oracle.

This is a necessary NAND/chip-occupancy bound, NOT a schedule or hardware run.
Every executed shard contains complete output rows.  No weights move online.
The exact integer shared-down work/depth/job-count frontier is included.
"""
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import struct
import time
import warnings

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02'
SOURCE = BASE / 'projection-study/deepseek.frozen_plan.json'
DEST = BASE / 'round2/deep-shared-boundary'
WAVE = 65536
SG_R, SG_C, SD_R, SD_C = 5632, 2048, 2048, 2816
TILE = 32


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def stats(a):
    a = np.asarray(a)
    return dict(mean=float(a.mean()), p95=float(np.percentile(a, 95)),
                p99=float(np.percentile(a, 99)), min=int(a.min()), max=int(a.max()))


def waves(rows, columns):
    return (np.asarray(rows, dtype=np.int64) * columns + WAVE - 1) // WAVE


def native(rows, columns):
    rows = np.asarray(rows, dtype=np.int64)
    return waves(rows, columns) * 28000 + np.where(rows > 0, 250 + (columns + 1)//2 + 2*rows, 0)


def shape_receipt(doc):
    p = Path(doc['metadata_path'])
    assert sha(p) == doc['metadata_sha256']
    m = json.loads(p.read_text())
    ts = [x for x in m['tensors'] if x.get('ffn_kind') == 'always_active_shared_experts']
    assert len(ts) == 78
    expected = {'gate': [2048, 2816], 'up': [2048, 2816], 'down': [2816, 2048]}
    for t in ts:
        assert t['shape_ggml'] == expected[t['shared_projection']]
    base = Path('/home/nfs_client/ymtuan2/models/DeepSeek-V2-Lite-Chat')
    index = json.loads((base/'model.safetensors.index.json').read_text())
    keys = [k for k in index['weight_map'] if '.shared_experts.' in k]
    headers = []
    found = 0
    for filename in sorted({index['weight_map'][k] for k in keys}):
        with (base/filename).open('rb') as f:
            n = struct.unpack('<Q', f.read(8))[0]
            raw = f.read(n)
        h = json.loads(raw)
        current = {k: h[k] for k in keys if k in h}
        for k, t in current.items():
            assert t['shape'] == ([2048, 2816] if '.down_proj.' in k else [2816, 2048])
        found += len(current)
        headers.append(dict(path=str(base/filename), header_bytes=n,
                            header_sha256=hashlib.sha256(raw).hexdigest(), tensors=current))
    assert found == 78
    config = base/'config.json'
    c = json.loads(config.read_text())
    assert (c['hidden_size'], c['moe_intermediate_size'], c['n_shared_experts']) == (2048, 1408, 2)
    return dict(metadata_path=str(p), metadata_sha256=sha(p), config_path=str(config),
                config_sha256=sha(config), safetensors_index_sha256=sha(base/'model.safetensors.index.json'),
                shared_tensor_count=78, tensor_header_receipts=headers,
                nominal_i8_geometry=dict(gu_rows=SG_R, gu_columns=SG_C, down_rows=SD_R, down_columns=SD_C),
                numeric_scope='Shapes only verified in GGUF Q8_0 and original BF16. I8/A8 timing remains a separate nominal service contract.')


def down_frontier():
    """Exact capacity DP: minimum waves for each max-depth and active count.

    For each number n of positive row shards and depth d, DP maximizes whole
    row capacity at a given total wave count.  Exact assigned rows are then
    materialized within each wave's integer ceil interval.  Pareto dominance
    uses (total waves, max waves, commands); not a phase-only timing proxy.
    """
    records = []
    max_sum = 104
    for depth in range(1, 89):
        dp = np.full((17, max_sum+1), -1, dtype=np.int64)
        parent = np.full_like(dp, -1)
        dp[0, 0] = 0
        for n in range(1, 17):
            for w in range(1, min(depth, max_sum)+1):
                candidate = dp[n-1, :max_sum+1-w] + (w*WAVE//SD_C)
                valid = dp[n-1, :max_sum+1-w] >= 0
                target = dp[n, w:]
                better = valid & (candidate > target)
                target[better] = candidate[better]
                parent[n, w:][better] = w
            feasible = np.flatnonzero(dp[n] >= SD_R)
            if not len(feasible):
                continue
            s = int(feasible[0]); rem = s; ws = []
            for nn in range(n, 0, -1):
                w = int(parent[nn, rem]); ws.append(w); rem -= w
            assert rem == 0
            ws = np.asarray(sorted(ws, reverse=True) + [0]*(16-n))
            lo = np.where(ws > 0, (ws-1)*WAVE//SD_C+1, 0)
            hi = ws*WAVE//SD_C
            assert int(lo.sum()) <= SD_R <= int(hi.sum())
            rr = lo.copy(); need = SD_R-int(rr.sum())
            while need:
                ix = np.flatnonzero(rr < hi)[:need]
                rr[ix] += 1; need -= len(ix)
            assert np.array_equal(waves(rr, SD_C), ws)
            records.append(dict(total_waves=s, max_waves=int(ws.max()), commands=n,
                                rows=rr.tolist(), waves=ws.tolist()))
    unique = {tuple((r[k] for k in ('total_waves','max_waves','commands'))):r for r in records}
    result = []
    for key, record in unique.items():
        if any(all(a <= b for a,b in zip(other,key)) and other != key for other in unique):
            continue
        result.append(record)
    # A smaller job count does NOT dominate a more even per-chip load when
    # overlapped with routed work.  Retain balanced active-n templates as well.
    for active in range(1,17):
        for total in range(88,105):
            ws=np.zeros(16,dtype=np.int64);ws[:active]=total//active
            ws[:total%active]+=1
            lo=np.where(ws>0,(ws-1)*WAVE//SD_C+1,0);hi=ws*WAVE//SD_C
            if lo.sum()>SD_R or hi.sum()<SD_R:continue
            rr=lo.copy();need=SD_R-int(rr.sum())
            while need:
                ix=np.flatnonzero(rr<hi)[:need];rr[ix]+=1;need-=len(ix)
            result.append(dict(total_waves=int(ws.sum()),max_waves=int(ws.max()),
                commands=active,rows=rr.tolist(),waves=ws.tolist(),control='balanced active-n template; retained despite scalar dominance'))
            break
    # The old equal128rows case is an explicit weak-control receipt only.
    rr=np.full(16,128,dtype=np.int64);ws=waves(rr,SD_C)
    result.append(dict(total_waves=int(ws.sum()),max_waves=int(ws.max()),commands=16,
        rows=rr.tolist(),waves=ws.tolist(),control='equal128rows;96waves reference, never sole baseline'))
    result=list({tuple(r['rows']):r for r in result}.values())
    result.sort(key=lambda r:(r['max_waves'],r['total_waves'],r['commands']))
    assert result[0]['max_waves'] == 6 and result[0]['total_waves'] == 89
    assert min(r['total_waves'] for r in result) == 88
    return result


def balanced_shares(demand, total):
    """Exact all floor/ceil shared-total-wave placements via subset zeta.

    If s_i is floor(total/16) or +1, a query pays the extra wave iff an argmax
    routed-demand chip intersects the high-chip set.  All 2^16 sets are scored.
    This is a restricted exact result, not arbitrary integer-placement proof.
    """
    base, high_count = divmod(total, 16)
    maxima = demand.max(axis=1)
    bits = ((demand == maxima[:,None]) * (1 << np.arange(16))).sum(axis=1)
    histogram = np.bincount(bits, minlength=65536).astype(np.int64)
    subset = histogram.copy()
    masks = np.arange(65536)
    for b in range(16):
        target = masks[(masks & (1 << b)) != 0]
        subset[target] += subset[target ^ (1 << b)]
    feasible = np.array([i for i in range(65536) if i.bit_count()==high_count])
    covered = len(demand)-subset[65535 ^ feasible]
    mask = int(feasible[np.argmin(covered)])
    s = np.full(16,base,dtype=np.int64)
    s += ((mask >> np.arange(16)) & 1)
    return s, float(maxima.mean()+base+covered.min()/len(demand))


def static_wave_milp(demand, total, seconds):
    unique, counts = np.unique(demand, axis=0, return_counts=True)
    n = len(unique)
    rr = np.repeat(np.arange(n*16),2)
    cc = np.empty(n*32,dtype=np.int64)
    cc[::2] = np.tile(np.arange(16),n)
    cc[1::2] = 16+np.repeat(np.arange(n),16)
    aa = coo_matrix((np.tile([-1.,1.],n*16),(rr,cc)),shape=(n*16,n+16)).tocsr()
    ss = coo_matrix((np.ones(16),(np.zeros(16),np.arange(16))),shape=(1,n+16)).tocsr()
    seed, restricted = balanced_shares(demand,total)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore',RuntimeWarning)
        res = milp(c=np.r_[np.zeros(16),counts/counts.sum()],
            integrality=np.r_[np.ones(16),np.zeros(n)],
            bounds=Bounds(np.zeros(n+16),np.r_[np.full(16,total),np.full(n,np.inf)]),
            constraints=[LinearConstraint(aa,unique.ravel(),np.inf),LinearConstraint(ss,total,total)],
            options={'time_limit':seconds,'mip_rel_gap':1e-8,'threads':1})
    if res.x is not None:
        other = np.rint(res.x[:16]).astype(np.int64)
        assert other.sum()==total
        if np.maximum(demand+other,0).max(axis=1).mean() < restricted:
            seed=other
    score = float((demand+seed).max(axis=1).mean())
    return seed,dict(total_shared_waves=total,success=bool(res.success),status=int(res.status),
        message=res.message,incumbent_mean_waves=score,dual_lower_bound_mean_waves=float(res.mip_dual_bound),
        absolute_gap_waves=score-float(res.mip_dual_bound),restricted_balanced_exact_mean_waves=restricted,
        shared_wave_counts=seed.tolist(),unique_routed_load_vectors=n)


def fit_gu(demand, sd, start, max_passes=20):
    """Full train mean native-chip occupancy greedy, 32-row one-copy moves.

    Each move changes fixed offline row partition boundaries.  It never moves
    data online and imposes no equal per-chip inventory quota.
    """
    t=np.maximum(np.asarray(start,dtype=np.int64)-waves(sd,SD_C),0)
    while t.sum()!=176:
        delta=1 if t.sum()<176 else -1
        scores=[]
        for c in range(16):
            if delta<0 and t[c]==0:continue
            trial=t.copy();trial[c]+=delta
            scores.append((float((demand+native(trial*TILE,SG_C)+native(sd,SD_C)).max(axis=1).mean()),c))
        t[min(scores)[1]]+=delta
    for _ in range(max_passes):
        occupancy=demand+native(t*TILE,SG_C)+native(sd,SD_C)
        value=float(occupancy.max(axis=1).mean())
        # An unaffected maximum can be found by deleting the two changed chips.
        best=(value,None)
        for a in range(16):
            if t[a]==0:continue
            for b in range(16):
                if a==b:continue
                x=t.copy();x[a]-=1;x[b]+=1
                da=int(native(x[a]*TILE,SG_C)-native(t[a]*TILE,SG_C))
                db=int(native(x[b]*TILE,SG_C)-native(t[b]*TILE,SG_C))
                q=occupancy.copy();q[:,a]+=da;q[:,b]+=db
                score=float(q.max(axis=1).mean())
                if score < best[0]-1e-8:best=(score,x)
        if best[1] is None:break
        t=best[1]
    return t*TILE


def shared_chain(gu_rows, sd_rows):
    """Optimistic whole-GU-to-down chain, excluding initial 2us submission.

    Per-stage max job includes controller command/input/channel output and
    its own PCIe drain.  It does not sum independent shard latencies.
    """
    g=native(gu_rows,SG_C)+(np.asarray(gu_rows)*4+7)//8
    d=native(sd_rows,SD_C)+(np.asarray(sd_rows)*4+7)//8
    return g.max(axis=-1)+2000+(SD_C+7)//8+int(d.max())


def boundary(demand_nand,demand_native,sd,r):
    choices=np.arange(11-r,12+r,dtype=np.int64)
    a=choices*TILE;b=(22-choices)*TILE
    gn_a=native(a,SG_C);gn_b=native(b,SG_C)
    sn=native(sd,SD_C);sw=waves(sd,SD_C)
    n=len(demand_native)
    rows=np.zeros((n,16),dtype=np.int64)
    nand_rows=np.zeros_like(rows)
    splits=np.zeros((n,8),dtype=np.int64)
    for p in range(8):
        score=np.maximum(demand_native[:,p,None]+sn[p]+gn_a,
                         demand_native[:,p+8,None]+sn[p+8]+gn_b)
        # Deterministic tie: smaller deviation from canonical split, then a.
        order=np.lexsort((choices,np.abs(choices-11)))
        ix=order[np.argmin(score[:,order],axis=1)]
        rows[:,p]=a[ix];rows[:,p+8]=b[ix];splits[:,p]=choices[ix]
        ns=np.maximum(demand_nand[:,p,None]+sw[p]+choices,
                      demand_nand[:,p+8,None]+sw[p+8]+22-choices)
        ni=order[np.argmin(ns[:,order],axis=1)]
        nand_rows[:,p]=a[ni];nand_rows[:,p+8]=b[ni]
    load=(demand_native+native(rows,SG_C)+sn).max(axis=1)+2000
    chain=shared_chain(rows,sd)+2000+(SG_C+7)//8
    nb=(demand_nand+waves(nand_rows,SG_C)+sw).max(axis=1)
    native_nb=(demand_nand+waves(rows,SG_C)+sw).max(axis=1)
    return dict(native_chip_bound_ns=load,shared_chain_bound_ns=chain,
        occupancy_chain_bound_ns=np.maximum(load,chain),nand_bound_waves=nb,
        native_selector_nand_bound_waves=native_nb,split_tiles=splits,gu_rows=rows)


def layer_worker(task):
    layer,frontier,seconds,outdir=task
    doc=json.loads(SOURCE.read_text())
    with np.load(doc['train_path']) as f:
        li=f['layers'].tolist().index(layer)
        selected=f['routes'][:32,:64,li].reshape(-1,doc['active'])
    phases={p['name']:p for p in doc['phases'] if p['layer']==layer}
    gu=np.asarray(phases['gate_up']['chosen_rows'],dtype=np.int64)
    down=np.asarray(phases['down']['chosen_rows'],dtype=np.int64)
    rw=waves(gu,2048)+waves(down,1408)
    rn=native(gu,2048)+native(down,1408)
    dw=rw[selected].sum(axis=1);dn=rn[selected].sum(axis=1)
    fits={};fit_receipts=[]
    for s in (264,265):
        fits[s],r=static_wave_milp(dw,s,seconds);fit_receipts.append(r)
    profiles=[]
    for fi,record in enumerate(frontier):
        for rotation in range(16):
            sd=np.roll(np.asarray(record['rows']),rotation)
            if 176+record['total_waves'] not in fits:continue
            s=fits[176+record['total_waves']]
            t=np.maximum(s-waves(sd,SD_C),0)
            if t.sum()!=176:continue
            gr=t*TILE
            occ=(dn+native(gr,SG_C)+native(sd,SD_C)).max(axis=1)+2000
            chain=2000+(SG_C+7)//8+shared_chain(gr,sd)
            score=float(np.maximum(occ,chain).mean())
            profiles.append((score,fi,rotation,gr,sd))
    assert profiles
    profiles.sort(key=lambda x:x[0])
    # Refit best three distinct SD frontier shapes on exact native row costs.
    chosen=[];seen=set()
    for _,fi,rot,gr,sd in profiles:
        if fi in seen:continue
        seen.add(fi);chosen.append((fi,rot,gr,sd))
        if len(chosen)==3:break
    onecopy=[]
    for fi,rot,gr,sd in chosen:
        gr=fit_gu(dn,sd,gr//TILE+waves(sd,SD_C))
        occ=(dn+native(gr,SG_C)+native(sd,SD_C)).max(axis=1)+2000
        chain=np.full(len(dw),2000+(SG_C+7)//8+shared_chain(gr,sd),dtype=np.int64)
        onecopy.append((float(np.maximum(occ,chain).mean()),fi,rot,gr,sd,occ,chain))
    best=min(onecopy,key=lambda x:x[0]);_,fi,rot,gr,sd,occ,chain=best
    one_waves=(dw+waves(gr,SG_C)+waves(sd,SD_C)).max(axis=1)
    dest=Path(outdir)
    arrays=dict(selected=selected,routed_nand_chip_waves=dw,routed_native_chip_ns=dn,
        onecopy_native_chip_bound_ns=occ,onecopy_chain_bound_ns=chain,
        onecopy_bound_ns=np.maximum(occ,chain),onecopy_nand_bound_waves=one_waves,
        onecopy_gu_rows=gr,onecopy_down_rows=sd)
    one=dict(gu_rows=gr.tolist(),shared_down_rows=sd.tolist(),down_frontier_index=fi,down_rotation=rot,
        gu_total_waves=int(waves(gr,SG_C).sum()),down_total_waves=int(waves(sd,SD_C).sum()),
        gu_max_waves=int(waves(gr,SG_C).max()),down_max_waves=int(waves(sd,SD_C).max()),
        native_chip_bound_ns=stats(occ),chain_bound_ns=stats(chain),combined_bound_ns=stats(np.maximum(occ,chain)),
        nand_bound_waves=stats(one_waves),fit='Exact floor/ceil total-load seed plus time-limited general integer MILP, materialized legal frontier and best3 native row-transfer refits; not globally optimal native placement.')
    replicas=[]
    for r in (0,1,2,3,11):
        candidates=[]
        for fi,record in enumerate(frontier):
            for rotation in range(16):
                sr=np.roll(np.asarray(record['rows']),rotation)
                # A single SD shard already exceeding best known full bound cannot win.
                if int(native(sr,SD_C).max())+int(dn.min())+2000 > best[0]+150000:continue
                b=boundary(dw,dn,sr,r)
                candidates.append((float(b['occupancy_chain_bound_ns'].mean()),fi,rotation,sr,b))
        _,fi,rotation,sr,b=min(candidates,key=lambda x:x[0])
        label='r'+str(r)
        for k,v in b.items():arrays[label+'_'+k]=v
        arrays[label+'_down_rows']=sr
        replicas.append(dict(r=r,shared_down_rows=sr.tolist(),down_frontier_index=fi,down_rotation=rotation,
            gu_canonical_rows=[352]*16,gu_range_splits_tiles=list(range(11-r,12+r)),
            native_chip_bound_ns=stats(b['native_chip_bound_ns']),chain_bound_ns=stats(b['shared_chain_bound_ns']),
            combined_bound_ns=stats(b['occupancy_chain_bound_ns']),nand_bound_waves=stats(b['nand_bound_waves']),
            mean_gain_vs_fitted_onecopy_pct=100*(1-b['occupancy_chain_bound_ns'].mean()/np.maximum(occ,chain).mean())))
    np.savez_compressed(dest/f'layer{layer}.npz',**arrays)
    return dict(layer=layer,samples=len(selected),onecopy=one,replicas=replicas,
        integer_wave_fit_certificates=fit_receipts,npz=str(dest/f'layer{layer}.npz'),npz_sha256=sha(dest/f'layer{layer}.npz'))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=8)
    ap.add_argument('--milp-seconds',type=float,default=8)
    ap.add_argument('--layers',type=int,default=26);ap.add_argument('--output',type=Path,default=DEST)
    args=ap.parse_args();args.output.mkdir(parents=True,exist_ok=True)
    began=time.monotonic();source_hash=sha(SOURCE);doc=json.loads(SOURCE.read_text())
    assert sha(doc['train_path'])==doc['train_sha256']
    shape=shape_receipt(doc)
    with np.load(doc['train_path']) as f:
        ids=f['prompt_ids'][:32].tolist();layers=f['layers'].tolist()[:args.layers]
        assert len(ids)==32 and all(x.startswith('train_') for x in ids)
    frontier=down_frontier()
    (args.output/'frontier.json').write_text(json.dumps(frontier,indent=2)+'\n')
    records=[]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures=[pool.submit(layer_worker,(layer,frontier,args.milp_seconds,str(args.output))) for layer in layers]
        for future in as_completed(futures):
            r=future.result();records.append(r)
            print('layer',r['layer'],'onecopy',r['onecopy']['combined_bound_ns']['mean'],
                  'r0/1/2/3/11 gains',[round(x['mean_gain_vs_fitted_onecopy_pct'],4) for x in r['replicas']],flush=True)
            (args.output/'partial.json').write_text(json.dumps(records,indent=2)+'\n')
    records.sort(key=lambda x:x['layer'])
    totals={}
    for label in ['onecopy']+['r'+str(r) for r in (0,1,2,3,11)]:
        key='onecopy_bound_ns' if label=='onecopy' else label+'_occupancy_chain_bound_ns'
        series=np.stack([np.load(x['npz'])[key] for x in records]).sum(axis=0)
        totals[label]=stats(series)
    base=totals['onecopy']['mean']
    for label in totals:totals[label]['mean_gain_vs_fitted_onecopy_pct']=100*(1-totals[label]['mean']/base)
    routed_bytes=26*64*3*2048*1408
    shared_bytes=26*3*2048*2816
    copies=[]
    for r in (0,1,2,3,11):
        extra_layer=16*r*TILE*SG_C
        copies.append(dict(r=r,extra_bytes_per_layer=extra_layer,extra_bytes_26layers=extra_layer*26,
            shared_gu_extra_fraction=r/11,full_routed_plus_shared_ffn_extra_pct=100*extra_layer*26/(routed_bytes+shared_bytes),
            stored_per_pair=dict(a_logical_start_tile=0,a_logical_end_tile=11+r,
                b_logical_start_tile=11-r,b_logical_end_tile=22),
            online_split_tiles=list(range(11-r,12+r)),pair_starts_rows=[704*p for p in range(8)],
            owners=[[p,p+8] for p in range(8)],
            legality='At split a, A computes [pairStart,pairStart+32a), B the remaining range. Each lies within its fixed stored interval. Full dot products; one selected copy per logical row.'))
    out=dict(status='complete train-only necessary-bound study; not schedule/device measurement',
        created_utc=datetime.now(timezone.utc).isoformat(),source_sha256=sha(__file__),source_plan=str(SOURCE),
        source_plan_sha256=source_hash,train_path=doc['train_path'],train_sha256=doc['train_sha256'],
        train_slice='[:32,:64]',prompt_ids=ids,layers=layers,shape_receipt=shape,
        native_cost_contract=dict(wave_ns=28000,command_ns=250,channel_bytes_per_ns=2,
            host_submission_ns=2000,pcie_bytes_per_ns=8,input_element_bytes=1,output_element_bytes=4,
            native_chip_ns='sum over jobs: 250+ceil(columns/2)+28000*ceil(rows*columns/65536)+2*rows; plus initial2us.',
            chain='Whole GU max native job plus own PCIe output; down2us submission, private PCIe upload and max down native job/output. Global channels/controller contention absent.'),
        shared_down_frontier=frontier,capacity=copies,records=records,paired26layer_totals_ns=totals,
        elapsed_seconds=time.monotonic()-began,
        limitations=['Only original training routes opened. No test/public routing accessed.',
            'Necessary NAND/native-chip occupancy and critical-path bounds; graph readiness/resource contention may erase all headroom.',
            'Copies fixed, but split selector uses current routed selections and no future information. Selector CPU cost omitted.',
            'Fitted onecopy uses minimum176wave sharedGU whole32rowtiles and exact sharedDown work/depth/jobs frontier. Native fitting is not certified globally optimal.',
            'The general shared-total-wave MILP is a relaxation when its16counts cannot be materialized with a frontier profile; reported dual gap is a NAND objective only.',
            'Depth/work/job frontier is exact for a single whole-row sharedDown matrix. Row placement permutation search is cyclic, not all16! permutations.',
            'Native format/firmware lowerings and numerical ABI remain unverified; no change to model pruning/precision/router or chip ISA claimed.',
            'Overlap-range replication and elastic tensor partitioning are prior concepts, not claimed novel here.'])
    assert sha(SOURCE)==source_hash and sha(doc['train_path'])==doc['train_sha256']
    target=args.output/'oracle.json';target.write_text(json.dumps(out,indent=2)+'\n')
    (args.output/'completion.json').write_text(json.dumps(dict(status='complete',oracle_sha256=sha(target),
        source_sha256=sha(__file__),source_plan_sha256=source_hash,train_sha256=doc['train_sha256']),indent=2)+'\n')
    print(json.dumps(totals,indent=2))


if __name__=='__main__':main()
