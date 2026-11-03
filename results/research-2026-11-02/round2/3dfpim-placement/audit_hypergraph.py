#!/usr/bin/env python3
"""Independent standard-connectivity objective audit; no new partitioner claim."""
from pathlib import Path
import gzip
import hashlib
import json
import numpy as np

ROOT=Path(__file__).resolve().parent
INPUT_IO=1280
OUTPUT_IO=1281


def nets_for(routes,aggregate):
    unique,counts=np.unique(routes,axis=0,return_counts=True)
    marginal=np.bincount(routes.ravel(),minlength=64)
    nets=[]
    for e in range(64):
        w=int(marginal[e])
        if not w:continue
        down=[e*20+8+r for r in range(12)]
        for j in range(4):
            nets.append(([e*20+j,e*20+4+j],w,"gate_up_pair"))
            nets.append(([e*20+4+j]+down,w,"activation"))
        if not aggregate:
            for d in down:nets.append(([d,OUTPUT_IO],w,"private_output"))
    for selected,count in zip(unique,counts):
        s=[int(e) for e in selected];w=int(count)
        nets.append(([INPUT_IO]+[e*20+j for e in s for j in range(8)],12*w,"common_input"))
        if aggregate:
            for r in range(12):
                nets.append(([OUTPUT_IO]+[e*20+8+r for e in s],w,"aggregate_output"))
    return nets


def main():
    data=json.loads((ROOT/"results.json").read_text())
    maps=np.load(ROOT/"fixed_mappings.npz")
    with np.load(data['routes_path']) as f: raw=f['routes'][:32,:64,:3]
    checks=[];exports=[]
    for entry in data['results']:
        layer=entry['layer'];policy=entry['policy'];aggregate=policy=='aggregate'
        routes=np.sort(raw[:,:,layer].reshape(-1,6),axis=1)
        nets=nets_for(routes,aggregate)
        path=ROOT/f'L{layer}_{policy}.hypergraph.json.gz'
        document=dict(vertices=1282,matrix_vertices=1280,
            fixed_terminal_parts={str(INPUT_IO):54,str(OUTPUT_IO):55},
            permitted_matrix_parts=list(range(54)),matrix_core_capacity_per_part=[24]*53+[8],
            scale='weighted connectivity cut * 128 / 2048 = mean send element-length units',
            nets=[dict(pins=p,weight=w,kind=k) for p,w,k in nets])
        with gzip.GzipFile(filename=str(path),mode='wb',mtime=0) as f:
            f.write(json.dumps(document,separators=(',',':')).encode())
        exports.append(dict(path=path.name,net_count=len(nets),
                            sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
        for label in [entry['strongest_static_baseline'],'query_aware_native_multicast']:
            mapping=maps[f'L{layer}_{policy}_{label}']
            part=np.r_[mapping,[54,55]]
            perkind={}
            for pins,weight,kind in nets:
                value=weight*(len({int(part[v]) for v in pins})-1)
                perkind[kind]=perkind.get(kind,0)+value
            measured=sum(perkind.values())*128/2048
            expected=next(s['mean_send_units'] for s in entry['scores'] if s['label']==label)
            assert measured==expected,(layer,policy,label,measured,expected)
            checks.append(dict(layer=layer,policy=policy,label=label,send_units=measured,
                               independent_standard_hypergraph_matches=True,weighted_cut_components=perkind))
    native=data['native_trace_crosschecks']
    fixed=[]
    for policy in ['private','aggregate']:
        a=next(c for c in native if c['label']==f'L0-{policy}-query_aware_native_multicast')
        b=next(c for c in native if c['label']==f'L0-{policy}-query-second-route')
        common=set(a['selected_matrix_fixed_assignment'])&set(b['selected_matrix_fixed_assignment'])
        assert common
        assert all(a['selected_matrix_fixed_assignment'][n]==b['selected_matrix_fixed_assignment'][n] for n in common)
        fixed.append(dict(policy=policy,shared_active_matrix_cores=len(common),addresses_invariant=True))
    result=dict(scope='Independent exact reduction of this fixed-atom movement objective to existing connectivity hypergraph partitioning',
        primary_prior={'title':'PaToH: Partitioning Tool for Hypergraphs','authors':'Umit V. Catalyurek and Cevdet Aykanat',
            'version':'November1999; manual revised March2011',
            'url':'https://faculty.cc.gatech.edu/~umit/PaToH/manual.pdf',
            'read_depth':'Section2.1 Equation4b connectivity objective; Section4.2.2 fixed-cell interface. No PaToH binary execution claimed.'},
        construction=['Common input: one query net contains its 48 active GU cores and fixed input terminal, weight12*query frequency.',
            'G/U pair: two-core net, marginal expert frequency weight.',
            'Private activation: U core and twelve down consumers form one multicast net per neuron128 chunk, marginal-frequency weight.',
            'Private output: each down core and fixed output terminal form one net.',
            'Aggregated output: selected down cores for one output-row128 range and fixed output terminal form one query net.'],
        metric='sum_n weight[n]*(number of tile parts connected by net n - 1)',
        implication='The existing-hardware movement problem in this conservative fixed-matrix-tile setting is an exact instance of standard connectivity hypergraph partitioning, including native multicast and local output aggregation. Backend expression/count validation is useful engineering; this is not a new objective or algorithm.',
        optimizer_limit='Our annealing/greedy results are heuristic train-only values; no optimality, heldout benefit or native cycle advantage proved.',
        private_output_lower_bound={'K':6,'send_units':15360,
            'proof':'Let g>=2 be touched input tiles (48 active GU cores, capacity24). At least max(0,120-24*g) of72 active down cores lie outside these tiles. Each external tile/expert needs all512 z elements and contains at most12 down cores for that expert, so activation send >=512/12*max(0,120-24*g). Add1536*g common input and6*1536 private output. Minimum over integer g>=2 is15360 atg2. Other G-U traffic is nonnegative.',
            'limits':'Relaxed route-specific bound, not proof a single fixed64expert layout attains it. Does not apply to output-aggregate policy or undocumented depth co-packing.'},
        exact_checks=checks,fixed_address_checks=fixed,exports=exports,
        native_crosschecks={'attempts':len(native),'passed':sum(c['exit_code']==0 for c in native),
             'failed_original_random':[c['label'] for c in native if c['exit_code']!=0]})
    (ROOT/'independent_objective_audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'exact_hypergraph_checks':len(checks),'native_successes':result['native_crosschecks']['passed'],
                      'fixed_address_checks':fixed}))

if __name__=='__main__':main()
