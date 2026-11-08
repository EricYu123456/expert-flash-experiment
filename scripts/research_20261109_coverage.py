#!/usr/bin/env python3
"""Build TRAIN/VALID-only eligible banks and matched enhanced raw-x controls."""
import argparse
import hashlib
import json
from pathlib import Path

H,I=2048,1408


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def row(layer,m):
    return '\t'.join(map(str,[layer,m['expert'],m['basis'],m['width'],m['transform_prefix'],m['readout_prefix']]))


def build(selections,output,levels):
    output=Path(output).resolve();output.mkdir(parents=True,exist_ok=False)
    cross,raw,source={},{},{}
    for path in selections:
        path=Path(path).resolve();m=json.loads(path.read_text());l=m['layer']
        assert m['status']=='formal_TRAIN_VALID_no_TEST'
        source[str(path)]=sha(path);owners=set(m['prototype_ids'])
        for f,bank in [('best_cross_prototype',cross),('raw_x_eq',raw)]:
            for rec in m['families'][f]:
                if rec['expert'] not in owners:
                    key=l,rec['expert'];assert key not in bank;bank[key]=rec
    assert set(cross)==set(raw)
    # Calibration error per replaced event per avoided coefficient MAC. This
    # is a candidate ordering only; true live quality determines eligibility.
    def risk(key):
        m=cross[key]
        return m['weighted_squared_error']/max(m['VALID_selected_event_count']*(3*H*I-H*m['width']),1)
    order=sorted(cross,key=lambda k:(risk(k),k))
    candidates=[]
    for n in sorted(set(min(n,len(order)) for n in levels)):
        chosen=order[:n]
        for family,bank in [('prototype',cross),('matched_raw',raw)]:
            registry=output/f'{family}.n{n:03d}.tsv'
            registry.write_text('\n'.join(row(l,bank[(l,e)]) for l,e in sorted(chosen))+'\n')
            candidates.append(dict(name=f'{family}_n{n:03d}',family=family,experts=n,
                                   registry=str(registry),registry_sha256=sha(registry),
                                   layer_experts=chosen,
                                   prototype_layer_owners=sorted({(l,p['prototype']) for (l,e),p in cross.items() if (l,e) in chosen})))
    result=dict(status='calibration_VALID_ordered_candidates_no_quality_TEST',
                ordering='selected alpha2 SSE / (selected event count * avoided coefficient MAC)',
                ordering_is_a_quality_guarantee=False,levels=levels,total_nonowner_experts=len(order),
                source_sha256=sha(__file__),selection_sha256=source,
                ranked=[dict(layer=k[0],expert=k[1],risk=risk(k),
                             prototype_nmse=cross[k]['selected_alpha2_nmse'],raw_nmse=raw[k]['selected_alpha2_nmse']) for k in order],
                candidates=candidates)
    (output/'candidates.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    return result


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--selections',nargs='+',required=True)
    p.add_argument('--output',required=True)
    p.add_argument('--levels',nargs='+',type=int,default=[4,8,16,32,64,128,168])
    a=p.parse_args();assert min(a.levels)>0
    r=build(a.selections,a.output,a.levels)
    print(json.dumps(dict(candidates=len(r['candidates']),total_experts=r['total_nonowner_experts'])))
