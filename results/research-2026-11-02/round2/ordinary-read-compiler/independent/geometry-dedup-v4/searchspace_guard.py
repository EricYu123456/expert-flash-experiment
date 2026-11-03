#!/usr/bin/env python3
"""Check conservative-key false negatives and explicit E8 counterexample."""
from collections import defaultdict
import importlib.util
import hashlib
import itertools
import json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
spec=importlib.util.spec_from_file_location('local_geometry_auditor',OUT/'audit.py')
aud=importlib.util.module_from_spec(spec);spec.loader.exec_module(aud)

def main():
    current=json.loads((aud.V4/'summary.json').read_text())['records']
    old={
        (r['model'],r['layer'],r['command_ns']):r
        for r in json.loads((aud.V3/'summary.json').read_text())['records']}
    remaining=[]
    for r in current:
        for phase,candidates in enumerate(r['fitting']):
            groups=defaultdict(list)
            for c in candidates:groups[aud.signature(c,r['shape'][2])[1]].append(c['label'])
            remaining += [dict(model=r['model'],layer=r['layer'],command_ns=r['command_ns'],phase=phase,labels=v)
                          for v in groups.values() if len(v)>1]
    assert not remaining
    # Same initial physical assignment but different unused/duplicate variants.
    E=8;base=[2048,2048]+[0]*14
    a=dict(label='disjoint_2',family='disjoint',base_rows=base)
    b=dict(label='cyclic_2',family='cyclic',base_rows=base)
    va,vb=aud.variants(a),aud.variants(b)
    ia,ib=aud.initial(a,E),aud.initial(b,E)
    assert np.array_equal(va[ia],vb[ib])
    ka,kb=aud.signature(a,E),aud.signature(b,E)
    assert ka[0]!=kb[0] and ka[1]==kb[1]
    # Every swap corresponds under ordinal map k->2k; enumerate all8!images.
    ra=[tuple(x) for x in va[ia]];rb=[tuple(x) for x in vb[ib]]
    assert len(set(ra))==8
    checked=0
    for permutation in itertools.permutations(range(8)):
        assert tuple(ra[x] for x in permutation)==tuple(rb[x] for x in permutation)
        checked+=1
    assert checked==40320
    high=[]
    for r in current:
        if r['model']!='qwen38' or r['command_ns']!=10000:continue
        p=old[(r['model'],r['layer'],r['command_ns'])]
        cy,wh=p['selection'][1:]
        if wh['mean_ns']>=cy['mean_ns']:continue
        aliases=[]
        for phase,index in enumerate(wh['key'][:2]):
            previous=p['fitting'][phase][index]
            canonical=next(c for c in r['fitting'][phase] if previous['label'] in c['aliases'])
            assert canonical['family'] in ('disjoint','cyclic')
            aliases.append(dict(old_label=previous['label'],canonical_label=canonical['label']))
        high.append(dict(layer=r['layer'],old_gain_ns=cy['mean_ns']-wh['mean_ns'],aliases=aliases))
    result=dict(status='PASS current342records; conservative-key incompleteness found outside current E64/E512 population',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),current_phases=len(current)*2,
        current_remaining_same_assignment_spaces=remaining,
        counterexample=dict(experts=E,chips=16,rows=4096,base_rows=base,
            candidate_A='disjoint_2',candidate_B='cyclic_2',initial_labels_A=ia.tolist(),initial_labels_B=ib.tolist(),
            assigned_physical_vectors_equal_pointwise=True,
            variant_multiset_A=dict(unique_rows=8,multiplicity_each=2),
            variant_multiset_B=dict(unique_rows=16,multiplicity_each=1),
            all_reachable_assignments_checked=checked,
            consequence='Current multiset+hist key retains these equivalent E8spaces, allowing another seeded search if such a campaign is run. This is a false negative/missed merge, not an unsound merge; current342records have none.'),
        current_highcost_qwen38_v3_positive_no_new_geometry_layers=high,
        recommendation='Keep v4sealed. Limit the one-fit-per-space statement to audited342records. Future arbitrary-E swaps may key on assigned physical histogram alone, provided expert-specific restrictions/ordinal costs/restricted swaps are absent. E8g1 inventories must remain distinct.')
    (OUT/'searchspace_guard.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(dict(status=result['status'],current_phases=len(current)*2,
                         remaining_same_spaces=len(remaining),E8_permutations=checked,old_highcost_positive_layers=len(high))))

if __name__=='__main__':main()
