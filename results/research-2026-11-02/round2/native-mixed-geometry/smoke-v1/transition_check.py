"""Force accepted geometry -> chip swap -> restore; not a fit or selection."""
import json
import sys
from pathlib import Path
import numpy as np

ROOT=Path('/home/ericyu/expert_flash_experiment');sys.path.insert(0,str(ROOT/'scripts'))
import research_20261103_native_mixed_geometry as m

OUT=Path(__file__).resolve().parent
p=json.loads((OUT/'protocol.frozen.json').read_text());checks=[]
assert m.sha(m.__file__)==p['source_sha256']
for case in p['cases']:
    r=json.loads(Path(case).read_text());w=r['winners']['independent_paired'];a=w[w['selected']]
    with np.load(a['arrays']) as z:rows=z['rows'];routes=z['routes'];initial=z['cost']
    f=m.MixedFit(rows,r['shape'],r['service'],routes,20261105,p['menus'][r['model']])
    e=int(routes[0,0]);ii=f.members[e];transitions=[]
    for phase in range(2):f.rows[e,phase]=min(f.menu[phase],key=lambda c:np.count_nonzero(c['rows']))['rows']
    for tag in ['geometry_accepted','geometry_then_permutation']:
        if tag.endswith('permutation'):
            for phase in range(2):f.rows[e,phase,[0,15]]=f.rows[e,phase,[15,0]]
        oldcost=f.values.copy();f.plan=f.lower()
        values,bounds=f.native.run(f.plan,f.unique[ii],f.config)
        f.values[ii]=values;f.bounds[ii]=bounds;f.total=int(f.values@f.weights)
        f.checkpoint(tag)
        full,fb=f.native.run(f.plan,routes,f.config)
        direct=f.backend.ReferenceGraph.build(f.plan,np.r_[routes[0],r['shape'][2]],f.service).run()
        assert direct['cost']==full[0]
        unaffected=np.ones(len(f.unique),bool);unaffected[ii]=False
        assert np.array_equal(f.values[unaffected],oldcost[unaffected])
        transitions.append(dict(tag=tag,expert=e,affected_unique=len(ii),jobs=len(f.plan['jobs']),python_first_query_cost_ns=int(full[0])))
    f.rows=rows.copy();f.plan=f.lower();f.values,f.bounds=f.native.run(f.plan,f.unique,f.config);f.total=int(f.values@f.weights)
    f.checkpoint('initial_restored')
    restored,_=f.native.run(f.plan,routes,f.config);assert np.array_equal(restored,initial)
    checks.append(dict(model=r['model'],case=case,case_sha256=m.sha(case),transitions=transitions,all512_restored_exact=True))
out=dict(status='passed_forced_geometry_to_permutation_and_restore',source_sha256=m.sha(__file__),fitter_sha256=m.sha(m.__file__),checks=checks,scope='Deterministic correctness probes, no parameter choice or output-driven selection.')
(OUT/'transition_check.json').write_text(json.dumps(out,indent=2)+'\n')
print(json.dumps(out,indent=2))
