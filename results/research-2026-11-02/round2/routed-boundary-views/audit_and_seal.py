import hashlib
import json
import shutil
from datetime import datetime,timezone
from pathlib import Path

import numpy as np

ROOT=Path('/home/ericyu/expert_flash_experiment')
OUT=Path(__file__).resolve().parent


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


screen=json.loads((OUT/'L0/screen.json').read_text())
control=json.loads((OUT/'L0/controls.json').read_text())
z=np.load(OUT/'L0/screen.npz');c=np.load(OUT/'L0/controls.npz')
rows=z['base_rows'];audit={}
for strategy in screen['cases']:
    edges=z[strategy+'__edges'];extra=0;page_extra=0
    for e,(s,t) in enumerate(edges):
        counts=rows[e,0];rhi=int(counts[s]);rlo=int(counts[t]);delta=rhi-rlo
        assert rhi==101 and rlo in [75,76]
        logical=[];cursor=0
        for r in counts:
            logical.append(np.arange(cursor,cursor+int(r)));cursor+=int(r)
        packet=logical[s][rlo:]
        physical=[a.copy() for a in logical]
        physical[t]=np.r_[logical[t],packet]
        for b in [False,True]:
            view=[]
            for chip in range(16):
                n=int(counts[chip])
                if b and chip==s:n=rlo
                if b and chip==t:n=rhi
                view.extend(physical[chip][:n].tolist())
            assert len(view)==1280 and sorted(view)==list(range(1280))
        assert sum(map(len,physical))==1280+delta
        extra+=delta*2560
        page_extra+=((rhi*2560+16383)//16384-(rlo*2560+16383)//16384)*16384
    item=screen['cases'][strategy]
    assert extra==item['duplicate_coefficient_bytes']
    assert page_extra==item['extra_programmed_page_bytes']==33554432
    delta=z[strategy+'__delta'];routes=z['routes'];base=z['base_joined_work']
    bits=((np.arange(1024)[:,None]>>np.arange(10))&1)
    best=np.asarray([(base[q]+bits@delta[r]).max(1).min() for q,r in enumerate(routes)])
    assert np.array_equal(best,z[strategy+'__dynamic_work'])
    assert np.all(z[strategy+'__dynamic_cost']>=best+2000)
    audit[strategy]=dict(logical_useful_rows_once_each_view=True,packet_ids_exact=True,
                         duplicate_coefficient_bytes=extra,extra_page_bytes=page_extra,
                         exact1024_work_oracle_reproduced=True,
                         short_view_pages=12,long_view_pages=16,
                         short_complete_outputs=76,long_complete_outputs=102,
                         native_prefix_ABI_not_proved_by_symbolic_audit=True)
bestwhole=min(v['mean_native_ns'] for v in control['controls'].values())
packet_lower=screen['cases']['learned']['mean_dynamic_mandatory_ns']
upper=100*(1-packet_lower/bestwhole)
for name,v in control['controls'].items():
    assert len(v['selected_experts'])==10
    assert v['extra_coefficient_bytes']<=control['same_capacity']['packet_extra_coefficient_bytes']
    assert v['extra_programmed_page_bytes']<=control['same_capacity']['packet_extra_programmed_page_bytes']
    assert abs(c[name+'__cost'].mean()-v['mean_native_ns'])<1e-8
(OUT/'independent_checks.json').write_text(json.dumps(dict(
    packet_cases=audit,all_controls_GU_only=True,best_whole_mean_ns=bestwhole,
    fixed_learned_packet_mean_lower_bound_ns=packet_lower,
    best_possible_fixed_packet_vs_whole_mean_advantage_upper_pct=upper,
    scope='Finite frozen edge configurations and64TRAIN queries, not all possible edge placements. No heldout.'),indent=2)+'\n')
snap=OUT/'source-snapshot';snap.mkdir(exist_ok=True)
sources=[]
for name in ['research_20261103_routed_boundary_views.py','research_20261103_routed_boundary_controls.py']:
    p=ROOT/'scripts'/name;shutil.copyfile(p,snap/name);sources.append(dict(path=str(p),sha256=sha(p)))
files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*'))
       if p.is_file() and p.name!='completion.json']
(OUT/'completion.json').write_text(json.dumps(dict(status='sealed_no_expansion',utc=datetime.now(timezone.utc).isoformat(),
    sources=sources,files=files,no_heldout_used=True,no_sealed_inputs_modified=True),indent=2)+'\n')
print('completion_sha256',sha(OUT/'completion.json'))
print('fixed_packet_vs_whole_upper_pct',upper)
