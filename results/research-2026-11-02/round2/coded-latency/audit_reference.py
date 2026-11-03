"""Independent ledger and unequal-padding algebra checks; no model inference."""
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path('/home/ericyu/expert_flash_experiment')
sys.path.insert(0,str(ROOT/'scripts'))
from research_20261102_coded_latency import Background

DEST=Path(__file__).resolve().parent
checks=[]
for subdir in (DEST,DEST/'external'):
    document=json.loads((subdir/'summary.json').read_text())
    assert document['source_sha256']==hashlib.sha256((ROOT/'scripts/research_20261102_coded_latency.py').read_bytes()).hexdigest()
    for model in document['models']:
        path=subdir/'phase_ledgers'/f'{model["model"]}.npz'
        with np.load(path) as z:
            base=z['data_useful_bytes'];code=z['coded_useful_bytes'];needed=z['needed']
            for erased in range(16):
                healthy=[c for c in range(16) if c!=erased]
                assert np.all(code[:,erased,erased]==0)
                for field in ('waves','pages','useful_bytes','commands','inputs','outputs','service_ns'):
                    data=z['data_'+field];coded=z['coded_'+field]
                    assert np.all(coded[:,erased,healthy]>=data[:,healthy])
                    assert np.all(coded[:,erased,erased]==0)
                    inactive=~needed[:,erased]
                    np.testing.assert_array_equal(coded[inactive,erased,:],data[inactive,:])
            source=json.loads((ROOT/f'results/research-2026-11-02/projection-study/{model["model"]}.frozen_plan.json').read_text())
            expected=[]
            data_capacity=parity_capacity=0
            for phase in source['phases']:
                rows=np.array(phase['chosen_rows']);cols=phase['columns']
                data_capacity+=int(rows.sum()*cols)
                parity_capacity+=int(sum(2*max(rows[e,h*8:h*8+8])*cols for e in range(len(rows)) for h in (0,1)))
                expected.append(source['active']*phase['total_rows']*cols)
            np.testing.assert_array_equal(base.sum(axis=1),np.tile(expected,model['tokens']))
            assert model['capacity']['data_weight_bytes']==data_capacity
            assert model['capacity']['parity_weight_bytes']==parity_capacity
        for scenario in model['scenarios']:
            policies=scenario['policies']
            for cap in ('none','0','10000','100000','1000000'):
                baseline=policies[f'chosen_wait_cap{cap}_decode0']
                # Different absolute traces can alter P95 alignment, so only
                # no-stall equality and per-phase min are structural checks.
                if scenario['duty']==0:
                    for decode in ('0','2000','10000','measured_median','measured_p95'):
                        coded=policies[f'coded_cap{cap}_decode{decode}']
                        np.testing.assert_array_equal(coded['per_token_ms'],baseline['per_token_ms'])
                if cap in ('0','10000'):
                    for decode in ('0','2000','10000','measured_median','measured_p95'):
                        assert policies[f'coded_cap{cap}_decode{decode}']['stats'].get('coded_repair_phases',0)==0
        checks.append(dict(split=document['split'],model=model['model'],phases=len(base),
                           ledger_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                           parity_extra_fraction=parity_capacity/data_capacity))

# Padding with zeros does not alter the radix256 integer identity. Test g8
# inactive halves and unequal rows, without calling the production codec.
rng=np.random.default_rng(20261102);algebra_checks=0
for row_counts in ([3]*8+[0]*8,[1,2,3,4,1,2,3,4]*2,[5,5,5,5,5,2,2,2]*2):
    maximum=max(row_counts);columns=31
    weight=rng.integers(-128,128,(16,maximum,columns),dtype=np.int16).astype(np.int8)
    for chip,count in enumerate(row_counts):weight[chip,count:]=0
    vector=rng.integers(-128,128,columns,dtype=np.int16).astype(np.int8)
    direct=np.einsum('cij,j->ci',weight.astype(np.int64),vector.astype(np.int64))
    for erased in range(16):
        half=erased//8;p=weight[half*8:half*8+8].astype(np.int64).sum(axis=0)
        low=((p+128)%256)-128;high=(p-low)//256
        assert np.all((-128<=high)&(high<=127))
        rebuilt=low@vector.astype(np.int64)+256*(high@vector.astype(np.int64))
        rebuilt-=sum((direct[c] for c in range(half*8,half*8+8) if c!=erased),np.zeros(maximum,dtype=np.int64))
        np.testing.assert_array_equal(rebuilt,direct[erased]);algebra_checks+=1

background=Background(5_000_000,.25)
start=background.offset+1
chip,end,event=background.current(start)
assert background.current(start+1000)==(chip,end,event)
assert background.current(end) is None
next_chip,next_end,next_event=background.current(background.offset+background.period+1)
assert next_chip==(chip+1)%16 and next_event==event+1
result=dict(checks=checks,padded_native_integer_erasure_checks=algebra_checks,
            persistent_background_identity_checked=True,all_checks_passed=True,
            source_sha256=hashlib.sha256((ROOT/'scripts/research_20261102_coded_latency.py').read_bytes()).hexdigest(),
            audit_reference_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
(DEST/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
