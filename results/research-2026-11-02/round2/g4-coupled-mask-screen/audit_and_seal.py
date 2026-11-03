import hashlib
import json
import math
import shutil
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path

import numpy as np

ROOT=Path('/home/ericyu/expert_flash_experiment')
OUT=Path(__file__).resolve().parent


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


d=json.loads((OUT/'result.json').read_text());z=np.load(OUT/'arrays.npz')
old=sum(any(all(c//4!=g for c in s) for g in range(4)) for s in combinations(range(16),5))
assert old==2832 and math.comb(16,5)-old==1536
audit={}
for record in d['runs']:
    k=record['key'];gu=z[k+'__gu_rows'];down=z[k+'__down_rows'];cat=z[k+'__categories'];routes=z['routes']
    assert np.all(gu[:64].sum(1)==1024) and np.all(down[:64].sum(1)==1536)
    assert np.all((cat==2).sum(1)==5) and np.all((cat==1).sum(1)==4)
    gp=(gu*1536+16383)//16384;dp=(down*512+16383)//16384
    # Shared Down has a different input dimension1024.
    dp[-1]=(down[-1]*1024+16383)//16384
    gw=(gp+3)//4;dw=(dp+3)//4
    assert np.all(gw[:64].sum(1)==25) and np.all(dw[:64].sum(1)==12)
    assert np.array_equal(gw[:64]+dw[:64],2+(cat==2))
    assert np.all((gu[:64]==83).sum(1)==6) and np.all((gu[:64]==82).sum(1)==3)
    assert np.all((gu[:64]==40).sum(1)==7)
    gc=gw*28000+(gu>0)*(250+768)+2*(gp*16384//1536)
    dc=dw*28000+(down>0)*(250+256)+2*(dp*16384//512)
    dc[-1]=dw[-1]*28000+(down[-1]>0)*(250+512)+2*(dp[-1]*16384//1024)
    mandatory=(gc[:64][routes].sum(1)+dc[:64][routes].sum(1)+gc[-1]+dc[-1]).max(1)
    assert abs(mandatory.mean()-record['fit']['mean_joint_chip_ns'])<1e-8
    native=z[k+'__cost'];assert np.all(native>=mandatory+2000)
    assert abs(native.mean()-record['mean_native_ns'])<1e-8
    physical_wave_inventory=(gw+dw).sum(0)
    assert physical_wave_inventory.max()<=192
    useful=(gu*1536+down*np.r_[np.full(64,512),1024][:,None]).sum(0)
    audit[k]=dict(complete_rows=True,categories_and_waves_exact=True,independent_joint_objective_exact=True,
                  perchip_wave_inventory=physical_wave_inventory.tolist(),oneblock_perchip_constructible=True,
                  perchip_useful_bytes=useful.tolist(),perchip_high_mask_experts=(cat==2).sum(0).tolist(),
                  programmed_page_bytes=int((gp.sum()+dp.sum())*16384))
(OUT/'independent_checks.json').write_text(json.dumps(dict(old_masks=old,total_masks=4368,
    missing_masks=1536,decompositions_per_new_mask=330,cases=audit),indent=2)+'\n')
snap=OUT/'source-snapshot';snap.mkdir(exist_ok=True)
source=ROOT/'scripts/research_20261103_g4_coupled_masks.py';assert sha(source)==d['script_sha256']
shutil.copyfile(source,snap/source.name)
files=[dict(path=str(p),sha256=sha(p),bytes=p.stat().st_size) for p in sorted(OUT.rglob('*'))
       if p.is_file() and p.name!='completion.json']
(OUT/'completion.json').write_text(json.dumps(dict(status='sealed_G4_L0_TRAIN_screen',utc=datetime.now(timezone.utc).isoformat(),
    source_sha256=sha(source),no_heldout_used=True,files=files),indent=2)+'\n')
print('completion_sha256',sha(OUT/'completion.json'))
