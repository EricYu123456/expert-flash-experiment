#!/usr/bin/env python3
"""Reconstruct unique one-block addresses for the frozen asymmetric winners."""
from pathlib import Path
import hashlib,json,sys
import numpy as np

OUT=Path(__file__).resolve().parent
sys.path.insert(0,str(OUT))
import asymmetric_down as d

_,_,menus,groups=d.checked_inputs()
chosen=d.f.read(OUT/'asymmetric-down/selection.frozen.json')['winners']
records=[];checked=0
for GU in d.LAYOUTS:
    menu=chosen[GU]
    for layer in range(3):
        occupied=set();counts=np.zeros((16,4),dtype=int)
        for e in range(65):
            if GU=='g8_fixed' and e<64:
                group=int(groups[layer,e]);resident=np.flatnonzero(groups[layer]==group)
                pos=int(np.flatnonzero(resident==e)[0]);chips=range(group*8,group*8+8);size=12
            else:chips=range(16);size=12 if e==64 else 6
            for j in range(size):
                if GU=='candidate':
                    plane=([3,2]*6)[j] if e==64 else [0,1,2,0,1,3][j]
                    wl=64+j//2 if e==64 else 2*e+j//3 if plane<2 else e
                elif GU=='RR':
                    page=384+j if e==64 else 6*e+j;plane=page%4;wl=page//4
                else:plane=j%4;wl=96+j//4 if e==64 else 3*pos+j//4
                for chip in chips:
                    key=(chip,plane,wl);assert key not in occupied
                    occupied.add(key);counts[chip,plane]+=1
        guard=max(w+1 for c,p,w in occupied);cursor=np.full(16,guard,dtype=int)
        downrows=menus[menu][layer];downpages=np.zeros((16,4),dtype=int)
        assert np.all(downrows.sum(axis=1)==1536)
        for e in range(65):
            columns=1024 if e==64 else 512
            for chip,rows in enumerate(downrows[e]):
                n=int(rows)*columns//16384;assert n*16384==int(rows)*columns
                for j in range(n):
                    key=(chip,j%4,int(cursor[chip])+j//4);assert key not in occupied
                    occupied.add(key);counts[chip,j%4]+=1;downpages[chip,j%4]+=1
                cursor[chip]+=(n+3)//4
        assert max(cursor)<=192 and int(counts.sum())*16384==155713536
        assert all(0<=wl<192 for c,p,wl in occupied)
        records.append(dict(GU=GU,Down=menu,layer=layer,GU_WL_end=guard,
                            full_model_WL_end=cursor.tolist(),resident_pages_per_chip_plane=counts.tolist(),
                            Down_pages_per_chip_plane=downpages.tolist(),useful_bytes=155713536,
                            programmed_bytes=155713536,allocated_four_plane_blocks_per_chip=1,
                            address_sha256=hashlib.sha256(np.asarray(sorted(occupied),dtype=np.int32).tobytes()).hexdigest()))
        checked+=len(occupied)
result=dict(status='pass',scope='Selected3GUfamilies x3layers; all64routed+shared; onecopy; actualstaticGU layout andfullprivateDown rows.',
            selection_sha256=d.f.sha(OUT/'asymmetric-down/selection.frozen.json'),
            source_sha256=d.f.sha(OUT/'asymmetric_down.py'),audit_source_sha256=d.f.sha(__file__),
            plans=records,physical_addresses_checked=checked,
            interpretation='Allwinners fitone192WL four-planeblock/chip: candidateGU128+Down50=178WL; RRandg8 GU99+Down50=149WL. Originalscreenretainscommon2blockbudgetforallmenus; thisisadditionalcapacityproof,notcapacitybasedspeedup.',
            timing_equivalence='EveryprivateDownjob beginsfreshcold anduploadsfullinput. Entire-job localblock1/offset -> assignedblock0/offset relocationpreservesdimensions/pages/source0123/coldcost. No genericcrossblockcacheclaim.')
d.f.dump(OUT/'asymmetric-down/selected_oneblock_capacity_audit.json',result)
print(json.dumps(dict(status='pass',plans=len(records),addresses=checked,max_WL=max(max(r['full_model_WL_end']) for r in records))))
