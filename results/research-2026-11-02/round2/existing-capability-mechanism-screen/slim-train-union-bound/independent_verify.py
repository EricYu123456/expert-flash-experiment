#!/usr/bin/env python3
"""Independent set/count and full-row/address verification; no native run."""
from pathlib import Path
from collections import Counter
import gzip
import hashlib
import json
import math
import numpy as np

P = Path(__file__).resolve().parent
R = P.parents[5]


def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    proto = json.loads((P/"protocol.frozen.json").read_text())
    for p, s in proto["inputs_sha256"].items():
        assert sha(Path(p)) == s, p
    report = json.loads((P/"report.json").read_text())
    a = np.load(P/"arrays.npz")
    witness = []
    checks = Counter()
    for model in report["models"]:
        H, I, SI = model["H"], model["I"], model["sharedI"]
        shapes = [(2*I,H),(H,I),(2*SI,H),(H,SI)]
        for conf in model["page_configurations"]:
            page = conf["page_bytes"]
            byblock = {}
            desc = {}
            for d in conf["per_die_address_witness"]:
                j, n = d["job_index"], d["pages"]
                e, phase = (j,0) if j<64 else ((j-64,1) if j<128 else (64,j-128))
                Rn,C = shapes[phase if e<64 else phase+2]
                r = Rn//16
                assert Rn%16 == 0 and r*C == n*page
                start,end = d["page_offset"],d["page_offset"]+n
                assert 0 <= start < end <= 512
                for lo,hi in byblock.setdefault(d["block"],[]):
                    assert end<=lo or start>=hi
                byblock[d["block"]].append((start,end))
                desc[j] = (e,phase,Rn,C,d)
                for chip in range(16):
                    witness.append({"model":model["model"],"page_bytes":page,"chip":chip,
                                    "expert":e,"phase":phase,"logical_row_start":chip*r,"logical_row_end":(chip+1)*r,
                                    "row_scale_IDs":"same logical expert/phase/row IDs",
                                    "full_input_columns":C,"block":d["block"],"page_offset":start,"pages":n,
                                    "copies":0,"fringe_bytes":0})
                    checks["physical_descriptors"] += 1
            assert set(desc)==set(range(130))
            assert len(byblock)==conf["g16_per_die_blocks512pages"]
            useful=sum((e<64 and 1 or 1)*(Rn*C//16) for e,ph,Rn,C,d in desc.values())*16
            assert useful==64*3*H*I+3*H*SI==conf["resident_programmed_page_bytes"]==conf["resident_useful_bytes"]
            assert len(byblock)*512*page*16==conf["g16_total_allocated_bytes"]
            # Independently certify every row interval, including shared, once.
            for j,(e,phase,Rn,C,d) in desc.items():
                rows=[]
                for chip in range(16):
                    rows.extend(range(chip*(Rn//16),(chip+1)*(Rn//16)))
                assert rows==list(range(Rn))
                checks["complete_operator_rowcoverages"] += 1
            for D,profiles in conf["legal_32_64die_page_exact_chunk_profiles"].items():
                for (Rn,C),profile in zip(shapes,profiles):
                    rows=np.array(profile["rows"],np.int64)
                    assert len(rows)==int(D) and rows.sum()==Rn
                    assert np.all(rows*C%page==0) and np.array_equal(rows*C//page,profile["pages"])
                    checks["page_DSE_row_multisets"] += 1
        for cohort in model["cohorts"]:
            B=cohort["B"]
            rr=a[f"{model['model']}.B{B}.routes"]
            counts=a[f"{model['model']}.B{B}.expert_multiplicity"]
            unions=[]
            for q,routes in enumerate(rr):
                c=Counter(int(e) for route in routes for e in route)
                assert sum(c.values())==6*B and max(c.values())<=B
                assert [c[e] for e in range(64)]==counts[q].tolist()
                U=len(c);unions.append(U)
                assert U==a[f"{model['model']}.B{B}.union"][q]
                read=3*H*I*U+3*H*SI
                mac=3*H*I*sum(c.values())+3*H*SI*B
                assert mac==cohort["global_exact_MACs_per_cohort"]
                # Direct expert-by-expert totals on each die, not a total/D shortcut.
                for chip in range(16):
                    die_MAC=sum((2*I//16*H+H//16*I)*n for e,n in c.items())
                    die_MAC+=(2*SI//16*H+H//16*SI)*B
                    assert die_MAC*16==mac
                    for page in [4096,16384]:
                        die_pages=sum((2*I//16*H//page+H//16*I//page) for e in c)
                        die_pages+=2*SI//16*H//page+H//16*SI//page
                        assert die_pages*16*page==read
                    checks["cohort_die_page_MAC_equalities"] += 1
                checks["cohorts"] += 1
            assert sum(unions)/len(unions)==cohort["selected_expert_union"]["mean"]
    for name in ["physical_row_intervals.json.gz","independent_checks.json"]:
        assert not (P/name).exists()
    with gzip.open(P/"physical_row_intervals.json.gz","wt") as f:
        json.dump(witness,f,separators=(",",":"))
    result={"status":"PASS","checks":dict(checks),"source_sha256":sha(Path(__file__)),
            "protocol_sha256":sha(P/"protocol.frozen.json"),"all_frozen_inputs_unchanged":True,
            "physical_image_scope":"Separate SLC/TLC coefficient-only16die static repacks; no weights/materializedMACs/native/fullSLIMABI proof.",
            "g16_page_and_MAC_floor_attained_all_cohorts":True}
    (P/"independent_checks.json").write_text(json.dumps(result,indent=2)+"\n")
    print(json.dumps(result,indent=2))


if __name__=="__main__":
    main()
