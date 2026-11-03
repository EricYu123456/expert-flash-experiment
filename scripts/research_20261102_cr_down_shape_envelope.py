#!/usr/bin/env python3
"""Route-free native Down-kernel envelope on both existing model shapes."""
import argparse
import importlib.util
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/research-2026-11-02/round3-cr-ipr"
OUT=BASE/"second-shape-native-envelope"
PIPE=ROOT/"scripts/research_20261102_cr_ipr_pipeline.py"
spec=importlib.util.spec_from_file_location("shape_pipeline",PIPE)
p=importlib.util.module_from_spec(spec)
import sys
sys.modules[spec.name]=p
spec.loader.exec_module(p)
import hashlib


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path,value):
    path.write_text(json.dumps(value,indent=2)+"\n")


def waves(rows,columns):
    size=rows*columns
    assert size%16384==0
    pages=size//16384
    return [dict(pages=[dict(page_id=f"v.p{j}",owner="v",page_index=j,
                            plane=j%4,block=0,wl=j//4,bytes=16384)
                       for j in range(i,min(i+4,pages))],source_order=list(range(min(4,pages-i))))
            for i in range(0,pages,4)]


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/"protocol.frozen.json").exists()
    dump(OUT/"protocol.frozen.json",dict(
        status="frozen_before_route_free_shape_sweep",
        inputs_sha256={str(x):sha(x) for x in [Path(__file__).resolve(),PIPE]},
        models=dict(granite4tiny=dict(H=1536,I=512,shared_I=1024),
                    deepseek=dict(H=2048,I=1408,shared_I=2816)),
        group_widths=[2,4,8,16],array_modes=["cr","ordinary"],
        policies=["strict_no_interval","cold_reset"],copy_ns=500,
        work="One complete routed Down shard R=H/group,C=I, and one half-sharedDown shard R=H/(2group),C=shared_I, each full input. Equal weight bytes where shared_I=2I. No routes or fitting.",
        limits="Microkernels only, no fullFFN/GU/input/output calendar, taskquality or hardwaremeasurement. Coldreset must be implementable while preserving the same native GEMV input/row accumulator context. No global geometry optimality claim."))


def run():
    protocol=json.loads((OUT/"protocol.frozen.json").read_text())
    for path,h in protocol["inputs_sha256"].items():
        assert sha(path)==h,path
    assert not (OUT/"report.json").exists()
    records=[]
    for model,shape in protocol["models"].items():
        for group in protocol["group_widths"]:
            for role in ["routed","half_shared"]:
                rows=shape["H"]//group//(2 if role=="half_shared" else 1)
                columns=shape["shared_I"] if role=="half_shared" else shape["I"]
                for array in protocol["array_modes"]:
                    for policy in protocol["policies"]:
                        settings=dict(cache_copy_ns=500,source_policy="fixed",issue_policy=policy,
                                      array_mode=array,row_bytes=columns,validate_row_stream=True,
                                      unselected_policy="reject")
                        result=p.simulate(waves(rows,columns),settings)
                        rec=dict(model=model,group=group,role=role,rows=rows,columns=columns,
                                 array=array,policy=policy,feasible=result["feasible"],
                                 duration_ns=result["makespan_ns"] if result["feasible"] else None,
                                 cold_waves=result["cold_waves"],cr_waves=result["cr_waves"],
                                 physical_pages=result["physical_read_pages"],
                                 useful_bytes=rows*columns,violations=result["violations"][:2])
                        records.append(rec)
                        if group==2:
                            dump(OUT/f"{model}.{role}.{array}.{policy}.json",result)
    dump(OUT/"report.json",dict(status="complete_native_kernel_only",records=records,
         protocol_sha256=sha(OUT/"protocol.frozen.json"),source_sha256=sha(__file__),scope=protocol["limits"]))
    for rec in records:
        if rec["role"]=="routed" and rec["group"] in [2,4]:
            print(json.dumps(rec))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["freeze","run"])
    action=parser.parse_args().mode
    freeze() if action=="freeze" else run()
