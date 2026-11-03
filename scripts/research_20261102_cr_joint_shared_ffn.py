#!/usr/bin/env python3
"""Frozen full-FFN replay of standard CBC with movable shared row halves."""
from __future__ import annotations

import argparse
from collections import Counter
import importlib.util
import json
from pathlib import Path
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/research-2026-11-02/round3-cr-ipr"
OUT=BASE/"joint-shared-full-ffn"


def module(name,path):
    spec=importlib.util.spec_from_file_location(name,path)
    obj=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(obj)
    return obj


g2=module("joint_g2_driver",ROOT/"scripts/research_20261102_cr_g2_availability_ffn.py")
joint=module("joint_cbc_builder",ROOT/"scripts/research_20261102_eight_pair_cbc.py")
native=module("native_group_driver",ROOT/"scripts/research_20261102_cr_native_group_control.py")
f,r,a=g2.f,g2.r,g2.a


def axes(diagnostic=False):
    return [(family,"cr",copies,df) for family in ["candidate","RR"]
            for copies in [1,4] for df in ([0] if diagnostic else [0,1])]+[
            ("g8_fixed",array,copies,0) for array in ["cr","ordinary"] for copies in [1,4]]


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/"protocol.frozen.json").exists()
    paths=[Path(__file__).resolve(),ROOT/"scripts/research_20261102_eight_pair_cbc.py",
           ROOT/"scripts/research_20261102_cr_g2_availability_ffn.py",
           ROOT/"scripts/research_20261102_cr_native_group_control.py",
           joint.OUT/"protocol.frozen.json",joint.OUT/"layout.frozen.json",
           joint.OUT/"resident_jobs.frozen.json",g2.DRIVER,g2.KERNEL,f.PIPE,f.CPP,f.TRAIN,
           r.BASE/"g8-addendum/group_maps.frozen.npz",
           r.OUT/"TRAIN512/service_arrays.npz",g2.OUT/"TRAIN512/service_arrays.npz",
           native.OUT/"service_arrays.npz",BASE/"prospective-full-ffn/service_arrays.npz",
           BASE/"prospective-routes/decode.routes.npz"]
    f.dump(OUT/"protocol.frozen.json",dict(status="frozen_before_joint_shared_causal_replay",utc=f.utc(),
        inputs_sha256={str(path):f.sha(path) for path in paths},
        TRAIN_axes=axes(),diagnostic_axes=axes(True),payload_MiB=303,
        native="Sixrouted+two sharedhalf jobs on8disjointpairs. Eachpair executes one24page/sixwave sameWL fixed0123 job/chip. Sharedhalves use same original fullC1024 input, rowintervals[0,768) and[768,1536). Every routed job uses private fullC512 input. Noextra MACcontexts/partialinput/columnpartial or buffer.",
        placement="64unique balanced4supports fromC(8,4);32E/pair plusS0/S1=34residentjobs=204WL/chip. GUreservation128,10jobs block0+24block1,nojob crossesblock. All6expert subsets match6pairs by Hall, two sharedhalfjobs take unusedpairs. Standard CBC, nofitting or newtheorem.",
        capacity="Payload303MiB versusg4triples244.5and sparseg2 289.5; same32chip-blockupperbudget, actual32blocks. Extra13.5MiB vs sparseg2 and58.5MiB vs g4triples are explicitlycharged.",
        GU="candidate/RR use inherited same first-feasibleconditionalpermutation/drop CRpolicy. g8fixed uses newlysealed whole-wavefixed0123/sameWL/noIPR/no shaping/noinactive pages; PBpressure pays explicitcoldreset. Ordinaryablation onlyg8fixed uses28us everywave plusPBcost.",
        data="TRAIN32prompts/512inputs/3layers; plus16previouslyobserved authored diagnosticprompts/256inputs/3layers. This design was proposed afterdiagnosticmatchingfailures; diagnosticreplayisreuse, NOT a newunseenvalidation. No layout/solver selection from these outcomes.",
        comparators="SameGU,arraymode,inputcopies,priority frozen g4triples and sparseg2 where available. Everyprimaryratio reports payloadincrease; noequalpayloadsuperiorityclaim. Nativeg8referencesGFonly.",
        FIFO="All primaryqueries must befinitePBvalid and<=128B perchip. Retain eventmean/P99,aligned3layersumP99,everypairedregression and resourcecounts.",
        limitations="Conditional service model, W8/A8reference,noactualdevice/fullmodel/qualityclaim. One manuallyinstantiatedshape,standardCBC. No globalcapacity/fullFFNoptimum; commonrelease Down-only89.94us stagecertificate excludesGUandtransfers. Matchingruntime notadded."))
    print(json.dumps(dict(status="frozen",protocol_sha256=f.sha(OUT/"protocol.frozen.json"))),flush=True)


def anchors(diagnostic):
    if diagnostic:
        path=BASE/"prospective-full-ffn/service_arrays.npz"
        with np.load(path) as z:
            return {key:z[key].copy() for key in z.files}
    result={}
    for path in [r.OUT/"TRAIN512/service_arrays.npz",g2.OUT/"TRAIN512/service_arrays.npz",
                 native.OUT/"service_arrays.npz"]:
        with np.load(path) as z:
            for key in z.files:
                result[key]=z[key].copy()
    return result


def run(diagnostic=False):
    protocol=f.read(OUT/"protocol.frozen.json")
    for path,h in protocol["inputs_sha256"].items():
        assert f.sha(path)==h,path
    # Builderverification is a required correctness gate, not an outcome filter.
    verification=f.read(joint.OUT/"report.json")
    assert verification["status"].startswith("pass"),verification["status"]
    dest=OUT/("diagnostic-reuse" if diagnostic else "TRAIN512")
    dest.mkdir(exist_ok=True)
    assert not (dest/"report.json").exists()
    if diagnostic:
        with np.load(BASE/"prospective-routes/decode.routes.npz") as z:
            assert z["completed"].all()
            routes=z["routes"].transpose(2,0,1,3).reshape(40,256,6)[:3]
    else:
        routes,_,_,_=r.inputs()
    with np.load(r.BASE/"g8-addendum/group_maps.frozen.npz") as z:
        groups=z["learned"].copy()
    layout=f.read(joint.OUT/"layout.frozen.json")
    data_anchors=anchors(diagnostic)
    ninputs=routes.shape[1]
    f.dump(dest/"run.frozen.json",dict(utc=f.utc(),source_sha256=f.sha(__file__),
        protocol_sha256=f.sha(OUT/"protocol.frozen.json"),builder_verification_sha256=f.sha(joint.OUT/"report.json"),
        samples=ninputs*3,axes=axes(diagnostic),data_scope="diagnostic reuse" if diagnostic else "TRAIN reuse"))
    arrays,records,cache={},{},{}
    recs=[]
    began=time.monotonic()
    executor=f.Executor(OUT)
    pipelines={}
    for ci,(family,array,copies,df) in enumerate(axes(diagnostic)):
        name=f"{family}.{array}.joint8.input{copies}.{'DF' if df else 'GF'}"
        key=(family=="g8_fixed",array)
        if key not in pipelines:
            pipelines[key]=native.Recipes(f.load_pipeline(),a.load_shaper(),array) if key[0] else r.Recipes(f.load_pipeline(),a.load_shaper())
        recipes=pipelines[key]
        costs=np.full((3,ninputs),-1,dtype=np.int64)
        fifos=costs.copy()
        aggregate=Counter()
        for layer in range(3):
            for qi,route in enumerate(routes[layer]):
                decision=joint.assign_route(route,layout)
                assert decision["feasible"] and decision["group_load"]==[1]*8
                jobs=decision["selected_jobs"]
                assert len(jobs)==16 and len({j["chip"] for j in jobs})==16
                maps={family+".joint8":[jobs]*3}
                graph=r.graph_for(route,layer,family,"joint8",f.Service(copies,df,0),recipes,groups,maps)
                assert graph is not None,(name,layer,qi)
                signature,data=graph.signature()
                signature=(signature,df,array,json.dumps(dict(graph.pipeline_totals),sort_keys=True))
                if signature in cache:
                    value,audit=cache[signature]
                else:
                    result=executor.run(graph,data)
                    audit=f.audit(graph,result,full=qi==0)
                    value=result["cost_ns"]
                    cache[signature]=value,audit
                    if qi==0:
                        f.dump_trace(dest/f"trace.L{layer}.{name}.json",graph,result,audit)
                costs[layer,qi]=value
                fifos[layer,qi]=audit["peak_output_fifo_bytes"]
                for count,value in graph.pipeline_totals.items():
                    aggregate[count]+=value
        assert np.all(costs>=0) and np.all(fifos<=128)
        arrays[name+".cost"]=costs
        arrays[name+".fifo"]=fifos
        priority="DF" if df else "GF"
        if family=="g8_fixed":
            refnames=[f"{array}.g4triples.input{copies}.GF",f"{array}.g2availability.input{copies}.GF"] if not diagnostic else []
        elif diagnostic:
            refnames=[f"{family}.g4triples.input{copies}.GF",f"{family}.g2_fitted.input{copies}.GF"]
        else:
            refnames=[f"{family}.triple_private_shared4.input{copies}.{priority}",
                      f"{family}.g2availability.input{copies}.{priority}"]
        comparisons=[]
        for ref in refnames:
            anchor=data_anchors[ref+".cost"]
            comparisons.append(dict(reference=ref,mean_reduction_pct=100*(1-float(costs.mean()/anchor.mean())),
                event_P99_reduction_pct=100*(1-float(np.percentile(costs,99)/np.percentile(anchor,99))),
                three_layer_sum_P99_reduction_pct=100*(1-float(np.percentile(costs.sum(axis=0),99)/np.percentile(anchor.sum(axis=0),99))),
                slower_events=int(np.count_nonzero(costs>anchor)),equal_events=int(np.count_nonzero(costs==anchor))))
        rec=dict(name=name,completed_events=int(costs.size),cost_ns=f.stats(costs),
                 three_layer_sum_ns=f.stats(costs.sum(axis=0)),FIFO_peak_bytes=f.stats(fifos),
                 payload_MiB=303,matching_failures=0,comparisons=comparisons,
                 pipeline_counts_per_event={count:value/costs.size for count,value in aggregate.items()})
        recs.append(rec)
        progress=dict(finished=ci+1,total=len(axes(diagnostic)),last=name,elapsed_seconds=time.monotonic()-began)
        f.dump(dest/"progress.json",progress)
        print(json.dumps(progress),flush=True)
    np.savez_compressed(dest/"service_arrays.npz",**arrays)
    f.dump(dest/"report.json",dict(status="complete_conditional_joint_shared_replay",records=recs,
        source_sha256=f.sha(__file__),protocol_sha256=f.sha(OUT/"protocol.frozen.json"),
        elapsed_seconds=time.monotonic()-began,unique_graphs=len(cache),
        data_scope="Previously observed authored diagnostic prompts reused after design, not fresh validation" if diagnostic else "Previously observed TRAIN",
        scope=protocol["limitations"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["freeze","train","diagnostic"])
    action=parser.parse_args().mode
    freeze() if action=="freeze" else run(action=="diagnostic")
