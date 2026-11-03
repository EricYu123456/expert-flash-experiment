#!/usr/bin/env python3
"""Whole-wave, fixed-source group controls for CR and ordinary reads.

Uses only selected four-plane page rows. No IPR, dummy reads, pre-ECC drop,
programmable source, or new buffers. CR backpressure explicitly ends a run and
pays another cold read. Private kernels remain strict.
"""
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
OUT=BASE/"native-group-control/v2"
SOURCE=ROOT/"scripts/research_20261102_cr_g2_availability_ffn.py"
spec=importlib.util.spec_from_file_location("g2_native_driver",SOURCE)
g2=importlib.util.module_from_spec(spec)
spec.loader.exec_module(g2)
f,r,a=g2.f,g2.r,g2.a


class Recipes(r.Recipes):
    def __init__(self,pipeline,shaper,array_mode):
        super().__init__(pipeline,shaper)
        self.array_mode=array_mode

    def make(self,waves,columns,ranks,source="fixed",pool=None):
        assert source=="fixed"
        for wave in waves:
            assert len(wave["pages"])==4
            assert [p["plane"] for p in wave["pages"]]==[0,1,2,3]
            assert len({p["block"] for p in wave["pages"]})==1
            assert len({p["wl"] for p in wave["pages"]})==1
            assert all(not p.get("unselected",False) for p in wave["pages"])
        settings=dict(cache_copy_ns=500,source_policy="fixed",issue_policy="strict_no_interval",
                      array_mode=self.array_mode,row_bytes=columns,validate_row_stream=True,
                      unselected_policy="reject")
        key=json.dumps(dict(waves=waves,settings=settings,ranks=ranks,GU=pool is not None),sort_keys=True)
        if key in self.cache:
            return self.cache[key]
        result=self.p.simulate(waves,settings)
        choice="strict"
        if not result["feasible"] and pool is not None:
            settings=dict(settings,issue_policy="cold_reset")
            result=self.p.simulate(waves,settings)
            choice="GU_explicit_cold_reset"
        if not result["feasible"]:
            self.cache[key]=None
            return None
        assert result["physical_read_pages"]==result["ecc_pages"]==len(waves)*4
        rows=Counter()
        milestones=[]
        for page in result["page_completions"]:
            owner=page["owner"]
            old=rows[owner]//columns
            rows[owner]+=page["bytes"]
            new=rows[owner]//columns
            if new>old:
                milestones.append((ranks[owner],page["end_ns"],(new-old)*4))
        counts={name:result[name] for name in
                ["waves","pages","bytes","physical_read_pages","physical_read_bytes",
                 "ecc_pages","ecc_bytes","cold_waves","cr_waves"]}
        import hashlib
        recipe=dict(duration=result["makespan_ns"],milestones=milestones,
                    signature=hashlib.sha256(key.encode()).hexdigest(),counts=counts,policy=choice,
                    summary=dict(settings=settings,policy=choice,**counts,finite_PB_feasible=True,
                                 source_policy="fixed",no_FFN_cost_oracle=True,
                                 no_IPR_or_dummy_or_drop=True))
        self.cache[key]=recipe
        return recipe


def freeze():
    OUT.mkdir(exist_ok=True)
    assert not (OUT/"protocol.frozen.json").exists()
    paths=[Path(__file__).resolve(),SOURCE,g2.DRIVER,g2.KERNEL,f.PIPE,f.CPP,f.TRAIN,
           g2.OUT/"resident_layouts.frozen.json",g2.OUT/"support_maps.frozen.npz",
           r.OUT/"Down_job_maps.frozen.json",r.BASE/"g8-addendum/group_maps.frozen.npz"]
    f.dump(OUT/"protocol.frozen.json",dict(utc=f.utc(),status="frozen_before_native_feature_ablation",
        inputs_sha256={str(p):f.sha(p) for p in paths},
        GU="Unchanged TRAIN balanced32expert/group g8 whole-rowGU, fixed0123 complete4plane sameWL waves. No shaping/fill/dummy/discard/permutation/IPR. Strict500 first; if PB fails, explicit coldreset with charged28us and finite-PB recurrence. One common input perchip unchanged.",
        Down="Unchanged frozen g4triples or g2 sparse252instances plus deterministic cap1/cap2 matching; all selected full waves fixed0123, sameWL, strict500, private complete inputs. No private fusion or input sharing.",
        axes=dict(array_mode=["cr","ordinary"],Down=["g4triples","g2availability"],input_copies=[1,4],priority="GF"),
        data="All existing TRAIN first3layers x512 inputs; no new prompts or heldout inspected by this driver, no fitting.",
        capacity="g8GU99WL perplane, g2Down<=252WL allocated below128and inblock1, no overlaps. g4payload244.5MiB, g2payload289.5MiB; same2blocks/chip upperbudget. g2 maximum allocatedWL380 includes conservativeGUreserved128; actualGU+Down351.",
        limits="Conditional timing model still requires correct PB copy/overwrite/ECC/PE/output and runrestart ABI. Removes source/drop/IPR dependencies for these controls only, not all previous candidate results. Ordinary28us page-sensing model is a feature ablation, not every in-flasharchitecture or actualdevice. Known grouping/replication, no newmethod claim."))
    print(json.dumps(dict(status="frozen",protocol_sha256=f.sha(OUT/"protocol.frozen.json"))),flush=True)


def run():
    protocol=f.read(OUT/"protocol.frozen.json")
    for path,h in protocol["inputs_sha256"].items():
        assert f.sha(path)==h,path
    assert not (OUT/"report.json").exists()
    routes,_,_,groups=r.inputs()
    layouts=f.read(g2.OUT/"resident_layouts.frozen.json")
    resident=[g2.resident_jobs(x) for x in layouts]
    oldmaps=f.read(r.OUT/"Down_job_maps.frozen.json")
    arrays,records,cache={},{},{}
    recs=[]
    began=time.monotonic()
    executor=f.Executor(OUT)
    for array in ["cr","ordinary"]:
        recipes=Recipes(f.load_pipeline(),a.load_shaper(),array)
        for copies in [1,4]:
            for plan in ["g4triples","g2availability"]:
                name=f"{array}.{plan}.input{copies}.GF"
                costs=np.full((3,512),-1,dtype=np.int64)
                fifos=costs.copy()
                caps=costs.copy()
                aggregate=Counter()
                for layer in range(3):
                    for qi,route in enumerate(routes[layer]):
                        if plan=="g4triples":
                            gm=oldmaps
                            graphplan="triple_private_shared4"
                            cap=0
                        else:
                            choice=g2.dispatch(route,layouts[layer])
                            selected=[job for expert,group in choice["assignment"].items()
                                      for job in resident[layer][(int(expert),int(group))]]
                            selected+=resident[layer][(64,0)]
                            gm={"g8_fixed.g2availability":[selected]*3}
                            graphplan="g2availability"
                            cap=choice["capacity"]
                        graph=r.graph_for(route,layer,"g8_fixed",graphplan,
                                          f.Service(copies,0,0),recipes,groups,gm)
                        assert graph is not None,(name,layer,qi)
                        signature,data=graph.signature()
                        signature=(signature,array,json.dumps(dict(graph.pipeline_totals),sort_keys=True))
                        if signature in cache:
                            value,audit=cache[signature]
                        else:
                            result=executor.run(graph,data)
                            audit=f.audit(graph,result,full=qi==0)
                            value=result["cost_ns"]
                            cache[signature]=value,audit
                            if qi==0:
                                f.dump_trace(OUT/f"trace.L{layer}.{name}.json",graph,result,audit)
                        costs[layer,qi]=value
                        fifos[layer,qi]=audit["peak_output_fifo_bytes"]
                        caps[layer,qi]=cap
                        for key,value in graph.pipeline_totals.items():
                            aggregate[key]+=value
                assert np.all(costs>=0) and np.all(fifos<=128)
                arrays[name+".cost"]=costs
                arrays[name+".fifo"]=fifos
                rec=dict(name=name,events=1536,cost_ns=f.stats(costs),
                         three_layer_sum_ns=f.stats(costs.sum(axis=0)),FIFO_peak_bytes=f.stats(fifos),
                         fallback_events=int(np.count_nonzero(caps==2)),
                         pipeline_counts_per_event={key:value/costs.size for key,value in aggregate.items()},
                         payload_MiB=244.5 if plan=="g4triples" else 289.5)
                if plan=="g2availability":
                    anchor=arrays[f"{array}.g4triples.input{copies}.GF.cost"]
                    rec.update(mean_reduction_pct=100*(1-float(costs.mean()/anchor.mean())),
                               event_P99_reduction_pct=100*(1-float(np.percentile(costs,99)/np.percentile(anchor,99))),
                               slower_events=int(np.count_nonzero(costs>anchor)))
                recs.append(rec)
                f.dump(OUT/"progress.json",dict(finished=len(recs),total=8,last=name,elapsed_seconds=time.monotonic()-began))
                print(json.dumps(dict(finished=len(recs),total=8,last=name,elapsed_seconds=time.monotonic()-began)),flush=True)
    np.savez_compressed(OUT/"service_arrays.npz",**arrays)
    f.dump(OUT/"report.json",dict(status="complete_conditional_native_feature_ablation",records=recs,
        source_sha256=f.sha(__file__),protocol_sha256=f.sha(OUT/"protocol.frozen.json"),
        elapsed_seconds=time.monotonic()-began,scope=protocol["limits"]))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode",choices=["freeze","run"])
    action=parser.parse_args().mode
    freeze() if action=="freeze" else run()
