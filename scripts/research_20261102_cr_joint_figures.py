#!/usr/bin/env python3
"""Descriptive figures from sealed experiments, never used for selection."""
import hashlib
import importlib.util
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/"results/research-2026-11-02/round3-cr-ipr"
PATHS=dict(
    g4=BASE/"down-replica-full-ffn/TRAIN512/report.json",
    sparse=BASE/"g2-availability-full-ffn/TRAIN512/report.json",
    joint=BASE/"joint-shared-full-ffn/TRAIN512/report.json",
    fixed=BASE/"joint-shared-fixed-capacity-control/TRAIN512/report.json",
    second=BASE/"second-prospective-routes/comparison/report.json",
    native=BASE/"native-group-control/v2/report.json")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    raw={key:json.loads(path.read_text()) for key,path in PATHS.items()}
    records={key:{r["name"]:r for r in doc["records"]} for key,doc in raw.items()}
    plt.rcParams.update({"font.size":10,"axes.spines.top":False,"axes.spines.right":False,"savefig.dpi":180})
    panels=[
        ("TRAIN: speed and extra stored weights",
         [records["g4"]["candidate.triple_private_shared4.input1.GF"],
          records["sparse"]["candidate.g2availability.input1.GF"],
          records["joint"]["candidate.cr.joint8.input1.GF"]],
         ["g4 triples\n244.5 MiB","Sparse g2\n289.5 MiB","Movable shared\n303 MiB"]),
        ("TRAIN: isolate shared placement, same 303 MiB",
         [records["fixed"]["candidate.cr.fixedShared.input1.GF"],
          records["joint"]["candidate.cr.joint8.input1.GF"]],
         ["Fixed shared\n166/1536 fallback","Movable shared\n0/1536 fallback"]),
        ("Second prospective diagnostic: no refitting",
         [records["second"][f"candidate.cr.{plan}.input1.GF"] for plan in ["g4triples","g2_fitted","joint8"]],
         ["g4 triples\n244.5 MiB","Frozen sparse g2\n289.5 MiB","Movable shared\n303 MiB"])]
    fig,axes=plt.subplots(1,3,figsize=(14.4,5.3),sharey=True)
    summary=[]
    for axis,(title,rows,labels) in zip(axes,panels):
        x=np.arange(len(rows))
        mean=[r["cost_ns"]["mean"]/1000 for r in rows]
        tail=[r["cost_ns"]["p99"]/1000 for r in rows]
        b1=axis.bar(x-.18,mean,.34,color="#356d9b",label="Mean")
        b2=axis.bar(x+.18,tail,.34,color="#d39432",label="Event P99")
        axis.bar_label(b1,fmt="%.1f",fontsize=8,padding=3)
        axis.bar_label(b2,fmt="%.1f",fontsize=8,padding=3)
        axis.set_xticks(x,labels,rotation=10,ha="right")
        axis.set_title(title,fontsize=10)
        axis.set_ylim(0,390)
        axis.grid(axis="y",alpha=.18)
        axis.set_axisbelow(True)
        summary.append(dict(panel=title,rows=[dict(name=r["name"],mean_us=m,event_P99_us=t) for r,m,t in zip(rows,mean,tail)]))
    axes[0].set_ylabel("Conditional FFN service time (microseconds)")
    axes[0].legend(frameon=False,loc="upper right")
    fig.suptitle("G4 W8/A8, first 3 MoE layers, input1 / GF: fixed resources and explicit replica cost",fontsize=12)
    fig.text(.035,.025,
        "TRAIN: 32 prompts / 512 aligned inputs / 1,536 layer-events. Second prospective diagnostic: 16 newly authored prompts / 768 events.\n"
        "Conditional NAND/PB/ECC/PE service model; not device timing, full-model inference or task quality. CBC and matching are established components.",fontsize=9)
    fig.tight_layout(rect=(0,.17,1,.93))
    outputs=[]
    for suffix in ["png","pdf"]:
        path=BASE/f"joint-overview.{suffix}"
        fig.savefig(path);outputs.append(path)
    plt.close(fig)

    fig,axes=plt.subplots(1,2,figsize=(9.8,4.9))
    feature=[]
    for axis,array,title in zip(axes,["cr","ordinary"],["CR reads","Ordinary reads (feature ablation)"]):
        old=records["native"][f"{array}.g4triples.input1.GF"]
        new=records["joint"][f"g8_fixed.{array}.joint8.input1.GF"]
        values=[old["cost_ns"]["mean"]/1000,new["cost_ns"]["mean"]/1000]
        bars=axis.bar([0,1],values,width=.58,color=["#7b8690","#3b8b76"])
        axis.bar_label(bars,fmt="%.2f",fontsize=10,padding=4)
        axis.set_xticks([0,1],["g4 triples\n244.5 MiB","Movable shared\n303 MiB"])
        axis.set_ylim(0,max(values)*1.26)
        axis.set_title(title)
        axis.set_ylabel("Conditional mean FFN time (microseconds)")
        gain=100*(1-values[1]/values[0])
        axis.text(.5,.9,f"Mean reduction: {gain:.2f}%",transform=axis.transAxes,ha="center",fontsize=10)
        axis.grid(axis="y",alpha=.18);axis.set_axisbelow(True)
        feature.append(dict(array=array,reference_mean_us=values[0],joint_mean_us=values[1],reduction_pct=gain))
    fig.suptitle("Fixed-source g8 GU: selected same-WL full waves, no IPR or pre-ECC drop",fontsize=11)
    fig.text(.035,.025,"TRAIN input1 / GF; GU PB pressure pays explicit cold restart. Extra payload +58.5 MiB/layer is charged.\n"
             "The ordinary-read gain is small; this is not evidence of a large speedup on every in-flash architecture.",fontsize=9)
    fig.tight_layout(rect=(0,.16,1,.93))
    for suffix in ["png","pdf"]:
        path=BASE/f"native-feature-ablation.{suffix}"
        fig.savefig(path);outputs.append(path)
    plt.close(fig)

    builder=ROOT/"scripts/research_20261102_eight_pair_cbc.py"
    spec=importlib.util.spec_from_file_location("figure_joint_builder",builder)
    joint=importlib.util.module_from_spec(spec);spec.loader.exec_module(joint)
    fig,axis=plt.subplots(figsize=(12.5,4.7))
    example=[]
    for row,route in enumerate([[0,1,2,3,4,5],[58,59,60,61,62,63]]):
        decision=joint.assign_route(route)
        example.append(decision)
        y=2.2-row*1.25
        axis.text(-.3,y+.24,f"Route {row+1}",ha="right",fontsize=10)
        for group,owner in enumerate(decision["group_owners"]):
            color="#d6a449" if owner.startswith("S") else "#6d9dc5"
            axis.add_patch(Rectangle((group,y),.86,.68,facecolor=color,alpha=.9,edgecolor="white"))
            axis.text(group+.43,y+.36,owner,ha="center",va="center",fontsize=12)
            axis.text(group+.43,y-.14,f"chips {2*group}, {2*group+1}",ha="center",va="top",fontsize=8)
    for group in range(8):
        axis.text(group+.43,3.15,f"Pair {group}",ha="center",fontsize=10)
    axis.text(3.9,.2,"Each pair stores 32 expert copies + both shared halves; matching assigns six experts, then fills the two unused pairs.",ha="center",fontsize=9)
    axis.text(3.9,-.1,"S0/S1 split OUTPUT rows, each takes the original full C1024 input. Every selected chip reads 24 pages in six fixed-source waves.",ha="center",fontsize=9)
    axis.set(xlim=(-1,8),ylim=(-.4,3.5));axis.axis("off")
    fig.suptitle("Shared Down follows the free pairs: two illustrative routes (not performance samples)",fontsize=12)
    fig.tight_layout()
    for suffix in ["png","pdf"]:
        path=BASE/f"joint-layout.{suffix}"
        fig.savefig(path);outputs.append(path)
    plt.close(fig)
    record=dict(status="descriptive_only_no_selection",source_sha256=sha(__file__),
                inputs_sha256={str(path):sha(path) for path in [*PATHS.values(),builder]},
                panels=summary,native_feature_ablation=feature,illustrative_routes=example,
                outputs_sha256={str(path):sha(path) for path in outputs})
    (BASE/"joint-figures.json").write_text(json.dumps(record,indent=2)+"\n")
    print(json.dumps(dict(status=record["status"],outputs=[str(p) for p in outputs])))


if __name__=="__main__":main()
