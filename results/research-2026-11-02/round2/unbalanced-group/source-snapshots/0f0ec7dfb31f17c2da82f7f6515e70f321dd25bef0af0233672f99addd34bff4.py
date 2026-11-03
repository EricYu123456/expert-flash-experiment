#!/usr/bin/env python3
"""Independently check the frozen strong baseline rows, true costs and capacity."""
from datetime import datetime, timezone
import json
from pathlib import Path
import shutil

import numpy as np

from materialize_plan import WaveAllocator
from research_20261102_projection_study import equal_rows,evaluate,scalar_cost
from research_20261102_unbalanced_group import DEST,PHYSICAL_CAP,inputs,write
from run_corpus_traces import ROOT,sha256


def main():
    audits=[]
    for model,kind in [('granite4tiny','screen'),('granite','screen'),('qwen3','screen'),('granite','full')]:
        path=DEST/(model+'.'+kind+'.strong.frozen.json');before=sha256(path);doc=json.loads(path.read_text())
        source_path,meta,route_path,routes,ids,layers,original_phases=inputs(model)
        assert sha256(source_path)==doc['source_plan_sha256']
        assert ids==doc['train_ids'] and len(ids)==32 and routes.shape[1]==64
        assert sha256(route_path)==doc['training_routes_sha256']
        original={(p['layer'],p['name']):p for p in original_phases}
        chosen={(p['layer'],p['name']):p for p in doc['phases']}
        total=balanced=old=0.
        for key,p in chosen.items():
            li=layers.index(p['layer']);selected=np.sort(routes[:,:,li].reshape(-1,meta['active']),axis=1)
            rows=np.asarray(p['chosen_rows']);g=p['chosen_g']
            templates={tuple(row) for row in equal_rows(p['total_rows'],16//g,g)}
            assert rows.shape==(meta['experts'],16)
            assert all(tuple(row) in templates for row in rows)
            assert np.all(rows.sum(axis=1)==p['total_rows'])
            values=evaluate(selected,rows,p['columns'],p['common'])
            assert abs(float(values['model_ns'].mean())-p['chosen_ns'])<1e-6
            balanced_values=evaluate(selected,np.asarray(p['balanced_refit_rows']),p['columns'],p['common'])
            assert abs(float(balanced_values['model_ns'].mean())-p['balanced_refit_ns'])<1e-6
            for idx in [0,len(selected)//3,len(selected)//2,len(selected)-1]:
                for k,v in scalar_cost(selected[idx],rows,p['columns'],p['common']).items():assert v==values[k][idx]
            prior=np.asarray(original[key]['rows'])
            np.testing.assert_array_equal(((rows*p['columns']+16383)//16384).sum(axis=1),((prior*p['columns']+16383)//16384).sum(axis=1))
            total+=p['chosen_ns'];balanced+=p['balanced_refit_ns'];old+=p['original_group_ns']
        alloc=[WaveAllocator() for _ in range(16)];last_end=[0]*16;pages=useful=streams=0
        for p in doc['whole_model_rows']:
            key=(p['layer'],p['name']);rows=np.asarray(p['rows'])
            expected=np.asarray(chosen[key]['chosen_rows'] if key in chosen else original[key]['rows'])
            np.testing.assert_array_equal(rows,expected)
            assert np.all(rows.sum(axis=1)==p['total_rows'])
            for e in range(meta['experts']):
                start=0
                for c,count in enumerate(rows[e]):
                    count=int(count)
                    if count:
                        nbytes=count*p['columns'];npages,nwaves,ncommands,segments=alloc[c].allocate(nbytes)
                        assert npages==(nbytes+16383)//16384 and nwaves==(npages+3)//4
                        assert ncommands==len(segments)
                        for block,wl,nw,offset,npag,plane_counts in segments:
                            first=block*192+wl
                            assert first>=last_end[c] and wl+nw<=192
                            assert sum(plane_counts)==npag and max(plane_counts)==nw
                            last_end[c]=first+nw
                        pages+=npages;useful+=nbytes;streams+=1
                    start+=count
                assert start==p['total_rows']
        chips=[a.summary(c) for c,a in enumerate(alloc)]
        claimed=doc['updated_whole_model_inventory']
        assert pages==claimed['total_physical_pages'] and useful==claimed['total_useful_bytes']
        assert chips==claimed['chips'] and all(c['reserved_lsb_bytes']<=PHYSICAL_CAP for c in chips)
        old_input=json.loads(Path(doc['original_relabel_plan']).read_text())
        original_inventory=old_input['baseline_whole_model_inventory']
        assert pages==original_inventory['total_physical_pages'] and useful==original_inventory['total_useful_bytes']
        assert sha256(path)==before
        audits.append(dict(model=model,scope=kind,path=str(path),sha256=before,phases=len(chosen),
            original_group_ms=old/1e6,balanced_refit_ms=balanced/1e6,strong_unbalanced_ms=total/1e6,
            gain_vs_original_group_pct=100*(1-total/old),additional_gain_vs_balanced_refit_pct=100*(1-total/balanced),
            useful_bytes=useful,physical_pages=pages,streams=streams,
            old_mean_payload_per_chip_bytes=useful/16,
            old_max_reserved_lsb_bytes_per_chip=original_inventory['max_reserved_lsb_bytes_per_chip'],
            max_useful_bytes_per_chip=max(c['useful_int8_bytes'] for c in chips),
            max_reserved_lsb_bytes_per_chip=max(c['reserved_lsb_bytes'] for c in chips),
            actual_cap_per_chip_bytes=PHYSICAL_CAP,
            min_reserved_capacity_headroom_bytes=PHYSICAL_CAP-max(c['reserved_lsb_bytes'] for c in chips),
            unequal_per_chip_inventory_allowed=True,same_total_useful_and_physical_pages=True,
            whole_output_rows_covered_once=True,ordinary_plane_WL_ranges_nonoverlapping=True,
            cpp_objective_rechecked_with_full_train_slice=True,independent_scalar_cost_samples_passed=True,
            heldout_routes_opened=False))
    snapshots=DEST/'source-snapshots';snapshots.mkdir(exist_ok=True)
    sources={}
    for filename in ['research_20261102_unbalanced_group.py','research_20261102_unbalanced_group.cpp','research_20261102_unbalanced_group_refit.py','research_20261102_unbalanced_group_finish.py','research_20261102_unbalanced_group_audit.py']:
        path=ROOT/'scripts'/filename;digest=sha256(path);target=snapshots/(digest+path.suffix)
        if not target.exists():shutil.copy2(path,target)
        sources[filename]=dict(sha256=digest,snapshot=str(target))
    result=dict(status='passed',frozen_utc=datetime.now(timezone.utc).isoformat(),audits=audits,sources=sources,
        historical_runner_snapshot=dict(sha256='7aca2a800cebc32c91f98c27b4616f9a8343071d1b8ccd646f7cefd989391f6b',
            path=str(snapshots/'7aca2a800cebc32c91f98c27b4616f9a8343071d1b8ccd646f7cefd989391f6b.py'),
            difference='After the completed single-relabel runs, the compile helper was changed to hash-addressed atomic .so publication to prevent concurrent compiler races. Optimizer, costs and search parameters were unchanged.'),
        scope='Granite4Tiny first3layers; Granite3.1 first2layers plus all24layers; Qwen first2layers. All fits32 training prompts x64 steps, no new heldout evaluation.',
        interpretation='Relaxing equal expert counts is a stronger known baseline, not a proposed method. Screen and full-model effects differ; only routed FFN service arithmetic is modeled.',
        limitations=['20 GiB/chip is an explicit capacity scenario, not a measured device disclosure; only routed FFN payload is allocated.',
            'No claim of equal per-chip payload, equal reserved bytes, measured page reads, or measured end-to-end speed.',
            'Baseline strengthening followed observation of earlier aggregate holdout outcomes; the original main plan was unchanged.',
            'Balanced controls approximately match proposal budgets; they do not prove global optimality.'])
    write(DEST/'completion.json',result)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
