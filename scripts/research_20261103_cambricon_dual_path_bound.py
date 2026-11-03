#!/usr/bin/env python3
"""Cambricon-S atomic work ledger and symbolic dual-path necessary constraints."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'results/research-2026-11-02/round2'
OUT = BASE/'cambricon-existing-dual-path-bound'
CASES = {'granite4tiny': (1536, 512, 1024), 'deepseek': (2048, 1408, 2816)}
PRIMARY = BASE/'qwen38-placement/joint-prior/cambricon.html'
TEXT = PRIMARY.with_suffix('.txt')


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def dump(path, value): Path(path).write_text(json.dumps(value, indent=2)+'\n')
def record(path): return dict(path=str(path), sha256=sha(path), bytes=path.stat().st_size)


def freeze():
    OUT.mkdir(exist_ok=False)
    refs = [Path(__file__), PRIMARY, TEXT, BASE/'existing-capability-mechanism-screen/notes.json',
            BASE/'chip-count-regime-screen/m32-prospective-tail-validation/independent/topology-portability/notes.json']
    refs += [BASE/f'w8-format-training/{model}/decode.routes.npz' for model in CASES]
    shutil.copy2(__file__, OUT/Path(__file__).name)
    dump(OUT/'protocol.frozen.json', dict(
        status='source_and_counting_rules_frozen_before_TRAIN_payload_read', utc=datetime.now(timezone.utc).isoformat(),
        source_sha256=sha(__file__), references=[record(p) for p in refs],
        scope='Existing TRAIN only, first MoE route layer, 32prompts x16decode x6experts +shared; no new inference, heldout, fit, native backend or numerical execution.',
        primary='https://arxiv.org/html/2409.15654v1 sectionsIV-B/C,V,VII-A/C TableII/IV',
        hardware=dict(channels=8, chips_per_channel=2, dies_per_chip=2, cores_per_die=1, planes_per_die=2,
                      page_bytes=16384, NAND_read_ns=30000, channel_bytes_per_second=1000000000,
                      NPU_reported_peak_ops_per_second=2000000000000, combined_core_input_output_buffer_bytes=2048),
        tiling='Fixed64rows x256columns W8 atomic pages. Row-major tile identity, original complete logical rows; pad only matrix-edge zero columns/rows, never omit bytes. Column slices reduced in original increasing-column order symbolically; no rawI32/output/reset ABI assumption.',
        controls='Two routeblind onecopy uniform cyclic atomic layouts: cyclic local tile ordinals; cyclic global ordinals across all64experts+shared. Every complete atomic tile has one address/plane, no replicas. Best static global PIM fraction must optimize the same symbolic plane/core/channel/NPU resources; original paper alpha retained as analytic comparator, not assumed universally optimal.',
        oracle='Perfect route-aware per-atomic read/PIM choice, no storage capacity or precedence overhead. Same physical page fetch is necessary on either path; same plane, shared core, channel, NPU constraints retained symbolically. No simultaneous core ownership by two planes.',
        calibration='Core occupancy time, external result bytes/scale/reset/merge ABI, request command timing, NPU SRAM capacity and effective utilization unavailable. Do not substitute NANDtR for core occupancy or infer actual PEcount from a20us illustrativeexample. Publish symbolic LP, plus a deliberately coarse safe known-resource floor; cannot derive actual latency/gain ceiling without calibrated comparator.',
        stop='No heavy simulator. If necessary resource headroom is unresolved due missing hardware service contract, seal as uncalibrated; never present differences between two lowerbounds as an upperbound on actual latency gain.'))
    print(json.dumps(record(OUT/'protocol.frozen.json')))


def tile_image(shapes, style):
    # expert,phase,row_start,col_start,valid_rows,valid_cols,core,plane,channel
    tiles = []
    ordinal = 0
    for expert in range(65):
        pair = shapes[2:] if expert == 64 else shapes[:2]
        local = 0
        for phase, (R, C) in enumerate(pair):
            for row in range(0, R, 64):
                for column in range(0, C, 256):
                    index = local if style == 'local_cyclic' else ordinal
                    owner = index % 64; core = owner % 32; plane = owner//32
                    tiles.append([expert, phase, row, column, min(64,R-row), min(256,C-column), core, plane, core%8])
                    local += 1; ordinal += 1
    return np.asarray(tiles, np.int32)


def summary(values):
    return dict(min=float(np.min(values)), mean=float(np.mean(values)), p99=float(np.quantile(values,.99)), max=float(np.max(values)))


def run():
    protocol = json.loads((OUT/'protocol.frozen.json').read_text())
    assert sha(__file__) == protocol['source_sha256']
    for ref in protocol['references']: assert sha(ref['path']) == ref['sha256']
    reports = []; arrays = {}
    for model, (H,I,SI) in CASES.items():
        path = BASE/f'w8-format-training/{model}/decode.routes.npz'
        with np.load(path,allow_pickle=False) as z:
            assert str(z['split']) == 'train' and z['completed'].all()
            assert all(str(x).startswith('train_') for x in z['prompt_ids'])
            routes = z['routes'][:,:,0].reshape(512,6).copy(); ids=z['prompt_ids'].copy()
        assert np.all((routes>=0)&(routes<64)) and np.all(np.diff(np.sort(routes,axis=1),axis=1)>0)
        shapes = [(2*I,H),(H,I),(2*SI,H),(H,SI)]
        projections = []
        for name,(R,C) in zip(('routed_GU','routed_Down','shared_GU','shared_Down'),shapes):
            row_tiles=(R+63)//64; col_tiles=(C+255)//256; pages=row_tiles*col_tiles
            projections.append(dict(projection=name, rows=R, columns=C, atomic_row_tiles=row_tiles,
                atomic_column_tiles=col_tiles, atomic_pages=pages, useful_coefficient_bytes=R*C,
                stored_page_bytes=pages*16384, zero_padding_bytes=pages*16384-R*C,
                useful_MACs=R*C, dense_padded_MAC_slots=pages*16384,
                column_slice_reduction_order=list(range(col_tiles)), original_row_scale_ID='expert,projection,logicalrow'))
        pages = 6*sum(p['atomic_pages'] for p in projections[:2])+sum(p['atomic_pages'] for p in projections[2:])
        useful=6*3*H*I+3*H*SI; padded=pages*16384
        plane_floor=int(np.ceil(pages/64))*30000
        # Paper's stated equations: preserve their request units, do not silently
        # reinterpret the 32-page global request as one core's compute time.
        hreq,wreq=256,2048; trc=30000+wreq/(8*1.)
        rate=(hreq+wreq/8)/30000; tr=16384/(1-rate)
        alpha=tr/(tr+trc)
        report=dict(model=model, queries=512, first_MoE_raw_layer=0 if model=='granite4tiny' else 1,
            projections=projections, per_query_atomic_pages=pages, per_query_useful_coefficient_bytes=useful,
            per_query_page_bytes=padded, per_query_zero_padding_bytes=padded-useful,
            per_query_useful_MACs=useful, per_query_dense_padded_MAC_slots=padded,
            phase_atomic_pages=[6*projections[0]['atomic_pages']+projections[2]['atomic_pages'],6*projections[1]['atomic_pages']+projections[3]['atomic_pages']],
            route_label_independent_total_count=True, best_any_address_64plane_array_service_floor_ns=plane_floor,
            read_compute_shared_core='Onecore/die; actual occupancy tau_core_ns/atomic is unspecified, not NANDtR.',
            layouts=[])
        for style in ('local_cyclic','global_cyclic'):
            image=tile_image(shapes,style)
            expected=64*sum(p['atomic_pages'] for p in projections[:2])+sum(p['atomic_pages'] for p in projections[2:])
            assert len(image)==expected
            assert np.sum(image[:,4].astype(np.int64)*image[:,5]) == 64*3*H*I+3*H*SI
            assert len(set(tuple(t[:4]) for t in image))==len(image)
            # Every byte of each original matrix is covered exactly once by its
            # disjoint row/column grid. Padding belongs to no extra logical row.
            expert_core=np.zeros((65,32),np.int64); expert_plane=np.zeros((65,64),np.int64)
            expert_channel=np.zeros((65,8),np.int64)
            for tile in image:
                e,_,_,_,_,_,core,plane,ch=map(int,tile)
                expert_core[e,core]+=1; expert_plane[e,core+32*plane]+=1; expert_channel[e,ch]+=1
            selected=np.zeros((512,65),np.int64)
            selected[np.arange(512)[:,None],routes]=1;selected[:,64]=1
            plane=selected@expert_plane; core=selected@expert_core; channel=selected@expert_channel
            assert np.all(plane.sum(1)==pages) and np.all(core.sum(1)==pages) and np.all(channel.sum(1)==pages)
            key=model+'.'+style
            arrays[key+'.resident_tile_ids']=image; arrays[key+'.plane_pages']=plane
            arrays[key+'.core_pages']=core;arrays[key+'.channel_pages']=channel
            arrays[key+'.expert_core_pages']=expert_core;arrays[key+'.expert_plane_pages']=expert_plane;arrays[key+'.expert_channel_pages']=expert_channel
            # With all transfer/core/command overhead removed, all-PIM is legal
            # only in the relaxation, and saturates the known plane floor.
            coarse=plane.max(1)*30000
            arrays[key+'.known_resource_relaxation_ns']=coarse
            report['layouts'].append(dict(name=style, onecopy=True, resident_pages=len(image),
                resident_useful_bytes=int(np.sum(image[:,4].astype(np.int64)*image[:,5])),
                resident_page_bytes=len(image)*16384, resident_zero_padding_bytes=len(image)*16384-int(np.sum(image[:,4].astype(np.int64)*image[:,5])),
                per_query_max_plane_pages=summary(plane.max(1)),per_query_max_core_pages=summary(core.max(1)),
                per_query_max_channel_pages=summary(channel.max(1)),known_resource_relaxation_ns=summary(coarse),
                static_and_oracle_known_resource_relaxations_equal=True,
                equality_scope='Set unknown core service/output/input/command/precedence to zero. Both beststatic andperfectsplit chooseallPIM; equality is only this coarse LB, not an actual gain ceiling.'))
        arrays[model+'.routes']=routes;arrays[model+'.prompt_ids']=ids
        report['paper_equations']=dict(global_tile_rows=hreq,global_tile_columns=wreq,atomic_pages_per_global_tile=32,
            t_rc_ns=trc,rate_rc=rate,t_r_ns=tr,alpha_read_compute=alpha,
            unit_caution='t_rc is globaltile service with32atomicpages; t_r numerator is1page. Preservepublishedformula, but no dimension-corrected optimum or physical minmax guarantee inferred.')
        reports.append(report)
    np.savez_compressed(OUT/'arrays.npz',**arrays)
    dump(OUT/'report.json',dict(status='sealed_atomic_counts_symbolic_only_uncalibrated_service',protocol_sha256=sha(OUT/'protocol.frozen.json'),models=reports,
        symbolic_LP=dict(variables='x_i in[0,1] PIM fraction/choice for each selected full atomic tile; T latency. Static comparator requires one fixed fraction and routeblind fixed onecopy tile addresses; oracle may choose x_i perquery, storage/precedence relaxed.',
            plane='For every physicalplane p: T >= 30000 * number of all selected pages on p (both read andPIM fetch once). Distinctplanes may read only as explicitly allowed; this lowerbound imposes no invented overlap.',
            core='For eachdie/core d: T >= tau_core_ns * sum_{tiles on d} x_i; tau_core is unknown and shared by bothplanes, never two cores/contexts per die.',
            channel='For eachchannel h: 1e9*T/1e9 >= 16384*sum_{tiles on h}(1-x_i) + actual_input_bytes_h(x) + actual_result_bytes_h(x). Inputs canbroadcast only whenidentical, perV-A; bytes/outputdtype/amortization notinvented. Dropping unknowntraffic gives safe weaker bound.',
            NPU='T >= (2*16384 /2000) * sum_i(1-x_i) ns at reported2TOPS peak; usingpaddedMACs assumesoriginaldenseatomicexecution. MoreconservativeusefulMACbound may replace16384 withvalidrows*validcols; NPU16x16@1GHz prose conflicts with2TOPS, so no effective-utilization claim.',
            buffers='Fixedcombined2KiB coreinput+output, unknownpartition; NPUcapacity absent. 256B A8input fits total but outputbuffer anddtype notverified. Oracle relaxes buffer/storageconstraints, never claims capacityfeasible.',
            reduction='Originalfixedatomiccolumnsliceorder preserved symbolically; completeGU/wholevectorA8/privateDown barrier retained incontract but ignored only fororacle lowerbound. No rawI32outputs/partialA8/newnumericexecution.'),
        conclusion='No calibrated quantitative placement-only latencyheadroom canbe deduced: missingcoreoccupancy/resultABI/commands/NPUbuffers prevent a feasiblematchedstaticservice comparator. ThecoarseknownresourceLBequality is not proof of <=2%/5% actualgain. Stopbeforebackend/native/fit; no firsthybrid/firsttiling contribution.',
        comparison_gate='Unresolved/uncalibrated, not positivegain andnot globalNO-GO. Anyfuture test mustfirstpin servicecontract andthen compareoriginaloptimalstaticfraction+2Datomicuniforminterleave usingidenticalresources/reductionorder.'))
    for ref in protocol['references']: assert sha(ref['path']) == ref['sha256']
    dump(OUT/'completion.frozen.json',dict(status='sealed_bounded_symbolic_screen_no_backend',utc=datetime.now(timezone.utc).isoformat(),files=[record(p) for p in sorted(OUT.rglob('*')) if p.is_file()]))
    print(json.dumps(dict(report_sha256=sha(OUT/'report.json'),completion_sha256=sha(OUT/'completion.frozen.json'),
        counts={r['model']:{'pages':r['per_query_atomic_pages'],'padding':r['per_query_zero_padding_bytes'],'array_floor_ns':r['best_any_address_64plane_array_service_floor_ns']} for r in reports}),indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('mode',choices=('freeze','run'));args=parser.parse_args()
    {'freeze':freeze,'run':run}[args.mode]()
