#!/usr/bin/env python3
"""Capacity-only arithmetic memo. No native costs, selectors or inference run."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
BASE = HERE.parents[1]
REPO = HERE.parents[4]
MI = 2**20
W = 65536


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(name, obj):
    (HERE/name).write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n')


def main():
    gu = BASE/'g4-elastic-prefix-screen/work-v3-independent-prefix/contract_arrays.npz'
    down = BASE/'g4-down-paired-merge-screen/arrays.npz'
    with np.load(gu) as g, np.load(down) as d:
        rows, columns = g['rows'], d['input_columns']
        assert np.array_equal(rows, d['rows'])
        assert rows.shape == (65,2,16) and columns.shape == (65,2)
        src = g['sources'][3]
        edges = d['layouts'][14]
        assert src.shape == (64,8) and edges.shape == (64,6,2)
        assert all(len(set(x)) == 8 for x in src)
        assert np.all(rows[np.arange(64)[:,None],0,src] > 40)
        base_waves = ((rows*columns[:,:,None]+W-1)//W).sum((0,1))
        source_count = np.bincount(src.reshape(-1), minlength=16)
        down_count = np.bincount(edges[:,:,1].reshape(-1), minlength=16)
        base_down = base_waves+down_count
        joint = base_down+source_count
        full_A1 = base_down+3*source_count
        full_A12 = base_down+192
        copied_gu = int(((g['resident_rows']-rows)*columns[:,:,None]).sum())
        copied_down = 384*W
        useful_base = int((rows*columns[:,:,None]).sum())
    assert int(base_waves.sum())==2440
    assert int(source_count.sum())==512 and int(down_count.sum())==384
    assert int(joint.sum())==3336 and int(full_A1.sum())==4360 and int(full_A12.sum())==5896
    assert int(base_down.max())<=192
    assert np.all((joint+191)//192==2)
    assert np.all((full_A1+191)//192==2) and np.all((full_A12+191)//192==2)
    def inventory(v):
        return dict(per_chip_waves=v.tolist(), resident_waves=int(v.sum()),
                    resident_wave_MiB=int(v.sum())/16,
                    per_chip_count_lower_bound_blocks=((v+191)//192).tolist(),
                    count_lower_bound_total_blocks=int(((v+191)//192).sum()))
    count_pairs=[(nG,nD) for nG in range(65) for nD in range(65) if 24*nG+12*nD<=896 and (1.5*nG+.75*nD)*MI<=copied_gu+copied_down]
    write('arithmetic.json', dict(created_utc=datetime.now(timezone.utc).isoformat(),
        inputs={str(p):sha(p) for p in [gu,down]}, no_native_or_route_replay=True,
        candidate_scope='Fixed GU v3 layout3 + Down paired graph14, G4L0;64routed experts and unchanged shared included in base inventory.',
        base_useful_MiB=useful_base/MI,
        candidate_extra=dict(GU_useful_MiB=copied_gu/MI,Down_useful_MiB=24,
            total_useful_MiB=(copied_gu+copied_down)/MI, pages=896*4, waves=896,
            wave_MiB=56),
        base=inventory(base_waves), base_plus_Down14=inventory(base_down),
        A1_source_count_per_chip=source_count.tolist(), Down14_append_count_per_chip=down_count.tolist(),
        joint_prefix=inventory(joint), complete_A1_plus_Down14=inventory(full_A1),
        complete_A1_A2_plus_Down14=inventory(full_A12),
        same32blocks_allocated_LSB_MiB=32*192/16,
        exact_wave_menu_count_inequality='2*nGU+nDown<=74,0<=nGU,nDown<=64, one extra complete copy per expert/projection for this counting menu.',
        vector_feasible_count_pairs=len(count_pairs), examples=[dict(nGU=a,nDown=b,waves=24*a+12*b,useful_MiB=1.5*a+.75*b) for a,b in [(37,0),(21,32),(5,64)]],
        count_menu_warning='Counts are not placement witnesses. IDs,profiles,chip capacity and contiguous native row images must also be certified. Coefficient bound is nonbinding for this exact-wave menu because at most888waves/55.5MiB can be used.',
        simple_packing_construction='On each chip, concatenate the maximal base GU/shared and Down14 core+append images in block0; their total is<=189waves. Store A1 full GU images in block1 for the service-matching control:3*source_count<=117waves. For A1+A2, each of64experts has one128-row full-GU image on every chip across its complementary8chip masks, hence64*3=192waves in block1. Each constituent image stays wave-aligned and inside its block. No baseline block address immutability is required.',
        proof_limits='Arithmetic establishes a simple packing construction under the same ordinary wave-aligned native model, not an independently emitted/validated address array or a device result. Boundary agent is responsible for actual rowID/address certificate. Native model is address-invariant within a block; physical latency heterogeneity and unavailable output ABI are not resolved here.'))

    notes = dict(created_utc=datetime.now(timezone.utc).isoformat(),
      status='Capacity fairness memo; no large control fit/native execution',
      conclusion='A vector-budget comparison and a same-allocated-LSB comparison answer different questions. They must both be reported. On these frozen single-layer inventories, full A1 plus Down14, and even full complementary A1+A2 plus Down14, admit the same32LSBblocks as the prefix candidate. Therefore a single-layer allocated-capacity saving cannot be inferred from the56MiB programmed increment.',
      budgets=[
        dict(name='Explicit resource-vector dominance control',definition='Require useful coefficient bytes,programmed pages,resident waves and allocated blocks each not to exceed the candidate, with per-chip/fault-domain constraints declared. Keep baseline resident weights/metadata and shared geometry common.',answers='How much latency can be bought with the same explicit multi-resource increment? This is legitimate but stricter than capacity-only fairness.',not_equivalent='Do not label this the only same-physical-capacity control when available allocated blocks contain usable holes.'),
        dict(name='Same allocated LSB physical capacity control',definition='Permit the baseline to fill any legal holes in the same per-chip allocated blocks. Useful/pages/waves may exceed the candidate while total allocated LSB space and hardware capacities stay equal. Repack all resident maximal row images; do not demand old offsets remain fixed.',answers='Does prefix sharing buy performance or model fit at equal physical capacity?',not_equivalent='A total block-count match alone is insufficient if it changes per-chip or LUN budgets. This case has a constructive2blocks/chip control; still emit/check addresses and one-copy+replica row identities.'),
        dict(name='Full-model global packing/Pareto',definition='Pack all40layers and unchanged shared/base weights together, within existing per-chip/native one-block constraints. Compare global allocated blocks,per-chip capacities and model fit as well as useful/pages/waves.',answers='Whether page/wave savings persist after block-step slack is amortized.',not_equivalent='Neither32L0blocks*40 nor the single-layer extra896W*40 is a proof of globally optimal allocated capacity; report bounds and feasible physical certificates separately.')],
      primary_service_match_control=dict(name='Separate full A1 GU images + original B + Down14',
        construction='Keep original B cores and Down14 paired images. Store every selected A1 source128-row GU image separately, with EXACT Bcore|sameappend row order. A long source request reads its A1copy; a short source reads original Bcore; offsource B-prefix requests and logical row scatter remain as in candidate.',
        selector='Replay exactly the candidate per-query per-source u bits, including dependent offsource ranges, and Down14merge bits in the original job order. This preserves the candidate policy freedom; do not replace it by only one whole-expert A/B bit.',
        equivalence_scope='Same C/rows/pages/waves/commands/input/output and GU->fullA8->Down descriptors imply identical service in the current address-invariant native model. Short-prefix fringe values may differ but discarded outputs cannot alter valid rows under the already-required row-separable/reset contract. No new native equivalence or numeric hardware measurement was executed.',
        attribution='Prefix sharing saves useful/programmed/resident-wave storage versus this control. Runtime latency comes from adaptive geometry/packet placement/selector. Equal allocated blocks plus service matching removes a single-layer allocated-capacity advantage, not every possible full-model or programmed-byte advantage.'),
      stronger_control=dict(name='Add full complementary A2 GU images',
        capacity='Exactly192GUbackup waves/chip for A1+A2; base+Down14 at most189waves/chip. Simple two-block packing exists in this model.',
        performance='Additional source choices may help but require fair TRAIN-only fitting and the same native coalescing/barriers. No measured or strict performance domination claimed; the original A1 policy is retained as an available control.'),
      required_if_joint_passes_gate=[
        'First emit the service-matching A1 physical image/address+rowID certificate and replay identical candidate bits/job order. This is an attribution check,not a new optimization campaign.',
        'Fit whole-GU and whole-Down alternatives independently under the explicit vector budget;do not force whole-FFN backups. Include37GU,21GU+32Down,and5GU+all64Down allocation corners if chip packing allows. Fit expert IDs,chip masks,legal geometries and choices with equal TRAIN effort.',
        'Run same-allocated controls starting from the exact-match A1 incumbent,then A1+A2 and selective complete/packet replicas permitted by remaining holes. Preserve all incumbents and compare marginal-hotness/routeblind plus query-aware fits.',
        'Sweep a small capacity Pareto: no-copy,GUonly512W,Downonly384W,joint896W, and the completeA1/A1+A2 physical budgets. Report useful bytes,programmed pages,resident waves,per-chip allocated blocks,global fit and latency together.',
        'Retain strong one-copy/static mixed geometry and same-image static packet/view controls;keep shared geometry,fullGU/fullA8 barrier,input fanout,command costs,host scatter and selector costs identical.',
        'Before a full-model capacity claim,global-pack every family and report feasible plans/lower bounds. A single-layer block step cannot establish or erase all model-wide savings.'
      ],
      reusable_existing_selectors=[
        dict(path=str(REPO/'scripts/research_20261103_g4_independent_prefix_native.py'),reuse='GU v3 independent-source u_i assembly, computed-versus-consumed row ledger and dependent offsource prefixes;relabel long reads to full A1 addresses while preserving the exact selected job order.',limit='Existing native script supplies finite greedy/representative evaluation,not an optimal copy selector or a generalized full-model fit.'),
        dict(path=str(REPO/'scripts/research_20261103_g4_down_paired_native_selected.py'),reuse='Graph14 merge_bits,complete128/256 row images,fullGU barrier and trace comparison scaffolding.',limit='Retain the packet-consistency constraints;flow/LP relaxation is not itself a feasible runtime policy.'),
        dict(path=str(REPO/'scripts/research_20261103_g4_down_copy_oracle.cpp'),reuse='Exact64choice necessary minmax-work oracle for six whole-expert binary choices,usable for simple backup placement screens.',limit='Not an exact selector for84independent GU+Down bits/top6 or a native scheduling optimum.'),
        dict(path=str(REPO/'scripts/research_20261103_g4_free_A_prefix_screen.py'),reuse='TRAIN-only source-mask/replica-location and static incumbent fitting patterns;retain same source/copy-budget accounting.',limit='Its original free-mask search is not same-budget constrained;add exact physical budgets before labeling a fair control.')],
      prior_scope='Reuse the immediately preceding primary-source audits of query-log replicated declustering,overlapping fragmentations,HDA and Tutel;no fresh literature search or unread-fulltext claims. These establish the control families,not the NAND native contract.',
      no_new_claim='No new algorithm/theorem,hardware extension,manufactured-chip performance,global optimum or heldout result claimed.')
    write('notes.json', notes)
    write('completion.json',dict(created_utc=datetime.now(timezone.utc).isoformat(),files={str(p.relative_to(HERE)):sha(p) for p in sorted(HERE.iterdir()) if p.is_file() and p.name!='completion.json'}))
    print(json.dumps(dict(notes_sha256=sha(HERE/'notes.json'),arithmetic_sha256=sha(HERE/'arithmetic.json'),extra_useful_MiB=(copied_gu+copied_down)/MI),indent=2))


if __name__=='__main__':main()
