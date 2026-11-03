#!/usr/bin/env python3
"""Public route-ID adapter to the separately sealed macro lowering core."""
from copy import deepcopy
import lowering

def problem_for_experts(kind,expert_ids,shared_geometry='native'):
    ids=sorted(map(int,expert_ids))
    if not ids or len(ids)>6 or len(set(ids))!=len(ids) or min(ids)<0 or max(ids)>=64:
        raise ValueError('Supply1..6unique resident expertIDs0..63.')
    even=sum(e%2==0 for e in ids)
    p=lowering.build_problem(kind,even_count=even,K=len(ids),shared_geometry=shared_geometry)
    groups=[ids] if kind=='candidate' else [[e for e in ids if e%2==0],[e for e in ids if e%2]]
    for typ,experts in zip(p['types'],groups):typ['experts']=experts
    return p

def compile_minimum(kind,expert_ids,source='fixed',shared_geometry='native'):
    """Exact macro-slot optimum, not an assertion of finite-PB feasibility."""
    problem=problem_for_experts(kind,expert_ids,shared_geometry)
    solution=lowering.solve_macros(problem,source)
    waves=lowering.slots_to_waves(problem,solution['slots'],source)
    certificate=lowering.validate_lowering(problem,solution,waves,source)
    return dict(problem=problem,solution=solution,waves=waves,certificate=certificate)

def compile_best_tested(kind,expert_ids,simulate,source='fixed',shared_geometry='native'):
    """Run the declared finite one-extra-wave menu using a caller simulator.

    simulate must be the sealed v1compatible callable. This grants exactly the
    same conditional drop and2weight-slot service contract as the experiment.
    A None best means this finite menu failed; it is not an impossibility proof.
    """
    out=compile_minimum(kind,expert_ids,source,shared_geometry)
    p,s=out['problem'],out['solution']
    choices=[(None,s['slots'])]+list(lowering.one_extra_wave_variants(s,source))
    tested=[];best=None
    for variant,slots in choices:
        waves=lowering.slots_to_waves(p,slots,source)
        certificate=lowering.validate_lowering(p,s,waves,source)
        result=simulate(waves,cache_copy_ns=500,source_policy=source,
                        issue_policy='strict_no_interval',weight_slots=2,
                        unselected_policy='drop_before_ecc_conditional')
        tested.append(dict(variant=variant,waves=len(waves),feasible=result['feasible'],
                           makespan_ns=result['makespan_ns'],violations=result['violations']))
        if result['feasible'] and (best is None or result['makespan_ns']<best['result']['makespan_ns']):
            best=dict(variant=variant,waves=waves,certificate=certificate,result=result)
    return dict(problem=p,minimum_solution=s,best=best,tested=tested,
                limitation='Exactminimum-macro objective; finite boundedtiming search only. DropbeforeECC and IPR addresses/ownerswitch conditional; noextraFIFO orhold.')
