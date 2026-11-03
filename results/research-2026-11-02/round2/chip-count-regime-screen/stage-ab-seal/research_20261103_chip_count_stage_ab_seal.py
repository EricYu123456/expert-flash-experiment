#!/usr/bin/env python3
"""Consolidate frozen finite regime controls and their anchor fanout addendum."""
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2/chip-count-regime-screen'


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n')


def main():
    output = BASE / 'stage-ab-seal'
    output.mkdir(exist_ok=False)
    a = json.loads((BASE / 'stage-a/report.json').read_text())
    b = json.loads((BASE / 'stage-b/report.json').read_text())
    add = json.loads((BASE / 'stage-b/anchor-fanout-addendum/report.json').read_text())
    geometry = json.loads((BASE / 'independent-geometry/audit.json').read_text())
    records = b['records'] + add['records']
    cases = []
    for case in a['cases']:
        relevant = [x for x in records if all(x[k] == case[k] for k in ('model', 'chips', 'input_copies'))]
        if not relevant:
            cases.append(dict(model=case['model'], chips=case['chips'], input_copies=case['input_copies'],
                              status='native_not_run', necessary=case['necessary'],
                              reason='StageA finite ownedwork gap below5%; this screening decision is not a no-headroom proof.'))
            continue
        best = min(relevant, key=lambda x: x['cost']['mean_ns'])
        lb = case['necessary']['universal_necessary_ns']
        cases.append(dict(model=case['model'], chips=case['chips'], input_copies=case['input_copies'],
                          status='finite_native_controls', best_id=best['id'], priority=best['priority'],
                          cost=best['cost'], necessary=case['necessary'],
                          optimistic_remaining_mean_pct=100 * (1 - lb / best['cost']['mean_ns']),
                          optimistic_remaining_p99_pct=100 * (1 - lb / best['cost']['p99_ns']),
                          warning='Necessary ceiling only, not attainable benefit. Finite deterministic labels are not learned strong joint controls.'))
    result = dict(status='sealed_stageAB_with_strong_anchor_mirror', utc=datetime.now(timezone.utc).isoformat(),
                  cases=cases, geometry_DP_calls=a['geometry_DP_calls'],
                  native_costs=b['native_costs'] + add['queries'],
                  original_stageB_native_costs=b['native_costs'], anchor_mirror_costs=add['queries'],
                  independent_geometry=dict(path=str(BASE / 'independent-geometry/audit.json'), sha256=sha(BASE / 'independent-geometry/audit.json'), status=geometry['status']),
                  attribution='The addendum only mirrors the same strong16chiprows across input1/4, with no fit. G4input4=CP0107; Deepinput1=seed20261106 native-refined incumbent. It supersedes the weaker corresponding finite winners in original StageB summary. Original artifacts unchanged.',
                  validation='StageB q0 C++/Python full-node checks and original16GF all512 cost/LB bridges; addendum q0 checks. Independent inverse-capacity DP verifies94frontiers and996profilemenus. These are synthetic cost-model checks, not device or end-to-end measurement.',
                  limitations='L0 actualW8 TRAIN512 only. Independent necessary minima may conflict. M8/32/64 are explicit hardware PE-scale variants; more NAND targets does not establish more AiF PEs. Never attribute comparisons across different chip counts to placement. No learned label fitting in these controls, no heldout/new weights/quality or replica advantage.')
    dump(output / 'summary.json', result)
    lines = ['Chip-count StageA/B finite controls (anchor addendum incorporated)', '',
             '| Model | Chips | Input copies | Best finite native mean / P99 (us) | Necessary LB (us) |',
             '|---|---:|---:|---:|---:|']
    for x in cases:
        if x['status'] == 'native_not_run':
            value = 'not run'
        else:
            value = f"{x['cost']['mean_ns']/1000:.6f} / {x['cost']['p99_ns']/1000:.6f}"
        lines.append(f"| {x['model']} | {x['chips']} | {x['input_copies']} | {value} | {x['necessary']['universal_necessary_ns']/1000:.3f} |")
    lines += ['', '94 minimum-work/depth frontier calls, independently checked together with996 menu profiles. Native total29696 query-costs, including2048 strong-anchor fanout mirrors. GF/DF choices are offered equally to each map.', '',
              'The maps are finite deterministic controls. Their gap to a necessary bound is not an algorithm benefit or proof that the bound is attainable. Subsequent learned CP controls are separate artifacts and should replace these controls when assessing novelty.', '',
              'Hardware count is held fixed inside each comparison;32/64PE AiF is a scale sensitivity, not an implemented hardware claim. FullGU/A8, privateDown, singlecontext, complete inputs, commands and page-fringe outputs are charged.', '']
    (output / 'report.md').write_text('\n'.join(lines))
    shutil.copy2(__file__, output / Path(__file__).name)
    files = []
    for folder in ('stage-a', 'stage-b', 'independent-geometry'):
        files += [x for x in (BASE / folder).rglob('*') if x.is_file()]
    files += [output / 'summary.json', output / 'report.md', output / Path(__file__).name]
    dump(output / 'completion.json', dict(status='sealed_stageAB', utc=datetime.now(timezone.utc).isoformat(),
                                          files=[dict(path=str(p), sha256=sha(p), bytes=p.stat().st_size) for p in sorted(files)]))
    print(json.dumps(dict(output=str(output), completion_sha256=sha(output / 'completion.json'), native_costs=result['native_costs']), indent=2))


if __name__ == '__main__':
    main()
