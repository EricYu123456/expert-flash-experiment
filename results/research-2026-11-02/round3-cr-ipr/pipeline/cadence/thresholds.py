#!/usr/bin/env python3
"""Post-grid integer-ns boundaries; no native timer granularity claim."""
import json
from pathlib import Path
from experiment import v1

out = []
for mode in ('fixed_recycle', 'fixed_evaluation'):
    for copy in (100, 500, 1000, 2000):
        def check(period):
            return v1.simulate(v1.build_g4_candidate(), cache_copy_ns=copy, cr_ns=period,
                               recycle_ns=6040 if mode == 'fixed_recycle' else period - 3660)
        lo, hi = 9700, 11000
        if check(lo)['feasible']:
            minimum = lo
        else:
            while lo + 1 < hi:
                mid = (lo + hi) // 2
                if check(mid)['feasible']:
                    hi = mid
                else:
                    lo = mid
            minimum = hi
        yes = check(minimum)
        previous = check(minimum - 1) if minimum > 9700 else None
        assert yes['feasible'] and (previous is None or not previous['feasible'])
        out.append(dict(mode=mode, copy_ns=copy, min_cr_ns_at_least9700=minimum,
                        increase_percent=(minimum / 9700 - 1) * 100,
                        makespan_ns=yes['makespan_ns'], previous_ns_failed=previous is not None))
Path(__file__).with_suffix('.json').write_text(json.dumps(dict(
    status='PASS', interpretation='Post-grid integer-ns threshold sensitivity within the conditional model; no native timer granularity or programmability claim.',
    records=out), indent=2) + '\n')
