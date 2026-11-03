#!/usr/bin/env python3
"""Build, cross-check and measure the native host output-recovery kernel."""
import ctypes
import json
from pathlib import Path
import platform
import subprocess

import numpy as np

from run_corpus_traces import ROOT, sha256

DEST = ROOT / 'results/research-2026-11-02/round2/integer-parity'


def main():
    source = ROOT / 'scripts/research_20261102_parity_decode_bench.cpp'
    binary, library = DEST / 'parity-decode-bench', DEST / 'parity-decode.so'
    commands = []
    for target, extra in ((binary, []), (library, ['-shared', '-fPIC'])):
        cmd = ['g++', '-O3', '-march=native', '-std=c++17', *extra, str(source), '-o', str(target)]
        subprocess.run(cmd, check=True)
        commands.append(cmd)
    lib = ctypes.CDLL(str(library))
    function = lib.recover_i32
    function.argtypes = [ctypes.POINTER(ctypes.c_int32)] * 3 + [ctypes.POINTER(ctypes.c_int64)] + [ctypes.c_int] * 3
    function.restype = None
    rng = np.random.default_rng(20261102)
    checks = 0
    for rows in (1, 31, 129, 1792):
        for experts in (1, 2, 8):
            for shards in (7, 15):
                low = rng.integers(-2**30, 2**30, (experts, rows), dtype=np.int32)
                high = rng.integers(-2**25, 2**25, (experts, rows), dtype=np.int32)
                known = rng.integers(-2**30, 2**30, (experts, shards, rows), dtype=np.int32)
                recovered = np.empty((experts, rows), dtype=np.int64)
                function(*(a.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) for a in (low, high, known)),
                         recovered.ctypes.data_as(ctypes.POINTER(ctypes.c_int64)), rows, experts, shards)
                expected = low.astype(np.int64) + 256 * high.astype(np.int64) - known.astype(np.int64).sum(axis=1)
                np.testing.assert_array_equal(recovered, expected)
                checks += 1
    cases = set()
    for path in (ROOT / 'results/research-2026-11-02/projection-study').glob('*.frozen_plan.json'):
        plan = json.loads(path.read_text())
        for phase in plan['phases']:
            table = np.array(phase['chosen_rows'])
            for half in range(2):
                count = int(table[:, half*8:half*8+8].max())
                if count:
                    for shards in (7, 15):
                        cases.add((count, plan['active'], shards))
    measurements = []
    for case in sorted(cases):
        cmd = [str(binary), *map(str, case)]
        measurements.append(json.loads(subprocess.check_output(cmd, text=True)))
        commands.append(cmd)
    document = dict(source_sha256=sha256(source), binary_sha256=sha256(binary),
                    library_sha256=sha256(library), cross_language_checks=checks,
                    measurements=measurements, commands=commands, platform=platform.platform(),
                    compiler=subprocess.check_output(['g++', '--version'], text=True).splitlines()[0],
                    scope='Warm host-memory native I64 decode kernel only. Does not include DMA, '
                          'interrupts, dispatch, accelerator compute, or cold-memory cache effects.')
    (DEST / 'host_decode_measurements.json').write_text(json.dumps(document, indent=2)+'\n')
    print(json.dumps(measurements, indent=2))


if __name__ == '__main__':
    main()
