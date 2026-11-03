#!/usr/bin/env python3
"""Whole-row ordinary-read compiler backend and explicit service contract.

This backend performs physical placement, not quantization or native numerical
simulation. I8 coefficient bytes are a declared input representation. Scale
metadata stays in host memory; its ledger/cost must be supplied separately.
"""
from dataclasses import asdict, dataclass
import ctypes
import hashlib
import heapq
import json
from pathlib import Path
import subprocess

import numpy as np

from research_20261102_whole_row_compiler import Capability, minimum_work

ROOT = Path(__file__).resolve().parents[1]


def ceildiv(a, b):
    return (a + b - 1) // b


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass(frozen=True)
class Service:
    chips: int = 16
    channels: int = 8
    planes: int = 4
    page_bytes: int = 16384
    block_wordlines: int = 192
    input_bytes: int = 1
    output_bytes: int = 4
    ordinary_read_ns: int = 28000
    command_ns: int = 250
    host_submit_ns: int = 2000
    pcie_bytes_per_second: int = 8_000_000_000
    channel_bytes_per_second: int = 2_000_000_000
    activation_ns: int = 0
    combine_ns: int = 0
    output_fringe: int = 0
    down_first: int = 0
    input_buffer_bytes: int = 0

    def validate(self):
        assert min(self.chips, self.channels, self.planes, self.page_bytes,
                   self.block_wordlines, self.input_bytes, self.output_bytes,
                   self.pcie_bytes_per_second, self.channel_bytes_per_second) > 0
        assert self.chips % self.channels == 0
        assert min(self.ordinary_read_ns, self.command_ns, self.host_submit_ns,
                   self.activation_ns, self.combine_ns) >= 0
        assert self.output_fringe in (0, 1, 2) and self.down_first in (0, 1)
        assert self.input_buffer_bytes >= 0  # 0 means unspecified, not infinite native capacity.

    @property
    def wave_bytes(self):
        return self.planes * self.page_bytes

    @property
    def resources(self):
        return 4 + self.channels + self.chips

    def transfer(self, nbytes, rate):
        return ceildiv(int(nbytes) * 1_000_000_000, rate)

    def array(self):
        self.validate()
        return np.asarray(list(asdict(self).values()), dtype=np.int64)


def split_complete_rows(rows, columns, service):
    """Minimum-wave splitting among the minimum number of one-block jobs.

    The existing standard capacity DP is reused with virtual job slots. It is
    a lowering rule for a fixed chip band, not an optimal cross-chip schedule.
    Every emitted command starts a fresh full-row dot product/full input.
    """
    assert rows > 0 and columns > 0
    max_rows = service.block_wordlines * service.wave_bytes // columns
    assert max_rows > 0, 'A complete row must fit the native one-block command.'
    count = ceildiv(rows, max_rows)
    if count == 1:
        return [int(rows)]
    cap = Capability(chips=count, planes=service.planes,
                     page_bytes=service.page_bytes,
                     block_wordlines=service.block_wordlines, weight_bytes=1)
    result = minimum_work(rows, columns, service.block_wordlines, cap)
    pieces = [r for r in result['witness']['rows'] if r]
    assert len(pieces) == count and sum(pieces) == rows
    return pieces


def lower_rows(gu_rows, down_rows, gu_columns, down_columns, service):
    """Lower rows[expert,chip] into ragged, single-block complete-row jobs."""
    service.validate()
    gu = np.asarray(gu_rows, dtype=np.int64)
    down = np.asarray(down_rows, dtype=np.int64)
    assert gu.shape == down.shape and gu.ndim == 2 and gu.shape[1] == service.chips
    assert np.all(gu >= 0) and np.all(down >= 0)
    assert np.all(gu.sum(1) > 0) and np.all(down.sum(1) > 0)
    columns = np.column_stack((np.broadcast_to(gu_columns, len(gu)),
                               np.broadcast_to(down_columns, len(gu)))).astype(np.int32)
    assert np.all(columns > 0) and np.all(columns[:, 0] == columns[0, 0])
    if service.input_buffer_bytes:
        assert np.all(columns.astype(np.int64) * service.input_bytes <= service.input_buffer_bytes), 'Native input-buffer bound exceeded.'
    offsets = np.zeros((len(gu), 2, 2), dtype=np.int32)
    jobs, logical = [], []
    split_cache = {}
    for expert in range(len(gu)):
        for phase, rows in enumerate((gu[expert], down[expert])):
            c = int(columns[expert, phase]); cursor = 0
            offsets[expert, phase, 0] = len(jobs)
            for chip, total in enumerate(rows):
                if not total:
                    continue
                key = int(total), c
                if key not in split_cache:
                    split_cache[key] = split_complete_rows(*key, service)
                for n in split_cache[key]:
                    useful = n * c; pages = ceildiv(useful, service.page_bytes)
                    waves = ceildiv(pages, service.planes)
                    assert waves <= service.block_wordlines
                    jobs.append([chip, n, c, pages, waves])
                    logical.append(dict(expert=expert, phase=phase, chip=chip,
                                        row_start=cursor, rows=n, columns=c,
                                        useful_bytes=useful, pages=pages, waves=waves))
                    cursor += n
            assert cursor == int(rows.sum())
            offsets[expert, phase, 1] = len(jobs)
    return dict(offsets=offsets, jobs=np.asarray(jobs, dtype=np.int32),
                input_columns=columns, logical=logical,
                total_rows=np.column_stack((gu.sum(1), down.sum(1))).astype(np.int64))


def emitted_rows(job, service):
    _, rows, columns, pages, waves = map(int, job)
    if service.output_fringe == 1:
        return pages * service.page_bytes // columns
    if service.output_fringe == 2:
        return waves * service.wave_bytes // columns
    return rows


def occupancy_table(plan, service):
    """Per-expert/phase chip ownership work after actual block lowering."""
    table = np.zeros((*plan['offsets'].shape[:2], service.chips), dtype=np.int64)
    counts = np.zeros_like(table)
    channel = np.zeros((*table.shape[:2], service.channels), dtype=np.int64)
    pcie = np.zeros(table.shape[:2], dtype=np.int64)
    for e, phases in enumerate(plan['offsets']):
        for p, (begin, end) in enumerate(phases):
            for job in plan['jobs'][begin:end]:
                chip, _, columns, _, waves = map(int, job)
                inp = service.transfer(columns * service.input_bytes, service.channel_bytes_per_second)
                outbytes = emitted_rows(job, service) * service.output_bytes
                out = service.transfer(outbytes, service.channel_bytes_per_second)
                table[e, p, chip] += service.command_ns + inp + waves * service.ordinary_read_ns + out
                counts[e, p, chip] += 1
                channel[e, p, chip % service.channels] += inp + out
                pcie[e, p] += service.transfer(outbytes, service.pcie_bytes_per_second)
    return dict(chip_ns=table, command_counts=counts, channel_ns=channel, pcie_ns=pcie)


def physical_addresses(plan, service, chip_blocks=None):
    """Best-fit decreasing, wave-aligned LSB block addresses for native jobs.

    Physical coefficient-only footprint excludes CSB/MSB, host metadata,
    attention/dense weights, overprovisioning and FTL relocation. It is a
    virtual address certificate; this does not program a device.
    """
    addresses = [None] * len(plan['logical']); block_counts = []
    for chip in range(service.chips):
        residual = []
        indices = [i for i, j in enumerate(plan['logical']) if j['chip'] == chip]
        for i in sorted(indices, key=lambda i: (-plan['logical'][i]['waves'], i)):
            j = plan['logical'][i]; w = j['waves']
            legal = [b for b, left in enumerate(residual) if left >= w]
            if legal:
                b = min(legal, key=lambda b: (residual[b] - w, b))
            else:
                b = len(residual); residual.append(service.block_wordlines)
            begin = service.block_wordlines - residual[b]
            addresses[i] = dict(job=i, chip=chip, block=b, wordline=begin,
                                reserved_wordlines=w, pages=j['pages'],
                                first_plane=0, page_order='wordline-major; plane=page_index%planes')
            residual[b] -= w
        block_counts.append(len(residual))
    if chip_blocks is not None:
        capacity = np.broadcast_to(chip_blocks, service.chips)
        assert np.all(np.asarray(block_counts) <= capacity), 'Per-chip LSB block capacity exceeded.'
    # Exact coverage and interval nonoverlap use independently sorted intervals.
    for e, phases in enumerate(plan['offsets']):
        for p, (begin, end) in enumerate(phases):
            bands = sorted((j['row_start'], j['row_start'] + j['rows'])
                           for j in plan['logical'][begin:end])
            assert bands[0][0] == 0 and bands[-1][1] == plan['total_rows'][e, p]
            assert all(a[1] == b[0] for a, b in zip(bands, bands[1:]))
    occupied = {}
    for a, j in zip(addresses, plan['logical']):
        assert a['wordline'] + j['waves'] <= service.block_wordlines
        occupied.setdefault((a['chip'], a['block']), []).append((a['wordline'], a['wordline'] + j['waves']))
    for ranges in occupied.values():
        ranges.sort(); assert all(a[1] <= b[0] for a, b in zip(ranges, ranges[1:]))
    useful = sum(j['useful_bytes'] for j in plan['logical'])
    pages = sum(j['pages'] for j in plan['logical'])
    waves = sum(j['waves'] for j in plan['logical'])
    return dict(addresses=addresses, chip_blocks=block_counts, useful_bytes=useful,
                programmed_page_bytes=pages * service.page_bytes,
                reserved_wave_bytes=waves * service.wave_bytes,
                allocated_lsb_block_bytes=sum(block_counts) * service.block_wordlines * service.wave_bytes,
                row_coverage_and_address_nonoverlap=True)


class NativeReplay:
    def __init__(self, build_dir):
        source = Path(__file__).with_name('research_20261103_ordinary_read_native.cpp')
        build_dir = Path(build_dir); build_dir.mkdir(parents=True, exist_ok=True)
        digest = sha(source); binary = build_dir / ('ordinary-read-' + digest[:16] + '.so')
        if not binary.exists():
            subprocess.run(['c++', '-std=c++17', '-O3', '-shared', '-fPIC', str(source), '-o', str(binary)], check=True)
        self.lib = ctypes.CDLL(str(binary)); self.call = self.lib.eval_ordinary_read
        p32 = np.ctypeslib.ndpointer(dtype=np.int32, flags='C_CONTIGUOUS')
        p64 = np.ctypeslib.ndpointer(dtype=np.int64, flags='C_CONTIGUOUS')
        self.call.argtypes = [p32, p32, p32, p32, ctypes.c_int, ctypes.c_int,
                              ctypes.c_int, ctypes.c_int, ctypes.c_int,
                              p64, p64, p64, ctypes.c_void_p, p64]
        self.call.restype = None
        self.provenance = dict(source=str(source), source_sha256=digest,
                               binary=str(binary), binary_sha256=sha(binary),
                               command='c++ -std=c++17 -O3 -shared -fPIC SOURCE -o BINARY')

    def run(self, plan, routes, experts, shared, service, trace=False):
        service.validate()
        validate_plan(plan, experts + shared, service)
        routes = np.ascontiguousarray(routes, dtype=np.int32)
        assert routes.ndim == 2 and len(routes) and routes.shape[1] > 0
        assert len(plan['offsets']) == experts + shared
        assert np.all((routes >= 0) & (routes < experts))
        assert all(len(set(r.tolist())) == len(r) for r in routes)
        for k in ('offsets', 'jobs', 'input_columns'):
            assert plan[k].dtype == np.int32 and plan[k].flags.c_contiguous
        counts = plan['offsets'][:, :, 1] - plan['offsets'][:, :, 0]
        chosen = np.r_[routes[0], np.arange(experts, experts + shared)]
        max_nodes = 3 + 5 * int(counts[chosen].sum()) + 4 * len(chosen)
        log = np.zeros((max_nodes if trace else 1, 8), dtype=np.int64)
        nlog = np.zeros(1, dtype=np.int64)
        values = np.zeros(len(routes), dtype=np.int64); bounds = values.copy()
        self.call(plan['offsets'], plan['jobs'], plan['input_columns'], routes,
                  experts, shared, len(plan['jobs']), len(routes), routes.shape[1], service.array(),
                  values, bounds, ctypes.c_void_p(log.ctypes.data) if trace else None, nlog)
        assert np.all(values >= bounds)
        return dict(cost=values, lower_bound=bounds,
                    trace=log[:nlog[0]].copy() if trace else None,
                    nodes=int(nlog[0]))


def validate_plan(plan, logical_experts, service):
    """Reject imported descriptors that could undercharge vectors or pages."""
    for key in ('offsets', 'jobs', 'input_columns'):
        assert isinstance(plan[key], np.ndarray)
        assert plan[key].dtype == np.int32 and plan[key].flags.c_contiguous
    offsets, jobs, inputs = (plan[k] for k in ('offsets', 'jobs', 'input_columns'))
    assert offsets.shape == (logical_experts, 2, 2)
    assert inputs.shape == (logical_experts, 2) and np.all(inputs > 0)
    assert jobs.ndim == 2 and jobs.shape[1] == 5 and len(jobs) > 0
    ranges = offsets.reshape(-1, 2).astype(np.int64)
    assert ranges[0, 0] == 0 and ranges[-1, 1] == len(jobs)
    assert np.all(ranges[:, 1] > ranges[:, 0])
    assert np.array_equal(ranges[:-1, 1], ranges[1:, 0])
    counts = ranges[:, 1] - ranges[:, 0]
    expected_columns = np.repeat(inputs.reshape(-1), counts)
    assert np.array_equal(jobs[:, 2], expected_columns), 'Job/full-vector dimensions disagree.'
    assert np.all(inputs[:, 0] == inputs[0, 0]), 'GU experts must share the same complete input.'
    if service.input_buffer_bytes:
        assert np.all(inputs.astype(np.int64) * service.input_bytes <= service.input_buffer_bytes), 'Native input-buffer bound exceeded.'
    j = jobs.astype(np.int64)
    assert np.all((j[:, 0] >= 0) & (j[:, 0] < service.chips))
    assert np.all(j[:, 1:] > 0)
    assert np.array_equal(j[:, 3], (j[:, 1] * j[:, 2] + service.page_bytes - 1) // service.page_bytes)
    assert np.array_equal(j[:, 4], (j[:, 3] + service.planes - 1) // service.planes)
    assert np.all(j[:, 4] <= service.block_wordlines)
    if 'total_rows' in plan:
        actual = np.add.reduceat(j[:, 1], ranges[:, 0]).reshape(logical_experts, 2)
        assert np.array_equal(actual, plan['total_rows']), 'Complete-row coverage count disagrees.'


class ReferenceGraph:
    """Slow independent ready-queue scan for regression of event semantics."""
    def __init__(self, service):
        self.service = service; self.nodes = []; self.job_count = 0

    def add(self, resource, duration, deps=(), phase=0, expert=-1, chip=-1,
            job=-1, acquire=False, release=False):
        i = len(self.nodes)
        self.nodes.append(dict(resource=resource, duration=int(duration), deps=list(deps),
                               phase=phase, expert=expert, chip=chip, job=job,
                               acquire=acquire, release=release))
        return i

    def matrix(self, expert, phase, d, submit, upload):
        s = self.service; chip, _, columns, _, waves = map(int, d)
        job = self.job_count; self.job_count += 1
        cmd = self.add(1, s.command_ns, [submit], phase, expert, chip, job, True)
        inp = self.add(3 + chip % s.channels, s.transfer(columns * s.input_bytes, s.channel_bytes_per_second),
                       [cmd, upload], phase, expert, chip, job)
        comp = self.add(3 + s.channels + chip, s.ordinary_read_ns * waves, [inp], phase, expert, chip, job)
        nbytes = emitted_rows(d, s) * s.output_bytes
        out = self.add(3 + chip % s.channels, s.transfer(nbytes, s.channel_bytes_per_second),
                       [comp], phase, expert, chip, job, False, True)
        return self.add(2, s.transfer(nbytes, s.pcie_bytes_per_second), [out], phase, expert)

    @classmethod
    def build(cls, plan, selected, service):
        g = cls(service); submit = g.add(0, service.host_submit_ns)
        upload = g.add(2, service.transfer(int(plan['input_columns'][0, 0]) * service.input_bytes,
                                           service.pcie_bytes_per_second), [submit])
        ready = []
        for ei, e in enumerate(selected):
            begin, end = plan['offsets'][e, 0]
            outputs = [g.matrix(ei, 0, d, submit, upload) for d in plan['jobs'][begin:end]]
            ready.append(g.add(service.resources - 1, 0, outputs, expert=ei))
        complete = []
        for ei, e in enumerate(selected):
            act = g.add(0, service.activation_ns, [ready[ei]], 1, ei)
            ds = g.add(0, service.host_submit_ns, [act], 1, ei)
            up = g.add(2, service.transfer(int(plan['input_columns'][e, 1]) * service.input_bytes,
                                           service.pcie_bytes_per_second), [ds], 1, ei)
            begin, end = plan['offsets'][e, 1]
            complete.extend(g.matrix(ei, 1, d, ds, up) for d in plan['jobs'][begin:end])
        g.add(0, service.combine_ns, complete, 1)
        return g

    def run(self):
        s = self.service; nodes = self.nodes
        successors = [[] for _ in nodes]; remaining = []; ready = [[] for _ in range(s.resources)]
        start = [-1] * len(nodes); end = start.copy(); events = []
        busy = [False] * s.resources; owner = [-1] * s.chips
        def enqueue(i):
            n = nodes[i]; p = 1 - n['phase'] if s.down_first else n['phase']
            heapq.heappush(ready[n['resource']], ((p, n['expert'], i), i))
        for i, n in enumerate(nodes):
            remaining.append(len(n['deps']))
            for q in n['deps']:
                assert q < i; successors[q].append(i)
            if not n['deps']:
                enqueue(i)
        now = done = 0
        while done < len(nodes):
            for r, q in enumerate(ready):
                if busy[r]:
                    continue
                held = []; chosen = None
                while q:
                    item = heapq.heappop(q); n = nodes[item[1]]
                    eligible = n['job'] < 0 or (owner[n['chip']] == -1 if n['acquire'] else owner[n['chip']] == n['job'])
                    if eligible:
                        chosen = item[1]; break
                    held.append(item)
                for item in held:
                    heapq.heappush(q, item)
                if chosen is None:
                    continue
                n = nodes[chosen]
                if n['acquire']:
                    owner[n['chip']] = n['job']
                start[chosen] = now; busy[r] = True
                heapq.heappush(events, (now + n['duration'], chosen))
            assert events, 'Deadlocked native job graph'
            now = events[0][0]
            while events and events[0][0] == now:
                _, i = heapq.heappop(events); n = nodes[i]
                end[i] = now; busy[n['resource']] = False
                if n['release']:
                    assert owner[n['chip']] == n['job']; owner[n['chip']] = -1
                done += 1
                for j in successors[i]:
                    remaining[j] -= 1
                    if not remaining[j]:
                        enqueue(j)
        assert all(o == -1 for o in owner)
        trace = np.asarray([[n['resource'], n['duration'], start[i], end[i],
                             n['phase'], n['expert'], n['chip'], n['job']]
                            for i, n in enumerate(nodes)], dtype=np.int64)
        return dict(cost=now, trace=trace)
