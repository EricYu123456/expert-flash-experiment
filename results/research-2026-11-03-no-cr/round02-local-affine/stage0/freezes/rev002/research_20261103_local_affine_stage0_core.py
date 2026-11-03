"""Frozen numerical/learning core. Imports perform no data I/O."""
import os
for _name in ['OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'OMP_THREAD_LIMIT']:
    os.environ[_name] = '8'
os.environ['OMP_DYNAMIC'] = 'FALSE'
import ctypes as ct
import hashlib
import json
import math
from pathlib import Path
import time
import subprocess
import numpy as np
import scipy.linalg as sla
from scipy.sparse.linalg import LinearOperator, eigsh, ArpackNoConvergence
from threadpoolctl import threadpool_limits
threadpool_limits(8)

GRID = [0., 1e-4, 1e-3, 1e-2, .1, 1., 10., 'infinity']
GUARDS = [50, 70, 85, 100]
SEED = 20261103


def save(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for b in iter(lambda: f.read(8 << 20), b''):
            h.update(b)
    return h.hexdigest()


def array_hash(*arrays):
    h = hashlib.sha256()
    for a in arrays:
        a = np.ascontiguousarray(a)
        h.update(str((a.dtype.str, a.shape)).encode())
        h.update(a.tobytes())
    return h.hexdigest()


class Numeric:
    def __init__(self, path):
        self.lib = ct.CDLL(str(path))
        self.activation_binary = Path(path).parent / 'numeric-self-test'
        self.worker = None
        self.activation_calls = 0
        self.activation_ipc_bytes = 0
        self.activation_seconds = 0.
        self.lib.la_environment.restype = ct.c_int
        assert self.lib.la_environment() >= 0
        assert self.lib.la_workers_environment() == 8
        self.dot_audits = 0

    @staticmethod
    def ptr(a):
        return ct.c_void_p(a.ctypes.data)

    def quantize(self, a):
        a = np.ascontiguousarray(a, dtype=np.float32)
        assert a.ndim == 2 and a.shape[1] > 0
        q = np.empty(a.shape, np.int8)
        s = np.empty(len(a), np.float32)
        assert self.lib.la_quantize(self.ptr(a), len(a), a.shape[1], self.ptr(q), self.ptr(s)) == 0, 'canonical quantizer fault'
        return q, s

    def dots(self, q, w, audit=True):
        assert q.dtype == w.dtype == np.int8 and q.shape[1] == w.shape[1]
        assert q.shape[1] * 127 ** 2 < 2 ** 53
        # Pure FP64 BLAS operands, alpha1/beta0 via ndarray matmul; integer
        # products and every signed partial are exactly representable.
        d = q.astype(np.float64) @ w.astype(np.float64).T
        assert d.dtype == np.float64 and np.isfinite(d).all() and np.equal(d, np.trunc(d)).all()
        if audit and len(q):
            qi = sorted(set([0, len(q) // 2, len(q) - 1]))
            wi = sorted(set([0, len(w) // 2, len(w) - 1]))
            ref = q[qi].astype(np.int64) @ w[wi].astype(np.int64).T
            assert np.array_equal(d[np.ix_(qi, wi)], ref)
            self.dot_audits += ref.size
        return np.ascontiguousarray(d)

    def rescale(self, dots, sw, sx):
        dots = np.ascontiguousarray(dots, np.float64)
        sw, sx = np.ascontiguousarray(sw, np.float32), np.ascontiguousarray(sx, np.float32)
        out = np.empty(dots.shape, np.float32)
        assert self.lib.la_rescale(self.ptr(dots), self.ptr(sw), self.ptr(sx), len(dots), dots.shape[1], self.ptr(out)) == 0
        return out

    def linear(self, q, sx, w, sw):
        return self.rescale(self.dots(q, w), sw, sx)

    def swiglu(self, gate, up):
        gate, up = np.ascontiguousarray(gate, np.float32), np.ascontiguousarray(up, np.float32)
        assert gate.shape == up.shape
        if self.worker is None:
            self.worker = subprocess.Popen([str(self.activation_binary), '--swiglu-worker'], stdin=subprocess.PIPE, stdout=subprocess.PIPE)
        begin = time.monotonic()
        self.worker.stdin.write(np.asarray(gate.shape, '<i4').tobytes())
        self.worker.stdin.write(gate.tobytes()); self.worker.stdin.write(up.tobytes()); self.worker.stdin.flush()
        def exact(n):
            chunks = []
            left = n
            while left:
                block = self.worker.stdout.read(left)
                assert block, 'activation worker short read/exit'
                chunks.append(block); left -= len(block)
            return b''.join(chunks)
        status = exact(4)
        assert len(status) == 4 and int(np.frombuffer(status, np.int32)[0]) == 0, 'pinned activation worker fault'
        blob = exact(gate.nbytes)
        assert len(blob) == gate.nbytes, 'incomplete activation worker output'
        self.activation_calls += 1
        self.activation_ipc_bytes += 12 + gate.nbytes * 3
        self.activation_seconds += time.monotonic() - begin
        return np.frombuffer(blob, np.float32).reshape(gate.shape).copy()

    def close(self):
        if self.worker is not None:
            self.worker.stdin.close()
            assert self.worker.wait(timeout=30) == 0
            self.worker = None

    def bias(self, y, p, w):
        y, p, w = np.ascontiguousarray(y, np.float32), np.ascontiguousarray(p, np.float32), np.ascontiguousarray(w, np.float64)
        out = np.empty(y.shape[1], np.float32)
        assert self.lib.la_calibrate(self.ptr(y), self.ptr(p), self.ptr(w), len(y), y.shape[1], self.ptr(out)) == 0
        return out

    def add_bias(self, p, bias):
        p, bias = np.ascontiguousarray(p, np.float32), np.ascontiguousarray(bias, np.float32)
        out = np.empty_like(p)
        assert self.lib.la_add_bias(self.ptr(p), self.ptr(bias), len(p), p.shape[1], self.ptr(out)) == 0
        return out

    def tree_project(self, x, normal):
        x, normal = np.ascontiguousarray(x, np.float32), np.ascontiguousarray(normal, np.float32)
        out = np.empty(len(x), np.float64)
        self.lib.la_tree_project(self.ptr(x), self.ptr(normal), len(x), x.shape[1], self.ptr(out))
        return out


def teacher(num, x, weights, batch=256):
    out = np.empty((len(x), weights['down'][0].shape[0]), np.float32)
    for start in range(0, len(x), batch):
        sl = slice(start, start + batch)
        q, sx = num.quantize(x[sl])
        g = num.linear(q, sx, *weights['gate'])
        u = num.linear(q, sx, *weights['up'])
        a = num.swiglu(g, u)
        qa, sa = num.quantize(a)
        out[sl] = num.linear(qa, sa, *weights['down'])
    return out


def signfix(v):
    j = int(np.argmax(np.abs(v)))
    return -v if v[j] < 0 else v


def leading_pc(x, seed):
    a = np.asarray(x, np.float64)
    a = a - a.mean(axis=0, dtype=np.float64)
    variance = np.sum(a * a, axis=0, dtype=np.float64)
    if not len(a) or float(variance.sum()) == 0:
        return None, {'solver': 'zero_variance', 'residual': 0.}
    H = a.shape[1]
    op = LinearOperator((H, H), matvec=lambda v: a.T @ (a @ v) / len(a), dtype=np.float64)
    v0 = np.random.default_rng(seed).standard_normal(H)
    method = 'ARPACK_eigsh_LA_tol1e-10_ncv20_maxiter4096'
    try:
        value, vec = eigsh(op, k=1, which='LA', v0=v0, tol=1e-10, maxiter=4096, ncv=min(H, 20))
        v = vec[:, 0]
        residual = float(np.linalg.norm(op @ v - value[0] * v) / max(abs(value[0]), 1e-300))
        if residual > 1e-8:
            raise ArpackNoConvergence('residual', value, vec)
    except (ArpackNoConvergence, TypeError):
        cov = a.T @ a / len(a)
        value, vec = sla.eigh(cov, subset_by_index=[H - 1, H - 1], driver='evr', check_finite=True)
        v = vec[:, 0]
        residual = float(np.linalg.norm(cov @ v - value[0] * v) / max(abs(value[0]), 1e-300))
        method = 'dense_symmetric_evr_fallback'
    assert np.isfinite(v).all() and residual <= 1e-8
    return signfix(v).astype(np.float32), {'solver': method, 'residual': residual, 'eigenvalue': float(value[0])}


def fit_tree(num, x, layer):
    nodes = {}
    def build(idx, rows, depth):
        mean = np.mean(x[rows].astype(np.float64), axis=0).astype(np.float32)
        node = {'mean': mean, 'depth': depth, 'fit_count': len(rows), 'normal': None, 'threshold': None}
        nodes[idx] = node
        if depth == 5:
            return
        normal, info = leading_pc(x[rows], SEED + layer * 1000 + idx)
        node['solver_info'] = info
        if normal is None:
            return
        z = num.tree_project(x[rows], normal)
        distinct = np.unique(z)
        if len(distinct) < 2:
            return
        # Choose a boundary between distinct projections, minimizing imbalance.
        candidates = distinct[:-1]
        sizes = np.searchsorted(np.sort(z), candidates, side='right')
        k = int(np.argmin(np.abs(2 * sizes - len(rows))))
        lo, hi = float(distinct[k]), float(distinct[k + 1])
        threshold = lo + (hi - lo) / 2
        if not lo <= threshold < hi:
            threshold = lo
        mask = z <= threshold
        assert mask.any() and (~mask).any()
        node['normal'], node['threshold'] = normal, threshold
        build(2 * idx + 1, rows[mask], depth + 1)
        build(2 * idx + 2, rows[~mask], depth + 1)
    build(0, np.arange(len(x)), 0)
    return nodes


def select_tree(num, tree, x, C):
    depth = int(math.log2(C))
    ids = np.full(len(x), -1, np.int32)
    def walk(nodeid, rows, d):
        n = tree[nodeid]
        if d == depth:
            ids[rows] = nodeid - (2 ** depth - 1)
        elif n['normal'] is not None:
            z = num.tree_project(x[rows], n['normal'])
            walk(nodeid * 2 + 1, rows[z <= n['threshold']], d + 1)
            walk(nodeid * 2 + 2, rows[z > n['threshold']], d + 1)
    walk(0, np.arange(len(x)), 0)
    return ids


def fit_projection(x):
    d = x.astype(np.float64)
    d -= d.mean(axis=0)
    cov = d.T @ d / len(d)
    values, vectors = sla.eigh(cov, driver='evd', check_finite=True)
    order = np.argsort(-values, kind='stable')
    values, vectors = values[order], vectors[:, order]
    vectors = np.stack([signfix(vectors[:, j]) for j in range(vectors.shape[1])])
    sigma = np.sqrt(np.maximum(values, 0.))
    return vectors.astype(np.float32), sigma


def design(num, x, tree, C, basis, sigma, rank, fmt):
    leaf = select_tree(num, tree, x, C)
    assert (leaf >= 0).all(), 'unsplittable tree: inconclusive configuration, never fake rows'
    means = np.stack([tree[C - 1 + j]['mean'] for j in range(C)])
    delta = np.subtract(x, means[leaf], dtype=np.float32)
    if fmt == 'identity':
        assert rank == x.shape[1]
        P = None
        z = delta
    else:
        P = basis[:rank].copy()
        if fmt == 'balanced':
            scales = np.maximum(sigma[:rank], sigma[0] / 16) if sigma[0] > 0 else np.ones(rank)
            P = (P.astype(np.float64) / scales[:, None]).astype(np.float32)
        # Declared host numeric implementation: F32 subtraction, pure FP64
        # BLAS projection using stored F32 P, then single F32 rounding.
        z = (delta.astype(np.float64) @ P.astype(np.float64).T).astype(np.float32)
    q, sx = num.quantize(z)
    v = sx.astype(np.float64)[:, None] * q.astype(np.float64)
    distance = np.sqrt(np.sum(delta.astype(np.float64) ** 2, axis=1, dtype=np.float64))
    return {'q': q, 'sx': sx, 'v': v, 'leaf': leaf, 'distance': distance, 'P': P}


def eigen(cov):
    cov = (cov + cov.T) * .5
    d, U = sla.eigh(cov, driver='evd', check_finite=True)
    assert d.min(initial=0) >= -1e-8 * max(float(d.max(initial=0)), 1e-300), 'negative covariance'
    return np.maximum(d, 0), U


def solve(cov, cross, reg, prior=None, factor=None):
    r = cov.shape[0]
    if prior is None:
        prior = np.zeros_like(cross)
    if reg == 'infinity':
        return prior.copy(), {'rank': 0, 'df': 0., 'condition': None}
    d, U = eigen(cov) if factor is None else factor
    trace = float(d.sum())
    if trace == 0:
        return prior.copy(), {'rank': 0, 'df': 0., 'condition': None}
    keep = d > d[-1] * 1e-10
    t = float(reg) * trace / r
    inverse = np.zeros_like(d)
    if t:
        inverse[:] = 1 / (d + t)
    else:
        inverse[keep] = 1 / d[keep]
    correction = U @ (inverse[:, None] * (U.T @ (cross - cov @ prior)))
    return prior + correction, {'rank': int(keep.sum()), 'df': float(np.sum(d * inverse)),
                                'condition': float(d[-1] / d[keep][0]) if keep.any() else None}


def local_smoother_df(pfactor, eta, localstats, localfactors, lam):
    d, U = pfactor
    def inverse(values, reg):
        out = np.zeros_like(values)
        if reg == 'infinity' or values.sum() == 0:
            return out
        t = float(reg) * float(values.sum()) / len(values)
        keep = values > values[-1] * 1e-10
        if t:
            out[:] = 1 / (values + t)
        else:
            out[keep] = 1 / values[keep]
        return out
    R = (U * inverse(d, eta)) @ U.T
    totalw = sum(s['sum_w'] for s in localstats)
    df = float(sum(s['supported'] for s in localstats))
    for st, (ld, Q) in zip(localstats, localfactors):
        li = inverse(ld, lam)
        direct = float(np.sum(ld * li))
        prior = float(np.sum(np.sum(Q * (R @ Q), axis=0) * ld * (1 - ld * li)))
        df += direct + st['sum_w'] / totalw * prior
    return df


def statistics(v, y, leaf, w, C, prompts):
    r, H = v.shape[1], y.shape[1]
    pooled_cov, pooled_cross = np.zeros((r, r)), np.zeros((r, H))
    local, total = [], float(w.sum())
    for c in range(C):
        ix = np.flatnonzero(leaf == c)
        ww = w[ix]
        sw = float(ww.sum())
        if not sw:
            local.append({'cov': np.zeros((r, r)), 'cross': np.zeros((r, H)), 'ix': ix,
                          'sum_w': 0., 'kish': 0., 'prompts': 0, 'supported': False})
            continue
        a, b = v[ix], y[ix].astype(np.float64)
        ma = np.sum(a * ww[:, None], axis=0) / sw
        mb = np.sum(b * ww[:, None], axis=0) / sw
        da, db = a - ma, b - mb
        cov = (da.T * ww) @ da / sw
        cross = (da.T * ww) @ db / sw
        pooled_cov += cov * (sw / total)
        pooled_cross += cross * (sw / total)
        local.append({'cov': cov, 'cross': cross, 'ix': ix, 'sum_w': sw,
                      'kish': sw * sw / float(ww @ ww), 'prompts': int(np.unique(prompts[ix]).size), 'supported': True})
    return pooled_cov, pooled_cross, local


def lower(num, slopes, dfit, y, w, C, pooled=False):
    H = y.shape[1]
    qmaps, scales, bias, unsupported = [], [], [], []
    for c in range(C):
        A = slopes[0] if pooled else slopes[c]
        q, sw = num.quantize(A.T.astype(np.float32))
        ix = np.flatnonzero(dfit['leaf'] == c)
        if len(ix):
            pred = num.linear(dfit['q'][ix], dfit['sx'][ix], q, sw)
            b = num.bias(y[ix], pred, w[ix])
        else:
            b = np.zeros(H, np.float32)
            unsupported.append(c)
        qmaps.append(q)
        scales.append(sw)
        bias.append(b)
    # Physically identical lowered maps are deduplicated, independently of
    # region-specific biases. Zero maps require no NAND coefficients.
    hashes = [array_hash(q, sw) if np.any(q) else 'zero' for q, sw in zip(qmaps, scales)]
    return {'qmaps': np.stack(qmaps), 'scales': np.stack(scales), 'bias': np.stack(bias),
            'map_hashes': hashes, 'unsupported': unsupported}


def predict(num, model, d):
    H = model['bias'].shape[1]
    p = np.empty((len(d['q']), H), np.float32)
    for c in range(len(model['qmaps'])):
        ix = np.flatnonzero(d['leaf'] == c)
        if len(ix):
            pred = num.linear(d['q'][ix], d['sx'][ix], model['qmaps'][c], model['scales'][c]) if np.any(model['qmaps'][c]) else np.zeros((len(ix), H), np.float32)
            p[ix] = num.add_bias(pred, model['bias'][c])
    return p


def regional_mean(num, y, leaf, w, C):
    b = []
    for c in range(C):
        ix = np.flatnonzero(leaf == c)
        b.append(num.bias(y[ix], np.zeros_like(y[ix]), w[ix]) if len(ix) else np.zeros(y.shape[1], np.float32))
    return np.stack(b)


def loss(pred, y, natural, covered):
    use = np.asarray(natural, bool)
    energy = float(np.sum(y[use].astype(np.float64) ** 2))
    sse = float(np.sum((pred[use & covered].astype(np.float64) - y[use & covered]) ** 2))
    return sse / energy if energy > 0 and use.any() else None


def quantile_threshold(distance, natural, guard):
    a = distance[natural]
    if not len(a):
        return None
    return None if guard == 100 else float(np.quantile(a, guard / 100, method='linear'))


def cost(model, rank, C, H, natural, covered, leaf, intermediate, fmt):
    unique = {h: c for c, h in enumerate(model['map_hashes']) if h != 'zero'}
    bank = sum(model['qmaps'][c].nbytes + model['scales'][c].nbytes for c in unique.values())
    count = int(natural.sum())
    coverage = float(covered[natural].mean()) if count else 0.
    nonzero_leaf = np.array([h != 'zero' for h in model['map_hashes']])
    active_map_fraction = float(np.sum(natural & covered & nonzero_leaf[leaf]) / count) if count else 0.
    mapbytes = H * rank
    Pbytes = 0 if fmt in ['identity', 'none'] else H * rank * 4
    tree_bytes = (C - 1 + C) * H * 4 + (C - 1) * 8
    storage = bank + model['bias'].nbytes + C * 8 + Pbytes + tree_bytes
    return {'active_coefficient_macs': active_map_fraction * mapbytes + (1 - coverage) * 3 * H * intermediate,
            'host_projection_macs': 0 if fmt in ['identity', 'none'] else H * rank,
            'retained_map_scale_bias_bytes': bank + model['bias'].nbytes,
            'retained_storage_bytes': storage, 'projection_F32_bytes': Pbytes, 'tree_normals_centers_threshold_bytes': tree_bytes,
            'map_nonzero_covered_natural_fraction': active_map_fraction,
            'unique_lowered_maps': len(unique), 'logical_nonzero_map_pages': math.ceil(mapbytes / 16384) if unique else 0,
            'fallback_fraction': 1 - coverage, 'original_fallback_retained': True,
            'shared_layer_projection_not_per_expert_additive': True,
            'physical_latency_claim': False}


def point_identity(model, params):
    key = array_hash(model['qmaps'], model['scales'], model['bias'])
    return hashlib.sha256((key + json.dumps(params, sort_keys=True)).encode()).hexdigest()


def archive_model(folder, model):
    key = array_hash(model['qmaps'], model['scales'], model['bias'])
    path = Path(folder) / (key + '.npz')
    if not path.exists():
        # Whole-map dedup in archive as well as cost accounting.
        hashes, ids, q, s = [], [], [], []
        for c, h in enumerate(model['map_hashes']):
            if h not in hashes:
                hashes.append(h); q.append(model['qmaps'][c]); s.append(model['scales'][c])
            ids.append(hashes.index(h))
        np.savez(path, qmaps=np.stack(q), scales=np.stack(s), region_map=np.array(ids), bias=model['bias'],
                 unsupported=np.array(model['unsupported'], np.int32))
    return str(path), key


def load_model(path):
    a = np.load(path, allow_pickle=False)
    ids = a['region_map']
    return {'qmaps': a['qmaps'][ids], 'scales': a['scales'][ids], 'bias': a['bias'],
            'unsupported': list(a['unsupported'])}


def bootstrap(comparisons, stats, prompt_ids, literal_ties, repeats=10000):
    assert len(prompt_ids) == 40
    rng = np.random.default_rng(SEED)
    draws = rng.integers(0, 40, size=(repeats, 40))
    counts = np.zeros((repeats, 40), np.float64)
    for b in range(repeats):
        counts[b] = np.bincount(draws[b], minlength=40)
    max_t = np.zeros(repeats)
    entries, distributions, family_undefined = [], [], False
    for comp in comparisons:
        a, b = stats[comp['candidate']], stats[comp['control']]
        def ratio(s):
            nums = np.asarray(s['sse'], np.float64)
            dens = np.asarray(s['energy'], np.float64)
            assert nums.shape == dens.shape and nums.shape[-1] == 40
            if nums.ndim == 1:
                nums, dens = nums[None], dens[None]
            den = counts @ dens.T
            num = counts @ nums.T
            undefined = np.any(den <= 0, axis=1)
            pointden = dens.sum(axis=1)
            estimate = None if np.any(pointden <= 0) else float(np.mean(nums.sum(axis=1) / pointden))
            with np.errstate(divide='ignore', invalid='ignore'):
                boot = np.mean(num / den, axis=1)
            return estimate, boot, undefined
        ah, ab, au = ratio(a)
        bh, bb, bu = ratio(b)
        undefined = au | bu
        delta = None if ah is None or bh is None else ah - bh
        literal = comp['id'] in literal_ties
        entry = {**comp, 'difference': delta, 'undefined_bootstrap_draws': int(undefined.sum()), 'literal_function_tie': literal}
        d = ab - bb
        if undefined.any() or delta is None or not np.isfinite(d).all():
            entry['inference'] = 'inconclusive_undefined_hit_or_energy'
            family_undefined = True
            distributions.append(None)
        else:
            sd = float(np.std(d, ddof=1))
            entry['bootstrap_sd'] = sd
            if sd == 0 and not literal:
                entry['inference'] = 'inconclusive_nonliteral_zero_studentization'
                family_undefined = True
                distributions.append(None)
            elif literal:
                assert delta == 0 and np.all(d == 0)
                entry['inference'] = 'literal_exact_tie'
                entry['simultaneous_band'] = [0., 0.]
                distributions.append(None)
            else:
                max_t = np.maximum(max_t, np.abs((d - delta) / sd))
                entry['descriptive_pointwise95'] = [float(x) for x in np.quantile(d, [.025, .975], method='linear')]
                entry['inference'] = 'pending_family'
                distributions.append(sd)
        entries.append(entry)
    cutoff = None if family_undefined else float(np.quantile(max_t, .95, method='linear'))
    for entry, sd in zip(entries, distributions):
        if sd is not None:
            entry['inference'] = 'inconclusive_family_undefined' if family_undefined else 'simultaneous_approximate_familywise_band'
            if cutoff is not None:
                entry['simultaneous_band'] = [entry['difference'] - cutoff * sd, entry['difference'] + cutoff * sd]
    return {'seed': SEED, 'repeats': repeats, 'same_40_prompt_ids': prompt_ids, 'max_t_cutoff95': cutoff,
            'family_inconclusive': family_undefined, 'comparisons': entries,
            'classification': 'Root_and_critic_only; no automatic scientific success/failure'}
