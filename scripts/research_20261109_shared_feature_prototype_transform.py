#!/usr/bin/env python3
"""Prototype metadata adapter over the frozen source-0/1 numerical kernel.

Source 0 is raw x, 1 native shared_z, 2 a native routed prototype's raw phi.
Concatenation only creates a source pool; it adds no quantization boundary.
The gathered transformed feature vector receives the single declared A8.
"""
import numpy as np


def remap(sources, indices, prototypes, shared_width=2816, prototype_width=1408):
    sources = np.asarray(sources, np.uint8)
    indices = np.asarray(indices, np.int32)
    prototypes = np.asarray(prototypes, np.int32)
    assert sources.shape == indices.shape == prototypes.shape
    assert np.isin(sources, [0, 1, 2]).all()
    assert (prototypes[sources != 2] == -1).all()
    assert ((prototypes[sources == 2] >= 0) & (prototypes[sources == 2] < 64)).all()
    assert ((indices[sources == 0] >= 0) & (indices[sources == 0] < 2048)).all()
    assert ((indices[sources == 1] >= 0) & (indices[sources == 1] < shared_width)).all()
    assert ((indices[sources == 2] >= 0) & (indices[sources == 2] < prototype_width)).all()
    owners = sorted(set(map(int, prototypes[sources == 2])))
    include_shared = bool((sources == 1).any())
    offsets = {e: (shared_width if include_shared else 0) + i * prototype_width for i, e in enumerate(owners)}
    kernel_sources = np.minimum(sources, 1).astype(np.uint8)
    kernel_indices = indices.copy()
    for j in np.flatnonzero(sources == 2):
        kernel_indices[j] += offsets[int(prototypes[j])]
    return kernel_sources, kernel_indices, owners, include_shared


def apply_cpp(numeric, x, shared_z, prototype_hidden, sources, indices, prototypes, center, inv_scale):
    x, shared_z = np.ascontiguousarray(x, np.float32), np.ascontiguousarray(shared_z, np.float32)
    kernel_sources, kernel_indices, owners, include_shared = remap(sources, indices, prototypes, shared_z.shape[1])
    parts = ([shared_z] if include_shared else []) + [np.asarray(prototype_hidden[e], np.float32) for e in owners]
    pool = np.ascontiguousarray(np.concatenate(parts, axis=1)) if parts else np.zeros((len(x), 1), np.float32)
    center, inv_scale = np.ascontiguousarray(center, np.float32), np.ascontiguousarray(inv_scale, np.float32)
    out = np.empty((len(x), len(sources)), np.float32)
    assert numeric.lib.sf_transform(len(x), x.shape[1], pool.shape[1], len(sources), x, pool,
                                    kernel_sources, kernel_indices, center, inv_scale, out) == 0
    return out


def apply_numpy(x, shared_z, prototype_hidden, sources, indices, prototypes, center, inv_scale):
    raw = np.stack([x[:, j] if s == 0 else shared_z[:, j] if s == 1 else prototype_hidden[int(e)][:, j]
                    for s, j, e in zip(sources, indices, prototypes)], axis=1)
    return np.multiply(np.subtract(raw, center, dtype=np.float32), inv_scale, dtype=np.float32)
