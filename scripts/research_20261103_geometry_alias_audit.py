#!/usr/bin/env python3
"""Detect equivalent geometry searches before attributing compiler gains."""
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2/ordinary-read-compiler'


def key(candidate, experts):
    base = np.asarray(candidate['base_rows'], dtype=np.int64)
    if candidate['family'] == 'disjoint':
        width = int(candidate['label'].split('_')[-1])
        variants = [tuple(np.roll(base, (v % (16 // width)) * width).tolist()) for v in range(16)]
        labels = np.arange(experts) % (16 // width)
    else:
        variants = [tuple(np.roll(base, v).tolist()) for v in range(16)]
        labels = np.arange(experts) * max(1, 16 // experts) % 16
    initial_histogram = Counter(variants[v] for v in labels)
    return tuple(sorted(variants)), tuple(sorted(initial_histogram.items()))


def main():
    source = BASE / 'full-campaign-v3/summary.json'
    doc = json.loads(source.read_text()); records = []
    for r in doc['records']:
        classes = []
        for phase in r['fitting']:
            groups = {}
            for c in phase:
                groups.setdefault(key(c, r['shape'][2]), []).append(c)
            lookup = {c['label']: members for members in groups.values() for c in members}
            classes.append(lookup)
        whole = next(s for s in r['selection'] if s['family'] == 'whole_rows')
        cyclic = next(s for s in r['selection'] if s['family'] == 'cyclic')
        labels = whole['gu_label'], whole['down_label']
        equivalent = [any(c['family'] in ('disjoint', 'cyclic') for c in classes[p][label]) for p, label in enumerate(labels)]
        improvement = 1 - whole['mean_ns'] / cyclic['mean_ns']
        records.append(dict(model=r['model'], layer=r['layer'], command_ns=r['command_ns'],
                            improvement_vs_cyclic=improvement,
                            winner_both_geometries_already_in_controls=all(equivalent),
                            aliases=[[c['label'] for c in classes[p][label]] for p, label in enumerate(labels)],
                            attribution='Extra label-search opportunity, not a distinct geometry; rerun with deduplication.' if improvement > 0 and all(equivalent) else
                                        'At least one distinct geometry; this does not itself establish novelty or isolate all search effects.'))
    totals = []
    for model in sorted({r['model'] for r in records}):
        for command in (250, 5000, 10000):
            subset = [r for r in records if r['model'] == model and r['command_ns'] == command]
            totals.append(dict(model=model, command_ns=command, layers=len(subset),
                positive_layers=sum(r['improvement_vs_cyclic'] > 1e-12 for r in subset),
                positive_layers_with_only_existing_geometries=sum(r['improvement_vs_cyclic'] > 1e-12 and r['winner_both_geometries_already_in_controls'] for r in subset),
                winner_has_distinct_geometry_layers=sum(not r['winner_both_geometries_already_in_controls'] for r in subset)))
    out = BASE / 'full-campaign-v3/geometry_alias_audit.json'
    out.write_text(json.dumps(dict(created_utc=datetime.now(timezone.utc).isoformat(),
        status='Completed attribution audit; v3 gains are not corrected causal estimates',
        source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        campaign_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        equivalence='Same multiset of16 physical variant row vectors and same initial histogram of expert-assigned physical row vectors. The compiler lowers both by ascending physical chip order.',
        records=records, totals=totals,
        remedy='Deduplicate before fitting and reuse the same labels for every equivalent family. A new seeded fit of the same geometry is a strong control, not a new physical mechanism.'), indent=2) + '\n')
    print(json.dumps(totals, indent=2))


if __name__ == '__main__':
    main()
