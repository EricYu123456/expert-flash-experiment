#!/usr/bin/env python3
"""Freeze complete W8 TRAIN archives as inputs to the ordinary-read compiler."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / 'results/research-2026-11-02/round2/w8-format-training'


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def main():
    output = BASE / 'compiler_inputs.frozen.json'
    assert not output.exists(), 'Retain the existing frozen compiler input manifest.'
    report = json.loads((BASE / 'summary.json').read_text())
    protocol = json.loads((BASE / 'protocol.frozen.json').read_text())
    assert report['status'] == 'complete'
    assert report['protocol_sha256'] == sha(BASE / 'protocol.frozen.json')
    records = {}; identities = None
    for model, layers in [('granite4tiny', list(range(40))), ('deepseek', list(range(1, 27)))]:
        archive = BASE / model / 'decode.routes.npz'
        entry = report['models'][model]
        assert entry['status'] == 'complete' and entry['completed_prompts'] == 32
        assert sha(archive) == entry['decode_archive_sha256']
        with np.load(archive) as f:
            assert f['routes'].shape == (32, 16, len(layers), 6)
            assert f['layers'].tolist() == layers and f['completed'].all()
            assert f['representation'].item() == 'fixed_full_FFN_W8A8_canonical_only'
            assert f['split'].item() == 'train' and np.all((f['routes'] >= 0) & (f['routes'] < 64))
            ids = f['prompt_ids'].tolist()
            assert len(set(ids)) == 32 and all(s.startswith('train_') for s in ids)
            if identities is None:
                identities = ids
            else:
                assert identities == ids
            for selected in f['routes'].reshape(-1, 6):
                assert len(set(selected.tolist())) == 6
        records[model] = dict(archive=str(archive), sha256=sha(archive), status='complete',
                              shape=[32, 16, len(layers), 6], layers=layers,
                              representation='fixed_full_FFN_W8A8_canonical_only',
                              source_model=protocol['models'][model]['model'])
    result = dict(status='complete actual-format TRAIN archives frozen before compiler fit',
                  created_utc=datetime.now(timezone.utc).isoformat(),
                  source_sha256=sha(__file__), capture_protocol_sha256=sha(BASE / 'protocol.frozen.json'),
                  capture_summary_sha256=sha(BASE / 'summary.json'),
                  selection_protocol_sha256=sha(BASE / 'selection_protocol.frozen.json'),
                  prompt_ids=identities, models=records,
                  limits='Canonical-only new TRAIN routing. No prefill mixed into decode; no heldout/quality/native-device claim. Prior full-FFN paired numerical proofs are separate artifacts.')
    output.write_text(json.dumps(result, indent=2) + '\n')
    print(str(output), sha(output), flush=True)


if __name__ == '__main__':
    main()
