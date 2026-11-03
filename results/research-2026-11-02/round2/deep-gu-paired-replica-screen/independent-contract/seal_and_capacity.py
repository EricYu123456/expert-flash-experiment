#!/usr/bin/env python3
"""Close the independent audit, including per-chip block lower-bound equality."""
import copy
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    frozen = json.loads((HERE / 'inputs.frozen.json').read_text())
    for item in frozen['inputs']:
        assert sha(item['path']) == item['sha256']
    protocol = json.loads((HERE.parent / 'protocol.frozen.json').read_text())
    for item in protocol['inputs']:
        assert sha(item['path']) == item['sha256']
    addresses = json.loads((HERE / 'resident_addresses.json').read_text())
    ledger = addresses['ledger']
    per_chip = [sum(j['waves'] for j in addresses['jobs'] if j['chip'] == c) for c in range(16)]
    lower = [(w + 191) // 192 for w in per_chip]
    assert len(addresses['jobs']) == 2080 and sum(ledger['chip_blocks']) == 63
    assert sum(lower) == 62
    assert sum(per_chip) == 9289
    assert ledger['useful_bytes'] == int(576.5 * 1048576)
    assert ledger['allocated_lsb_block_bytes'] == 756 * 1048576
    dump = lambda p, x: Path(p).write_text(json.dumps(x, indent=2) + '\n')
    # BFD leaves one unnecessary block on chip11, whose total is exactly576waves.
    # A direct packing of its 64x6,48x3,16x2,11,5-wave jobs attains three blocks.
    repaired = copy.deepcopy(addresses)
    chip_jobs = [j for j in repaired['jobs'] if j['chip'] == 11]
    six = [j for j in chip_jobs if j['waves'] == 6]
    two = [j for j in chip_jobs if j['waves'] == 2]
    other = [j for j in chip_jobs if j['waves'] in (5, 11)]
    three = [j for j in chip_jobs if j['waves'] == 3]
    assert [len(six), len(two), len(other), len(three)] == [64, 16, 2, 48]
    bins = [six[:29] + two[:1] + other, six[29:61], six[61:] + two[1:] + three]
    assert all(sum(j['waves'] for j in jobs) == 192 for jobs in bins)
    assert sorted(j['job'] for jobs in bins for j in jobs) == sorted(j['job'] for j in chip_jobs)
    for block, jobs in enumerate(bins):
        wordline = 0
        for j in jobs:
            j.update(block=block, wordline=wordline, end_wordline=wordline + j['waves'])
            wordline += j['waves']
    repaired['ledger']['chip_blocks'] = lower
    repaired['ledger']['allocated_lsb_block_bytes'] = sum(lower) * 192 * 65536
    repaired['placement'] += ' One explicit chip11 packing repair after BFD reaches fixed-inventory block lower bound.'
    dump(HERE / 'resident_addresses.repaired62.json', repaired)
    dump(HERE / 'capacity_lower_bound.json', dict(per_chip_reserved_waves=per_chip,
                                                per_chip_minimum_blocks=lower,
                                                BFD_blocks=ledger['chip_blocks'], BFD_total_blocks=63,
                                                repaired_total_blocks=62,
                                                repaired_allocated_lsb_block_MiB=744,
                                                chip11_block_partition_waves=[192, 192, 192],
                                                caveat='Fixed chip inventory with static repack, not optimal arbitrary placement.'))
    target = HERE / 'completion.json'
    assert not target.exists()
    paths = sorted(p for p in HERE.rglob('*') if p.is_file() and p != target)
    dump(target, dict(status='SEALED_PASS_BOUNDED_NO_GO',
                      audit_source=str(frozen['inputs'][0]['path']),
                      audit_source_sha256=frozen['inputs'][0]['sha256'],
                      root_inputs_unchanged=True, no_weights_or_heldout_read=True,
                      files=[dict(path=str(p), bytes=p.stat().st_size, sha256=sha(p)) for p in paths]))
    print(json.dumps(dict(completion=str(target), completion_sha256=sha(target),
                          per_chip_reserved_waves=per_chip, files=len(paths)), indent=2))


if __name__ == '__main__':
    main()
