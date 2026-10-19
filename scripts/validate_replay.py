#!/usr/bin/env python3
"""Compare NVMeVirt kernel counters to independently generated expected demand."""
import argparse
import csv
import io
import json
from pathlib import Path


def validate(expected_path, stats_path, wall_path, v2_stats_path=None):
    expected=json.loads(expected_path.read_text())
    lines=stats_path.read_text().splitlines()
    header={}
    for line in lines[:4]:
        key,value=line.split(maxsplit=1)
        header[key]=value
    assert header['model']==expected.get('model','metadata_only_int8_v1'),header
    for key in ('commands','model_ns','active_chip_commands'):
        assert int(header[key])==expected[key],(key,header[key],expected[key])
    chips=list(csv.DictReader(io.StringIO('\n'.join(lines[4:]))))
    assert len(chips)==16
    fields={'useful_bytes':'useful_bytes','physical_pages':'physical_pages',
            'read_rounds':'read_rounds','cold_reads':'cold_reads','service_ns':'chip_service_ns',
            'active_commands':'active_chip_commands'}
    for actual,total in fields.items():
        assert sum(int(c[actual]) for c in chips)==expected[total],actual
    for chip,row in enumerate(chips):
        assert int(row['chip'])==chip
        assert int(row['useful_bytes'])==expected['chip_bytes'][chip]
        assert int(row['service_ns'])==expected['chip_service_ns_by_chip'][chip]
    wall=json.loads(wall_path.read_text())
    assert wall['commands']==expected['commands']
    # The modeled completion target is a lower bound; VM/driver costs remain visible.
    assert wall['ioctl_wall_ns'] >= expected['model_ns'],(wall,expected['model_ns'])
    if v2_stats_path:
        more={k:int(v) for k,v in (line.split(maxsplit=1) for line in v2_stats_path.read_text().splitlines())}
        for key in ('flash_ns','control_ns','host_ns','input_ns','output_ns',
                    'vector_input_bytes','vector_output_bytes','chip_matrix_commands'):
            assert more[key]==expected[key],(key,more[key],expected[key])
        if expected.get('model')=='metadata_only_int8_v3':
            for key in ('pcie_input_bytes','pcie_output_bytes'):
                assert more[key]==expected[key],(key,more[key],expected[key])
    return dict(label=expected['label'],commands=expected['commands'],
                all_kernel_counters_match=True, model_ns=expected['model_ns'],
                overhead_counters_checked=bool(v2_stats_path),
                ioctl_wall_ns=wall['ioctl_wall_ns'],
                ioctl_overhead_ns=wall['ioctl_wall_ns']-expected['model_ns'])


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('expected',type=Path)
    p.add_argument('stats',type=Path)
    p.add_argument('wall',type=Path)
    p.add_argument('--output',type=Path)
    p.add_argument('--v2-stats',type=Path)
    a=p.parse_args()
    r=validate(a.expected,a.stats,a.wall,a.v2_stats)
    text=json.dumps(r,indent=2)+'\n'
    if a.output:
        a.output.write_text(text)
    print(text,end='')


if __name__=='__main__':
    main()
