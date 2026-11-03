#!/usr/bin/env python3
"""Independent raw-I8 rectangle coverage/cost audit; no route files opened.

AiF waves are four16KiB pages. These are not analog256x256 crossbar operations.
One dense command per rectangle and full-precision partial outputs remain an
existing-interface condition for AiF, not a demonstrated device ABI.
"""
import collections
import hashlib
import json
import math
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PAGE = 16384
WAVE = 4 * PAGE


def grid_h26():
    r = [(i*128,128,2*512,3*512) for i in range(10)]
    r += [(i*128,3*128,j*512,512) for i in (0,3) for j in (0,1)]
    r += [(i*128,2*128,0,2*512) for i in (6,8)]
    return r


def stripe_min_output():
    r = [(i*128,128,0,1536) for i in range(10)]
    start = 0
    for h in [192]*4+[256]*2:
        r.append((start,h,1536,1024))
        start += h
    assert start == 1280
    return r


def stripe_min_io():
    r = [(i*256,256,j*768,768) for j in (0,1) for i in range(5)]
    start = 0
    for h in [192]*4+[256]*2:
        r.append((start,h,1536,1024))
        start += h
    assert start == 1280
    return r


def down_min_output():
    r = [(i*256,256,0,512) for i in range(9)]
    r += [(2304+i*128,128,0,512) for i in range(2)]
    r += [(i*512,512,512,128) for i in range(5)]
    return r


def original_h38():
    r = [(i*128,3*128,j*512,512) for i in (0,3) for j in range(5)]
    r += [(i*128,2*128,0,1024) for i in (6,8)]
    r += [(i*128,128,1024,1536) for i in range(6,10)]
    return r


def metric(rectangles, rows, cols):
    coverage = np.zeros((rows,cols), dtype=np.uint8)
    jobs = []
    for idx, (start,h,col,w) in enumerate(rectangles):
        assert h>0 and w>0 and 0<=start<start+h<=rows and 0<=col<col+w<=cols
        coverage[start:start+h,col:col+w] += 1
        size = h*w
        jobs.append(dict(job=idx,row_start=start,rows=h,col_start=col,columns=w,
                         weight_bytes=size,pages=math.ceil(size/PAGE),waves=math.ceil(size/WAVE),
                         input_i8_bytes=w,partial_output_elements=h,
                         native_256_square_MVMs=math.ceil(w/256)*math.ceil(h/256)))
    assert np.all(coverage == 1), collections.Counter(coverage.flatten())
    partials = sum(j['rows'] for j in jobs)
    inputs = sum(j['columns'] for j in jobs)
    return dict(coverage_exactly_once=True, matrix_rows=rows,matrix_columns=cols,
                jobs=jobs,num_jobs=len(jobs),weight_bytes=sum(j['weight_bytes'] for j in jobs),
                page_reads=sum(j['pages'] for j in jobs),total_chip_waves=sum(j['waves'] for j in jobs),
                waves_histogram=dict(collections.Counter(j['waves'] for j in jobs)),
                all_page_and_wave_exact=all(j['weight_bytes']%WAVE==0 for j in jobs),
                input_i8_bytes=inputs,partial_output_elements=partials,
                extra_scalar_output_adds=partials-rows,
                io_by_output_bytes={str(beta):dict(output_bytes=partials*beta,total_input_plus_output_bytes=inputs+partials*beta) for beta in (1,2,4,8)},
                native_256_square_MVMs=sum(j['native_256_square_MVMs'] for j in jobs),
                native_256_scope='Independent fixed256x256 idealTier decomposition count only; not AiF commands/waves or calibrated timing.')


def whole_rows(rows,cols,counts):
    starts=np.cumsum([0]+counts[:-1]).tolist()
    return metric([(s,h,0,cols) for s,h in zip(starts,counts)],rows,cols)


def divisor_io_min(area, max_rows, max_cols):
    choices=[]
    for h in range(1,max_rows+1):
        if area%h==0 and area//h<=max_cols:
            choices.append((area//h+4*h,h,area//h))
    best=min(x[0] for x in choices)
    return dict(min_input_plus_four_byte_output=best,shapes_rows_columns=[list(x[1:]) for x in choices if x[0]==best])


def main():
    definitions={
      'gate_up_grid_h26':dict(phase='gate_up',matrix_shape=[1280,2560],rectangles=grid_h26()),
      'gate_up_two_stripes_min_output':dict(phase='gate_up',matrix_shape=[1280,2560],rectangles=stripe_min_output()),
      'gate_up_three_stripes_min_io4':dict(phase='gate_up',matrix_shape=[1280,2560],rectangles=stripe_min_io()),
      'down_min_output':dict(phase='down',matrix_shape=[2560,640],rectangles=down_min_output()),
    }
    for d in definitions.values():
        assert len(d['rectangles'])==16
    inventory=dict(schema_version=1,rectangle_fields=['row_start','rows','col_start','columns'],
                   scope='RawI8 one-copy dense rectangles; no routes/heldout read. No physical-chip assignment is implied by array order.',
                   controls=definitions)
    (HERE/'independent_rectangles.json').write_text(json.dumps(inventory,indent=2)+'\n')
    controls={name:metric(d['rectangles'],*d['matrix_shape']) for name,d in definitions.items()}
    for name,d in controls.items():
        assert d['num_jobs']==16
        assert d['total_chip_waves']==(25 if name=='down_min_output' else 50)
        assert d['waves_histogram']==({1:7,2:9} if name=='down_min_output' else {3:14,4:2})
        assert d['all_page_and_wave_exact']
    root_cert=json.loads((HERE/'geometry_certificates.json').read_text())
    baselines={}
    for phase in root_cert['phases']:
        baselines[phase['name']]=whole_rows(phase['rows'],phase['columns'],phase['balanced_whole_wave']['rows'])
        assert baselines[phase['name']]['total_chip_waves']==phase['balanced_whole_wave']['total_waves']
    original=metric(original_h38(),1280,2560)
    low=divisor_io_min(3*WAVE,1280,2560)
    high=divisor_io_min(4*WAVE,1280,2560)
    assert low['min_input_plus_four_byte_output']==1792
    assert high['min_input_plus_four_byte_output']==2048
    assert 14*1792+2*2048==controls['gate_up_three_stripes_min_io4']['io_by_output_bytes']['4']['total_input_plus_output_bytes']
    report=dict(
      status='PASS independent cell-level geometry and cost audit',
      source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
      root_certificate_sha256=hashlib.sha256((HERE/'geometry_certificates.json').read_bytes()).hexdigest(),
      rectangle_inventory_sha256=hashlib.sha256((HERE/'independent_rectangles.json').read_bytes()).hexdigest(),
      controls=controls,original_root_grid_h38=original,whole_row_balanced_baselines=baselines,
      lower_bounds={
        'pooled_weight_work':dict(gate_up=50,down=25,whole_row_balanced_total=77,exact2D_total=75,relative_work_reduction=2/77),
        'GU_min_output':dict(partial_rows=2560,achieved_by='gate_up_two_stripes_min_output',proof='Every exact3/4wave rectangle has area3*65536 or4*65536. Neither is divisible by full input width2560, so no rectangle spans the whole width. Thus each of1280 output rows has at least2 contributions; two stripes attain2560.'),
        'GU_grid_H26':dict(partial_rows=3328,scope='128output-row by512input-column grid,14area3 and2area4 rectangles',proof='LetV be vertical3x1 bars, H be horizontal1x4 bars,S squares,U vertical4x1 bars. At most one width>=3 rectangle fits each of10 grid rows, so V>=4+H. Output height sum=16+2V+S+3U>=26 because H+S+U=2. The construction attains26.'),
        'GU_min_io_beta4':dict(low_job=low,high_job=high,total_bytes=29184,achieved_by='gate_up_three_stripes_min_io4',scope='Exactly16 jobs,14 exact3waves+2exact4waves; oneI8 input byte percolumn percommand and4outputbytes perrow, no intercommandinput residency'),
        'down_min_output':dict(partial_rows=5120,achieved_by='down_min_output',proof='Exact1/2wave areas have only power-of-two integer widths. Full640 inputcolumns cannot occur; maximum legalwidth<=640 is512. Every outputrow therefore has at least2 contributions. The512+128 stripe construction attains2*2560.'),
        'down_min_input_among_min_output_grid':dict(input_bytes=6272,scope='128x128 quarterwave grid with9area8 and7area4 rectangles, minimum40rowunits',proof='Each row must partition4+1. If a low and b high jobs usewidth4 then a+2b=20,a<=7,b<=9. Suminputwidthunits=16+3(a+b), minimized byb9,a2, yielding49*128.'),
      },
      costs_not_inferred=['PhysicalAiF outputdtype/scale/rounding; rawpartial ABI remains unverified','Hostcopy/reduction/nonlinearity time','Channel-specific placement, inputbroadcast reuse and PCIetraffic policy','Block packing, commandrelease and selected-route tail latency','NVFP4 checkpoint byte representation or its precision'],
      native_backend_boundary='Original idealFP32 Tier can express eachrectangle via dense256x256 chunks and digital_add, but oneAiFrectangle is not oneTier op. RootGUgridH38 needs72 nativeTier MVMs, gridH26 andtwo-stripes need84, three-stripes needs54; the dense matrix requires50. Downrectangle scheme needs32 vs dense30. Do not transfer AiF50/25wave claims to this backend.',
    )
    (HERE/'independent_geometry_audit.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(rectangles=str((HERE/'independent_rectangles.json').resolve()),status=report['status'],summary={k:dict(waves=v['total_chip_waves'],inputs=v['input_i8_bytes'],partial_rows=v['partial_output_elements'],adds=v['extra_scalar_output_adds'],io4=v['io_by_output_bytes']['4']['total_input_plus_output_bytes'],tier256=v['native_256_square_MVMs']) for k,v in controls.items()})))


if __name__=='__main__':main()
