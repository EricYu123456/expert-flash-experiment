#!/usr/bin/env python3
"""Frozen full512 continuation with explicit selection-gate naming erratum.

Preserve all original source/protocol/selection artifacts, including the false
prefix-only gate flag. Re-evaluate exactly the already-declared numeric gate.
"""
from pathlib import Path
import argparse
import shutil
import full_ffn_addendum as driver

HERE=Path(__file__).resolve().parent
SCREEN=HERE/'two-block-addendum'
DEST=HERE/'final-run'


def freeze():
    DEST.mkdir(exist_ok=True);assert not (DEST/'protocol.frozen.json').exists()
    selected=driver.f.read(SCREEN/'selection.frozen.json')
    assert selected['screen_report_sha256']==driver.f.sha(SCREEN/'screen/report.json')
    assert selected['screen_arrays_sha256']==driver.f.sha(SCREEN/'screen/service_arrays.npz')
    gate=any(not selected['winners'][g].startswith('original_') and
             selected['comparisons'][g]['mean_gain']>0.01 for g in driver.LAYOUTS)
    assert gate,'Numeric protocol gate must pass; no override of a failed gain threshold.'
    # Exact copies keep the chosen plans and screen evidence immutable.
    (DEST/'screen').mkdir(exist_ok=True)
    for relative in ('menus.frozen.npz','screen/report.json','screen/service_arrays.npz'):
        shutil.copyfile(SCREEN/relative,DEST/relative)
    protocol=driver.f.read(SCREEN/'protocol.frozen.json')
    protocol=dict(protocol,inputs_sha256=dict(protocol['inputs_sha256']))
    additions=[Path(__file__),SCREEN/'protocol.frozen.json',SCREEN/'selection.frozen.json',
               DEST/'menus.frozen.npz',SCREEN/'screen/report.json',SCREEN/'screen/service_arrays.npz',
               driver.BASE/'asymmetric-down/TRAIN512/service_arrays.npz']
    for p in additions:protocol['inputs_sha256'][str(p)]=driver.f.sha(p)
    protocol.update(status='frozen_full512_continuation',frozen_utc=driver.f.utc(),
        parent_two_block_protocol_sha256=driver.f.sha(SCREEN/'protocol.frozen.json'),
        parent_two_block_selection_sha256=driver.f.sha(SCREEN/'selection.frozen.json'),
        gate_erratum='Parent executor classified newmenus using dedicated-prefix only; all64_* names were omitted. The quantitative>1% rule passed and allwinners/PB/FIFOvalidity are unchanged. Preserve falseparent flag; this newcontinuation evaluates thedeclared numericcriterion explicitly.',
        execution='Same unchanged full_ffn_addendum.execute and asymmetric_down.graph_for/KernelRecipes. Selectedfamilywinner and originalfamilyincumbent, input1/4 GF/DF on all512TRAINqueries ×3layers; nonewfit orheldout.')
    chosen=dict(selected,full512_gate_passed=True,gate_corrected_by=str(Path(__file__)),
                parent_selection_sha256=driver.f.sha(SCREEN/'selection.frozen.json'),
                correction='Naming classification only; no changed plans, cost outcomes, threshold, or validity rule.')
    driver.f.dump(DEST/'protocol.frozen.json',protocol)
    driver.f.dump(DEST/'selection.frozen.json',chosen)
    driver.f.dump(DEST/'gate_metadata_erratum.json',dict(parent_selection=str(SCREEN/'selection.frozen.json'),
        parent_gate=selected['full512_gate_passed'],correct_numeric_gate=gate,winners_unchanged=True,
        reason=protocol['gate_erratum']))
    print('Frozen full512 continuation; selected maps unchanged.',flush=True)


if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);args=ap.parse_args()
    if args.mode=='freeze':freeze()
    else:
        driver.DEST=DEST
        driver.execute('run')
