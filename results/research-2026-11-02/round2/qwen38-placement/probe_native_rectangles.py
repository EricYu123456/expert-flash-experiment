#!/usr/bin/env python3
"""Full Qwen38 matrix shapes, known integer weights, native idealFP32 Tier.

This verifies algebra and fixed256x256 decomposition, not AiF numeric ABI or
real-model quality. Host scatter/add combines partial outputs. No timing model
or new native operation is inserted. Original setter fringe bug is avoided by
explicitly programming existing public Tier.weight at the same addresses.
"""
import functools
import hashlib
import json
from pathlib import Path
import subprocess
import sys

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parent/'moewithpim-audit/upstream'
sys.path.insert(0,str(UPSTREAM/'src'))
import torch
from threedsim.accelerator import Accelerator, AcceleratorConfig
from threedsim.modules import Linear
from threedsim.modules.base import assign_acc

torch.set_num_threads(1)
torch.manual_seed(20)


def execute(definition, spread_atoms=False):
    rows,cols=definition['matrix_shape']
    W=((torch.arange(rows)[:,None]*17+torch.arange(cols)[None,:]*11)%7-3).float()
    x=(torch.arange(cols)%5-2).float()
    target=W@x
    output=torch.zeros(rows)
    # Identical native geometry/capacity for all layouts; no new context.
    acc=Accelerator(AcceleratorConfig(tiles=16,tiers=16,tier_shape=(256,256)),device='cpu')
    calls=[]
    addresses=[]
    for job,(start,h,col,w) in enumerate(definition['rectangles']):
        layer=Linear(w,h,device='cpu')
        assign_acc(layer,acc)
        mapping=[]
        for i in range((w+255)//256):
            line=[]
            for j in range((h+255)//256):
                nr,nc=min(256,w-i*256),min(256,h-j*256)
                local_atom=i*((h+255)//256)+j
                tile,tier=(local_atom%16,local_atom//16) if spread_atoms else (job,local_atom)
                line.append((tile,tier,nr*nc/(256*256),nr,nc))
            mapping.append(line)
        layer.set_mapping(mapping)
        layer.set_name(f'job{job}')
        weight=W[start:start+h,col:col+w].T.contiguous()
        layer.weight.data=weight
        for i,line in enumerate(mapping):
            for j,(tile,tier,_,nr,nc) in enumerate(line):
                t=acc.tiles[tile].tiers[tier]
                t.weight.zero_()
                t.weight[:nr,:nc]=weight[i*256:i*256+nr,j*256:j*256+nc]
                original=t.forward
                @functools.wraps(original)
                def counted(*a,_original=original,_address=(tile,tier),**kw):
                    calls.append(_address)
                    return _original(*a,**kw)
                t.forward=counted
                addresses.append((tile,tier))
        y=layer(x[col:col+w],op_info={'token_id':0,'seq_len':1,'decoding_id':0})
        output[start:start+h]+=y
    assert len(addresses)==len(set(addresses))
    assert len(calls)==len(addresses) and set(calls)==set(addresses)
    assert torch.equal(output,target), (output-target).abs().max().item()
    return dict(numeric_exact=True,max_abs_error=(output-target).abs().max().item(),
                shape=[rows,cols],native_tier_shape=[256,256],
                observed_native_tier_calls=len(calls),unique_fixed_addresses=len(addresses),
                useful_coefficients=rows*cols,reserved_native_coefficients=len(addresses)*256*256,
                native_coefficient_utilization=rows*cols/(len(addresses)*256*256),
                dense_original_grid_MVMs=((cols+255)//256)*((rows+255)//256),
                output_sha256=hashlib.sha256(output.numpy().tobytes()).hexdigest(),
                addresses_sha256=hashlib.sha256(json.dumps(addresses).encode()).hexdigest())


def main():
    inventory=json.loads((HERE/'independent_rectangles.json').read_text())
    expected=json.loads((HERE/'independent_geometry_audit.json').read_text())
    results={k:execute(v) for k,v in inventory['controls'].items()}
    for name,r in results.items():
        assert r['observed_native_tier_calls']==expected['controls'][name]['native_256_square_MVMs']
    dense={phase:execute(dict(matrix_shape=shape,rectangles=[(0,shape[0],0,shape[1])]),spread_atoms=True)
           for phase,shape in [('gate_up',[1280,2560]),('down',[2560,640])]}
    assert dense['gate_up']['observed_native_tier_calls']==50 and dense['down']['observed_native_tier_calls']==30
    report=dict(status='PASS full-matrix native idealFP32 rectangle composition',
                upstream_commit=subprocess.check_output(['git','rev-parse','HEAD'],cwd=UPSTREAM,text=True).strip(),
                hardware=dict(tiles=16,tiers_per_tile=16,tier_shape=[256,256]),
                results=results,native_dense_atom_mapping_controls=dense,
                numerical_scope='Known small integer values in fullQwen38 matrix dimensions; exactfloat32 dot products and host sums. No pretrainedweights, checkpointquantization, scaledoutputABI or physicalADC error simulation.',
                programming='Outside adapter programs existingTier.weight slices with explicitfixedgeometry; no changes to originalTier.forward or Linear except per-instanceforward wrapper recording calls and returning originalresult.',
                timing_scope='No schedule or chiptime executed here. AiFwave, command and channel quantities are not substituted for nativeTier MVM cost.',
                source_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                upstream_git_diff=subprocess.check_output(['git','diff','--stat'],cwd=UPSTREAM,text=True))
    assert report['upstream_git_diff']==''
    out=HERE/'native_rectangle_probe.json'
    out.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(path=str(out),status=report['status'],native_calls={k:v['observed_native_tier_calls'] for k,v in results.items()})))


if __name__=='__main__':main()
