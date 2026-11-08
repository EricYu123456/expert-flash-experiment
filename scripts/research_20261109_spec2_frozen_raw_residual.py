#!/usr/bin/env python3
"""Fit small shared corrections to an immutable optimized canonical raw-x.

The original raw registry stays the exact fallback. A folded concat input has
a different A8 context; B=0 controls measure that difference explicitly.
"""
import argparse
import json
from pathlib import Path
import time

import numpy as np
from threadpoolctl import threadpool_limits

import research_20261109_spec2_diagnostics as d
from research_20261109_shared_feature_targets import Numeric


def raw_entry(layer, expert):
    folder = d.DATA / f'spec2-screen16/layer{layer}'
    sel = json.loads((folder / 'selection.json').read_text())
    entry = next(r for r in sel['models'] if r['basis'] == 'raw_x_eq' and r['expert'] == expert)
    prefix = entry['readout_prefix']
    model = {'qweight': np.fromfile(prefix + '.i8', np.int8).reshape(2048, 2048),
             'wscale': np.fromfile(prefix + '.scales.f32', np.float32),
             'bias': np.fromfile(prefix + '.bias.f32', np.float32)}
    tp = entry['transform_prefix']
    sources, indices = np.fromfile(tp + '.sources.u8', np.uint8), np.fromfile(tp + '.indices.i32', np.int32)
    assert np.all(sources == 0) and np.array_equal(indices, np.arange(2048))
    transform = {'sources': sources, 'indices': indices,
                 'center': np.fromfile(tp + '.center.f32', np.float32),
                 'inv_scale': np.fromfile(tp + '.inv_scale.f32', np.float32)}
    rawline = f'{layer}\t{expert}\traw_x_eq\t2048\t{tp}\t{prefix}'
    identities = {str(folder / 'selection.json'): d.fit.old.sha(folder / 'selection.json')}
    for pre, suffixes in ((prefix, ('.i8','.scales.f32','.bias.f32')), (tp, ('.sources.u8','.indices.i32','.center.f32','.inv_scale.f32'))):
        identities.update({pre+s: d.fit.old.sha(pre+s) for s in suffixes})
    return entry, model, transform, rawline, identities


def checked_cpp(numeric, raw, model, expected):
    x = np.ascontiguousarray(raw[:7], np.float32)
    out = np.empty((len(x), len(model['bias'])), np.float32)
    assert numeric.lib.sf_predict(*x.shape, out.shape[1], x,
                                  np.ascontiguousarray(model['qweight']),
                                  np.ascontiguousarray(model['wscale']),
                                  np.ascontiguousarray(model['bias']), out) == 0
    assert np.array_equal(out.view(np.uint32), np.ascontiguousarray(expected[:len(x)]).view(np.uint32))
    return out.size


def score(caps, meta, i, expert, pt, pv):
    return {s: d.errors(caps[s].get('expert_outputs')[:,i], pred, d.fit.alpha_squared(caps[s],expert),meta[s][2])
            for s,pred in (('train',pt),('validation',pv))}


def objective(metrics):
    return metrics['validation']['all']['selected_alpha2_nmse']


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--layer',type=int,required=True,choices=d.FIXED)
    p.add_argument('--threads',type=int,default=8)
    a=p.parse_args()
    folder=d.DATA/f'spec2-frozen-raw-residual/layer{a.layer}'
    if folder.exists():
        raise FileExistsError(folder)
    folder.mkdir(parents=True)
    start=time.monotonic()
    numeric=Numeric(a.threads)
    result={'status':'executing','layer':a.layer,'models':[],'grid':[],'controls':[],'raw_baselines':[],
            'source_sha256':d.fit.old.sha(Path(__file__)),'frozen_numeric_binary_sha256':numeric.binary_sha256,
            'raw_inputs_sha256':{},'canonical_CPP_prediction_floats_bitchecked':0,
            'contract':'Original optimized raw canonical prediction defines TRAIN residual. B fit uses z subvector of final concat whole-vector A8 QDQ. Original raw W8 dequantized coefficients are frozen, concat head row-W8 quantized anew, separate TRAIN weighted bias calibration. Original raw payload is retained exactly.',
            'TEST':'No TEST accepted or opened.'}
    with threadpool_limits(limits=a.threads):
        caps,meta=d.load_inputs(a.layer)
        zc,zg=d.fit.mean_std(caps['train'].get('shared_z'))
        rawlines=[]
        chosen_lines={128:[],256:[]}
        fallback_lines={128:[],256:[]}
        control_lines={128:[],256:[]}
        for i,expert in enumerate(d.FIXED[a.layer]):
            entry,rawmodel,rawdef,rawline,identities=raw_entry(a.layer,expert)
            result['raw_inputs_sha256'].update(identities)
            rawlines.append(rawline)
            xr={s:d.fit.transform(c.get('x'),rawdef['center'],rawdef['inv_scale']) for s,c in caps.items()}
            re={s:d.fit.encode(x) for s,x in xr.items()}
            rawpred={s:d.fit.predict_encoded(re[s],rawmodel) for s in caps}
            rawmetrics=score(caps,meta,i,expert,rawpred['train'],rawpred['validation'])
            assert rawmetrics['validation']['all']['selected_alpha2_nmse'] == entry['valid']['selected_alpha2_nmse']
            result['raw_baselines'].append({'expert':expert,'original_beta':entry['beta'],'original_lambda':entry['lambda'],'metrics':rawmetrics})
            residual=np.subtract(caps['train'].get('expert_outputs')[:,i],rawpred['train'],dtype=np.float32)
            a2=d.fit.alpha_squared(caps['train'],expert)
            basecoeff=np.multiply(rawmodel['qweight'].astype(np.float32),rawmodel['wscale'][:,None],dtype=np.float32)
            for width in (128,256):
                ip=d.DATA/f'spec2-diagnostics/layer{a.layer}/shared_indices{width}.i32.npy'
                indices=np.load(ip)
                definition={'sources':np.concatenate((np.zeros(2048,np.uint8),np.ones(width,np.uint8))),
                            'indices':np.concatenate((np.arange(2048,dtype=np.int32),indices)),
                            'center':np.concatenate((rawdef['center'],zc[indices])),
                            'inv_scale':np.concatenate((rawdef['inv_scale'],zg[indices]))}
                tp=folder/f'transform.z{width}'
                d.fit.save_transform(tp,definition)
                inputs={s:np.concatenate((xr[s],d.fit.transform(c.get('shared_z')[:,indices],zc[indices],zg[indices])),axis=1) for s,c in caps.items()}
                enc={s:d.fit.encode(x) for s,x in inputs.items()}
                # Independent frozen-C++ gather/F32 transform check.
                rawx=np.ascontiguousarray(caps['validation'].get('x')[:7])
                rawz=np.ascontiguousarray(caps['validation'].get('shared_z')[:7])
                transformed=np.empty((7,2048+width),np.float32)
                assert numeric.lib.sf_transform(7,2048,2816,2048+width,rawx,rawz,
                        definition['sources'],definition['indices'],definition['center'],definition['inv_scale'],transformed)==0
                assert np.array_equal(transformed.view(np.uint32),inputs['validation'][:7].view(np.uint32))
                zero_coeff=np.concatenate((basecoeff,np.zeros((2048,width),np.float32)),axis=1)
                qw,sw=d.fit.old.quantize_rows(zero_coeff)
                zero={'coefficient':zero_coeff,'qweight':qw,'wscale':sw,'bias':rawmodel['bias'].copy()}
                zero_pre={s:d.fit.predict_encoded(enc[s],zero,use_bias=False) for s in caps}
                best,best_zero=(float('inf'),None),(float('inf'),None)
                for beta in d.BETAS:
                    weights=d.fit.blend_weights(a2,beta)
                    zb=((weights @ (caps['train'].get('expert_outputs')[:,i].astype(np.float64)-zero_pre['train']))/weights.sum()).astype(np.float32)
                    zmodel={**zero,'bias':zb}
                    zpred={s:np.add(v,zb,dtype=np.float32) for s,v in zero_pre.items()}
                    zm=score(caps,meta,i,expert,zpred['train'],zpred['validation'])
                    zr={'expert':expert,'shared_width':width,'beta':beta,'metrics':zm,'kind':'B0_concat_requantized_bias_recalibrated'}
                    result['controls'].append(zr)
                    if objective(zm)<best_zero[0]:
                        best_zero=(objective(zm),(zr,{k:v.copy() for k,v in zmodel.items()},zpred['validation'].copy()))
                    gram=d.fit.prepare_gram(enc['train']['qdq'][:,2048:],weights)
                    bm,rhs=d.fit.solve_components(gram,residual)
                    for ridge in (.03,.3,3.):
                        b=d.fit.solve_one(gram,bm,rhs,ridge)
                        coeff=np.concatenate((basecoeff,b['coefficient']),axis=1)
                        qweight,wscale=d.fit.old.quantize_rows(coeff)
                        model={'coefficient':coeff,'qweight':qweight,'wscale':wscale,
                               'bias':np.add(rawmodel['bias'],b['bias'],dtype=np.float32)}
                        pretrain=d.fit.predict_encoded(enc['train'],model,use_bias=False)
                        model['bias']=((weights @ (caps['train'].get('expert_outputs')[:,i].astype(np.float64)-pretrain))/weights.sum()).astype(np.float32)
                        pred={'train':np.add(pretrain,model['bias'],dtype=np.float32),
                              'validation':d.fit.predict_encoded(enc['validation'],model)}
                        metrics=score(caps,meta,i,expert,pred['train'],pred['validation'])
                        raw_nmse=objective(rawmetrics)
                        r={'expert':expert,'shared_width':width,'beta':beta,'ridge_B':ridge,'metrics':metrics,
                           'VALID_relative_selected_alpha2_NMSE_reduction_vs_original_raw':(raw_nmse-objective(metrics))/raw_nmse,
                           'VALID_relative_selected_alpha2_NMSE_reduction_vs_matched_B0':(objective(zm)-objective(metrics))/objective(zm),
                           'row_weight_N_eff':d.neff(weights),
                           'B_effective_df':float(np.sum(gram['values']/(gram['values']+ridge)))}
                        result['grid'].append(r)
                        if objective(metrics)<best[0]:
                            best=(objective(metrics),(r,{k:v.copy() for k,v in model.items()},pred['validation'].copy()))
                record,model,predv=best[1]
                prefix=folder/f'conditional_z{width}.expert{expert}'
                d.fit.save_model(prefix,model)
                result['canonical_CPP_prediction_floats_bitchecked']+=checked_cpp(numeric,inputs['validation'],model,predv)
                basis=f'spec2_frozen_raw_residual_z{width}'
                line=f'{a.layer}\t{expert}\t{basis}\t{2048+width}\t{tp}\t{prefix}'
                chosen_lines[width].append(line)
                fallback_lines[width].append(line if best[0]<objective(rawmetrics) else rawline)
                record={**record,'readout_prefix':str(prefix),'transform_prefix':str(tp),
                        'fallback_to_original_raw':best[0]>=objective(rawmetrics),
                        'payload_sha256':{str(prefix)+s:d.fit.old.sha(str(prefix)+s) for s in ('.i8','.scales.f32','.bias.f32','.npz')}}
                result['models'].append(record)
                zr,zm,zpv=best_zero[1]
                zp=folder/f'B0_z{width}.expert{expert}'
                d.fit.save_model(zp,zm)
                control_lines[width].append(f'{a.layer}\t{expert}\tspec2_B0_z{width}\t{2048+width}\t{tp}\t{zp}')
                result['canonical_CPP_prediction_floats_bitchecked']+=checked_cpp(numeric,inputs['validation'],zm,zpv)
                print(json.dumps({'event':'frozen_raw_residual_chosen','layer':a.layer,'expert':expert,'width':width,'beta':record['beta'],'ridge_B':record['ridge_B'],'VALID_gain':record['VALID_relative_selected_alpha2_NMSE_reduction_vs_original_raw']}),flush=True)
        (folder/'registry.raw_fallback.tsv').write_text('\n'.join(rawlines)+'\n')
        for width in (128,256):
            for name,lines in [('candidates',chosen_lines[width]),('with_raw_fallback',fallback_lines[width]),('B0_control',control_lines[width])]:
                (folder/f'registry.z{width}.{name}.tsv').write_text('\n'.join(lines)+'\n')
    for path,identity in result['raw_inputs_sha256'].items():
        assert d.fit.old.sha(path)==identity
    result['status']='complete'
    result['seconds']=time.monotonic()-start
    result['interpretation']='VALID selects B hyperparameters, so this is a calibration candidate, not an independent TEST result. Concat B0 may differ from original raw due to A8 context, re-W8 and bias calibration. Original fallback registry/payload never changes.'
    d.fit.old.dump(folder/'selection.json',result)
    d.fit.old.dump(d.ROOT/f'results/research-2026-11-09-shared-features/spec2_frozen_raw_residual_layer{a.layer}.json',result)
    print(json.dumps({'status':'complete','layer':a.layer,'seconds':result['seconds']}),flush=True)


if __name__=='__main__':
    main()
