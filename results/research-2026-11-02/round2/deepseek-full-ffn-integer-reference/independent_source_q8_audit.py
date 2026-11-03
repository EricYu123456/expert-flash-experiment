#!/usr/bin/env python3
"""Read-only GGUF offsets/Q8_0 row-scale audit; no inference/quantizer fitting."""
import hashlib,json,pathlib
import numpy as np
OUT=pathlib.Path(__file__).resolve().parent
MODEL=pathlib.Path('/home/nfs_client/ericyu/expert_flash_experiment/models/deepseek-v2-lite-chat-q8_0.gguf')
META=MODEL.with_name('deepseek-metadata.json')
SOURCE=pathlib.Path('/home/nfs_client/ericyu/expert_flash_experiment/sources/llama.cpp-927dada6c')
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def main():
 meta=json.loads(META.read_text());tensors={t['name']:t for t in meta['tensors']};rows=[json.loads(l) for l in (OUT/'smoke/weight_cache_ledger.jsonl').read_text().splitlines()]
 grouped={}
 for row in rows:
  t=tensors[row['weight']];C,R=t['shape_ggml'][:2];assert t['ggml_type'].upper()==row['source_type'].upper()=='Q8_0' and row['columns']==C and row['rows']==R and C%32==0
  expert=row['expert'];expected=t['data_offset']+expert*R*(C//32)*34;assert row['source_byte_offset']==expected
  assert row['chip_rows']==[R//16]*16
  grouped.setdefault(row['weight'],[]).append(row)
 assert len(grouped)==159
 dtype=np.dtype([('d','<f2'),('qs','i1',(32,))]);assert dtype.itemsize==34
 weight=np.memmap(MODEL,mode='r',dtype='u1');checks=[]
 for name,entries in sorted(grouped.items()):
  chosen={min(r['expert'] for r in entries),max(r['expert'] for r in entries)}
  for expert in sorted(chosen):
   c=next(r for r in entries if r['expert']==expert);C,R=c['columns'],c['rows'];path=OUT/'smoke/scales'/f'{name}.expert{expert}.f32';scales=np.fromfile(path,dtype='<f4');assert len(scales)==R
   for r in [0,R//2,R-1]:
    at=c['source_byte_offset']+r*(C//32)*34;blocks=np.frombuffer(weight[at:at+(C//32)*34],dtype=dtype)
    # This independently follows ggml block_q8_0 and dequantize_row_q8_0:
    # FP16 delta converted toF32, then each signed byte multiplied inF32.
    original=(blocks['qs'].astype(np.float32)*blocks['d'].astype(np.float32)[:,None]).reshape(C)
    maximum=np.max(np.abs(original));expected=np.float32(1) if maximum==0 else np.float32(maximum/np.float32(127))
    a=np.asarray(expected,dtype=np.float32).view(np.uint32).item();b=scales[r:r+1].view(np.uint32).item();assert a==b,(name,expert,r,a,b)
    checks.append({'weight':name,'expert':expert,'row':r,'columns':C,'expected_scale_f32_bits':a,'stored_scale_f32_bits':b})
 result={'status':'PASS','cache_entries_all_source_offsets_geometry_and_Q8_types_checked':len(rows),'target_weight_names_checked':len(grouped),'deterministic_rows_per_cached_expert_boundary':3,'independent_GGUF_dequant_row_scale_samples':len(checks),'row_scale_bit_mismatches':0,'sample_selection':'For every logical target weight, minimum and maximum cached expert IDs (one for dense/shared); rows0,R//2,R-1.','read_only_no_model_inference_or_weight_requantization':True,'original_model_sha256_expected':meta['sha256'],'source_layout_evidence':{'block_q8_0':str(SOURCE/'ggml/src/ggml-common.h')+':241','dequantize_row_q8_0':str(SOURCE/'ggml/src/ggml-quants.c')+':491'},'input_sha256':{str(META):sha(META),str(OUT/'protocol.frozen.json'):sha(OUT/'protocol.frozen.json'),str(OUT/'smoke/weight_cache_ledger.jsonl'):sha(OUT/'smoke/weight_cache_ledger.jsonl'),str(SOURCE/'ggml/src/ggml-common.h'):sha(SOURCE/'ggml/src/ggml-common.h'),str(SOURCE/'ggml/src/ggml-quants.c'):sha(SOURCE/'ggml/src/ggml-quants.c')},'audit_source_sha256':sha(__file__)}
 (OUT/'independent_source_q8_audit.json').write_text(json.dumps(result,indent=2)+'\n');(OUT/'independent_source_q8_scale_samples.json').write_text(json.dumps(checks,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
