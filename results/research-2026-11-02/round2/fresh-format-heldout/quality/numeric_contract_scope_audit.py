#!/usr/bin/env python3
"""Read existing frozen artifacts; no model inference or quantizer rerun."""
import collections,hashlib,json,pathlib
ROOT=pathlib.Path('/home/ericyu/expert_flash_experiment')
BASE=ROOT/'results/research-2026-11-02/round2'
FRESH=BASE/'fresh-format-heldout'
DEST=FRESH/'quality'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 protocol=json.loads((DEST/'protocol.frozen.json').read_text());proof=FRESH/'quality-build/unchanged_numeric_contract.json';contract=json.loads(proof.read_text())
 assert all(x['old_new_byte_exact'] for x in contract)
 for p in [ROOT/'scripts/research_20261103_teacher_forced_quality.cpp',ROOT/'scripts/research_20261102_full_ffn_integer_reference.cpp']:
  expected=protocol['inputs_sha256'].get(str(p))
  if expected:assert sha(p)==expected
 sources={'earlier_train':BASE/'full-ffn-integer-reference/main','fresh_teacher':FRESH/'quality/run','fresh_generation':FRESH/'generation/run'}
 ledgers={};inputhash={str(proof):sha(proof),str(DEST/'protocol.frozen.json'):sha(DEST/'protocol.frozen.json')}
 for label,folder in sources.items():
  path=folder/'weight_cache_ledger.jsonl';rows=[json.loads(l) for l in path.read_text().splitlines()];mapping={(r['weight'],r['expert']):r for r in rows};assert len(mapping)==len(rows);ledgers[label]=mapping;inputhash[str(path)]=sha(path)
 comparisons=[]
 fields=['columns','rows','source_type','source_byte_offset','canonical_bytes_fnv1a64','row_scales_fnv1a64']
 for left,right in [('earlier_train','fresh_teacher'),('earlier_train','fresh_generation'),('fresh_teacher','fresh_generation')]:
  common=sorted(ledgers[left].keys()&ledgers[right].keys());row_count=scale_bytes=0
  for key in common:
   a,b=ledgers[left][key],ledgers[right][key];assert all(a[k]==b[k] for k in fields)
   name=f'{key[0]}.expert{key[1]}.f32';aa=(sources[left]/'scales'/name).read_bytes();bb=(sources[right]/'scales'/name).read_bytes();assert aa==bb and len(aa)==a['rows']*4
   row_count+=a['rows'];scale_bytes+=len(aa)
  comparisons.append({'left':left,'right':right,'common_expert_projection_entries':len(common),'canonical_W8_and_fixed_row_scale_ledger_hashes_exact':True,'scale_rows_byte_compared':row_count,'scale_bytes_compared':scale_bytes,'all_F32_row_scale_bytes_identical':True})
 result={'status':'PASS','inference_or_quantization_rerun':False,'existing_function_body_proof_path':str(proof),'existing_function_body_proof_sha256':sha(proof),'unchanged_functions':contract,'cache_entry_counts':{k:len(v) for k,v in ledgers.items()},'fixed_cache_key':'weight_name/expert only; independent of token, activation, chip, phase-group or native job','weight_scope':'Original readonly GGUF full-row to_float, followed by one fixed full-C per-row W8 quantization before any chip assignment; cache reused thereafter.','activation_scope':'Whole full-C input vector quantized once per logical operator invocation/input vector before selected-expert and output-row loops, then identical bytes/scale reused for every chip/full-row job. Separate gate/up operators each quantize their identical full input; no global once-across-all-operators claim.','row_scales_depend_on_activation_or_layout':False,'A8_scale_depends_on_chip_or_row_job_partition':False,'source_lines':{'quantize_full_vector':'scripts/research_20261103_teacher_forced_quality.cpp:52','weight_row_to_float_and_quantize':'scripts/research_20261103_teacher_forced_quality.cpp:73','placement_after_quantization':'scripts/research_20261103_teacher_forced_quality.cpp:82','A8_once_before_expert_row_loops':'scripts/research_20261103_teacher_forced_quality.cpp:121','full_C_dot_and_one_rescale':'scripts/research_20261103_teacher_forced_quality.cpp:129'},'actual_cross_workload_cache_comparisons':comparisons,'inputs_sha256':inputhash,'audit_source_sha256':sha(__file__),'limits':['Existing frozen-artifact/source audit, not a new quality experiment or native-AiF ABI claim.','A8 is deliberately dynamic with the actual whole activation vector; numerical references can differ, while physical row partition does not enter quantization.']}
 write(DEST/'numeric_contract_scope_audit.json',result);print(json.dumps({k:v for k,v in result.items() if k not in ['unchanged_functions','inputs_sha256']},indent=2))
if __name__=='__main__':main()
