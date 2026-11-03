#!/usr/bin/env python3
"""Independent raw-source ranking/rendering/provenance check; no model use."""
from datetime import datetime
import hashlib
import json
from pathlib import Path
import pyarrow.parquet as pq

BASE=Path(__file__).resolve().parent
ROOT=Path('/home/ericyu/expert_flash_experiment')
ANCHOR_SHA='00507ff56f9197fb18511ad58303919214c03cba6f0c1440f6733ffc42944d00'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def time(s):return datetime.fromisoformat(s)
def rank(source_id,source_key):return hashlib.sha256(('20261104|'+source_key+'|'+source_id).encode('utf-8')).hexdigest()

def main():
 manifest_path=BASE/'manifest.json';m=json.loads(manifest_path.read_text())
 p=Path(m['acquisition_protocol_path']);protocol=json.loads(p.read_text())
 assert sha(p)==m['acquisition_protocol_sha256']
 assert protocol['seed']==20261104
 assert sha(protocol['source_script_path'])==protocol['source_script_sha256']
 assert time(protocol['bindings']['root_freeze_anchor']['frozen_utc'])<time(protocol['frozen_utc'])<time(m['acquisition_started_utc'])
 anchor=protocol['bindings']['root_freeze_anchor'];assert sha(anchor['path'])==ANCHOR_SHA==anchor['sha256']
 for r in protocol['bindings']['frozen_inputs']:
  f=Path(r['path']);f=f if f.is_absolute() else ROOT/f
  assert sha(f)==r['sha256']
 models=protocol['bindings']['model_checkpoint_sha256'];assert models==m['model_checkpoint_sha256']
 for r in protocol['bindings']['frozen_inputs']:
  if r['role']=='numeric_protocol':
   d=json.loads(Path(r['path']).read_text());source_models=[v for k,v in d['input_sha256'].items() if k.endswith('.gguf')]
   assert len(source_models)==1
   assert source_models[0]==models[r['models'][0]]
 state=json.loads(Path(protocol['metadata_state_path']).read_text())
 assert sha(protocol['metadata_state_path'])==protocol['metadata_state_sha256']
 for r in state['metadata_receipts']+state['official_repository_receipts']:assert sha(r['path'])==r['sha256']
 for r in state['official_repository_receipts']:
  raw=Path(r['path']).read_bytes()
  assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\x00'+raw).hexdigest()==r['blob_sha']
 items=m['items'];assert len(items)==16 and [x['kind'] for x in items]==['qa_mc']*8+['qa_bool']*8
 answers=json.loads(Path(m['answers_path']).read_text());answer_map={x['id']:x for x in answers};assert len(answer_map)==16
 assert sha(m['answers_path'])==m['answers_sha256']
 reports=[]
 for spec in protocol['source_specs']:
  receipt=next(x for x in m['source_receipts'] if x['source_key']==spec['key'])
  raw=Path(receipt['path']);assert raw.stat().st_size==spec['bytes']==receipt['bytes']
  assert sha(raw)==spec['expected_sha256']==receipt['sha256']
  assert time(receipt['retrieved_utc'])>=time(protocol['frozen_utc'])
  # Independent parser includes answer column, but ranks only structural fields.
  rows=pq.read_table(raw).to_pylist();assert len(rows)==spec['rows']
  ids=[r['id'] for r in rows] if spec['kind']=='qa_mc' else [str(i) for i in range(len(rows))]
  assert len(ids)==len(set(ids))
  eligible=[]
  for ix,row in enumerate(rows):
   if spec['kind']=='qa_mc':
    question=row['question_stem'];labels=row['choices']['label'];choices=row['choices']['text']
    if not isinstance(question,str) or not question.strip() or len(labels)!=4 or len(choices)!=4:continue
    if not all(isinstance(x,str) and x.strip() for x in labels+choices):continue
    if len({x.strip() for x in labels})!=4:continue
    rendered=question.strip()+'\n'+''.join(label.strip()+'. '+choice.strip()+'\n' for label,choice in zip(labels,choices))
   else:
    passage,question=row['passage'],row['question']
    if not all(isinstance(x,str) and x.strip() for x in (passage,question)):continue
    rendered='Passage:\n'+passage.strip()+'\n\nQuestion:\n'+question.strip()+'\n'
   eligible.append((rank(ids[ix],spec['key']),ids[ix],ix,rendered))
  chosen=sorted(eligible)[:8]
  target=[x for x in items if x['dataset']==spec['repo']];assert len(target)==8
  for pos,(ranking,sid,ix,rendered) in enumerate(chosen):
   item=target[pos]
   assert item['dataset_revision']==spec['revision'] and item['dataset_config']==spec['config'] and item['split']==spec['split']
   assert item['source_id']==sid and item['source_row_index']==ix and item['selection_rank']==pos and item['ranking_sha256']==ranking
   prompt=Path(item['prompt_path']).read_bytes();assert prompt==rendered.encode('utf-8')
   assert sha(item['prompt_path'])==item['prompt_sha256']
   assert len(prompt)==item['prompt_utf8_bytes'] and len(rendered.split())==item['prompt_whitespace_words']
   answer=answer_map[item['id']];row=rows[ix]
   if spec['kind']=='qa_mc':
    assert answer['answer_label']==row['answerKey']
    k=row['choices']['label'].index(row['answerKey'])
    assert answer['answer_index']==k and answer['answer_text']==row['choices']['text'][k]
   else:
    assert answer['answer_bool']==row['answer'] and answer['answer_verbalization']==('yes' if row['answer'] else 'no')
  reports.append(dict(dataset=spec['repo'],revision=spec['revision'],config=spec['config'],split=spec['split'],
   source_rows=len(rows),eligible_rows=len(eligible),selected_rows=8,raw_sha256=sha(raw),
   selected_source_ids=[x[1] for x in chosen],prompt_word_range=[min(len(x[3].split()) for x in chosen),max(len(x[3].split()) for x in chosen)]))
 assert len({x['prompt_sha256'] for x in items})==16
 selected_records=json.loads(Path(m['selected_source_records_path']).read_text());assert len(selected_records)==16
 assert sha(m['selected_source_records_path'])==m['selected_source_records_sha256']
 for row in selected_records:assert 'answer' not in row['source_fields_without_answers'] and 'answerKey' not in row['source_fields_without_answers']
 result=dict(status='PASS independent acquisition, ranking, full rendering, answer separation and freeze/provenance check',
  source_sha256=sha(__file__),manifest_path=str(manifest_path),manifest_sha256=sha(manifest_path),
  acquisition_protocol_sha256=sha(p),root_anchor_sha256=ANCHOR_SHA,models=models,items=16,datasets=reports,
  root_anchor_before_acquisition_protocol_before_target=True,answers_appended_to_prompts=False,
  selected_records_exclude_answer_fields=True,no_model_or_placement_execution=True,
  payload_bytes=sum(x['bytes'] for x in m['source_receipts']),
  limits=['Fresh within-project sources, not model-pretraining decontamination.','Code repository license is not substituted for unknown OpenBookQAdata license.','No benchmark accuracy or model quality claim from16free-generation tokens.'])
 out=BASE/'independent_acquisition_check.json';assert not out.exists();out.write_text(json.dumps(result,indent=2)+'\n')
 print(json.dumps({k:v for k,v in result.items() if k!='datasets'},indent=2))

if __name__=='__main__':main()
