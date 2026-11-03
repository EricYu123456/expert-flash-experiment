#!/usr/bin/env python3
"""Prepare metadata now; freeze after plan hashes; acquire only after freeze.

No model, target dataset payload, or Parquet footer is opened by `prepare` or
`freeze`. Answers never participate in eligibility, ranking, or prompt rendering.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

ROOT=Path(__file__).resolve().parents[1]
DEST=ROOT/'results/research-2026-11-02/round2/second-format-heldout'
SEED=20261104
SOURCES=[
 dict(key='openbookqa_main_test',repo='allenai/openbookqa',revision='388097ea7776314e93a529163e0fea805b8a6454',
      config='main',split='test',kind='qa_mc',file='main/test-00000-of-00001.parquet',rows=500,bytes=55535,
      expected_sha256='cd5483e366daa230c1c87bbdc512d8b7229f14f6dd04d19fc8b1a3855aaaa8a3',
      metadata_license=['unknown'],
      license_note='Pinned HF card marks data license unknown. Official model-code repository has Apache2.0; it is not asserted to license this externally hosted dataset.',
      publisher_repo='https://github.com/allenai/OpenBookQA',publisher_commit='b51971646e9371a61508d9953fc706645e194a71'),
 dict(key='boolq_validation',repo='google/boolq',revision='35b264d03638db9f4ce671b711558bf7ff0f80d5',
      config='default',split='validation',kind='qa_bool',file='data/validation-00000-of-00001.parquet',rows=3270,bytes=1257630,
      expected_sha256='52355d11524b4b874a9b9dcc278feb10f672d52c4f4eff9872e695ede59820f8',
      metadata_license=['cc-by-sa-3.0'],
      license_note='Official Google dataset README and pinned HF card both state CC-BY-SA3.0.',
      publisher_repo='https://github.com/google-research-datasets/boolean-questions',publisher_commit='90af34107399cc7a446b373dc4ee35b8001da7c2')]

def now():return datetime.now(timezone.utc).isoformat()
def sha(path):
 h=hashlib.sha256()
 with Path(path).open('rb') as f:
  for block in iter(lambda:f.read(8*1024*1024),b''):h.update(block)
 return h.hexdigest()
def textsha(s):return hashlib.sha256(s.encode('utf-8')).hexdigest()
def write_new(path,data):
 path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
 with path.open('xb') as f:f.write(data)
def json_new(path,data):write_new(path,(json.dumps(data,ensure_ascii=False,indent=2)+'\n').encode())
def get(url,limit,allow_payload=False):
 if not allow_payload:
  assert not re.search(r'\.(parquet|jsonl|zip)(?:[?]|$)',url), 'Metadata stage cannot open target payloads.'
 req=urllib.request.Request(url,headers={'User-Agent':'expert-flash-heldout-metadata/1.0'})
 with urllib.request.urlopen(req,timeout=60) as r:
  raw=r.read(limit+1);assert len(raw)<=limit
  return raw,dict(url=url,bytes=len(raw),sha256=hashlib.sha256(raw).hexdigest(),retrieved_utc=now(),
                 etag=r.headers.get('ETag'),final_url_without_query=urllib.parse.urlunsplit(urllib.parse.urlsplit(r.url)._replace(query='',fragment='')))
def rank(key,source_id):return hashlib.sha256(f'{SEED}|{key}|{source_id}'.encode()).hexdigest()

def prepare():
 assert not (DEST/'metadata_state.json').exists()
 assert not (DEST/'acquisition_protocol.frozen.json').exists()
 receipts=[];schemas=[]
 for spec in SOURCES:
  repo,rev=spec['repo'],spec['revision'];parent=spec['file'].rsplit('/',1)[0]
  urls=[('metadata',f'https://huggingface.co/api/datasets/{repo}/revision/{rev}'),
        ('tree',f'https://huggingface.co/api/datasets/{repo}/tree/{rev}/{parent}'),
        ('README',f'https://huggingface.co/datasets/{repo}/resolve/{rev}/README.md')]
  for role,url in urls:
   raw,receipt=get(url,2*1024*1024)
   path=DEST/'metadata'/(spec['key']+'.'+('README.md' if role=='README' else role+'.json'))
   if role=='metadata':
    d=json.loads(raw);assert d['sha']==rev and d['cardData']['license']==spec['metadata_license']
    info=d['cardData']['dataset_info']
    if isinstance(info,list):info=next(x for x in info if x['config_name']==spec['config'])
    split=next(x for x in info['splits'] if x['name']==spec['split']);assert split['num_examples']==spec['rows']
    schemas.append(dict(source_key=spec['key'],features=info['features'],target_split_metadata=split))
   elif role=='tree':
    d=json.loads(raw);target=next(x for x in d if x['path']==spec['file'])
    assert target['size']==spec['bytes'] and target['lfs']['oid']==spec['expected_sha256']
   write_new(path,raw);receipts.append(dict(source=spec['key'],role=role,path=str(path),**receipt))
 official_path=DEST/'metadata/official_repository_receipts.json'
 official=json.loads(official_path.read_text())
 for record in official:
  record['sha256']=sha(record['path'])
 state=dict(schema='expert_flash.second_heldout_metadata.v1',prepared_utc=now(),source_script_sha256=sha(__file__),
            source_specs=SOURCES,schemas=schemas,metadata_receipts=receipts,official_repository_receipts=official,
            official_receipts_path=str(official_path),official_receipts_sha256=sha(official_path),
            target_payload_downloaded=False,plan_freeze_pending=True,
            metadata_exposure='Official repository README schema examples were viewed; their membership in a target split was not tested. AI2homepage redirected to HFlanding, which may carry automatic previews. No target Parquet/JSONL/footer/viewer-rows endpoint was fetched, and no example is used for ranking or eligibility.')
 json_new(DEST/'metadata_state.json',state)
 draft=dict(schema='expert_flash.second_heldout_protocol.draft.v1',state='DRAFT only; waiting for root frozen W8 train-compiled plan hashes',
            seed=SEED,source_specs=SOURCES,selection=selection_contract(),
            required_binding_schema=dict(models=['granite4tiny','deepseek'],
              model_checkpoint_sha256={'granite4tiny':'64hex','deepseek':'64hex'},
              frozen_inputs=[dict(path='absolute or workspace-relative',sha256='64hex',role='train_compiled_plans',models=['granite4tiny','deepseek'])]),
            no_target_payload_downloaded=True,metadata_state_sha256=sha(DEST/'metadata_state.json'))
 json_new(DEST/'acquisition_protocol.draft.json',draft)
 print(json.dumps(dict(status='metadata prepared; target payload unopened; protocol not frozen',
                      metadata_state_path=str(DEST/'metadata_state.json'),metadata_state_sha256=sha(DEST/'metadata_state.json'),
                      seed=SEED,expected_target_payload_bytes=sum(s['bytes'] for s in SOURCES)),indent=2))

def selection_contract():
 return dict(count_per_dataset=8,
  ranking='Ascending SHA256(UTF8(decimal_seed + verticalbar + source_key + verticalbar + source_id)); tie break source_id. Seed20261104. No tokenizer/model/answer-based selection.',
  openbookqa_source_id='Publisher id string; duplicates or empty ids fail integrity checks.',
  boolq_source_id='Canonical decimal zero-based pinned validation Parquet row index; HFschema has no publisher id field. It is a source-row id, not a claimed original Google id.',
  openbookqa_eligibility='Nonempty string question_stem; exactly4nonempty string choices with4distinct nonempty labels. No length truncation or answerKey inspection.',
  boolq_eligibility='Nonempty string passage and question. Full text preserved; no length, difficulty, answer or model-based filter.',
  openbookqa_render='question_stem.strip()+LF+source-order label.strip()+dot+space+choice_text.strip(), LF-separated, final LF. No facts, answer or answer cue added.',
  boolq_render='Literal Passage: +LF+passage.strip()+LF+LF+literal Question: +LF+question.strip()+LF. No answer or answer cue added.',
  item_order='OpenBookQA ranks00..07, then BoolQ ranks00..07, same16prompt bytes supplied to both G4andDeep.',
  failures='Fewer than8eligible, duplicate ids, duplicate selected prompts or invalid selected answers fail; do not relax/resample.',
  answers='Read answer columns only after chosen row indices are fixed. Store labels/booleans in answers/answers.json; never append them to prompts.')

def check_metadata(state):
 assert state['source_specs']==SOURCES and state['source_script_sha256']==sha(__file__)
 for r in state['metadata_receipts']+state['official_repository_receipts']:assert sha(r['path'])==r['sha256']
 assert sha(state['official_receipts_path'])==state['official_receipts_sha256']

def checked_bindings(path):
 d=json.loads(Path(path).read_text());assert set(d['models'])=={'granite4tiny','deepseek'}
 for model in d['models']:assert re.fullmatch('[0-9a-f]{64}',d['model_checkpoint_sha256'][model])
 covered=set()
 for r in d['frozen_inputs']:
  p=Path(r['path']);p=p if p.is_absolute() else ROOT/p
  assert re.fullmatch('[0-9a-f]{64}',r['sha256']) and sha(p)==r['sha256']
  if r['role']=='train_compiled_plans':covered.update(r['models'])
 assert covered>={'granite4tiny','deepseek'},'Both new train-compiled model plans must be bound before target acquisition.'
 return d

def freeze(bindings):
 assert bindings is not None, 'Root plan-freeze bindings are required; do not freeze the draft early.'
 assert not (DEST/'acquisition_protocol.frozen.json').exists()
 assert not (DEST/'raw').exists() and not (DEST/'prompts').exists()
 state=json.loads((DEST/'metadata_state.json').read_text());check_metadata(state)
 bound=checked_bindings(bindings)
 protocol=dict(schema='expert_flash.second_heldout_protocol.v1',frozen_utc=now(),seed=SEED,
  source_script_path=str(Path(__file__).resolve()),source_script_sha256=sha(__file__),
  metadata_state_path=str(DEST/'metadata_state.json'),metadata_state_sha256=sha(DEST/'metadata_state.json'),
  source_specs=SOURCES,selection=selection_contract(),
  bindings_path=str(Path(bindings).resolve()),bindings_sha256=sha(bindings),bindings=bound,
  workload_contract=dict(models=['granite4tiny','deepseek'],prompts=16,decode_tokens=16,
   input_policy='Complete question/options and complete BoolQpassage/question; no128token truncation. Record actual model-tokenized lengths. No chat template in stored prompt bytes.',
   placement='Frozen W8train-compiled plans unchanged; no placement/weight/scale calibration fit after target payload.',
   quality='A16token free generation is not a benchmark accuracy estimate. Runner freezes and records any teacher-forcing/scoring contract separately before inference.',
   provenance='Dataset commits are not checkpoint commits. Fresh within-project source formats do not establish model-pretraining decontamination or a representative full benchmark.'),
  preprotocol_exposure=state['metadata_exposure'],target_payload_opened_before_freeze=False,
  acquisition_limits=dict(payload_files=2,total_expected_payload_bytes=sum(s['bytes'] for s in SOURCES),
                          max_each_payload_bytes=2*1024*1024,no_new_model=True,no_other_split=True))
 path=DEST/'acquisition_protocol.frozen.json';json_new(path,protocol)
 print(json.dumps(dict(status='frozen before target payload',path=str(path),sha256=sha(path)),indent=2))

def rows_without_answers(local,spec):
 import pyarrow.parquet as pq
 cols=['id','question_stem','choices'] if spec['kind']=='qa_mc' else ['question','passage']
 rows=pq.read_table(local,columns=cols).to_pylist();assert len(rows)==spec['rows']
 if spec['kind']=='qa_mc':
  ids=[r['id'] for r in rows];assert all(isinstance(x,str) and x for x in ids) and len(ids)==len(set(ids))
 eligible=[]
 for index,row in enumerate(rows):
  if spec['kind']=='qa_mc':
   q=row['question_stem'];labels=row['choices']['label'];texts=row['choices']['text']
   if not isinstance(q,str) or not q.strip() or len(labels)!=4 or len(texts)!=4:continue
   if not all(isinstance(x,str) and x.strip() for x in labels+texts):continue
   if len(set(x.strip() for x in labels))!=4:continue
   prompt=q.strip()+'\n'+'\n'.join(l.strip()+'. '+t.strip() for l,t in zip(labels,texts))+'\n';source_id=row['id']
  else:
   passage,q=row['passage'],row['question']
   if not all(isinstance(x,str) and x.strip() for x in (passage,q)):continue
   prompt='Passage:\n'+passage.strip()+'\n\nQuestion:\n'+q.strip()+'\n';source_id=str(index)
  eligible.append(dict(source_id=source_id,row_index=index,prompt=prompt,record=row,ranking_sha256=rank(spec['key'],source_id)))
 return rows,eligible

def acquire():
 import pyarrow.parquet as pq
 protocol_path=DEST/'acquisition_protocol.frozen.json';protocol=json.loads(protocol_path.read_text());protocol_sha=sha(protocol_path)
 assert protocol['source_specs']==SOURCES and protocol['source_script_sha256']==sha(__file__)
 assert protocol['selection']==selection_contract() and protocol['seed']==SEED
 assert sha(protocol['bindings_path'])==protocol['bindings_sha256'];checked_bindings(protocol['bindings_path'])
 state=json.loads(Path(protocol['metadata_state_path']).read_text());check_metadata(state)
 assert sha(protocol['metadata_state_path'])==protocol['metadata_state_sha256']
 assert not (DEST/'manifest.json').exists()
 started=now();receipts=[];items=[];records=[];answers=[];reports=[]
 for spec in SOURCES:
  url=f'https://huggingface.co/datasets/{spec["repo"]}/resolve/{spec["revision"]}/{spec["file"]}?download=true'
  raw,receipt=get(url,protocol['acquisition_limits']['max_each_payload_bytes'],allow_payload=True)
  assert receipt['bytes']==spec['bytes'] and receipt['sha256']==spec['expected_sha256']
  local=DEST/'raw'/(spec['key']+'.parquet');write_new(local,raw)
  receipts.append(dict(source_key=spec['key'],path=str(local),**receipt))
  rows,eligible=rows_without_answers(local,spec);assert len(eligible)>=8
  chosen=sorted(eligible,key=lambda x:(x['ranking_sha256'],x['source_id']))[:8]
  # Labels are first materialized only after selected row indices are fixed.
  col='answerKey' if spec['kind']=='qa_mc' else 'answer'
  labels=pq.read_table(local,columns=[col]).column(col).to_pylist()
  reports.append(dict(source_key=spec['key'],total_rows=len(rows),eligible_rows=len(eligible),selected_source_ids=[x['source_id'] for x in chosen],
                      selected_row_indices=[x['row_index'] for x in chosen],payload_sha256=sha(local)))
  for index,row in enumerate(chosen):
   key='openbookqa' if spec['kind']=='qa_mc' else 'boolq';item_id=f'second20261104_{key}_{index:02d}'
   prompt=row['prompt'];p=DEST/'prompts'/(item_id+'.txt');write_new(p,prompt.encode())
   records.append(dict(id=item_id,source_key=spec['key'],source_id=row['source_id'],row_index=row['row_index'],source_fields_without_answers=row['record']))
   answer=labels[row['row_index']]
   if spec['kind']=='qa_mc':
    source_labels=row['record']['choices']['label'];assert answer in source_labels
    answer_doc=dict(answer_label=answer,answer_index=source_labels.index(answer),answer_text=row['record']['choices']['text'][source_labels.index(answer)])
   else:
    assert isinstance(answer,bool);answer_doc=dict(answer_bool=answer,answer_verbalization='yes' if answer else 'no')
   answers.append(dict(id=item_id,dataset=spec['repo'],source_id=row['source_id'],**answer_doc))
   items.append(dict(id=item_id,dataset=spec['repo'],dataset_revision=spec['revision'],dataset_config=spec['config'],split=spec['split'],kind=spec['kind'],
    source_id=row['source_id'],source_id_origin='publisher_id' if spec['kind']=='qa_mc' else 'pinned_parquet_row_index',source_row_index=row['row_index'],
    selection_rank=index,ranking_sha256=row['ranking_sha256'],prompt_path=str(p),prompt_sha256=sha(p),
    prompt_utf8_bytes=len(prompt.encode()),prompt_unicode_characters=len(prompt),prompt_whitespace_words=len(prompt.split()),
    source_payload_path=str(local),source_payload_sha256=sha(local),license_as_publisher=spec['metadata_license'],answers_sidecar_key=item_id))
 assert len(items)==16 and len({i['prompt_sha256'] for i in items})==16
 rp=DEST/'selected_source_records.json';ap=DEST/'answers/answers.json';json_new(rp,records);json_new(ap,answers)
 checked_bindings(protocol['bindings_path']);assert sha(protocol_path)==protocol_sha
 manifest=dict(schema='expert_flash.second_heldout_manifest.v1',created_utc=now(),acquisition_started_utc=started,
  acquisition_protocol_path=str(protocol_path),acquisition_protocol_sha256=protocol_sha,items=items,source_reports=reports,source_receipts=receipts,
  selected_source_records_path=str(rp),selected_source_records_sha256=sha(rp),answers_path=str(ap),answers_sha256=sha(ap),
  bindings=protocol['bindings'],model_checkpoint_sha256=protocol['bindings']['model_checkpoint_sha256'],
  same16prompt_bytes_for_both_models=True,answers_appended_to_prompts=False,placement_fitting_performed=False,
  no_model_download=True,no_benchmark_accuracy_claim=True,
  provenance_note='Only two fixed publisher-heldout splits; fresh project source formats, not guaranteed absent from pretraining. README examples/metadata exposure recorded in frozen protocol.')
 path=DEST/'manifest.json';json_new(path,manifest)
 print(json.dumps(dict(status='acquired frozen16prompts',path=str(path),sha256=sha(path),items=16,payload_bytes=sum(x['bytes'] for x in receipts)),indent=2))

def main():
 ap=argparse.ArgumentParser();ap.add_argument('stage',choices=['prepare','freeze','acquire']);ap.add_argument('--bindings',type=Path)
 args=ap.parse_args()
 if args.stage=='prepare':prepare()
 elif args.stage=='freeze':freeze(args.bindings)
 else:acquire()

if __name__=='__main__':main()
