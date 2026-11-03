#!/usr/bin/env python3
"""Integrity audit on fixed prospective IDs; no reselection or cost access."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,hashlib,json
import pyarrow.parquet as pq

ROOT=Path(__file__).resolve().parents[1]
ANCHOR=ROOT/'results/research-2026-11-02/round2/chip-count-regime-screen/m32-prospective-tail-validation'
DATA=ANCHOR/'data-capture'
OUT=DATA/'independent-source-check'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def hs(s):return hashlib.sha256(s.encode()).hexdigest()
def write(name,x):
 p=OUT/name;assert not p.exists();p.write_text(json.dumps(x,indent=2)+'\n')
def dt(s):return datetime.fromisoformat(s.replace('Z','+00:00'))

def freeze():
 OUT.mkdir(exist_ok=False)
 paths=[Path(__file__),DATA/'acquisition_capture_protocol.frozen.json',DATA/'manifest.json',DATA/'selected_IDs.pre_cap.json',
        DATA/'exclusion_catalog.json',DATA/'tokenizer.receipt.tsv',ANCHOR/'protocol.frozen.json',ANCHOR/'completion.frozen.json']
 paths+=list((DATA/'raw').iterdir())
 paths+=list((DATA/'granite4tiny').glob('batch*/run_command.json'))
 write('protocol.frozen.json',{'scope':'Fixed32sample integrity only; no labels, newroutes, costs, filtering or changing IDs. Captured values/types/hashes are checked, not selected.',
                              'inputs_sha256':{str(p):sha(p) for p in paths},'frozen_utc':datetime.now(timezone.utc).isoformat()})

def run():
 for path,h in json.loads((OUT/'protocol.frozen.json').read_text())['inputs_sha256'].items():assert sha(path)==h,path
 protocol=json.loads((DATA/'acquisition_capture_protocol.frozen.json').read_text());m=json.loads((DATA/'manifest.json').read_text())
 anchor=json.loads((ANCHOR/'protocol.frozen.json').read_text());assert dt(anchor['utc'])<dt(protocol['frozen_utc'])<dt(m['acquisition_started_utc'])
 assert m['protocol_sha256']==sha(DATA/'acquisition_capture_protocol.frozen.json')
 assert protocol['anchor_protocol_sha256']==sha(ANCHOR/'protocol.frozen.json')
 excluded=json.loads((DATA/'exclusion_catalog.json').read_text())['records'];old={x['prompt_sha256'] for x in excluded}
 oldIDs={x['id'] for x in excluded};newhash=set();expected=[];records=[]
 selected=m['items'];assert len(selected)==32 and len({x['id'] for x in selected})==32
 assert json.loads((DATA/'selected_IDs.pre_cap.json').read_text())==[{k:v for k,v in j.items() if k not in ['prompt_sha256','prompt_bytes','prompt_tokens_including_BOS','untruncated_prompt_tokens','removed_codepoints','task','length_bucket']}|{'split':'validation'} for j in selected]
 for spec in protocol['sources']:
  payload=next(Path(j['source_payload_path']) for j in selected if j['source_key']==spec['key'])
  assert payload.stat().st_size==spec['bytes']
  if spec['format']=='jsonl':
   raw=payload.read_bytes();lines=raw.decode().splitlines();rows=[json.loads(x) for x in lines]
   assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==spec['expected_git_blob_sha1']
  else:
   assert sha(payload)==spec['expected_sha256'];rows=pq.read_table(payload,columns=spec['fields']).to_pylist()
  order=sorted(range(len(rows)),key=lambda i:(hs('\0'.join([protocol['seed'],spec['key'],spec['revision'],spec['split'],str(i)])),str(i)))
  found=[]
  for i in order:
   sid=str(i)
   if any(x.get('dataset')==spec['dataset'] and x.get('split')==spec['split'] and x.get('source_id')==sid for x in excluded):continue
   # Match the frozen normalization for ranking/dedup; only the selected32
   # receive the stricter type audit below. There is no new eligibility filter.
   values=[' '.join(str(rows[i][f]).split()) for f in spec['fields']]
   full=values[0]+'\n1. '+values[1]+'\n2. '+values[2]+'\n'
   if hs(full) in old|newhash:continue
   found.append(i);newhash.add(hs(full))
   if len(found)==16:break
  items=[j for j in selected if j['source_key']==spec['key']]
  assert len(items)==16 and [j['source_row_index'] for j in items]==found
  for j in items:
   i=j['source_row_index'];row=rows[i]
   assert all(isinstance(row[f],str) and row[f].strip() for f in spec['fields'])
   fields={f:' '.join(row[f].split()) for f in spec['fields']};v=list(fields.values())
   full=v[0]+'\n1. '+v[1]+'\n2. '+v[2]+'\n';actual=Path(j['prompt_path']).read_bytes()
   assert j['sample_noanswer_fields_sha256']==hs(json.dumps(fields,ensure_ascii=False,sort_keys=True,separators=(',',':')))
   assert j['full_rendered_prompt_sha256']==hs(full)
   assert sha(payload)==j['source_payload_sha256'] and j['dataset_split']=='validation'
   assert j['removed_codepoints']==0 and actual==full.encode()
   assert sha(j['prompt_path'])==j['prompt_sha256'] and j['prompt_sha256'] not in old and j['id'] not in oldIDs
   assert 0<j['prompt_tokens_including_BOS']==j['untruncated_prompt_tokens']<=256
   if spec['format']=='jsonl':assert hs(lines[i])==j['source_JSONL_line_sha256']
   records.append({'id':j['id'],'source_id':j['source_id'],'three_original_fields_all_nonempty_strings':True,
                   'noanswer_source_fields_sample_SHA_exact':True,'rendered_prompt_exact':True,'removed_codepoints':0,
                   'prompt_sha256':j['prompt_sha256'],'source_payload_sha256':j['source_payload_sha256']})
 for r in m['source_receipts']:assert dt(m['acquisition_started_utc'])<=dt(r['retrieved_utc'])<=dt(m['created_utc'])
 for f in (DATA/'granite4tiny').glob('batch*/run_command.json'):
  r=json.loads(f.read_text());assert dt(m['created_utc'])<dt(r['started_utc']) and r['manifest_sha256']==sha(DATA/'manifest.json')
 assert len(records)==32 and len({r['prompt_sha256'] for r in records})==32
 write('report.json',{'status':'PASS','protocol_sha256':sha(OUT/'protocol.frozen.json'),'main_protocol_sha256':sha(DATA/'acquisition_capture_protocol.frozen.json'),
                      'manifest_sha256':sha(DATA/'manifest.json'),'selected32_raw_fields_all_nonempty_strings':True,
                      'all_deterministic_selectedIDs_recomputed':True,'all_prompt_renderings_hashes_exact_no_truncation':True,
                      'old332_exactSHA_ID_exclusion_checked':True,'strict_planfreeze_then_datafreeze_then_payload_then_capture_UTC_order':True,
                      'all_selectedIDs_unchanged_no_filter_or_replacement':True,'labels_loaded':False,'newroutes_or_costs_accessed':False,'records':records})
 print(json.dumps({'status':'PASS','selected':32,'report_sha256':sha(OUT/'report.json')}))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['freeze','run']);a=ap.parse_args()
 freeze() if a.mode=='freeze' else run()
