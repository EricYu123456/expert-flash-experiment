#!/usr/bin/env python3
"""Prospective sources/canonical W8 capture. Frozen plans precede new payload."""
from pathlib import Path
from datetime import datetime,timezone
import argparse,collections,hashlib,json,os,signal,subprocess,time,urllib.request
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
BASE=ROOT/'results/research-2026-11-02/round2'
ANCHOR=BASE/'chip-count-regime-screen/m32-prospective-tail-validation'
OUT=ANCHOR/'data-capture'
TRAIN=BASE/'w8-format-training'
NFS=Path('/home/nfs_client/ericyu/expert_flash_experiment')
MODEL=NFS/'models/granite4tiny-q4_k_m.gguf'
BINARY=TRAIN/'build/granite4tiny-canonical-capture'
COLLECTOR=ROOT/'scripts/research_20261103_w8_training_capture_granite4tiny.cpp'
TOKENIZER=ROOT/'scripts/research_20261103_prospective_prompt_tokenizer.cpp'
PLAN=BASE/'full-ffn-integer-reference/placement.tsv'
SEED='M32-prospective-W8-v1-20261115'
SOURCES=[
 dict(key='piqa_validation',dataset='ybisk/piqa',publisher='ybisk/ybisk.github.io',revision='16c6be5562fdf2e56a6d5ee4f39a047494eb0617',split='validation',format='jsonl',
      file='piqa/data/valid.jsonl',bytes=495929,expected_git_blob_sha1='da35972044f4a35e0053ff09a97ed7926ed3b8b4',
      payload_url='https://raw.githubusercontent.com/ybisk/ybisk.github.io/16c6be5562fdf2e56a6d5ee4f39a047494eb0617/piqa/data/valid.jsonl',
      fields=['goal','sol1','sol2'],metadata_license='HF card unknown; no additional data-license claim'),
 dict(key='winogrande_validation',dataset='allenai/winogrande',publisher='allenai/winogrande',revision='01e74176c63542e6b0bcb004dcdea22d94fb67b5',split='validation',config='winogrande_xl',format='parquet',
      file='winogrande_xl/validation-00000-of-00001.parquet',bytes=85928,expected_sha256='f9d914d1818c0aba98cabdf8af2bc4bd943c462b9c5e7062647e05c2ce3315d1',
      payload_url='https://huggingface.co/datasets/allenai/winogrande/resolve/01e74176c63542e6b0bcb004dcdea22d94fb67b5/winogrande_xl/validation-00000-of-00001.parquet?download=true',
      fields=['sentence','option1','option2'],metadata_license='Not specified in fetched HF cardData; no inferred data-license claim')]

def now():return datetime.now(timezone.utc).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def htext(s):return hashlib.sha256(s.encode()).hexdigest()
def new(p,data):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True)
 with p.open('xb') as f:f.write(data)
def save(p,x):new(p,(json.dumps(x,ensure_ascii=False,indent=2)+'\n').encode())
def fetch(url,limit):
 with urllib.request.urlopen(urllib.request.Request(url,headers={'User-Agent':'expert-flash-prospective/1.0'}),timeout=60) as r:
  b=r.read(limit+1);assert len(b)<=limit
  return b,dict(url=url,bytes=len(b),sha256=hashlib.sha256(b).hexdigest(),retrieved_utc=now(),etag=r.headers.get('ETag'))
def check_anchor():
 assert sha(ANCHOR/'protocol.frozen.json')=='911ecbdb77dcd18ccc880870f70a6a231c0a67730bfac7ac9144eea061e9b516'
 assert sha(ANCHOR/'completion.frozen.json')=='36eb40d994a2163a9d066bd73777510f1b6bff1ede05bf894f686f551c7ef895'
 for p in json.loads((ANCHOR/'protocol.frozen.json').read_text())['plans'].values():assert sha(p['path'])==p['sha256']

def metadata():
 check_anchor();OUT.mkdir(exist_ok=False);(OUT/'metadata').mkdir()
 endpoints=[('piqa_hf','https://huggingface.co/api/datasets/ybisk/piqa/revision/2e8ac2dffd59bac8c3c6714948f4c551a0848bb0'),
            ('piqa_tree','https://api.github.com/repos/ybisk/ybisk.github.io/git/trees/0947af65f14626ad58254027ac6de49de50af48c?recursive=1'),
            ('wino_hf','https://huggingface.co/api/datasets/allenai/winogrande/revision/01e74176c63542e6b0bcb004dcdea22d94fb67b5'),
            ('wino_tree','https://huggingface.co/api/datasets/allenai/winogrande/tree/01e74176c63542e6b0bcb004dcdea22d94fb67b5/winogrande_xl')]
 receipts=[]
 for key,url in endpoints:
  raw,r=fetch(url,4<<20);doc=json.loads(raw);p=OUT/'metadata'/f'{key}.json';new(p,raw);r['path']=str(p);receipts.append(r)
  if key=='piqa_tree':
   target=next(x for x in doc['tree'] if x['path']==SOURCES[0]['file']);assert (target['size'],target['sha'])==(SOURCES[0]['bytes'],SOURCES[0]['expected_git_blob_sha1'])
  if key=='wino_tree':
   target=next(x for x in doc if x['path']==SOURCES[1]['file']);assert (target['size'],target['lfs']['oid'])==(SOURCES[1]['bytes'],SOURCES[1]['expected_sha256'])
 records=[];inputs=[]
 # These validation receipts contain IDs/hashes only, never corpus prompt text.
 for p in sorted((NFS/'runs/corpus-300-v1').glob('*/batches/*.json')):
  a=json.loads(p.read_text());inputs.append(p)
  for r in a.get('validation',[]):
   records.append(dict(id=r['id'],prompt_sha256=r['prompt_sha256'],metadata_path=str(p),origin='corpus300_capture_receipt'))
 for p in [BASE/'fresh-format-heldout/manifest.json',BASE/'second-format-heldout/manifest.json']:
  a=json.loads(p.read_text());inputs.append(p)
  for r in a['items']:
   records.append({k:r[k] for k in ['id','dataset','dataset_revision','split','source_id','prompt_sha256','source_text_sha256'] if k in r}|dict(metadata_path=str(p),origin='previous_capture_manifest'))
 p=TRAIN/'selected_train_prompts.json';inputs.append(p)
 for r in json.loads(p.read_text()):records.append(dict(id=r['id'],prompt_sha256=r['prompt_sha256'],metadata_path=str(p),origin='W8_TRAIN_capture'))
 # Filename-only coverage guards against a missing corpus receipt; no text opens.
 names={p.stem for p in (NFS/'runs/corpus-300-v1/corpus/prompts').glob('*.txt')}
 covered={r['id'] for r in records};assert len(names)==300 and names<=covered
 distinct={}
 for r in records:distinct.setdefault((r['id'],r['prompt_sha256']),r)
 records=list(distinct.values())
 assert not any('piqa' in r.get('dataset','').lower() or 'winogrande' in r.get('dataset','').lower() for r in records)
 save(OUT/'exclusion_catalog.json',dict(records=records,corpus300_IDs=sorted(names),inputs_sha256={str(p):sha(p) for p in inputs},
      unique_IDs=len({r['id'] for r in records}),unique_prompt_SHA256s=len({r['prompt_sha256'] for r in records}),
      scope='All300templatecorpuscaptured IDs/SHAs plus W8TRAIN and priorWikiText/ARC/OpenBookQA/BoolQ manifests. No oldprompt/sourcepayload/route/output opened; exactID/hash dedup, not semantic or pretraining-overlap guarantee.'))
 save(OUT/'metadata_state.json',dict(receipts=receipts,sources=SOURCES,payloads_opened=False,anchor_protocol_sha256=sha(ANCHOR/'protocol.frozen.json'),
      HellaSwag_rejection='Official rowanz/hellaswag metadata GitHub451 DMCA; not fetched, no mirror workaround.',previous_sources=['synthetic_corpus300','Salesforce/wikitext','allenai/ai2_arc','allenai/openbookqa','google/boolq']))
 # Compile only a vocab tokenizer helper. Original collector/binary unchanged.
 build=json.loads((TRAIN/'build/granite4tiny.build.json').read_text());argv=list(build['argv']);argv[argv.index(str(COLLECTOR))]=str(TOKENIZER)
 target=OUT/'prompt-tokenizer';argv[argv.index('-o')+1]=str(target);subprocess.run(argv,check=True)
 save(OUT/'tokenizer.build.json',dict(argv=argv,source_sha256=sha(TOKENIZER),binary_sha256=sha(target)))
 print(json.dumps(dict(status='metadata_only_ready',excluded_IDs=len({r['id'] for r in records}),exclusion_catalog_sha256=sha(OUT/'exclusion_catalog.json'))))

def freeze():
 check_anchor();paths=[Path(__file__),TOKENIZER,OUT/'prompt-tokenizer',OUT/'tokenizer.build.json',OUT/'metadata_state.json',OUT/'exclusion_catalog.json',
  COLLECTOR,BINARY,TRAIN/'build/granite4tiny.build.json',TRAIN/'numeric_helper_identity.json',TRAIN/'protocol.frozen.json',
  MODEL,MODEL.with_name('granite4tiny-source-manifest.json'),MODEL.with_name('granite4tiny-metadata.json'),PLAN,ANCHOR/'protocol.frozen.json',ANCHOR/'completion.frozen.json',TRAIN/'granite4tiny/batch0/capture_report.json']
 paths += [Path(x['path']) for x in json.loads((ANCHOR/'protocol.frozen.json').read_text())['plans'].values()]
 paths+=list((OUT/'metadata').glob('*.json'))
 assert sha(BINARY)=='8f5171680fdab9c0284c11af11ac214998a2087314d2302e0af869f5a8476b80'
 assert sha(COLLECTOR)=='4e99c2a318e17e85c8c91fd72b41c45fe53f1528234c4679f291edeaa5859380'
 assert sha(MODEL)=='5a38b08c441ae1adbafb1d2b8a7167e0d48734d83af68b268cefea1eec553dcd'
 assert os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize >= 40*2**30
 save(OUT/'acquisition_capture_protocol.frozen.json',dict(frozen_utc=now(),inputs_sha256={str(p):sha(p) for p in paths},seed=SEED,sources=SOURCES,
  anchor_protocol_sha256=sha(ANCHOR/'protocol.frozen.json'),anchor_completion_sha256=sha(ANCHOR/'completion.frozen.json'),payloads_opened_before_freeze=False,
  selection=dict(per_source=16,source_ID='Zero-based rowindex of the pinned file, canonical decimal string; optional publisherIDs recorded separately, never used to change ranking.',
   ranking='Ascending SHA256(seed+NUL+sourcekey+NUL+revision+NUL+split+NUL+sourceID), then sourceID. No route/output/answer/length-based ranking.',
   exclusion='Beforeselection exclude matching namespaced dataset/split/sourceID and previouscapturedID. In rank order skip exact full-rendered promptSHA matching old exclusioncatalog or earliernewselectedprompt. Afterthe32IDsarefixed, final-capped duplicates abortwithoutreplacement. No semantic filtering. Invalidselectedsourcefields are integrityfailure, not reselection.',
   samplehash='SHA256 of canonical sorted compact UTF8 JSON of only three noanswer sourcefields; keep fullJSONL linehash separately forPIQA. No gold labels read or appended.',
   order='PIQA rank0..15 thenWino rank0..15; batchb receives selectionindices b::4, exactly8prompts (4persource).'),
  prompt=dict(fields={'piqa_validation':['goal','sol1','sol2'],'winogrande_validation':['sentence','option1','option2']},normalization='Each field: Unicode whitespace split then join byoneASCIIspace; no answer labels or instruction/chat template.',
   format='field0+LF+1.dot.space+field1+LF+2.dot.space+field2+LF',max_tokens=256,tokenizer='SameGGUF llama vocab_only=true; llama_tokenize add_special=true parse_special=true identical flags tocollector. IncludesBOS/specialtokens.',
   truncation='Beforeinferenceonly: while fullformattedtokenlength>256, choosefieldwithlargestUTF8bytelength (ties0,1,2), remove itslastUnicodecodepoint, rerender/recount. Retain all3fields nonempty andbothoptionlabels. No character-count approximation.',
   failure='Aftercanonicalcapturestarts, no truncation, replacement, retryselection or outputdependent filtering. Preserve allfailures/partialfiles.'),
  capture=dict(binary=str(BINARY),model=str(MODEL),original_plan_argument=str(PLAN),steps=16,threads_per_process=8,max_processes=4,batches=4,prompts_per_batch=8,
   inference_wall_budget_seconds=1200,termination='Stopall at1195s toreservecleanup within1200; retainpartialprompt/mask and originalselection, no rerun.',minimum_free_local_bytes=40*2**30,
   representation='fixed_full_FFN_W8A8_canonical_only',layers=list(range(40)),active_experts=6,split='prospective_blind',new_capture_routed_rows_paired_with_three_M32_plans=False,
   numeric='Originalcollector/source/binary unchanged. Complete-row GGUFtofloat W8[-127,127]/F32scale; completeoperatorvector dynamicA8; I64dot thenoneF32 rescale, FMAoff. CompleteGU→SwiGLU→privateDown;attention/router GGUF. Theold16chipplanargument doesnot numerically execute anyM32candidate.',
   output='All40layers prefill/decode routes, coverage/weight-cache/scales/logitfinite/hash artifacts. Extract32x16x40x6 withcompleted mask, prompt/sourceID andtokenposition. Incompletearchivesnever512primary.'),
  interpretation='Newprojectsource prompts, not proof absentfrommodelpretraining. No benchmarkaccuracy claim/goldlabels. Priorfullrow numericalproof is referenced, not threeplan pairednumeric validation. No placementfit/modeldownload.'))
 print(json.dumps(dict(protocol_sha256=sha(OUT/'acquisition_capture_protocol.frozen.json'),source_sha256=sha(__file__))))

def verify():
 check_anchor();p=json.loads((OUT/'acquisition_capture_protocol.frozen.json').read_text())
 for path,h in p['inputs_sha256'].items():assert sha(path)==h,path
 return p

def acquire():
 p=verify();ex=json.loads((OUT/'exclusion_catalog.json').read_text())['records']
 acquisition_started_utc=now()
 oldhash={r['prompt_sha256'] for r in ex};oldIDs={r['id'] for r in ex};newhash=set();items=[];receipts=[];source_counts={}
 for spec in SOURCES:
  raw,rec=fetch(spec['payload_url'],1<<20);assert len(raw)==spec['bytes']
  if 'expected_sha256' in spec:assert rec['sha256']==spec['expected_sha256']
  if 'expected_git_blob_sha1' in spec:assert hashlib.sha1(b'blob '+str(len(raw)).encode()+b'\0'+raw).hexdigest()==spec['expected_git_blob_sha1']
  local=OUT/'raw'/spec['file'].split('/')[-1];new(local,raw);rec['path']=str(local);receipts.append(rec)
  if spec['format']=='jsonl':
   rawlines=raw.decode().splitlines();rows=[json.loads(line) for line in rawlines]
  else:
   import pyarrow.parquet as pq
   rows=pq.read_table(local,columns=spec['fields']).to_pylist() # no answer column loaded
  source_counts[spec['key']]=len(rows)
  ranked=sorted(range(len(rows)),key=lambda i:(htext('\0'.join([SEED,spec['key'],spec['revision'],spec['split'],str(i)])),str(i)))
  selected=[]
  for i in ranked:
   sid=str(i);fields={f:' '.join(str(rows[i][f]).split()) for f in spec['fields']};assert all(fields.values())
   itemid=f'prospective_m32_{spec["key"]}_{len(selected):02d}'
   if itemid in oldIDs:raise RuntimeError('CapturedIDcollision')
   if any(r.get('dataset')==spec['dataset'] and r.get('split')==spec['split'] and r.get('source_id')==sid for r in ex):continue
   values=list(fields.values());text=values[0]+'\n1. '+values[1]+'\n2. '+values[2]+'\n'
   if htext(text) in oldhash|newhash:continue
   # Frozen selectedID beforecap; any postcapduplicate aborts (no hidden new sample).
   item=dict(id=itemid,source_key=spec['key'],dataset=spec['dataset'],dataset_revision=spec['revision'],dataset_split=spec['split'],split=spec['split'],source_id=sid,
     source_row_index=i,selection_rank=len(selected),ranking_sha256=htext('\0'.join([SEED,spec['key'],spec['revision'],spec['split'],sid])),
     sample_noanswer_fields_sha256=htext(json.dumps(fields,ensure_ascii=False,sort_keys=True,separators=(',',':'))),
     full_rendered_prompt_sha256=htext(text),source_payload_sha256=rec['sha256'],source_payload_path=str(local),prompt_path=str(OUT/'prompts'/f'{itemid}.txt'))
   if spec['format']=='jsonl':item['source_JSONL_line_sha256']=htext(rawlines[i])
   for k,value in enumerate(values):new(OUT/'fields'/f'{itemid}.{k}.txt',value.encode())
   selected.append(item);newhash.add(htext(text))
   if len(selected)==16:break
  assert len(selected)==16;items.extend(selected)
 assert len(items)==32
 save(OUT/'selected_IDs.pre_cap.json',items)
 (OUT/'prompts').mkdir(exist_ok=False)
 tsv=OUT/'tokenizer.jobs.tsv';new(tsv,''.join('\t'.join([j['id'],*[str(OUT/'fields'/f'{j["id"]}.{k}.txt') for k in range(3)],j['prompt_path']])+'\n' for j in items).encode())
 argv=[str(OUT/'prompt-tokenizer'),str(MODEL),str(tsv),'256',str(OUT/'tokenizer.receipt.tsv')]
 with (OUT/'tokenizer.stdout.log').open('xb') as out,(OUT/'tokenizer.stderr.log').open('xb') as err:subprocess.run(argv,stdout=out,stderr=err,check=True)
 caps={r[0]:list(map(int,r[1:])) for r in (line.split('\t') for line in (OUT/'tokenizer.receipt.tsv').read_text().splitlines())}
 seen=set()
 for j in items:
  sh=sha(j['prompt_path']);assert sh not in oldhash|seen,'Postcapduplicate: stopwithoutreplacement';seen.add(sh)
  before,after,removed,size=caps[j['id']];assert after<=256
  j.update(prompt_sha256=sh,prompt_bytes=size,prompt_tokens_including_BOS=after,untruncated_prompt_tokens=before,removed_codepoints=removed,
           task=j['source_key'],length_bucket='fixed_max256tokens',split='prospective_blind')
 root=OUT/'granite4tiny';root.mkdir(exist_ok=False)
 for b in range(4):new(root/f'batch{b}.jobs.tsv',''.join(f"{j['prompt_path']}\t{j['id']}\n" for j in items[b::4]).encode())
 save(OUT/'manifest.json',dict(status='payload_and_selection_fixed_beforeinference',created_utc=now(),acquisition_started_utc=acquisition_started_utc,protocol_sha256=sha(OUT/'acquisition_capture_protocol.frozen.json'),
  anchor_protocol_sha256=sha(ANCHOR/'protocol.frozen.json'),items=items,source_receipts=receipts,answers_in_prompts=False,placement_fitting_performed=False,
  source_rows_counts=source_counts,tokenizer_receipt_sha256=sha(OUT/'tokenizer.receipt.tsv')))
 print(json.dumps(dict(status='acquired32_no_answers',manifest_sha256=sha(OUT/'manifest.json'),prompt_token_min=min(j['prompt_tokens_including_BOS'] for j in items),prompt_token_max=max(j['prompt_tokens_including_BOS'] for j in items))))

def capture():
 verify();manifest=json.loads((OUT/'manifest.json').read_text());root=OUT/'granite4tiny'
 assert not (OUT/'capture_execution.json').exists();assert len(manifest['items'])==32
 for j in manifest['items']:assert sha(j['prompt_path'])==j['prompt_sha256']
 assert os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize>=40*2**30
 processes=[];t0=time.monotonic()
 for b in range(4):
  dest=root/f'batch{b}';dest.mkdir(exist_ok=False);argv=[str(BINARY),str(MODEL),str(root/f'batch{b}.jobs.tsv'),'16','8',str(PLAN),str(dest)]
  save(dest/'run_command.json',dict(argv=argv,started_utc=now(),manifest_sha256=sha(OUT/'manifest.json'),protocol_sha256=sha(OUT/'acquisition_capture_protocol.frozen.json')))
  out=(dest/'stdout.log').open('xb');err=(dest/'stderr.log').open('xb')
  env=dict(os.environ,OMP_NUM_THREADS='8',OPENBLAS_NUM_THREADS='1')
  proc=subprocess.Popen(argv,stdout=out,stderr=err,env=env,start_new_session=True);processes.append((b,proc,out,err))
 reason='completed'
 while any(proc.poll() is None for _,proc,_,_ in processes):
  free=os.statvfs(OUT).f_bavail*os.statvfs(OUT).f_frsize
  if time.monotonic()-t0>=1195 or free<40*2**30:
   reason='wall_budget' if time.monotonic()-t0>=1195 else 'free_space_floor'
   for _,proc,_,_ in processes:
    if proc.poll() is None:os.killpg(proc.pid,signal.SIGTERM)
   time.sleep(.5)
   for _,proc,_,_ in processes:
    if proc.poll() is None:os.killpg(proc.pid,signal.SIGKILL)
   break
  time.sleep(1)
 records=[]
 for b,proc,out,err in processes:
  proc.wait();out.close();err.close();records.append(dict(batch=b,exit_code=proc.returncode))
 wall=time.monotonic()-t0
 save(OUT/'capture_execution.json',dict(status='complete' if all(r['exit_code']==0 for r in records) else 'partial',reason=reason,wall_seconds=wall,
  batches=records,manifest_sha256=sha(OUT/'manifest.json'),all_original_selections_retained=True,new_routed_rows_paired_with_M32=False))
 print(json.dumps(dict(wall_seconds=wall,batches=records)))

def extract():
 p=verify();manifest=json.loads((OUT/'manifest.json').read_text());items=manifest['items'];root=OUT/'granite4tiny'
 assert not (OUT/'capture_summary.json').exists()
 routes=np.full((32,16,40,6),-1,np.int32);mask=np.zeros(32,bool);positions=np.full((32,16),-1,np.int32);tokens=np.full((32,16),-1,np.int32)
 pref=[];offsets=[0];stats=[];coverage_by_id=collections.defaultdict(list)
 for b in range(4):
  path=root/f'batch{b}/coverage_ledger.jsonl'
  if path.exists():
   for line in path.read_text().splitlines():
    try:r=json.loads(line)
    except json.JSONDecodeError:continue
    coverage_by_id[r['job']].append(r)
 expected={f'blk.{l}.ffn_{proj}.weight' for l in range(40) for proj in ['gate_exps','up_exps','down_exps','gate_shexp','up_shexp','down_shexp']}
 failures=[]
 for i,j in enumerate(items):
  try:
   stem=root/f'batch{i%4}/traces'/f"{j['id']}.unsharded";summary=Path(str(stem)+'.summary.json');routing=Path(str(stem)+'.routing.jsonl')
   if not summary.exists() or not routing.exists():offsets.append(offsets[-1]);continue
   s=json.loads(summary.read_text());n=s['prompt_tokens'];assert n==j['prompt_tokens_including_BOS']<=256 and s['decoded_tokens']==16
   pre=np.full((n,40,6),-1,np.int32);seen=set()
   for line in routing.read_text().splitlines():
    r=json.loads(line);L=r['layer'];assert 0<=L<40 and len(set(r['expert_ids']))==6 and all(0<=x<64 for x in r['expert_ids'])
    if r['phase']=='decode':
     t=r['step'];assert 0<=t<16 and r['position']==n+t and r['token_id']==s['generated_token_ids'][t]
     key=('decode',t,L);routes[i,t,L]=r['expert_ids'];positions[i,t]=r['position'];tokens[i,t]=r['token_id']
    else:
     t=r['position'];assert r['phase']=='prefill' and 0<=t<n and r['token_id']==s['prompt_token_ids'][t]
     key=('prefill',t,L);pre[t,L]=r['expert_ids']
    assert key not in seen;seen.add(key)
   assert np.all(routes[i]>=0) and np.all(pre>=0) and len(seen)==(n+16)*40
   totals=collections.Counter()
   for r in coverage_by_id[j['id']]:
    assert r['weight'] in expected and r['mode']=='unsharded'
    assert r['integer_mismatches']==r['scaled_bit_mismatches']==r['cross_run_output_bit_mismatches']==0
    C,R,E=r['weight_shape'][:3];K=6 if E==64 else 1
    assert r['selected_experts']==K and r['output_floats']==r['tokens']*K*R and r['input_shape'][0]==C
    totals[r['phase'],r['weight']]+=r['tokens']
   for w in expected:assert totals['prefill',w]==n and totals['decode',w]==16
   mask[i]=True;pref.append(pre);offsets.append(offsets[-1]+n);stats.append(dict(id=j['id'],source_key=j['source_key'],prefill_tokens=n,decode_tokens=16,routing_records=len(seen),routing_sha256=sha(routing)))
  except Exception as exc:
   routes[i].fill(-1);positions[i].fill(-1);tokens[i].fill(-1);mask[i]=False
   offsets.append(offsets[-1]);failures.append(dict(id=j['id'],error_type=type(exc).__name__,error=str(exc)))
 np.savez_compressed(root/'decode.routes.npz',routes=routes,prompt_ids=np.array([j['id'] for j in items]),source_ids=np.array([j['source_id'] for j in items]),
  source_keys=np.array([j['source_key'] for j in items]),layers=np.arange(40,dtype=np.int32),completed=mask,token_positions=positions,token_ids=tokens,
  representation=np.array('fixed_full_FFN_W8A8_canonical_only'),split=np.array('prospective_blind'))
 np.savez_compressed(root/'prefill.routes.npz',routes=np.concatenate(pref) if pref else np.empty((0,40,6),np.int32),offsets=np.array(offsets,np.int64),completed=mask)
 execution=json.loads((OUT/'capture_execution.json').read_text());complete=bool(mask.all()) and execution['status']=='complete'
 batch_reports=[]
 expected_finite_per_batch=[]
 vocab=json.loads((TRAIN/'granite4tiny/batch0/capture_report.json').read_text())['continuation_logit_values_checked_finite']//(8*17)
 for b in range(4):
  f=root/f'batch{b}/capture_report.json'
  if f.exists():
   r=json.loads(f.read_text());assert r['status']=='complete' and r['mode']=='canonical_only' and r['jobs']==8 and r['decode_steps']==16
   assert r['full_FFN_outputs_all_tokens_finite'] and not r['new_capture_routed_rows_paired_with_placed']
   needed=sum(j['prompt_tokens_including_BOS']+16 for j in items[b::4])*40*(6*(512+512+1536)+1024+1024+1536)
   assert r['finite_FFN_values_checked']==needed and r['continuation_logit_values_checked_finite']==8*17*vocab
   batch_reports.append(dict(batch=b,path=str(f),sha256=sha(f),finite_FFN_values=r['finite_FFN_values_checked'],finite_continuation_logits=r['continuation_logit_values_checked_finite']))
 if complete:assert len(batch_reports)==4
 save(OUT/'capture_summary.json',dict(status='complete' if complete else 'partial',prompts_completed=int(mask.sum()),complete_queries=int(mask.sum())*16,
  archive_shape=list(routes.shape),all40layers_present_for_completed=True,full_FFN_outputs_all_tokens_finite=complete,batch_reports=batch_reports,canonical_only=True,new_routed_rows_paired_with_three_M32_plans=False,
  manifest_sha256=sha(OUT/'manifest.json'),protocol_sha256=sha(OUT/'acquisition_capture_protocol.frozen.json'),decode_routes_sha256=sha(root/'decode.routes.npz'),
  prompt_records=stats,incomplete_not_primary=not complete,execution=execution,prompt_validation_failures=failures,completed_prompts=int(mask.sum()),all40layersfinite=complete,decode_archive_sha256=sha(root/'decode.routes.npz')))
 print(json.dumps(dict(status='complete' if complete else 'partial',completed=int(mask.sum()),routes_sha256=sha(root/'decode.routes.npz'))))

if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('mode',choices=['metadata','freeze','acquire','capture','extract']);a=ap.parse_args()
 globals()[a.mode]()
