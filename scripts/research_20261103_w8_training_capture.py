#!/usr/bin/env python3
"""Balanced TRAIN-only canonical full-FFN W8 routing capture, no refitting."""
from datetime import datetime,timezone
import argparse,collections,hashlib,json,os,pathlib,re,signal,subprocess,time
import numpy as np
ROOT=pathlib.Path('/home/ericyu/expert_flash_experiment')
NFS=pathlib.Path('/home/nfs_client/ericyu/expert_flash_experiment')
BASE=ROOT/'results/research-2026-11-02/round2'
OUT=BASE/'w8-format-training'
CORPUS=NFS/'runs/corpus-300-v1/corpus'
SELF=pathlib.Path(__file__).resolve()
SEED='20261103'
MODELS={
 'deepseek':dict(old_source=ROOT/'scripts/research_20261103_deepseek_full_ffn.cpp',old_binary=BASE/'deepseek-full-ffn-integer-reference/build/deepseek-full-ffn',old_protocol=BASE/'deepseek-full-ffn-integer-reference/protocol.frozen.json',old_build=BASE/'deepseek-full-ffn-integer-reference/build_command.json',old_summary=BASE/'deepseek-full-ffn-integer-reference/summary.json',model=NFS/'models/deepseek-v2-lite-chat-q8_0.gguf',metadata=NFS/'models/deepseek-metadata.json',layers=list(range(1,27)),dense_layer=0,projection_inventory=BASE/'deepseek-full-ffn-integer-reference/frozen_equal_projection_rows.json'),
 'granite4tiny':dict(old_source=ROOT/'scripts/research_20261102_full_ffn_integer_reference.cpp',old_binary=BASE/'full-ffn-integer-reference/build/full-ffn-integer-reference',old_protocol=BASE/'full-ffn-integer-reference/frozen_protocol.json',old_build=BASE/'full-ffn-integer-reference/build_command.json',old_summary=BASE/'fresh-format-heldout/generation/summary.json',model=NFS/'models/granite4tiny-q4_k_m.gguf',metadata=NFS/'models/granite4tiny-metadata.json',layers=list(range(40)),dense_layer=None,placement=BASE/'full-ffn-integer-reference/placement.tsv')}
# old_summary for G4 is prior proof only; no fresh prompt, route or test payload
# is opened by this collector, and no new selection depends on that proof.
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def write(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def jsonlines(p):return [json.loads(l) for l in p.read_text().splitlines()]
def select():
 OUT.mkdir(exist_ok=True)
 if (OUT/'selection_protocol.frozen.json').exists():raise RuntimeError('Never overwrite selection freeze')
 manifest_path=CORPUS/'manifest.json';manifest=json.loads(manifest_path.read_text());rows=[]
 # The aggregate corpus JSONL contains test text and is deliberately never opened.
 for p in sorted((CORPUS/'prompts').glob('train_*.txt')):
  match=re.fullmatch(r'train_(.+)_(short|medium|long)_(\d+)',p.stem);assert match and match[1] in manifest['task_counts']
  rows.append(dict(id=p.stem,task=match[1],length_bucket=match[2],path=str(p)))
 assert len(rows)==manifest['split_counts']['train']==200
 counts=collections.Counter((r['task'],r['length_bucket']) for r in rows)
 for key,n in manifest['cells'].items():
  split,task,length=key.split('/')
  if split=='train':assert counts[task,length]==n
 metadata_path=OUT/'eligible_train_filename_metadata.json';write(metadata_path,rows)
 rank=lambda domain,value:hashlib.sha256((SEED+'\0'+domain+'\0'+value).encode()).hexdigest()
 tasks=sorted({r['task'] for r in rows},key=lambda t:(rank('task',t),t))
 ranked={t:sorted([r for r in rows if r['task']==t],key=lambda r:(rank('prompt',r['id']),r['id'])) for t in tasks}
 chosen=[];cursor=collections.Counter()
 while len(chosen)<32:
  for task in tasks:
   if len(chosen)==32:break
   if cursor[task]<len(ranked[task]):
    row=dict(ranked[task][cursor[task]]);row.update(selection_index=len(chosen),task_rank=rank('task',task),prompt_rank=rank('prompt',row['id']));chosen.append(row);cursor[task]+=1
 assert len({r['id'] for r in chosen})==32 and max(cursor.values())-min(cursor.values())<=1
 # Freeze selection before opening ANY chosen text bytes.
 protocol={'frozen_utc':datetime.now(timezone.utc).isoformat(),'seed':SEED,'source_manifest':str(manifest_path),'source_manifest_sha256':sha(manifest_path),'eligible_metadata_path':str(metadata_path),'eligible_metadata_sha256':sha(metadata_path),'eligible_train_prompts':200,'task_order':tasks,'task_order_rule':'SHA256(seed+NUL+task+NUL+task_name), then task name','within_task_rule':'SHA256(seed+NUL+prompt+NUL+prompt_id), then ID','selection_rule':'Balanced round-robin over seeded task order, next ranked TRAIN ID per task, skip exhausted tasks, stop at32. No length/content/output filtering.','selected_task_counts':dict(cursor),'selected_length_counts':dict(collections.Counter(r['length_bucket'] for r in chosen)),'selected':chosen,'test_or_fresh_text_opened':False,'aggregate_corpus_JSONL_opened':False,'collector_source_sha256':sha(SELF)}
 write(OUT/'selection_protocol.frozen.json',protocol);return protocol

def make_source(name,cfg):
 old=cfg['old_source'].read_text();prefix=old[:old.index('int main(')]
 addition='''
static uint64_t finite_ffn_values=0,finite_vocab_values=0;
static std::map<std::string,uint64_t> training_logit_hash;
static void verify_training_ffn_finite(const ggml_tensor*t){
 if(t->type!=GGML_TYPE_F32||t->ne[3]!=1)throw std::runtime_error("Unsupported training FFN output");
 for(int64_t z=0;z<t->ne[2];++z)for(int64_t y=0;y<t->ne[1];++y)for(int64_t r=0;r<t->ne[0];++r){float f;std::memcpy(&f,(const char*)t->data+z*t->nb[2]+y*t->nb[1]+r*t->nb[0],4);if(!std::isfinite(f))throw std::runtime_error("Nonfinite training FFN output");++finite_ffn_values;}
}
'''
 prefix=prefix.replace('struct Trace {FILE*route',addition+'\nstruct Trace {FILE*route')
 assert 'audit(t,layer,p);return true;' in prefix
 prefix=prefix.replace('audit(t,layer,p);return true;','audit(t,layer,p);verify_training_ffn_finite(t);return true;')
 a=prefix.index('static void check_logits(');b=prefix.index('static void run(',a)
 prefix=prefix[:a]+'''static void check_logits(llama_context*ctx,int vocab){
 const float*p=llama_get_logits_ith(ctx,-1);if(!p)throw std::runtime_error("Missing training continuation logits");
 for(int i=0;i<vocab;++i){if(!std::isfinite(p[i]))throw std::runtime_error("Nonfinite training continuation logits");++finite_vocab_values;}
 auto it=training_logit_hash.find(current_job);if(it==training_logit_hash.end())it=training_logit_hash.emplace(current_job,H0).first;
 it->second=hash_bytes(it->second,p,(size_t)vocab*4);
 if(current_phase=="prefill"||(current_phase=="decode"&&current_step==0)){
  std::string path=root+"/snapshots/"+current_job+"."+current_phase+"."+std::to_string(current_step)+".logits.f32";
  std::ofstream f(path,std::ios::binary);f.write((const char*)p,(size_t)vocab*4);if(!f)throw std::runtime_error("Training logit snapshot write");
 }
}
''' +prefix[b:]
 argc=7 if name=='granite4tiny' else 6;rootarg=6 if name=='granite4tiny' else 5;load='load_plan(argv[5]);' if name=='granite4tiny' else ''
 main=r'''
int main(int argc,char**argv){if(argc!=ARGC){std::fprintf(stderr,"Training capture arguments\n");return 2;}try{
 int steps=std::stoi(argv[3]),threads=std::stoi(argv[4]);if(steps!=16||threads!=8)throw std::runtime_error("Fixed training capture steps/threads");root=argv[ROOTARG];
 std::filesystem::create_directories(root+"/scales");std::filesystem::create_directories(root+"/traces");std::filesystem::create_directories(root+"/snapshots");LOADPLANomp_set_num_threads(threads);
 ledger.open(root+"/coverage_ledger.jsonl");cache_ledger.open(root+"/weight_cache_ledger.jsonl");
 gguf_init_params gp{true,nullptr};gguf=gguf_init_from_file(argv[1],gp);if(!gguf)throw std::runtime_error("GGUF metadata");int fd=open(argv[1],O_RDONLY);struct stat st;if(fd<0||fstat(fd,&st))throw std::runtime_error("Model open");mapping_size=st.st_size;void*m=mmap(nullptr,mapping_size,PROT_READ,MAP_SHARED,fd,0);close(fd);if(m==MAP_FAILED)throw std::runtime_error("Model mmap");mapping=(const char*)m;
 llama_backend_init();ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto*model=llama_model_load_from_file(argv[1],mp);if(!model)throw std::runtime_error("Model loading");
 mode="unsharded";reference_outputs=std::fopen("/dev/null","wb");if(!reference_outputs)throw std::runtime_error("Capture FFN sink");
 std::ifstream f(argv[2]);std::string line;int njobs=0;while(std::getline(f,line)){auto tab=line.find('\t');if(tab==std::string::npos)throw std::runtime_error("Training TSV");run(model,line.substr(0,tab),line.substr(tab+1),steps,threads);ledger.flush();cache_ledger.flush();++njobs;}
 if(njobs!=8)throw std::runtime_error("Expected8 training prompts per batch");std::fclose(reference_outputs);
 std::ofstream report(root+"/capture_report.json");report<<"{\"status\":\"complete\",\"model\":\"MODELNAME\",\"mode\":\"canonical_only\",\"new_capture_routed_rows_paired_with_placed\":false,\"jobs\":"<<njobs<<",\"decode_steps\":16,\"full_FFN_outputs_all_tokens_finite\":true,\"finite_FFN_values_checked\":"<<finite_ffn_values<<",\"continuation_logit_values_checked_finite\":"<<finite_vocab_values<<",\"cache_entries\":"<<caches.size()<<",\"prompt_logit_stream_hashes\":{";
 bool comma=false;for(const auto&e:training_logit_hash){if(comma)report<<',';comma=true;report<<'\"'<<e.first<<"\":\""<<e.second<<'\"';}report<<"}}\n";report.close();ledger.close();cache_ledger.close();llama_model_free(model);llama_backend_free();munmap((void*)mapping,mapping_size);gguf_free(gguf);return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"w8-training-capture: %s\n",e.what());return 1;}}
'''
 main=main.replace('ARGC',str(argc)).replace('ROOTARG',str(rootarg)).replace('LOADPLAN',load).replace('MODELNAME',name)
 target=ROOT/'scripts'/f'research_20261103_w8_training_capture_{name}.cpp'
 if target.exists():raise RuntimeError('Never overwrite a capture source')
 target.write_text(prefix+main)
 # Numeric math, cache and operator implementation remain byte-identical.
 spans=[('quantize','static float quantize(','static int64_t dot('),('dot','static int64_t dot(','static float scale_output('),('scale_output','static float scale_output(','static bool equal_bits('),('target','static bool target(','static const Cache &get_cache('),('weight_cache','static const Cache &get_cache(','struct Stats'),('operator','static void audit(','struct Trace'),('run','static void run(','int main(')]
 proofs=[]
 new=target.read_text()
 for label,begin,end in spans:
  aa=old[old.index(begin):old.index(end,old.index(begin))].strip();newend='static uint64_t finite_ffn_values' if label=='operator' else end
  bb=new[new.index(begin):new.index(newend,new.index(begin))].strip();assert aa==bb,(name,label)
  proofs.append(dict(function=label,byte_exact=True,body_sha256=hashlib.sha256(aa.encode()).hexdigest()))
 return target,proofs

def prepare():
 selection=select();chosen=selection['selected']
 # Only the32 frozen TRAIN text files are opened from this point onward.
 jobs=[dict(r,split='train',prompt_sha256=sha(r['path']),prompt_bytes=pathlib.Path(r['path']).stat().st_size) for r in chosen]
 builddir=OUT/'build';builddir.mkdir(exist_ok=True);sources=[];binaries={};proofs={}
 for name,cfg in MODELS.items():
  src,proof=make_source(name,cfg);binary=builddir/f'{name}-canonical-capture';argv=list(json.loads(cfg['old_build'].read_text())['argv']);argv[argv.index(str(cfg['old_source']))]=str(src);argv[argv.index('-o')+1]=str(binary);subprocess.run(argv,check=True)
  write(builddir/f'{name}.build.json',{'argv':argv,'source_sha256':sha(src),'binary_sha256':sha(binary)});sources.append(src);binaries[name]=str(binary);proofs[name]=proof
 write(OUT/'numeric_helper_identity.json',proofs);write(OUT/'selected_train_prompts.json',jobs)
 for name in MODELS:
  (OUT/name).mkdir(exist_ok=True)
  for b in range(4):(OUT/name/f'batch{b}.jobs.tsv').write_text(''.join(f"{j['path']}\t{j['id']}\n" for j in jobs[b::4]))
 inputs=[SELF,CORPUS/'manifest.json',OUT/'eligible_train_filename_metadata.json',OUT/'selection_protocol.frozen.json',OUT/'selected_train_prompts.json',OUT/'numeric_helper_identity.json',*sources]
 for name,cfg in MODELS.items():inputs += [cfg['old_source'],cfg['old_binary'],cfg['old_protocol'],cfg['old_build'],cfg['old_summary'],cfg['model'],cfg['metadata'],pathlib.Path(binaries[name]),builddir/f'{name}.build.json']+[OUT/name/f'batch{b}.jobs.tsv' for b in range(4)]+([cfg['placement']] if 'placement' in cfg else [cfg['projection_inventory']])
 # The prior G4 proof summary is metadata only; no fresh text/routes are opened.
 inputs.extend(pathlib.Path(x) for x in json.loads(MODELS['granite4tiny']['old_build'].read_text())['argv'] if x.endswith('.a'))
 hashes={str(p):sha(p) for p in set(inputs)}
 protocol={'frozen_utc':datetime.now(timezone.utc).isoformat(),'inputs_sha256':hashes,'selection_protocol_sha256':sha(OUT/'selection_protocol.frozen.json'),'prompts':jobs,'same32IDs_for_both_models':True,'steps':16,'threads_per_process':8,'max_concurrent_processes':4,'inference_wall_budget_seconds':1200,'batch_rule':'Each model has4batches of8 IDs, indices batch::4 in selection order. Queue DeepSeek batches0..3 thenG4batches0..3; launch next when a worker exits.','models':{name:{'model':str(cfg['model']),'binary':binaries[name],'routed_raw_layers':cfg['layers'],'dense_leading_layer':cfg['dense_layer'],'active_experts':6,'experts':64} for name,cfg in MODELS.items()},'capture_scope':'Canonical-only full-FFN route collection. New routed rows/logits are NOT rerun in a placed mode. Prior paired G4 andDeepSeek proofs are referenced, not extended by this capture. Shared/dense still execute the unchanged packed-vs-canonical I64 check in the old operator.','numeric_contract':'Unchanged complete-row GGUF-to_float fixedrowW8[-127,127], full-vector dynamicA8 once per operator vector, exact I64 dot andoneF32rescale. All FFN/down/shared/prefill tokens replaced; DeepSeek raw0dense separately included, shared hidden2816single down. Attention/router remainGGUF.','output_policy':'Record all route JSONL, operator input/output hashes, all-FFN finite counts, continuation-logit finite hashes and2logit snapshots perprompt. FFN reference stream goes to/dev/null; no massive FFN scalar snapshot or paired-placement claim.','data_policy':'Only existing200-TRAIN filename metadata and32selectedTRAIN text files. No aggregate JSONL/test text/freshWikiText/ARC text opened. No refit, model download, quality measurement or holdout selection.','timeout_policy':'Preserve completed prompts/partiallogs; never truncate/reselect. Archive incomplete slots as-1 pluscompleted mask; statuspartial forbids treating incomplete archive asfull32.'}
 write(OUT/'protocol.frozen.json',protocol);print(json.dumps({'status':'frozen','task_counts':selection['selected_task_counts'],'length_counts':selection['selected_length_counts'],'selected_text_bytes':sum(j['prompt_bytes'] for j in jobs),'protocol_sha256':sha(OUT/'protocol.frozen.json')},indent=2),flush=True)

def verify():
 spec=json.loads((OUT/'protocol.frozen.json').read_text())
 for p,h in spec['inputs_sha256'].items():assert sha(p)==h,p
 for j in spec['prompts']:assert sha(j['path'])==j['prompt_sha256']
 return spec

def command(name,b,spec):
 cfg=MODELS[name];argv=[spec['models'][name]['binary'],str(cfg['model']),str(OUT/name/f'batch{b}.jobs.tsv'),'16','8']
 if 'placement' in cfg:argv.append(str(cfg['placement']))
 return argv+[str(OUT/name/f'batch{b}')]

def extract(spec,status):
 reports={}
 for name,cfg in MODELS.items():
  layers=cfg['layers'];L=len(layers);route=np.full((32,16,L,6),-1,dtype=np.int32);mask=np.zeros(32,dtype=np.bool_);raw_to_i={l:i for i,l in enumerate(layers)};prefill=[];tokenids=[];offsets=[0];promptdata=[];coverage_records=0;kind_totals=collections.Counter();entries=collections.defaultdict(list);batch_reports=[]
  for b in range(4):
   ledger=OUT/name/f'batch{b}/coverage_ledger.jsonl'
   if ledger.exists():
    for raw in ledger.read_text().splitlines():
     try:r=json.loads(raw)
     except json.JSONDecodeError:continue
     entries[r['job']].append(r)
   report_path=OUT/name/f'batch{b}/capture_report.json'
   if report_path.exists():
    report=json.loads(report_path.read_text());assert report['status']=='complete' and report['jobs']==8 and report['full_FFN_outputs_all_tokens_finite'];batch_reports.append(dict(batch=b,report=report,sha256=sha(report_path)))
  expected={f'blk.{l}.ffn_{p}.weight' for l in layers for p in ['gate_exps','up_exps','down_exps','gate_shexp','up_shexp','down_shexp']}
  if cfg['dense_layer'] is not None:expected|={f'blk.0.ffn_{p}.weight' for p in ['gate','up','down']}
  for i,j in enumerate(spec['prompts']):
   batch=i%4;stem=OUT/name/f'batch{batch}/traces'/f"{j['id']}.unsharded";summarypath=pathlib.Path(str(stem)+'.summary.json');routepath=pathlib.Path(str(stem)+'.routing.jsonl')
   if not summarypath.exists() or not routepath.exists():offsets.append(offsets[-1]);continue
   summary=json.loads(summarypath.read_text());n=summary['prompt_tokens'];assert summary['decoded_tokens']==16 and len(summary['generated_token_ids'])==16
   pref=np.full((n,L,6),-1,dtype=np.int32);seen=set();routing=jsonlines(routepath)
   for r in routing:
    l=r['layer'];assert l in raw_to_i and len(set(r['expert_ids']))==6 and all(0<=e<64 for e in r['expert_ids']);li=raw_to_i[l]
    if r['phase']=='decode':
     step=r['step'];assert 0<=step<16 and r['position']==n+step and r['token_id']==summary['generated_token_ids'][step];key=('decode',step,l);assert key not in seen;route[i,step,li]=r['expert_ids']
    else:
     assert r['phase']=='prefill';p=r['position'];assert 0<=p<n and r['token_id']==summary['prompt_token_ids'][p];key=('prefill',p,l);assert key not in seen;pref[p,li]=r['expert_ids']
    seen.add(key)
   assert np.all(pref>=0) and np.all(route[i]>=0) and len(routing)==(n+16)*L
   totals=collections.Counter();callbacks=collections.Counter()
   for r in entries[j['id']]:
    assert r['weight'] in expected and r['mode']=='unsharded';T=r['tokens'];C,R,E=r['weight_shape'][:3];K=6 if E==64 else 1
    assert r['selected_experts']==K and r['output_floats']==T*K*R and r['input_shape'][0]==C and r['integer_mismatches']==r['scaled_bit_mismatches']==r['cross_run_output_bit_mismatches']==0
    assert r['op']==('MUL_MAT_ID' if K==6 else 'MUL_MAT')
    if K==6:assert r['output_shape']==[R,6,T,1] and r['input_shape'][2:]==[T,1]
    else:assert r['output_shape']==[R,T,1,1] and r['input_shape']==[C,T,1,1]
    if r['phase']=='decode':assert T==1
    totals[r['phase'],r['weight']]+=T;callbacks[r['phase'],r['weight']]+=1;coverage_records+=1;kind_totals['dense' if r['layer']==0 and name=='deepseek' else 'shared' if 'shexp' in r['projection'] else 'routed']+=r['output_floats']
   for w in expected:assert totals['prefill',w]==n and totals['decode',w]==16 and callbacks['prefill',w]==(n+511)//512 and callbacks['decode',w]==16
   mask[i]=True;prefill.append(pref);tokenids.extend(summary['prompt_token_ids']);offsets.append(offsets[-1]+n);promptdata.append({'selection_index':i,'id':j['id'],'task':j['task'],'length_bucket':j['length_bucket'],'prefill_tokens':n,'decode_tokens':16,'routing_sha256':sha(routepath),'route_records':len(routing),'generated_token_ids':summary['generated_token_ids']})
  assert len(offsets)==33
  np.savez_compressed(OUT/name/'decode.routes.npz',routes=route,prompt_ids=np.asarray([j['id'] for j in spec['prompts']]),tasks=np.asarray([j['task'] for j in spec['prompts']]),length_buckets=np.asarray([j['length_bucket'] for j in spec['prompts']]),layers=np.asarray(layers,dtype=np.int32),completed=mask,representation=np.asarray('fixed_full_FFN_W8A8_canonical_only'),split=np.asarray('train'))
  np.savez_compressed(OUT/name/'prefill.routes.npz',routes=np.concatenate(prefill,axis=0) if prefill else np.empty((0,L,6),dtype=np.int32),prompt_offsets=np.asarray(offsets,dtype=np.int64),token_ids=np.asarray(tokenids,dtype=np.int32),prompt_ids=np.asarray([j['id'] for j in spec['prompts']]),layers=np.asarray(layers,dtype=np.int32),completed=mask,split=np.asarray('train'))
  result={'model':name,'status':'complete' if mask.all() else 'partial','completed_prompts':int(mask.sum()),'requested_prompts':32,'decode_route_shape':list(route.shape),'routed_raw_layers':layers,'prefill_token_offsets_shape':[33],'prefill_tokens_completed':int(offsets[-1]),'completed_decode_tokens':int(mask.sum())*16,'coverage_callback_records_completed_prompts':coverage_records,'full_FFN_weight_names':len(expected),'FFN_output_rows_by_kind_completed_prompts':dict(kind_totals),'new_routed_placement_pairing':False,'source_split':'TRAIN','quality_measurement':False,'complete_batch_finite_reports':batch_reports,'prompt_data':promptdata,'decode_archive_sha256':sha(OUT/name/'decode.routes.npz'),'prefill_archive_sha256':sha(OUT/name/'prefill.routes.npz')};write(OUT/name/'summary.json',result);reports[name]=result
 overall={'status':'complete' if all(r['status']=='complete' for r in reports.values()) else 'partial','protocol_sha256':sha(OUT/'protocol.frozen.json'),'execution_status':status,'models':reports,'same_frozen32TRAIN_IDs':True,'selection_counts':json.loads((OUT/'selection_protocol.frozen.json').read_text())['selected_task_counts'],'new_data_paired_with_placed':False,'compiler_refit_performed':False};write(OUT/'summary.json',overall);return overall

def run():
 
 if (OUT/'run_started.json').exists():raise RuntimeError('Never overwrite an executed capture')
 spec=verify();started=time.monotonic();deadline=started+1200;queue=[(name,b) for name in MODELS for b in range(4)];active={};completed=[];next_print=started+30
 write(OUT/'run_started.json',{'started_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(OUT/'protocol.frozen.json'),'max_concurrent_processes':4})
 while queue or active:
  while queue and len(active)<4 and time.monotonic()<deadline:
   name,b=queue.pop(0);dest=OUT/name/f'batch{b}'
   if dest.exists():raise RuntimeError('Never overwrite batch capture')
   dest.mkdir();argv=command(name,b,spec);write(dest/'run_command.json',{'argv':argv,'started_utc':datetime.now(timezone.utc).isoformat(),'protocol_sha256':sha(OUT/'protocol.frozen.json')});stdout=open(dest/'stdout.log','w');stderr=open(dest/'stderr.log','w');proc=subprocess.Popen(argv,stdout=stdout,stderr=stderr,env=dict(os.environ,OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1'),start_new_session=True);active[proc.pid]=(proc,name,b,time.monotonic(),stdout,stderr)
  for pid,(proc,name,b,began,out,err) in list(active.items()):
   code=proc.poll()
   if code is not None:
    out.close();err.close();row={'model':name,'batch':b,'exit_code':code,'wall_seconds':time.monotonic()-began};completed.append(row);write(OUT/name/f'batch{b}/execution.json',row);del active[pid];print('BATCH_COMPLETE',json.dumps(row),flush=True)
  now=time.monotonic()
  if now>=next_print:
   counts={name:len(list((OUT/name).glob('batch*/traces/*.summary.json'))) for name in MODELS};print('PROGRESS',json.dumps({'elapsed_seconds':round(now-started,1),'completed_prompts':counts,'active_batches':[(v[1],v[2]) for v in active.values()],'queued_batches':queue}),flush=True);write(OUT/'progress.json',{'elapsed_seconds':now-started,'completed_prompts':counts,'active_batches':[(v[1],v[2]) for v in active.values()],'queued_batches':queue});next_print=now+30
  if now>=deadline:
   for proc,*_ in active.values():os.killpg(proc.pid,signal.SIGTERM)
   for pid,(proc,name,b,began,out,err) in list(active.items()):
    try:proc.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(proc.pid,signal.SIGKILL);proc.wait()
    out.close();err.close();row={'model':name,'batch':b,'exit_code':'BUDGET_STOP','wall_seconds':time.monotonic()-began};completed.append(row);write(OUT/name/f'batch{b}/execution.json',row)
   active.clear();break
  if queue or active:time.sleep(2)
 status={'wall_seconds':time.monotonic()-started,'completed_batches':completed,'unlaunched_batches':queue,'budget_seconds':1200}
 write(OUT/'execution.json',status);verify();result=extract(spec,status);print(json.dumps({name:{k:v for k,v in row.items() if k!='prompt_data'} for name,row in result['models'].items()},indent=2),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run','audit']);a=p.parse_args()
 if a.action=='prepare':prepare()
 elif a.action=='run':run()
 else:extract(verify(),json.loads((OUT/'execution.json').read_text()))
if __name__=='__main__':main()
