#!/usr/bin/env python3
"""Existing frozen WikiText windows: original DeepSeek Q8 GGUF vs fixed FFN W8A8."""
import argparse,collections,hashlib,json,math,os,pathlib,signal,subprocess,time
import numpy as np
ROOT=pathlib.Path('/home/ericyu/expert_flash_experiment');NFS=pathlib.Path('/home/nfs_client/ericyu/expert_flash_experiment');BASE=ROOT/'results/research-2026-11-02/round2';OUT=BASE/'deepseek-teacher-quality';SELF=pathlib.Path(__file__).resolve();OLD=ROOT/'scripts/research_20261103_deepseek_full_ffn.cpp';TEACHER=ROOT/'scripts/research_20261103_teacher_forced_quality.cpp';CPP=ROOT/'scripts/research_20261103_deepseek_teacher_forced_quality.cpp';MODEL=NFS/'models/deepseek-v2-lite-chat-q8_0.gguf';PRIOR=BASE/'fresh-format-heldout/quality/protocol.frozen.json';BINARY=OUT/'build/deepseek-teacher-quality'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def lines(p):return [json.loads(x) for x in p.read_text().splitlines()]
def prepare():
 OUT.mkdir(exist_ok=True);(OUT/'build').mkdir(exist_ok=True)
 if CPP.exists() or (OUT/'protocol.frozen.json').exists():raise RuntimeError('Never overwrite quality source/protocol')
 old=OLD.read_text();teacher=TEACHER.read_text();prefix=old[:old.index('struct Trace')];suffix=teacher[teacher.index('static void audit_source_gguf('):teacher.index('static void load_plan(')]
 finite=r'''
static uint64_t finite_W8_FFN_values=0;
static void verify_W8_FFN_finite(const ggml_tensor*t){
 if(t->type!=GGML_TYPE_F32||t->ne[3]!=1)throw std::runtime_error("Finite FFN layout");
 for(int64_t z=0;z<t->ne[2];++z)for(int64_t y=0;y<t->ne[1];++y)for(int64_t r=0;r<t->ne[0];++r){float v;std::memcpy(&v,(const char*)t->data+z*t->nb[2]+y*t->nb[1]+r*t->nb[0],4);if(!std::isfinite(v))throw std::runtime_error("Nonfinite W8 FFN output");++finite_W8_FFN_values;}
}
'''
 assert suffix.count('else audit(t,layer,p);return true;')==1;suffix=suffix.replace('else audit(t,layer,p);return true;','else{audit(t,layer,p);verify_W8_FFN_finite(t);}return true;')
 main=r'''
int main(int argc,char**argv){if(argc!=7){std::fprintf(stderr,"Usage: MODEL JOBS TOKENS THREADS OUTDIR REPEAT_FIRST\n");return 2;}try{
 int limit=std::stoi(argv[3]),threads=std::stoi(argv[4]),repeat_first=std::stoi(argv[6]);if(limit!=128||threads!=8||(repeat_first!=0&&repeat_first!=1))throw std::runtime_error("Fixed teacher bounds");root=argv[5];std::filesystem::create_directories(root+"/traces");std::filesystem::create_directories(root+"/scales");omp_set_num_threads(threads);ledger.open(root+"/coverage_ledger.jsonl");cache_ledger.open(root+"/weight_cache_ledger.jsonl");
 gguf_init_params gp{true,nullptr};gguf=gguf_init_from_file(argv[1],gp);if(!gguf)throw std::runtime_error("GGUF metadata");int fd=open(argv[1],O_RDONLY);struct stat st;if(fd<0||fstat(fd,&st))throw std::runtime_error("GGUF open");mapping_size=st.st_size;void*m=mmap(nullptr,mapping_size,PROT_READ,MAP_SHARED,fd,0);close(fd);if(m==MAP_FAILED)throw std::runtime_error("GGUF mmap");mapping=(const char*)m;
 std::ifstream list(argv[2]);std::string line;std::vector<std::pair<std::string,std::string>>jobs;while(std::getline(list,line)){auto tab=line.find('\t');if(tab==std::string::npos)throw std::runtime_error("Teacher jobs TSV");jobs.push_back({line.substr(0,tab),line.substr(tab+1)});}if(jobs.size()!=4)throw std::runtime_error("Expected exactly4 fixed windows perbatch");if(repeat_first){std::ofstream first(root+"/repeat_first_id.txt");first<<jobs[0].second<<'\n';}
 llama_backend_init();ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto*model=llama_model_load_from_file(argv[1],mp);if(!model)throw std::runtime_error("Model load");
 for(const std::string&variant:{std::string("source-gguf"),std::string("unsharded")}){mode=variant;for(const auto&j:jobs){quality_run(model,j.first,j.second,limit,threads,false);ledger.flush();cache_ledger.flush();}if(mode=="unsharded"&&repeat_first){quality_run(model,jobs[0].first,jobs[0].second,limit,threads,true);ledger.flush();}}
 std::ofstream out(root+"/completion_numeric.json");out<<"{\"status\":\"complete\",\"teacher_windows\":4,\"tokens_per_window\":128,\"scored_tokens_per_window\":127,\"MoE_raw_layers\":26,\"dense_raw0_included\":true,\"FFN_weight_names\":159,\"all_logits_finite\":true,\"finite_W8_FFN_values\":"<<finite_W8_FFN_values<<",\"repeat_first_W8\":"<<(repeat_first?"true":"false")<<",\"cache_entries\":"<<caches.size()<<"}\n";out.close();ledger.close();cache_ledger.close();llama_model_free(model);llama_backend_free();munmap((void*)mapping,mapping_size);gguf_free(gguf);return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"deepseek-teacher-quality: %s\n",e.what());return 1;}}
'''
 source=prefix+finite+suffix+main;CPP.write_text(source);proof=[]
 spans=[('quantize','static float quantize(','static int64_t dot('),('dot','static int64_t dot(','static float scale_output('),('rescale','static float scale_output(','static bool equal_bits('),('target','static bool target(','static const Cache &get_cache('),('weight_cache','static const Cache &get_cache(','struct Stats'),('operator','static void audit(','struct Trace')]
 for label,start,end in spans:
  a=old[old.index(start):old.index(end,old.index(start))].strip();newend='static uint64_t finite_W8_FFN_values' if label=='operator' else end;b=source[source.index(start):source.index(newend,source.index(start))].strip();assert a==b,label;proof.append(dict(span=label,source='priorDeepfullFFN',byte_exact=True,body_sha256=hashlib.sha256(a.encode()).hexdigest()))
 for label,start,end in [('source_observer','static void audit_source_gguf(','struct Trace'),('tokenization','static std::vector<llama_token> tokenize_fixed(','static void quality_run('),('teacher_measurement_and_context','static void quality_run(','static void load_plan(')]:
  a=teacher[teacher.index(start):teacher.index(end,teacher.index(start))].strip();newend='int main(' if label=='teacher_measurement_and_context' else end;b=source[source.index(start):source.index(newend,source.index(start))].strip();assert a==b,label;proof.append(dict(span=label,source='existingG4teacher',byte_exact=True,body_sha256=hashlib.sha256(a.encode()).hexdigest()))
 build=json.loads((BASE/'deepseek-full-ffn-integer-reference/build_command.json').read_text());argv=build['argv'];argv[argv.index(str(OLD))]=str(CPP);argv[argv.index('-o')+1]=str(BINARY);subprocess.run(argv,check=True);dump(OUT/'build/build_command.json',dict(argv=argv,source_sha256=sha(CPP),binary_sha256=sha(BINARY)));dump(OUT/'helper_identity.json',proof)
 prior=json.loads(PRIOR.read_text());jobs=prior['jobs'];assert len(jobs)==8 and all(j['kind']=='lm' for j in jobs)
 for j in jobs:assert sha(j['prompt_path'])==j['prompt_sha256']
 for b in range(2):(OUT/f'batch{b}.jobs.tsv').write_text(''.join(f"{j['prompt_path']}\t{j['id']}\n" for j in jobs[b::2]))
 required=[SELF,OLD,TEACHER,CPP,BINARY,MODEL,PRIOR,BASE/'deepseek-full-ffn-integer-reference/protocol.frozen.json',OUT/'helper_identity.json',OUT/'build/build_command.json',OUT/'batch0.jobs.tsv',OUT/'batch1.jobs.tsv']+[pathlib.Path(j['prompt_path']) for j in jobs]+[pathlib.Path(x) for x in argv if x.endswith('.a')]
 spec=dict(frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),inputs_sha256={str(p):sha(p) for p in required},jobs=jobs,job_order='Same original acquisition-frozen lm00..07 textitems; 2batches indices batch::2, no reselect.',model=str(MODEL),binary=str(BINARY),modes=['source-gguf','unsharded'],threads_per_process=8,max_processes=2,inference_budget_seconds=1200,tokens_per_window=128,scored_next_tokens_per_window=127,total_scored_next_tokens=1016,tokenization='Same8 old frozen rawtext items, first128 DeepGGUF llama_tokenize(add_special=true,parse_special=true) tokens with originalBOS rules andnochattemplate. Exact IDs shared across Deepcontrols; tokenIDs/textwindow endpoint neednot match G4tokenization.',numeric_contract='Identical originalDeep fullFFN fixedrowW8[-127,127]/fullvector dynamicA8/I64fullrowdot/F32rescale helpers, includingraw0dense andallrouted/shared down; attention/router sourceGGUF unchanged. No quantizer orplacement tuning.',teacher_context='Eachwindow/reference initializesindependent context; all128tokens prefill andalllogits requested. No free-running generation.',loss='F64 maxshift fullvocabulary logsumexp minusF32 targetlogit; aggregate exp(totalNLL/1016).',repeat='First old frozen item W8 repeatedin independentcontext; all128x102400 F32logits compared bitwise.',outputs='Perposition NLL/logsumexp/targetlogit/top1; all128 routertop6 eachrawMoElayer; first2fullvocablogit vectors; firstW8window full128vectors +repeat comparison; allFFN finitecoverage ledgers.',limitations='Supplemental quality sensitivity on existing text relative tosourceQ8GGUF, not newblindtest, BF16equivalence, taskaccuracy, newquantization method orhardware performance.',timeout='Keepcompleted windows andpartial logs; no pickingreplacement/shorter windows. Fullquality summary requiresall8source+W8 windowsandrepeat.')
 dump(OUT/'protocol.frozen.json',spec);print(json.dumps({'status':'frozen','protocol_sha256':sha(OUT/'protocol.frozen.json'),'source_sha256':sha(CPP),'binary_sha256':sha(BINARY)},indent=2))

def verify():
 spec=json.loads((OUT/'protocol.frozen.json').read_text())
 for p,h in spec['inputs_sha256'].items():assert sha(p)==h,p
 return spec

def run():
 spec=verify()
 if (OUT/'started.json').exists():raise RuntimeError('Never overwrite quality execution')
 dump(OUT/'started.json',dict(protocol_sha256=sha(OUT/'protocol.frozen.json'),utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime())));active={};begin=time.monotonic();deadline=begin+1200;next_print=begin+30;finished=[]
 for b in range(2):
  d=OUT/f'batch{b}';d.mkdir();argv=[str(BINARY),str(MODEL),str(OUT/f'batch{b}.jobs.tsv'),'128','8',str(d),'1' if b==0 else '0'];dump(d/'command.json',dict(argv=argv));out=(d/'stdout.log').open('w');err=(d/'stderr.log').open('w');p=subprocess.Popen(argv,stdout=out,stderr=err,env=dict(os.environ,OMP_NUM_THREADS='8',OMP_DYNAMIC='FALSE',OPENBLAS_NUM_THREADS='1'),start_new_session=True);active[p.pid]=(p,b,out,err,time.monotonic())
 while active:
  for pid,(p,b,out,err,start) in list(active.items()):
   code=p.poll()
   if code is not None:out.close();err.close();row=dict(batch=b,exit_code=code,wall_seconds=time.monotonic()-start);finished.append(row);dump(OUT/f'batch{b}/execution.json',row);del active[pid];print('BATCH_COMPLETE',json.dumps(row),flush=True)
  now=time.monotonic()
  if now>=next_print:
   counts={f'batch{b}':len(list((OUT/f'batch{b}/traces').glob('*.summary.json'))) for b in range(2)};print('PROGRESS',json.dumps(dict(elapsed_seconds=now-begin,completed_reference_contexts=counts)),flush=True);next_print=now+30
  if now>=deadline:
   for p,*_ in active.values():os.killpg(p.pid,signal.SIGTERM)
   for p,b,out,err,start in active.values():
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    out.close();err.close();finished.append(dict(batch=b,exit_code='BUDGET_STOP',wall_seconds=time.monotonic()-start))
   active.clear();break
  if active:time.sleep(2)
 dump(OUT/'execution.json',dict(wall_seconds=time.monotonic()-begin,budget_seconds=1200,batches=finished));verify();print('QUALITY_EXECUTION_FINISHED',flush=True)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('action',choices=['prepare','run']);a=p.parse_args();prepare() if a.action=='prepare' else run()
