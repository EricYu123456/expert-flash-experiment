#define LAQ_NO_MAIN
#ifndef LARF_HAS_OLD_HELPERS
#include "/home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/stage2-quality/freezes/rev004/research_20261103_local_affine_stage2_harness.cpp"
#endif
static bool admitted(const Header&h){return (h.mode==CANONICAL&&h.token_phase==0)||((h.mode==CANONICAL||h.mode==LOCAL||h.mode==FACTOR)&&h.token_phase==1);}
struct AlphaLedger {
 std::ofstream file;uint64_t frames=0,rows=0;
 AlphaLedger(const std::string&p):file(p,std::ios::binary){need(bool(file),"create fresh alpha ledger");file.write("LARFALP1",8);}
 void append(Header h,const Snapshot&s,const ggml_tensor*a,uint64_t request,const std::set<int>&selected){
  if(!admitted(h))return;h.request=request;h.kind=4;h.slots=mask(s.ids,h.T,h.K,selected).size();h.bytes=uint64_t(h.T)*h.K*8;header_valid(h);std::vector<float>alpha(h.T*h.K);
  for(uint32_t t=0;t<h.T;++t)for(uint32_t k=0;k<h.K;++k){ggml_backend_tensor_get(a,&alpha[t*h.K+k],t*a->nb[2]+k*a->nb[1],4);need(std::isfinite(alpha[t*h.K+k])&&alpha[t*h.K+k]>=0,"live alpha ledger finite");}
  file.write((char*)&h,sizeof(h));file.write((char*)s.ids.data(),s.ids.size()*4);file.write((char*)alpha.data(),alpha.size()*4);file.flush();need(bool(file),"alpha ledger exact write");++frames;rows+=h.T;
 }
};
struct RefreshHook:Hook {
 AlphaLedger*ledger=nullptr;std::map<int,uint64_t>requests;
 bool collect(ggml_tensor*t,bool ask){
  bool ok=Hook::callback(t,ask);if(!ok||ask)return ok;
  try{int l=-1;if(role(t,"down",l)&&pilot.count(l))requests[l]=worker->serial;
   if(named(t->name,"ffn_moe_weighted",l)&&pilot.count(l)){need(ledger&&requests.count(l),"alpha join missing evaluated Down request");ledger->append(HeaderWithLayer(l),snap.at(l),t->src[1],requests.at(l),pilot.at(l));}
   return true;
  }catch(const std::exception&e){failed=true;error=e.what();return false;}
 }
};
static bool refresh_cb(ggml_tensor*t,bool ask,void*p){return ((RefreshHook*)p)->collect(t,ask);}
static std::vector<llama_token> refresh_tokens(llama_model*m,const std::string&path){
 std::ifstream f(path,std::ios::binary);need(bool(f),"approved preprocessed suffix open");std::string suffix((std::istreambuf_iterator<char>(f)),{});auto*v=llama_model_get_vocab(m);int n=-llama_tokenize(v,suffix.data(),suffix.size(),nullptr,0,true,false);need(n>=768,"calibration incomplete: suffix has fewer than768 native tokens; no substitution");std::vector<llama_token>ids(n);need(llama_tokenize(v,suffix.data(),suffix.size(),ids.data(),n,true,false)==n,"refresh native tokenizer");ids.resize(768);return ids;
}
static json rollout(llama_model*m,ggml_threadpool*pool,Worker&worker,AlphaLedger&ledger,uint32_t model,uint32_t mode,uint64_t runid,uint64_t context,const std::vector<llama_token>&prefix){
 need(prefix.size()==768&&(mode==CANONICAL||mode==LOCAL||mode==FACTOR),"fixed refresh rollout");need(environment()&&check_pool(pool),"refresh actual graph environment");RefreshHook tr;tr.epoch.run=runid;tr.epoch.context=context;tr.epoch.subrun=4;tr.epoch.arm=tr.epoch.mode=mode;tr.epoch.model=model;tr.epoch.K=model==1?6:8;tr.pilot=pilots(model);tr.I=model==1?1408:1024;tr.worker=&worker;tr.ledger=&ledger;
 auto cp=llama_context_default_params();cp.n_ctx=1024;cp.n_batch=cp.n_ubatch=128;cp.n_threads=cp.n_threads_batch=8;cp.n_seq_max=1;cp.cb_eval=refresh_cb;cp.cb_eval_user_data=&tr;auto*c=llama_init_from_model(m,cp);need(c,"fresh refresh context");llama_attach_threadpool(c,pool,pool);auto batch=llama_batch_init(128,0,1);auto*v=llama_model_get_vocab(m);int V=llama_vocab_n_tokens(v),next=0;uint64_t call=0,hash=FNV;std::vector<llama_token>gen;std::string stop="max32";int eog=-1;auto start=std::chrono::steady_clock::now();
 try{
  auto eval=[&](const std::vector<llama_token>&tokens,int pos,uint32_t phase){batch.n_tokens=tokens.size();for(int j=0;j<batch.n_tokens;++j){batch.token[j]=tokens[j];batch.pos[j]=pos+j;batch.n_seq_id[j]=1;batch.seq_id[j][0]=0;batch.logits[j]=true;}
   need(check_pool(pool),"refresh graph workers before call");tr.begin(++call,tokens.size(),pos,phase);int rc=llama_decode(c,batch);llama_synchronize(c);if(tr.failed)throw std::runtime_error(tr.error);need(rc==0,"refresh llama decode");tr.finish();tr.trace.clear();need(check_pool(pool),"refresh graph workers after call");
   for(int j=0;j<batch.n_tokens;++j){auto*p=llama_get_logits_ith(c,j);need(p,"refresh logits");next=0;for(int i=0;i<V;++i){need(std::isfinite(p[i]),"refresh fullvocab finite");if(p[i]>p[next])next=i;}hash=hash_bytes(hash,p,size_t(V)*4);}
  };
  for(int a=0;a<768;a+=128)eval(std::vector<llama_token>(prefix.begin()+a,prefix.begin()+a+128),a,0);
  for(int step=0;step<32;++step){if(llama_vocab_is_eog(v,next)){stop="EOG";eog=next;break;}gen.push_back(next);eval(std::vector<llama_token>{next},768+step,1);}
  json r={{"model",model},{"mode",mode},{"arm",mode},{"run",runid},{"context",context},{"subrun",4},{"prefix_ids",prefix},{"generated_ids",gen},{"stop",stop},{"EOG_not_fed",eog},{"decode_input_rows",gen.size()},{"prefix_input_rows",768},{"fresh_KV",true},{"pilot_hits",tr.hits},{"layer_calls",tr.calls},{"full_vocab_fnv1a64",std::to_string(hash)},{"trace_fnv1a64",std::to_string(tr.trace_hash)},{"seconds_quality_harness_only",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()}};llama_batch_free(batch);llama_free(c);return r;
 }catch(...){llama_batch_free(batch);llama_free(c);throw;}
}
static int collect_refresh(const json&plan){
 need(plan.at("phase")=="COLLECT_REFRESH","only explicit collection phase");uint32_t mid=plan.at("model_id");need(mid==1||mid==2,"refresh model adapter");Worker worker(plan.at("worker_argv").get<std::vector<std::string>>(),mid);AlphaLedger alpha(plan.at("alpha_ledger"));metadata(plan.at("gguf"),mid);llama_backend_init();ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto*m=llama_model_load_from_file(plan.at("gguf").get<std::string>().c_str(),mp);need(m&&llama_model_n_embd(m)==2048&&llama_model_n_layer(m)==(mid==1?27:16),"refresh pinned native model");auto pp=ggml_threadpool_params_default(8);pp.poll=0;auto*pool=ggml_threadpool_new(&pp);need(pool&&check_pool(pool),"refresh pool");json reports=json::array();uint64_t context=0,runid=plan.at("run_id");
 try{for(auto job:plan.at("jobs")){auto prefix=refresh_tokens(m,job.at("suffix_path"));json jr={{"book_id",job.at("book_id")},{"immutable_prefix_ids",prefix},{"runs",json::array()}};for(uint32_t mode:{CANONICAL,LOCAL,FACTOR})jr["runs"].push_back(rollout(m,pool,worker,alpha,mid,mode,runid,++context,prefix));reports.push_back(jr);save(plan.at("output"),{{"status","partial_complete"},{"jobs",reports},{"alpha_frames",alpha.frames},{"alpha_rows_both_layers",alpha.rows},{"TEST_opened",false}});std::cerr<<"COLLECT_REFRESH completed book "<<job.at("book_id")<<" model "<<mid<<'\n';}
  worker.finish();ggml_threadpool_free(pool);llama_model_free(m);llama_backend_free();save(plan.at("output"),{{"status","complete"},{"jobs",reports},{"alpha_frames",alpha.frames},{"alpha_rows_both_layers",alpha.rows},{"TEST_opened",false},{"teacher_offroute_targets_queried",false}});return 0;
 }catch(...){ggml_threadpool_free(pool);llama_model_free(m);llama_backend_free();throw;}
}
#ifndef LARF_NO_MAIN
int main(int argc,char**argv){try{signal(SIGPIPE,SIG_IGN);need(environment(),"refresh caller strict RN");need(argc==3&&std::string(argv[1])=="collect","collect approved-plan.json");std::ifstream f(argv[2]);need(bool(f),"refresh approved plan");json p;f>>p;return collect_refresh(p);}catch(const std::exception&e){std::cerr<<"Refresh collection tooling invalid/incomplete: "<<e.what()<<'\n';return 1;}}
#endif
