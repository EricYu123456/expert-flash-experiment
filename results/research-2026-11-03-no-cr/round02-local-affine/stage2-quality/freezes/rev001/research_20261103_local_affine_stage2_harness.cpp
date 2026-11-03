#include "research_20261103_local_affine_stage2_wire.hpp"
#include "llama.h"
#include "gguf.h"
#include "json.hpp"
#include <fstream>
#include <iostream>
#include <sstream>
using namespace laq;using json=nlohmann::json;
static constexpr uint64_t FNV=14695981039346656037ULL;
static bool named(const char*name,const char*prefix,int&layer){char tail;return std::sscanf(name,(std::string(prefix)+"-%d%c").c_str(),&layer,&tail)==1;}
static bool role(const ggml_tensor*t,const char*which,int&layer){
 if(!t||t->op!=GGML_OP_MUL_MAT_ID||!t->src[0])return false;
 for(int l:{0,1,8,13})if(std::string(t->src[0]->name)=="blk."+std::to_string(l)+".ffn_"+which+"_exps.weight"){layer=l;return true;}return false;
}
static std::map<int,std::set<int>> pilots(uint32_t model){return model==1?std::map<int,std::set<int>>{{1,{14,40}},{13,{10,56}}}:std::map<int,std::set<int>>{{0,{6,17}},{8,{3,9}}};}
struct Hook {
 Header epoch;Worker*worker=nullptr;int I=0;std::map<int,std::set<int>>pilot;std::map<int,Snapshot>snap;
 bool failed=false;std::string error;uint64_t trace_hash=FNV,hits=0,calls=0;std::vector<uint8_t>trace;
 void begin(uint64_t call,int T,uint64_t position=0,uint32_t token_phase=0){need(snap.empty(),"cross-call scratch");epoch.decode=call;epoch.micro=0;epoch.T=T;epoch.position=position;epoch.token_phase=token_phase;failed=false;error.clear();}
 void finish(){need(!failed,error.c_str());need(snap.size()==pilot.size(),"missing pilot-layer callback closure");
  for(auto&[l,s]:snap){need(s.consumed&&s.weighted&&s.closed,"up/down/weighted/out incomplete epoch");uint32_t fields[]={uint32_t(l),s.epoch.T};auto add=[&](const void*p,size_t n){trace_hash=hash_bytes(trace_hash,p,n);auto*b=(const uint8_t*)p;trace.insert(trace.end(),b,b+n);};add(fields,sizeof(fields));add(s.x.data(),s.x.size()*4);add(s.ids.data(),s.ids.size()*4);}
  snap.clear();
 }
 bool callback(ggml_tensor*t,bool ask){try{
  int l=-1;bool up=role(t,"up",l)&&pilot.count(l);bool down=role(t,"down",l)&&pilot.count(l);bool weighted=named(t->name,"ffn_moe_weighted",l)&&pilot.count(l);bool out=named(t->name,"ffn_moe_out",l)&&pilot.count(l);
  if(ask)return up||down||weighted||out;
  need(environment(),"hook main x87/SSE RN FTZ0 DAZ0");
  if(up){
   need(!snap.count(l),"duplicate evaluated up snapshot");auto*w=t->src[0];need(w->ne[0]==2048&&w->ne[1]==I&&w->ne[2]==64&&w->ne[3]==1,"Up weight geometry");
   Snapshot s;s.epoch=epoch;s.epoch.layer=l;s.x=read_x(t->src[1],epoch.T,2048);s.ids=read_ids(t->src[2],epoch.T,epoch.K);snap.emplace(l,std::move(s));return true;
  }
  if(down){
   auto it=snap.find(l);need(it!=snap.end()&&!it->second.consumed,"Down missing/stale/duplicate up epoch");auto&s=it->second;
   need(same_epoch(s.epoch,HeaderWithLayer(l)),"Down epoch mismatch");auto*w=t->src[0];need(w->ne[0]==I&&w->ne[1]==2048&&w->ne[2]==64&&w->ne[3]==1,"Down weight geometry");
   need(t->src[1]&&t->src[1]->type==GGML_TYPE_F32&&t->src[1]->ne[0]==I&&t->src[1]->ne[1]==epoch.K&&t->src[1]->ne[2]==epoch.T&&t->src[1]->ne[3]==1,"actual Down input geometry");
   need(read_ids(t->src[2],epoch.T,epoch.K)==s.ids,"Down IDs do not match owned up snapshot");down_layout(t,epoch.T,2048,epoch.K);
   auto expected=mask(s.ids,epoch.T,epoch.K,pilot.at(l));std::vector<Slot>replacement;
   for(int a=0;a<int(epoch.T);++a)for(int k=0;k<int(epoch.K);++k){std::vector<float>v(2048);ggml_backend_tensor_get(t,v.data(),a*t->nb[2]+k*t->nb[1],8192);for(float x:v)need(std::isfinite(x),"source Down nonfinite");}
   if(epoch.mode==COPY_ORIGINAL){replacement=expected;for(auto&r:replacement){r.y.resize(2048);ggml_backend_tensor_get(t,r.y.data(),r.t*t->nb[2]+r.k*t->nb[1],8192);}}
   else if(epoch.mode>=CANONICAL){need(worker,"missing numerical worker");replacement=worker->compute(s.epoch,s.x,s.ids,expected);if(epoch.mode==COPY_CANONICAL){auto owned=replacement;replacement=std::move(owned);}}
   if(epoch.mode!=OBSERVER)scatter(t,replacement,epoch.T,2048,epoch.K);
   s.consumed=true;hits+=expected.size();++calls;return true;
  }
  if(weighted){auto it=snap.find(l);need(it!=snap.end()&&it->second.consumed&&!it->second.weighted,"weighted epoch/order");need(t->op==GGML_OP_MUL&&t->src[1],"final live alpha MUL");auto*a=t->src[1];host(a);need(a->type==GGML_TYPE_F32&&a->ne[0]==1&&a->ne[1]==epoch.K&&a->ne[2]==epoch.T&&a->ne[3]==1,"final alpha geometry");
   std::vector<std::pair<size_t,size_t>>iv;std::vector<float>alpha(epoch.T*epoch.K);for(uint32_t x=0;x<epoch.T;++x)for(uint32_t k=0;k<epoch.K;++k)iv.push_back({x*a->nb[2]+k*a->nb[1],4});intervals(iv,ggml_nbytes(a));
   for(uint32_t x=0;x<epoch.T;++x)for(uint32_t k=0;k<epoch.K;++k){float v;ggml_backend_tensor_get(a,&v,x*a->nb[2]+k*a->nb[1],4);need(std::isfinite(v)&&v>=0,"final alpha nonfinite/negative");alpha[x*epoch.K+k]=v;}
   trace_hash=hash_bytes(trace_hash,alpha.data(),alpha.size()*4);auto*b=(uint8_t*)alpha.data();trace.insert(trace.end(),b,b+alpha.size()*4);it->second.weighted=true;return true;
  }
  if(out){auto it=snap.find(l);need(it!=snap.end()&&it->second.weighted&&!it->second.closed,"moe_out epoch/order");it->second.closed=true;return true;}
  return true;
 }catch(const std::exception&e){failed=true;error=e.what();return false;}}
 Header HeaderWithLayer(int l){auto e=epoch;e.layer=l;return e;}
};
static bool hook_cb(ggml_tensor*t,bool ask,void*opaque){return ((Hook*)opaque)->callback(t,ask);}
struct Result {
 json report;std::vector<llama_token>generated;std::vector<float>logits;std::vector<uint8_t>trace;
};
static double nll(const float*p,int V,int target){need(target>=0&&target<V,"NLL target index");double mx=-INFINITY;for(int j=0;j<V;++j){need(std::isfinite(p[j]),"nonfinite logits");mx=std::max(mx,double(p[j]));}double total=0;for(int j=0;j<V;++j)total=total+std::exp(double(p[j])-mx);return mx+std::log(total)-double(p[target]);}
static std::vector<std::pair<int,int>> target_index(int prefix,const std::vector<llama_token>&gt){std::vector<std::pair<int,int>>out;for(int j=0;j<int(gt.size());++j)out.push_back({prefix-1+j,gt[j]});return out;}
static Result run(llama_model*m,ggml_threadpool*pool,Worker&worker,uint32_t model,uint32_t mode,uint64_t runid,uint64_t context,uint32_t subrun,const std::vector<llama_token>&prefix,const std::vector<llama_token>&gt,bool free_run,bool keep_logits){
 need(prefix.size()>0&&prefix.size()+gt.size()<=1024&&mode<=FACTOR,"run size/mode");need(environment()&&check_pool(pool),"actual graph worker environment");Hook tr;tr.epoch.run=runid;tr.epoch.context=context;tr.epoch.subrun=subrun;tr.epoch.arm=tr.epoch.mode=mode;tr.epoch.model=model;tr.epoch.K=model==1?6:8;tr.pilot=pilots(model);tr.I=model==1?1408:1024;tr.worker=&worker;
 auto cp=llama_context_default_params();cp.n_ctx=1024;cp.n_batch=cp.n_ubatch=128;cp.n_threads=cp.n_threads_batch=8;cp.n_seq_max=1;if(mode!=SOURCE){cp.cb_eval=hook_cb;cp.cb_eval_user_data=&tr;}
 auto*c=llama_init_from_model(m,cp);need(c,"fresh context");llama_attach_threadpool(c,pool,pool);auto batch=llama_batch_init(128,0,1);auto*v=llama_model_get_vocab(m);int V=llama_vocab_n_tokens(v);Result out;uint64_t call=0,lh=FNV;int next=0;std::map<int,int>targets;for(auto [pos,tok]:target_index(prefix.size(),gt))targets[pos]=tok;json scores=json::array();auto start=std::chrono::steady_clock::now();
 try{
  auto eval=[&](const std::vector<llama_token>&tokens,int offset,int count,uint32_t token_phase){batch.n_tokens=0;for(int j=0;j<count;++j){int i=batch.n_tokens++;batch.token[i]=tokens[j];batch.pos[i]=offset+j;batch.n_seq_id[i]=1;batch.seq_id[i][0]=0;batch.logits[i]=true;}
   need(check_pool(pool),"actual graph workers before call");tr.begin(++call,count,offset,token_phase);int rc=llama_decode(c,batch);llama_synchronize(c);if(tr.failed)throw std::runtime_error(tr.error);need(rc==0,"llama decode");if(mode!=SOURCE)tr.finish();need(check_pool(pool),"actual graph workers after call");
   for(int j=0;j<count;++j){auto*p=llama_get_logits_ith(c,j);need(p,"requested logits missing");next=0;for(int i=0;i<V;++i){need(std::isfinite(p[i]),"finite full vocabulary");if(p[i]>p[next])next=i;}lh=hash_bytes(lh,p,size_t(V)*4);if(keep_logits)out.logits.insert(out.logits.end(),p,p+V);int pos=offset+j;if(targets.count(pos))scores.push_back({{"input_position",pos},{"target_position",pos+1},{"target_id",targets.at(pos)},{"nll",nll(p,V,targets.at(pos))}});}
  };
  for(size_t a=0;a<prefix.size();a+=128){std::vector<llama_token>x(prefix.begin()+a,prefix.begin()+std::min(prefix.size(),a+128));eval(x,a,x.size(),0);}
  std::string stopped="fixed_replay";int eog=-1;
  if(free_run){stopped="max32";for(int j=0;j<32;++j){if(llama_vocab_is_eog(v,next)){stopped="EOG";eog=next;break;}out.generated.push_back(next);eval(std::vector<llama_token>{next},prefix.size()+j,1,1);}}
  else if(!gt.empty()){
   // The last target need not be fed: input position p-1+j predicts GT[j].
   for(size_t j=0;j+1<gt.size();){int count=subrun==3?std::min<size_t>(128,gt.size()-1-j):1;std::vector<llama_token>x(gt.begin()+j,gt.begin()+j+count);eval(x,prefix.size()+j,count,2);j+=count;}
   need(scores.size()==gt.size(),"groundtruth NLL target-index closure");
  }
  out.trace=std::move(tr.trace);out.report={{"model",model},{"mode",mode},{"run",runid},{"context",context},{"subrun",subrun},{"prefix_ids",prefix},{"groundtruth_ids",gt},{"generated_ids",out.generated},{"stop",stopped},{"EOG_not_fed",eog},{"full_vocab_fnv1a64",std::to_string(lh)},{"trace_fnv1a64",std::to_string(tr.trace_hash)},{"pilot_hits",tr.hits},{"layer_calls",tr.calls},{"nll_targets",scores},{"seconds_quality_harness_only",std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()},{"fresh_KV",true}};
  llama_batch_free(batch);llama_free(c);return out;
 }catch(...){llama_batch_free(batch);llama_free(c);throw;}
}
static void equal(const Result&a,const Result&b,bool trace){need(a.generated==b.generated&&a.logits.size()==b.logits.size()&&std::memcmp(a.logits.data(),b.logits.data(),a.logits.size()*4)==0,"full vocabulary/source-copy bit control failed");if(trace)need(a.trace==b.trace,"owned input/IDs/final-alpha bit control failed");}
static void metadata(const std::string&path,int model){ggml_context*ctx=nullptr;gguf_init_params p{true,&ctx};auto*f=gguf_init_from_file(path.c_str(),p);need(f&&ctx,"GGUF metadata");int I=model==1?1408:1024;
 for(auto [l,es]:pilots(model))for(std::string role:{"gate","up","down"}){std::string prefix="blk."+std::to_string(l)+".ffn_"+role+"_exps";auto*t=ggml_get_tensor(ctx,(prefix+".weight").c_str());need(t&&t->ne[0]==(role=="down"?I:2048)&&t->ne[1]==(role=="down"?2048:I)&&t->ne[2]==64&&t->ne[3]==1,"adapter GUDown geometry");for(int64_t j=0;j<gguf_get_n_tensors(f);++j){std::string name=gguf_get_tensor_name(f,j);if(name.rfind(prefix,0)==0)need(name==prefix+".weight","unsupported expert bias/scale tensor");}need(!ggml_get_tensor(ctx,("blk."+std::to_string(l)+".ffn_gate_up_exps.weight").c_str()),"merged GU unsupported");}
 gguf_free(f);ggml_free(ctx);
}
static std::vector<llama_token> tokenize(llama_model*m,const std::string&path){std::ifstream f(path,std::ios::binary);need(bool(f),"prompt open");std::string text((std::istreambuf_iterator<char>(f)),{});auto*v=llama_model_get_vocab(m);int n=-llama_tokenize(v,text.data(),text.size(),nullptr,0,true,true);need(n>0,"tokenize size");std::vector<llama_token>tokens(n);need(llama_tokenize(v,text.data(),text.size(),tokens.data(),n,true,true)==n,"tokenize");tokens.resize(std::min(n,128));return tokens;}
static void save(const std::string&path,const json&j){std::ofstream f(path);need(bool(f),"result create");f<<j.dump(2)<<'\n';need(bool(f),"result write");}
static int actual(const json&plan){
 uint32_t mid=plan.at("model_id");need(mid==1||mid==2,"model adapter");std::string phase=plan.at("phase");need(phase=="TOOLING"||phase=="DEVELOPMENT"||phase=="QUALITY","run phase");
 // Worker verifies Root receipt/source hashes before any operator or GGUF read.
 Worker worker(plan.at("worker_argv").get<std::vector<std::string>>(),mid);metadata(plan.at("gguf"),mid);llama_backend_init();ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto*m=llama_model_load_from_file(plan.at("gguf").get<std::string>().c_str(),mp);need(m&&llama_model_n_embd(m)==2048&&llama_model_n_layer(m)==(mid==1?27:16),"native model adapter");auto pp=ggml_threadpool_params_default(8);pp.poll=0;auto*pool=ggml_threadpool_new(&pp);need(pool&&check_pool(pool),"actual graph pool");uint64_t context=0,runid=plan.at("run_id");json reports=json::array();
 try{
  for(auto job:plan.at("jobs")){
   std::vector<llama_token>prefix=job.contains("prefix_ids")?job.at("prefix_ids").get<std::vector<llama_token>>():tokenize(m,job.at("path"));
   auto invoke=[&](uint32_t mode,uint32_t sub,const std::vector<llama_token>&gt,bool free,bool keep){return run(m,pool,worker,mid,mode,runid,++context,sub,prefix,gt,free,keep);};
   json jr={{"id",job.at("id")},{"immutable_prefix_ids",prefix},{"runs",json::array()}};
   if(phase=="TOOLING"){
    auto a=invoke(SOURCE,0,{},true,true);auto o=invoke(OBSERVER,0,{},true,true);equal(a,o,false);auto c=invoke(COPY_ORIGINAL,0,{},true,true);equal(a,c,false);equal(o,c,true);auto repeat=invoke(OBSERVER,0,{},true,true);equal(o,repeat,true);auto b=invoke(CANONICAL,0,{},true,true);auto bc=invoke(COPY_CANONICAL,0,{},true,true);equal(b,bc,true);
    for(auto*r:{&a,&o,&c,&repeat,&b,&bc})jr["runs"].push_back(r->report);jr["tooling_bit_controls_passed"]=true;
   }else if(phase=="DEVELOPMENT"){
    auto source=invoke(SOURCE,1,{},true,false);jr["runs"].push_back(source.report);auto gt=source.generated;
    for(uint32_t mode:{SOURCE,CANONICAL,LOCAL,FACTOR}){auto replay=invoke(mode,2,gt,false,false);jr["runs"].push_back(replay.report);if(mode!=SOURCE){auto free=invoke(mode,1,{},true,false);jr["runs"].push_back(free.report);}}
   }else{
    auto gt=job.at("groundtruth_ids").get<std::vector<llama_token>>();need(prefix.size()==256&&gt.size()==256,"fixed256 context+256 groundtruth interface");for(uint32_t mode:{SOURCE,CANONICAL,LOCAL,FACTOR})jr["runs"].push_back(invoke(mode,3,gt,false,false).report);
   }
   reports.push_back(jr);save(plan.at("output"),{{"status","partial_complete"},{"jobs",reports},{"TEST_opened",false},{"quality_harness_not_compressed_timing",true}});
  }
  worker.finish();ggml_threadpool_free(pool);llama_model_free(m);llama_backend_free();save(plan.at("output"),{{"status","complete"},{"jobs",reports},{"TEST_opened",false},{"quality_harness_not_compressed_timing",true}});return 0;
 }catch(...){ggml_threadpool_free(pool);llama_model_free(m);llama_backend_free();throw;}
}
#ifndef LAQ_NO_MAIN
int main(int argc,char**argv){try{signal(SIGPIPE,SIG_IGN);need(environment(),"harness x87/SSE RN FTZ0 DAZ0");need(argc==3&&std::string(argv[1])=="run","run approved-plan.json");std::ifstream f(argv[2]);need(bool(f),"approved plan");json plan;f>>plan;return actual(plan);}catch(const std::exception&e){std::cerr<<"Stage2 harness invalid/tooling: "<<e.what()<<'\n';return 1;}}
#endif
