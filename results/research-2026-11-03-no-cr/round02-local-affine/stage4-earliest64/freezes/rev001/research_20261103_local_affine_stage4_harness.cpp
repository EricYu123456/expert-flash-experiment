// Original helpers stay immutable. Only earliest pilot set/new modes are mounted.
#define LAQ_QUALITY_NO_MAIN
#include "/home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/stage2-pg19-quality/freezes/rev001/research_20261103_local_affine_stage2_pg19_quality_harness.cpp"
static constexpr uint32_t B64=9,COPY_B64=10,C64=11,COPY_C64=12;
static int early_layer(uint32_t model){need(model==1||model==2,"Stage4 model");return model==1?1:0;}
static std::set<int> all_experts(){std::set<int>s;for(int e=0;e<64;++e)s.insert(e);return s;}
struct OwnedBatch{uint32_t layer=0,T=0,position=0;std::vector<float>x,alpha;std::vector<int32_t>ids;};
struct EarlyHook:Hook{
 std::vector<OwnedBatch>owned;std::map<int,std::array<uint64_t,3>>exposure;
 void setup(uint32_t model){pilot={{early_layer(model),all_experts()}};for(int e=0;e<64;++e)exposure[e]={0,0,0};}
 bool observe(ggml_tensor*t,bool ask){bool ok=Hook::callback(t,ask);if(!ok||ask)return ok;try{
  int layer=-1;
  if(role(t,"down",layer)&&pilot.count(layer)){auto&s=snap.at(layer);need(s.consumed,"Stage4 scatter must complete first");auto slots=mask(s.ids,epoch.T,epoch.K,pilot.at(layer));need(slots.size()==size_t(epoch.T)*epoch.K,"Stage4 all64 selected slots");for(auto&v:slots)++exposure[v.e][zone_index(epoch.position+v.t)];}
  if(named(t->name,"ffn_moe_weighted",layer)&&pilot.count(layer)){
   auto&s=snap.at(layer);need(s.weighted&&trace.size()>=size_t(epoch.T)*epoch.K*4,"Stage4 actualalpha closure");
   OwnedBatch b;b.layer=layer;b.T=epoch.T;b.position=epoch.position;b.x=s.x;b.ids=s.ids;b.alpha.resize(size_t(epoch.T)*epoch.K);
   // The base Hook appended actual final-MUL alpha bytes immediately above.
   std::memcpy(b.alpha.data(),trace.data()+trace.size()-b.alpha.size()*4,b.alpha.size()*4);owned.push_back(std::move(b));
  }return true;
 }catch(const std::exception&e){failed=true;error=e.what();return false;}}
 json report(){json j=json::object();for(auto[e,n]:exposure)for(int z=0;z<3;++z)j[std::to_string(early_layer(epoch.model))][std::to_string(e)][zones[z]]=n[z];return j;}
};
static bool early_callback(ggml_tensor*t,bool ask,void*opaque){return ((EarlyHook*)opaque)->observe(t,ask);}
struct EarlyResult{QualityResult quality;std::vector<OwnedBatch>owned;};
static json invariant(const EarlyResult&a,const EarlyResult&b,int K){
 need(a.owned.size()==4&&b.owned.size()==4,"four owned earliest batches required");uint64_t nx=0,ni=0,na=0;int positions=0;
 auto different=[](const void*a,const void*b,int n){uint64_t count=0;auto*x=(const uint8_t*)a;auto*y=(const uint8_t*)b;for(int i=0;i<n;++i){uint32_t u,v;std::memcpy(&u,x+4*i,4);std::memcpy(&v,y+4*i,4);count+=u!=v;}return count;};
 for(int bi=0;bi<4;++bi){auto&x=a.owned[bi];auto&y=b.owned[bi];need(x.T==128&&y.T==128&&x.position==uint32_t(bi*128)&&y.position==x.position&&x.layer==y.layer,"owned batch position/layer");
  need(x.x.size()==128*2048&&y.x.size()==x.x.size()&&x.ids.size()==size_t(128*K)&&y.ids.size()==x.ids.size()&&x.alpha.size()==x.ids.size()&&y.alpha.size()==x.ids.size(),"owned invariant shapes");
  for(int t=0;t<128;++t){nx+=different(x.x.data()+t*2048,y.x.data()+t*2048,2048);ni+=different(x.ids.data()+t*K,y.ids.data()+t*K,K);na+=different(x.alpha.data()+t*K,y.alpha.data()+t*K,K);++positions;}
 }
 return {{"positions_compared",positions},{"X_F32_u32_mismatches",nx},{"ordered_ID_I32_mismatches",ni},{"final_actual_alpha_F32_u32_mismatches",na},{"literal_owned_bytes_compared",true},{"passed",nx==0&&ni==0&&na==0}};
}
static EarlyResult early_run(llama_model*m,ggml_threadpool*pool,Worker&worker,uint32_t mid,uint32_t mode,uint64_t run,uint64_t context,const std::vector<llama_token>&ids,bool keep){
 need(ids.size()==512&&(mode<=2||mode==9||mode==10||mode==11||mode==12),"Stage4 mode and GT512");need(environment()&&check_pool(pool),"Stage4 graph numeric workers");
 EarlyHook tr;tr.epoch.run=run;tr.epoch.context=context;tr.epoch.subrun=9;tr.epoch.mode=tr.epoch.arm=mode;tr.epoch.model=mid;tr.epoch.K=mid==1?6:8;tr.I=mid==1?1408:1024;tr.worker=&worker;tr.setup(mid);
 auto cp=llama_context_default_params();cp.n_ctx=1024;cp.n_batch=cp.n_ubatch=128;cp.n_threads=cp.n_threads_batch=8;cp.n_seq_max=1;if(mode!=0){cp.cb_eval=early_callback;cp.cb_eval_user_data=&tr;}auto*c=llama_init_from_model(m,cp);need(c,"Stage4 fresh KV context");llama_attach_threadpool(c,pool,pool);auto batch=llama_batch_init(128,0,1);EarlyResult result;auto&out=result.quality;out.V=llama_vocab_n_tokens(llama_model_get_vocab(m));uint64_t hash=FNV;json scores=json::array();
 try{for(int pos=0;pos<512;pos+=128){batch.n_tokens=128;for(int t=0;t<128;++t){batch.token[t]=ids[pos+t];batch.pos[t]=pos+t;batch.n_seq_id[t]=1;batch.seq_id[t][0]=0;batch.logits[t]=true;}
  need(check_pool(pool),"Stage4 before batch RN");tr.begin(pos/128+1,128,pos,pos<256?0:2);int rc=llama_decode(c,batch);llama_synchronize(c);if(tr.failed)throw std::runtime_error(tr.error);need(rc==0,"Stage4 decode");if(mode!=0)tr.finish();need(check_pool(pool),"Stage4 after batch RN");
  for(int t=0;t<128;++t){auto*p=llama_get_logits_ith(c,t);need(p,"Stage4 vocab logits");for(int j=0;j<out.V;++j)need(std::isfinite(p[j]),"Stage4 nonfinite logits");hash=hash_bytes(hash,p,size_t(out.V)*4);if(keep)out.logits.insert(out.logits.end(),p,p+out.V);int i=pos+t;if(i>=255&&i<=510)scores.push_back({{"input_position",i},{"target_position",i+1},{"target_id",ids[i+1]},{"nll",nll(p,out.V,ids[i+1])}});}
  if(!keep)tr.trace.clear();
 }
 need(scores.size()==256,"Stage4 exactly256 NLL targets");result.owned=std::move(tr.owned);out.trace=std::move(tr.trace);
 out.report={{"model",mid},{"reported_arm",mode<=1?0:mode},{"internal_hook_mode",mode},{"run",run},{"context",context},{"subrun",9},{"fresh_KV",true},{"four_batches_128",true},{"unused511_fed",true},{"nll_targets",scores},{"expert_exposure",tr.report()},{"exposure_scope",mode==1?"source potential live all64 hits; no replacements":"actual successful all64 Down/scattered rows"},{"full_vocab_fnv1a64",std::to_string(hash)},{"trace_fnv1a64",std::to_string(tr.trace_hash)}};
 llama_batch_free(batch);llama_free(c);return result;
 }catch(...){llama_batch_free(batch);llama_free(c);throw;}
}
static void early_metadata(const std::string&path,int mid){
 ggml_context*ctx=nullptr;gguf_init_params p{true,&ctx};auto*f=gguf_init_from_file(path.c_str(),p);need(f&&ctx,"Stage4 GGUF metadata");int l=early_layer(mid),I=mid==1?1408:1024;
 for(std::string role:{"gate","up","down"}){auto prefix="blk."+std::to_string(l)+".ffn_"+role+"_exps";auto*t=ggml_get_tensor(ctx,(prefix+".weight").c_str());need(t&&t->ne[0]==(role=="down"?I:2048)&&t->ne[1]==(role=="down"?2048:I)&&t->ne[2]==64&&t->ne[3]==1,"Stage4 GUDown shapes");for(int64_t i=0;i<gguf_get_n_tensors(f);++i){std::string n=gguf_get_tensor_name(f,i);if(n.rfind(prefix,0)==0)need(n==prefix+".weight","Stage4 bias/scale unsupported");}}
 need(!ggml_get_tensor(ctx,("blk."+std::to_string(l)+".ffn_gate_up_exps.weight").c_str()),"Stage4 mergedGU unsupported");gguf_free(f);ggml_free(ctx);
}
static int tokenize_only(const json&p){
 need(p.at("phase")=="ACQUIRE_FRESH16_TOKENIZE","separate tokenize phase");llama_backend_init();auto mp=llama_model_default_params();mp.vocab_only=true;mp.n_gpu_layers=0;auto*m=llama_model_load_from_file(p.at("gguf").get<std::string>().c_str(),mp);need(m,"Stage4 vocab-only model");json jobs=json::array();
 try{for(auto&j:p.at("jobs")){auto ids=joint_tokens(m,j.at("suffix_path"));jobs.push_back({{"book_id",j.at("book_id")},{"status",ids.size()==512?"complete":"insufficient_window"},{"joint_ids",ids},{"native_at_least_tokens",ids.size()},{"add_special",true},{"parse_special",false}});}save(p.at("output"),{{"status","complete"},{"model_id",p.at("model_id")},{"vocab_only",true},{"jobs",jobs}});llama_model_free(m);llama_backend_free();return 0;}catch(...){llama_model_free(m);llama_backend_free();throw;}
}
static int quality_only(const json&p){
 need(p.at("phase")=="QUALITY_C64","separate quality phase");uint32_t mid=p.at("model_id");need(mid==1||mid==2,"Stage4 adapter");Worker worker(p.at("worker_argv").get<std::vector<std::string>>(),mid);early_metadata(p.at("gguf"),mid);llama_backend_init();ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto*m=llama_model_load_from_file(p.at("gguf").get<std::string>().c_str(),mp);need(m&&llama_model_n_embd(m)==2048&&llama_model_n_layer(m)==(mid==1?27:16),"Stage4 native model geometry");auto pp=ggml_threadpool_params_default(8);pp.poll=0;auto*pool=ggml_threadpool_new(&pp);need(pool&&check_pool(pool),"Stage4 actual pool");json reports=json::array();int actual=0;bool insufficient=false;
 try{need(p.at("jobs").size()==16,"Stage4 all16 books");for(size_t bi=0;bi<16;++bi){auto&j=p.at("jobs")[bi];auto ids=j.at("joint_ids").get<std::vector<llama_token>>();
  if(ids.size()!=512){need(ids.size()<512,"Stage4 IDs length");insufficient=true;reports.push_back({{"book_id",j.at("book_id")},{"status","insufficient_window"},{"native_tokens",ids.size()},{"no_substitution",true}});continue;}
  auto invoke=[&](uint32_t mode,const char*key,bool keep){++actual;return early_run(m,pool,worker,mid,mode,p.at("run_id"),j.at("contexts").at(key),ids,keep);};
  EarlyResult source,B,C;json controls=nullptr;
  if(bi==0){auto nohook=invoke(0,"source_nohook",true);source=invoke(1,"source",true);quality_equal(nohook.quality,source.quality,false);auto copy=invoke(2,"copyoriginal",true);quality_equal(source.quality,copy.quality,true);B=invoke(9,"B64",true);auto bc=invoke(10,"copyB64",true);quality_equal(B.quality,bc.quality,true);C=invoke(11,"C64",true);auto cc=invoke(12,"copyC64",true);quality_equal(C.quality,cc.quality,true);
   controls={{"firstfixedbook",true},{"nohook_observer_fullvocab_memcmp",true},{"observer_copyoriginal_fullvocab_ownedtrace_memcmp",true},{"B64_COPY_B64_fullvocab_ownedtrace_memcmp",true},{"C64_COPY_C64_fullvocab_ownedtrace_memcmp",true}};
  }else{source=invoke(1,"source",false);B=invoke(9,"B64",false);C=invoke(11,"C64",false);}
  auto sb=invariant(source,B,mid==1?6:8),sc=invariant(source,C,mid==1?6:8);need(sb.at("passed")&&sc.at("passed"),"Stage4 earliest X/orderedIDs/finalactualalpha invariance failed");
  reports.push_back({{"book_id",j.at("book_id")},{"status","complete"},{"immutable_joint_ids",ids},{"runs",json::array({source.quality.report,B.quality.report,C.quality.report})},{"firstbook_controls",controls},{"source_vs_B64_invariant",sb},{"source_vs_C64_invariant",sc}});save(p.at("output"),{{"status","partial"},{"jobs",reports},{"actual_contexts",actual}});std::cerr<<"STAGE4_QUALITY completed book "<<j.at("book_id")<<" model "<<mid<<'\n';
 }
 worker.finish();ggml_threadpool_free(pool);llama_model_free(m);llama_backend_free();need(insufficient||actual==52,"Stage4 actual52 contexts");save(p.at("output"),{{"status","complete"},{"jobs",reports},{"actual_contexts",actual},{"insufficient_books",insufficient},{"TEST_opened",false}});return 0;
 }catch(...){ggml_threadpool_free(pool);llama_model_free(m);llama_backend_free();throw;}
}
static int generated(const std::string&python,const std::string&script){
 need(environment(),"Stage4 generated caller environment");auto pp=ggml_threadpool_params_default(8);pp.poll=0;auto*pool=ggml_threadpool_new(&pp);need(pool&&check_pool(pool),"Stage4 generated actualGGML8 RN");for(int i=0;i<8;++i)need(!check_pool(pool,i,0x2000u),"Stage4 MXCSR-only worker mutation rejected");ggml_threadpool_free(pool);
 for(int K:{6,8}){EarlyResult a,b;for(int q=0;q<4;++q){OwnedBatch x;x.layer=K==6?1:0;x.T=128;x.position=q*128;x.x.resize(128*2048);x.ids.resize(128*K);x.alpha.resize(128*K);for(int i=0;i<128*K;++i){x.ids[i]=i%64;x.alpha[i]=float(i%7)/7;}a.owned.push_back(x);}b=a;need(invariant(a,b,K).at("passed"),"Stage4 generated exact invariant");b.owned[2].x[7]=-0.f;need(!invariant(a,b,K).at("passed"),"signedzero X bit mutation detected");b=a;b.owned[1].ids[3]^=1;need(!invariant(a,b,K).at("passed"),"ordered IDs mutation detected");b=a;b.owned[3].alpha[9]=1.f;need(!invariant(a,b,K).at("passed"),"finalalpha mutation detected");
  std::vector<int32_t>ids(64*K);for(int t=0;t<64;++t)for(int k=0;k<K;++k)ids[t*K+k]=(t+k)%64;auto slots=mask(ids,64,K,all_experts());need(slots.size()==size_t(64*K),"all64 everyK slot eligible");std::set<int>seen;for(auto&v:slots)seen.insert(v.e);need(seen.size()==64,"all64 generated coverage");
  uint32_t mid=K==6?1:2;Worker worker({python,script,"--model",std::to_string(mid),"--generated"},mid);std::vector<float>x(64*2048);for(size_t i=0;i<x.size();++i)x[i]=float(i%17)/32.f;
  for(uint32_t mode:{9u,10u,11u,12u}){Header h;h.model=mid;h.layer=early_layer(mid);h.mode=h.arm=mode;h.run=1;h.context=mode;h.subrun=8;h.decode=1;h.T=64;h.K=K;auto reply=worker.compute(h,x,ids,slots);need(reply.size()==slots.size(),"newmodes all64 binary IPC");for(auto&v:reply)for(int j=0;j<2048;++j){float expected=x[v.t*2048+j]+float(v.e)/64.f;need(std::memcmp(&expected,&v.y[j],4)==0&&v.flags==0,"all64 slot/row binary values");}}
  worker.finish();
 }
 std::cout<<"{\"status\":\"passed\",\"generated_only\":true,\"all64_K6_K8\":true,\"512pos_literal_X_ID_finalalpha_and_mutations\":true,\"actual8GGMLworkers_RN\":true}\n";return 0;
}
int main(int argc,char**argv){try{signal(SIGPIPE,SIG_IGN);need(environment(),"Stage4 actual callingthread strict numeric environment");if(argc==4&&std::string(argv[1])=="synthetic")return generated(argv[2],argv[3]);need(argc==3,"Stage4 tokenize/quality approvedplan.json");std::ifstream f(argv[2]);need(bool(f),"Stage4 plan open");json p;f>>p;if(std::string(argv[1])=="tokenize")return tokenize_only(p);need(std::string(argv[1])=="quality","Stage4 explicit command");return quality_only(p);}catch(const std::exception&e){std::cerr<<"Stage4 harness technical-invalid: "<<e.what()<<'\n';return 1;}}
