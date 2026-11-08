// Functional live-replacement quality adapter. Original/native FFN is still
// computed by this harness, so its wall time is not a compressed deployment.
// Reuse the new CPU numeric implementation; no historical file is modified.
#define main shared_feature_capture_entry_unused
#include "research_20261108_shared_feature_capture.cpp"
#undef main

struct Readout{std::string basis;Weight w;std::vector<float>bias;};
static std::map<std::pair<int,int>,Readout> readouts;
static uint64_t total_replaced=0,total_shared_checks=0;
struct LiveLayer{int T=0;std::vector<float>x,z,shared;};
struct LiveTrace{bool failed=false;std::string error,id,phase;int step=-1;std::map<int,LiveLayer> live;std::ofstream routes;uint64_t replaced=0,checks=0;};
template<class T>static std::vector<T> load_vector(const std::string&p,size_t n){std::ifstream f(p,std::ios::binary);need(bool(f),"Readout open "+p);std::vector<T>v(n);f.read((char*)v.data(),n*sizeof(T));need((size_t)f.gcount()==n*sizeof(T)&&f.peek()==EOF,"Readout bytes "+p);return v;}
static void load_readouts(const std::string&registry){
 if(registry=="-")return;std::ifstream f(registry);need(bool(f),"Readout registry");std::string line;
 while(std::getline(f,line)){if(line.empty()||line[0]=='#')continue;std::vector<std::string>col;std::stringstream s(line);std::string part;while(std::getline(s,part,'\t'))col.push_back(part);need(col.size()==5,"Registry needs layer, expert, basis, width, prefix");
  int l=std::stoi(col[0]),e=std::stoi(col[1]),width=std::stoi(col[3]);need(l>=1&&l<=26&&e>=0&&e<64&&!readouts.count({l,e}),"Registry identity");Readout r;r.basis=col[2];r.w.C=width;r.w.R=H;
  need((r.basis=="shared_z"&&width==IS)||((r.basis=="x"||r.basis=="raw_x")&&width==H)||(r.basis=="shared_output"&&width==H)||(r.basis=="x_shared_output"&&width==2*H),"Readout feature dimension");
  std::string prefix=col[4];r.w.q=load_vector<int8_t>(prefix+".i8",(size_t)H*width);r.w.scales=load_vector<float>(prefix+".scales.f32",H);r.bias=load_vector<float>(prefix+".bias.f32",H);
  for(float v:r.w.scales)need(v>0&&std::isfinite(v),"Readout scale");for(float v:r.bias)need(std::isfinite(v),"Readout bias");readouts.emplace(std::make_pair(l,e),std::move(r));
 }
 need(!readouts.empty(),"Empty registry");
}
static bool has_layer(int l){auto it=readouts.lower_bound({l,0});return it!=readouts.end()&&it->first.first==l;}
static void precompute_shared(int l,LiveLayer&c){
 auto gate=matvecs(weight(wn(l,"gate_shexp"),0),c.x,c.T),up=matvecs(weight(wn(l,"up_shexp"),0),c.x,c.T);c.z.resize((size_t)c.T*IS);
 for(int t=0;t<c.T;++t)ggml_vec_swiglu_f32(IS,c.z.data()+(size_t)t*IS,gate.data()+(size_t)t*IS,up.data()+(size_t)t*IS);
 bool need_output=false;for(auto it=readouts.lower_bound({l,0});it!=readouts.end()&&it->first.first==l;++it)need_output|=it->second.basis=="shared_output"||it->second.basis=="x_shared_output";
 if(need_output)c.shared=matvecs(weight(wn(l,"down_shexp"),0),c.z,c.T);
}
static std::vector<float> predict(const Readout&r,const LiveLayer&c){
 const std::vector<float>*feature=&c.z;std::vector<float>joint;
 if(r.basis=="x"||r.basis=="raw_x")feature=&c.x;
 if(r.basis=="shared_output")feature=&c.shared;
 if(r.basis=="x_shared_output"){joint.resize((size_t)c.T*2*H);for(int t=0;t<c.T;++t){std::memcpy(joint.data()+(size_t)t*2*H,c.x.data()+(size_t)t*H,H*4);std::memcpy(joint.data()+(size_t)t*2*H+H,c.shared.data()+(size_t)t*H,H*4);}feature=&joint;}
 auto y=matvecs(r.w,*feature,c.T);for(int t=0;t<c.T;++t)for(int h=0;h<H;++h)y[(size_t)t*H+h]+=r.bias[h];return y;
}
static bool live_callback(ggml_tensor*t,bool ask,void*opaque){auto&tr=*(LiveTrace*)opaque;try{
 int l=-1;std::string p;bool op=target(t,l,p);if(ask)return op;if(!op)return true;replace_operator(t);
 if(p=="up_exps"){
  auto*ids=t->src[2];int T=ids->ne[1];
  for(int a=0;a<T;++a){tr.routes<<"{\"phase\":\""<<tr.phase<<"\",\"step\":"<<tr.step<<",\"token_in_batch\":"<<a<<",\"layer\":"<<l<<",\"expert_ids\":[";for(int k=0;k<K;++k){int32_t e;std::memcpy(&e,(char*)ids->data+a*ids->nb[1]+k*ids->nb[0],4);tr.routes<<(k?",":"")<<e;}tr.routes<<"]}\n";}
  if(has_layer(l)){auto&c=tr.live[l];need(c.x.empty(),"Duplicate live x");c.T=T;c.x=read_f32(t->src[1],H,1,T,true);precompute_shared(l,c);}
 }else if(p=="down_exps"&&has_layer(l)){
  auto&c=tr.live.at(l);auto*ids=t->src[2];need(ids->ne[1]==c.T,"Live token alignment");
  for(auto it=readouts.lower_bound({l,0});it!=readouts.end()&&it->first.first==l;++it){int expert=it->first.second;auto y=predict(it->second,c);for(int a=0;a<c.T;++a)for(int k=0;k<K;++k){int32_t e;std::memcpy(&e,(char*)ids->data+a*ids->nb[1]+k*ids->nb[0],4);if(e==expert){std::memcpy((char*)t->data+a*t->nb[2]+k*t->nb[1],y.data()+(size_t)a*H,H*4);++tr.replaced;++total_replaced;}}}
 }else if(p=="down_shexp"&&has_layer(l)){
  auto&c=tr.live.at(l);auto native_z=read_f32(t->src[1],IS,1,c.T,false);need(native_z.size()==c.z.size()&&std::memcmp(native_z.data(),c.z.data(),c.z.size()*4)==0,"Owned live shared z differs from native shared Down input");tr.checks+=c.z.size();total_shared_checks+=c.z.size();
  if(!c.shared.empty()){auto sy=read_f32(t,H,1,c.T,false);need(std::memcmp(sy.data(),c.shared.data(),sy.size()*4)==0,"Owned live shared output differs");}tr.live.erase(l);
 }
 return true;
 }catch(const std::exception&e){tr.failed=true;tr.error=e.what();std::cerr<<"quality callback: "<<e.what()<<'\n';return false;}}
static int greedy_checked(llama_context*c,int index,int V){auto*p=llama_get_logits_ith(c,index);need(p,"Quality logits");int top=0;for(int j=0;j<V;++j){need(std::isfinite(p[j]),"Nonfinite quality logits");if(p[j]>p[top])top=j;}return top;}
static void quality_run(llama_model*m,const std::string&path,const std::string&id,int ji,int limit,int steps){
 std::ifstream in(path);need(bool(in),"Quality prompt");std::string text((std::istreambuf_iterator<char>(in)),{});auto*v=llama_model_get_vocab(m);int n=-llama_tokenize(v,text.data(),text.size(),nullptr,0,true,true);need(n>1,"Quality tokenizer count");std::vector<llama_token>tok(n);need(llama_tokenize(v,text.data(),text.size(),tok.data(),n,true,true)==n,"Quality tokenizer");tok.resize(std::min(n,limit));n=tok.size();int V=llama_vocab_n_tokens(v);
 LiveTrace tr;tr.id=id;tr.phase="prefill";tr.routes.open(root+"/"+id+".routing.jsonl");auto cp=llama_context_default_params();cp.n_ctx=std::max(256,((n+steps+255)/256)*256);cp.n_batch=cp.n_ubatch=std::max(n,1);cp.n_threads=cp.n_threads_batch=threads;cp.n_seq_max=1;cp.cb_eval=live_callback;cp.cb_eval_user_data=&tr;auto*c=llama_init_from_model(m,cp);need(c,"Quality context");auto b=llama_batch_init(std::max(n,1),0,1);b.n_tokens=n;for(int a=0;a<n;++a){b.token[a]=tok[a];b.pos[a]=a;b.n_seq_id[a]=1;b.seq_id[a][0]=0;b.logits[a]=true;}
 auto decode=[&](){int rc=llama_decode(c,b);llama_synchronize(c);need(!tr.failed,tr.error);need(rc==0&&tr.live.empty(),"Quality decode closure");};auto start=Clock::now();decode();
 std::ofstream score(root+"/"+id+".positions.jsonl");score<<std::setprecision(17);double total=0;
 for(int a=0;a<n;++a){int top=greedy_checked(c,a,V);auto*p=llama_get_logits_ith(c,a);double sum=0;for(int j=0;j<V;++j)sum+=std::exp(double(p[j])-p[top]);double lse=p[top]+std::log(sum);if(a+1<n){double nll=lse-p[tok[a+1]];total+=nll;score<<"{\"position\":"<<a<<",\"input_token_id\":"<<tok[a]<<",\"target_token_id\":"<<tok[a+1]<<",\"top1_token_id\":"<<top<<",\"nll_F64\":"<<nll<<"}\n";}}
 int next=greedy_checked(c,-1,V);std::vector<llama_token>generated;tr.phase="decode";bool stopped_eog=false;int decoded_inputs=0;
 for(int a=0;a<steps;++a){generated.push_back(next);if(llama_vocab_is_eog(v,next)){stopped_eog=true;break;}tr.step=a;b.n_tokens=1;b.token[0]=next;b.pos[0]=n+a;b.n_seq_id[0]=1;b.seq_id[0][0]=0;b.logits[0]=true;decode();++decoded_inputs;next=greedy_checked(c,-1,V);}
 double seconds=std::chrono::duration<double>(Clock::now()-start).count();std::ofstream out(root+"/"+id+".summary.json");out<<std::setprecision(17)<<"{\"prompt_id\":\""<<id<<"\",\"prompt_index\":"<<ji<<",\"input_tokens\":"<<n<<",\"scored_tokens\":"<<n-1<<",\"decode_steps\":"<<steps<<",\"actual_generated_length\":"<<generated.size()<<",\"decoded_generated_inputs\":"<<decoded_inputs<<",\"stopped_eog\":"<<(stopped_eog?"true":"false")<<",\"sum_nll_F64\":"<<total<<",\"mean_nll_F64\":"<<total/(n-1)<<",\"replaced_expert_token_events\":"<<tr.replaced<<",\"native_shared_z_floats_bitchecked\":"<<tr.checks<<",\"seconds\":"<<seconds<<",\"token_ids\":";ids_json(out,tok);out<<",\"generated_token_ids\":";ids_json(out,generated);out<<"}\n";
 std::cout<<"QUALITY_COMPLETE "<<ji<<' '<<id<<" nll="<<total/(n-1)<<" replaced="<<tr.replaced<<" seconds="<<seconds<<std::endl;llama_batch_free(b);llama_free(c);
}
int main(int argc,char**argv){try{
 need(argc==9,"Usage: MODEL JOBS_TSV OUTDIR MAX_TOKENS THREADS AR_STEPS REGISTRY_TSV|- w8");root=argv[3];int limit=std::stoi(argv[4]),steps=std::stoi(argv[6]);threads=std::stoi(argv[5]);mode=argv[8];need(limit>=2&&limit<=512&&threads>=1&&threads<=64&&steps>=0&&steps<=128&&mode=="w8","Quality arguments");need(!std::filesystem::exists(root),"Refuse quality output overwrite");std::filesystem::create_directories(root);omp_set_num_threads(threads);load_readouts(argv[7]);weight_ledger.open(root+"/weight_cache.jsonl");gguf_init_params gp{true,&meta_tensors};meta=gguf_init_from_file(argv[1],gp);need(meta&&meta_tensors,"Quality GGUF metadata");int fd=open(argv[1],O_RDONLY);struct stat st;need(fd>=0&&fstat(fd,&st)==0,"Quality GGUF open");mapping_size=st.st_size;mapping=(const char*)mmap(nullptr,mapping_size,PROT_READ,MAP_SHARED,fd,0);close(fd);need(mapping!=MAP_FAILED,"Quality mmap");llama_backend_init();ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto*m=llama_model_load_from_file(argv[1],mp);need(m&&llama_model_n_embd(m)==H&&llama_model_n_layer(m)==27,"Quality DeepSeek dimensions");
 std::ifstream jobs(argv[2]);need(bool(jobs),"Quality jobs");std::string line;int count=0;while(std::getline(jobs,line)){auto tab=line.find('\t');need(tab!=std::string::npos,"Quality jobs TSV");quality_run(m,line.substr(0,tab),line.substr(tab+1),count++,limit,steps);weight_ledger.flush();}
 need(count>0,"Empty quality jobs");std::ofstream out(root+"/completion.json");out<<"{\"status\":\"complete\",\"jobs\":"<<count<<",\"readouts\":"<<readouts.size()<<",\"max_tokens\":"<<limit<<",\"decode_steps\":"<<steps<<",\"gpu_layers\":0,\"replaced_expert_token_events\":"<<total_replaced<<",\"native_shared_z_floats_bitchecked\":"<<total_shared_checks<<",\"harness_recomputes_native_FFN\":true}\n";llama_model_free(m);llama_backend_free();munmap((void*)mapping,mapping_size);gguf_free(meta);ggml_free(meta_tensors);return 0;
 }catch(const std::exception&e){std::cerr<<"shared-feature-quality: "<<e.what()<<'\n';return 1;}}
