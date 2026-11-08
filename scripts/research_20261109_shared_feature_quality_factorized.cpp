// Independent factorized-readout revision; source0/1/2 and native owners unchanged.
// 11/09 prototype/shared feature adapter; owner branches always remain native. Original/native FFN is still
// computed by this harness, so its wall time is not a compressed deployment.
// Reuse the new CPU numeric implementation; no historical file is modified.
#define main shared_feature_capture_entry_unused
#include "research_20261108_shared_feature_capture.cpp"
#undef main

#include "research_20261109_shared_feature_transform.hpp"
#include "research_20261109_factorized_readout.hpp"
#include "nlohmann/json.hpp"
struct Transform{int width=0,pool_width=1;bool uses_shared=false;std::string prefix;std::vector<uint8_t>sources,kernel_sources;std::vector<int32_t>indices,prototypes,kernel_indices,owners;std::vector<float>center,invscale;};
struct Readout{std::string basis;Weight w,v;int input_width=0;bool factorized=false;std::vector<float>bias,latent_center,latent_gain;};
static std::map<std::pair<int,std::string>,Transform> transforms;
static std::map<std::pair<int,int>,Readout> readouts;
static uint64_t total_factorized_rows=0,total_latent_values=0,total_factor_V_MAC=0,total_factor_U_MAC=0;
static uint64_t total_replaced=0,total_shared_checks=0,total_transform_values=0,total_new_A8_rows=0,total_proto_gu_rows=0,total_proto_overlap_floats=0,total_proto_cross_replacements=0;
struct Feature{std::vector<float>v,sx;std::vector<int8_t>q;std::vector<uint8_t>ready;};
struct NativePrototype{std::vector<float>raw;std::vector<uint8_t>ready;};
struct LiveLayer{int T=0;std::vector<float>x,z,shared;std::map<std::string,Feature>features;std::map<int,NativePrototype>prototypes;};
struct LiveTrace{bool failed=false;std::string error,id,phase;int step=-1;std::map<int,LiveLayer> live;std::ofstream routes;uint64_t factorized_rows=0,latent_values=0,factor_V_MAC=0,factor_U_MAC=0;uint64_t replaced=0,checks=0,proto_gu_rows=0,proto_overlap_floats=0,proto_cross_replacements=0,basis_A8_rows=0;};
template<class T>static std::vector<T> load_vector(const std::string&p,size_t n){std::ifstream f(p,std::ios::binary);need(bool(f),"Readout open "+p);std::vector<T>v(n);f.read((char*)v.data(),n*sizeof(T));need((size_t)f.gcount()==n*sizeof(T)&&f.peek()==EOF,"Readout bytes "+p);return v;}
static void load_readouts(const std::string&registry){
 if(registry=="-")return;std::ifstream f(registry);need(bool(f),"Readout registry");std::string line;
 while(std::getline(f,line)){if(line.empty()||line[0]=='#')continue;std::vector<std::string>col;std::stringstream s(line);std::string part;while(std::getline(s,part,'\t'))col.push_back(part);need(col.size()==6,"Registry needs layer, expert, basis_id, width, transform_prefix, C_prefix");
  int l=std::stoi(col[0]),e=std::stoi(col[1]),width=std::stoi(col[3]);need(l>=1&&l<=26&&e>=0&&e<64&&width>0&&width<=2816&&!readouts.count({l,e}),"Registry identity/dimensions");Readout r;r.basis=col[2];r.w.C=width;r.w.R=H;r.input_width=width;
  auto tk=std::make_pair(l,r.basis);auto ti=transforms.find(tk);
  if(ti==transforms.end()){Transform tr;tr.width=width;tr.prefix=col[4];tr.sources=load_vector<uint8_t>(tr.prefix+".sources.u8",width);tr.indices=load_vector<int32_t>(tr.prefix+".indices.i32",width);tr.center=load_vector<float>(tr.prefix+".center.f32",width);tr.invscale=load_vector<float>(tr.prefix+".inv_scale.f32",width);
   const std::string proto_path=tr.prefix+".prototypes.i32";tr.prototypes=std::filesystem::exists(proto_path)?load_vector<int32_t>(proto_path,width):std::vector<int32_t>(width,-1);
   std::set<int>owners;
   for(int j=0;j<width;++j){need(tr.sources[j]<=2&&tr.indices[j]>=0&&std::isfinite(tr.center[j])&&tr.invscale[j]>0&&std::isfinite(tr.invscale[j]),"Transform metadata");if(tr.sources[j]==0)need(tr.prototypes[j]==-1&&tr.indices[j]<H,"Raw x metadata");else if(tr.sources[j]==1){need(tr.prototypes[j]==-1&&tr.indices[j]<IS,"Shared z metadata");tr.uses_shared=true;}else{need(tr.prototypes[j]>=0&&tr.prototypes[j]<64&&tr.indices[j]<I,"Prototype metadata");owners.insert(tr.prototypes[j]);}}
   tr.owners.assign(owners.begin(),owners.end());tr.pool_width=(tr.uses_shared?IS:0)+(int)tr.owners.size()*I;tr.pool_width=std::max(tr.pool_width,1);tr.kernel_sources=tr.sources;tr.kernel_indices=tr.indices;
   for(int j=0;j<width;++j)if(tr.sources[j]==2){tr.kernel_sources[j]=1;auto pos=std::find(tr.owners.begin(),tr.owners.end(),tr.prototypes[j])-tr.owners.begin();tr.kernel_indices[j]=(tr.uses_shared?IS:0)+(int)pos*I+tr.indices[j];}
   transforms.emplace(tk,std::move(tr));
  }else need(ti->second.width==width&&ti->second.prefix==col[4],"Layer/basis transform must be identical across experts");
  const auto&transform=transforms.at(tk);need(std::find(transform.owners.begin(),transform.owners.end(),e)==transform.owners.end(),"Prototype owners must stay native; registry violates owner-excluded policy");
  std::string prefix=col[5];const std::string factor_path=prefix+".factor.json";
  if(std::filesystem::exists(factor_path)){
   std::ifstream jf(factor_path);need(bool(jf),"Factor descriptor open");nlohmann::json cfg;jf>>cfg;
   need(cfg.at("version").get<int>()==1&&cfg.at("H").get<int>()==H,"Factor descriptor version/H");
   int input=cfg.contains("input_width")?cfg.at("input_width").get<int>():cfg.at("width").get<int>();
   need(input==width&&(!cfg.contains("width")||cfg.at("width").get<int>()==input),"Factor input width");
   int rank=cfg.at("rank").get<int>();need(rank>0&&rank<=1408,"Factor rank");
   if(cfg.contains("input_transform_prefix"))need(cfg.at("input_transform_prefix").get<std::string>()==col[4],"Factor input transform differs from registry");
   if(cfg.contains("V_bias"))need(!cfg.at("V_bias").get<bool>(),"V bias is not supported");
   r.factorized=true;r.v.C=width;r.v.R=rank;r.w.C=rank;
   r.v.q=load_vector<int8_t>(prefix+".V.i8",(size_t)rank*width);r.v.scales=load_vector<float>(prefix+".V.scales.f32",rank);
   r.latent_center=load_vector<float>(prefix+".latent.center.f32",rank);r.latent_gain=load_vector<float>(prefix+".latent.inv_scale.f32",rank);
   r.w.q=load_vector<int8_t>(prefix+".U.i8",(size_t)H*rank);r.w.scales=load_vector<float>(prefix+".U.scales.f32",H);
   for(int j=0;j<rank;++j)need(r.v.scales[j]>0&&std::isfinite(r.v.scales[j])&&std::isfinite(r.latent_center[j])&&r.latent_gain[j]>0&&std::isfinite(r.latent_gain[j]),"Factor V/affine metadata");
  }else{r.w.q=load_vector<int8_t>(prefix+".i8",(size_t)H*width);r.w.scales=load_vector<float>(prefix+".scales.f32",H);}
  r.bias=load_vector<float>(prefix+".bias.f32",H);
  for(float v:r.w.scales)need(v>0&&std::isfinite(v),"Readout scale");for(float v:r.bias)need(std::isfinite(v),"Readout bias");readouts.emplace(std::make_pair(l,e),std::move(r));
 }
 need(!readouts.empty(),"Empty registry");
 // A hosted native GU already gives its owner's native phi. Keep every owner
 // of a resident prototype native, even if another private basis was proposed.
 std::map<int,std::set<int>>owner_union;for(const auto&kv:transforms)owner_union[kv.first.first].insert(kv.second.owners.begin(),kv.second.owners.end());
 for(const auto&kv:readouts)need(!owner_union[kv.first.first].count(kv.first.second),"Every active resident prototype owner must stay native");
}
static bool has_layer(int l){auto it=readouts.lower_bound({l,0});return it!=readouts.end()&&it->first.first==l;}
static void precompute_shared(int l,LiveLayer&c){
 auto gate=matvecs(weight(wn(l,"gate_shexp"),0),c.x,c.T),up=matvecs(weight(wn(l,"up_shexp"),0),c.x,c.T);c.z.resize((size_t)c.T*IS);
 for(int t=0;t<c.T;++t)ggml_vec_swiglu_f32(IS,c.z.data()+(size_t)t*IS,gate.data()+(size_t)t*IS,up.data()+(size_t)t*IS);
}
static void ensure_prototype(int l,int owner,const std::vector<int>&rows,LiveLayer&c,LiveTrace&trace){
 auto&proto=c.prototypes[owner];if(proto.ready.empty()){proto.ready.assign(c.T,0);proto.raw.resize((size_t)c.T*I);}
 std::vector<int>missing;for(int t:rows)if(!proto.ready[t])missing.push_back(t);if(missing.empty())return;
 int N=missing.size();std::vector<float>x((size_t)N*H);for(int n=0;n<N;++n)std::memcpy(x.data()+(size_t)n*H,c.x.data()+(size_t)missing[n]*H,H*4);
 auto gate=matvecs(weight(wn(l,"gate_exps"),owner),x,N),up=matvecs(weight(wn(l,"up_exps"),owner),x,N);std::vector<float>phi((size_t)N*I);
 for(int n=0;n<N;++n)ggml_vec_swiglu_f32(I,phi.data()+(size_t)n*I,gate.data()+(size_t)n*I,up.data()+(size_t)n*I);
 for(int n=0;n<N;++n){std::memcpy(proto.raw.data()+(size_t)missing[n]*I,phi.data()+(size_t)n*I,I*4);proto.ready[missing[n]]=1;}
 trace.proto_gu_rows+=N;total_proto_gu_rows+=N;
}
static const Feature& feature_for(int l,const std::string&basis,const std::vector<std::pair<int,int>>&locations,LiveLayer&c,LiveTrace&trace){
 const auto&transform=transforms.at({l,basis});auto&feature=c.features[basis];if(feature.ready.empty()){feature.ready.assign(c.T,0);feature.v.resize((size_t)c.T*transform.width);feature.q.resize(feature.v.size());feature.sx.resize(c.T);}
 std::vector<int>missing;for(auto loc:locations)if(!feature.ready[loc.first])missing.push_back(loc.first);if(missing.empty())return feature;
 for(int owner:transform.owners)ensure_prototype(l,owner,missing,c,trace);
 int N=missing.size();std::vector<float>x((size_t)N*H),pool((size_t)N*transform.pool_width,0),v((size_t)N*transform.width);
 for(int n=0;n<N;++n){int t=missing[n];std::memcpy(x.data()+(size_t)n*H,c.x.data()+(size_t)t*H,H*4);int offset=0;if(transform.uses_shared){std::memcpy(pool.data()+(size_t)n*transform.pool_width,c.z.data()+(size_t)t*IS,IS*4);offset=IS;}
  for(int owner:transform.owners){std::memcpy(pool.data()+(size_t)n*transform.pool_width+offset,c.prototypes.at(owner).raw.data()+(size_t)t*I,I*4);offset+=I;}}
 need(sf_apply_transform(N,H,transform.pool_width,transform.width,x.data(),pool.data(),transform.kernel_sources.data(),transform.kernel_indices.data(),transform.center.data(),transform.invscale.data(),v.data())==0,"Live prototype transform");
 for(int n=0;n<N;++n){int t=missing[n];std::memcpy(feature.v.data()+(size_t)t*transform.width,v.data()+(size_t)n*transform.width,transform.width*4);feature.sx[t]=quantize(v.data()+(size_t)n*transform.width,transform.width,feature.q.data()+(size_t)t*transform.width);feature.ready[t]=1;}
 total_transform_values+=v.size();total_new_A8_rows+=N;trace.basis_A8_rows+=N;return feature;
}
static void predict_selected(int l,int expert,const Readout&r,LiveLayer&c,ggml_tensor*dst,LiveTrace&tr){
 auto*ids=dst->src[2];std::vector<std::pair<int,int>>locations;
 for(int t=0;t<c.T;++t)for(int k=0;k<K;++k){int32_t e;std::memcpy(&e,(char*)ids->data+t*ids->nb[1]+k*ids->nb[0],4);if(e==expert)locations.emplace_back(t,k);}
 if(locations.empty())return;const auto&feature=feature_for(l,r.basis,locations,c,tr);int C=r.input_width;
 if(r.factorized){
  int N=locations.size(),Q=r.v.R;std::vector<int8_t>qinput((size_t)N*C);std::vector<float>sx(N),output((size_t)N*H);
  for(int n=0;n<N;++n){int t=locations[n].first;std::memcpy(qinput.data()+(size_t)n*C,feature.q.data()+(size_t)t*C,C);sx[n]=feature.sx[t];}
  need(sf_factorized_qinput(N,C,Q,H,qinput.data(),sx.data(),r.v.q.data(),r.v.scales.data(),r.latent_center.data(),r.latent_gain.data(),r.w.q.data(),r.w.scales.data(),r.bias.data(),output.data())==0,"Factorized live prediction");
  for(int n=0;n<N;++n){int t=locations[n].first,k=locations[n].second;std::memcpy((char*)dst->data+t*dst->nb[2]+k*dst->nb[1],output.data()+(size_t)n*H,H*4);}
  tr.factorized_rows+=N;total_factorized_rows+=N;tr.latent_values+=(uint64_t)N*Q;total_latent_values+=(uint64_t)N*Q;
  tr.factor_V_MAC+=(uint64_t)N*C*Q;total_factor_V_MAC+=(uint64_t)N*C*Q;tr.factor_U_MAC+=(uint64_t)N*H*Q;total_factor_U_MAC+=(uint64_t)N*H*Q;
 }else{
 // Exactly one canonical vector-A8 encoding per transformed input row, shared
 // only by readouts that declare this identical layer+basis transform.
 #pragma omp parallel for schedule(static)
 for(int64_t task=0;task<(int64_t)locations.size()*H;++task){int n=task/H,h=task%H,t=locations[n].first,k=locations[n].second;int32_t sum=0;const int8_t*q=feature.q.data()+(size_t)t*C;const int8_t*w=r.w.q.data()+(size_t)h*C;
  #pragma omp simd reduction(+:sum)
  for(int j=0;j<C;++j)sum+=(int32_t)q[j]*(int32_t)w[j];
  const float product=scale_output((int64_t)sum,r.w.scales[h],feature.sx[t]);const float y=product+r.bias[h];std::memcpy((char*)dst->data+t*dst->nb[2]+k*dst->nb[1]+h*dst->nb[0],&y,4);
 }
 }
 tr.replaced+=locations.size();total_replaced+=locations.size();if(!transforms.at({l,r.basis}).owners.empty()){tr.proto_cross_replacements+=locations.size();total_proto_cross_replacements+=locations.size();}
}
static bool live_callback(ggml_tensor*t,bool ask,void*opaque){auto&tr=*(LiveTrace*)opaque;try{
 int l=-1;std::string p;bool op=target(t,l,p);if(ask)return op;if(!op)return true;replace_operator(t);
 if(p=="up_exps"){
  auto*ids=t->src[2];int T=ids->ne[1];
  for(int a=0;a<T;++a){std::set<int>required;std::vector<int>eligible,factorized;tr.routes<<"{\"phase\":\""<<tr.phase<<"\",\"step\":"<<tr.step<<",\"token_in_batch\":"<<a<<",\"layer\":"<<l<<",\"expert_ids\":[";for(int k=0;k<K;++k){int32_t e;std::memcpy(&e,(char*)ids->data+a*ids->nb[1]+k*ids->nb[0],4);tr.routes<<(k?",":"")<<e;auto found=readouts.find({l,e});if(found!=readouts.end()){eligible.push_back(e);if(found->second.factorized)factorized.push_back(e);const auto&tf=transforms.at({l,found->second.basis});required.insert(tf.owners.begin(),tf.owners.end());}}
   tr.routes<<"],\"eligible_replacement_expert_ids\":[";for(size_t j=0;j<eligible.size();++j)tr.routes<<(j?",":"")<<eligible[j];tr.routes<<"],\"required_prototype_ids\":[";bool comma=false;for(int e:required){tr.routes<<(comma?",":"")<<e;comma=true;}tr.routes<<"],\"factorized_replacement_expert_ids\":[";for(size_t j=0;j<factorized.size();++j)tr.routes<<(j?",":"")<<factorized[j];tr.routes<<"]}\n";}
  if(has_layer(l)){auto&c=tr.live[l];need(c.x.empty(),"Duplicate live x");c.T=T;c.x=read_f32(t->src[1],H,1,T,true);precompute_shared(l,c);}
 }else if(p=="down_exps"&&has_layer(l)){
  auto&c=tr.live.at(l);auto*ids=t->src[2];need(ids->ne[1]==c.T,"Live token alignment");
  for(auto it=readouts.lower_bound({l,0});it!=readouts.end()&&it->first.first==l;++it)predict_selected(l,it->first.second,it->second,c,t,tr);
  // Owner branches remain native. This independently validates any hosted
  // prototype phi that coincides with a naturally selected owner/row.
  for(const auto&entry:c.prototypes){int owner=entry.first;const auto&proto=entry.second;for(int a=0;a<c.T;++a)if(proto.ready[a])for(int k=0;k<K;++k){int32_t e;std::memcpy(&e,(char*)ids->data+a*ids->nb[1]+k*ids->nb[0],4);if(e==owner){need(std::memcmp(proto.raw.data()+(size_t)a*I,(char*)t->src[1]->data+a*t->src[1]->nb[2]+k*t->src[1]->nb[1],I*4)==0,"Hosted prototype phi differs from native owner Down input");tr.proto_overlap_floats+=I;total_proto_overlap_floats+=I;}}}

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
 double seconds=std::chrono::duration<double>(Clock::now()-start).count();std::ofstream out(root+"/"+id+".summary.json");out<<std::setprecision(17)<<"{\"prompt_id\":\""<<id<<"\",\"prompt_index\":"<<ji<<",\"input_tokens\":"<<n<<",\"scored_tokens\":"<<n-1<<",\"decode_steps\":"<<steps<<",\"actual_generated_length\":"<<generated.size()<<",\"decoded_generated_inputs\":"<<decoded_inputs<<",\"stopped_eog\":"<<(stopped_eog?"true":"false")<<",\"sum_nll_F64\":"<<total<<",\"mean_nll_F64\":"<<total/(n-1)<<",\"replaced_expert_token_events\":"<<tr.replaced<<",\"native_shared_z_floats_bitchecked\":"<<tr.checks<<",\"prototype_GU_row_pairs_computed\":"<<tr.proto_gu_rows<<",\"prototype_owner_selected_phi_floats_bitchecked\":"<<tr.proto_overlap_floats<<",\"prototype_cross_replaced_events\":"<<tr.proto_cross_replacements<<",\"basis_A8_unique_rows\":"<<tr.basis_A8_rows<<",\"private_latent_A8_rows\":"<<tr.factorized_rows<<",\"factorized_latent_affine_values\":"<<tr.latent_values<<",\"factorized_V_coefficient_MAC\":"<<tr.factor_V_MAC<<",\"factorized_U_coefficient_MAC\":"<<tr.factor_U_MAC<<",\"seconds\":"<<seconds<<",\"token_ids\":";ids_json(out,tok);out<<",\"generated_token_ids\":";ids_json(out,generated);out<<"}\n";
 std::cout<<"QUALITY_COMPLETE "<<ji<<' '<<id<<" nll="<<total/(n-1)<<" replaced="<<tr.replaced<<" seconds="<<seconds<<std::endl;llama_batch_free(b);llama_free(c);
}
int main(int argc,char**argv){try{
 need(argc==9,"Usage: MODEL JOBS_TSV OUTDIR MAX_TOKENS THREADS AR_STEPS REGISTRY_TSV|- w8");root=argv[3];int limit=std::stoi(argv[4]),steps=std::stoi(argv[6]);threads=std::stoi(argv[5]);mode=argv[8];need(limit>=2&&limit<=512&&threads>=1&&threads<=64&&steps>=0&&steps<=128&&mode=="w8","Quality arguments");need(!std::filesystem::exists(root),"Refuse quality output overwrite");std::filesystem::create_directories(root);omp_set_num_threads(threads);load_readouts(argv[7]);weight_ledger.open(root+"/weight_cache.jsonl");gguf_init_params gp{true,&meta_tensors};meta=gguf_init_from_file(argv[1],gp);need(meta&&meta_tensors,"Quality GGUF metadata");int fd=open(argv[1],O_RDONLY);struct stat st;need(fd>=0&&fstat(fd,&st)==0,"Quality GGUF open");mapping_size=st.st_size;mapping=(const char*)mmap(nullptr,mapping_size,PROT_READ,MAP_SHARED,fd,0);close(fd);need(mapping!=MAP_FAILED,"Quality mmap");llama_backend_init();ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto*m=llama_model_load_from_file(argv[1],mp);need(m&&llama_model_n_embd(m)==H&&llama_model_n_layer(m)==27,"Quality DeepSeek dimensions");
 std::ifstream jobs(argv[2]);need(bool(jobs),"Quality jobs");std::string line;int count=0;while(std::getline(jobs,line)){auto tab=line.find('\t');need(tab!=std::string::npos,"Quality jobs TSV");quality_run(m,line.substr(0,tab),line.substr(tab+1),count++,limit,steps);weight_ledger.flush();}
 need(count>0,"Empty quality jobs");std::ofstream out(root+"/completion.json");out<<"{\"status\":\"complete\",\"jobs\":"<<count<<",\"readouts\":"<<readouts.size()<<",\"max_tokens\":"<<limit<<",\"decode_steps\":"<<steps<<",\"gpu_layers\":0,\"replaced_expert_token_events\":"<<total_replaced<<",\"native_shared_z_floats_bitchecked\":"<<total_shared_checks<<",\"transformed_feature_values\":"<<total_transform_values<<",\"new_basis_A8_rows\":"<<total_new_A8_rows<<",\"private_latent_A8_rows\":"<<total_factorized_rows<<",\"factorized_latent_affine_values\":"<<total_latent_values<<",\"factorized_V_coefficient_MAC\":"<<total_factor_V_MAC<<",\"factorized_U_coefficient_MAC\":"<<total_factor_U_MAC<<",\"transform_bases\":"<<transforms.size()<<",\"prototype_GU_row_pairs_computed\":"<<total_proto_gu_rows<<",\"prototype_owner_selected_phi_floats_bitchecked\":"<<total_proto_overlap_floats<<",\"prototype_cross_replaced_events\":"<<total_proto_cross_replacements<<",\"prototype_owners_kept_native\":true,\"harness_recomputes_native_FFN\":true}\n";llama_model_free(m);llama_backend_free();munmap((void*)mapping,mapping_size);gguf_free(meta);ggml_free(meta_tensors);return 0;
 }catch(const std::exception&e){std::cerr<<"shared-feature-quality: "<<e.what()<<'\n';return 1;}}
