// Full routed + shared FFN CPU functional reference; upstream llama.cpp is unchanged.
// Numeric contract is deliberately fixed and common to both physical placements.
// This is not original-GGUF numerical equivalence or a native AiF output ABI.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
using Clock=std::chrono::steady_clock;
static constexpr uint64_t H0=14695981039346656037ULL;
static uint64_t hash_bytes(uint64_t h,const void *v,size_t n){const auto*p=(const unsigned char*)v;for(size_t i=0;i<n;++i){h^=p[i];h*=1099511628211ULL;}return h;}
static std::string mode,root,current_job,current_phase;
static int current_step=-1;
static gguf_context *gguf=nullptr;
static const char *mapping=nullptr;
static size_t mapping_size=0;
static std::ofstream ledger,cache_ledger;
static FILE *reference_outputs=nullptr,*reference_logits=nullptr;
static uint64_t cross_output_floats=0,cross_output_differences=0,cross_logits_floats=0,cross_logits_differences=0;
static uint64_t integer_checks=0,integer_differences=0,scaled_differences=0;
struct Plan {int total_rows=0,columns=0;std::array<std::array<int,16>,64> rows{};};
static std::map<std::string,Plan> plans;
struct PackedChip {std::vector<int8_t> bytes;std::vector<float> scales;};
static std::array<PackedChip,16> chips;
struct RowAddress {int chip=0;size_t byte=0,scale=0;int logical=0;};
struct Cache {
 std::string name;int expert=0,C=0,R=0;size_t source_offset=0;
 std::vector<int8_t> canonical;std::vector<float> scales;std::vector<RowAddress> addresses;
};
static std::map<std::string,Cache> caches;
static float quantize(const float *v,int n,int8_t *q){
 float amax=0;for(int k=0;k<n;++k){if(!std::isfinite(v[k]))throw std::runtime_error("Nonfinite input");amax=std::max(amax,std::abs(v[k]));}
 float s=amax==0?1.0f:amax/127.0f;if(!(s>0)||!std::isfinite(s))throw std::runtime_error("Invalid quantization scale");
 for(int k=0;k<n;++k)q[k]=(int8_t)std::clamp((int)std::round(v[k]/s),-127,127);return s;
}
static int64_t dot(const int8_t *w,const int8_t *x,int n){
 int64_t sum=0;
 #pragma omp simd reduction(+:sum)
 for(int k=0;k<n;++k)sum+=(int32_t)w[k]*(int32_t)x[k];return sum;
}
static float scale_output(int64_t sum,float sw,float sx){const float s=sw*sx;const float f=(float)sum;return f*s;}
static bool equal_bits(float a,float b){uint32_t x,y;std::memcpy(&x,&a,4);std::memcpy(&y,&b,4);return x==y;}
static bool target(const ggml_tensor*t,int &layer,std::string &projection){
 if((t->op!=GGML_OP_MUL_MAT&&t->op!=GGML_OP_MUL_MAT_ID)||!t->src[0])return false;
 char p[64];int end=0;if(std::sscanf(t->src[0]->name,"blk.%d.ffn_%63[^.]%n",&layer,p,&end)!=2)return false;
 projection=p;
 const bool routed=projection=="gate_exps"||projection=="up_exps"||projection=="down_exps";
 const bool shared=projection=="gate_shexp"||projection=="up_shexp"||projection=="down_shexp";
 if(!routed&&!shared){if(projection!="gate_inp")throw std::runtime_error("Uncovered FFN matrix "+std::string(t->src[0]->name));return false;}
 if(layer<0||layer>=40||(routed!=(t->op==GGML_OP_MUL_MAT_ID)))throw std::runtime_error("FFN op/layer mismatch");return true;
}
static const Cache &get_cache(const ggml_tensor*w,int expert,int layer,const std::string &projection){
 const std::string key=std::string(w->name)+"/"+std::to_string(expert);auto it=caches.find(key);if(it!=caches.end())return it->second;
 Cache c;c.name=w->name;c.expert=expert;c.C=w->ne[0];c.R=w->ne[1];
 int64_t tid=gguf_find_tensor(gguf,w->name);if(tid<0||gguf_get_tensor_type(gguf,tid)!=w->type)throw std::runtime_error("GGUF weight mismatch");
 const auto *traits=ggml_get_type_traits(w->type);if(!traits->to_float)throw std::runtime_error("Missing to_float");
 size_t rb=ggml_row_size(w->type,c.C);c.source_offset=gguf_get_data_offset(gguf)+gguf_get_tensor_offset(gguf,tid)+expert*c.R*rb;
 if(c.source_offset+c.R*rb>mapping_size)throw std::runtime_error("Weight out of range");
 c.canonical.resize((size_t)c.R*c.C);c.scales.resize(c.R);std::vector<float> row(c.C);
 for(int r=0;r<c.R;++r){traits->to_float(mapping+c.source_offset+r*rb,row.data(),c.C);c.scales[r]=quantize(row.data(),c.C,c.canonical.data()+(size_t)r*c.C);}
 std::array<std::vector<int>,16> chip_rows;
 const bool shared=projection.find("shexp")!=std::string::npos;
 if(shared){if(expert!=0||c.R%16)throw std::runtime_error("Shared fullrow partition invalid");for(int r=0;r<c.R;++r)chip_rows[r/(c.R/16)].push_back(r);}
 else {
  const std::string phase=projection=="down_exps"?"down":"gate_up";const auto&p=plans.at(std::to_string(layer)+"/"+phase);
  const int base=projection=="up_exps"?512:0;
  if(c.C!=p.columns||(phase=="down"?c.R!=p.total_rows:c.R!=512))throw std::runtime_error("Frozen plan shape mismatch");
  int start=0;for(int chip=0;chip<16;++chip){int end=start+p.rows[expert][chip];for(int logical=std::max(start,base);logical<std::min(end,base+c.R);++logical)chip_rows[chip].push_back(logical-base);start=end;}
  if(start!=p.total_rows)throw std::runtime_error("Frozen expert row sum mismatch");
 }
 std::vector<int> seen(c.R,0);uint64_t physical_hash=H0;
 for(int chip=0;chip<16;++chip){
  // Physical slots within every chip run in reverse logical-row order. Computing
  // packed slots and scattering is observably different from canonical indexing.
  std::reverse(chip_rows[chip].begin(),chip_rows[chip].end());
  for(int r:chip_rows[chip]){auto &pc=chips[chip];RowAddress a{chip,pc.bytes.size(),pc.scales.size(),r};
   pc.bytes.insert(pc.bytes.end(),c.canonical.data()+(size_t)r*c.C,c.canonical.data()+(size_t)(r+1)*c.C);pc.scales.push_back(c.scales[r]);
   c.addresses.push_back(a);++seen[r];physical_hash=hash_bytes(physical_hash,&chip,sizeof(chip));physical_hash=hash_bytes(physical_hash,&r,sizeof(r));
  }
 }
 if(c.addresses.size()!=(size_t)c.R||std::any_of(seen.begin(),seen.end(),[](int n){return n!=1;}))throw std::runtime_error("Physical fullrow bijection failed");
 const std::string scale_path=root+"/scales/"+key.substr(0,key.find('/'))+".expert"+std::to_string(expert)+".f32";
 std::ofstream sf(scale_path,std::ios::binary);sf.write((const char*)c.scales.data(),c.scales.size()*4);if(!sf)throw std::runtime_error("Scale write failed");
 cache_ledger<<"{\"weight\":\""<<w->name<<"\",\"expert\":"<<expert<<",\"columns\":"<<c.C<<",\"rows\":"<<c.R
 <<",\"source_type\":\""<<ggml_type_name(w->type)<<"\",\"source_byte_offset\":"<<c.source_offset
 <<",\"canonical_bytes_fnv1a64\":\""<<hash_bytes(H0,c.canonical.data(),c.canonical.size())<<"\",\"row_scales_fnv1a64\":\""<<hash_bytes(H0,c.scales.data(),c.scales.size()*4)
 <<"\",\"physical_logical_map_fnv1a64\":\""<<physical_hash<<"\",\"shared_layout\":"<<(shared?"true":"false")<<",\"chip_rows\":[";
 for(int chip=0;chip<16;++chip)cache_ledger<<(chip?",":"")<<chip_rows[chip].size();cache_ledger<<"]}\n";
 return caches.emplace(key,std::move(c)).first->second;
}
struct Stats {uint64_t callbacks=0,tokens=0,outputs=0,input_vectors=0;};
static std::map<std::string,Stats> coverage;
static void shape(std::ostream&o,const ggml_tensor*t){o<<'[';for(int i=0;i<4;++i)o<<(i?",":"")<<t->ne[i];o<<']';}
static void audit(ggml_tensor*dst,int layer,const std::string &projection){
 const auto*w=dst->src[0],*x=dst->src[1],*ids=dst->src[2];const bool routed=dst->op==GGML_OP_MUL_MAT_ID;
 if(!x->data||!dst->data||x->type!=GGML_TYPE_F32||dst->type!=GGML_TYPE_F32||x->ne[0]!=w->ne[0]||x->nb[0]!=4||dst->nb[0]!=4||x->ne[3]!=1||w->ne[3]!=1)throw std::runtime_error("Unsupported FFN tensor layout");
 int C=w->ne[0],R=w->ne[1],K=routed?ids->ne[0]:1,T=routed?ids->ne[1]:x->ne[1],V=routed?x->ne[1]:1;
 if(routed){if(!ids->data||ids->type!=GGML_TYPE_I32||x->ne[2]!=T||dst->ne[0]!=R||dst->ne[1]!=K||dst->ne[2]!=T)throw std::runtime_error("Unsupported routed layout");}
 else if(x->ne[2]!=1||dst->ne[0]!=R||dst->ne[1]!=T||dst->ne[2]!=1||w->ne[2]!=1)throw std::runtime_error("Unsupported shared layout");
 const size_t nvec=(size_t)V*T;std::vector<int8_t> q(nvec*C);std::vector<float> sx(nvec);
 uint64_t ih=H0;for(int t=0;t<T;++t)for(int v=0;v<V;++v){size_t i=(size_t)t*V+v;const float*src=(const float*)((const char*)x->data+(routed?t*x->nb[2]+v*x->nb[1]:t*x->nb[1]));sx[i]=quantize(src,C,q.data()+i*C);ih=hash_bytes(ih,q.data()+i*C,C);ih=hash_bytes(ih,&sx[i],4);}
 std::vector<const Cache*>selected((size_t)T*K);std::vector<int32_t> expert_ids((size_t)T*K);
 for(int t=0;t<T;++t)for(int k=0;k<K;++k){int32_t e=0;if(routed)std::memcpy(&e,(const char*)ids->data+t*ids->nb[1]+k*ids->nb[0],4);if(e<0||e>=w->ne[2])throw std::runtime_error("Invalid expert");expert_ids[(size_t)t*K+k]=e;selected[(size_t)t*K+k]=&get_cache(w,e,layer,projection);}
 size_t count=(size_t)T*K*R;std::vector<float> out(count);uint64_t local_int=0,local_scaled=0;
 // Every placed result is read from the packed per-chip buffers and scattered.
 // A direct canonical I64 dot independently checks every placed integer result.
 #pragma omp parallel for schedule(static) reduction(+:local_int,local_scaled)
 for(int64_t task=0;task<(int64_t)T*K*R;++task){int r=task%R,k=(task/R)%K,t=task/(R*K);const auto&c=*selected[(size_t)t*K+k];size_t vi=(size_t)t*V+k%V;const auto*input=q.data()+vi*C;
  if(mode=="unsharded"&&projection.find("shexp")==std::string::npos){int64_t s=dot(c.canonical.data()+(size_t)r*C,input,C);out[task]=scale_output(s,c.scales[r],sx[vi]);}
  else {const auto&a=c.addresses[r];const auto&chip=chips[a.chip];int64_t placed=dot(chip.bytes.data()+a.byte,input,C);int64_t ref=dot(c.canonical.data()+(size_t)a.logical*C,input,C);local_int+=placed!=ref;
   float p=scale_output(placed,chip.scales[a.scale],sx[vi]);float u=scale_output(ref,c.scales[a.logical],sx[vi]);local_scaled+=!equal_bits(p,u);out[((size_t)t*K+k)*R+a.logical]=p;}
 }
 if(mode=="placed"){integer_checks+=count;integer_differences+=local_int;scaled_differences+=local_scaled;}
 uint64_t cross=0;if(mode=="unsharded"){if(std::fwrite(out.data(),4,count,reference_outputs)!=count)throw std::runtime_error("Reference output write failed");}
 else {std::vector<float> ref(count);if(std::fread(ref.data(),4,count,reference_outputs)!=count)throw std::runtime_error("Reference output read failed");for(size_t i=0;i<count;++i)cross+=!equal_bits(out[i],ref[i]);cross_output_floats+=count;cross_output_differences+=cross;}
 for(int t=0;t<T;++t)for(int k=0;k<K;++k)for(int r=0;r<R;++r){float *p=(float*)((char*)dst->data+(routed?t*dst->nb[2]+k*dst->nb[1]:t*dst->nb[1])+r*dst->nb[0]);*p=out[((size_t)t*K+k)*R+r];}
 auto&st=coverage[mode+"/"+current_phase+"/"+std::string(w->name)];++st.callbacks;st.tokens+=T;st.outputs+=count;st.input_vectors+=nvec;
 ledger<<"{\"mode\":\""<<mode<<"\",\"job\":\""<<current_job<<"\",\"phase\":\""<<current_phase<<"\",\"step\":"<<current_step<<",\"weight\":\""<<w->name<<"\",\"layer\":"<<layer<<",\"projection\":\""<<projection
 <<"\",\"op\":\""<<(routed?"MUL_MAT_ID":"MUL_MAT")<<"\",\"weight_shape\":";shape(ledger,w);ledger<<",\"input_shape\":";shape(ledger,x);ledger<<",\"output_shape\":";shape(ledger,dst);
 ledger<<",\"tokens\":"<<T<<",\"selected_experts\":"<<K<<",\"output_floats\":"<<count<<",\"input_fnv1a64\":\""<<ih<<"\",\"expert_ids_fnv1a64\":\""<<hash_bytes(H0,expert_ids.data(),expert_ids.size()*4)
 <<"\",\"output_fnv1a64\":\""<<hash_bytes(H0,out.data(),count*4)<<"\",\"integer_checks\":"<<(mode=="placed"?count:0)<<",\"integer_mismatches\":"<<local_int<<",\"scaled_bit_mismatches\":"<<local_scaled<<",\"cross_run_output_bit_mismatches\":"<<cross<<"}\n";
 if(local_int||local_scaled||cross)throw std::runtime_error("Physical placement equivalence failed");
}

static uint64_t finite_ffn_values=0,finite_vocab_values=0;
static std::map<std::string,uint64_t> training_logit_hash;
static void verify_training_ffn_finite(const ggml_tensor*t){
 if(t->type!=GGML_TYPE_F32||t->ne[3]!=1)throw std::runtime_error("Unsupported training FFN output");
 for(int64_t z=0;z<t->ne[2];++z)for(int64_t y=0;y<t->ne[1];++y)for(int64_t r=0;r<t->ne[0];++r){float f;std::memcpy(&f,(const char*)t->data+z*t->nb[2]+y*t->nb[1]+r*t->nb[0],4);if(!std::isfinite(f))throw std::runtime_error("Nonfinite training FFN output");++finite_ffn_values;}
}

struct Trace {FILE*route=nullptr;bool failed=false;std::vector<int>positions;std::vector<llama_token>tokens;};
static bool callback(ggml_tensor*t,bool ask,void*opaque){auto&s=*(Trace*)opaque;try{
 int layer=-1;std::string p;if(target(t,layer,p)){if(ask)return true;audit(t,layer,p);verify_training_ffn_finite(t);return true;}
 char tail;bool route=std::sscanf(t->name,"ffn_moe_topk-%d%c",&layer,&tail)==1;if(ask)return route;if(!route)return true;
 if(t->type!=GGML_TYPE_I32||t->ne[1]!=(int64_t)s.tokens.size()||t->ne[2]!=1||t->ne[3]!=1)throw std::runtime_error("Routing shape mismatch");
 std::vector<unsigned char>data(ggml_nbytes(t));ggml_backend_tensor_get(t,data.data(),0,data.size());
 for(size_t i=0;i<s.tokens.size();++i){std::fprintf(s.route,"{\"phase\":\"%s\",\"step\":%d,\"position\":%d,\"token_id\":%d,\"layer\":%d,\"expert_ids\":[",current_phase.c_str(),current_step,s.positions[i],s.tokens[i],layer);
 for(int64_t k=0;k<t->ne[0];++k){int32_t e;std::memcpy(&e,data.data()+i*t->nb[1]+k*t->nb[0],4);std::fprintf(s.route,"%s%d",k?",":"",e);}std::fprintf(s.route,"]}\n");}
 return true;
 }catch(const std::exception&e){std::fprintf(stderr,"Full FFN callback failure: %s\n",e.what());s.failed=true;return false;}}
static void ids_json(std::ostream&o,const std::vector<llama_token>&v){o<<'[';for(size_t i=0;i<v.size();++i)o<<(i?",":"")<<v[i];o<<']';}
static std::string piece(const llama_vocab*v,llama_token t){std::vector<char>b(128);int n=llama_token_to_piece(v,t,b.data(),b.size(),0,true);if(n<0){b.resize(-n);n=llama_token_to_piece(v,t,b.data(),b.size(),0,true);}if(n<0)throw std::runtime_error("Piece conversion failed");return {b.data(),(size_t)n};}
static void check_logits(llama_context*ctx,int vocab){
 const float*p=llama_get_logits_ith(ctx,-1);if(!p)throw std::runtime_error("Missing training continuation logits");
 for(int i=0;i<vocab;++i){if(!std::isfinite(p[i]))throw std::runtime_error("Nonfinite training continuation logits");++finite_vocab_values;}
 auto it=training_logit_hash.find(current_job);if(it==training_logit_hash.end())it=training_logit_hash.emplace(current_job,H0).first;
 it->second=hash_bytes(it->second,p,(size_t)vocab*4);
 if(current_phase=="prefill"||(current_phase=="decode"&&current_step==0)){
  std::string path=root+"/snapshots/"+current_job+"."+current_phase+"."+std::to_string(current_step)+".logits.f32";
  std::ofstream f(path,std::ios::binary);f.write((const char*)p,(size_t)vocab*4);if(!f)throw std::runtime_error("Training logit snapshot write");
 }
}
static void run(llama_model*model,const std::string&prompt_path,const std::string&id,int steps,int threads){
 current_job=id;std::ifstream f(prompt_path);if(!f)throw std::runtime_error("Prompt read failed");std::string prompt((std::istreambuf_iterator<char>(f)),{});const auto*vocab=llama_model_get_vocab(model);
 int n=-llama_tokenize(vocab,prompt.data(),prompt.size(),nullptr,0,true,true);if(n<=0)throw std::runtime_error("Tokenization failed");std::vector<llama_token>tokens(n),generated;
 if(llama_tokenize(vocab,prompt.data(),prompt.size(),tokens.data(),n,true,true)!=n)throw std::runtime_error("Tokenization mismatch");
 std::string stem=root+"/traces/"+id+"."+mode;Trace trace;trace.route=std::fopen((stem+".routing.jsonl").c_str(),"w");if(!trace.route)throw std::runtime_error("Trace open failed");
 auto params=llama_context_default_params();params.n_ctx=((n+steps+255)/256)*256;params.n_batch=params.n_ubatch=512;params.n_threads=params.n_threads_batch=threads;params.n_seq_max=1;params.no_perf=false;params.cb_eval=callback;params.cb_eval_user_data=&trace;
 llama_context*ctx=llama_init_from_model(model,params);if(!ctx)throw std::runtime_error("Context init failed");llama_batch b=llama_batch_init(params.n_batch,0,1);auto*sampler=llama_sampler_init_greedy();
 auto add=[&](llama_token tok,int pos,bool logits){int i=b.n_tokens++;b.token[i]=tok;b.pos[i]=pos;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=logits;trace.tokens.push_back(tok);trace.positions.push_back(pos);};
 auto clear=[&](){b.n_tokens=0;trace.tokens.clear();trace.positions.clear();};
 const auto start=Clock::now();current_phase="prefill";current_step=-1;
 for(int off=0;off<n;off+=params.n_batch){clear();int count=std::min((int)params.n_batch,n-off);for(int k=0;k<count;++k)add(tokens[off+k],off+k,true);if(llama_decode(ctx,b)||trace.failed)throw std::runtime_error("Full FFN prefill failed");llama_synchronize(ctx);if(trace.failed)throw std::runtime_error("Prefill callback failed");}
 check_logits(ctx,llama_vocab_n_tokens(vocab));llama_token next=llama_sampler_sample(sampler,ctx,-1);double prefill=std::chrono::duration<double>(Clock::now()-start).count();
 std::cout<<"Completed prefill job="<<id<<" mode="<<mode<<" tokens="<<n<<" seconds="<<prefill<<" caches="<<caches.size()<<std::endl;
 const auto ds=Clock::now();std::string text;current_phase="decode";
 for(int step=0;step<steps;++step){current_step=step;generated.push_back(next);text+=piece(vocab,next);clear();add(next,n+step,true);if(llama_decode(ctx,b)||trace.failed)throw std::runtime_error("Full FFN decode failed");llama_synchronize(ctx);if(trace.failed)throw std::runtime_error("Decode callback failed");check_logits(ctx,llama_vocab_n_tokens(vocab));next=llama_sampler_sample(sampler,ctx,-1);}
 double decode=std::chrono::duration<double>(Clock::now()-ds).count();std::fclose(trace.route);std::ofstream output(stem+".output.txt");output<<text;
 std::ofstream summary(stem+".summary.json");summary<<std::setprecision(17)<<"{\"prompt_tokens\":"<<n<<",\"decoded_tokens\":"<<steps<<",\"threads\":"<<threads<<",\"prefill_seconds\":"<<prefill<<",\"decode_seconds\":"<<decode<<",\"prefill_replaced\":true,\"prefill_state_reused\":false,\"prompt_token_ids\":";ids_json(summary,tokens);summary<<",\"generated_token_ids\":";ids_json(summary,generated);summary<<"}\n";
 llama_perf_context_print(ctx);llama_sampler_free(sampler);llama_batch_free(b);llama_free(ctx);std::cout<<"Completed decode job="<<id<<" mode="<<mode<<" steps="<<steps<<" seconds="<<decode<<std::endl;
}
static void load_plan(const std::string&path){std::ifstream f(path);std::string line;while(std::getline(f,line)){std::istringstream s(line);int layer,expert;std::string phase;Plan p;if(!(s>>layer>>phase>>expert>>p.total_rows>>p.columns))throw std::runtime_error("Plan TSV invalid");auto&dst=plans[std::to_string(layer)+"/"+phase];if(expert<0||expert>=64)throw std::runtime_error("Plan expert invalid");dst.total_rows=p.total_rows;dst.columns=p.columns;for(int c=0;c<16;++c)if(!(s>>dst.rows[expert][c]))throw std::runtime_error("Plan chip count missing");}if(plans.size()!=80)throw std::runtime_error("Expected 80 frozen phase plans");}

int main(int argc,char**argv){if(argc!=7){std::fprintf(stderr,"Training capture arguments\n");return 2;}try{
 int steps=std::stoi(argv[3]),threads=std::stoi(argv[4]);if(steps!=16||threads!=8)throw std::runtime_error("Fixed training capture steps/threads");root=argv[6];
 std::filesystem::create_directories(root+"/scales");std::filesystem::create_directories(root+"/traces");std::filesystem::create_directories(root+"/snapshots");load_plan(argv[5]);omp_set_num_threads(threads);
 ledger.open(root+"/coverage_ledger.jsonl");cache_ledger.open(root+"/weight_cache_ledger.jsonl");
 gguf_init_params gp{true,nullptr};gguf=gguf_init_from_file(argv[1],gp);if(!gguf)throw std::runtime_error("GGUF metadata");int fd=open(argv[1],O_RDONLY);struct stat st;if(fd<0||fstat(fd,&st))throw std::runtime_error("Model open");mapping_size=st.st_size;void*m=mmap(nullptr,mapping_size,PROT_READ,MAP_SHARED,fd,0);close(fd);if(m==MAP_FAILED)throw std::runtime_error("Model mmap");mapping=(const char*)m;
 llama_backend_init();ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto*model=llama_model_load_from_file(argv[1],mp);if(!model)throw std::runtime_error("Model loading");
 mode="unsharded";reference_outputs=std::fopen("/dev/null","wb");if(!reference_outputs)throw std::runtime_error("Capture FFN sink");
 std::ifstream f(argv[2]);std::string line;int njobs=0;while(std::getline(f,line)){auto tab=line.find('\t');if(tab==std::string::npos)throw std::runtime_error("Training TSV");run(model,line.substr(0,tab),line.substr(tab+1),steps,threads);ledger.flush();cache_ledger.flush();++njobs;}
 if(njobs!=8)throw std::runtime_error("Expected8 training prompts per batch");std::fclose(reference_outputs);
 std::ofstream report(root+"/capture_report.json");report<<"{\"status\":\"complete\",\"model\":\"granite4tiny\",\"mode\":\"canonical_only\",\"new_capture_routed_rows_paired_with_placed\":false,\"jobs\":"<<njobs<<",\"decode_steps\":16,\"full_FFN_outputs_all_tokens_finite\":true,\"finite_FFN_values_checked\":"<<finite_ffn_values<<",\"continuation_logit_values_checked_finite\":"<<finite_vocab_values<<",\"cache_entries\":"<<caches.size()<<",\"prompt_logit_stream_hashes\":{";
 bool comma=false;for(const auto&e:training_logit_hash){if(comma)report<<',';comma=true;report<<'\"'<<e.first<<"\":\""<<e.second<<'\"';}report<<"}}\n";report.close();ledger.close();cache_ledger.close();llama_model_free(model);llama_backend_free();munmap((void*)mapping,mapping_size);gguf_free(gguf);return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"w8-training-capture: %s\n",e.what());return 1;}}
