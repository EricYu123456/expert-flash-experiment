// CPU-only DeepSeek shared-feature capture. Pinned llama.cpp remains unchanged.
// W8 reference: source GGUF -> complete-row symmetric W8, vector A8, exact
// integer dot, F32(sw*sx), F32(integer), F32 product. Rounding is std::round.
#include "llama.h"
#include "ggml-backend.h"
#include "gguf.h"
#include <algorithm>
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
extern "C" void ggml_vec_swiglu_f32(int, float *, const float *, const float *);
using Clock = std::chrono::steady_clock;
static void need(bool b,const std::string&s){if(!b)throw std::runtime_error(s);}
static constexpr int H=2048, I=1408, IS=2816, K=6;
static std::string root, mode;
static gguf_context *meta=nullptr;
static ggml_context *meta_tensors=nullptr;
static const char *mapping=nullptr;
static size_t mapping_size=0;
static int threads=8;
static std::map<int,std::vector<int>> selected;
static std::ofstream weight_ledger;
static uint64_t operator_calls=0,operator_values=0,fixed_overlap_checks=0;

static float quantize(const float *v,int n,int8_t *q){
 float amax=0;for(int k=0;k<n;++k){need(std::isfinite(v[k]),"Nonfinite input");amax=std::max(amax,std::abs(v[k]));}
 float s=amax==0?1.0f:amax/127.0f;need(s>0&&std::isfinite(s),"Invalid A8 scale");
 for(int k=0;k<n;++k)q[k]=(int8_t)std::clamp((int)std::round(v[k]/s),-127,127);
 return s;
}
static int64_t dot(const int8_t*w,const int8_t*x,int n){
 // All supported rows are <=10944, so even the sum of absolute products is
 // <=176515776. I32 SIMD accumulation is exact and equals the prior I64 dot.
 need(n<=10944,"I32 dot bound");int32_t sum=0;
 #pragma omp simd reduction(+:sum)
 for(int k=0;k<n;++k)sum+=(int32_t)w[k]*(int32_t)x[k];return sum;
}
static float scale_output(int64_t sum,float sw,float sx){const float s=sw*sx;const float f=(float)sum;return f*s;}
struct Weight{int C=0,R=0;std::vector<int8_t> q;std::vector<float> scales;};
static std::map<std::string,Weight> cache;
static const Weight& weight(const std::string&name,int e){
 std::string key=name+"/"+std::to_string(e);auto it=cache.find(key);if(it!=cache.end())return it->second;
 auto*w=ggml_get_tensor(meta_tensors,name.c_str());need(w,"Missing weight "+name);need(e>=0&&e<w->ne[2],"Expert index");
 Weight out;out.C=w->ne[0];out.R=w->ne[1];out.q.resize((size_t)out.C*out.R);out.scales.resize(out.R);
 int64_t id=gguf_find_tensor(meta,name.c_str());auto*traits=ggml_get_type_traits(w->type);need(id>=0&&traits->to_float,"GGUF dequantizer");
 size_t rb=ggml_row_size(w->type,out.C),offset=gguf_get_data_offset(meta)+gguf_get_tensor_offset(meta,id)+(size_t)e*out.R*rb;
 need(offset+(size_t)out.R*rb<=mapping_size,"Weight extent");
 #pragma omp parallel
 {
  std::vector<float> row(out.C);
  #pragma omp for schedule(static)
  for(int r=0;r<out.R;++r){traits->to_float(mapping+offset+(size_t)r*rb,row.data(),out.C);out.scales[r]=quantize(row.data(),out.C,out.q.data()+(size_t)r*out.C);}
 }
 weight_ledger<<"{\"name\":\""<<name<<"\",\"expert\":"<<e<<",\"rows\":"<<out.R<<",\"columns\":"<<out.C<<",\"source_type\":\""<<ggml_type_name(w->type)<<"\"}\n";
 return cache.emplace(key,std::move(out)).first->second;
}
static std::vector<float> matvecs(const Weight&w,const std::vector<float>&x,int T){
 need(x.size()==(size_t)T*w.C,"Matvec shape");std::vector<int8_t>qx(x.size());std::vector<float>sx(T),y((size_t)T*w.R);
 for(int t=0;t<T;++t)sx[t]=quantize(x.data()+(size_t)t*w.C,w.C,qx.data()+(size_t)t*w.C);
 #pragma omp parallel for schedule(static)
 for(int64_t j=0;j<(int64_t)T*w.R;++j){int r=j%w.R,t=j/w.R;y[j]=scale_output(dot(w.q.data()+(size_t)r*w.C,qx.data()+(size_t)t*w.C,w.C),w.scales[r],sx[t]);}
 return y;
}
static std::string wn(int l,const std::string&p){return "blk."+std::to_string(l)+".ffn_"+p+".weight";}
static std::vector<float> expert_forward(int l,int e,const std::vector<float>&x,int T){
 auto gate=matvecs(weight(wn(l,"gate_exps"),e),x,T),up=matvecs(weight(wn(l,"up_exps"),e),x,T);std::vector<float>z((size_t)T*I);
 for(int t=0;t<T;++t)ggml_vec_swiglu_f32(I,z.data()+(size_t)t*I,gate.data()+(size_t)t*I,up.data()+(size_t)t*I);
 return matvecs(weight(wn(l,"down_exps"),e),z,T);
}
static bool target(const ggml_tensor*t,int&l,std::string&p){
 if((t->op!=GGML_OP_MUL_MAT&&t->op!=GGML_OP_MUL_MAT_ID)||!t->src[0])return false;
 char role[64];if(std::sscanf(t->src[0]->name,"blk.%d.ffn_%63[^.]",&l,role)!=2)return false;p=role;
 bool routed=p=="gate_exps"||p=="up_exps"||p=="down_exps",shared=p=="gate_shexp"||p=="up_shexp"||p=="down_shexp",dense=p=="gate"||p=="up"||p=="down";
 if(!routed&&!shared&&!dense){need(p=="gate_inp","Uncovered FFN "+p);return false;}
 need((dense&&l==0)||((shared||routed)&&l>=1&&l<=26),"Raw layer mismatch");need(routed==(t->op==GGML_OP_MUL_MAT_ID),"Projection op mismatch");return true;
}
static void replace_operator(ggml_tensor*dst){
 auto*w=dst->src[0];auto*x=dst->src[1];auto*ids=dst->src[2];bool routed=dst->op==GGML_OP_MUL_MAT_ID;
 need(x->data&&dst->data&&x->type==GGML_TYPE_F32&&dst->type==GGML_TYPE_F32&&x->nb[0]==4&&dst->nb[0]==4,"CPU operator layout");
 int C=w->ne[0],R=w->ne[1],nk=routed?ids->ne[0]:1,T=routed?ids->ne[1]:x->ne[1],nv=routed?x->ne[1]:1;
 need(x->ne[0]==C&&x->ne[3]==1&&dst->ne[3]==1,"Projection dimensions");
 if(routed)need(ids->type==GGML_TYPE_I32&&x->ne[2]==T&&dst->ne[1]==nk&&dst->ne[2]==T,"Routed dimensions");
 std::vector<int8_t>q((size_t)T*nv*C);std::vector<float>sx((size_t)T*nv);std::vector<const Weight*>ws((size_t)T*nk);
 for(int t=0;t<T;++t)for(int v=0;v<nv;++v){const float*in=(const float*)((const char*)x->data+(routed?t*x->nb[2]+v*x->nb[1]:t*x->nb[1]));sx[(size_t)t*nv+v]=quantize(in,C,q.data()+((size_t)t*nv+v)*C);}
 for(int t=0;t<T;++t)for(int k=0;k<nk;++k){int32_t e=0;if(routed)std::memcpy(&e,(char*)ids->data+t*ids->nb[1]+k*ids->nb[0],4);ws[(size_t)t*nk+k]=&weight(w->name,e);}
 #pragma omp parallel for schedule(static)
 for(int64_t j=0;j<(int64_t)T*nk*R;++j){int r=j%R,k=(j/R)%nk,t=j/(R*nk);const auto&ww=*ws[(size_t)t*nk+k];size_t vi=(size_t)t*nv+k%nv;float y=scale_output(dot(ww.q.data()+(size_t)r*C,q.data()+vi*C,C),ww.scales[r],sx[vi]);std::memcpy((char*)dst->data+(routed?t*dst->nb[2]+k*dst->nb[1]:t*dst->nb[1])+r*dst->nb[0],&y,4);}
 ++operator_calls;operator_values+=(uint64_t)T*nk*R;
}
static std::vector<float> read_f32(const ggml_tensor*t,int R,int nk,int T,bool routed){
 need(t->type==GGML_TYPE_F32&&t->ne[0]==R&&t->ne[3]==1,"Capture tensor type");std::vector<float>v((size_t)T*nk*R);
 for(int a=0;a<T;++a)for(int k=0;k<nk;++k){const auto*p=(const char*)t->data+(routed?a*t->nb[2]+k*t->nb[1]:a*t->nb[1]);std::memcpy(v.data()+((size_t)a*nk+k)*R,p,(size_t)R*4);}
 for(float f:v)need(std::isfinite(f),"Nonfinite captured feature");return v;
}
struct Captured{int T=0;std::vector<float>x,z,shared,routed,alpha;std::vector<int32_t>ids;};
struct Trace{bool failed=false;std::string error;std::map<int,Captured> layers;};
static bool named(const char*n,const char*p,int&l){char tail;std::string f=std::string(p)+"-%d%c";return std::sscanf(n,f.c_str(),&l,&tail)==1&&selected.count(l);}
static bool callback(ggml_tensor*t,bool ask,void*opaque){auto&tr=*(Trace*)opaque;try{
 int l=-1;std::string p;bool op=target(t,l,p),capture=op&&selected.count(l);int al=-1;bool alpha=false;
 for(const char*n:{"ffn_moe_weights","ffn_moe_weights_softmax","ffn_moe_weights_norm","ffn_moe_weights_scaled"})if(named(t->name,n,al))alpha=true;
 if(ask)return (op&&mode=="w8")||capture||alpha;
 if(op&&mode=="w8")replace_operator(t);
 if(alpha){auto&c=tr.layers[al];int T=t->ne[0]==1?t->ne[2]:t->ne[1],nk=t->ne[0]==1?t->ne[1]:t->ne[0];need(nk==K&&t->type==GGML_TYPE_F32,"Router weights");c.alpha.resize((size_t)T*K);for(int a=0;a<T;++a)for(int k=0;k<K;++k){size_t off=t->ne[0]==1?a*t->nb[2]+k*t->nb[1]:a*t->nb[1]+k*t->nb[0];std::memcpy(&c.alpha[(size_t)a*K+k],(char*)t->data+off,4);}return true;}
 if(!capture)return true;auto&c=tr.layers[l];bool routed=t->op==GGML_OP_MUL_MAT_ID;int T=routed?t->src[2]->ne[1]:t->src[1]->ne[1];c.T=T;
 if(p=="up_exps"){
  need(c.x.empty(),"Duplicate layer x");c.x=read_f32(t->src[1],H,1,T,true);c.ids.resize((size_t)T*K);auto*ids=t->src[2];
  for(int a=0;a<T;++a)for(int k=0;k<K;++k)std::memcpy(&c.ids[(size_t)a*K+k],(char*)ids->data+a*ids->nb[1]+k*ids->nb[0],4);
 }else if(p=="down_exps")c.routed=read_f32(t,H,K,T,true);
 else if(p=="down_shexp"){c.z=read_f32(t->src[1],IS,1,T,false);c.shared=read_f32(t,H,1,T,false);}
 return true;
 }catch(const std::exception&e){tr.failed=true;tr.error=e.what();std::cerr<<"capture callback: "<<e.what()<<'\n';return false;}}
template<class T>static void append(std::ofstream&o,const std::vector<T>&v){o.write((const char*)v.data(),v.size()*sizeof(T));need(bool(o),"Array append");}
struct LayerFiles{uint64_t rows=0;std::ofstream x,z,shared,routed,fixed,ids,alpha,prompt_index,position,metadata;};
static std::map<int,LayerFiles> files;
static void open_files(){for(const auto&kv:selected){std::string p=root+"/layer"+std::to_string(kv.first);std::filesystem::create_directories(p);auto&f=files[kv.first];f.x.open(p+"/x.f32",std::ios::binary);f.z.open(p+"/shared_z.f32",std::ios::binary);f.shared.open(p+"/shared_output.f32",std::ios::binary);f.routed.open(p+"/routed_outputs.f32",std::ios::binary);f.fixed.open(p+"/expert_outputs.f32",std::ios::binary);f.ids.open(p+"/selected_experts.i32",std::ios::binary);f.alpha.open(p+"/routing_weights.f32",std::ios::binary);f.prompt_index.open(p+"/prompt_indices.i64",std::ios::binary);f.position.open(p+"/positions.i32",std::ios::binary);f.metadata.open(p+"/rows.jsonl");}}
static void emit(Trace&tr,int job_index,const std::string&id){
 need(tr.layers.size()==selected.size(),"Missing capture layers");
 for(const auto&kv:selected){int l=kv.first;auto&c=tr.layers.at(l);int T=c.T,E=kv.second.size();need(c.x.size()==(size_t)T*H&&c.z.size()==(size_t)T*IS&&c.shared.size()==(size_t)T*H&&c.routed.size()==(size_t)T*K*H&&c.alpha.size()==(size_t)T*K,"Capture closure");
  std::vector<float>fixed((size_t)T*E*H);
  need(mode=="w8"||E==0,"Off-route fixed targets require W8 mode");
  for(int j=0;j<E;++j){int e=kv.second[j];auto y=expert_forward(l,e,c.x,T);for(int a=0;a<T;++a){std::memcpy(fixed.data()+((size_t)a*E+j)*H,y.data()+(size_t)a*H,H*4);for(int k=0;k<K;++k)if(c.ids[(size_t)a*K+k]==e){need(std::memcmp(y.data()+(size_t)a*H,c.routed.data()+((size_t)a*K+k)*H,H*4)==0,"Fixed/off-route expert does not match graph native W8");++fixed_overlap_checks;}}}
  auto&f=files[l];append(f.x,c.x);append(f.z,c.z);append(f.shared,c.shared);append(f.routed,c.routed);append(f.fixed,fixed);append(f.ids,c.ids);append(f.alpha,c.alpha);
  std::vector<int64_t>pi(T,job_index);std::vector<int32_t>pos(T);for(int a=0;a<T;++a){pos[a]=a;f.metadata<<"{\"row\":"<<f.rows+a<<",\"prompt_index\":"<<job_index<<",\"prompt_id\":\""<<id<<"\",\"position\":"<<a<<"}\n";}append(f.prompt_index,pi);append(f.position,pos);f.rows+=T;
 }
 tr.layers.clear();
}
static void ids_json(std::ostream&o,const std::vector<llama_token>&v){o<<'[';for(size_t i=0;i<v.size();++i)o<<(i?",":"")<<v[i];o<<']';}
static void run(llama_model*m,const std::string&path,const std::string&id,int ji,int limit){
 std::ifstream in(path);need(bool(in),"Prompt read");std::string text((std::istreambuf_iterator<char>(in)),{});auto*v=llama_model_get_vocab(m);int n=-llama_tokenize(v,text.data(),text.size(),nullptr,0,true,true);need(n>1,"Prompt token count");std::vector<llama_token>tok(n);need(llama_tokenize(v,text.data(),text.size(),tok.data(),n,true,true)==n,"Tokenization");tok.resize(std::min(n,limit));n=tok.size();
 Trace tr;auto cp=llama_context_default_params();cp.n_ctx=std::max(256,((n+255)/256)*256);cp.n_batch=cp.n_ubatch=n;cp.n_threads=cp.n_threads_batch=threads;cp.n_seq_max=1;cp.cb_eval=callback;cp.cb_eval_user_data=&tr;auto*c=llama_init_from_model(m,cp);need(c,"Context init");auto b=llama_batch_init(n,0,1);b.n_tokens=n;for(int a=0;a<n;++a){b.token[a]=tok[a];b.pos[a]=a;b.n_seq_id[a]=1;b.seq_id[a][0]=0;b.logits[a]=true;}
 auto start=Clock::now();int rc=llama_decode(c,b);llama_synchronize(c);need(!tr.failed,tr.error);need(rc==0,"Teacher decode");emit(tr,ji,id);
 std::ofstream scores(root+"/scores/"+id+".jsonl");scores<<std::setprecision(17);int V=llama_vocab_n_tokens(v);double total=0;
 for(int a=0;a<n;++a){auto*p=llama_get_logits_ith(c,a);need(p,"Teacher logits");int top=0;for(int j=0;j<V;++j){need(std::isfinite(p[j]),"Finite logits");if(p[j]>p[top])top=j;}double sum=0;for(int j=0;j<V;++j)sum+=std::exp(double(p[j])-p[top]);double lse=p[top]+std::log(sum);if(a+1<n){double nll=lse-p[tok[a+1]];total+=nll;scores<<"{\"position\":"<<a<<",\"input_token_id\":"<<tok[a]<<",\"target_token_id\":"<<tok[a+1]<<",\"top1_token_id\":"<<top<<",\"nll_F64\":"<<nll<<"}\n";}}
 double seconds=std::chrono::duration<double>(Clock::now()-start).count();std::ofstream summary(root+"/scores/"+id+".summary.json");summary<<std::setprecision(17)<<"{\"prompt_id\":\""<<id<<"\",\"prompt_index\":"<<ji<<",\"tokens\":"<<n<<",\"scored_tokens\":"<<n-1<<",\"sum_nll_F64\":"<<total<<",\"mean_nll_F64\":"<<total/(n-1)<<",\"seconds\":"<<seconds<<",\"token_ids\":";ids_json(summary,tok);summary<<"}\n";
 std::cout<<"COMPLETE "<<ji<<' '<<id<<" tokens="<<n<<" seconds="<<seconds<<" cache_entries="<<cache.size()<<std::endl;llama_batch_free(b);llama_free(c);
}
static void manifest(int jobs,int limit){
 std::ofstream o(root+"/capture_manifest.json");o<<"{\"status\":\"complete\",\"mode\":\""<<mode<<"\",\"jobs\":"<<jobs<<",\"max_tokens\":"<<limit<<",\"hidden\":2048,\"shared_width\":2816,\"topk\":6,\"gpu_layers\":0,\"fixed_target_graph_overlap_bit_checks\":"<<fixed_overlap_checks<<",\"operator_calls\":"<<operator_calls<<",\"operator_values\":"<<operator_values<<",\"layers\":{";
 bool comma=false;for(const auto&kv:selected){if(comma)o<<',';comma=true;auto&f=files[kv.first];std::string p="layer"+std::to_string(kv.first)+"/";o<<'\"'<<kv.first<<"\":{\"rows\":"<<f.rows<<",\"expert_ids\":[";for(size_t j=0;j<kv.second.size();++j)o<<(j?",":"")<<kv.second[j];o<<"],\"arrays\":{";
  auto a=[&](const char*name,const char*file,const char*dtype,const std::string&shape,bool first=false){o<<(first?"":",")<<'\"'<<name<<"\":{\"path\":\""<<p<<file<<"\",\"dtype\":\""<<dtype<<"\",\"shape\":["<<f.rows<<shape<<"]}";};
  a("x","x.f32","float32",",2048",true);a("shared_z","shared_z.f32","float32",",2816");a("shared_output","shared_output.f32","float32",",2048");a("routed_outputs","routed_outputs.f32","float32",",6,2048");a("expert_outputs","expert_outputs.f32","float32",","+std::to_string(kv.second.size())+",2048");a("selected_experts","selected_experts.i32","int32",",6");a("routing_weights","routing_weights.f32","float32",",6");a("prompt_ids","prompt_indices.i64","int64","");a("positions","positions.i32","int32","");o<<"}}";
 }o<<"}}\n";
}
int main(int argc,char**argv){try{
 need(argc==8,"Usage: MODEL JOBS_TSV OUTDIR MAX_TOKENS THREADS w8|source-gguf LAYER:EXPERT,EXPERT;...");root=argv[3];int limit=std::stoi(argv[4]);threads=std::stoi(argv[5]);mode=argv[6];need(limit>=2&&limit<=512&&threads>=1&&threads<=64&&(mode=="w8"||mode=="source-gguf"),"Arguments");
 std::stringstream layers(argv[7]);std::string part;while(std::getline(layers,part,';')){auto colon=part.find(':');need(colon!=std::string::npos,"Layer specification");int l=std::stoi(part.substr(0,colon));need(l>=1&&l<=26&&!selected.count(l),"Target layer");auto&es=selected[l];std::stringstream experts(part.substr(colon+1));std::string e;while(std::getline(experts,e,',')){int ei=std::stoi(e);need(ei>=0&&ei<64&&std::find(es.begin(),es.end(),ei)==es.end(),"Target expert");es.push_back(ei);}}
 need(!selected.empty(),"No layers");need(!std::filesystem::exists(root+"/capture_manifest.json"),"Refuse completed capture overwrite");std::filesystem::create_directories(root+"/scores");omp_set_num_threads(threads);weight_ledger.open(root+"/weight_cache.jsonl");open_files();gguf_init_params gp{true,&meta_tensors};meta=gguf_init_from_file(argv[1],gp);need(meta&&meta_tensors,"GGUF metadata");int fd=open(argv[1],O_RDONLY);struct stat st;need(fd>=0&&fstat(fd,&st)==0,"GGUF open");mapping_size=st.st_size;mapping=(const char*)mmap(nullptr,mapping_size,PROT_READ,MAP_SHARED,fd,0);close(fd);need(mapping!=MAP_FAILED,"GGUF mmap");
 llama_backend_init();ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto*m=llama_model_load_from_file(argv[1],mp);need(m&&llama_model_n_embd(m)==H&&llama_model_n_layer(m)==27,"DeepSeek model dimensions");
 std::ifstream jobs(argv[2]);need(bool(jobs),"Jobs read");std::string line;int count=0;while(std::getline(jobs,line)){auto tab=line.find('\t');need(tab!=std::string::npos,"Jobs TSV");auto id=line.substr(tab+1);need(!id.empty()&&id.find_first_not_of("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-. ")==std::string::npos,"Job ID");run(m,line.substr(0,tab),id,count++,limit);weight_ledger.flush();}
 need(count>0,"No jobs");manifest(count,limit);llama_model_free(m);llama_backend_free();munmap((void*)mapping,mapping_size);gguf_free(meta);ggml_free(meta_tensors);return 0;
 }catch(const std::exception&e){std::cerr<<"shared-feature-capture: "<<e.what()<<'\n';return 1;}}
