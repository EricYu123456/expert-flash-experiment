// Round 1 bounded Stage0. See the independently frozen JSON protocol.
// CPU canonical row-W8/dynamic-A8 reference; original non-FFN GGUF graph.
// No physical placement, AiF timing, original-checkpoint equivalence or quality claim.
#include "llama.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "gguf.h"
#include <algorithm>
#include <array>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <numeric>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <omp.h>
#include <xmmintrin.h>
// Original C linkage in ggml-cpu/vec.h. Link the pinned library unchanged.
extern "C" void ggml_vec_silu_f32(int n, float * y, const float * x);
extern "C" void ggml_vec_swiglu_f32(int n, float * y, const float * x, const float * g);
using U128=unsigned __int128; using I128=__int128;
using Clock=std::chrono::steady_clock;
static void require(bool yes,const char *s){if(!yes)throw std::runtime_error(s);}
static void require(bool yes,const std::string &s){if(!yes)throw std::runtime_error(s);}
static constexpr uint64_t FNV0=14695981039346656037ULL;
static uint64_t hash(uint64_t h,const void *v,size_t n){auto*p=(const unsigned char*)v;while(n--){h^=*p++;h*=1099511628211ULL;}return h;}
static uint32_t bits(float f){uint32_t u;std::memcpy(&u,&f,4);return u;}
static bool eq(float a,float b){return bits(a)==bits(b);}
static uint64_t abs64(int64_t x){return x<0?(uint64_t)(-(x+1))+1:(uint64_t)x;}
static constexpr U128 IMAX=(U128(1)<<127)-1;
static bool mul128(U128 a,U128 b,U128 &r){if(b&&a>IMAX/b)return false;r=a*b;return true;}
// sqrtl supplies only an estimate; division-based corrections make this exact.
static uint64_t isqrt(U128 n){
 if(!n)return 0;long double f=std::sqrt((long double)n);
 uint64_t r=f>=(long double)UINT64_MAX?UINT64_MAX:(uint64_t)f;
 while(U128(r)>n/r)--r;
 while(r<UINT64_MAX&&U128(r+1)<=n/(r+1))++r;
 return r;
}
static I128 floor_div(I128 a,I128 b){require(b>0,"floor divisor");I128 q=a/b,r=a%b;return q-(r<0);}
static I128 ceil_div(I128 a,I128 b){require(b>0,"ceil divisor");I128 q=a/b,r=a%b;return q+(r>0);}
static uint64_t global_bound(uint64_t W,uint64_t X){U128 p;require(mul128(W,X,p),"global overflow");return isqrt(p);}
static uint64_t conditional_bound(uint64_t G,uint64_t V,int64_t C,uint64_t X,int64_t d,uint64_t fallback,bool &valid){
 valid=false;if(!G)return fallback;
 U128 gv,gx,cc,dd,p;
 if(!mul128(G,V,gv)||!mul128(G,X,gx)||!mul128(abs64(C),abs64(C),cc)||!mul128(abs64(d),abs64(d),dd)||gv<cc||gx<dd||!mul128(gv-cc,gx-dd,p))return fallback;
 const uint64_t s=isqrt(p);U128 acd;if(!mul128(abs64(C),abs64(d),acd)||acd>IMAX-s)return fallback;
 const I128 cd=(C<0)!=(d<0)?-I128(acd):I128(acd);
 const I128 lo=ceil_div(cd-I128(s),G),hi=floor_div(cd+I128(s),G);
 if(lo>hi)return fallback;
 U128 a=lo<0?U128(-lo):U128(lo),b=hi<0?U128(-hi):U128(hi),m=std::max(a,b);
 if(m>UINT64_MAX)return fallback;valid=true;return std::min(fallback,(uint64_t)m);
}
struct Quant{float amax=0,scale=1;std::vector<int8_t> q;};
static Quant quantize(const float*v,int n,int qmax=127){
 Quant a;a.q.resize(n);for(int j=0;j<n;++j){require(std::isfinite(v[j]),"nonfinite quantizer input");a.amax=std::max(a.amax,std::abs(v[j]));}
 a.scale=a.amax==0?1.f:a.amax/float(qmax);require(std::isfinite(a.scale)&&a.scale>0,"underflow/nonfinite quantizer scale");
 for(int j=0;j<n;++j)a.q[j]=(int8_t)std::clamp((int)std::round(v[j]/a.scale),-qmax,qmax);return a;
}
static int64_t dot(const int8_t*w,const int8_t*x,int n){int64_t s=0;
 #pragma omp simd reduction(+:s)
 for(int j=0;j<n;++j)s+=(int32_t)w[j]*(int32_t)x[j];return s;}
static uint64_t norm(const int8_t*x,int n){return (uint64_t)dot(x,x,n);}
static float scale_output(uint64_t b,float sw,float sx){const float s=sw*sx,f=(float)b;return f*s;}
static float signed_output(int64_t b,float sw,float sx){const float s=sw*sx,f=(float)b;return f*s;}
static bool certificate(float h,uint64_t B,float sw,float sx,float L,int qmax){
 const float U=scale_output(B,sw,sx),Ba=std::abs(h)*U,sL=L/float(qmax);
 return std::isfinite(U)&&U>=0&&std::isfinite(Ba)&&std::isfinite(sL)&&sL>0&&Ba<L&&(Ba/sL)<.5f;
}
static std::array<uint64_t,5> pages(const std::vector<uint8_t>&base,const std::vector<uint8_t>&cand,int H,int unit){
 uint64_t nb=0,nc=0,added=0,lost=0;const uint64_t bytes=base.size()*(uint64_t)H;
 for(uint64_t off=0;off<bytes;off+=unit){int first=off/H,last=std::min<uint64_t>(base.size()-1,(std::min(bytes,off+unit)-1)/H);bool b=true,c=true;for(int r=first;r<=last;++r){b&=base[r];c&=cand[r];}nb+=b;nc+=c;added+=c&&!b;lost+=b&&!c;}
 return {nb,nc,added,lost,(bytes+unit-1)/unit};
}
static void synthetic(){
 require(std::fegetround()==FE_TONEAREST,"requires nearest F32 rounding");require((_mm_getcsr()&((1<<15)|(1<<6)))==0,"FTZ/DAZ unsupported");
 for(uint64_t i=0;i<10000;++i){U128 n=U128(i)*i;require(isqrt(n)==i,"isqrt square");if(i)require(isqrt(n-1)==i-1,"isqrt minus1");require(isqrt(n+2*i)==i,"isqrt upper");}
 require(isqrt(IMAX)==13043817825332782212ULL,"isqrt 127bit");U128 z;require(!mul128(IMAX,2,z)&&mul128(IMAX,1,z),"overflow sentinel");
 for(int a=-11;a<=11;++a)for(int b=1;b<=7;++b){require(floor_div(a,b)==(I128)std::floor(double(a)/b),"floor signed");require(ceil_div(a,b)==(I128)std::ceil(double(a)/b),"ceil signed");}
 uint64_t checks=0;for(int g0=-2;g0<=2;++g0)for(int g1=-2;g1<=2;++g1)for(int v0=-2;v0<=2;++v0)for(int v1=-2;v1<=2;++v1)for(int x0=-2;x0<=2;++x0)for(int x1=-2;x1<=2;++x1){int8_t g[2]={(int8_t)g0,(int8_t)g1},v[2]={(int8_t)v0,(int8_t)v1},x[2]={(int8_t)x0,(int8_t)x1};auto G=norm(g,2),V=norm(v,2),X=norm(x,2),b=global_bound(V,X);bool ok;auto c=conditional_bound(G,V,dot(g,v,2),X,dot(g,x,2),b,ok);require(c>=abs64(dot(v,x,2))&&c<=b,"conditional exhaustive");++checks;}
 bool valid;require(conditional_bound(1,1,2,1,0,4,valid)==4&&!valid,"invalid Gram fallback");
 require(std::round(.5f)==1&&std::round(-.5f)==-1,"ties away");
 require(!certificate(1,1,1,1,254,127),"reject exact half");require(certificate(1,1,1,1,std::nextafter(254.f,INFINITY),127),"accept below half");
 require(!certificate(1,1,1,1,0,127)&&!certificate(1,1,INFINITY,1,1000,127),"zero/nonfinite fallback");
 for(int n:{1024,1408,2816,10944}){std::vector<float>g(n),u(n),h(n),a(n);for(int j=0;j<n;++j){g[j]=(j%101-50)*.25f;u[j]=(j%79-39)*.125f;}ggml_vec_silu_f32(n,h.data(),g.data());ggml_vec_swiglu_f32(n,a.data(),g.data(),u.data());for(int j=0;j<n;++j)require(eq(h[j]*u[j],a[j]),"full-I activation product");}
 std::vector<uint8_t>b(9,1),c=b;for(int i=0;i<8;++i)c[i]=0;auto p=pages(b,c,2048,16384);require(p[0]==2&&p[1]==1&&p[3]==1,"seed paid including padding");
 b={1,0,1};c={1,1,1};p=pages(b,c,3000,4096);require(p[0]==1&&p[1]==3&&p[2]==2,"cross-row page");
 std::cout<<"{\"synthetic\":\"passed\",\"conditional_exhaustive_cases\":"<<checks<<",\"full_I\":[1024,1408,2816,10944],\"rounding\":"<<std::fegetround()<<",\"mxcsr\":"<<_mm_getcsr()<<"}\n";
}
struct Cache{std::string name;int expert=0,C=0,R=0;std::vector<int8_t>w;std::vector<float>s;std::vector<uint64_t>n,blocks;std::vector<int64_t>cross;std::vector<uint64_t>gn;};
static std::map<std::string,Cache> caches;
static gguf_context*gf=nullptr;static const char*mapping=nullptr;static size_t mapping_size=0;
static std::string root,modelname,job,phase;static int jobindex=-1,step=-1,model_layers=0;
static std::ofstream weightlog,coverage_log,dispatch_log,fitlog,inputlog,auditlog;static std::ofstream inputbin;
static uint64_t product_checks=0,integer_bound_checks=0,finite_outputs=0,finite_logits=0,invariant_checks=0,conditional_fallbacks=0;
static uint64_t activation_mismatches=0;
static uint64_t primary_row_evaluations=0,block_isqrt_evaluations=0,deflation_coordinate_products=0,conditional_gram_dot_products=0;
static std::map<std::string,int> prompt_counts;
static std::map<std::tuple<std::string,int,std::string>,std::array<uint64_t,6>> prefill_diagnostics;
static std::map<std::string,std::array<uint64_t,3>> coverage;
static Cache& cache(const ggml_tensor*w,int e){
 std::string key=std::string(w->name)+"/"+std::to_string(e);auto it=caches.find(key);if(it!=caches.end())return it->second;
 Cache c;c.name=w->name;c.expert=e;c.C=w->ne[0];c.R=w->ne[1];int64_t tid=gguf_find_tensor(gf,w->name);require(tid>=0&&gguf_get_tensor_type(gf,tid)==w->type,"GGUF tensor identity");
 auto*traits=ggml_get_type_traits(w->type);require(traits->to_float,"missing dequantizer");size_t rb=ggml_row_size(w->type,c.C),off=gguf_get_data_offset(gf)+gguf_get_tensor_offset(gf,tid)+e*c.R*rb;require(off+c.R*rb<=mapping_size,"weight range");
 c.w.resize((size_t)c.C*c.R);c.s.resize(c.R);c.n.resize(c.R);const int N=(c.C+127)/128;c.blocks.resize((size_t)c.R*N);std::vector<float> row(c.C);
 for(int r=0;r<c.R;++r){traits->to_float(mapping+off+r*rb,row.data(),c.C);auto q=quantize(row.data(),c.C);c.s[r]=q.scale;std::copy(q.q.begin(),q.q.end(),c.w.begin()+(size_t)r*c.C);c.n[r]=norm(q.q.data(),c.C);for(int b=0;b<N;++b)c.blocks[(size_t)r*N+b]=norm(q.q.data()+128*b,std::min(128,c.C-128*b));}
 weightlog<<"{\"weight\":\""<<c.name<<"\",\"expert\":"<<e<<",\"columns\":"<<c.C<<",\"rows\":"<<c.R<<",\"source_type\":\""<<ggml_type_name(w->type)<<"\",\"source_offset\":"<<off<<",\"w8_fnv1a64\":\""<<hash(FNV0,c.w.data(),c.w.size())<<"\",\"scales_fnv1a64\":\""<<hash(FNV0,c.s.data(),c.s.size()*4)<<"\",\"global_norm_bytes\":"<<c.n.size()*8<<",\"block128_norm_bytes\":"<<c.blocks.size()*8<<"}\n";
 return caches.emplace(key,std::move(c)).first->second;
}
struct Target{int layer=-1;std::string proj,kind,role;bool routed=false;};
static bool target(const ggml_tensor*t,Target&a){
 if((t->op!=GGML_OP_MUL_MAT&&t->op!=GGML_OP_MUL_MAT_ID)||!t->src[0])return false;
 char p[80];if(std::sscanf(t->src[0]->name,"blk.%d.ffn_%79[^.]",&a.layer,p)!=2)return false;a.proj=p;
 if(a.proj=="gate_inp")return false;
 a.routed=a.proj.find("_exps")!=std::string::npos;a.kind=a.routed?"routed":a.proj.find("_shexp")!=std::string::npos?"shared":"dense";
 a.role=a.proj.substr(0,a.proj.find('_'));require(a.role=="gate"||a.role=="up"||a.role=="down","uncovered FFN matrix "+std::string(t->src[0]->name));
 require(a.routed==(t->op==GGML_OP_MUL_MAT_ID),"FFN op/role mismatch");require(a.layer>=0&&a.layer<model_layers,"FFN layer range");
 const auto*w=t->src[0];int I=a.kind=="dense"?10944:a.kind=="shared"?2816:modelname=="deepseek-v2-lite"?1408:1024;
 require((a.role=="down"?w->ne[0]:w->ne[1])==I&&(a.role=="down"?w->ne[1]:w->ne[0])==2048&&w->ne[2]==(a.routed?64:1)&&w->ne[3]==1,"unsupported FFN dimensions");
 if(modelname=="deepseek-v2-lite")require((a.layer==0)==(a.kind=="dense"),"Deep dense0/remaining MoE coverage");else require(a.kind=="routed","OLMoE all-routed coverage");return true;
}
struct Record{int layer=0,token=0,e=0;std::string kind,job,phase;int ji=0,step=0;Cache*up=nullptr,*gate=nullptr;std::vector<int8_t>qx;float sx=0;std::vector<float>g,u,h,a;std::vector<int64_t>gd,ud;bool haveg=false,haveu=false;};
using PairKey=std::tuple<int,std::string,int,int>;
static std::map<PairKey,Record> pending;static std::map<std::pair<int,int>,Quant> common;
static std::map<int,std::vector<uint64_t>> energy;static std::map<int,std::vector<int>> coords;static bool fit_frozen=false;
static std::map<int,int> fit_input_counts;
static std::vector<Record> fit_records;
static void common_input(int layer,int token,const Quant&q){
 auto key=std::make_pair(layer,token);auto it=common.find(key);if(it!=common.end()){require(it->second.q==q.q&&eq(it->second.scale,q.scale),"common gate/up/shared qx mismatch");return;}
 common.emplace(key,q);
 if(phase=="decode"){
  if(jobindex<4){++fit_input_counts[layer];auto&en=energy[layer];if(en.empty())en.resize(q.q.size());require(en.size()==q.q.size(),"fit shape");for(size_t j=0;j<en.size();++j)en[j]+=(int)q.q[j]*(int)q.q[j];}
  auto offset=inputbin.tellp();inputbin.write((const char*)q.q.data(),q.q.size());inputlog<<"{\"job\":\""<<job<<"\",\"step\":"<<step<<",\"layer\":"<<layer<<",\"offset\":"<<offset<<",\"count\":"<<q.q.size()<<",\"sx_bits\":"<<bits(q.scale)<<",\"qx_fnv1a64\":\""<<hash(FNV0,q.q.data(),q.q.size())<<"\"}\n";
 }
}
static constexpr int NM=21;
using Metrics=std::array<uint64_t,NM>;
static const char*metric_names="dispatches,rows,up_bytes,h0_rows,true_zero_rows,extra_rows,candidate_rows,seed_rows,seed_lost_h0,base4,cand4,added4,lost4,total4,base16,cand16,added16,lost16,total16,invariant_rows,conditional_fallback_rows";
static const std::array<std::string,6> boundnames={"global","block128","conditional","primary_rowmin","deflation16","all_rowmin"};
using AggregateKey=std::tuple<std::string,std::string,int,std::string,int,int,int>; // job,phase,layer,class,bits,witness,bound
static std::map<AggregateKey,Metrics> aggregated;
static void score(const Record&r,bool deflation){
 const int I=r.a.size(),H=r.qx.size(),N=(H+127)/128;auto&up=*r.up;auto&gate=*r.gate;
 require(I==up.R&&H==up.C&&gate.R==I&&gate.C==H,"paired physical shape");
 if(up.cross.empty()){up.cross.resize(I);up.gn=gate.n;
  conditional_gram_dot_products+=(uint64_t)I*H;
  #pragma omp parallel for schedule(static)
  for(int i=0;i<I;++i)up.cross[i]=dot(up.w.data()+(size_t)i*H,gate.w.data()+(size_t)i*H,H);
 }
 require(up.gn==gate.n,"gate/up cache pair");
 uint64_t X=norm(r.qx.data(),H);std::vector<uint64_t> xb(N);for(int b=0;b<N;++b)xb[b]=norm(r.qx.data()+128*b,std::min(128,H-128*b));
 std::array<std::vector<uint64_t>,6> B;for(auto&v:B)v.resize(I);std::vector<uint8_t> h0(I);uint64_t fallback=0;
 uint64_t selectedX=0;if(deflation)for(int j:coords.at(r.layer))selectedX+=(int)r.qx[j]*(int)r.qx[j];
 for(int i=0;i<I;++i){auto global=global_bound(up.n[i],X);B[0][i]=global;uint64_t block=0;for(int b=0;b<N;++b)block+=global_bound(up.blocks[(size_t)i*N+b],xb[b]);B[1][i]=std::min(global,block);bool valid;B[2][i]=conditional_bound(gate.n[i],up.n[i],up.cross[i],X,r.gd[i],global,valid);fallback+=!valid;B[3][i]=std::min({B[0][i],B[1][i],B[2][i]});
  if(deflation){int64_t selectedDot=0;uint64_t selectedW=0;for(int j:coords.at(r.layer)){int w=up.w[(size_t)i*H+j];selectedW+=w*w;selectedDot+=w*(int)r.qx[j];}require(selectedW<=up.n[i]&&selectedX<=X,"integer residual");B[4][i]=std::min(global,abs64(selectedDot)+global_bound(up.n[i]-selectedW,X-selectedX));B[5][i]=std::min(B[3][i],B[4][i]);}
  for(int k=0;k<(deflation?6:4);++k){require(B[k][i]>=abs64(r.ud[i]),"integer bound violated");float U=scale_output(B[k][i],up.s[i],r.sx);require(std::isfinite(U)&&U>=std::abs(r.u[i]),"scaled Up bound violated");++integer_bound_checks;}
  // This conservative finite-Up guarantee is independent of the observed Up.
  float U=scale_output(global,up.s[i],r.sx);h0[i]=r.h[i]==0&&std::isfinite(U);require(std::isfinite(r.a[i])&&std::isfinite(r.h[i]),"nonfinite actual activation");
 }
 conditional_fallbacks+=fallback;
 primary_row_evaluations+=I;block_isqrt_evaluations+=(uint64_t)I*N;if(deflation)deflation_coordinate_products+=(uint64_t)I*16;
 if(r.step==0){for(int i=0;i<std::min(I,8);++i){auditlog<<"{\"job\":\""<<r.job<<"\",\"step\":"<<r.step<<",\"layer\":"<<r.layer<<",\"class\":\""<<r.kind<<"\",\"expert\":"<<r.e<<",\"row\":"<<i<<",\"gate_dot\":"<<r.gd[i]<<",\"up_dot\":"<<r.ud[i]<<",\"gate_bits\":"<<bits(r.g[i])<<",\"up_bits\":"<<bits(r.u[i])<<",\"h_bits\":"<<bits(r.h[i])<<",\"a_bits\":"<<bits(r.a[i])<<",\"sw_bits\":"<<bits(up.s[i])<<",\"sx_bits\":"<<bits(r.sx)<<",\"X2\":"<<X<<",\"V\":"<<up.n[i]<<",\"G\":"<<gate.n[i]<<",\"C\":"<<up.cross[i]<<",\"bounds\":[";for(int k=0;k<(deflation?6:4);++k)auditlog<<(k?",":"")<<B[k][i];auditlog<<"]}\n";}}
 for(int qbits:{8,6,4}){int qm=qbits==8?127:qbits==6?31:7;auto q=quantize(r.a.data(),I,qm);uint64_t nz=std::count(q.q.begin(),q.q.end(),0),nh0=std::count(h0.begin(),h0.end(),1);
  for(int wit=0;wit<2;++wit){int seed=wit?std::min(I,(16384+H-1)/H):0;float L=q.amax;if(wit){L=0;for(int i=0;i<seed;++i)L=std::max(L,std::abs(r.a[i]));}
   for(int k=0;k<(deflation?6:4);++k){std::vector<uint8_t> mask(I);uint64_t extra=0,skip=0,losth0=0;float after_amax=0;
    for(int i=0;i<I;++i){bool cert=certificate(r.h[i],B[k][i],up.s[i],r.sx,L,qm);mask[i]=i>=seed&&(h0[i]||cert);if(i<seed)losth0+=h0[i];if(mask[i]){require(q.q[i]==0,"claimed zero has nonzero ownq");extra+=!h0[i];++skip;}else after_amax=std::max(after_amax,std::abs(r.a[i]));}
    float after_scale=after_amax==0?1.f:after_amax/float(qm);require(eq(after_amax,q.amax)&&eq(after_scale,q.scale),"claimed zeros changed amax/scale bits");
    // Identical scale means unchanged remaining inputs reproduce own q bits;
    // explicitly check all removed and surviving rows, without reusing q8.
    for(int i=0;i<I;++i){int qi=mask[i]?0:std::clamp((int)std::round(r.a[i]/after_scale),-qm,qm);require(qi==q.q[i],"ownq bit invariant");}invariant_checks+=I;
    auto p4=pages(h0,mask,H,4096),p16=pages(h0,mask,H,16384);Metrics m={(uint64_t)1,(uint64_t)I,(uint64_t)I*H,nh0,nz,extra,skip,(uint64_t)seed,losth0,p4[0],p4[1],p4[2],p4[3],p4[4],p16[0],p16[1],p16[2],p16[3],p16[4],(uint64_t)I,fallback};
    auto&ag=aggregated[{r.job,r.phase,r.layer,r.kind,qbits,wit,k}];for(int j=0;j<NM;++j)ag[j]+=m[j];
    if(r.phase=="decode"){dispatch_log<<r.job<<','<<r.ji<<','<<r.step<<','<<r.layer<<','<<r.kind<<','<<r.e<<','<<qbits<<','<<wit<<','<<boundnames[k]<<','<<bits(L)<<','<<bits(q.amax)<<','<<bits(q.scale);for(auto v:m)dispatch_log<<','<<v;dispatch_log<<'\n';}
   }
  }
 }
}
static void freeze_fit(){require(!fit_frozen&&energy.size()==(size_t)model_layers,"fit coverage");
 for(const auto&it:energy){require(fit_input_counts[it.first]==32,"fit inputs must be unique layer/token 4x8");std::vector<int> order(it.second.size());std::iota(order.begin(),order.end(),0);std::stable_sort(order.begin(),order.end(),[&](int a,int b){return it.second[a]>it.second[b];});order.resize(16);coords[it.first]=order;fitlog<<"{\"layer\":"<<it.first<<",\"unique_decode_inputs\":32,\"coordinates\":[";for(int j=0;j<16;++j)fitlog<<(j?",":"")<<order[j];fitlog<<"],\"selected_energy\":[";for(int j=0;j<16;++j)fitlog<<(j?",":"")<<it.second[order[j]];fitlog<<"]}\n";}
 fitlog.flush();require(bool(fitlog),"fit freeze write");fit_frozen=true;std::cout<<"FROZEN deflation coords before job4; scoring first4 records="<<fit_records.size()<<std::endl;for(const auto&r:fit_records)score(r,true);fit_records.clear();fit_records.shrink_to_fit();
}
static void audit(ggml_tensor*dst,const Target&a){
 auto*w=dst->src[0];auto*x=dst->src[1];auto*ids=dst->src[2];bool routed=a.routed;
 require(x->data&&dst->data&&x->type==GGML_TYPE_F32&&dst->type==GGML_TYPE_F32&&x->ne[0]==w->ne[0]&&x->nb[0]==4&&dst->nb[0]==4&&x->ne[3]==1,"FFN tensor layout");
 int C=w->ne[0],R=w->ne[1],K=routed?ids->ne[0]:1,T=routed?ids->ne[1]:x->ne[1],V=routed?x->ne[1]:1;
 require(K==(routed?(modelname=="deepseek-v2-lite"?6:8):1),"selected expert count");
 if(routed)require(ids->data&&ids->type==GGML_TYPE_I32&&x->ne[2]==T&&dst->ne[0]==R&&dst->ne[1]==K&&dst->ne[2]==T&&(V==1||V==K),"routed shape");else require(x->ne[2]==1&&dst->ne[0]==R&&dst->ne[1]==T&&dst->ne[2]==1,"dense/shared shape");
 std::vector<Quant> q((size_t)T*V);uint64_t ih=FNV0;
 for(int t=0;t<T;++t)for(int v=0;v<V;++v){const float*src=(const float*)((char*)x->data+(routed?t*x->nb[2]+v*x->nb[1]:t*x->nb[1]));q[(size_t)t*V+v]=quantize(src,C);ih=hash(ih,q[(size_t)t*V+v].q.data(),C);if(a.role!="down")common_input(a.layer,t,q[(size_t)t*V+v]);}
 std::vector<Cache*> chosen((size_t)T*K);std::vector<int> es((size_t)T*K);
 for(int t=0;t<T;++t){std::set<int> distinct;for(int k=0;k<K;++k){int e=0;if(routed)std::memcpy(&e,(char*)ids->data+t*ids->nb[1]+k*ids->nb[0],4);require(e>=0&&e<w->ne[2]&&distinct.insert(e).second,"invalid/duplicate expert");es[(size_t)t*K+k]=e;chosen[(size_t)t*K+k]=&cache(w,e);}}
 std::vector<float> out((size_t)T*K*R);std::vector<int64_t> dots(out.size());
 #pragma omp parallel for schedule(static)
 for(int64_t task=0;task<(int64_t)out.size();++task){int row=task%R,k=(task/R)%K,t=task/(R*K);const auto&c=*chosen[(size_t)t*K+k];auto&input=q[(size_t)t*V+k%V];int64_t d=dot(c.w.data()+(size_t)row*C,input.q.data(),C);dots[task]=d;out[task]=signed_output(d,c.s[row],input.scale);}
 for(float f:out){require(std::isfinite(f),"nonfinite canonical FFN output");++finite_outputs;}
 for(int t=0;t<T;++t)for(int k=0;k<K;++k){int e=es[(size_t)t*K+k];auto key=PairKey(a.layer,a.kind,t,e);auto&input=q[(size_t)t*V+k%V];size_t off=((size_t)t*K+k)*R;
  if(a.role!="down"){auto&r=pending[key];if(!r.haveg&&!r.haveu){r.layer=a.layer;r.kind=a.kind;r.token=t;r.e=e;r.job=job;r.phase=phase;r.ji=jobindex;r.step=step;r.qx=input.q;r.sx=input.scale;}else require(r.qx==input.q&&eq(r.sx,input.scale),"paired qx/sx mismatch");
   if(a.role=="gate"){require(!r.haveg,"duplicate gate callback");r.haveg=true;r.gate=chosen[(size_t)t*K+k];r.g.assign(out.begin()+off,out.begin()+off+R);r.gd.assign(dots.begin()+off,dots.begin()+off+R);}else{require(!r.haveu,"duplicate up callback");r.haveu=true;r.up=chosen[(size_t)t*K+k];r.u.assign(out.begin()+off,out.begin()+off+R);r.ud.assign(dots.begin()+off,dots.begin()+off+R);}
  }else{auto it=pending.find(key);require(it!=pending.end()&&it->second.haveg&&it->second.haveu,"missing gate/up pair at Down");auto&r=it->second;require((int)r.g.size()==C,"original full-I size");r.h.resize(C);r.a.resize(C);ggml_vec_silu_f32(C,r.h.data(),r.g.data());const float*actual=(const float*)((char*)x->data+(routed?t*x->nb[2]+(k%V)*x->nb[1]:t*x->nb[1]));
   for(int i=0;i<C;++i){r.a[i]=actual[i];++product_checks;if(!eq(r.h[i]*r.u[i],r.a[i]))++activation_mismatches;}require(!activation_mismatches,"actual graph product differs from full-I GGML h*up: run invalid, no certificates");
   if(phase=="prefill"){auto&pd=prefill_diagnostics[{job,a.layer,a.kind}];++pd[0];pd[1]+=C;pd[2]+=std::count(r.h.begin(),r.h.end(),0.f);int ix=3;for(int qm:{127,31,7}){auto own=quantize(r.a.data(),C,qm);pd[ix++]+=std::count(own.q.begin(),own.q.end(),0);}}
   else if(jobindex<4)fit_records.push_back(std::move(r));else score(r,fit_frozen);pending.erase(it);
  }
 }
 for(int t=0;t<T;++t)for(int k=0;k<K;++k)for(int r=0;r<R;++r){float*p=(float*)((char*)dst->data+(routed?t*dst->nb[2]+k*dst->nb[1]:t*dst->nb[1])+r*4);*p=out[((size_t)t*K+k)*R+r];}
 auto&cv=coverage[job+"/"+phase+"/"+w->name];++cv[0];cv[1]+=T;cv[2]+=out.size();coverage_log<<"{\"job\":\""<<job<<"\",\"phase\":\""<<phase<<"\",\"step\":"<<step<<",\"weight\":\""<<w->name<<"\",\"layer\":"<<a.layer<<",\"class\":\""<<a.kind<<"\",\"role\":\""<<a.role<<"\",\"tokens\":"<<T<<",\"selected\":"<<K<<",\"input_vectors\":"<<T*V<<",\"output_floats\":"<<out.size()<<",\"qx_hash\":\""<<ih<<"\",\"output_hash\":\""<<hash(FNV0,out.data(),out.size()*4)<<"\"}\n";
}
struct Trace{bool failed=false;std::string error;};
static bool callback(ggml_tensor*t,bool ask,void*opaque){auto&tr=*(Trace*)opaque;try{Target a;if(!target(t,a))return false;if(ask)return true;audit(t,a);return true;}catch(const std::exception&e){tr.failed=true;tr.error=e.what();std::cerr<<"callback: "<<e.what()<<std::endl;return false;}}
static llama_token greedy(llama_context*ctx,int vocab,uint64_t&h){const float*p=llama_get_logits_ith(ctx,-1);require(p,"missing logits");int best=0;for(int j=0;j<vocab;++j){require(std::isfinite(p[j]),"nonfinite logits");++finite_logits;if(p[j]>p[best])best=j;}h=hash(h,p,(size_t)vocab*4);return best;}
static void idsjson(std::ostream&o,const std::vector<llama_token>&v){o<<'[';for(size_t i=0;i<v.size();++i)o<<(i?",":"")<<v[i];o<<']';}
static void verify_batch(){require(pending.empty(),"unmatched FFN pairs");require(common.size()>0,"no canonical FFN callback");for(int l=0;l<model_layers;++l)require(common.count({l,0}),"uncovered model layer");common.clear();}
static void run(llama_model*m,const std::string&path,const std::string&id){
 job=id;std::ifstream f(path,std::ios::binary);require(bool(f),"prompt open");std::string prompt((std::istreambuf_iterator<char>(f)),{});auto*vocab=llama_model_get_vocab(m);int n=-llama_tokenize(vocab,prompt.data(),prompt.size(),nullptr,0,true,true);require(n>0,"tokenizer size");std::vector<llama_token> full(n);require(llama_tokenize(vocab,prompt.data(),prompt.size(),full.data(),n,true,true)==n,"tokenizer count");std::vector<llama_token>tokens(full.begin(),full.begin()+std::min(n,128)),generated;
 prompt_counts[job]=tokens.size();Trace trace;auto cp=llama_context_default_params();cp.n_ctx=256;cp.n_batch=cp.n_ubatch=128;cp.n_threads=cp.n_threads_batch=8;cp.n_seq_max=1;cp.cb_eval=callback;cp.cb_eval_user_data=&trace;auto*ctx=llama_init_from_model(m,cp);require(ctx,"context init");auto b=llama_batch_init(128,0,1);
 auto add=[&](llama_token tok,int pos){int i=b.n_tokens++;b.token[i]=tok;b.pos[i]=pos;b.n_seq_id[i]=1;b.seq_id[i][0]=0;b.logits[i]=true;};
 auto decode=[&](){int rc=llama_decode(ctx,b);llama_synchronize(ctx);require(!trace.failed,"callback failed: "+trace.error);require(rc==0,"llama_decode code "+std::to_string(rc));verify_batch();};
 auto start=Clock::now();phase="prefill";step=-1;b.n_tokens=0;for(size_t j=0;j<tokens.size();++j)add(tokens[j],j);decode();uint64_t lh=FNV0;auto next=greedy(ctx,llama_vocab_n_tokens(vocab),lh);double prefill=std::chrono::duration<double>(Clock::now()-start).count();std::cout<<"PREFILL "<<modelname<<' '<<job<<" tokens="<<tokens.size()<<" seconds="<<prefill<<" caches="<<caches.size()<<std::endl;
 start=Clock::now();phase="decode";for(step=0;step<8;++step){generated.push_back(next);b.n_tokens=0;add(next,tokens.size()+step);decode();next=greedy(ctx,llama_vocab_n_tokens(vocab),lh);}double secs=std::chrono::duration<double>(Clock::now()-start).count();
 std::ofstream summary(root+"/"+job+".json");summary<<std::setprecision(17)<<"{\"job\":\""<<job<<"\",\"selection_index\":"<<jobindex<<",\"full_prompt_tokens\":"<<n<<",\"retained_prompt_tokens\":"<<tokens.size()<<",\"decoded_steps\":8,\"prefill_seconds\":"<<prefill<<",\"decode_seconds\":"<<secs<<",\"logit_stream_fnv1a64\":\""<<lh<<"\",\"full_token_ids\":";idsjson(summary,full);summary<<",\"retained_token_ids\":";idsjson(summary,tokens);summary<<",\"generated_token_ids\":";idsjson(summary,generated);summary<<"}\n";require(bool(summary),"job summary write");
 llama_batch_free(b);llama_free(ctx);coverage_log.flush();weightlog.flush();dispatch_log.flush();inputbin.flush();inputlog.flush();std::cout<<"DECODE "<<modelname<<' '<<job<<" steps=8 seconds="<<secs<<std::endl;
}
static void finish(){
 std::ofstream pd(root+"/prefill_diagnostics.csv");pd<<"job,layer,class,dispatches,rows,observed_h0_rows,q8_zero_rows,q6_zero_rows,q4_zero_rows\n";for(const auto&kv:prefill_diagnostics){pd<<std::get<0>(kv.first)<<','<<std::get<1>(kv.first)<<','<<std::get<2>(kv.first);for(auto v:kv.second)pd<<','<<v;pd<<'\n';}require(bool(pd),"prefill diagnostic write");
 std::ofstream ag(root+"/aggregates.csv");ag<<"job,phase,layer,class,bits,witness,bound,"<<metric_names<<'\n';for(const auto&kv:aggregated){const auto&k=kv.first;ag<<std::get<0>(k)<<','<<std::get<1>(k)<<','<<std::get<2>(k)<<','<<std::get<3>(k)<<','<<std::get<4>(k)<<','<<std::get<5>(k)<<','<<boundnames[std::get<6>(k)];for(auto v:kv.second)ag<<','<<v;ag<<'\n';}require(bool(ag),"aggregate write");
 const int branches=modelname=="deepseek-v2-lite"?53:model_layers;require(coverage.size()==(size_t)8*2*branches*3,"all job/phase/FFN branch/projection coverage");for(const auto&kv:coverage){bool dec=kv.first.find("/decode/")!=std::string::npos;auto jid=kv.first.substr(0,kv.first.find('/'));require(kv.second[0]==(dec?8:1)&&kv.second[1]==(dec?8:prompt_counts.at(jid)),"exact callback/token coverage");}
 uint64_t wbytes=0,normbytes=0,blockbytes=0,condbytes=0,deflbytes=0;for(const auto&kv:caches){const auto&c=kv.second;wbytes+=c.w.size();if(!c.cross.empty()){normbytes+=c.n.size()*8;blockbytes+=c.blocks.size()*8;condbytes+=c.cross.size()*16;deflbytes+=c.R*(16+8);}}
 std::ofstream out(root+"/report.json");out<<"{\"status\":\"complete\",\"model\":\""<<modelname<<"\",\"canonical_all_prefill_decode\":true,\"jobs\":8,\"decode_steps\":8,\"layers\":"<<model_layers<<",\"classes_per_job_phase\":"<<branches<<",\"coverage_keys\":"<<coverage.size()<<",\"cache_entries\":"<<caches.size()<<",\"cached_w8_bytes\":"<<wbytes<<",\"up_global_norm_metadata_bytes\":"<<normbytes<<",\"up_block128_metadata_bytes\":"<<blockbytes<<",\"conditional_G_C_metadata_bytes\":"<<condbytes<<",\"deflation16_weight_residual_metadata_bytes\":"<<deflbytes<<",\"deflation_coordinates_bytes\":"<<model_layers*16*4<<",\"full_I_product_bit_checks\":"<<product_checks<<",\"product_bit_mismatches\":"<<activation_mismatches<<",\"integer_and_scaled_bound_checks\":"<<integer_bound_checks<<",\"ownq_amax_scale_invariant_rows\":"<<invariant_checks<<",\"conditional_fallback_rows\":"<<conditional_fallbacks<<",\"primary_row_evaluations\":"<<primary_row_evaluations<<",\"block_isqrt_evaluations\":"<<block_isqrt_evaluations<<",\"deflation_coordinate_products\":"<<deflation_coordinate_products<<",\"conditional_gram_dot_products\":"<<conditional_gram_dot_products<<",\"finite_ffn_outputs\":"<<finite_outputs<<",\"finite_logits\":"<<finite_logits<<",\"f32_rounding_mode\":"<<std::fegetround()<<",\"mxcsr\":"<<_mm_getcsr()<<",\"physical_latency_claim\":false,\"deflation_fit_jobs\":4,\"deflation_unique_inputs_per_layer\":32}\n";require(bool(out),"report write");
}
int main(int argc,char**argv){try{
 if(argc==2&&std::string(argv[1])=="--self-test"){synthetic();return 0;}
 require(argc==5,"usage: collector model-name gguf frozen-jobs.tsv output-dir");modelname=argv[1];require(modelname=="deepseek-v2-lite"||modelname=="olmoe-1b-7b","unsupported model adapter");root=argv[4];std::filesystem::create_directories(root);omp_set_num_threads(8);synthetic();
 weightlog.open(root+"/weight_cache.jsonl");coverage_log.open(root+"/coverage.jsonl");dispatch_log.open(root+"/decode_dispatch.csv");fitlog.open(root+"/deflation_coordinates.frozen.jsonl");inputlog.open(root+"/common_decode_inputs.jsonl");auditlog.open(root+"/audit_samples.jsonl");inputbin.open(root+"/common_decode_inputs.i8",std::ios::binary);dispatch_log<<"job,selection_index,step,layer,class,expert,bits,witness,bound,L_bits,amax_bits,scale_bits,"<<metric_names<<'\n';
 gguf_init_params gp{true,nullptr};gf=gguf_init_from_file(argv[2],gp);require(gf,"GGUF metadata init");int fd=open(argv[2],O_RDONLY);struct stat st;require(fd>=0&&fstat(fd,&st)==0,"GGUF stat");mapping_size=st.st_size;void*mp=mmap(nullptr,mapping_size,PROT_READ,MAP_SHARED,fd,0);close(fd);require(mp!=MAP_FAILED,"GGUF mmap");mapping=(const char*)mp;
 llama_backend_init();ggml_backend_load_all();auto lp=llama_model_default_params();lp.n_gpu_layers=0;auto*m=llama_model_load_from_file(argv[2],lp);require(m,"llama model load");model_layers=llama_model_n_layer(m);require(llama_model_n_embd(m)==2048&&model_layers==(modelname=="deepseek-v2-lite"?27:16),"model identity dimensions");
 std::ifstream jobs(argv[3]);require(bool(jobs),"jobs TSV");std::string line;while(std::getline(jobs,line)){++jobindex;require(jobindex<8,"too many jobs");auto tab=line.find('\t');require(tab!=std::string::npos,"jobs TSV format");run(m,line.substr(0,tab),line.substr(tab+1));if(jobindex==3)freeze_fit();}require(jobindex==7&&fit_frozen&&fit_records.empty(),"expected frozen eight jobs");finish();llama_model_free(m);llama_backend_free();munmap((void*)mapping,mapping_size);gguf_free(gf);return 0;
 }catch(const std::exception&e){std::cerr<<"stage0 invalid/tooling: "<<e.what()<<std::endl;return 1;}}
