#pragma once
// New Stage3 implementation. Every predecessor remains immutable.
#include <immintrin.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdint>
#include <cstring>
#include <exception>
#include <functional>
#include <mutex>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <vector>
#include <pthread.h>
#include <sched.h>
namespace las3 {
inline void need(bool v,const char*s){if(!v)throw std::runtime_error(s);}
inline bool environment(){return std::fegetround()==FE_TONEAREST&&(_mm_getcsr()&0xe040u)==0;}
constexpr int NL=7,COPY_NL=8;
inline int32_t exact_dot(const int8_t*a,const int8_t*b,int n){
 need(n>0&&n<=2048,"exactI32 supported width1..2048");
 // API accepts [-128,127]. abs(all products/partials) <=2048*16384
 // =33554432 < INT32_MAX; sign-extended I16 products use nonsaturating
 // VPMADDWD pair sums and I32 additions. No maddubs or wraparound.
 __m256i sum=_mm256_setzero_si256();int k=0;
 for(;k+16<=n;k+=16){auto x=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(a+k)));auto y=_mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)(b+k)));sum=_mm256_add_epi32(sum,_mm256_madd_epi16(x,y));}
 alignas(32) int32_t lane[8];_mm256_store_si256((__m256i*)lane,sum);int32_t out=0;for(int j=0;j<8;++j)out+=lane[j];for(;k<n;++k)out+=int32_t(a[k])*int32_t(b[k]);return out;
}
template<class W> inline float fma_dot(const float*x,const W*w,int n){
 __m256 a=_mm256_setzero_ps();int k=0;
 for(;k+8<=n;k+=8){__m256 v;if constexpr(sizeof(W)==1)v=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*)(w+k))));else v=_mm256_loadu_ps(w+k);a=_mm256_fmadd_ps(_mm256_loadu_ps(x+k),v,a);}
 alignas(32) float s[8];_mm256_store_ps(s,a);for(;k<n;++k)s[k%8]=std::fma(x[k],float(w[k]),s[k%8]);float p0=s[0]+s[1],p1=s[2]+s[3],p2=s[4]+s[5],p3=s[6]+s[7];float lo=p0+p1,hi=p2+p3;return lo+hi;
}
inline float rescale(int32_t d,float sw,float sx){float s=sw*sx;float f=float(d);float z=f*s;need(std::isfinite(z),"nonfinite integer rescale");return z;}
inline float quantize(const float*x,int n,int8_t*q){float m=0;for(int k=0;k<n;++k){need(std::isfinite(x[k]),"nonfinite A8 input");m=std::max(m,std::abs(x[k]));}float s=m==0?1.f:m/127.f;need(s>0&&std::isfinite(s),"invalid A8 scale");for(int k=0;k<n;++k)q[k]=int8_t(std::clamp(int(std::round(x[k]/s)),-127,127));return s;}
inline double tree_dot(const float*x,const float*w,int n){double s=0;for(int j=0;j<n;++j){double p=double(x[j])*double(w[j]);s=s+p;}return s;}
inline float projection_dot(const float*x,const float*w,int n){double s[8]={};int j=0;for(;j+7<n;j+=8)for(int k=0;k<8;++k){double p=double(x[j+k])*double(w[j+k]);s[k]=s[k]+p;}for(;j<n;++j){double p=double(x[j])*double(w[j]);s[0]=s[0]+p;}double lo=(s[0]+s[1])+(s[2]+s[3]),hi=(s[4]+s[5])+(s[6]+s[7]);return float(lo+hi);}
struct Team {
 std::mutex mu;std::condition_variable start,done;std::vector<std::thread>threads;std::function<void(int)>job;uint64_t generation=0;int complete=0,ready=0;bool stop=false;std::exception_ptr error;
 Team(){need(environment(),"caller RN FTZ0 DAZ0");for(int id=0;id<8;++id)threads.emplace_back([this,id]{cpu_set_t cp;CPU_ZERO(&cp);CPU_SET(id,&cp);int rc=pthread_setaffinity_np(pthread_self(),sizeof(cp),&cp);std::unique_lock<std::mutex>lk(mu);if(rc||!environment()){if(!error)error=std::make_exception_ptr(std::runtime_error("actual persistent worker affinity/numeric environment"));}++ready;done.notify_one();uint64_t seen=0;for(;;){start.wait(lk,[&]{return stop||generation!=seen;});if(stop)return;seen=generation;auto f=job;lk.unlock();try{need(environment(),"actual worker RN FTZ0 DAZ0");f(id);need(environment(),"worker environment changed");}catch(...){lk.lock();if(!error)error=std::current_exception();lk.unlock();}lk.lock();++complete;if(complete==8)done.notify_one();}});std::unique_lock<std::mutex>lk(mu);done.wait(lk,[&]{return ready==8;});if(error){auto e=error;lk.unlock();shutdown();std::rethrow_exception(e);}}
 void run(std::function<void(int)>f){std::unique_lock<std::mutex>lk(mu);need(!stop,"stopped team");error=nullptr;job=std::move(f);complete=0;++generation;start.notify_all();done.wait(lk,[&]{return complete==8;});job={};if(error)std::rethrow_exception(error);}
 void shutdown(){{std::lock_guard<std::mutex>lk(mu);stop=true;}start.notify_all();for(auto&t:threads)if(t.joinable())t.join();}
 ~Team(){shutdown();}
};
struct W8 {int rows=0,cols=0;std::vector<int8_t>q;std::vector<float>s;};
struct Operator {int expert=0,I=0,R=128,Q=512;bool global=false;W8 g,u,down;std::vector<int8_t>cq,bq,uq;std::vector<float>cs,cb,bs,us,db,ng,nu,gb,ub,yb;std::array<bool,32>unsupported{};std::array<bool,32>dunsupported{};};
struct Layer {int H=2048,K=6,layer=0;std::vector<float>normals,means,P;std::vector<double>thresholds;std::vector<Operator>ops;};
struct Ledger {uint64_t coefficient_bytes=0,scale_bytes=0,bias_bytes=0,integer_MAC=0,FMA=0,common_metadata_bytes=0,P_FP64_MAC=0,tree_FP64_MAC=0,A8_elements=0,alpha_multiplies=0,slot_adds=0;int fallbacks=0,live=0,leaf=-1;double selector=0,projection=0,private_projection_GU=0,activation_A8=0,down_lift=0,bias_alpha=0;};
using Clock=std::chrono::steady_clock;
inline double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
inline int select(const Layer&l,const float*x){int node=0;for(int depth=0;depth<5;++depth){double z=tree_dot(x,l.normals.data()+size_t(node)*l.H,l.H);node=node*2+1+(z>l.thresholds.at(node));}return node-31;}
// Workspace is allocated before warmups/timing and fully counted in resident
// bytes. Every required transform/quantizer still executes in each invocation.
struct Scratch {
 int H=0,I=0,S=0,Q=0;float sx=1,sh=1,sp=1;std::vector<float>delta,p,fq,g,u,a,h,fused,latent,y,out;std::vector<int8_t>qx,qh,qp,qa;std::array<float,8>sa{};
 Scratch(const Layer&l,int slots=8):H(l.H),S(slots){for(auto&o:l.ops){I=std::max(I,o.I);Q=std::max(Q,o.Q);}delta.resize(H);p.resize(128);fq.resize(128);g.resize(size_t(S)*I);u=g;a=g;h=g;fused=g;latent.resize(size_t(S)*Q);y.resize(size_t(S)*H);out.resize(H);qx.resize(H);qh.resize(H);qp.resize(128);qa.resize(size_t(S)*I);}
 uint64_t bytes()const{return uint64_t(delta.size()+p.size()+fq.size()+g.size()+u.size()+a.size()+h.size()+fused.size()+latent.size()+y.size()+out.size())*4+qx.size()+qh.size()+qp.size()+qa.size()+sizeof(sa);}
};
using Activation=void(*)(int,float*,const float*,const float*,float*,float*);
inline const std::vector<float>&forward(Team&team,const Layer&l,int mode,const float*x,const std::vector<int>&which,const std::vector<float>&alpha,Activation activation,Ledger&log,Scratch&w){
 need(environment()&&(mode==3||mode==5||mode==6||mode==NL||mode==COPY_NL),"forward mode/environment");need(which.size()==alpha.size()&&which.size()<=8&&w.H==l.H,"slot/workspace dimensions");int H=l.H,S=int(which.size());if(!S){std::fill(w.out.begin(),w.out.end(),0.f);return w.out;}for(float a:alpha)need(std::isfinite(a),"finite final alpha (signed directAPI)");log.live=S;
 // kinds0 canonical/fallback,1 local,2 factor,3 nonlinear.
 std::array<int,8>kind{},prefix{};int total=0,c=-1;bool needP=false,needH=false,needX=mode==3,nonlinear=false;float sx=1,sh=1,sp=1;
 if(mode!=3){auto t=Clock::now();c=select(l,x);log.leaf=c;for(int j=0;j<H;++j)w.delta[j]=x[j]-l.means[size_t(c)*H+j];log.selector=seconds(t);log.common_metadata_bytes=uint64_t(6)*H*4+5*8;log.tree_FP64_MAC=uint64_t(5)*H;}
 for(int slot=0;slot<S;++slot){auto&o=l.ops.at(which[slot]);need(o.I<=w.I&&o.Q<=w.Q,"workspace capacity");bool fallback=mode!=3&&(mode==6?o.dunsupported[c]:o.unsupported[c]);kind[slot]=mode==3||fallback?0:mode==5?1:mode==6?2:3;log.fallbacks+=fallback;needX|=kind[slot]==0;needH|=kind[slot]==2&&o.global;needP|=kind[slot]==1||kind[slot]==3||(kind[slot]==2&&!o.global);nonlinear|=kind[slot]==0||kind[slot]==3;prefix[slot]=total;total+=kind[slot]==2?o.Q:(kind[slot]==0||kind[slot]==3?o.I:0);
  if(kind[slot]==0){log.coefficient_bytes+=uint64_t(3)*o.I*H;log.scale_bytes+=uint64_t(2*o.I+H)*4;log.integer_MAC+=uint64_t(3)*o.I*H;}
  else if(kind[slot]==1){log.coefficient_bytes+=uint64_t(H)*128;log.scale_bytes+=uint64_t(H)*4;log.bias_bytes+=uint64_t(H)*4;log.integer_MAC+=uint64_t(H)*128;}
  else if(kind[slot]==2){int R=o.global?H:128;log.coefficient_bytes+=uint64_t(o.Q)*R+uint64_t(H)*o.Q;log.scale_bytes+=uint64_t(o.Q+H)*4;log.bias_bytes+=uint64_t(H)*4;log.integer_MAC+=uint64_t(o.Q)*R;log.FMA+=uint64_t(H)*o.Q;}
  else{log.coefficient_bytes+=uint64_t(2)*o.I*128*4+uint64_t(H)*o.I;log.scale_bytes+=uint64_t(H)*4;log.bias_bytes+=uint64_t(2*o.I+H)*4;log.FMA+=uint64_t(2)*o.I*128;log.integer_MAC+=uint64_t(H)*o.I;}}
 auto t=Clock::now();if(needX){sx=quantize(x,H,w.qx.data());log.A8_elements+=H;}if(needH){sh=quantize(w.delta.data(),H,w.qh.data());log.A8_elements+=H;}if(needP){team.run([&](int id){for(int r=id;r<128;r+=8)w.p[r]=projection_dot(w.delta.data(),l.P.data()+size_t(r)*H,H);});sp=quantize(w.p.data(),128,w.qp.data());for(int k=0;k<128;++k)w.fq[k]=float(w.qp[k]);log.common_metadata_bytes+=uint64_t(128)*H*4;log.P_FP64_MAC=uint64_t(128)*H;log.A8_elements+=128;}w.sx=sx;w.sh=sh;w.sp=sp;log.projection=seconds(t);
 // One first-stage dispatch across ALL selected slots/rows, never K teams.
 t=Clock::now();if(total)team.run([&](int id){for(int task=id;task<total;task+=8){int slot=S-1;while(slot>0&&task<prefix[slot])--slot;while(slot+1<S&&prefix[slot+1]==prefix[slot])++slot;int row=task-prefix[slot];auto&o=l.ops[which[slot]];size_t base=size_t(slot)*w.I;
  if(kind[slot]==0){w.g[base+row]=rescale(exact_dot(w.qx.data(),o.g.q.data()+size_t(row)*H,H),o.g.s[row],sx);w.u[base+row]=rescale(exact_dot(w.qx.data(),o.u.q.data()+size_t(row)*H,H),o.u.s[row],sx);}
  else if(kind[slot]==3){float gg=fma_dot(w.fq.data(),o.ng.data()+size_t(row)*128,128),uu=fma_dot(w.fq.data(),o.nu.data()+size_t(row)*128,128);float gs=gg*sp,us=uu*sp;w.g[base+row]=gs+o.gb[size_t(c)*o.I+row];w.u[base+row]=us+o.ub[size_t(c)*o.I+row];need(std::isfinite(w.g[base+row])&&std::isfinite(w.u[base+row]),"NL GU finite");}
  else{int R=o.global?H:128,j=o.global?0:c;w.latent[size_t(slot)*w.Q+row]=rescale(exact_dot(o.global?w.qh.data():w.qp.data(),o.bq.data()+(size_t(j)*o.Q+row)*R,R),o.bs[size_t(j)*o.Q+row],o.global?sh:sp);}}});log.private_projection_GU=seconds(t);
 if(nonlinear){for(int slot=0;slot<S;++slot)if(kind[slot]==0||kind[slot]==3)log.A8_elements+=l.ops[which[slot]].I;t=Clock::now();team.run([&](int id){for(int slot=id;slot<S;slot+=8)if(kind[slot]==0||kind[slot]==3){auto&o=l.ops[which[slot]];size_t b=size_t(slot)*w.I;activation(o.I,w.a.data()+b,w.g.data()+b,w.u.data()+b,w.h.data()+b,w.fused.data()+b);w.sa[slot]=quantize(w.a.data()+b,o.I,w.qa.data()+b);}});log.activation_A8=seconds(t);}
 // One completed-output team. A row owner evaluates all slots and applies
 // separate F32 alpha multiplies/adds in the original selected-slot order.
 log.alpha_multiplies=uint64_t(S)*H;log.slot_adds=uint64_t(S-1)*H;t=Clock::now();team.run([&](int id){for(int row=id;row<H;row+=8){float sum=0;for(int slot=0;slot<S;++slot){auto&o=l.ops[which[slot]];float v;
   if(kind[slot]==0||kind[slot]==3){v=rescale(exact_dot(w.qa.data()+size_t(slot)*w.I,o.down.q.data()+size_t(row)*o.I,o.I),o.down.s[row],w.sa[slot]);if(kind[slot]==3)v=v+o.yb[size_t(c)*H+row];}
   else if(kind[slot]==1){v=rescale(exact_dot(w.qp.data(),o.cq.data()+(size_t(c)*H+row)*128,128),o.cs[size_t(c)*H+row],sp);v=v+o.cb[size_t(c)*H+row];}
   else{float a=fma_dot(w.latent.data()+size_t(slot)*w.Q,o.uq.data()+size_t(row)*o.Q,o.Q);float b=a*o.us[row];v=b+o.db[size_t(c)*H+row];}
   need(std::isfinite(v),"cohort output/bias finite");w.y[size_t(slot)*H+row]=v;float weighted=v*alpha[slot];sum=slot==0?weighted:sum+weighted;need(std::isfinite(sum),"finite slot ordered alpha combination");}w.out[row]=sum;}});log.down_lift=seconds(t);return w.out;
}
}
