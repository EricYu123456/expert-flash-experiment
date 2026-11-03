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
struct Ledger {uint64_t coefficient_bytes=0,scale_bytes=0,bias_bytes=0,integer_MAC=0,FMA=0;int fallbacks=0,live=0,leaf=-1;double selector=0,projection=0,private_projection_GU=0,activation_A8=0,down_lift=0,bias_alpha=0;};
using Clock=std::chrono::steady_clock;
inline double seconds(Clock::time_point t){return std::chrono::duration<double>(Clock::now()-t).count();}
inline int select(const Layer&l,const float*x){int node=0;for(int depth=0;depth<5;++depth){double z=tree_dot(x,l.normals.data()+size_t(node)*l.H,l.H);node=node*2+1+(z>l.thresholds.at(node));}return node-31;}
// Full-I pinned activation callback, identical in executable and batch oracle.
using Activation=void(*)(int,float*,const float*,const float*);
inline std::vector<float> forward(Team&team,const Layer&l,int mode,const float*x,const std::vector<int>&which,const std::vector<float>&alpha,Activation activation,Ledger&log){
 need(environment()&&(mode==3||mode==5||mode==6||mode==NL||mode==COPY_NL),"forward mode/environment");need(which.size()==alpha.size(),"slot alpha dimensions");int H=l.H;std::vector<float>out(H,0.f);if(which.empty())return out;for(float a:alpha)need(std::isfinite(a)&&a>=0,"final alpha");log.live=int(which.size());
 int c=-1;std::vector<float>delta(H),p(128);std::vector<int8_t>qx(H),qh(H),qp(128);float sx=1,sh=1,sp=1;bool needP=false,needH=false,needX=mode==3;
 if(mode!=3){auto t=Clock::now();c=select(l,x);log.leaf=c;for(int j=0;j<H;++j)delta[j]=x[j]-l.means[size_t(c)*H+j];log.selector=seconds(t);for(int z:which){auto&o=l.ops.at(z);bool fallback=(mode==6?o.dunsupported[c]:o.unsupported[c]);needX|=fallback;if(!fallback){needH|=mode==6&&o.global;needP|=mode!=6||!o.global;}}}
 auto t=Clock::now();if(needX)sx=quantize(x,H,qx.data());if(needH)sh=quantize(delta.data(),H,qh.data());if(needP){team.run([&](int id){for(int r=id;r<128;r+=8)p[r]=projection_dot(delta.data(),l.P.data()+size_t(r)*H,H);});sp=quantize(p.data(),128,qp.data());}log.projection=seconds(t);
 std::vector<std::vector<float>>ys(which.size(),std::vector<float>(H));
 for(size_t slot=0;slot<which.size();++slot){auto&o=l.ops.at(which[slot]);bool fallback=mode!=3&&(mode==6?o.dunsupported[c]:o.unsupported[c]);if(fallback)++log.fallbacks;auto&y=ys[slot];
  if(mode==3||fallback||mode==NL||mode==COPY_NL){std::vector<float>g(o.I),u(o.I),a(o.I);std::vector<int8_t>qa(o.I);t=Clock::now();
   if(mode==3||fallback){team.run([&](int id){for(int r=id;r<o.I;r+=8){g[r]=rescale(exact_dot(qx.data(),o.g.q.data()+size_t(r)*H,H),o.g.s[r],sx);u[r]=rescale(exact_dot(qx.data(),o.u.q.data()+size_t(r)*H,H),o.u.s[r],sx);}});log.coefficient_bytes+=uint64_t(2)*o.I*H;log.scale_bytes+=uint64_t(2)*o.I*4;log.integer_MAC+=uint64_t(2)*o.I*H;
   }else{std::array<float,128>fq;for(int k=0;k<128;++k)fq[k]=float(qp[k]);team.run([&](int id){for(int r=id;r<o.I;r+=8){float gg=fma_dot(fq.data(),o.ng.data()+size_t(r)*128,128),uu=fma_dot(fq.data(),o.nu.data()+size_t(r)*128,128);float gs=gg*sp,us=uu*sp;g[r]=gs+o.gb[size_t(c)*o.I+r];u[r]=us+o.ub[size_t(c)*o.I+r];need(std::isfinite(g[r])&&std::isfinite(u[r]),"NL GU finite");}});log.coefficient_bytes+=uint64_t(2)*o.I*128*4;log.bias_bytes+=uint64_t(2)*o.I*4;log.FMA+=uint64_t(2)*o.I*128;}
   log.private_projection_GU+=seconds(t);t=Clock::now();activation(o.I,a.data(),g.data(),u.data());float sa=quantize(a.data(),o.I,qa.data());log.activation_A8+=seconds(t);t=Clock::now();team.run([&](int id){for(int r=id;r<H;r+=8)y[r]=rescale(exact_dot(qa.data(),o.down.q.data()+size_t(r)*o.I,o.I),o.down.s[r],sa);});log.down_lift+=seconds(t);log.coefficient_bytes+=uint64_t(H)*o.I;log.scale_bytes+=uint64_t(H)*4;log.integer_MAC+=uint64_t(H)*o.I;
   if(mode!=3&&!fallback){t=Clock::now();for(int r=0;r<H;++r){y[r]=y[r]+o.yb[size_t(c)*H+r];need(std::isfinite(y[r]),"NL output bias finite");}log.bias_alpha+=seconds(t);log.bias_bytes+=uint64_t(H)*4;}
  }else if(mode==5){t=Clock::now();team.run([&](int id){for(int r=id;r<H;r+=8){y[r]=rescale(exact_dot(qp.data(),o.cq.data()+(size_t(c)*H+r)*128,128),o.cs[size_t(c)*H+r],sp);y[r]=y[r]+o.cb[size_t(c)*H+r];need(std::isfinite(y[r]),"local bias finite");}});log.down_lift+=seconds(t);log.coefficient_bytes+=uint64_t(H)*128;log.scale_bytes+=uint64_t(H)*4;log.bias_bytes+=uint64_t(H)*4;log.integer_MAC+=uint64_t(H)*128;
  }else{t=Clock::now();std::vector<float>h(o.Q);int R=o.global?H:128,j=o.global?0:c;const auto&q=o.global?qh:qp;float s=o.global?sh:sp;team.run([&](int id){for(int r=id;r<o.Q;r+=8)h[r]=rescale(exact_dot(q.data(),o.bq.data()+(size_t(j)*o.Q+r)*R,R),o.bs[size_t(j)*o.Q+r],s);});log.private_projection_GU+=seconds(t);t=Clock::now();team.run([&](int id){for(int r=id;r<H;r+=8){float a=fma_dot(h.data(),o.uq.data()+size_t(r)*o.Q,o.Q);float b=a*o.us[r];y[r]=b+o.db[size_t(c)*H+r];need(std::isfinite(y[r]),"factor output finite");}});log.down_lift+=seconds(t);log.coefficient_bytes+=uint64_t(o.Q)*R+uint64_t(H)*o.Q;log.scale_bytes+=uint64_t(o.Q+H)*4;log.bias_bytes+=uint64_t(H)*4;log.integer_MAC+=uint64_t(o.Q)*R;log.FMA+=uint64_t(H)*o.Q;}
 }
 t=Clock::now();for(size_t slot=0;slot<ys.size();++slot)for(int r=0;r<H;++r){float weighted=ys[slot][r]*alpha[slot];out[r]=slot==0?weighted:out[r]+weighted;need(std::isfinite(out[r]),"alpha/slot sum finite");}log.bias_alpha+=seconds(t);return out;
}
}
