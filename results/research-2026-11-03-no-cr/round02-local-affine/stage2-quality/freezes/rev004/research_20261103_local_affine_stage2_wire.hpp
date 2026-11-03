#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <immintrin.h>
#include <algorithm>
#include <atomic>
#include <cerrno>
#include <cfenv>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <map>
#include <set>
#include <stdexcept>
#include <string>
#include <vector>
#include <poll.h>
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>

namespace laq {
inline void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
inline bool environment(){return std::fegetround()==FE_TONEAREST&&(_mm_getcsr()&((1u<<15)|(1u<<6)|0x6000u))==0;}
constexpr uint32_t MAGIC=0x3251414c,VERSION=1;
constexpr uint64_t MAX_BYTES=16u<<20;
enum Mode:uint32_t{SOURCE=0,OBSERVER=1,COPY_ORIGINAL=2,CANONICAL=3,COPY_CANONICAL=4,LOCAL=5,FACTOR=6};
#pragma pack(push,1)
struct Header {
 uint32_t magic=MAGIC,version=VERSION,kind=1,mode=0;
 uint64_t request=0,run=0,context=0;
 uint32_t subrun=0,arm=0;uint64_t decode=0;
 uint32_t micro=0,model=0,layer=0,T=0,H=2048,K=0,slots=0;
 uint64_t bytes=0;uint32_t status=0,reserved=0;
 uint64_t position=0;uint32_t token_phase=0,reserved2=0;
};
#pragma pack(pop)
static_assert(sizeof(Header)==116,"fixed little endian IPC header");
inline void header_valid(const Header&h){uint16_t x=1;need(*(uint8_t*)&x==1,"little endian host required");need(h.magic==MAGIC&&h.version==VERSION&&h.reserved==0&&h.reserved2==0&&h.token_phase<=2&&h.bytes<=MAX_BYTES,"IPC header/version/size");}
inline bool same_epoch(const Header&a,const Header&b){return a.request==b.request&&a.run==b.run&&a.context==b.context&&a.subrun==b.subrun&&a.arm==b.arm&&a.decode==b.decode&&a.micro==b.micro&&a.model==b.model&&a.layer==b.layer&&a.mode==b.mode&&a.T==b.T&&a.H==b.H&&a.K==b.K&&a.position==b.position&&a.token_phase==b.token_phase;}
inline void exact_io(int fd,void*ptr,size_t n,bool writing,int timeout_ms=120000){
 auto*p=(uint8_t*)ptr;auto deadline=std::chrono::steady_clock::now()+std::chrono::milliseconds(timeout_ms);
 while(n){auto left=std::chrono::duration_cast<std::chrono::milliseconds>(deadline-std::chrono::steady_clock::now()).count();need(left>0,"IPC timeout");pollfd f{fd,short(writing?POLLOUT:POLLIN),0};int r=poll(&f,1,int(left));if(r<0&&errno==EINTR)continue;need(r>0,"IPC timeout/poll failure");need(!(f.revents&(POLLERR|POLLNVAL)),"IPC pipe error");ssize_t k=writing?write(fd,p,n):read(fd,p,n);if(k<0&&errno==EINTR)continue;need(k>0,"IPC short read/write or worker exit");p+=k;n-=size_t(k);}
}
struct Slot {uint32_t t,k,e,flags=0;std::vector<float> y;};
inline std::vector<Slot> mask(const std::vector<int32_t>&ids,int T,int K,const std::set<int>&pilots){
 need(ids.size()==size_t(T)*K,"ID owned shape");std::vector<Slot>out;
 for(int t=0;t<T;++t){std::set<int>seen;for(int k=0;k<K;++k){int e=ids[t*K+k];need(e>=0&&e<64&&seen.insert(e).second,"invalid/duplicate routed ID");if(pilots.count(e))out.push_back({uint32_t(t),uint32_t(k),uint32_t(e),0,{}});}}
 return out;
}
inline void validate_response(const Header&req,const Header&h,const std::vector<uint8_t>&blob,const std::vector<Slot>&expected,std::vector<Slot>&out){
 header_valid(h);need(h.kind==2&&h.status==0&&same_epoch(req,h),"IPC response kind/status/epoch mismatch");
 need(h.slots==expected.size()&&h.bytes==blob.size()&&h.bytes==uint64_t(h.slots)*(16+4*h.H),"IPC response slot count/length");
 std::map<std::pair<uint32_t,uint32_t>,uint32_t>want;for(auto&s:expected)want[{s.t,s.k}]=s.e;
 std::set<std::pair<uint32_t,uint32_t>>seen;out.clear();size_t off=0;
 for(uint32_t z=0;z<h.slots;++z){uint32_t v[4];std::memcpy(v,blob.data()+off,16);off+=16;auto key=std::make_pair(v[0],v[1]);need(want.count(key)&&want.at(key)==v[2]&&seen.insert(key).second&&v[3]<=1,"IPC unsolicited/duplicate/invalid slot");Slot s{v[0],v[1],v[2],v[3],std::vector<float>(h.H)};std::memcpy(s.y.data(),blob.data()+off,4*h.H);off+=4*h.H;for(float a:s.y)need(std::isfinite(a),"IPC nonfinite output");out.push_back(std::move(s));}
 need(seen.size()==want.size(),"IPC missing pilot slot");
}
inline void host(const ggml_tensor*t){need(t&&t->buffer&&ggml_backend_buffer_is_host(t->buffer)&&t->data,"unverified/nonhost graph buffer");}
inline void intervals(const std::vector<std::pair<size_t,size_t>>&v,size_t nbytes){auto s=v;std::sort(s.begin(),s.end());size_t prev=0;for(auto [off,n]:s){need(n&&off<=nbytes&&n<=nbytes-off&&off>=prev,"tensor stride overlap/range");prev=off+n;}}
inline std::vector<int32_t> read_ids(const ggml_tensor*t,int T,int K){
 host(t);need(t->type==GGML_TYPE_I32&&t->ne[0]==K&&t->ne[1]==T&&t->ne[2]==1&&t->ne[3]==1,"IDs tensor shape/type");
 std::vector<std::pair<size_t,size_t>>iv;std::vector<int32_t>out(T*K);for(int a=0;a<T;++a)for(int k=0;k<K;++k)iv.push_back({a*t->nb[1]+k*t->nb[0],4});intervals(iv,ggml_nbytes(t));for(int a=0;a<T;++a)for(int k=0;k<K;++k)ggml_backend_tensor_get(t,&out[a*K+k],a*t->nb[1]+k*t->nb[0],4);mask(out,T,K,{});return out;
}
inline std::vector<float> read_x(const ggml_tensor*t,int T,int H){
 host(t);need(t->type==GGML_TYPE_F32&&t->ne[0]==H&&t->ne[1]==1&&t->ne[2]==T&&t->ne[3]==1&&t->nb[0]==4,"up source input shape/type");std::vector<std::pair<size_t,size_t>>iv;for(int a=0;a<T;++a)iv.push_back({a*t->nb[2],size_t(H)*4});intervals(iv,ggml_nbytes(t));std::vector<float>out(size_t(T)*H);for(int a=0;a<T;++a)ggml_backend_tensor_get(t,out.data()+size_t(a)*H,a*t->nb[2],H*4);for(float x:out)need(std::isfinite(x),"input nonfinite");return out;
}
inline void down_layout(const ggml_tensor*t,int T,int H,int K){
 host(t);need(t->type==GGML_TYPE_F32&&t->ne[0]==H&&t->ne[1]==K&&t->ne[2]==T&&t->ne[3]==1&&t->nb[0]==4,"Down shape/type");std::vector<std::pair<size_t,size_t>>iv;for(int a=0;a<T;++a)for(int k=0;k<K;++k)iv.push_back({a*t->nb[2]+k*t->nb[1],size_t(H)*4});intervals(iv,ggml_nbytes(t));
}
inline void scatter(ggml_tensor*t,const std::vector<Slot>&slots,int T,int H,int K){
 down_layout(t,T,H,K);std::set<std::pair<uint32_t,uint32_t>>seen;
 for(auto&s:slots){need(s.t<uint32_t(T)&&s.k<uint32_t(K)&&s.y.size()==size_t(H)&&seen.insert({s.t,s.k}).second,"scatter invalid/duplicate slot");for(float x:s.y)need(std::isfinite(x),"scatter nonfinite");}
 // Validation of all rows finishes before the first graph write.
 for(auto&s:slots)ggml_backend_tensor_set(t,s.y.data(),s.t*t->nb[2]+s.k*t->nb[1],H*4);
}
struct PoolCheck {std::atomic<uint32_t>seen{0},bad{0};int inject=-1;uint32_t mode=0;};
inline void env_op(ggml_tensor*dst,const ggml_tensor*,int ith,int nth,void*opaque){auto&s=*(PoolCheck*)opaque;uint32_t orig=_mm_getcsr();if(ith==s.inject)_mm_setcsr((orig&~0x6000u)|s.mode);if(nth!=8||!environment())s.bad|=1;s.seen|=1u<<ith;((float*)dst->data)[ith]=0;if(ith==s.inject)_mm_setcsr(orig);}
inline bool check_pool(ggml_threadpool*pool,int inject=-1,uint32_t mode=0){
 PoolCheck state;state.inject=inject;state.mode=mode;ggml_init_params ip{2u<<20,nullptr,false};auto*ctx=ggml_init(ip);need(ctx,"pool environment context");auto*a=ggml_new_tensor_1d(ctx,GGML_TYPE_F32,8);auto*out=ggml_map_custom1(ctx,a,env_op,8,&state);auto*g=ggml_new_graph(ctx);ggml_build_forward_expand(g,out);auto plan=ggml_graph_plan(g,8,pool);std::vector<uint8_t>work(plan.work_size);plan.work_data=work.data();auto rc=ggml_graph_compute(g,&plan);ggml_free(ctx);return rc==GGML_STATUS_SUCCESS&&state.seen==255&&state.bad==0;
}
struct Worker {
 pid_t pid=-1;int input=-1,output=-1;uint64_t serial=0;uint32_t model=0;
 Worker(const std::vector<std::string>&argv,uint32_t mid):model(mid){
  int to[2],from[2];need(pipe(to)==0&&pipe(from)==0,"worker pipe");pid=fork();need(pid>=0,"worker fork");
  if(pid==0){setpgid(0,0);dup2(to[0],STDIN_FILENO);dup2(from[1],STDOUT_FILENO);close(to[0]);close(to[1]);close(from[0]);close(from[1]);std::vector<char*>args;for(auto&s:argv)args.push_back(const_cast<char*>(s.c_str()));args.push_back(nullptr);execvp(args[0],args.data());_exit(127);}
  setpgid(pid,pid);close(to[0]);close(from[1]);input=to[1];output=from[0];
  try{Header h;exact_io(output,&h,sizeof(h),false);header_valid(h);need(h.kind==3&&h.model==mid&&h.status==0&&h.bytes==0&&h.slots==0,"worker ready/authentication failure");}
  catch(...){kill(-pid,SIGKILL);waitpid(pid,nullptr,0);close(input);close(output);pid=-1;input=output=-1;throw;}
 }
 std::vector<Slot> compute(Header h,const std::vector<float>&x,const std::vector<int32_t>&ids,const std::vector<Slot>&expected){
  h.request=++serial;h.kind=1;h.slots=expected.size();h.bytes=(x.size()+ids.size())*4;header_valid(h);need(h.bytes==uint64_t(h.T)*(h.H+h.K)*4&&h.model==model,"request shape/model");exact_io(input,&h,sizeof(h),true);exact_io(input,(void*)x.data(),x.size()*4,true);exact_io(input,(void*)ids.data(),ids.size()*4,true);Header reply;exact_io(output,&reply,sizeof(reply),false);header_valid(reply);std::vector<uint8_t>blob(reply.bytes);if(reply.bytes)exact_io(output,blob.data(),blob.size(),false);std::vector<Slot>out;validate_response(h,reply,blob,expected,out);return out;
 }
 void finish(){if(input>=0){close(input);input=-1;}if(pid>0){int status=0;auto end=std::chrono::steady_clock::now()+std::chrono::seconds(30);for(;;){pid_t r=waitpid(pid,&status,WNOHANG);if(r==pid)break;if(r<0&&errno==EINTR)continue;need(r==0,"worker waitpid failed");need(std::chrono::steady_clock::now()<end,"worker shutdown timeout");usleep(10000);}pid=-1;need(WIFEXITED(status)&&WEXITSTATUS(status)==0,"worker nonzero exit");}if(output>=0){close(output);output=-1;}}
 ~Worker(){if(pid>0){kill(-pid,SIGKILL);waitpid(pid,nullptr,0);}if(input>=0)close(input);if(output>=0)close(output);}
};
struct Snapshot {Header epoch;std::vector<float>x;std::vector<int32_t>ids;bool consumed=false,weighted=false,closed=false;};
inline uint64_t hash_bytes(uint64_t h,const void*p,size_t n){auto*a=(const uint8_t*)p;for(size_t j=0;j<n;++j){h^=a[j];h*=1099511628211ULL;}return h;}
}
