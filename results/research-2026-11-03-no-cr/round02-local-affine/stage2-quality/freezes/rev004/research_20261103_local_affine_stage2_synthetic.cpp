#define LAQ_NO_MAIN
#include "research_20261103_local_affine_stage2_harness.cpp"
#include <functional>
static void rejects(const std::function<void()>&f){bool bad=false;try{f();}catch(const std::exception&){bad=true;}need(bad,"expected fail-closed rejection");}
template<class T>struct AlignedAllocator {using value_type=T;AlignedAllocator()=default;template<class U>AlignedAllocator(const AlignedAllocator<U>&){};T*allocate(size_t n){void*p=nullptr;if(posix_memalign(&p,64,n*sizeof(T)))throw std::bad_alloc();return (T*)p;}void deallocate(T*p,size_t){free(p);}template<class U>bool operator==(const AlignedAllocator<U>&)const{return true;}template<class U>bool operator!=(const AlignedAllocator<U>&)const{return false;}};
struct Tensor {
 ggml_tensor t{};std::vector<uint8_t,AlignedAllocator<uint8_t>> bytes;ggml_backend_buffer_t buffer=nullptr;
 Tensor(ggml_type type,int64_t a,int64_t b,int64_t c,int64_t d,size_t n0=0,size_t n1=0,size_t n2=0){
  t.type=type;t.ne[0]=a;t.ne[1]=b;t.ne[2]=c;t.ne[3]=d;t.nb[0]=n0?n0:4;t.nb[1]=n1?n1:t.nb[0]*a;t.nb[2]=n2?n2:t.nb[1]*b;t.nb[3]=t.nb[2]*c;bytes.assign(ggml_nbytes(&t),0xa5);buffer=ggml_backend_cpu_buffer_from_ptr(bytes.data(),bytes.size());need(buffer,"synthetic CPU buffer");t.buffer=buffer;t.data=bytes.data();
 }
 ~Tensor(){ggml_backend_buffer_free(buffer);}Tensor(const Tensor&)=delete;
 template<class V>void put(size_t offset,V x){std::memcpy(bytes.data()+offset,&x,sizeof(x));}
};
struct Fixture {
 int mid,K,I,l;Tensor x,ids,down,alpha;ggml_tensor wu{},wd{},up{},din{},weighted{},out{};
 Fixture(int model,int layer,bool zero=false):mid(model),K(model==1?6:8),I(model==1?1408:1024),l(layer),x(GGML_TYPE_F32,2048,1,2,1,4,8192,8224),ids(GGML_TYPE_I32,K,2,1,1,8,8*K+16),down(GGML_TYPE_F32,2048,K,2,1,4,8240,8240*K+32),alpha(GGML_TYPE_F32,1,K,2,1,4,12,12*K+16){
  std::vector<int> route;auto selected=pilots(mid).at(l);if(!zero)for(int e:selected)route.push_back(e);for(int e=0;route.size()<size_t(K);++e)if(!selected.count(e))route.push_back(e);
  for(int t=0;t<2;++t){for(int j=0;j<2048;++j)x.put(t*x.t.nb[2]+4*j,float(t+j)/2048);for(int k=0;k<K;++k){int e=route[k];ids.put(t*ids.t.nb[1]+k*ids.t.nb[0],int32_t(e));alpha.put(t*alpha.t.nb[2]+k*alpha.t.nb[1],float(k+1)/32);for(int j=0;j<2048;++j)down.put(t*down.t.nb[2]+k*down.t.nb[1]+4*j,float(-t-k-j)/128);}}
  wu.ne[0]=2048;wu.ne[1]=I;wu.ne[2]=64;wu.ne[3]=1;wd.ne[0]=I;wd.ne[1]=2048;wd.ne[2]=64;wd.ne[3]=1;std::snprintf(wu.name,sizeof(wu.name),"blk.%d.ffn_up_exps.weight",l);std::snprintf(wd.name,sizeof(wd.name),"blk.%d.ffn_down_exps.weight",l);
  up.op=GGML_OP_MUL_MAT_ID;up.src[0]=&wu;up.src[1]=&x.t;up.src[2]=&ids.t;down.t.op=GGML_OP_MUL_MAT_ID;down.t.src[0]=&wd;down.t.src[1]=&din;down.t.src[2]=&ids.t;din.type=GGML_TYPE_F32;din.ne[0]=I;din.ne[1]=K;din.ne[2]=2;din.ne[3]=1;
  weighted.op=GGML_OP_MUL;weighted.src[1]=&alpha.t;std::snprintf(weighted.name,sizeof(weighted.name),"ffn_moe_weighted-%d",l);std::snprintf(out.name,sizeof(out.name),"ffn_moe_out-%d",l);
 }
 Hook hook(uint32_t mode,Worker*w=nullptr){Hook h;h.epoch.run=11;h.epoch.context=17;h.epoch.arm=h.epoch.mode=mode;h.epoch.model=mid;h.epoch.K=K;h.pilot={{l,pilots(mid).at(l)}};h.I=I;h.worker=w;h.begin(1,2,128,2);return h;}
 void close(Hook&h){need(h.callback(&weighted,false)&&h.callback(&out,false),"synthetic alpha/out closure");h.finish();}
};
static void hook_tests(const std::string&python,const std::string&worker){
 for(int mid:{1,2}){int l=mid==1?1:0;std::vector<std::string>argv{python,worker,"--synthetic","--model",std::to_string(mid)};Worker w(argv,mid);
  Fixture f(mid,l);auto before=f.down.bytes;auto h=f.hook(CANONICAL,&w);need(h.callback(&f.up,true)&&h.snap.empty()&&f.down.bytes==before,"ask metadata does not read/write");need(h.callback(&f.up,false),"owned up");auto owned=h.snap.at(l).x;f.x.put(0,999.f);need(h.snap.at(l).x==owned,"snapshot owns input");need(h.callback(&f.down.t,false),h.error.c_str());auto ids=read_ids(&f.ids.t,2,f.K);auto selected=mask(ids,2,f.K,pilots(mid).at(l));auto expected=before;
  for(auto&s:selected)for(int j=0;j<2048;++j){float value=owned[s.t*2048+j]+float(s.e)/64;std::memcpy(expected.data()+s.t*f.down.t.nb[2]+s.k*f.down.t.nb[1]+4*j,&value,4);}need(f.down.bytes==expected,"only pilot slots changed; padding/nonpilot preserved");f.close(h);
  Fixture copy(mid,l);auto hc=copy.hook(COPY_ORIGINAL);auto prior=copy.down.bytes;need(hc.callback(&copy.up,false)&&hc.callback(&copy.down.t,false),"copyoriginal");need(copy.down.bytes==prior,"copyoriginal byte identity");copy.close(hc);
  // Request with zero pilot mask still round-trips and consumes exactly once.
  Fixture zero(mid,l,true);auto hz=zero.hook(LOCAL,&w);hz.epoch.context=18;auto bz=zero.down.bytes;need(hz.callback(&zero.up,false)&&hz.callback(&zero.down.t,false)&&hz.snap.at(l).consumed,"zero-mask consume");need(zero.down.bytes==bz,"zero-mask untouched");zero.close(hz);
  Fixture duplicate(mid,l);auto hd=duplicate.hook(OBSERVER);need(hd.callback(&duplicate.up,false)&&hd.callback(&duplicate.down.t,false),"first consume");need(!hd.callback(&duplicate.down.t,false)&&hd.failed,"duplicate Down rejects");
  Fixture stale(mid,l);auto hs=stale.hook(OBSERVER);need(hs.callback(&stale.up,false),"stale setup");hs.epoch.context++;need(!hs.callback(&stale.down.t,false)&&hs.failed,"cross-context snapshot rejects");
  Fixture missing(mid,l);auto hm=missing.hook(OBSERVER);need(!hm.callback(&missing.down.t,false),"missing up rejects");
  Fixture badlayout(mid,l);auto hb=badlayout.hook(OBSERVER);need(hb.callback(&badlayout.up,false),"layout setup");badlayout.down.t.nb[1]=4;need(!hb.callback(&badlayout.down.t,false),"overlapping Down layout rejects");
  w.finish();
 }
 for(std::string fault:{"short","misepoch","duplicate","nonfinite","die"}){
  Fixture f(1,1);auto prior=f.down.bytes;Worker w({python,worker,"--synthetic","--model","1","--fault",fault},1);auto h=f.hook(CANONICAL,&w);need(h.callback(&f.up,false),"fault up");need(!h.callback(&f.down.t,false)&&h.failed,"fault abort prompt");need(f.down.bytes==prior,"bad reply never writes any graph slot");
 }
}
int main(int argc,char**argv){try{signal(SIGPIPE,SIG_IGN);need(argc==3&&environment(),"synthetic CLI/environment");llama_backend_init();auto pp=ggml_threadpool_params_default(8);pp.poll=0;auto*p=ggml_threadpool_new(&pp);need(check_pool(p),"all actual GGML workers strict RN");for(uint32_t mode:{0x2000u,0x4000u,0x6000u})need(!check_pool(p,7,mode)&&check_pool(p),"MXCSR-only worker non-RN rejected/restored");uint32_t mx=_mm_getcsr();_mm_setcsr(mx|0x2000);need(std::fegetround()==FE_TONEAREST&&!environment(),"x87 RN is insufficient");_mm_setcsr(mx);hook_tests(argv[1],argv[2]);std::vector<llama_token>gt(256);for(int j=0;j<256;++j)gt[j]=j%3;auto ix=target_index(256,gt);need(ix.size()==256&&ix.front()==std::make_pair(255,0)&&ix.back()==std::make_pair(510,0),"groundtruth target index");float logits[]={-10000,0,10000};need(nll(logits,3,2)==0&&nll(logits,3,0)==20000,"stable groundtruth NLL");rejects([&]{nll(logits,3,3);});ggml_threadpool_free(p);llama_backend_free();std::cout<<json({{"status","passed"},{"generated_synthetic_only",true},{"wire_bytes",sizeof(Header)},{"GGML_workers",8},{"mask_stride_snapshot_epoch_fault_tests",true},{"target_index_256",true}}).dump()<<'\n';return 0;}catch(const std::exception&e){std::cerr<<"Stage2 synthetic invalid: "<<e.what()<<'\n';return 1;}}
