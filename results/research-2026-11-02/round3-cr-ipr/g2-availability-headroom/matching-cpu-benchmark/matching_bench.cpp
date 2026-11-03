#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

using Masks = std::array<uint8_t, 6>;
struct Input { std::array<int, 6> experts; Masks masks; };
struct Result { uint8_t feasible; uint8_t attempts; std::array<int8_t, 6> group; };
static_assert(sizeof(Result) == 8);
static std::atomic<uint64_t> observable_sink{0};

template<int Cap>
bool augment(int e, const Masks& masks, std::array<int8_t,6*Cap>& owner,
             std::array<int8_t,6>& assigned, uint16_t& seen) {
  for (int g=0; g<6; ++g) if (masks[e] & (1u<<g)) {
    for (int slot=0; slot<Cap; ++slot) {
      const int key=g*Cap+slot;
      if (seen & (1u<<key)) continue;
      seen |= uint16_t(1u<<key);
      if (owner[key]<0 || augment<Cap>(owner[key],masks,owner,assigned,seen)) {
        owner[key]=int8_t(e); assigned[e]=int8_t(g); return true;
      }
    }
  }
  return false;
}

template<int Cap>
__attribute__((noinline)) Result matching(const Masks& masks) {
  std::array<int8_t,6*Cap> owner; owner.fill(-1);
  Result out{1,1,{}}; out.group.fill(-1);
  for (int e=0; e<6; ++e) {
    uint16_t seen=0;
    if (!augment<Cap>(e,masks,owner,out.group,seen)) { out.feasible=0; break; }
  }
  return out;
}

__attribute__((noinline)) Result fallback(const Masks& masks) {
  Result out=matching<1>(masks);
  if (!out.feasible) { out=matching<2>(masks); out.attempts=2; }
  return out;
}

__attribute__((noinline)) Result control(const Masks& masks) {
  Result out{1,0,{}};
  for (int i=0;i<6;++i) out.group[i]=int8_t(masks[i]&7u);
  return out;
}

std::array<uint8_t,64> supports() {
  std::array<uint8_t,64> out{}; int e=0;
  for(int a=0;a<6;++a) for(int b=a+1;b<6;++b)
    for(int c=b+1;c<6;++c) for(int d=c+1;d<6;++d)
      for(int copy=0;copy<4;++copy) out[e++]=uint8_t((1u<<a)|(1u<<b)|(1u<<c)|(1u<<d));
  out[60]=7; out[61]=56; out[62]=25; out[63]=38;
  if(e!=60) throw std::runtime_error("support construction");
  return out;
}

uint64_t hash_bytes(uint64_t h,const void* data,size_t n) {
  const auto* bytes=static_cast<const uint8_t*>(data);
  for(size_t i=0;i<n;++i) {h^=bytes[i];h*=1099511628211ull;}
  return h;
}
std::string hex(uint64_t x) {std::ostringstream s;s<<std::hex<<std::setw(16)<<std::setfill('0')<<x;return s.str();}

template<int Mode>
std::pair<int64_t,uint64_t> measure(const std::vector<Input>& inputs,int calls,int sample,
                                   std::vector<Result>& outputs) {
  const auto begin=std::chrono::steady_clock::now();
  for(int i=0;i<calls;++i) {
    const Masks& masks=inputs[(size_t(i)+size_t(sample)*17u)%inputs.size()].masks;
    std::atomic_signal_fence(std::memory_order_seq_cst);
    if constexpr(Mode==0) outputs[i]=matching<1>(masks);
    if constexpr(Mode==1) outputs[i]=matching<2>(masks);
    if constexpr(Mode==2) outputs[i]=fallback(masks);
    if constexpr(Mode==3) outputs[i]=control(masks);
    std::atomic_signal_fence(std::memory_order_seq_cst);
  }
  const auto end=std::chrono::steady_clock::now();
  // Outside timed interval: all outputs feed an observable atomic checksum.
  const uint64_t h=hash_bytes(1469598103934665603ull,outputs.data(),outputs.size()*sizeof(Result));
  observable_sink.fetch_xor(h,std::memory_order_relaxed);
  return {std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count(),h};
}

template<int Mode>
void bench(const std::string& profile,const std::vector<Input>& inputs,int samples,int calls,
           std::ofstream& csv,uint64_t& checksum) {
  static const char* names[]={"cap1","cap2","cap1_then_cap2","harness_control"};
  std::vector<Result> output(static_cast<size_t>(calls));
  for(int warm=0;warm<4;++warm) measure<Mode>(inputs,calls,warm,output);
  for(int sample=0;sample<samples;++sample) {
    const auto [ns,h]=measure<Mode>(inputs,calls,sample,output);
    csv<<profile<<'\t'<<names[Mode]<<'\t'<<sample<<'\t'<<ns<<'\t'<<calls<<'\t'<<hex(h)<<'\n';
    std::string key=profile+":"+names[Mode]+":"+std::to_string(sample)+":"+hex(h);
    checksum^=hash_bytes(1469598103934665603ull,key.data(),key.size());
  }
}

int main(int argc,char** argv) {
  if(argc!=6) {std::cerr<<"routes.txt outputprefix samples calls run_index\n";return 2;}
  const std::string prefix=argv[2];const int samples=std::stoi(argv[3]),calls=std::stoi(argv[4]),run=std::stoi(argv[5]);
  const auto resident=supports();std::ifstream input(argv[1]);std::vector<Input> all;
  while(true) {
    Input x{}; if(!(input>>x.experts[0]))break;
    for(int i=1;i<6;++i)if(!(input>>x.experts[i]))throw std::runtime_error("truncated route");
    if(!std::is_sorted(x.experts.begin(),x.experts.end()) || std::adjacent_find(x.experts.begin(),x.experts.end())!=x.experts.end())throw std::runtime_error("bad route");
    for(int i=0;i<6;++i) {if(x.experts[i]<0||x.experts[i]>=64)throw std::runtime_error("bad expert");x.masks[i]=resident[x.experts[i]];}
    all.push_back(x);
  }
  if(all.empty()||samples<0||calls<=0)throw std::runtime_error("empty workload");
  std::array<std::string,4> pname={"mixed_fresh_synthetic","cap1_success","cap1_failure","deliberate_Hall_failure"};
  std::array<std::vector<Input>,4> profiles;profiles[0]=all;
  for(const auto& x:all)profiles[matching<1>(x.masks).feasible?1:2].push_back(x);
  Input hall{{40,41,42,43,44,45},{}};for(int i=0;i<6;++i)hall.masks[i]=resident[hall.experts[i]];profiles[3].push_back(hall);
  for(const auto& v:profiles)if(v.empty())throw std::runtime_error("empty stratified profile");
  std::ofstream semantic(prefix+".semantic.tsv");
  semantic<<"input_index\tmode\tfeasible\tattempts\te0\te1\te2\te3\te4\te5\tg0\tg1\tg2\tg3\tg4\tg5\n";
  std::vector<Input> checks=all;checks.push_back(hall);
  for(size_t i=0;i<checks.size();++i) for(int mode=0;mode<3;++mode) {
    const auto& x=checks[i];Result r= mode==0 ? matching<1>(x.masks) : mode==1 ? matching<2>(x.masks) : fallback(x.masks);
    semantic<<i<<'\t'<<mode<<'\t'<<int(r.feasible)<<'\t'<<int(r.attempts);
    for(int e:x.experts)semantic<<'\t'<<e;for(int8_t g:r.group)semantic<<'\t'<<int(g);semantic<<'\n';
  }
  if(samples==0) {
    std::ofstream receipt(prefix+".receipt.json");
    receipt<<"{\"status\":\"SEMANTICS_ONLY\",\"routes\":"<<all.size()<<",\"profile_sizes\":{";
    for(int i=0;i<4;++i)receipt<<(i?",":"")<<'\"'<<pname[i]<<"\":"<<profiles[i].size();
    receipt<<"}}\n";return 0;
  }
  std::ofstream csv(prefix+".samples.tsv");csv<<"profile\tmode\tsample\telapsed_ns\tcalls\tchecksum\n";
  uint64_t checksum=0;
  for(int pi=0;pi<4;++pi) {
    int p=(pi+run)%4;
    for(int mi=0;mi<4;++mi) {
      int mode=(mi+run)%4;
      if(mode==0)bench<0>(pname[p],profiles[p],samples,calls,csv,checksum);
      if(mode==1)bench<1>(pname[p],profiles[p],samples,calls,csv,checksum);
      if(mode==2)bench<2>(pname[p],profiles[p],samples,calls,csv,checksum);
      if(mode==3)bench<3>(pname[p],profiles[p],samples,calls,csv,checksum);
    }
    std::cout<<"run "<<run<<" profile "<<pname[p]<<" complete\n"<<std::flush;
  }
  std::ofstream receipt(prefix+".receipt.json");
  receipt<<"{\"status\":\"PASS\",\"routes\":"<<all.size()<<",\"samples_per_mode_profile\":"<<samples
         <<",\"calls_per_sample\":"<<calls<<",\"warmup_batches\":4,\"checksum\":\""<<hex(checksum)<<"\",\"sink\":\""<<hex(observable_sink.load())<<"\",\"profile_sizes\":{";
  for(int i=0;i<4;++i)receipt<<(i?",":"")<<'\"'<<pname[i]<<"\":"<<profiles[i].size();
  receipt<<"}}\n";
}
