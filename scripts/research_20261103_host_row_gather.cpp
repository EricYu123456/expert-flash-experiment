// Optional host fallback microbenchmark, not DMA/device timing calibration.
#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <numeric>
#include <sched.h>
#include <stdexcept>
#include <string>
#include <vector>

struct Band { uint32_t chip, source, destination, rows; };
static_assert(sizeof(Band)==16);
struct Workload {
  std::string name, model, provenance;
  uint32_t layers, output_rows;
  std::array<uint32_t,16> source_rows{};
  std::vector<Band> bands;
};
struct Replica {
  std::array<uint32_t*,16> source{};
  uint32_t* output=nullptr;
  float* scales=nullptr;
  ~Replica() { for(auto p:source)std::free(p);std::free(output);std::free(scales); }
};
template<typename T> T* allocate(size_t count) {
  void* p=nullptr;
  if(posix_memalign(&p,64,std::max<size_t>(64,count*sizeof(T))))throw std::bad_alloc();
  return static_cast<T*>(p);
}
uint32_t bits(float x) { uint32_t b;std::memcpy(&b,&x,4);return b; }
int32_t integer_pattern(uint32_t row) {return int32_t(row%2048)-1024;}
float scale_pattern(uint32_t row) {return float(1u<<(row%4))*0.5f;}
uint32_t expected(uint32_t row,bool rescale) {
  return bits(rescale ? float(integer_pattern(row))*scale_pattern(row)
                      : float(int32_t(row%4096)-2048)*0.125f);
}
void initialize(Replica& a,const Workload& w,bool rescale) {
  for(unsigned c=0;c<16;++c) {
    a.source[c]=allocate<uint32_t>(w.source_rows[c]);
    std::fill_n(a.source[c],w.source_rows[c],0x7fc00000u);
  }
  a.output=allocate<uint32_t>(w.output_rows);
  std::fill_n(a.output,w.output_rows,0xdeadbeefu);
  if(rescale) {
    a.scales=allocate<float>(w.output_rows);
    for(uint32_t i=0;i<w.output_rows;++i)a.scales[i]=scale_pattern(i);
  }
  for(const auto& b:w.bands)for(uint32_t i=0;i<b.rows;++i) {
    const uint32_t r=b.destination+i;
    if(rescale) {
      const int32_t v=integer_pattern(r);
      std::memcpy(a.source[b.chip]+b.source+i,&v,4);
    } else a.source[b.chip][b.source+i]=expected(r,false);
  }
}
__attribute__((noinline)) void gather(Replica& a,const Workload& w,bool rescale) {
  for(const auto& b:w.bands) {
    const auto* src=a.source[b.chip]+b.source;
    auto* dst=a.output+b.destination;
    if(!rescale)std::memcpy(dst,src,size_t(b.rows)*4);
    else {
      const auto* scales=a.scales+b.destination;
      for(uint32_t i=0;i<b.rows;++i) {
        int32_t value;std::memcpy(&value,src+i,4);
        const float result=float(value)*scales[i];
        std::memcpy(dst+i,&result,4);
      }
    }
  }
  asm volatile("" : : "r"(a.output) : "memory");
}
uint64_t verify(const Replica& a,const Workload& w,bool rescale) {
  uint64_t checksum=0;
  for(uint32_t i=0;i<w.output_rows;++i) {
    if(a.output[i]!=expected(i,rescale))throw std::runtime_error("bitwise copy/rescale mismatch");
    checksum+=uint64_t(a.output[i])*(uint64_t(i)+1);
  }
  return checksum;
}
int main(int argc,char** argv) {
  if(argc!=6)throw std::runtime_error("workload output cpu samples stream_target_MiB");
  const int cpu=std::stoi(argv[3]);const size_t samples=std::stoull(argv[4]);
  const size_t stream_target=std::stoull(argv[5])<<20;
  cpu_set_t mask;CPU_ZERO(&mask);CPU_SET(cpu,&mask);
  if(sched_setaffinity(0,sizeof(mask),&mask))throw std::runtime_error("CPU affinity failed");
  std::ifstream input(argv[1]);size_t count;input>>count;
  std::vector<Workload> workloads(count);
  for(auto& w:workloads) {
    size_t bands;input>>w.name>>w.model>>w.layers>>w.output_rows>>bands>>w.provenance;
    for(auto& n:w.source_rows)input>>n;
    w.bands.resize(bands);
    for(auto& b:w.bands)input>>b.chip>>b.source>>b.destination>>b.rows;
    std::vector<unsigned char> cover(w.output_rows);
    for(const auto& b:w.bands) {
      if(b.chip>=16 || uint64_t(b.source)+b.rows>w.source_rows[b.chip] ||
         uint64_t(b.destination)+b.rows>w.output_rows)throw std::runtime_error("invalid band");
      for(uint32_t i=0;i<b.rows;++i)if(cover[b.destination+i]++)throw std::runtime_error("overlap");
    }
    if(std::find(cover.begin(),cover.end(),0)!=cover.end())throw std::runtime_error("missing row");
  }
  if(!input)throw std::runtime_error("workload parse failed");
  std::ofstream output(argv[2]);output<<std::setprecision(17);
  output<<"{\"cpu\":"<<cpu<<",\"thread_count\":1,\"band_descriptor_bytes\":"<<sizeof(Band)
        <<",\"records\":[";
  bool first=true;
  for(const auto& w:workloads)for(bool rescale:{false,true}) {
    // Conditional I32+row-scale mode is intentionally limited to the two
    // relevant Qwen controls; no native-I32 capability is asserted.
    if(rescale && w.name!="qwen38_whole_row" && w.name!="qwen38_overlap_best")continue;
    for(bool streaming:{false,true}) {
      const size_t source_rows=std::accumulate(w.source_rows.begin(),w.source_rows.end(),size_t(0));
      const size_t one_bytes=4*(source_rows+w.output_rows+(rescale?w.output_rows:0));
      const size_t replicas=streaming?std::max<size_t>(2,(stream_target+one_bytes-1)/one_bytes):1;
      std::vector<Replica> arena(replicas);
      for(auto& a:arena)initialize(a,w,rescale);
      // Initialize and check every replica before measuring; two complete
      // rotating passes prevent the final verification pass warming the set.
      uint64_t checksum=0;
      for(auto& a:arena) {gather(a,w,rescale);verify(a,w,rescale);}
      for(size_t pass=0;pass<2;++pass)for(auto& a:arena)gather(a,w,rescale);
      std::vector<int64_t> times;times.reserve(samples);
      for(size_t i=0;i<samples;++i) {
        auto& a=arena[i%replicas];
        const auto begin=std::chrono::steady_clock::now();gather(a,w,rescale);
        const auto end=std::chrono::steady_clock::now();
        times.push_back(std::chrono::duration_cast<std::chrono::nanoseconds>(end-begin).count());
      }
      for(auto& a:arena)checksum+=verify(a,w,rescale);
      auto ordered=times;std::sort(ordered.begin(),ordered.end());
      const double mean=std::accumulate(times.begin(),times.end(),0.0)/samples;
      const int64_t med=ordered[samples/2],p99=ordered[size_t(std::ceil(.99*samples))-1];
      if(!first)output<<',';first=false;
      output<<"{\"case\":\""<<w.name<<"\",\"model\":\""<<w.model
            <<"\",\"operation\":\""<<(rescale?"conditional_I32_rowScale":"opaque_F32_copy")
            <<"\",\"cache\":\""<<(streaming?"rotating_stream":"warm_single_token")
            <<"\",\"layers\":"<<w.layers<<",\"useful_output_bytes\":"<<4ull*w.output_rows
            <<",\"staging_bytes_including_fringe\":"<<4ull*source_rows
            <<",\"bands\":"<<w.bands.size()<<",\"replicas\":"<<replicas
            <<",\"working_set_bytes\":"<<one_bytes*replicas
            <<",\"samples\":"<<samples<<",\"median_ns\":"<<med<<",\"p99_ns\":"<<p99
            <<",\"mean_ns\":"<<mean<<",\"min_ns\":"<<ordered.front()<<",\"max_ns\":"<<ordered.back()
            <<",\"median_payload_GBps\":"<<double(4ull*w.output_rows)/med
            <<",\"verified_bitwise\":true,\"verification_checksum_sum\":"<<checksum
            <<",\"durations_ns\":[";
      for(size_t i=0;i<times.size();++i){if(i)output<<',';output<<times[i];}
      output<<"]}";
      std::cerr<<w.name<<' '<<(rescale?"I32scale":"F32copy")<<' '<<(streaming?"stream":"warm")
               <<" median_us "<<med/1000.0<<" p99_us "<<p99/1000.0<<'\n';
    }
  }
  output<<"]}\n";
}
