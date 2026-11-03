// Host partial-result reduction only. No storage or inference timing claims.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>
#include <sched.h>

using timer = std::chrono::steady_clock;
static volatile float sink_value = 0;

template<class F> static double trial(F fn, int repeat) {
    auto begin = timer::now();
    for (int i=0; i<repeat; ++i) {
        fn(i);
        asm volatile("" ::: "memory");
    }
    return std::chrono::duration<double,std::nano>(timer::now()-begin).count()/repeat;
}

int main(int argc, char **argv) {
    try {
        if (argc != 2) throw std::runtime_error("usage: BENCH OUT.json");
        cpu_set_t allowed, selected;
        CPU_ZERO(&allowed); CPU_ZERO(&selected);
        int cpu = -1;
        if (sched_getaffinity(0,sizeof(allowed),&allowed)==0) {
            for (int i=0;i<CPU_SETSIZE;++i) if (CPU_ISSET(i,&allowed)) {cpu=i;break;}
            if (cpu>=0) { CPU_SET(cpu,&selected); if(sched_setaffinity(0,sizeof(selected),&selected))cpu=-1; }
        }
        std::ofstream out(argv[1]);
        if (!out) throw std::runtime_error("cannot open output");
        out << "{\"purpose\":\"same-host array reduction microbenchmark, not native AiF\","
               "\"pinned_cpu\":" << cpu << ",\"warmup_trials\":3,\"reported_trials\":31,\"cases\":[";
        std::mt19937 gen(20261102);
        int case_index=0;
        for (int rows : {6*1024,8*2048}) for (int slots : {1,1024}) {
            const size_t length=(size_t)rows*slots;
            std::vector<float>a(length),b(length),dest(rows),scale(rows);
            std::vector<int32_t>ia(length),ib(length);
            for(size_t i=0;i<length;++i) {
                ia[i]=(int32_t)(gen()%16000001)-8000000;
                ib[i]=(int32_t)(gen()%16000001)-8000000;
                a[i]=(float)ia[i];b[i]=(float)ib[i];
            }
            for(int i=0;i<rows;++i)scale[i]=1e-6f+(float)(gen()%1000)*1e-8f;
            // Each partial integer is exactly representable by FP32. The sum
            // fits INT32; INT64 conversion models a conservative host contract.
            for(int i=0;i<rows;++i) {
                const int64_t integer=(int64_t)a[i]+(int64_t)b[i];
                if(integer!=(int64_t)ia[i]+ib[i])throw std::runtime_error("integer control failed");
            }
            const int repeat=slots==1?10000:2048;
            std::vector<std::vector<double>>times(4);
            for(int t=-3;t<31;++t) {
                // Rotate the order to limit monotonic clock/frequency bias.
                for(int order=0;order<4;++order) {
                    const int kind=(order+(t+3))%4;
                    const double value=trial([&](int j){
                        const size_t offset=(size_t)(j%slots)*rows;
                        const float *x=a.data()+offset,*y=b.data()+offset;
                        const int32_t *u=ia.data()+offset,*v=ib.data()+offset;
                        float *z=dest.data();
                        if(kind==0)std::memcpy(z,x,rows*sizeof(float));
                        else if(kind==1)for(int i=0;i<rows;++i)z[i]=x[i]+y[i];
                        else if(kind==2)for(int i=0;i<rows;++i)
                            z[i]=(float)((int64_t)u[i]+(int64_t)v[i])*scale[i];
                        else for(int i=0;i<rows;++i)
                            z[i]=(float)((int64_t)x[i]+(int64_t)y[i])*scale[i];
                        sink_value=z[j%rows];
                    },repeat);
                    if(t>=0)times[kind].push_back(value);
                }
            }
            if(case_index++)out << ',';
            out << "{\"elements\":" << rows << ",\"source_slots\":" << slots
                << ",\"source_pair_bytes\":" << length*sizeof(float)*2
                << ",\"destination_bytes\":" << rows*sizeof(float) << ",\"operations\":[";
            const char *labels[]={"baseline_fp32_copy","fp32_partial_add","i32_add_i64_rescale","fp32_exact_integer_extract_i64_rescale"};
            for(int k=0;k<4;++k) {
                auto values=times[k];std::sort(values.begin(),values.end());
                if(k)out<<',';
                out << "{\"operation\":\"" << labels[k] << "\",\"median_ns\":" << values[15]
                    << ",\"p10_ns\":" << values[3] << ",\"p90_ns\":" << values[27]
                    << ",\"trial_ns\":[";
                for(size_t i=0;i<times[k].size();++i)out<<(i?",":"")<<times[k][i];
                out<<"]}";
            }
            out<<"]}";
        }
        out<<"],\"limitations\":[\"ordinary heap buffers, one local CPU, warm allocation; streaming slots may still hit LLC\","
               "\"does not include NVMe DMA, completion, lookup, page gather, activation or full model time\","
               "\"i32/FP32 integer output contracts are benchmark alternatives, not verified native AiF interfaces\","
               "\"row scale application is a synthetic array operation; real GGUF block scales differ\"]}\n";
        std::printf("host reduction microbenchmark completed; sink=%g\n",(double)sink_value);
        return 0;
    } catch(const std::exception &e) {std::fprintf(stderr,"%s\n",e.what());return 1;}
}
