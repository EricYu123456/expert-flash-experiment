#include <immintrin.h>
#include <omp.h>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <atomic>

// Explicit AVX2/FMA: eight independent F32 lanes, then this exact pair tree.
// A sample's arithmetic is identical in B1 and batched calls.
static bool environment() {
    return std::fegetround()==FE_TONEAREST && (_mm_getcsr() & ((1u<<15)|(1u<<6)|0x6000u))==0;
}
// Focused environment-test ABI; the kernels never alter caller rounding.
extern "C" uint32_t lac_get_mxcsr() {return _mm_getcsr();}
extern "C" void lac_set_mxcsr(uint32_t value) {_mm_setcsr(value);}
extern "C" int lac_x87_nearest() {return std::fegetround()==FE_TONEAREST;}
extern "C" int lac_environment() {
    std::atomic<int> good{0};
    #pragma omp parallel num_threads(8)
    { if (environment()) ++good; }
    return good.load();
}
template<class W> static float dot(const float* x,const W* w,int K,bool scalar) {
    alignas(32) float lane[8]={};
    if (scalar) {
        for(int k=0;k<K;++k) lane[k%8]=std::fma(x[k],float(w[k]),lane[k%8]);
    } else {
        __m256 a=_mm256_setzero_ps(); int k=0;
        for(;k+8<=K;k+=8) {
            __m256 v;
            if constexpr(sizeof(W)==1) {
                __m128i b=_mm_loadl_epi64(reinterpret_cast<const __m128i*>(w+k));
                v=_mm256_cvtepi32_ps(_mm256_cvtepi8_epi32(b));
            } else v=_mm256_loadu_ps(reinterpret_cast<const float*>(w+k));
            a=_mm256_fmadd_ps(_mm256_loadu_ps(x+k),v,a);
        }
        _mm256_store_ps(lane,a);
        for(;k<K;++k) lane[k%8]=std::fma(x[k],float(w[k]),lane[k%8]);
    }
    float p0=lane[0]+lane[1],p1=lane[2]+lane[3],p2=lane[4]+lane[5],p3=lane[6]+lane[7];
    float a=p0+p1,b=p2+p3; return a+b;
}
template<class W> static int kernel(const float* x,const W* w,int N,int K,int R,float* out,bool scalar) {
    if(N<0||K<=0||R<=0||lac_environment()!=8)return 1;
    std::atomic<int> fault{0};
    #pragma omp parallel for schedule(static) num_threads(8)
    for(int64_t j=0;j<int64_t(N)*R;++j) {
        float a=dot(x+(j/R)*K,w+(j%R)*K,K,scalar);
        if(!std::isfinite(a))fault=1;
        out[j]=a;
    }
    return fault.load();
}
extern "C" int lac_f32(const float* x,const float* w,int N,int K,int R,float* out) {return kernel(x,w,N,K,R,out,false);}
extern "C" int lac_i8(const float* x,const int8_t* w,int N,int K,int R,float* out) {return kernel(x,w,N,K,R,out,false);}
extern "C" int lac_reference(const float* x,const float* w,int N,int K,int R,float* out) {return kernel(x,w,N,K,R,out,true);}
