#pragma once
// New two-stage readout kernel; frozen 11/08/11/09 numeric files are untouched.
// Input is the already encoded layer+basis A8 row. Intermediate encoding is
// private to this expert: V -> F32 affine -> full-vector A8 -> U + final bias.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

static inline int sf_factorized_qinput(
    int N,int C,int Q,int R,const int8_t*qx,const float*sx,
    const int8_t*V,const float*sv,const float*center,const float*gain,
    const int8_t*U,const float*su,const float*bias,float*out,
    float*raw_latent=nullptr,float*normalized_latent=nullptr,
    int8_t*latent_codes=nullptr,float*latent_scales=nullptr){
    if(N<1||C<1||C>10944||Q<1||Q>10944||R<1)return 1;
    for(int n=0;n<N;++n)if(!(sx[n]>0&&std::isfinite(sx[n])))return 2;
    for(int q=0;q<Q;++q)if(!(sv[q]>0&&std::isfinite(sv[q])&&std::isfinite(center[q])&&gain[q]>0&&std::isfinite(gain[q])))return 2;
    for(int r=0;r<R;++r)if(!(su[r]>0&&std::isfinite(su[r])&&std::isfinite(bias[r])))return 2;
    std::vector<float>affine((size_t)N*Q),scales(N);
    std::vector<int8_t>codes((size_t)N*Q);
    int bad=0;
    #pragma omp parallel for schedule(static) reduction(|:bad)
    for(int64_t k=0;k<(int64_t)N*Q;++k){
        int n=k/Q,q=k%Q;int32_t sum=0;
        #pragma omp simd reduction(+:sum)
        for(int j=0;j<C;++j)sum+=(int32_t)qx[(size_t)n*C+j]*(int32_t)V[(size_t)q*C+j];
        const float scale=sv[q]*sx[n];
        const float integer=(float)(int64_t)sum;
        const float value=integer*scale;
        const float shifted=value-center[q];
        const float transformed=shifted*gain[q];
        if(!std::isfinite(value)||!std::isfinite(transformed))bad=1;
        if(raw_latent)raw_latent[k]=value;
        if(normalized_latent)normalized_latent[k]=transformed;
        affine[k]=transformed;
    }
    if(bad)return 3;
    #pragma omp parallel for schedule(static) reduction(|:bad)
    for(int n=0;n<N;++n){
        float amax=0;
        for(int q=0;q<Q;++q)amax=std::max(amax,std::abs(affine[(size_t)n*Q+q]));
        const float scale=amax==0?1.0f:amax/127.0f;
        if(!(scale>0&&std::isfinite(scale))){bad=1;continue;}
        scales[n]=scale;
        if(latent_scales)latent_scales[n]=scale;
        for(int q=0;q<Q;++q){
            const float quotient=affine[(size_t)n*Q+q]/scale;
            const int8_t code=(int8_t)std::clamp((int)std::round(quotient),-127,127);
            codes[(size_t)n*Q+q]=code;
            if(latent_codes)latent_codes[(size_t)n*Q+q]=code;
        }
    }
    if(bad)return 3;
    #pragma omp parallel for schedule(static) reduction(|:bad)
    for(int64_t k=0;k<(int64_t)N*R;++k){
        int n=k/R,r=k%R;int32_t sum=0;
        #pragma omp simd reduction(+:sum)
        for(int q=0;q<Q;++q)sum+=(int32_t)codes[(size_t)n*Q+q]*(int32_t)U[(size_t)r*Q+q];
        const float scale=su[r]*scales[n];
        const float integer=(float)(int64_t)sum;
        const float product=integer*scale;
        const float value=product+bias[r];
        if(!std::isfinite(value))bad=1;
        out[k]=value;
    }
    return bad?3:0;
}
