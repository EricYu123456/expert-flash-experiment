// Canonical CPU arithmetic for transformed shared features. Compile with
// -fno-fast-math -ffp-contract=off. Native SwiGLU comes from pinned llama.cpp.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <exception>
#include <stdexcept>
#include <vector>
#include <omp.h>
#include "research_20261109_shared_feature_transform.hpp"
extern "C" void ggml_vec_swiglu_f32(int,float*,const float*,const float*);
static float quantize(const float*x,int n,int8_t*q){
 float amax=0;for(int j=0;j<n;++j){if(!std::isfinite(x[j]))throw std::runtime_error("nonfinite input");amax=std::max(amax,std::abs(x[j]));}
 const float s=amax==0?1.0f:amax/127.0f;if(!(s>0)||!std::isfinite(s))throw std::runtime_error("quantization scale");
 for(int j=0;j<n;++j)q[j]=(int8_t)std::clamp((int)std::round(x[j]/s),-127,127);return s;
}
extern "C" void sf_set_threads(int n){omp_set_num_threads(n);}
extern "C" int sf_transform(int N,int H,int Z,int R,const float*x,const float*z,const uint8_t*sources,const int32_t*indices,const float*center,const float*invscale,float*out){try{
 return sf_apply_transform(N,H,Z,R,x,z,sources,indices,center,invscale,out);
 }catch(...){return 9;}}
extern "C" int sf_quantize(int N,int C,const float*x,int8_t*q,float*s){try{
 if(N<1||C<1)return 1;for(int t=0;t<N;++t)s[t]=quantize(x+(int64_t)t*C,C,q+(int64_t)t*C);return 0;
 }catch(...){return 9;}}
extern "C" int sf_rescale_exact_dots(int N,int R,int C,const double*d,const float*sw,const float*sx,float*out){
 if(N<1||R<1||C<1||C>10944)return 1;const double limit=(double)C*127*127;
 for(int r=0;r<R;++r)if(!(sw[r]>0)||!std::isfinite(sw[r]))return 2;
 for(int t=0;t<N;++t)if(!(sx[t]>0)||!std::isfinite(sx[t]))return 2;
 int failed=0;
 #pragma omp parallel for schedule(static) reduction(|:failed)
 for(int64_t k=0;k<(int64_t)N*R;++k){double v=d[k];if(!std::isfinite(v)||std::abs(v)>limit||std::trunc(v)!=v){failed=1;continue;}int64_t sum=(int64_t)v;const float scale=sw[k%R]*sx[k/R];const float f=(float)sum;out[k]=f*scale;}
 return failed?3:0;
}
extern "C" int sf_swiglu(int N,int I,const float*gate,const float*up,float*out){
 if(N<1||I<1)return 1;
 #pragma omp parallel for schedule(static)
 for(int t=0;t<N;++t)ggml_vec_swiglu_f32(I,out+(int64_t)t*I,gate+(int64_t)t*I,up+(int64_t)t*I);
 for(int64_t j=0;j<(int64_t)N*I;++j)if(!std::isfinite(out[j]))return 2;return 0;
}
extern "C" int sf_predict(int N,int C,int R,const float*x,const int8_t*w,const float*sw,const float*bias,float*out){try{
 if(N<1||C<1||C>10944||R<1)return 1;
 std::vector<int8_t>q((int64_t)N*C);std::vector<float>sx(N);for(int t=0;t<N;++t)sx[t]=quantize(x+(int64_t)t*C,C,q.data()+(int64_t)t*C);
 #pragma omp parallel for schedule(static)
 for(int64_t k=0;k<(int64_t)N*R;++k){int t=k/R,r=k%R;int32_t sum=0;
  #pragma omp simd reduction(+:sum)
  for(int j=0;j<C;++j)sum+=(int32_t)q[(int64_t)t*C+j]*(int32_t)w[(int64_t)r*C+j];
  const float scale=sw[r]*sx[t];const float f=(float)(int64_t)sum;const float product=f*scale;out[k]=product+bias[r];}
 return 0;
 }catch(...){return 9;}}
