#pragma once
#include <cmath>
#include <cstdint>
// Exact deployment order: gather -> F32 subtraction -> F32 multiplication.
// Both the numerical sanity library and the live adapter include this body.
inline int sf_apply_transform(int N,int H,int Z,int R,const float*x,const float*z,const uint8_t*sources,const int32_t*indices,const float*center,const float*invscale,float*out){
 if(N<1||H<1||Z<1||R<1||R>10944)return 1;
 for(int j=0;j<R;++j)if(sources[j]>1||indices[j]<0||indices[j]>=(sources[j]?Z:H)||!std::isfinite(center[j])||!(invscale[j]>0)||!std::isfinite(invscale[j]))return 2;
 #pragma omp parallel for schedule(static)
 for(int64_t k=0;k<(int64_t)N*R;++k){int t=k/R,j=k%R;const float raw=sources[j]?z[(int64_t)t*Z+indices[j]]:x[(int64_t)t*H+indices[j]];const float shifted=raw-center[j];out[k]=shifted*invscale[j];}
 for(int64_t k=0;k<(int64_t)N*R;++k)if(!std::isfinite(out[k]))return 3;return 0;
}
