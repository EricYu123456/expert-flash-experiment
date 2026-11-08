// Standalone, independently compiled numerical probe for the new runtime.
#include "research_20261109_factorized_readout.hpp"
#include <omp.h>
extern "C" void sf_factorized_set_threads(int n){omp_set_num_threads(n);}
extern "C" int sf_factorized_predict_q(
    int N,int C,int Q,int R,const int8_t*qx,const float*sx,
    const int8_t*V,const float*sv,const float*center,const float*gain,
    const int8_t*U,const float*su,const float*bias,float*out,
    float*raw,float*affine,int8_t*codes,float*scales){
    try{return sf_factorized_qinput(N,C,Q,R,qx,sx,V,sv,center,gain,U,su,bias,out,raw,affine,codes,scales);}
    catch(...){return 9;}
}
