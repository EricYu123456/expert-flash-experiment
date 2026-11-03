// Frozen Round2 numeric bridge. No model or prompt reads in this translation unit.
#include "ggml.h"
#include <algorithm>
#include <cfenv>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <omp.h>
#include <xmmintrin.h>
extern "C" void ggml_vec_silu_f32(int,float*,const float*);
extern "C" void ggml_vec_swiglu_f32(int,float*,const float*,const float*);
static void need(bool b,const char*s){if(!b)throw std::runtime_error(s);}
static float scale(int64_t d,float sw,float sx){float s=sw*sx;float f=(float)d;return f*s;}
extern "C" int la_environment(){return std::fegetround()==FE_TONEAREST&&(_mm_getcsr()&((1<<15)|(1<<6)))==0?int(_mm_getcsr()):-1;}
extern "C" int la_workers_environment(){int bad=0,count=0;
 #pragma omp parallel reduction(|:bad) reduction(+:count)
 {bad|=la_environment()<0;++count;}return bad||count!=8?-1:count;}
extern "C" int la_quantize(const float*x,int n,int width,int8_t*q,float*scales){
 if(la_environment()<0)return 1;
 for(int t=0;t<n;++t){float m=0;for(int j=0;j<width;++j){float a=x[(size_t)t*width+j];if(!std::isfinite(a))return 2;m=std::max(m,std::abs(a));}float s=m==0?1.f:m/127.f;if(!(s>0)||!std::isfinite(s))return 3;scales[t]=s;for(int j=0;j<width;++j)q[(size_t)t*width+j]=(int8_t)std::clamp((int)std::round(x[(size_t)t*width+j]/s),-127,127);}return 0;
}
extern "C" int la_rescale(const double*dots,const float*sw,const float*sx,int n,int outdim,float*out){
 if(la_environment()<0||la_workers_environment()!=8)return 1;int bad=0;
 #pragma omp parallel for schedule(static) reduction(|:bad)
 for(int64_t k=0;k<(int64_t)n*outdim;++k){double d=dots[k];if(!std::isfinite(d)||std::abs(d)>=9007199254740992.||d!=std::trunc(d)){bad=1;continue;}float f=scale((int64_t)d,sw[k%outdim],sx[k/outdim]);out[k]=f;bad|=!std::isfinite(f);}return bad;
}
#ifndef LOCAL_AFFINE_NO_GGML
extern "C" int la_swiglu(const float*g,const float*u,int n,int width,float*out){
 if(la_environment()<0||la_workers_environment()!=8)return 1;int bad=0;
 #pragma omp parallel for schedule(static) reduction(|:bad)
 for(int t=0;t<n;++t){std::vector<float>h(width),fused(width);ggml_vec_silu_f32(width,h.data(),g+(size_t)t*width);ggml_vec_swiglu_f32(width,fused.data(),g+(size_t)t*width,u+(size_t)t*width);for(int j=0;j<width;++j){size_t k=(size_t)t*width+j;out[k]=h[j]*u[k];uint32_t a,b;std::memcpy(&a,&out[k],4);std::memcpy(&b,&fused[j],4);bad|=a!=b||!std::isfinite(out[k]);}}return bad;
}
#endif
extern "C" int la_add_bias(const float*pred,const float*bias,int n,int width,float*out){
 for(int64_t k=0;k<(int64_t)n*width;++k){out[k]=pred[k]+bias[k%width];if(!std::isfinite(out[k]))return 1;}return 0;
}
extern "C" int la_calibrate(const float*y,const float*pred,const double*w,int n,int width,float*bias){
 std::vector<double> sums(width,0.);double sw=0.;for(int t=0;t<n;++t){if(!(w[t]>0)||!std::isfinite(w[t]))return 1;sw=sw+w[t];for(int j=0;j<width;++j){size_t k=(size_t)t*width+j;double residual=double(y[k])-double(pred[k]);double term=w[t]*residual;sums[j]=sums[j]+term;}}if(!(sw>0))return 2;for(int j=0;j<width;++j){bias[j]=(float)(sums[j]/sw);if(!std::isfinite(bias[j]))return 3;}return 0;
}
// Runtime tree's actual rule: F32 input and stored F32 normal, sequential FP64
// multiply then add, with contraction disabled. No IDs or training order here.
extern "C" void la_tree_project(const float*x,const float*normal,int n,int width,double*out){
 for(int t=0;t<n;++t){double s=0;for(int j=0;j<width;++j){double p=double(x[(size_t)t*width+j])*double(normal[j]);s=s+p;}out[t]=s;}
}
extern "C" int la_project(const float*x,const float*P,int n,int width,int rank,float*out){
 if(la_environment()<0||la_workers_environment()!=8)return 1;int bad=0;
 #pragma omp parallel for schedule(static) reduction(|:bad)
 for(int64_t task=0;task<(int64_t)n*rank;++task){const float*a=x+(task/rank)*width,*b=P+(task%rank)*width;double sums[8]={};int j=0;for(;j+7<width;j+=8){for(int k=0;k<8;++k){double p=double(a[j+k])*double(b[j+k]);sums[k]=sums[k]+p;}}for(;j<width;++j){double p=double(a[j])*double(b[j]);sums[0]=sums[0]+p;}double lo=(sums[0]+sums[1])+(sums[2]+sums[3]);double hi=(sums[4]+sums[5])+(sums[6]+sums[7]);out[task]=(float)(lo+hi);bad|=!std::isfinite(out[task]);}return bad;
}
extern "C" void la_round(const float*x,int n,float*out){for(int j=0;j<n;++j)out[j]=std::round(x[j]);}
#ifndef LOCAL_AFFINE_LIBRARY
int main(int argc,char**argv){try{
 omp_set_num_threads(8);if(argc==2&&std::string(argv[1])=="--swiglu-worker"){for(;;){int32_t dims[2];std::cin.read((char*)dims,sizeof(dims));if(std::cin.eof()&&std::cin.gcount()==0)return 0;need(bool(std::cin)&&dims[0]>0&&dims[0]<=256&&dims[1]>0&&dims[1]<=10944,"activation worker framing");size_t count=(size_t)dims[0]*dims[1];std::vector<float>g(count),u(count),a(count);std::cin.read((char*)g.data(),count*4);std::cin.read((char*)u.data(),count*4);need(bool(std::cin),"activation worker input");int32_t rc=la_swiglu(g.data(),u.data(),dims[0],dims[1],a.data());std::cout.write((char*)&rc,4);if(!rc)std::cout.write((char*)a.data(),count*4);std::cout.flush();if(rc)return 1;}}
 omp_set_num_threads(8);need(la_environment()>=0&&la_workers_environment()==8,"all8 worker environments");float v[]={0.f,-0.f,.5f,-.5f,std::nextafter(.5f,0.f),std::nextafter(-.5f,0.f),127.f,-127.f};float y[8];la_round(v,8,y);need(y[2]==1&&y[3]==-1&&y[4]==0&&y[5]==0,"ties away near-half");
 for(int I:{1024,1408,2816,10944}){std::vector<float>g(I),u(I),a(I);for(int j=0;j<I;++j){g[j]=(j%101-50)*.25f;u[j]=(j%79-39)*.125f;}need(la_swiglu(g.data(),u.data(),1,I,a.data())==0,"full-I product bits");}
 float z[4]={};int8_t q[4];float s;need(la_quantize(z,1,4,q,&s)==0&&s==1&&q[0]==0,"zero rule");std::cout<<"{\"status\":\"passed\",\"mxcsr\":"<<la_environment()<<",\"ties_away\":true,\"full_I\":[1024,1408,2816,10944]}\n";return 0;
 }catch(const std::exception&e){std::cerr<<e.what()<<'\n';return 1;}}
#endif
