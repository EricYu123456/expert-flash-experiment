#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
static float quantize(const float *v,int n,int8_t *q){
 float amax=0;for(int k=0;k<n;++k){if(!std::isfinite(v[k]))throw std::runtime_error("Nonfinite input");amax=std::max(amax,std::abs(v[k]));}
 float s=amax==0?1.0f:amax/127.0f;if(!(s>0)||!std::isfinite(s))throw std::runtime_error("Invalid quantization scale");
 for(int k=0;k<n;++k)q[k]=(int8_t)std::clamp((int)std::round(v[k]/s),-127,127);return s;
}
static int64_t dot(const int8_t *w,const int8_t *x,int n){
 int64_t sum=0;
 #pragma omp simd reduction(+:sum)
 for(int k=0;k<n;++k)sum+=(int32_t)w[k]*(int32_t)x[k];return sum;
}
static float scale_output(int64_t sum,float sw,float sx){const float s=sw*sx;const float f=(float)sum;return f*s;}
static bool equal_bits(float a,float b){uint32_t x,y;std::memcpy(&x,&a,4);std::memcpy(&y,&b,4);return x==y;}

template<class T>static void read(FILE*f,T*p,size_t n){if(std::fread(p,sizeof(T),n,f)!=n)throw std::runtime_error("fixture read");}
int main(int argc,char**argv){if(argc!=2)return 2;FILE*f=std::fopen(argv[1],"rb");if(!f)return 3;uint32_t cases;read(f,&cases,1);uint64_t rows=0,components=0;
for(uint32_t j=0;j<cases;++j){uint32_t C,R;read(f,&C,1);read(f,&R,1);std::vector<float>x(C),sc(R),ref(R);std::vector<int8_t>q(C),expected(C);std::vector<int64_t>integer(R);float expected_scale;read(f,x.data(),C);read(f,expected.data(),C);read(f,&expected_scale,1);read(f,integer.data(),R);read(f,sc.data(),R);read(f,ref.data(),R);float sx=quantize(x.data(),C,q.data());if(!equal_bits(sx,expected_scale)||q!=expected)throw std::runtime_error("quantizer mismatch");for(uint32_t r=0;r<R;++r)if(!equal_bits(scale_output(integer[r],sc[r],sx),ref[r]))throw std::runtime_error("rescale mismatch");rows+=R;components+=C;}
if(std::fgetc(f)!=EOF)throw std::runtime_error("trailing fixture bytes");std::fclose(f);std::printf("{\"status\":\"PASS\",\"operator_vectors\":%u,\"A8_components\":%llu,\"I64_to_F32_rescaled_rows\":%llu,\"bit_mismatches\":0}\n",cases,(unsigned long long)components,(unsigned long long)rows);}
