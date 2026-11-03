#include <algorithm>
#include <cstdint>
#include <limits>
extern "C" void down_copy_oracle(const int64_t*base,const int32_t*routes,
 const int64_t*delta,const int32_t*indices,int nq,int64_t*values,int32_t*masks){
 for(int qi=0;qi<nq;++qi){int q=indices[qi];int64_t loads[16];std::copy(base+q*16,base+q*16+16,loads);
  int64_t best=*std::max_element(loads,loads+16);int bestmask=0,oldgray=0;
  for(int i=1;i<64;++i){int gray=i^(i>>1),changed=gray^oldgray;int bit=__builtin_ctz((unsigned)changed);int e=routes[q*6+bit];int sign=(gray&changed)?1:-1;
   for(int c=0;c<16;++c)loads[c]+=sign*delta[e*16+c];
   int64_t value=*std::max_element(loads,loads+16);if(value<best||(value==best&&gray<bestmask)){best=value;bestmask=gray;}oldgray=gray;
  }
  values[qi]=best;masks[qi]=bestmask;
 }
}
