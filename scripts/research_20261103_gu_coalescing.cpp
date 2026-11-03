#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>
struct Interval{int b,s,e;};
struct Cost{int64_t ns;int w,cmd,out;};
extern "C" void gu_coalescing(const int32_t* layout,const int32_t* routes,int count,int align,int columns,int64_t* answer){
 for(int q=0;q<count;++q)for(int chip=0;chip<16;++chip){
  std::vector<Interval>a;
  for(int k=0;k<11;++k){int expert=k<10?routes[q*10+k]:512;
   for(int seg=0;seg<2;++seg){const int32_t*d=layout+((expert*16+chip)*2+seg)*3;if(d[2]>d[1])a.push_back({d[0],d[1],d[2]});}}
  std::sort(a.begin(),a.end(),[](auto x,auto y){return std::tie(x.b,x.s,x.e)<std::tie(y.b,y.s,y.e);});
  std::vector<Cost>dp(a.size()+1,{std::numeric_limits<int64_t>::max()/4,0,0,0});dp[0]={0,0,0,0};
  for(size_t i=0;i<a.size();++i){int start=a[i].s/align*align;
   for(size_t j=i;j<a.size()&&a[j].b==a[i].b;++j){
    int rows=a[j].e-start;int waves=(int64_t(rows)*columns+65535)/65536;
    int out=int64_t(waves)*65536/columns;
    Cost v{dp[i].ns+28000LL*waves+250+(columns+1)/2+2LL*out,dp[i].w+waves,dp[i].cmd+1,dp[i].out+out};
    if(std::tie(v.ns,v.w,v.cmd)<std::tie(dp[j+1].ns,dp[j+1].w,dp[j+1].cmd))dp[j+1]=v;
   }
  }
  auto z=dp.back();int64_t*d=answer+(q*16+chip)*4;d[0]=z.ns;d[1]=z.w;d[2]=z.cmd;d[3]=z.out;
 }
}
