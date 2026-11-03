#include <algorithm>
#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>
struct Seg {int block,start,end;};
struct Cost {int64_t ns,waves,commands,outputs;};
extern "C" void tile_sharing(const int32_t* ranges,const int32_t* routes,
 int count,int experts,int topk,int command_ns,int copies,int64_t* result) {
 for(int q=0;q<count;++q)for(int chip=0;chip<16;++chip){
  std::vector<Seg> segments;
  for(int k=0;k<=topk;++k){int e=k==topk?experts:routes[q*topk+k];
   const int32_t* r=ranges+(e*16+chip)*2;
   for(int at=r[0];at<r[1];){int b=at/6144,end=std::min(r[1],(b+1)*6144);
    segments.push_back({b,at-b*6144,end-b*6144});at=end;}}
  std::sort(segments.begin(),segments.end(),[](auto a,auto b){return std::tie(a.block,a.start)<std::tie(b.block,b.start);});
  const int64_t dispatch=command_ns+1024LL*copies;
  Cost singleton{0,0,0,0};
  for(auto s:segments){int waves=(s.end+31)/32-s.start/32;
   singleton.ns+=28064LL*waves+dispatch;singleton.waves+=waves;
   singleton.commands++;singleton.outputs+=32LL*waves;}
  std::vector<Cost> dp(segments.size()+1,{std::numeric_limits<int64_t>::max()/4,0,0,0});dp[0]={0,0,0,0};
  for(size_t i=0;i<segments.size();++i)for(size_t j=i;j<segments.size()&&segments[j].block==segments[i].block;++j){
   int waves=(segments[j].end+31)/32-segments[i].start/32;
   Cost c{dp[i].ns+28064LL*waves+dispatch,dp[i].waves+waves,dp[i].commands+1,dp[i].outputs+32LL*waves};
   if(std::tie(c.ns,c.waves,c.commands)<std::tie(dp[j+1].ns,dp[j+1].waves,dp[j+1].commands))dp[j+1]=c;}
  auto best=dp.back();int64_t* out=result+(q*16+chip)*8;
  out[0]=singleton.ns;out[1]=singleton.waves;out[2]=singleton.commands;out[3]=singleton.outputs;
  out[4]=best.ns;out[5]=best.waves;out[6]=best.commands;out[7]=best.outputs;
 }
}
