// Small standard bipartite matching oracle. No full-FFN scheduling or arithmetic.
#include <algorithm>
#include <array>
#include <cstdint>
#include <vector>
static bool dfs(int rank,const uint8_t* masks,const int32_t* route,const int32_t* caps,
                std::array<int,12>& owner,std::array<bool,12>& seen){
 for(int g=0;g<6;++g)if(masks[route[rank]]&(1<<g))for(int j=0;j<caps[g];++j){int slot=2*g+j;if(seen[slot])continue;seen[slot]=true;
  if(owner[slot]<0||dfs(owner[slot],masks,route,caps,owner,seen)){owner[slot]=rank;return true;}}
 return false;
}
extern "C" int match_route(const uint8_t* masks,const int32_t* route,const int32_t* caps,int32_t* result){
 std::array<int,12> owner;owner.fill(-1);for(int r=0;r<6;++r){std::array<bool,12> seen{};if(!dfs(r,masks,route,caps,owner,seen))return 0;}
 if(result){std::fill(result,result+6,-1);for(int s=0;s<12;++s)if(owner[s]>=0)result[owner[s]]=s/2;}return 1;
}
extern "C" int score_map(const uint8_t* masks,const int32_t* routes,int n,int8_t* flags){
 const int32_t caps[6]={1,1,1,1,1,1};int score=0;for(int q=0;q<n;++q){int v=match_route(masks,routes+6*q,caps,nullptr);score+=v;if(flags)flags[q]=v;}return score;
}
extern "C" int train_swaps(uint8_t* masks,const int32_t* routes,int n,const int32_t* pairs,int budget,
                            int32_t* history,int8_t* accepted,int32_t* attempted){
 std::vector<int8_t>flags(n);int score=score_map(masks,routes,n,flags.data());const int32_t caps[6]={1,1,1,1,1,1};*attempted=0;
 for(int k=0;k<budget;++k){accepted[k]=0;if(score==n){history[k]=score;continue;}++*attempted;int a=pairs[2*k],b=pairs[2*k+1];std::vector<int>affected;int old=0;
  for(int q=0;q<n;++q){bool used=false;for(int j=0;j<6;++j)used|=routes[6*q+j]==a||routes[6*q+j]==b;if(used){affected.push_back(q);old+=flags[q];}}
  std::swap(masks[a],masks[b]);std::vector<int8_t>now;int fresh=0;for(int q:affected){int v=match_route(masks,routes+6*q,caps,nullptr);fresh+=v;now.push_back(v);}
  if(fresh>old){score+=fresh-old;accepted[k]=1;for(size_t j=0;j<affected.size();++j)flags[affected[j]]=now[j];}else std::swap(masks[a],masks[b]);history[k]=score;
 }return score;
}
