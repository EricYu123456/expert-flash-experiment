// Exact finite matching-orientation oracle. No device or native backend changes.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <queue>
#include <vector>

struct Edge { int v, rev, cap; };
struct Flow {
 std::vector<std::vector<Edge>> g;
 std::vector<int> level, it;
 explicit Flow(int n):g(n),level(n),it(n){}
 int add(int a,int b,int c){int ix=g[a].size();g[a].push_back({b,int(g[b].size()),c});g[b].push_back({a,ix,0});return ix;}
 bool bfs(int s,int t){std::fill(level.begin(),level.end(),-1);std::queue<int> q;q.push(s);level[s]=0;
  while(!q.empty()){int a=q.front();q.pop();for(auto e:g[a])if(e.cap&&level[e.v]<0){level[e.v]=level[a]+1;q.push(e.v);}}
  return level[t]>=0;}
 int dfs(int a,int t,int f){if(a==t)return f;for(int &i=it[a];i<int(g[a].size());++i){auto &e=g[a][i];if(e.cap&&level[e.v]==level[a]+1){int z=dfs(e.v,t,std::min(f,e.cap));if(z){e.cap-=z;g[e.v][e.rev].cap+=z;return z;}}}return 0;}
 int run(int s,int t){int f=0,z;while(bfs(s,t)){std::fill(it.begin(),it.end(),0);while((z=dfs(s,t,1000)))f+=z;}return f;}
};

extern "C" void paired_orientation(const int64_t* base,const int32_t* routes,
 const int32_t* matching,int nq,int64_t delta,int64_t* value,int32_t* assignment){
 assert(delta>0);
 for(int q=0;q<nq;++q){
  std::array<std::array<int,2>,48> pairs;std::array<int,16> degree{};
  for(int k=0;k<6;++k)for(int j=0;j<8;++j){int e=routes[q*6+k];assert(e>=0&&e<64);
   for(int side=0;side<2;++side){int c=matching[e*16+j*2+side];assert(c>=0&&c<16);pairs[k*8+j][side]=c;++degree[c];}}
  for(int d:degree)assert(d==6);
  auto feasible=[&](int64_t bound,int32_t* take){
   Flow f(66);std::array<std::array<int,2>,48> ix;
   for(int j=0;j<48;++j){f.add(0,j+1,1);for(int side=0;side<2;++side)ix[j][side]=f.add(j+1,49+pairs[j][side],1);}
   for(int c=0;c<16;++c){int64_t space=bound-base[q*16+c];if(space<0)return false;f.add(49+c,65,int(std::min<int64_t>(6,space/delta)));}
   if(f.run(0,65)!=48)return false;
   if(take)for(int j=0;j<48;++j){int side=f.g[j+1][ix[j][0]].cap==0?0:1;assert(f.g[j+1][ix[j][side]].cap==0);take[j]=pairs[j][side];}
   return true;
  };
  std::vector<int64_t> levels;
  for(int c=0;c<16;++c)for(int d=0;d<=6;++d)levels.push_back(base[q*16+c]+d*delta);
  std::sort(levels.begin(),levels.end());levels.erase(std::unique(levels.begin(),levels.end()),levels.end());
  int lo=0,hi=int(levels.size())-1;assert(feasible(levels[hi],nullptr));
  while(lo<hi){int mid=(lo+hi)/2;if(feasible(levels[mid],nullptr))hi=mid;else lo=mid+1;}
  value[q]=levels[lo];assert(feasible(value[q],assignment?assignment+q*48:nullptr));
 }
}
