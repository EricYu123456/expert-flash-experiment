// Existing-native-command whole-copy baseline: two fixed omit-color profiles.
#include <algorithm>
#include <array>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

static int64_t profile_cost(const int32_t* route,int active,const int32_t* base,
                            const int32_t* alt) {
  std::array<int,4> counts{};int64_t best=INT64_MAX;
  const auto visit=[&](auto&& self,int i)->void {
    if(i==active) {
      int peak=active-*std::min_element(counts.begin(),counts.end());
      int pair=std::min(counts[0]+counts[2],counts[1]+counts[3]);
      int64_t cost=peak*28000+active*12*250+2000+
          256*(2*active-pair)+active*1536*4/8;
      best=std::min(best,cost);return;
    }
    int e=route[i];++counts[base[e]];self(self,i+1);--counts[base[e]];
    if(alt[e]>=0 && alt[e]!=base[e]) {++counts[alt[e]];self(self,i+1);--counts[alt[e]];}
  };
  visit(visit,0);return best;
}

extern "C" double optimize_whole_copy(const int32_t* base,int32_t* alt,
    const int32_t* routes,const int64_t* weights,int experts,int events,int active,
    int sweeps,uint64_t seed,int64_t* stats) {
  std::vector<std::vector<int>> incidence(experts);
  std::vector<int64_t> costs(events);int64_t total=0,count=0;
  for(int t=0;t<events;++t) {
    for(int k=0;k<active;++k) incidence[routes[t*active+k]].push_back(t);
    costs[t]=profile_cost(routes+t*active,active,base,alt);
    total+=costs[t]*weights[t];count+=weights[t];
  }
  std::vector<int> order(experts);std::iota(order.begin(),order.end(),0);std::mt19937_64 rng(seed);
  int64_t proposals=0,accepted=0;
  for(int sweep=0;sweep<sweeps;++sweep) {
    int changed=0;std::shuffle(order.begin(),order.end(),rng);
    for(int e:order) if(alt[e]>=0) {
      int original=alt[e],best_color=original;int64_t best_delta=0;
      for(int color=0;color<4;++color) if(color!=base[e] && color!=original) {
        alt[e]=color;int64_t delta=0;++proposals;
        for(int t:incidence[e]) delta+=(profile_cost(routes+t*active,active,base,alt)-costs[t])*weights[t];
        if(delta<best_delta) {best_delta=delta;best_color=color;}
      }
      alt[e]=best_color;
      if(best_color!=original) {
        ++changed;++accepted;total+=best_delta;
        for(int t:incidence[e]) costs[t]=profile_cost(routes+t*active,active,base,alt);
      }
    }
    if(!changed) break;
  }
  stats[0]=proposals;stats[1]=accepted;return double(total)/count;
}

extern "C" void evaluate_whole_copy(const int32_t* base,const int32_t* alt,
    const int32_t* routes,int events,int active,int64_t* costs) {
  for(int t=0;t<events;++t) costs[t]=profile_cost(routes+t*active,active,base,alt);
}
