// Train-only native-rectangle family audit; no new hardware or algorithm claim.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

extern "C" double fit_columns(
    int32_t *g, const int32_t *allowed, int nallowed,
    const int32_t *routes, const int64_t *weights,
    int experts, int events, int active, int rows, int columns,
    int iterations, int sweeps, uint64_t seed, double temperature_ns,
    int64_t *stats) {
  std::vector<std::vector<int>> incidence(experts);
  std::vector<int> sum(events), interior(events);
  std::vector<int64_t> costs(events);
  const auto inside=[](int x) {return x==1 || x==2;};
  const int64_t penalty=2000+rows/2;
  const auto cost=[&](int s,int z) {
    return int64_t(std::max(s,3*active-s))*28000+
      penalty*(active+z)+2000+int64_t(active)*columns/2;
  };
  int64_t total=0,count=0;
  for(int t=0;t<events;++t) {
    for(int k=0;k<active;++k) {
      int e=routes[t*active+k]; incidence[e].push_back(t);
      sum[t]+=g[e]; interior[t]+=inside(g[e]);
    }
    costs[t]=cost(sum[t],interior[t]);
    total+=costs[t]*weights[t]; count+=weights[t];
  }
  const auto move=[&](int e,int to,bool commit) {
    int ds=to-g[e],dz=inside(to)-inside(g[e]); int64_t delta=0;
    for(int t:incidence[e]) {
      auto next=cost(sum[t]+ds,interior[t]+dz);
      delta+=(next-costs[t])*weights[t];
      if(commit) {sum[t]+=ds;interior[t]+=dz;costs[t]=next;}
    }
    if(commit) {g[e]=to;total+=delta;}
    return delta;
  };
  int64_t best=total;
  std::vector<int32_t> best_g(g,g+experts);
  const auto save=[&]() {
    if(total<best) {best=total;std::copy(g,g+experts,best_g.begin());}
  };
  std::mt19937_64 rng(seed);std::uniform_real_distribution<double> u(0,1);
  int64_t valid=0,accepted=0,greedy=0;
  for(int j=0;j<iterations;++j) {
    int e=rng()%experts,to=allowed[rng()%nallowed];
    if(to==g[e])continue;
    ++valid;auto delta=move(e,to,false);
    double temp=temperature_ns*std::pow(.001,double(j)/std::max(1,iterations-1));
    if(delta<0 || (temp>0 && u(rng)<std::exp(-double(delta)/count/temp))) {
      move(e,to,true);++accepted;save();
    }
  }
  for(int e=0;e<experts;++e)if(g[e]!=best_g[e])move(e,best_g[e],true);
  std::vector<int> order(experts);std::iota(order.begin(),order.end(),0);
  bool converged=false;
  for(int sweep=0;sweep<sweeps;++sweep) {
    int changed=0;std::shuffle(order.begin(),order.end(),rng);
    for(int e:order) {
      int to=g[e];int64_t delta=0;
      for(int j=0;j<nallowed;++j)if(allowed[j]!=g[e]) {
        auto d=move(e,allowed[j],false);++valid;
        if(d<delta) {delta=d;to=allowed[j];}
      }
      if(to!=g[e]) {move(e,to,true);++changed;++greedy;save();}
    }
    if(!changed) {converged=true;break;}
  }
  std::copy(best_g.begin(),best_g.end(),g);
  stats[0]=valid;stats[1]=accepted;stats[2]=greedy;stats[3]=converged;
  return double(best)/count;
}
