// Fixed physical geometries, balanced label swaps, native chip-occupancy fit.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

extern "C" double fit_native_labels(int32_t *labels, const int64_t *table,
    const int32_t *routes, const int64_t *weights, int experts, int events,
    int active, int variants, int objective, int iterations, uint64_t seed,
    double initial_temperature_ns, int64_t *stats) {
  using A = std::array<int64_t,16>;
  std::vector<A> load[2];
  for (auto &a:load) a.resize(events);
  std::vector<std::vector<int>> incidence(experts);
  std::vector<std::vector<uint8_t>> member(experts,std::vector<uint8_t>(events));
  auto value = [&](int phase,int label,int chip) {
    return table[(phase*variants+label)*16+chip];
  };
  int64_t count=0;
  for(int t=0;t<events;++t) {
    count+=weights[t];
    for(int k=0;k<active;++k) {
      int e=routes[t*active+k];
      incidence[e].push_back(t); member[e][t]=1;
      for(int p=0;p<2;++p) for(int c=0;c<16;++c)
        load[p][t][c]+=value(p,labels[p*experts+e],c);
    }
  }
  auto cost = [&](int t) {
    int64_t g=0,d=0,j=0;
    for(int c=0;c<16;++c) {
      g=std::max(g,load[0][t][c]); d=std::max(d,load[1][t][c]);
      j=std::max(j,load[0][t][c]+load[1][t][c]);
    }
    return objective==0 ? (g+d)*1000+j : j*1000+g+d;
  };
  std::vector<int64_t> costs(events);
  int64_t total=0;
  for(int t=0;t<events;++t) {costs[t]=cost(t);total+=costs[t]*weights[t];}
  int64_t best=total, accepted=0, valid=0;
  std::vector<int32_t> best_labels(labels,labels+2*experts);
  std::mt19937_64 rng(seed); std::uniform_real_distribution<double> uniform(0,1);
  std::vector<int> affected; std::vector<int64_t> proposed;
  for(int it=0;it<iterations;++it) {
    int p=rng()%2,a=rng()%experts,b=rng()%experts;
    int la=labels[p*experts+a],lb=labels[p*experts+b];
    if(a==b || la==lb)continue;
    A delta_work{}; bool changed=false;
    for(int c=0;c<16;++c) {
      delta_work[c]=value(p,lb,c)-value(p,la,c);
      changed|=delta_work[c]!=0;
    }
    if(!changed)continue;
    ++valid; affected.clear();
    for(int t:incidence[a])if(!member[b][t])affected.push_back(t);
    for(int t:incidence[b])if(!member[a][t])affected.push_back(t);
    proposed.resize(affected.size()); int64_t delta=0;
    for(size_t q=0;q<affected.size();++q) {
      int t=affected[q], sign=member[a][t] ? 1:-1;
      for(int c=0;c<16;++c)load[p][t][c]+=sign*delta_work[c];
      proposed[q]=cost(t); delta+=(proposed[q]-costs[t])*weights[t];
    }
    double temperature=initial_temperature_ns*std::pow(.001,double(it)/std::max(1,iterations-1));
    bool take=delta<=0 || (temperature>0 && uniform(rng)<std::exp(-double(delta)/(count*1000.0*temperature)));
    if(take) {
      ++accepted; total+=delta; std::swap(labels[p*experts+a],labels[p*experts+b]);
      for(size_t q=0;q<affected.size();++q)costs[affected[q]]=proposed[q];
      if(total<best) {best=total;std::copy(labels,labels+2*experts,best_labels.begin());}
    } else {
      for(int t:affected) {
        int sign=member[a][t] ? 1:-1;
        for(int c=0;c<16;++c)load[p][t][c]-=sign*delta_work[c];
      }
    }
  }
  std::copy(best_labels.begin(),best_labels.end(),labels);
  stats[0]=valid;stats[1]=accepted;
  return double(best)/count;
}
