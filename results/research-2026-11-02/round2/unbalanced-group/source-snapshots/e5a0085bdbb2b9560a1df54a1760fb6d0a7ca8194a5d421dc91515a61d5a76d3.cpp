// Baseline audit: single-expert relabels among existing disjoint chip groups.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <numeric>
#include <random>
#include <vector>

extern "C" double optimize_unbalanced_group(
    int32_t *labels, const int32_t *templates, const int32_t *routes,
    const int64_t *weights, int experts, int groups, int events, int active,
    int columns, int common, int iterations, int sweeps, uint64_t seed,
    double temperature_ns, int64_t *stats) {
  using A = std::array<int64_t,16>;
  std::vector<A> wt(groups), pt(groups), it(groups), ot(groups);
  std::vector<int64_t> ct(groups);
  for(int g=0;g<groups;++g) for(int c=0;c<16;++c) {
    int64_t r=templates[g*16+c];
    pt[g][c]=(r*columns+16383)/16384;
    wt[g][c]=(pt[g][c]+3)/4;
    it[g][c]=r>0; ot[g][c]=r*4;
    ct[g]+=(wt[g][c]+191)/192;
  }
  std::vector<A> wave(events), page(events), input(events), output(events);
  std::vector<int64_t> command(events), costs(events);
  std::vector<std::vector<int>> incidence(experts);
  for(int t=0;t<events;++t) for(int k=0;k<active;++k) {
    int e=routes[t*active+k],g=labels[e]; incidence[e].push_back(t);
    for(int c=0;c<16;++c) {
      wave[t][c]+=wt[g][c]; page[t][c]+=pt[g][c];
      input[t][c]+=it[g][c]; output[t][c]+=ot[g][c];
    }
    command[t]+=ct[g];
  }
  const int64_t host_input=int64_t(columns)*(common?1:active);
  const auto cost=[&](int t) {
    int64_t maxwave=0,maxpage=0,inp=0,out=0,totalout=0;
    for(int c=0;c<16;++c) {
      maxwave=std::max(maxwave,wave[t][c]);
      maxpage=std::max(maxpage,page[t][c]); totalout+=output[t][c];
    }
    for(int c=0;c<8;++c) {
      int64_t a=common?(input[t][c]>0):input[t][c];
      int64_t b=common?(input[t][c+8]>0):input[t][c+8];
      inp=std::max(inp,(a+b)*columns);
      out=std::max(out,output[t][c]+output[t][c+8]);
    }
    return std::max(maxwave*28000,maxpage*2560)+command[t]*250+2000+
      std::max((host_input+7)/8,(inp+1)/2)+
      std::max((totalout+7)/8,(out+1)/2);
  };
  int64_t count=0,total=0;
  for(int t=0;t<events;++t) {
    costs[t]=cost(t); count+=weights[t]; total+=costs[t]*weights[t];
  }
  int64_t best=total;
  std::vector<int32_t> best_labels(labels,labels+experts);
  // Apply one relabel to caches. Passing commit=false restores them exactly.
  const auto move=[&](int e,int to,bool commit) {
    int from=labels[e]; int64_t delta=0;
    for(int t:incidence[e]) {
      for(int c=0;c<16;++c) {
        wave[t][c]+=wt[to][c]-wt[from][c];
        page[t][c]+=pt[to][c]-pt[from][c];
        input[t][c]+=it[to][c]-it[from][c];
        output[t][c]+=ot[to][c]-ot[from][c];
      }
      command[t]+=ct[to]-ct[from];
      int64_t next=cost(t); delta+=(next-costs[t])*weights[t];
      if(commit) costs[t]=next;
      else {
        for(int c=0;c<16;++c) {
          wave[t][c]-=wt[to][c]-wt[from][c];
          page[t][c]-=pt[to][c]-pt[from][c];
          input[t][c]-=it[to][c]-it[from][c];
          output[t][c]-=ot[to][c]-ot[from][c];
        }
        command[t]-=ct[to]-ct[from];
      }
    }
    if(commit) {labels[e]=to; total+=delta;}
    return delta;
  };
  const auto save=[&]() {
    if(total<best) {best=total;std::copy(labels,labels+experts,best_labels.begin());}
  };
  std::mt19937_64 rng(seed); std::uniform_real_distribution<double> uniform(0,1);
  int64_t valid=0,accepted=0,greedy=0;
  if(groups>1) for(int j=0;j<iterations;++j) {
    int e=rng()%experts,to=rng()%groups;
    if(to==labels[e]) continue;
    ++valid; int64_t delta=move(e,to,false);
    double temperature=temperature_ns*std::pow(0.001,double(j)/std::max(1,iterations-1));
    if(delta<0 || (temperature>0 && uniform(rng)<std::exp(-double(delta)/count/temperature))) {
      move(e,to,true);++accepted;save();
    }
  }
  // Restore the best annealing state before exact coordinate descent.
  for(int e=0;e<experts;++e) if(labels[e]!=best_labels[e]) move(e,best_labels[e],true);
  std::vector<int> order(experts);std::iota(order.begin(),order.end(),0);
  for(int sweep=0;sweep<sweeps && groups>1;++sweep) {
    int changed=0;std::shuffle(order.begin(),order.end(),rng);
    for(int e:order) {
      int best_to=labels[e];int64_t best_delta=0;
      for(int to=0;to<groups;++to) if(to!=labels[e]) {
        int64_t delta=move(e,to,false);++valid;
        if(delta<best_delta) {best_delta=delta;best_to=to;}
      }
      if(best_to!=labels[e]) {move(e,best_to,true);++changed;++greedy;save();}
    }
    if(!changed) break;
  }
  std::copy(best_labels.begin(),best_labels.end(),labels);
  stats[0]=valid;stats[1]=accepted;stats[2]=greedy;
  return double(best)/count;
}
