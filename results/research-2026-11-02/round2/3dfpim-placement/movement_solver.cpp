#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <random>
#include <vector>

namespace {
constexpr int E=64, A=20, N=1280, T=54, K=6;
inline int pc(uint64_t x) { return __builtin_popcountll(x); }
struct Model {
 const int* route; int Q; bool aggregate;
 double p[E]{}; double logmiss[E]{};
 Model(const int* r,int q,bool a):route(r),Q(q),aggregate(a) {
  for(int i=0;i<Q*K;++i) p[r[i]]+=1.0/Q;
  for(int e=0;e<E;++e) logmiss[e]=std::log1p(-std::min(p[e],1.0-1e-15));
 }
 double covered(uint64_t mask) const {
  double log=0; while(mask) {int e=__builtin_ctzll(mask);mask&=mask-1;log+=logmiss[e];}
  return -std::expm1(log);
 }
 double cost(const int* tile,int objective) const {
  uint64_t gu[E]{},d[E]{},gusers[T]{},dusers[12][T]{};
  double internal=0;
  for(int e=0;e<E;++e) {
   for(int j=0;j<8;++j) { gu[e]|=1ull<<tile[e*A+j];gusers[tile[e*A+j]]|=1ull<<e; }
   for(int j=0;j<12;++j) { d[e]|=1ull<<tile[e*A+8+j];dusers[j][tile[e*A+8+j]]|=1ull<<e; }
   for(int j=0;j<4;++j) internal+=p[e]*((tile[e*A+j]!=tile[e*A+4+j])+pc(d[e]&~(1ull<<tile[e*A+4+j])));
  }
  if(objective==0) {
   double sum=0;
   for(int q=0;q<Q;++q) {
    uint64_t in=0; for(int k=0;k<K;++k)in|=gu[route[q*K+k]];
    sum+=12*pc(in);
    if(aggregate) for(int j=0;j<12;++j) {
     uint64_t out=0;for(int k=0;k<K;++k)out|=1ull<<tile[route[q*K+k]*A+8+j];sum+=pc(out);
    }
    else sum+=12*K;
   }
   return sum/Q+internal;
  }
  if(objective==1) {
   double sum=internal;
   for(int t=0;t<T;++t)sum+=12*covered(gusers[t]);
   if(aggregate) for(int j=0;j<12;++j)for(int t=0;t<T;++t)sum+=covered(dusers[j][t]);
   else sum+=12*K;
   return sum;
  }
  // Correct consumer endpoint graph, greedy solver (not KaHIP).
  // Input's affinity anchor is the last shared-GU user; each native edge
  // is weighted by the consumer expert's activation marginal. Evaluation
  // of this layout still uses the exact same native multicast accounting.
  double graph=0;int anchor=tile[63*A+7];
  for(int e=0;e<E;++e) {
   for(int j=0;j<8;++j)graph+=12*p[e]*(tile[e*A+j]!=anchor);
   for(int j=0;j<4;++j) {
    graph+=p[e]*(tile[e*A+j]!=tile[e*A+4+j]);
    for(int r=0;r<12;++r)graph+=p[e]*(tile[e*A+4+j]!=tile[e*A+8+r]);
   }
  }
  if(aggregate) for(int r=0;r<12;++r)for(int e=0;e<E;++e)for(int f=e+1;f<E;++f)
    graph+=p[e]*p[f]*(tile[e*A+8+r]!=tile[f*A+8+r]);
  return graph;
 }
};
}
extern "C" double score_movement(const int* mapping,const int* routes,int q,int aggregate,int objective) {
 return Model(routes,q,aggregate).cost(mapping,objective);
}
extern "C" void original_shuffle(int* ranks,unsigned seed) {
 std::vector<int> order(N);std::iota(order.begin(),order.end(),0);
 std::srand(seed);std::random_shuffle(order.begin(),order.end());
 for(int i=0;i<N;++i)ranks[order[i]]=i;
}
extern "C" double optimize_movement(int* mapping,const int* routes,int q,int aggregate,int objective,int steps,unsigned seed) {
 Model model(routes,q,aggregate);std::mt19937 rng(seed);
 std::vector<int> current(mapping,mapping+N),best=current;
 double value=model.cost(current.data(),objective),lowest=value;
 std::uniform_real_distribution<double> u(0,1);
 for(int it=0;it<steps;++it) {
  int e=rng()%E,f=rng()%E; if(e==f)continue;
  std::vector<std::pair<int,int>> swap;
  unsigned kind=rng()%10;
  if(kind<3)swap.push_back({int(rng()%N),int(rng()%N)});
  else if(kind<5) {int j=rng()%4;swap.push_back({e*A+j,f*A+j});swap.push_back({e*A+4+j,f*A+4+j});}
  else if(kind<7)for(int j=0;j<8;++j)swap.push_back({e*A+j,f*A+j});
  else if(kind==7)for(int j=8;j<20;++j)swap.push_back({e*A+j,f*A+j});
  else for(int j=0;j<20;++j)swap.push_back({e*A+j,f*A+j});
  bool changed=false;for(auto pair:swap)if(current[pair.first]!=current[pair.second])changed=true;
  if(!changed)continue;
  for(auto pair:swap)std::swap(current[pair.first],current[pair.second]);
  double next=model.cost(current.data(),objective);
  double temperature=it>steps*4/5?0:0.08*std::pow(0.002,double(it)/(steps*0.8));
  if(next<=value || (temperature>0 && u(rng)<std::exp((value-next)/temperature))) {
   value=next;if(value<lowest-1e-12){lowest=value;best=current;}
  } else for(auto pair:swap)std::swap(current[pair.first],current[pair.second]);
 }
 std::copy(best.begin(),best.end(),mapping);return lowest;
}
