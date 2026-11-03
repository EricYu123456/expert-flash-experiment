// Capacity-preserving baseline search. No new hardware or novelty claim.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

extern "C" double fit_workdepth_profiles(int32_t *rows, const int32_t *columns,
    const int64_t *shared, const int32_t *routes, const int64_t *weights,
    int experts, int events, int active, int move_kind, int objective,
    int valid_budget, uint64_t seed, double temperature_ns, int64_t *stats) {
  using A = std::array<int64_t,16>;
  const int size=2*experts*16;
  auto at=[&](int p,int e,int c)->int32_t& {return rows[(p*experts+e)*16+c];};
  auto occupancy=[&](int p,int r) -> int64_t {
    if(r==0)return 0;
    return ((int64_t(r)*columns[p]+65535)/65536)*28000+250+(columns[p]+1)/2+2*r;
  };
  std::vector<A> load[2]; for(auto &v:load)v.resize(events);
  std::vector<std::vector<int>> incidence(experts);
  std::vector<std::vector<uint8_t>> member(experts,std::vector<uint8_t>(events));
  int64_t count=0;
  for(int t=0;t<events;++t) {
    count+=weights[t];
    for(int p=0;p<2;++p) for(int c=0;c<16;++c) load[p][t][c]=shared[p*16+c];
    for(int k=0;k<active;++k) {
      int e=routes[t*active+k];incidence[e].push_back(t);member[e][t]=1;
      for(int p=0;p<2;++p)for(int c=0;c<16;++c)load[p][t][c]+=occupancy(p,at(p,e,c));
    }
  }
  auto value=[&](int t) {
    int64_t g=0,d=0,j=0;
    for(int c=0;c<16;++c){g=std::max(g,load[0][t][c]);d=std::max(d,load[1][t][c]);j=std::max(j,load[0][t][c]+load[1][t][c]);}
    return objective==0 ? (g+d)*1000+j : j*1000+g+d;
  };
  std::vector<int64_t> values(events);int64_t total=0;
  for(int t=0;t<events;++t){values[t]=value(t);total+=values[t]*weights[t];}
  int64_t best=total,accepted=0,valid=0,attempts=0;
  std::vector<int32_t> best_rows(rows,rows+size);
  std::mt19937_64 rng(seed);std::uniform_real_distribution<double> uniform(0,1);
  std::vector<int> affected;std::vector<int64_t> proposed;
  while(valid<valid_budget && attempts<10000000) {
    ++attempts;int a=rng()%experts,b=rng()%experts,c=rng()%16,d=rng()%16;
    if(a==b)continue;
    int p0=0,p1=2;
    if(move_kind==2){p0=rng()%2;p1=p0+1;}
    A delta[2]{};bool changed=false;
    if(move_kind==0) {
      for(int p=0;p<2;++p)for(int chip=0;chip<16;++chip){delta[p][chip]=occupancy(p,at(p,b,chip))-occupancy(p,at(p,a,chip));changed|=delta[p][chip]!=0;}
    } else {
      if(c==d)continue;bool match=true;
      for(int p=p0;p<p1;++p)match &= at(p,a,c)==at(p,b,d) && at(p,a,d)==at(p,b,c);
      if(!match)continue;
      for(int p=p0;p<p1;++p){delta[p][c]=occupancy(p,at(p,a,d))-occupancy(p,at(p,a,c));delta[p][d]=-delta[p][c];changed|=delta[p][c]!=0;}
    }
    if(!changed)continue;
    ++valid;affected.clear();
    for(int t:incidence[a])if(!member[b][t])affected.push_back(t);
    for(int t:incidence[b])if(!member[a][t])affected.push_back(t);
    proposed.resize(affected.size());int64_t difference=0;
    for(size_t q=0;q<affected.size();++q){int t=affected[q],sign=member[a][t]?1:-1;
      for(int p=0;p<2;++p)for(int chip=0;chip<16;++chip)load[p][t][chip]+=sign*delta[p][chip];
      proposed[q]=value(t);difference+=(proposed[q]-values[t])*weights[t];}
    double temp=temperature_ns*std::pow(.001,double(valid)/std::max(1,valid_budget));
    bool take=difference<=0 || (temp>0 && uniform(rng)<std::exp(-double(difference)/(count*1000.0*temp)));
    if(take){++accepted;total+=difference;
      if(move_kind==0){for(int p=0;p<2;++p)for(int chip=0;chip<16;++chip)std::swap(at(p,a,chip),at(p,b,chip));}
      else for(int p=p0;p<p1;++p){std::swap(at(p,a,c),at(p,a,d));std::swap(at(p,b,c),at(p,b,d));}
      for(size_t q=0;q<affected.size();++q)values[affected[q]]=proposed[q];
      if(total<best){best=total;std::copy(rows,rows+size,best_rows.begin());}
    } else for(int t:affected){int sign=member[a][t]?1:-1;for(int p=0;p<2;++p)for(int chip=0;chip<16;++chip)load[p][t][chip]-=sign*delta[p][chip];}
  }
  std::copy(best_rows.begin(),best_rows.end(),rows);stats[0]=attempts;stats[1]=valid;stats[2]=accepted;
  return double(best)/count;
}
