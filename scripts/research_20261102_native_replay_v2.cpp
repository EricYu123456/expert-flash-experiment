// Exact compiled replay of the sealed Python Graph semantics, version2.
// Per-chip ready-command heaps avoid repeatedly removing blocked commands.
// This is a software cost simulator, not a NAND, PE or numerical simulator.
#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <functional>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

namespace {
struct Node {
  int resource,phase,expert,chip,job;
  int64_t duration;
  bool acquire=false,release=false;
  std::vector<int> deps,next;
};
struct Graph {
  std::vector<Node> nodes;
  int jobs=0;
  Graph() { nodes.reserve(1900); }
  int add(int r,int64_t d,const std::vector<int>& deps={},int p=0,int e=-1,int c=-1,int j=-1,bool ac=false,bool re=false) {
    int i=nodes.size(); nodes.push_back(Node{r,p,e,c,j,d,ac,re,deps,{}});
    for(int q:deps) {assert(q<i);nodes[q].next.push_back(i);}
    return i;
  }
  int matrix(int e,int chip,int phase,int rows,int columns,int waves,int submit,int upload) {
    assert(waves==(int64_t(rows)*columns+65535)/65536 && waves<=192);
    int j=jobs++;
    int cmd=add(1,250,{submit},phase,e,chip,j,true,false);
    int inp=add(3+chip%8,(columns+1)/2,{cmd,upload},phase,e,chip,j);
    int comp=add(11+chip,28000LL*waves,{inp},phase,e,chip,j);
    int out=add(3+chip%8,rows*2LL,{comp},phase,e,chip,j,false,true);
    return add(2,(rows*4LL+7)/8,{out},phase,e);
  }
  int64_t run(bool down_first) {
    using Key=std::tuple<int,int,int>;
    using Item=std::pair<Key,int>;
    using Queue=std::priority_queue<Item,std::vector<Item>,std::greater<Item>>;
    std::array<Queue,28> ready;
    std::array<Queue,16> commands;
    std::array<bool,28> busy{};
    std::array<int,16> owner;owner.fill(-1);
    std::vector<int> remaining(nodes.size());
    std::vector<int64_t> start(nodes.size(),-1),end(nodes.size(),-1);
    using Event=std::pair<int64_t,int>;
    std::priority_queue<Event,std::vector<Event>,std::greater<Event>> events;
    auto enqueue=[&](int i) {
      const auto &n=nodes[i];int p=down_first ? 1-n.phase:n.phase;
      Item item={{p,n.expert,i},i};
      if(n.acquire) {assert(n.resource==1);commands[n.chip].push(item);}
      else ready[n.resource].push(item);
    };
    for(int i=0;i<int(nodes.size());++i) {
      remaining[i]=nodes[i].deps.size();if(!remaining[i])enqueue(i);
    }
    int64_t now=0;size_t done=0;
    std::vector<Item> held;held.reserve(100);
    while(done<nodes.size()) {
      for(int r=0;r<28;++r) {
        auto &q=ready[r];if(busy[r])continue;
        int selected=-1;
        if(r==1) {
          assert(q.empty());int best_chip=-1;
          for(int c=0;c<16;++c)if(owner[c]==-1 && !commands[c].empty()) {
            if(best_chip<0 || commands[c].top()<commands[best_chip].top())best_chip=c;
          }
          if(best_chip>=0) {selected=commands[best_chip].top().second;commands[best_chip].pop();}
        } else {
          if(q.empty())continue;
          selected=q.top().second;q.pop();
          const auto &n=nodes[selected];assert(n.job<0 || owner[n.chip]==n.job);
        }
        if(selected<0)continue;
        const auto &n=nodes[selected];
        if(n.acquire) {assert(owner[n.chip]==-1);owner[n.chip]=n.job;}
        start[selected]=now;busy[r]=true;events.push({now+n.duration,selected});
      }
      assert(!events.empty());now=events.top().first;
      while(!events.empty() && events.top().first==now) {
        int i=events.top().second;events.pop();const auto &n=nodes[i];
        end[i]=now;busy[n.resource]=false;
        if(n.release) {assert(owner[n.chip]==n.job);owner[n.chip]=-1;}
        ++done;
        for(int j:n.next) {--remaining[j];if(!remaining[j])enqueue(j);}
      }
    }
    for(int o:owner)assert(o==-1);
    for(size_t i=0;i<nodes.size();++i) {
      assert(start[i]>=0 && end[i]-start[i]==nodes[i].duration);
      for(int d:nodes[i].deps)assert(end[d]<=start[i]);
    }
    return *std::max_element(end.begin(),end.end());
  }
};
}

// descriptors[logical_expert,phase,job_slot,(chip,rows,columns,waves)]
// contains at most16 native jobs per phase. Zero rows denote unused slots.
// The extra final logical expert is always-active shared FFN.
extern "C" void eval_native_ffn(const int32_t *descriptors,const int32_t *partial,
    const int32_t *routes,int experts,int events,int active,int64_t reduction_ns,
    int64_t activation_ns,int down_first,int64_t *result) {
  for(int t=0;t<events;++t) {
    Graph g;int submit=g.add(0,2000);
    int upload=g.add(2,320,{submit});
    std::vector<int> chosen(active+1),ready;
    for(int ei=0;ei<active;++ei)chosen[ei]=routes[t*active+ei];chosen[active]=experts;
    for(int ei=0;ei<=active;++ei) {
      int e=chosen[ei];std::vector<int> outputs;
      for(int s=0;s<16;++s) {
        const int32_t *d=descriptors+((e*2)*16+s)*4;
        if(d[1])outputs.push_back(g.matrix(ei,d[0],0,d[1],d[2],d[3],submit,upload));
      }
      int barrier=g.add(27,0,outputs,0,ei);
      if(partial[e*2])barrier=g.add(0,reduction_ns,{barrier},0,ei);
      ready.push_back(barrier);
    }
    std::vector<int> complete;
    for(int ei=0;ei<=active;++ei) {
      int e=chosen[ei];int act=g.add(0,activation_ns,{ready[ei]},1,ei);
      int ds=g.add(0,2000,{act},1,ei);int up=g.add(2,80,{ds},1,ei);
      std::vector<int> outputs;
      for(int s=0;s<16;++s) {
        const int32_t *d=descriptors+((e*2+1)*16+s)*4;
        if(d[1])outputs.push_back(g.matrix(ei,d[0],1,d[1],d[2],d[3],ds,up));
      }
      if(partial[e*2+1])complete.push_back(g.add(0,reduction_ns,outputs,1,ei));
      else complete.insert(complete.end(),outputs.begin(),outputs.end());
    }
    g.add(0,0,complete,1);result[t]=g.run(down_first!=0);
  }
}
