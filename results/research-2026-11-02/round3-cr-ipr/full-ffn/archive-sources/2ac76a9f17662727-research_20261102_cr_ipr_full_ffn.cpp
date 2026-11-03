// Generic causal graph executor for conditional CR/IPR FFN service models.
// No arithmetic, no implicit weight FIFO, and no measured hardware timing.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

extern "C" int64_t run_graph(int count, const int64_t *data,
    const int32_t *offsets, const int32_t *dependencies, int down_first,
    int64_t *trace) {
  constexpr int resources=27, chips=16;
  using Key=std::tuple<int,int,int>;
  using Item=std::pair<Key,int>;
  using Queue=std::priority_queue<Item,std::vector<Item>,std::greater<Item>>;
  std::vector<Queue> ready(resources), commands(chips);
  std::vector<std::vector<int>> next(count);
  std::vector<int> remain(count), owner(chips,-1);
  std::vector<bool> busy(resources,false);
  std::vector<int64_t> start(count,-1),end(count,-1);
  using Event=std::pair<int64_t,int>;
  std::priority_queue<Event,std::vector<Event>,std::greater<Event>> events;
  int64_t now=0; int done=0;
  auto node=[&](int i,int k){return data[i*8+k];};
  auto enqueue=[&](int i){
    int r=node(i,0),phase=node(i,2),expert=node(i,3);
    if(r<0){start[i]=now;events.push({now+node(i,1),i});return;}
    Key key={down_first?1-phase:phase,expert,i};
    if(node(i,6)){assert(r==1);commands[node(i,4)].push({key,i});}
    else ready[r].push({key,i});
  };
  for(int i=0;i<count;++i){
    assert(node(i,0)>=-1&&node(i,0)<resources&&node(i,1)>=0);
    remain[i]=offsets[i+1]-offsets[i];
    for(int j=offsets[i];j<offsets[i+1];++j){int d=dependencies[j];assert(d>=0&&d<i);next[d].push_back(i);}
    if(!remain[i])enqueue(i);
  }
  while(done<count){
    for(int r=0;r<resources;++r){
      if(busy[r])continue;
      int selected=-1;
      if(r==1){
        int best=-1;
        for(int c=0;c<chips;++c)if(owner[c]<0&&!commands[c].empty())
          if(best<0||commands[c].top()<commands[best].top())best=c;
        if(best>=0){selected=commands[best].top().second;commands[best].pop();}
      }else if(!ready[r].empty()){selected=ready[r].top().second;ready[r].pop();}
      if(selected<0)continue;
      int c=node(selected,4),j=node(selected,5);
      if(j>=0){
        if(node(selected,6)){assert(owner[c]<0);owner[c]=j;}
        else assert(owner[c]==j);
      }
      start[selected]=now;busy[r]=true;
      events.push({now+node(selected,1),selected});
    }
    assert(!events.empty());now=events.top().first;
    while(!events.empty()&&events.top().first==now){
      int i=events.top().second;events.pop();end[i]=now;
      int r=node(i,0);if(r>=0)busy[r]=false;
      if(node(i,7)){int c=node(i,4);assert(owner[c]==node(i,5));owner[c]=-1;}
      ++done;
      for(int q:next[i])if(--remain[q]==0)enqueue(q);
    }
  }
  for(int o:owner)assert(o<0);
  for(int i=0;i<count;++i){
    assert(start[i]>=0&&end[i]-start[i]==node(i,1));
    for(int k=offsets[i];k<offsets[i+1];++k)assert(end[dependencies[k]]<=start[i]);
    trace[i*2]=start[i];trace[i*2+1]=end[i];
  }
  return now;
}
