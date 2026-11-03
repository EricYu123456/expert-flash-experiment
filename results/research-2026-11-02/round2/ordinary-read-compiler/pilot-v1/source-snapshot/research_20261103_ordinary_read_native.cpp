// Hardware-parametric ordinary-read FFN service replay. No PE arithmetic.
// Every job contains complete rows, reloads its input, fits one block and owns
// one chip until its output leaves the channel. No cached weights or CR-read.
#include <algorithm>
#include <cassert>
#include <cstdint>
#include <functional>
#include <queue>
#include <tuple>
#include <utility>
#include <vector>

namespace {
int64_t ceildiv(int64_t a,int64_t b) { assert(a>=0 && b>0);return (a+b-1)/b; }
struct Config {
  int chips,channels,planes,page,block,input_bytes,output_bytes;
  int64_t read,command,submit,pcie,channel,activation,combine;
  int fringe,down_first;
  explicit Config(const int64_t *c):chips(c[0]),channels(c[1]),planes(c[2]),
    page(c[3]),block(c[4]),input_bytes(c[5]),output_bytes(c[6]),read(c[7]),
    command(c[8]),submit(c[9]),pcie(c[10]),channel(c[11]),activation(c[12]),
    combine(c[13]),fringe(c[14]),down_first(c[15]) {
    assert(chips>0 && channels>0 && chips%channels==0 && planes>0 && page>0);
    assert(block>0 && input_bytes>0 && output_bytes>0 && read>=0 && command>=0);
    assert(submit>=0 && pcie>0 && channel>0 && activation>=0 && combine>=0);
    assert(fringe>=0 && fringe<=2);
  }
  int resources()const{return 4+channels+chips;}
  int chip_resource(int c)const{return 3+channels+c;}
  int join_resource()const{return 3+channels+chips;}
  int64_t transfer(int64_t bytes,int64_t rate)const{return ceildiv(bytes*1000000000LL,rate);}
};
struct Node {
  int resource,phase,expert,chip,job;
  int64_t duration;
  bool acquire=false,release=false;
  std::vector<int> deps,next;
};
struct Graph {
  const Config &c;std::vector<Node> nodes;int jobs=0;
  explicit Graph(const Config &cfg):c(cfg){nodes.reserve(2400);}
  int add(int r,int64_t d,const std::vector<int>& deps={},int p=0,int e=-1,
      int chip=-1,int j=-1,bool acquire=false,bool release=false) {
    assert(r>=0 && r<c.resources() && d>=0);
    int i=nodes.size();nodes.push_back(Node{r,p,e,chip,j,d,acquire,release,deps,{}});
    for(int q:deps){assert(q>=0 && q<i);nodes[q].next.push_back(i);}return i;
  }
  // Descriptor: chip, rows, columns, physical pages, wave count.
  int matrix(int expert,int phase,const int32_t *d,int submit,int upload) {
    int chip=d[0],rows=d[1],columns=d[2],pages=d[3],waves=d[4];
    assert(chip>=0 && chip<c.chips && rows>0 && columns>0);
    assert(pages==ceildiv(int64_t(rows)*columns,c.page)); // I8 coefficients.
    assert(waves==ceildiv(pages,c.planes) && waves<=c.block);
    int64_t emitted=rows;
    if(c.fringe==1)emitted=int64_t(pages)*c.page/columns;
    if(c.fringe==2)emitted=int64_t(waves)*c.planes*c.page/columns;
    assert(emitted>=rows);
    int j=jobs++;
    int cmd=add(1,c.command,{submit},phase,expert,chip,j,true,false);
    int inp=add(3+chip%c.channels,c.transfer(int64_t(columns)*c.input_bytes,c.channel),
      {cmd,upload},phase,expert,chip,j);
    int comp=add(c.chip_resource(chip),c.read*waves,{inp},phase,expert,chip,j);
    int out=add(3+chip%c.channels,c.transfer(emitted*c.output_bytes,c.channel),
      {comp},phase,expert,chip,j,false,true);
    return add(2,c.transfer(emitted*c.output_bytes,c.pcie),{out},phase,expert);
  }
  int64_t run(int64_t *trace,int64_t *bound) {
    using Key=std::tuple<int,int,int>;using Item=std::pair<Key,int>;
    using Queue=std::priority_queue<Item,std::vector<Item>,std::greater<Item>>;
    std::vector<Queue> ready(c.resources()),commands(c.chips);
    std::vector<bool> busy(c.resources());std::vector<int> owner(c.chips,-1);
    std::vector<int> remaining(nodes.size());
    std::vector<int64_t> start(nodes.size(),-1),end(nodes.size(),-1),
      work(c.resources()),chip_work(c.chips),critical(nodes.size());
    using Event=std::pair<int64_t,int>;
    std::priority_queue<Event,std::vector<Event>,std::greater<Event>> events;
    auto enqueue=[&](int i){
      const auto &n=nodes[i];int p=c.down_first?1-n.phase:n.phase;
      Item item={{p,n.expert,i},i};
      if(n.acquire){assert(n.resource==1);commands[n.chip].push(item);}
      else ready[n.resource].push(item);
    };
    int64_t critical_max=0;
    for(int i=0;i<int(nodes.size());++i){
      const auto &n=nodes[i];remaining[i]=n.deps.size();if(!remaining[i])enqueue(i);
      work[n.resource]+=n.duration;
      if(n.job>=0)chip_work[n.chip]+=n.duration;
      int64_t prior=0;for(int d:n.deps)prior=std::max(prior,critical[d]);
      critical[i]=prior+n.duration;critical_max=std::max(critical_max,critical[i]);
    }
    int64_t now=0;size_t done=0;
    while(done<nodes.size()){
      for(int r=0;r<c.resources();++r){
        if(busy[r])continue;int selected=-1;
        if(r==1){
          assert(ready[r].empty());int best=-1;
          for(int chip=0;chip<c.chips;++chip)if(owner[chip]<0&&!commands[chip].empty())
            if(best<0||commands[chip].top()<commands[best].top())best=chip;
          if(best>=0){selected=commands[best].top().second;commands[best].pop();}
        }else if(!ready[r].empty()){
          selected=ready[r].top().second;ready[r].pop();
          const auto &n=nodes[selected];assert(n.job<0||owner[n.chip]==n.job);
        }
        if(selected<0)continue;const auto &n=nodes[selected];
        if(n.acquire){assert(owner[n.chip]<0);owner[n.chip]=n.job;}
        start[selected]=now;busy[r]=true;events.push({now+n.duration,selected});
      }
      assert(!events.empty());now=events.top().first;
      while(!events.empty()&&events.top().first==now){
        int i=events.top().second;events.pop();const auto &n=nodes[i];
        end[i]=now;busy[n.resource]=false;
        if(n.release){assert(owner[n.chip]==n.job);owner[n.chip]=-1;}
        ++done;for(int j:n.next)if(--remaining[j]==0)enqueue(j);
      }
    }
    for(int o:owner)assert(o<0);
    for(size_t i=0;i<nodes.size();++i){
      const auto &n=nodes[i];assert(start[i]>=0&&end[i]-start[i]==n.duration);
      for(int d:n.deps)assert(end[d]<=start[i]);
      if(trace){
        int64_t *t=trace+i*8;
        t[0]=n.resource;t[1]=n.duration;t[2]=start[i];t[3]=end[i];
        t[4]=n.phase;t[5]=n.expert;t[6]=n.chip;t[7]=n.job;
      }
    }
    int64_t lb=*std::max_element(work.begin(),work.end());
    lb=std::max(lb,critical_max);
    lb=std::max(lb,c.submit+*std::max_element(chip_work.begin(),chip_work.end()));
    lb=std::max(lb,c.submit+work[1]);assert(now>=lb);
    if(bound)*bound=lb;return now;
  }
};
}

// offsets[expert,phase,(begin,end)] indexes a ragged flat job array.
// input_columns[expert,phase] stores the original full input vector size.
// Shared FFNs are the final `shared` experts and are always selected once.
// Optional trace contains the first event only; trace_nodes returns its count.
extern "C" void eval_ordinary_read(const int32_t *offsets,const int32_t *jobs,
    const int32_t *input_columns,const int32_t *routes,int experts,int shared,
    int events,int active,const int64_t *config,int64_t *result,int64_t *bounds,
    int64_t *trace,int64_t *trace_nodes){
  Config c(config);assert(experts>0&&shared>=0&&events>0&&active>0);
  assert(input_columns[0]>0);
  for(int t=0;t<events;++t){
    Graph g(c);int submit=g.add(0,c.submit);
    int upload=g.add(2,c.transfer(int64_t(input_columns[0])*c.input_bytes,c.pcie),{submit});
    std::vector<int> chosen,ready;chosen.reserve(active+shared);
    for(int k=0;k<active;++k){int e=routes[t*active+k];assert(e>=0&&e<experts);chosen.push_back(e);}
    for(int s=0;s<shared;++s)chosen.push_back(experts+s);
    for(int ei=0;ei<int(chosen.size());++ei){
      int e=chosen[ei];assert(input_columns[e*2]==input_columns[0]);
      std::vector<int> outputs;const int32_t *o=offsets+e*4;
      for(int s=o[0];s<o[1];++s)outputs.push_back(g.matrix(ei,0,jobs+s*5,submit,upload));
      assert(!outputs.empty());ready.push_back(g.add(c.join_resource(),0,outputs,0,ei));
    }
    std::vector<int> complete;
    for(int ei=0;ei<int(chosen.size());++ei){
      int e=chosen[ei];int act=g.add(0,c.activation,{ready[ei]},1,ei);
      int ds=g.add(0,c.submit,{act},1,ei);
      int up=g.add(2,c.transfer(int64_t(input_columns[e*2+1])*c.input_bytes,c.pcie),{ds},1,ei);
      const int32_t *o=offsets+e*4;
      for(int s=o[2];s<o[3];++s)complete.push_back(g.matrix(ei,1,jobs+s*5,ds,up));
    }
    assert(!complete.empty());g.add(0,c.combine,complete,1);
    if(t==0&&trace_nodes)*trace_nodes=g.nodes.size();
    result[t]=g.run(t==0?trace:nullptr,bounds?bounds+t:nullptr);
  }
}
