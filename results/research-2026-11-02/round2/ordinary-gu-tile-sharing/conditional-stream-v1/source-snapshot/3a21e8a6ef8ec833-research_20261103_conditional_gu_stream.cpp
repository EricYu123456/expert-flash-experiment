// Conditional FIFO/DMA sensitivity. This does not modify the sealed backend.
#include "research_20261103_ordinary_gu_tile_dag.cpp"

struct StreamWave {int chip,compute,out,pcie;};

// Exact same ready queues/context ownership as the sealed scheduler. Only the
// chip-context LB changes: streamed read/out nodes can overlap within one job.
static int64_t stream_run(Graph& g,int64_t* trace,int64_t* bound) {
 const Config& c=g.c;auto& nodes=g.nodes;
 using Key=std::tuple<int,int,int>;using Item=std::pair<Key,int>;
 using Queue=std::priority_queue<Item,std::vector<Item>,std::greater<Item>>;
 std::vector<Queue> ready(c.resources()),commands(c.chips);
 std::vector<bool> busy(c.resources());std::vector<int> owner(c.chips,-1),remaining(nodes.size());
 std::vector<int64_t> start(nodes.size(),-1),end(nodes.size(),-1),work(c.resources()),context(c.chips),critical(nodes.size()),local(nodes.size());
 using Event=std::pair<int64_t,int>;std::priority_queue<Event,std::vector<Event>,std::greater<Event>> events;
 auto enqueue=[&](int i){const auto& n=nodes[i];int p=c.down_first?1-n.phase:n.phase;Item item={{p,n.expert,i},i};
  if(n.acquire)commands[n.chip].push(item);else ready[n.resource].push(item);};
 int64_t critical_max=0;
 for(int i=0;i<int(nodes.size());++i){const auto& n=nodes[i];remaining[i]=n.deps.size();if(!remaining[i])enqueue(i);work[n.resource]+=n.duration;
  int64_t prior=0,own=0;for(int d:n.deps){prior=std::max(prior,critical[d]);if(n.job>=0&&nodes[d].job==n.job)own=std::max(own,local[d]);}
  critical[i]=prior+n.duration;critical_max=std::max(critical_max,critical[i]);local[i]=own+n.duration;
  if(n.release)context[n.chip]+=local[i];}
 int64_t now=0;size_t done=0;
 while(done<nodes.size()){
  for(int r=0;r<c.resources();++r){if(busy[r])continue;int selected=-1;
   if(r==1){assert(ready[r].empty());int best=-1;for(int chip=0;chip<c.chips;++chip)if(owner[chip]<0&&!commands[chip].empty())
    if(best<0||commands[chip].top()<commands[best].top())best=chip;
    if(best>=0){selected=commands[best].top().second;commands[best].pop();}}
   else if(!ready[r].empty()){selected=ready[r].top().second;ready[r].pop();const auto& n=nodes[selected];assert(n.job<0||owner[n.chip]==n.job);}
   if(selected<0)continue;const auto& n=nodes[selected];if(n.acquire){assert(owner[n.chip]<0);owner[n.chip]=n.job;}
   start[selected]=now;busy[r]=true;events.push({now+n.duration,selected});}
  assert(!events.empty());now=events.top().first;
  while(!events.empty()&&events.top().first==now){int i=events.top().second;events.pop();const auto& n=nodes[i];end[i]=now;busy[n.resource]=false;
   if(n.release){assert(owner[n.chip]==n.job);owner[n.chip]=-1;}++done;for(int j:n.next)if(--remaining[j]==0)enqueue(j);}}
 for(int o:owner)assert(o<0);
 for(size_t i=0;i<nodes.size();++i){const auto& n=nodes[i];assert(start[i]>=0&&end[i]-start[i]==n.duration);for(int d:n.deps)assert(end[d]<=start[i]);
  if(trace){int64_t* t=trace+i*8;t[0]=n.resource;t[1]=n.duration;t[2]=start[i];t[3]=end[i];t[4]=n.phase;t[5]=n.expert;t[6]=n.chip;t[7]=n.job;}}
 int64_t lb=*std::max_element(work.begin(),work.end());lb=std::max(lb,critical_max);lb=std::max(lb,c.submit+*std::max_element(context.begin(),context.end()));lb=std::max(lb,c.submit+work[1]);
 assert(now>=lb);if(bound)*bound=lb;return now;
}

static int whole_matrix(Graph& g,int expert,int phase,const int32_t* d,int submit,int upload,int64_t dma){
 int out=g.matrix(expert,phase,d,submit,upload);g.nodes[out].duration+=dma;return out;
}

static std::vector<std::pair<int,int>> wave_matrix(Graph& g,int priority,int chip,int waves,int owners,
 const std::vector<int>& wave_owners,int submit,int upload,int64_t dma,bool expert_dma,std::vector<StreamWave>& logs){
 const Config& c=g.c;int job=g.jobs++;
 int cmd=g.add(1,c.command,{submit},0,priority,chip,job,true,false);
 int input=g.add(3+chip%c.channels,c.transfer(2048LL*c.input_bytes*c.input_copies,c.channel),{cmd,upload},0,priority,chip,job);
 int previous_read=input,previous_out=-1;std::vector<std::pair<int,int>> result;
 for(int w=0;w<waves;++w){
  int read=g.add(c.chip_resource(chip),c.read,{previous_read},0,priority,chip,job);previous_read=read;
  std::vector<int> deps{read};if(previous_out>=0)deps.push_back(previous_out);
  int out=g.add(3+chip%c.channels,c.transfer(32LL*c.output_bytes,c.channel),deps,0,priority,chip,job,false,w+1==waves);previous_out=out;
  int pci=expert_dma?-1:g.add(2,dma+c.transfer(32LL*c.output_bytes,c.pcie),{out},0,priority);
  logs.push_back({chip,read,out,pci});result.push_back({expert_dma?out:pci,wave_owners[w]});
 }
 return result;
}

// stream0: legacy whole-view production/drain; stream1: wave transport but
// view-wide completion; stream2: per-expert full-row prefix completion;
// stream3: collect whole expert in controller, then one vectorDMA/finalCQE.
extern "C" void eval_conditional_stream(const int32_t* offsets,const int32_t* jobs,const int32_t* columns,
 const int32_t* ranges,const int32_t* routes,int experts,int job_count,int events,int active,int mode,int stream,
 int64_t cid_submit,int64_t cid_complete,int64_t dma,const int64_t* config,
 int64_t* result,int64_t* bounds,int64_t* fifo,int64_t* controller_peak,int64_t* trace,int64_t* trace_nodes){
 Config c(config);assert(experts==64&&active==6&&c.chips==16&&columns[0]==2048&&stream>=0&&stream<=3);
 for(int q=0;q<events;++q){
  std::vector<int> selected;for(int k=0;k<active;++k)selected.push_back(routes[q*active+k]);selected.push_back(experts);
  Graph g(c);int prefix=-1;
  if(cid_submit)for(int k=0;k<int(selected.size());++k)prefix=g.add(0,cid_submit,prefix<0?std::vector<int>{}:std::vector<int>{prefix},0,k);
  int submit=g.add(0,c.submit,prefix<0?std::vector<int>{}:std::vector<int>{prefix});
  int upload=g.add(2,c.transfer(2048LL*c.input_bytes,c.pcie),{submit});
  std::vector<std::vector<int>> outputs(selected.size());std::vector<int> ready;std::vector<StreamWave> logs;std::vector<std::pair<int,int64_t>> vector_dma;
  auto make_view=[&](int priority,int chip,int waves,int owners,const std::vector<int>& perwave){
   if(stream==0){int32_t d[]={chip,waves*32,2048,waves*4,waves};int done=whole_matrix(g,priority,0,d,submit,upload,dma);
    for(int k=0;k<int(selected.size());++k)if(owners&(1<<k))outputs[k].push_back(done);}
   else {auto pieces=wave_matrix(g,priority,chip,waves,owners,perwave,submit,upload,dma,stream==3,logs);
    for(int k=0;k<int(selected.size());++k)if(owners&(1<<k))for(auto piece:pieces)
     if(stream==1||(piece.second&(1<<k)))outputs[k].push_back(piece.first);}
  };
  if(mode==0){
   for(int k=0;k<int(selected.size());++k){int e=selected[k];const int32_t* o=offsets+e*4;
    for(int j=o[0];j<o[1];++j){const int32_t* d=jobs+j*5;assert(d[1]==d[4]*32&&d[3]==d[4]*4);
     make_view(k,d[0],d[4],1<<k,std::vector<int>(d[4],1<<k));}
    assert(!outputs[k].empty());if(stream!=3)ready.push_back(g.add(c.join_resource(),0,outputs[k],0,k));}
  }else {
   auto views=tile_views(ranges,selected,c,mode);
   for(auto v:views){std::vector<int> perwave;
    for(int w=0;w<v.waves;++w){int row=v.block*6144+(v.start+w)*32,mask=0;
     for(int k=0;k<int(selected.size());++k){const int32_t* r=ranges+(selected[k]*16+v.chip)*2;
      assert(r[0]%32==0&&r[1]%32==0);if(r[0]<=row&&row+32<=r[1])mask|=1<<k;}
     assert(mask>0&&(mask&(mask-1))==0);perwave.push_back(mask);}
    make_view(v.priority,v.chip,v.waves,v.owners,perwave);}
   for(int k=0;k<int(selected.size());++k){assert(!outputs[k].empty());if(stream!=3)ready.push_back(g.add(c.join_resource(),0,outputs[k],0,k));}
  }
  if(stream==3)for(int k=0;k<int(selected.size());++k){
   int e=selected[k];const int32_t* o=offsets+e*4;int64_t rows=0;for(int j=o[0];j<o[1];++j)rows+=jobs[j*5+1];
   int64_t bytes=rows*c.output_bytes;int pci=g.add(2,dma+c.transfer(bytes,c.pcie),outputs[k],0,k);
   ready.push_back(g.add(c.join_resource(),0,{pci},0,k));vector_dma.push_back({pci,bytes});
  }
  std::vector<int> complete;
  for(int k=0;k<int(selected.size());++k){int e=selected[k],done=ready[k];if(cid_complete)done=g.add(0,cid_complete,{done},0,k);
   int act=g.add(0,c.activation,{done},1,k);int ds=g.add(0,c.submit,{act},1,k);
   int up=g.add(2,c.transfer(int64_t(columns[e*2+1])*c.input_bytes,c.pcie),{ds},1,k);
   const int32_t* o=offsets+e*4;for(int j=o[2];j<o[3];++j)complete.push_back(whole_matrix(g,k,1,jobs+j*5,ds,up,dma));}
  g.add(0,c.combine,complete,1);
  std::vector<int64_t> log(g.nodes.size()*8);
  result[q]=stream_run(g,log.data(),bounds?bounds+q:nullptr);
  if(q==0){if(trace_nodes)*trace_nodes=g.nodes.size();if(trace)std::copy(log.begin(),log.end(),trace);}
  if(fifo)for(int chip=0;chip<16;++chip)fifo[q*16+chip]=stream?0:-1;
  if(controller_peak)controller_peak[q]=stream?0:-1;
  if(stream){
   std::vector<std::vector<std::pair<int64_t,int>>> f(16);std::vector<std::pair<int64_t,int>> controller;
   for(auto w:logs){f[w.chip].push_back({log[w.compute*8+3],128});f[w.chip].push_back({log[w.out*8+3],-128});
    controller.push_back({log[w.out*8+3],128});if(w.pcie>=0)controller.push_back({log[w.pcie*8+3],-128});}
   for(auto v:vector_dma)controller.push_back({log[v.first*8+3],-int(v.second)});
   for(int chip=0;chip<16;++chip){auto& ev=f[chip];std::sort(ev.begin(),ev.end());int64_t live=0,peak=0;
    for(auto e:ev){live+=e.second;assert(live>=0);peak=std::max(peak,live);}assert(live==0);if(fifo)fifo[q*16+chip]=peak;}
   std::sort(controller.begin(),controller.end());int64_t live=0,peak=0;for(auto e:controller){live+=e.second;assert(live>=0);peak=std::max(peak,live);}assert(live==0);if(controller_peak)controller_peak[q]=peak;
  }
 }
}
