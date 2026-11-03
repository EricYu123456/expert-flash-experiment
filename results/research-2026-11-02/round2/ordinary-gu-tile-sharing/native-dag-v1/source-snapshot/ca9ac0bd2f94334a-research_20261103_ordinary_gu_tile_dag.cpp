// Reuse the sealed single-context resource scheduler, without editing it.
#include "../results/research-2026-11-02/round2/ordinary-read-compiler/full-campaign-v6-W8-input4/source-snapshot/ef7994fa2ba37a4b-research_20261103_ordinary_read_native.cpp"

struct TileSegment {int block,start,end,ordinal;};
struct TileView {int chip,block,start,waves,owners,priority;};
struct TileCost {int64_t ns,waves,commands;};

static std::vector<TileView> tile_views(const int32_t* ranges,
 const std::vector<int>& selected,const Config& c,int mode) {
 std::vector<TileView> result;
 for(int chip=0;chip<c.chips;++chip){
  std::vector<TileSegment> a;
  for(int ordinal=0;ordinal<int(selected.size());++ordinal){
   const int32_t* r=ranges+(selected[ordinal]*16+chip)*2;
   for(int at=r[0];at<r[1];){int b=at/6144,end=std::min(r[1],(b+1)*6144);
    a.push_back({b,at-b*6144,end-b*6144,ordinal});at=end;}}
  std::sort(a.begin(),a.end(),[](auto x,auto y){return std::tie(x.block,x.start)<std::tie(y.block,y.start);});
  std::vector<std::pair<int,int>> groups;
  if(mode==1){for(int i=0;i<int(a.size());++i)groups.push_back({i,i+1});}
  else {
   assert(mode==2);std::vector<TileCost> dp(a.size()+1,{INT64_MAX/4,0,0});
   std::vector<int> back(a.size()+1,-1);dp[0]={0,0,0};
   int64_t perwave=c.read+c.transfer(32LL*c.output_bytes,c.channel);
   int64_t dispatch=c.command+c.transfer(2048LL*c.input_bytes*c.input_copies,c.channel);
   for(int i=0;i<int(a.size());++i)for(int j=i;j<int(a.size())&&a[i].block==a[j].block;++j){
    int waves=(a[j].end+31)/32-a[i].start/32;
    TileCost v{dp[i].ns+perwave*waves+dispatch,dp[i].waves+waves,dp[i].commands+1};
    if(std::tie(v.ns,v.waves,v.commands)<std::tie(dp[j+1].ns,dp[j+1].waves,dp[j+1].commands)){dp[j+1]=v;back[j+1]=i;}}
   for(int j=a.size();j>0;){int i=back[j];assert(i>=0);groups.push_back({i,j});j=i;}
   std::reverse(groups.begin(),groups.end());
  }
  for(auto group:groups){int i=group.first,j=group.second,owners=0;
   for(int k=i;k<j;++k)owners|=1<<a[k].ordinal;
   result.push_back({chip,a[i].block,a[i].start/32,(a[j-1].end+31)/32-a[i].start/32,owners,__builtin_ctz(unsigned(owners))});}
 }
 std::sort(result.begin(),result.end(),[](auto a,auto b){return std::tie(a.priority,a.chip,a.block,a.start,a.waves)<std::tie(b.priority,b.chip,b.block,b.start,b.waves);});
 return result;
}

extern "C" void eval_gu_tile_dag(const int32_t* offsets,const int32_t* jobs,
 const int32_t* columns,const int32_t* ranges,const int32_t* routes,
 int experts,int job_count,int events,int active,int mode,const int64_t* config,
 int64_t* result,int64_t* bounds,int64_t* trace,int64_t* trace_nodes) {
 if(mode==0){eval_ordinary_read(offsets,jobs,columns,routes,experts,1,job_count,events,active,config,result,bounds,trace,trace_nodes);return;}
 Config c(config);assert(c.chips==16&&c.planes==4&&c.page==16384&&c.block==192);
 assert(experts==64&&active==6&&columns[0]==2048);
 for(int q=0;q<events;++q){
  std::vector<int> selected;for(int k=0;k<active;++k)selected.push_back(routes[q*active+k]);selected.push_back(experts);
  auto views=tile_views(ranges,selected,c,mode);
  Graph g(c);int submit=g.add(0,c.submit);
  int upload=g.add(2,c.transfer(2048LL*c.input_bytes,c.pcie),{submit});
  std::vector<std::vector<int>> outputs(selected.size());
  for(auto v:views){assert(v.start>=0&&v.start+v.waves<=192);
   int32_t d[]={v.chip,v.waves*32,2048,v.waves*4,v.waves};
   int done=g.matrix(v.priority,0,d,submit,upload);
   for(int k=0;k<int(selected.size());++k)if(v.owners&(1<<k))outputs[k].push_back(done);
  }
  std::vector<int> ready;
  for(int k=0;k<int(selected.size());++k){assert(!outputs[k].empty());ready.push_back(g.add(c.join_resource(),0,outputs[k],0,k));}
  std::vector<int> complete;
  for(int k=0;k<int(selected.size());++k){
   int e=selected[k];int activation=g.add(0,c.activation,{ready[k]},1,k);
   int ds=g.add(0,c.submit,{activation},1,k);
   int up=g.add(2,c.transfer(int64_t(columns[e*2+1])*c.input_bytes,c.pcie),{ds},1,k);
   const int32_t* o=offsets+e*4;
   for(int j=o[2];j<o[3];++j)complete.push_back(g.matrix(k,1,jobs+j*5,ds,up));
  }
  g.add(0,c.combine,complete,1);
  if(q==0&&trace_nodes)*trace_nodes=g.nodes.size();
  result[q]=g.run(q==0?trace:nullptr,bounds?bounds+q:nullptr);
 }
}
