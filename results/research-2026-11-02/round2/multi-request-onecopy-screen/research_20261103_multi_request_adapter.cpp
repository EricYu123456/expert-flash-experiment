// Appended to frozen ordinary_read_native.cpp; reuses the identical Graph engine.
extern "C" void eval_ready_cohorts(const int32_t*offsets,const int32_t*jobs,
 const int32_t*columns,const int32_t*routes,int experts,int groups,int batch,int active,
 const int64_t*config,int64_t*cost,int64_t*bounds,int64_t*trace,int64_t*ntrace){
 Config c(config);
 for(int q=0;q<groups;++q){Graph g(c);
  for(int request=0;request<batch;++request){
   int submit=g.add(0,c.submit);int upload=g.add(2,c.transfer(int64_t(columns[0])*c.input_bytes,c.pcie),{submit});
   std::vector<int> chosen,ready;
   for(int k=0;k<active;++k)chosen.push_back(routes[(q*batch+request)*active+k]);chosen.push_back(experts);
   for(int ei=0;ei<int(chosen.size());++ei){int e=chosen[ei],ordinal=request*(active+1)+ei;std::vector<int> out;const int32_t*o=offsets+e*4;
    for(int j=o[0];j<o[1];++j)out.push_back(g.matrix(ordinal,0,jobs+j*5,submit,upload));ready.push_back(g.add(c.join_resource(),0,out,0,ordinal));}
   std::vector<int> complete;
   for(int ei=0;ei<int(chosen.size());++ei){int e=chosen[ei],ordinal=request*(active+1)+ei;int act=g.add(0,c.activation,{ready[ei]},1,ordinal);int ds=g.add(0,c.submit,{act},1,ordinal);int up=g.add(2,c.transfer(int64_t(columns[e*2+1])*c.input_bytes,c.pcie),{ds},1,ordinal);const int32_t*o=offsets+e*4;
    for(int j=o[2];j<o[3];++j)complete.push_back(g.matrix(ordinal,1,jobs+j*5,ds,up));}
   g.add(0,c.combine,complete,1,request*(active+1));
  }
  if(q==0&&ntrace)*ntrace=g.nodes.size();cost[q]=g.run(q==0?trace:nullptr,bounds+q);
 }
}
