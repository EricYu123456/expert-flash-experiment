// Export canonical native W8 expert weights without loading a llama context.
// Reuses the frozen 11/08 dequantization/rounding; historical files unchanged.
#define main shared_feature_capture_entry_unused
#include "research_20261108_shared_feature_capture.cpp"
#undef main
int main(int argc,char**argv){try{
 need(argc==6,"Usage: MODEL RAW_LAYER EXPERT_IDS_CSV THREADS OUTDIR");int layer=std::stoi(argv[2]);threads=std::stoi(argv[4]);root=argv[5];need(layer>=1&&layer<=26&&threads>0,"Export arguments");need(!std::filesystem::exists(root),"Refuse weight export overwrite");std::filesystem::create_directories(root);omp_set_num_threads(threads);weight_ledger.open(root+"/weight_cache.jsonl");
 gguf_init_params gp{true,&meta_tensors};meta=gguf_init_from_file(argv[1],gp);need(meta&&meta_tensors,"Export metadata");int fd=open(argv[1],O_RDONLY);struct stat st;need(fd>=0&&fstat(fd,&st)==0,"Export model open");mapping_size=st.st_size;mapping=(const char*)mmap(nullptr,mapping_size,PROT_READ,MAP_SHARED,fd,0);close(fd);need(mapping!=MAP_FAILED,"Export mapping");
 std::stringstream list(argv[3]);std::string s;std::set<int>done;std::ofstream manifest(root+"/weights.json");manifest<<"{\"layer\":"<<layer<<",\"numeric_reference\":\"canonical_rowW8_from_Q8_GGUF\",\"experts\":{";bool comma=false;
 while(std::getline(list,s,',')){int e=std::stoi(s);need(e>=0&&e<64&&done.insert(e).second,"Export expert");if(comma)manifest<<',';comma=true;manifest<<'\"'<<e<<"\":{";
  bool sep=false;for(const std::string role:{"gate","up","down"}){const auto&w=weight(wn(layer,role+"_exps"),e);std::string prefix="expert"+std::to_string(e)+"."+role;std::ofstream q(root+"/"+prefix+".i8",std::ios::binary),scale(root+"/"+prefix+".scales.f32",std::ios::binary);q.write((const char*)w.q.data(),w.q.size());scale.write((const char*)w.scales.data(),w.scales.size()*4);need(bool(q)&&bool(scale),"Weight export write");manifest<<(sep?",":"")<<'\"'<<role<<"\":{\"prefix\":\""<<prefix<<"\",\"rows\":"<<w.R<<",\"columns\":"<<w.C<<"}";sep=true;}manifest<<'}';
 }
 manifest<<"}}\n";need(bool(manifest)&&!done.empty(),"Weight export manifest");munmap((void*)mapping,mapping_size);gguf_free(meta);ggml_free(meta_tensors);std::cout<<"EXPORTED layer="<<layer<<" experts="<<done.size()<<std::endl;return 0;
 }catch(const std::exception&e){std::cerr<<"native-target-export: "<<e.what()<<'\n';return 1;}}
