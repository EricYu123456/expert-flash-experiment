// Export-only wrapper around the exact immutable GGUF row dequant/row-I8 code.
#define main stage0_immutable_entry_not_called
#include "/home/ericyu/expert_flash_experiment/results/research-2026-11-03-no-cr/round02-local-affine/stage0/freezes/rev003/research_20261103_local_affine_stage0_capture.cpp"
#undef main
#include <cfenv>
#include <immintrin.h>
static bool strict_environment(){return std::fegetround()==FE_TONEAREST&&(_mm_getcsr()&((1u<<15)|(1u<<6)|0x6000u))==0;}
int main(int argc,char**argv){try{
 need(strict_environment(),"Stage4 exporter x87/SSE RN FTZDAZ0");omp_set_num_threads(8);
 if(argc==2&&std::string(argv[1])=="synthetic"){
  std::vector<float>x={0.f,-0.f,.5f,-.5f,1.f,-1.f,127.f,-127.f};std::vector<int8_t>q(x.size());float scale=0;
  need(la_quantize(x.data(),1,x.size(),q.data(),&scale)==0&&scale==1.f,"synthetic scale");
  for(size_t i=0;i<x.size();++i)need(q[i]==int8_t(std::round(x[i])),"immutable ties-away row quantizer");
  auto orig=_mm_getcsr();for(unsigned mode:{0x2000u,0x4000u,0x6000u}){_mm_setcsr((orig&~0x6000u)|mode);need(!strict_environment(),"MXCSR-only non-RN must fail");}_mm_setcsr(orig);
  need(strict_environment(),"exporter restored environment");std::cout<<"{\"status\":\"passed\",\"generated_only\":true,\"row_ties_away\":true,\"MXCSR_only_modes_rejected\":true}\n";return 0;
 }
 need(argc==7&&std::string(argv[1])=="weights","weights GGUF model layer expert output");std::string name=argv[3];int layer=std::stoi(argv[4]),expert=std::stoi(argv[5]);
 need((name=="deepseek"&&layer==1)||(name=="olmoe"&&layer==0),"Stage4 earliest-only adapter");need(expert>=0&&expert<64,"Stage4 expert range");
 ggml_context*ctx=nullptr;gguf_init_params gp{true,&ctx};auto*gf=gguf_init_from_file(argv[2],gp);need(gf&&ctx,"Stage4 export metadata");int I=name=="deepseek"?1408:1024;
 for(const std::string role:{"gate","up","down"}){auto n="blk."+std::to_string(layer)+".ffn_"+role+"_exps.weight";auto*t=ggml_get_tensor(ctx,n.c_str());need(t&&t->ne[0]==(role=="down"?I:2048)&&t->ne[1]==(role=="down"?2048:I)&&t->ne[2]==64&&t->ne[3]==1,"Stage4 exact modelI GUDown shape");}
 need(!ggml_get_tensor(ctx,("blk."+std::to_string(layer)+".ffn_gate_up_exps.weight").c_str()),"Stage4 mergedGU unsupported");gguf_free(gf);ggml_free(ctx);
 export_weights(argv[2],layer,expert,argv[6]);need(strict_environment(),"exporter environment unchanged");return 0;
 }catch(const std::exception&e){std::cerr<<"Stage4 weights technical-invalid: "<<e.what()<<'\n';return 1;}}
