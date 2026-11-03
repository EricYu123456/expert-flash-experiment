// Vocab-only deterministic prompt formatting. No graph, FFN or inference.
#include "llama.h"
#include "ggml-backend.h"
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

static std::string read(const std::string&p){std::ifstream f(p);if(!f)throw std::runtime_error("field read");return {(std::istreambuf_iterator<char>(f)),{}};}
static std::string render(const std::array<std::string,3>&s){return s[0]+"\n1. "+s[1]+"\n2. "+s[2]+"\n";}
static int count(const llama_vocab*v,const std::string&s){int n=-llama_tokenize(v,s.data(),s.size(),nullptr,0,true,true);if(n<=0)throw std::runtime_error("token count");return n;}
static void trim_last(std::string&s){if(s.empty())throw std::runtime_error("empty field");size_t k=s.size()-1;while(k>0&&((unsigned char)s[k]&0xc0)==0x80)--k;if(k==0)throw std::runtime_error("Cannot retain nonempty fields");s.resize(k);}
int main(int argc,char**argv){try{
 if(argc!=5)throw std::runtime_error("model fields.tsv tokenlimit receipt.tsv");
 int limit=std::stoi(argv[3]);if(limit!=256)throw std::runtime_error("Fixed limit256");
 llama_backend_init();ggml_backend_load_all();auto params=llama_model_default_params();params.n_gpu_layers=0;params.vocab_only=true;
 auto*m=llama_model_load_from_file(argv[1],params);if(!m)throw std::runtime_error("vocab-only model load");auto*v=llama_model_get_vocab(m);
 std::ifstream jobs(argv[2]);std::ofstream receipt(argv[4]);if(!jobs||!receipt)throw std::runtime_error("job/receipt");
 std::string line;int n_jobs=0;
 while(std::getline(jobs,line)){
  std::vector<std::string> f;std::istringstream ss(line);std::string x;while(std::getline(ss,x,'\t'))f.push_back(x);
  if(f.size()!=5)throw std::runtime_error("TSV id+3fields+out");
  std::array<std::string,3> s={read(f[1]),read(f[2]),read(f[3])};for(auto&a:s)if(a.empty())throw std::runtime_error("Empty source field");
  auto original=render(s);int before=count(v,original),removed=0;auto prompt=original;
  while(count(v,prompt)>limit){size_t k=0;for(size_t i=1;i<3;++i)if(s[i].size()>s[k].size())k=i;trim_last(s[k]);++removed;prompt=render(s);}
  int after=count(v,prompt);if(std::filesystem::exists(f[4]))throw std::runtime_error("Refuse overwrite prompt");
  std::ofstream out(f[4]);out<<prompt;if(!out)throw std::runtime_error("Prompt write");
  receipt<<f[0]<<'\t'<<before<<'\t'<<after<<'\t'<<removed<<'\t'<<prompt.size()<<'\n';++n_jobs;
 }
 if(n_jobs!=32)throw std::runtime_error("Expected32 fixed prompts");
 llama_model_free(m);llama_backend_free();return 0;
 }catch(const std::exception&e){std::cerr<<"prospective-tokenizer: "<<e.what()<<'\n';return 1;}}
