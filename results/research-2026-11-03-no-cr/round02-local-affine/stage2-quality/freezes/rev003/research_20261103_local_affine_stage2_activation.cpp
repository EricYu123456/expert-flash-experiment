#include "research_20261103_local_affine_stage2_wire.hpp"
#include <iostream>
#include <omp.h>
extern "C" int la_swiglu(const float*,const float*,int,int,float*);
static bool strict_workers(){int bad=0,count=0;
 #pragma omp parallel num_threads(8) reduction(|:bad) reduction(+:count)
 {bad|=!laq::environment();++count;}return bad==0&&count==8;
}
int main(int argc,char**argv){try{
 omp_set_num_threads(8);laq::need(argc==2&&std::string(argv[1])=="--swiglu-worker","activation worker CLI");
 for(;;){laq::need(strict_workers(),"activation actual workers x87/SSE RN FTZ0 DAZ0");int32_t dims[2];std::cin.read((char*)dims,8);if(std::cin.eof()&&std::cin.gcount()==0)return 0;
  laq::need(bool(std::cin)&&dims[0]>0&&dims[0]<=256&&dims[1]>0&&dims[1]<=10944,"activation framing");size_t count=size_t(dims[0])*dims[1];std::vector<float>g(count),u(count),a(count);std::cin.read((char*)g.data(),4*count);std::cin.read((char*)u.data(),4*count);laq::need(bool(std::cin),"activation short input");
  int32_t rc=la_swiglu(g.data(),u.data(),dims[0],dims[1],a.data());laq::need(strict_workers()&&rc==0,"pinned full-I activation/worker environment");std::cout.write((char*)&rc,4);std::cout.write((char*)a.data(),4*count);std::cout.flush();laq::need(bool(std::cout),"activation write");
 }
 }catch(const std::exception&e){std::cerr<<"Stage2 activation invalid/tooling: "<<e.what()<<'\n';return 1;}}
