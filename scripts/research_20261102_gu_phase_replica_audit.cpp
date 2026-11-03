// Independent physical-byte interpreter: one common input and one accumulator.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
constexpr int H=1536,P=16384,CH=16,RR=1024,SR=2048,WR=192;
static void need(bool x,const std::string&s){if(!x)throw std::runtime_error(s);}
template<class T>static std::vector<T> read(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);need(bool(f),"open "+p);size_t n=f.tellg();need(n%sizeof(T)==0,"file geometry");f.seekg(0);std::vector<T>v(n/sizeof(T));f.read((char*)v.data(),n);need(bool(f),"read "+p);return v;}
struct Map{int fd;const int8_t*p;size_t n;explicit Map(const std::string&s){fd=open(s.c_str(),O_RDONLY);struct stat st;need(fd>=0&&fstat(fd,&st)==0,"mmap open");n=st.st_size;p=(const int8_t*)mmap(nullptr,n,PROT_READ,MAP_SHARED,fd,0);need(p!=MAP_FAILED,"mmap");}~Map(){munmap((void*)p,n);close(fd);}};
static int64_t dot(const int8_t*w,const int8_t*x,int n){int64_t z=0;
 #pragma omp simd reduction(+:z)
 for(int k=0;k<n;++k)z+=(int32_t)w[k]*(int32_t)x[k];return z;}
static uint32_t bits(float f){uint32_t x;std::memcpy(&x,&f,4);return x;}
struct Page{int e,ph,lp,plane,wl;bool drop;};
int main(int argc,char**argv){try{need(argc==4||argc==5,"directory canonical-fixture-dir output-prefix [fault|g8]");std::string d=argv[1],src=argv[2],out=argv[3],fault=argc==5?argv[4]:"";bool g8=fault=="g8";if(g8)fault="";
 Map canonical(src+"/canonical.w8");auto sw=read<float>(src+"/canonical.scales.f32");auto ids=read<int32_t>(d+"/routes.i32");auto xs=read<int8_t>(d+"/inputs.a8");auto sx=read<float>(d+"/inputs.scales.f32");auto co=g8?std::vector<int32_t>{}:read<int32_t>(d+"/coords.i32");int cases=ids.size()/6;need(ids.size()==cases*6&&xs.size()==cases*H&&sx.size()==cases&&(g8||co.size()==780*2),"input geometry");
 std::array<Map*,CH>banks;for(int c=0;c<CH;++c){char b[40];std::snprintf(b,sizeof(b),"/chip%02d.bin",c);banks[c]=new Map(d+b);need(banks[c]->n==4*WR*P,"bank size");}
 auto page=[&](int e,int ph,int lp,bool drop=false){int ix=e==64?768+lp:e*12+(ph==2?6:0)+lp;int plane=co[2*ix],wl=co[2*ix+1];need(plane>=0&&wl>=0&&wl<186,"unavailable phase read");need(plane==(e==64?lp%4:(ph+lp)%4),"physical plane phase");return Page{e,ph,lp,plane,wl,drop};};
 uint64_t rows=0,bytes=0,phys=0,dropped=0,dummy_checksum=0,partial_dummy=0;std::array<uint64_t,3>phases{};std::ofstream outputs(out+".outputs.bin",std::ios::binary);
 for(int t=0;t<cases;++t){std::array<bool,64>selected{};std::vector<int>A,B,D;for(int k=0;k<6;++k){int e=ids[t*6+k];need(e>=0&&e<64&&!selected[e],"route distinct");selected[e]=true;if(e<3)A.push_back(e);else if(e<6)B.push_back(e);else D.push_back(e);}std::sort(D.begin(),D.end());while(A.size()<3){need(!D.empty(),"Hall unavailable");A.push_back(D.front());D.erase(D.begin());}B.insert(B.end(),D.begin(),D.end());need(A.size()==3&&B.size()==3,"phase capacity");std::sort(A.begin(),A.end());std::sort(B.begin(),B.end());
  for(int mode=0;mode<2;++mode){std::vector<Page>stream;if(!g8){for(int k=0;k<3;++k){for(int lp=0;lp<6;++lp)stream.push_back(page(A[k],0,lp));for(int lp=0;lp<6;++lp)stream.push_back(page(B[k],2,lp));}for(int lp=0;lp<12;++lp)stream.push_back(page(64,0,lp));need(stream.size()==48,"useful pages");
   if(mode){int e=0;while(selected[e])++e;int ph=e>=3&&e<6?2:0;std::vector<Page>dummy;for(int pl=0;pl<4;++pl){int lp=0;while((ph+lp)%4!=pl)++lp;dummy.push_back(page(e,ph,lp,true));}stream.insert(stream.begin()+40,dummy.begin(),dummy.end());}}
   std::vector<int64_t>result(8192);std::vector<float>fp(8192);std::vector<uint8_t>seen(8192);
   for(int chip=0;chip<CH;++chip){int phase=0,owner=-1,ownerrow=0;int64_t acc=0;std::array<bool,4*WR>readseen{};int emitted=0;auto localstream=stream;
    if(g8){localstream.clear();const auto&group=chip<8?A:B;for(int e:group)for(int lp=0;lp<12;++lp){int pos=e<6?e%3:e-3;localstream.push_back(Page{e,0,lp,lp%4,3*pos+lp/4,false});}for(int lp=0;lp<12;++lp)localstream.push_back(Page{64,0,lp,lp%4,183+lp/4,false});if(mode){int e=chip<8?0:3;while(selected[e]){++e;if(chip<8&&e==3)e=6;if(chip>=8&&e==6)e=6;}int pos=e<6?e%3:e-3;std::vector<Page>dummy;for(int pl=0;pl<4;++pl)dummy.push_back(Page{e,0,pl,pl,3*pos,true});localstream.insert(localstream.begin()+40,dummy.begin(),dummy.end());}}
    for(size_t pi=0;pi<localstream.size();++pi){auto p=localstream[pi];need(p.plane==pi%4,"fixed0123 source mismatch");if(g8&&pi%4==0)for(int q=1;q<4;++q)need(localstream[pi+q].wl==p.wl,"ordinary read unexpectedly requires IPR");need(p.drop||p.e==64||selected[p.e],"unselected expert reached ECC");need(!readseen[p.wl*4+p.plane],"physical page read twice");readseen[p.wl*4+p.plane]=true;++phys;const int8_t*w=banks[chip]->p+((size_t)p.wl*4+p.plane)*P;
     if(p.drop){need(!selected[p.e],"dummy must be genuinely unselected");for(int k=0;k<P;++k)dummy_checksum+=(uint8_t)w[k];++dropped;partial_dummy+=phase!=0;need(fault!="decode_dummy","dummy entered ECC");continue;}
     if(owner!=p.e){need(phase==0&&acc==0,"owner changed inside row / extra context required");owner=p.e;ownerrow=0;}
     need(phase==(p.lp*P)%H,"input phase mismatch");++phases[phase/512];if(fault=="phase"&&t==0&&chip==0&&pi==1)phase=0;
     int off=0;while(off<P){int n=std::min(H-phase,P-off);acc+=dot(w+off,xs.data()+t*H+phase,n);bytes+=n;off+=n;phase+=n;
      if(phase==H){int localrow=g8?(chip%8)*128+ownerrow:chip*64+ownerrow;int logical=p.e==64?64*RR+chip*128+ownerrow:p.e*RR+localrow;int slot=6;if(p.e<64){slot=0;while(ids[t*6+slot]!=p.e)++slot;}int dst=p.e==64?6*RR+chip*128+ownerrow:slot*RR+localrow;need(dst<8192&&seen[dst]++==0,"scatter duplicate / overflow");int64_t ref=dot(canonical.p+(size_t)logical*H,xs.data()+t*H,H);need(acc==ref,"packed integer differs");float scale=sw[logical];if(fault=="scale"&&t==0&&chip==0&&pi==0)scale*=2;float y=(float)acc*(scale*sx[t]),yr=(float)ref*(sw[logical]*sx[t]);need(bits(y)==bits(yr),"row scale differs");result[dst]=acc;fp[dst]=y;++ownerrow;++emitted;++rows;phase=0;acc=0;}
     }
     if(p.lp==((p.e==64||g8)?11:5))need(phase==0&&acc==0&&ownerrow==((p.e==64||g8)?128:64),"expert full-row completion");
    }need(emitted==512&&phase==0&&acc==0,"perchip full output");
   }need(std::all_of(seen.begin(),seen.end(),[](int x){return x==1;}),"output coverage");outputs.write((char*)result.data(),result.size()*8);outputs.write((char*)fp.data(),fp.size()*4);
  }
 }
 for(auto*b:banks)delete b;std::ofstream r(out+".json");r<<"{\"status\":\"pass\",\"cases\":"<<cases<<",\"modes\":2,\"chips\":16,\"row_comparisons\":"<<rows<<",\"scaled_f32_bit_comparisons\":"<<rows<<",\"ECC_weight_bytes\":"<<bytes<<",\"physical_pages\":"<<phys<<",\"dummy_pages_dropped_before_ECC\":"<<dropped<<",\"dummy_checksum\":"<<dummy_checksum<<",\"dummy_read_pages_while_partial_row_live\":"<<partial_dummy<<",\"page_start_phase_counts\":["<<phases[0]<<","<<phases[1]<<","<<phases[2]<<"],\"mismatches\":0}\n";std::cout<<"PASS "<<cases<<" cases "<<rows<<" rows\n";return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<'\n';return 1;}}
