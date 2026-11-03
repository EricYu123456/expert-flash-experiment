// Independent page-stream interpreter for the conditional CR+IPR GU stencil.
// One whole-row integer accumulator; one common A8 vector; positional scatter.
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
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
constexpr int H=1536,P=16384,E=64,CH=16,RR=1024,SR=2048,WR=192;
constexpr size_t NROWS=E*RR+SR;
static void need(bool x,const std::string&s){if(!x)throw std::runtime_error(s);}
template<class T>static std::vector<T> read(const std::string&p){std::ifstream f(p,std::ios::binary|std::ios::ate);need(bool(f),"open "+p);size_t n=f.tellg();need(n%sizeof(T)==0,"size "+p);f.seekg(0);std::vector<T> v(n/sizeof(T));f.read((char*)v.data(),n);need(bool(f),"read "+p);return v;}
static int64_t dot(const int8_t*w,const int8_t*x,int n){int64_t z=0;
 #pragma omp simd reduction(+:z)
 for(int i=0;i<n;++i)z+=(int32_t)w[i]*(int32_t)x[i];return z;}
static uint32_t bits(float f){uint32_t v;std::memcpy(&v,&f,4);return v;}
static float scale(int64_t z,float sw,float sx){float f=(float)z;float s=sw*sx;return f*s;}
struct Map{int fd;const int8_t*p;size_t n;explicit Map(const std::string&s){fd=open(s.c_str(),O_RDONLY);struct stat st;need(fd>=0&&fstat(fd,&st)==0,"mmap open");n=st.st_size;p=(const int8_t*)mmap(nullptr,n,PROT_READ,MAP_SHARED,fd,0);need(p!=MAP_FAILED,"mmap");}~Map(){munmap((void*)p,n);close(fd);}};
struct Pg{int plane,wl,expert,lp;bool shared;};
static Pg page(bool shared,int e,int lp){if(!shared){static const int pp[]={0,1,2,0,1,3};return {pp[lp],lp<2?2*e:lp==2?e:lp<5?2*e+1:e,e,lp,false};}static const int pp[]={3,2,3,2,3,2,3,2,3,2,3,2};static const int ww[]={64,64,65,65,66,66,67,67,68,68,69,69};return {pp[lp],ww[lp],-1,lp,true};}
static size_t row_index(bool sh,int expert,int r){return sh?(size_t)E*RR+r:(size_t)expert*RR+r;}
int main(int argc,char**argv){try{need(argc==4||argc==5,"arguments: directory case-count output-prefix [fault]");std::string dir=argv[1],out=argv[3],fault=argc==5?argv[4]:"";int cases=std::stoi(argv[2]);
 auto weights=read<int8_t>(dir+"/canonical.w8");auto scales=read<float>(dir+"/canonical.scales.f32");auto ids=read<int32_t>(dir+"/routes.i32");auto xs=read<int8_t>(dir+"/inputs.a8");auto sx=read<float>(dir+"/inputs.scales.f32");need(weights.size()==NROWS*H&&scales.size()==NROWS&&ids.size()==cases*6&&xs.size()==cases*H&&sx.size()==cases,"input geometry");
 std::array<Map*,CH> maps;for(int c=0;c<CH;++c){char b[50];std::snprintf(b,sizeof(b),"/chip%02d.bin",c);maps[c]=new Map(dir+b);need(maps[c]->n==4*WR*P,"bank geometry");}
 std::ofstream output(out+".outputs.bin",std::ios::binary);need(bool(output),"output open");uint64_t rows=0,pages=0,macs=0,cross_page_rows=0;int64_t maxabs=0;std::array<uint64_t,3> phases{};
 const int orders[6][4]={{0,1,2,3},{2,3,0,1},{2,0,1,3},{0,1,3,2},{3,2,0,1},{2,0,1,3}};
 for(int t=0;t<cases;++t){std::array<bool,E> selected{};for(int k=0;k<6;++k){int e=ids[t*6+k];need(e>=0&&e<E&&!selected[e],"route distinct/range");selected[e]=true;}
  std::vector<int64_t> result(6*RR+SR);std::vector<float> fresult(result.size());std::vector<uint8_t> seen(result.size());const int8_t*x=xs.data()+(size_t)t*H;
  for(int chip=0;chip<CH;++chip){int phase=0;int64_t acc=0;int row_start_page=-1;int emitted=0;std::array<uint8_t,4*WR> readseen{};
   for(int g=0;g<2;++g){std::vector<Pg> stream;std::vector<std::pair<int,size_t>> metadata;
    // Macro sequence is A1,D,A2,B1,B2,C,A3,B3; metadata is only a host FIFO map.
    const int which[]={0,-1,1,0,1,-2,2,2};const int halves[]={0,0,0,1,1,0,0,1};
    for(int m=0;m<8;++m){bool sh=which[m]<0;int expert=sh?-1:ids[t*6+g*3+which[m]];int lp=sh?g*6+(which[m]==-2?3:0):halves[m]*3;for(int j=0;j<3;++j)stream.push_back(page(sh,expert,lp+j));
     int start=sh?chip*128+g*64+(which[m]==-2?32:0):chip*64+halves[m]*32;
     for(int j=0;j<32;++j){int dst=sh?6*RR+start+j:(g*3+which[m])*RR+start+j;metadata.emplace_back(dst,row_index(sh,expert,start+j));}}
    need(stream.size()==24&&metadata.size()==256,"macro geometry");int emit_group=0;
    for(int wave=0;wave<6;++wave){std::array<const int8_t*,4> pb{};std::array<Pg,4> pg{};
     for(int q=0;q<4;++q){auto a=stream[wave*4+q];if(fault=="unselected"&&t==0&&chip==0&&g==0&&wave==0&&q==0){int bad=0;while(selected[bad])++bad;a=page(false,bad,0);}need(a.shared||selected[a.expert],"unselected expert read");need(!pb[a.plane]&&a.wl<WR,"duplicate plane / block overflow");need(readseen[a.wl*4+a.plane]++==0,"physical page reread");size_t off=((size_t)a.wl*4+a.plane)*P;pb[a.plane]=maps[chip]->p+off;pg[a.plane]=a;++pages;}
     for(int q=0;q<4;++q){int pl=orders[wave][q];if(fault=="permutation"&&t==0&&chip==0&&g==0&&wave==0)pl=(pl+1)%4;const auto a=pg[pl];const auto expected=stream[wave*4+q];need(a.shared==expected.shared&&a.expert==expected.expert&&a.lp==expected.lp,"plane consumption order");need(phase==((wave*4+q)*P)%H,"input phase at page boundary");++phases[phase/512];if(fault=="phase"&&t==0&&chip==0&&g==0&&wave==0&&q==1)phase=0;
      int offset=0;while(offset<P){if(phase==0)row_start_page=wave*4+q;int n=std::min(H-phase,P-offset);int64_t partial=dot(pb[pl]+offset,x+phase,n);if(fault=="weight"&&t==0&&chip==0&&g==0&&wave==0&&q==0&&offset==0)partial+=1;acc+=partial;phase+=n;offset+=n;macs+=n;
       if(phase==H){need(emit_group<256,"too many rows");auto md=metadata[emit_group++];if(fault=="scatter"&&t==0&&chip==0&&g==0&&emit_group==1)++md.first;need(md.first>=0&&md.first<(int)result.size()&&seen[md.first]++==0,"scatter duplicate");size_t logical=md.first<6*RR?(size_t)ids[t*6+md.first/RR]*RR+md.first%RR:(size_t)E*RR+md.first-6*RR;need(logical==md.second,"scatter logical row identity");int64_t ref=dot(weights.data()+md.second*H,x,H);need(acc==ref,"integer result differs from canonical row");float sw=scales[md.second];if(fault=="scale"&&t==1&&chip==0&&g==0&&emit_group==1)sw*=2.0f;float f=scale(acc,sw,sx[t]),rf=scale(ref,scales[md.second],sx[t]);need(bits(f)==bits(rf),"scaled F32 bits differ");result[md.first]=acc;fresult[md.first]=f;maxabs=std::max(maxabs,std::abs(acc));cross_page_rows+=row_start_page!=wave*4+q;++rows;++emitted;phase=0;acc=0;}
      }
     }
    }
    need(phase==0&&acc==0&&emit_group==256,"group leaves row partial");
   }need(emitted==512,"perchip output count");
  }
  need(std::all_of(seen.begin(),seen.end(),[](uint8_t n){return n==1;}),"output missing / duplicated");output.write((char*)result.data(),result.size()*8);output.write((char*)fresult.data(),fresult.size()*4);need(bool(output),"output write");if(t%100==0)std::cout<<"completed "<<t+1<<"/"<<cases<<std::endl;
 }
 for(auto*m:maps)delete m;std::ofstream r(out+".json");r<<"{\"status\":\"pass\",\"cases\":"<<cases<<",\"chips\":16,\"row_comparisons\":"<<rows<<",\"scaled_f32_comparisons\":"<<rows<<",\"weight_bytes_consumed\":"<<macs<<",\"physical_pages_read\":"<<pages<<",\"cross_page_rows\":"<<cross_page_rows<<",\"page_start_phase_counts\":["<<phases[0]<<","<<phases[1]<<","<<phases[2]<<"],\"max_abs_integer\":"<<maxabs<<",\"mismatches\":0}\n";
 return 0;
 }catch(const std::exception&e){std::cerr<<"FAIL: "<<e.what()<<std::endl;return 1;}}
