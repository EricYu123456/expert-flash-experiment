#!/usr/bin/env python3
"""Compare all fixture input quantizers and rescalings with frozen C++ helpers."""
import hashlib,json,pathlib,struct,subprocess
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];OUT=ROOT/'results/research-2026-11-02/round2/routed-boundary-view-numeric'
REF=ROOT/'scripts/research_20261102_full_ffn_integer_reference.cpp'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def quant(x):
 m=np.max(np.abs(x));s=np.float32(1) if m==0 else np.float32(m/np.float32(127));v=(x/s).astype(np.float32)
 return np.clip(np.copysign(np.floor(np.abs(v.astype(np.float64))+.5),v),-127,127).astype(np.int8),s
def main():
 dest=OUT/'cpp-helper-audit';dest.mkdir(exist_ok=False);text=REF.read_text();body=text[text.index('static float quantize('):text.index('static bool target(')]
 prefix='#include <algorithm>\n#include <cmath>\n#include <cstdint>\n#include <cstdio>\n#include <cstring>\n#include <stdexcept>\n#include <vector>\n'
 driver=r'''
template<class T>static void read(FILE*f,T*p,size_t n){if(std::fread(p,sizeof(T),n,f)!=n)throw std::runtime_error("fixture read");}
int main(int argc,char**argv){if(argc!=2)return 2;FILE*f=std::fopen(argv[1],"rb");if(!f)return 3;uint32_t cases;read(f,&cases,1);uint64_t rows=0,components=0;
for(uint32_t j=0;j<cases;++j){uint32_t C,R;read(f,&C,1);read(f,&R,1);std::vector<float>x(C),sc(R),ref(R);std::vector<int8_t>q(C),expected(C);std::vector<int64_t>integer(R);float expected_scale;read(f,x.data(),C);read(f,expected.data(),C);read(f,&expected_scale,1);read(f,integer.data(),R);read(f,sc.data(),R);read(f,ref.data(),R);float sx=quantize(x.data(),C,q.data());if(!equal_bits(sx,expected_scale)||q!=expected)throw std::runtime_error("quantizer mismatch");for(uint32_t r=0;r<R;++r)if(!equal_bits(scale_output(integer[r],sc[r],sx),ref[r]))throw std::runtime_error("rescale mismatch");rows+=R;components+=C;}
if(std::fgetc(f)!=EOF)throw std::runtime_error("trailing fixture bytes");std::fclose(f);std::printf("{\"status\":\"PASS\",\"operator_vectors\":%u,\"A8_components\":%llu,\"I64_to_F32_rescaled_rows\":%llu,\"bit_mismatches\":0}\n",cases,(unsigned long long)components,(unsigned long long)rows);}
'''
 src=dest/'frozen_helper_check.cpp';src.write_text(prefix+body+driver);man=json.loads((OUT/'manifest.json').read_text());s=json.loads((OUT/'summary.json').read_text());fx=np.load(OUT/'fixtures_and_canonical_outputs.npz');sc=np.fromfile(OUT/'logical_scales.f32',dtype='<f4');keys=[f'routed.{e}' for e in s['experts']]+['shared'];inp=dest/'operator_vectors_and_reference_rescales.bin'
 with inp.open('wb') as f:
  f.write(struct.pack('<I',28))
  for fi in [0,1]:
   for k in keys:
    for phase,x in [('gu',fx[f'x{fi}']),('down',fx[f'f{fi}.{k}.activation'])]:
     q,sx=quant(x);i=fx[f'f{fi}.{k}.{phase}_i64'];y=fx[f'f{fi}.{k}.{phase}'];sm=man['scale_meta'][k+'.'+phase];sw=sc[sm['offset']//4:sm['offset']//4+sm['rows']];f.write(struct.pack('<II',len(x),len(i)))
     for a in [x,q,np.asarray(sx),i,sw,y]:f.write(a.tobytes())
 binary=dest/'frozen_helper_check';argv=['g++','-std=c++17','-O2','-fopenmp','-fno-fast-math','-ffp-contract=off',str(src),'-o',str(binary)];subprocess.run(argv,check=True);run=subprocess.run([str(binary),str(inp)],check=True,capture_output=True,text=True);report=json.loads(run.stdout);report.update(reference_source=str(REF),reference_source_sha256=sha(REF),helper_body_sha256=hashlib.sha256(body.encode()).hexdigest(),helper_extraction_byte_exact=True,build_argv=argv,source_sha256=sha(src),binary_sha256=sha(binary),fixture_sha256=sha(inp),scope='Fixture A8 and I64-to-F32 scaling checked with verbatim frozen C++ quantize/scale_output/equal_bits; exact integer dots are independently checked by page-pread replay. No weight conversion, inference or activation implementation claim.')
 (dest/'report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
