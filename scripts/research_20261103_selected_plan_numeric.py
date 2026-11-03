#!/usr/bin/env python3
"""Paired fixed-W8 full-FFN reference for frozen compiler complete-row bands."""
import argparse,collections,hashlib,json,pathlib,subprocess,time
import numpy as np
ROOT=pathlib.Path('/home/ericyu/expert_flash_experiment');NFS=pathlib.Path('/home/nfs_client/ericyu/expert_flash_experiment');BASE=ROOT/'results/research-2026-11-02/round2';OUT=BASE/'selected-plan-numeric';SELF=pathlib.Path(__file__).resolve()
CFG={
 'granite4tiny':dict(source=ROOT/'scripts/research_20261102_full_ffn_integer_reference.cpp',build=BASE/'full-ffn-integer-reference/build_command.json',protocol=BASE/'full-ffn-integer-reference/frozen_protocol.json',model=NFS/'models/granite4tiny-q4_k_m.gguf',H=1536,I=512,SI=1024,layers=list(range(40)),dense=None,vocab=100352),
 'deepseek':dict(source=ROOT/'scripts/research_20261103_deepseek_full_ffn.cpp',build=BASE/'deepseek-full-ffn-integer-reference/build_command.json',protocol=BASE/'deepseek-full-ffn-integer-reference/protocol.frozen.json',model=NFS/'models/deepseek-v2-lite-chat-q8_0.gguf',H=2048,I=1408,SI=2816,layers=list(range(1,27)),dense=10944,vocab=102400)}
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(8<<20),b''):h.update(b)
 return h.hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def split_gu(start,rows,intermediate):
 assert 0<=start and rows>=0 and start+rows<=2*intermediate
 return [(projection,max(start,base)-base,min(start+rows,base+intermediate)-max(start,base)) for projection,base in [('gate',0),('up',intermediate)] if min(start+rows,base+intermediate)>max(start,base)]
def lower_layer(plan,cfg):
 h,i,experts,k,L,si=plan['shape'];assert (h,i,experts,k,L,si)==(cfg['H'],cfg['I'],64,6,len(cfg['layers']),cfg['SI'])
 layer=plan['layer'];assert layer in cfg['layers'] and plan['row_coverage_and_address_nonoverlap']
 covered={(e,p):np.zeros(2*(si if e==64 else i) if p==0 else h,dtype=np.uint8) for e in range(65) for p in range(2)};bands=[];ledger=[]
 for ji,b in enumerate(plan['logical']):
  e,p,chip,start,n,c=(b[k] for k in ['expert','phase','chip','row_start','rows','columns']);assert 0<=e<=64 and p in (0,1) and 0<=chip<16 and start>=0 and n>0
  mid=si if e==64 else i;assert c==(h if p==0 else mid) and start+n<=len(covered[e,p]);covered[e,p][start:start+n]+=1
  parts=split_gu(start,n,mid) if p==0 else [('down',start,n)]
  for projection,local,count in parts:
   projection+= '_shexp' if e==64 else '_exps';source_expert=0 if e==64 else e;R=mid if p==0 else h
   bands.append((layer,projection,source_expert,c,R,chip,local,count));ledger.append(dict(layer=layer,native_job_index=ji,native_expert=e,native_phase=p,native_row_start=start,native_rows=n,projection=projection,source_expert=source_expert,chip=chip,columns=c,projection_row_start=local,rows=count))
 assert all(np.all(v==1) for v in covered.values()),'Native logical row intervals must cover exactly once'
 return bands,ledger

def selfcheck():
 count=0
 # All splits at/around the GU boundary, including zero and a band crossing both halves.
 for mid in [1,7,512,1024,1408,2816]:
  cases={(0,0),(0,2*mid),(mid,0),(mid,mid),(0,mid),(max(0,mid-1),min(2,mid+1)),(2*mid,0)}
  for start,n in cases:
   parts=split_gu(start,n,mid);expanded=[]
   for p,r,k in parts:expanded.extend((0 if p=='gate' else mid)+x for x in range(r,r+k))
   assert expanded==list(range(start,start+n));count+=1
 # Synthetic complete plans with zero chips and a GU band across the gate/up boundary.
 for name,cfg in CFG.items():
  logical=[]
  for e in range(65):
   mid=cfg['SI'] if e==64 else cfg['I']
   for p,total in [(0,2*mid),(1,cfg['H'])]:
    # Deliberately only two active chips, one GU band straddles the boundary.
    cut=mid-1 if p==0 else total//3
    for chip,start,n in [(0,0,cut),(15,cut,total-cut)]:
     logical.append(dict(expert=e,phase=p,chip=chip,row_start=start,rows=n,columns=cfg['H'] if p==0 else mid))
  plan=dict(shape=[cfg['H'],cfg['I'],64,6,len(cfg['layers']),cfg['SI']],layer=cfg['layers'][0],row_coverage_and_address_nonoverlap=True,logical=logical)
  bands,_=lower_layer(plan,cfg);assert {b[5] for b in bands}=={0,15};count+=1
 return dict(status='pass',GU_interval_and_zero_chip_cases=count)

BAND_GLOBALS=r'''
struct PhysicalBand{int chip=0,start=0,rows=0;};
struct PhysicalRows{int columns=0,rows=0;std::vector<PhysicalBand> bands;};
static std::map<std::string,PhysicalRows> physical_rows;
static std::string physical_key(int layer,const std::string &projection,int expert){return std::to_string(layer)+"/"+projection+"/"+std::to_string(expert);}
static void load_physical_rows(const std::string&path){
 std::ifstream f(path);if(!f)throw std::runtime_error("Physical row TSV missing");std::string line;size_t lines=0;
 while(std::getline(f,line)){std::istringstream s(line);int layer,e,C,R,chip,start,n;std::string p,extra;
  if(!(s>>layer>>p>>e>>C>>R>>chip>>start>>n)||(s>>extra)||e<0||e>=64||C<=0||R<=0||chip<0||chip>=16||start<0||n<0||start>R||n>R-start)throw std::runtime_error("Physical row TSV invalid");
  auto &dst=physical_rows[physical_key(layer,p,e)];if(dst.rows&&(dst.rows!=R||dst.columns!=C))throw std::runtime_error("Physical row TSV shape drift");dst.columns=C;dst.rows=R;dst.bands.push_back({chip,start,n});++lines;
 }
 if(!lines||physical_rows.size()!=EXPECTEDKEYS)throw std::runtime_error("Physical row key coverage invalid");
 for(const auto &entry:physical_rows){std::vector<int>seen(entry.second.rows,0);for(const auto&b:entry.second.bands)for(int r=b.start;r<b.start+b.rows;++r)++seen[r];if(std::any_of(seen.begin(),seen.end(),[](int n){return n!=1;}))throw std::runtime_error("Physical row TSV is not a complete-row bijection");}
}
static uint64_t finite_ffn_values=0,finite_continuation_logits=0;
static void verify_ffn_finite(const ggml_tensor*t){
 if(t->type!=GGML_TYPE_F32||t->ne[3]!=1)throw std::runtime_error("FFN finite-check layout");
 for(int64_t z=0;z<t->ne[2];++z)for(int64_t y=0;y<t->ne[1];++y)for(int64_t r=0;r<t->ne[0];++r){float v;std::memcpy(&v,(const char*)t->data+z*t->nb[2]+y*t->nb[1]+r*t->nb[0],4);if(!std::isfinite(v))throw std::runtime_error("Nonfinite FFN output");++finite_ffn_values;}
}
'''
LAYOUT=r''' std::array<std::vector<int>,16> chip_rows;
 const bool shared=projection.find("shexp")!=std::string::npos;
 const auto &rowmap=physical_rows.at(physical_key(layer,projection,expert));
 if(rowmap.columns!=c.C||rowmap.rows!=c.R)throw std::runtime_error("Physical row map vs GGUF mismatch");
 for(const auto &band:rowmap.bands)for(int r=band.start;r<band.start+band.rows;++r)chip_rows[band.chip].push_back(r);
'''
MAIN=r'''
int main(int argc,char**argv){if(argc!=7){std::fprintf(stderr,"Usage: %s MODEL JOBS STEPS THREADS ROWS_TSV OUTDIR\n",argv[0]);return 2;}try{
 int steps=std::stoi(argv[3]),threads=std::stoi(argv[4]);if(steps<1||threads<1||threads>32)throw std::runtime_error("Invalid bounds");root=argv[6];load_physical_rows(argv[5]);
 std::filesystem::create_directories(root+"/scales");std::filesystem::create_directories(root+"/traces");omp_set_num_threads(threads);ledger.open(root+"/coverage_ledger.jsonl");cache_ledger.open(root+"/weight_cache_ledger.jsonl");
 gguf_init_params gp{true,nullptr};gguf=gguf_init_from_file(argv[1],gp);if(!gguf)throw std::runtime_error("GGUF metadata failed");int fd=open(argv[1],O_RDONLY);struct stat st;if(fd<0||fstat(fd,&st))throw std::runtime_error("Model open failed");mapping_size=st.st_size;void*m=mmap(nullptr,mapping_size,PROT_READ,MAP_SHARED,fd,0);close(fd);if(m==MAP_FAILED)throw std::runtime_error("Model mmap failed");mapping=(const char*)m;
 llama_backend_init();ggml_backend_load_all();auto mp=llama_model_default_params();mp.n_gpu_layers=0;auto*model=llama_model_load_from_file(argv[1],mp);if(!model)throw std::runtime_error("Model load failed");
 std::ifstream jobs(argv[2]);std::string line;int njobs=0;while(std::getline(jobs,line)){auto tab=line.find('\t');if(tab==std::string::npos)throw std::runtime_error("Jobs TSV invalid");std::string path=line.substr(0,tab),id=line.substr(tab+1);++njobs;
  std::string output_path=root+"/traces/"+id+".unsharded.ffn_outputs.f32",logit_path=root+"/traces/"+id+".unsharded.logits.f32";
  for(const std::string &variant:{std::string("unsharded"),std::string("placed")}){mode=variant;reference_outputs=std::fopen(output_path.c_str(),mode=="unsharded"?"wb":"rb");reference_logits=std::fopen(logit_path.c_str(),mode=="unsharded"?"wb":"rb");if(!reference_outputs||!reference_logits)throw std::runtime_error("Snapshot open failed");run(model,path,id,steps,threads);
   if(mode=="placed"&&(std::fgetc(reference_outputs)!=EOF||std::fgetc(reference_logits)!=EOF))throw std::runtime_error("Placed consumed fewer values");std::fclose(reference_outputs);std::fclose(reference_logits);ledger.flush();cache_ledger.flush();
  }
 }
 if(!njobs||integer_differences||scaled_differences||cross_output_differences||cross_logits_differences)throw std::runtime_error("Incomplete or nonexact numeric comparison");
 std::ofstream report(root+"/numeric_report.json");report<<"{\"status\":\"complete\",\"model\":\"MODELNAME\",\"jobs\":"<<njobs<<",\"steps\":"<<steps<<",\"all_FFN_canonical_baseline\":true,\"shared_uses_selected_rows\":true,\"GGUF_fullrow_W8_fullvector_A8_unchanged\":true,\"finite_FFN_values_both_modes\":"<<finite_ffn_values<<",\"finite_continuation_logit_values_both_modes\":"<<finite_continuation_logits<<",\"integer_checks\":"<<integer_checks<<",\"integer_mismatches\":"<<integer_differences<<",\"same_input_scaled_bit_mismatches\":"<<scaled_differences<<",\"cross_run_output_floats\":"<<cross_output_floats<<",\"cross_run_output_bit_mismatches\":"<<cross_output_differences<<",\"cross_run_logits_floats\":"<<cross_logits_floats<<",\"cross_run_logits_bit_mismatches\":"<<cross_logits_differences<<",\"weight_cache_entries\":"<<caches.size()<<",\"physical_rowmap_keys\":"<<physical_rows.size()<<",\"native_flash_programming_claimed\":false}\n";
 report.close();ledger.close();cache_ledger.close();llama_model_free(model);llama_backend_free();munmap((void*)mapping,mapping_size);gguf_free(gguf);return 0;
 }catch(const std::exception&e){std::fprintf(stderr,"selected-plan-numeric: %s\n",e.what());return 1;}}
'''

def build():
 OUT.mkdir(exist_ok=True);b=OUT/'build';b.mkdir(exist_ok=True);proofs={};inventory={}
 for name,cfg in CFG.items():
  src=ROOT/'scripts'/f'research_20261103_selected_plan_numeric_{name}.cpp';binary=b/name
  if src.exists() or binary.exists():raise RuntimeError('Never overwrite selected numeric source/binary')
  old=cfg['source'].read_text();prefix=old[:old.index('int main(')];insert=prefix.index('static const Cache &get_cache(')
  keys=len(cfg['layers'])*195+(3 if cfg['dense'] else 0);prefix=prefix[:insert]+BAND_GLOBALS.replace('EXPECTEDKEYS',str(keys))+prefix[insert:]
  start=prefix.index(' std::array<std::vector<int>,16> chip_rows;');end=prefix.index(' std::vector<int> seen',start);prefix=prefix[:start]+LAYOUT+prefix[end:]
  predicate=('mode=="unsharded"&&projection.find("shexp")==std::string::npos' if name=='granite4tiny' else 'mode=="unsharded"&&projection.find("_exps")!=std::string::npos');assert prefix.count(predicate)==1;prefix=prefix.replace(predicate,'mode=="unsharded"')
  assert prefix.count('audit(t,layer,p);return true;')==1;prefix=prefix.replace('audit(t,layer,p);return true;','audit(t,layer,p);verify_ffn_finite(t);return true;')
  needle='if(!p)throw std::runtime_error("No logits");';assert prefix.count(needle)==1;prefix=prefix.replace(needle,needle+'for(int i=0;i<vocab;++i){if(!std::isfinite(p[i]))throw std::runtime_error("Nonfinite logits");++finite_continuation_logits;}')
  source=prefix+MAIN.replace('MODELNAME',name);src.write_text(source)
  spans=[('quantize','static float quantize(','static int64_t dot('),('dot','static int64_t dot(','static float scale_output('),('rescale_and_bitcompare','static float scale_output(','static bool target('),('target','static bool target(','static const Cache &get_cache('),('weight_conversion_prefix','static const Cache &get_cache(',' std::array<std::vector<int>,16> chip_rows;'),('packing_scatter_cache_suffix',' std::vector<int> seen','struct Stats'),('operator','static void audit(','struct Trace'),('run','static void run(', 'static void load_plan(' if name=='granite4tiny' else 'int main(')]
  proofs[name]=[]
  for label,start,end in spans:
   aa=old[old.index(start):old.index(end,old.index(start))].strip();bb=source[source.index(start):source.index(end,source.index(start))].strip()
   if label=='target':bb=bb[:bb.index('struct PhysicalBand')].strip()
   if label=='operator':bb=bb.replace('if(mode=="unsharded"){int64_t s=dot','if('+predicate+'){int64_t s=dot')
   assert aa==bb,(name,label);proofs[name].append(dict(span=label,byte_exact_after_declared_branch_normalization=True,old_body_sha256=hashlib.sha256(aa.encode()).hexdigest(),normalization='Only canonical predicate broadens from routed-only to allFFN' if label=='operator' else None))
  argv=json.loads(cfg['build'].read_text())['argv'];argv[argv.index(str(cfg['source']))]=str(src);argv[argv.index('-o')+1]=str(binary);subprocess.run(argv,check=True);dump(b/f'{name}.build.json',{'argv':argv,'source_sha256':sha(src),'binary_sha256':sha(binary)});inventory[name]=dict(source=str(src),source_sha256=sha(src),binary=str(binary),binary_sha256=sha(binary))
 dump(OUT/'numeric_helper_identity.json',proofs);dump(OUT/'build_inventory.json',dict(driver_source_sha256=sha(SELF),models=inventory,mapping_selfcheck=selfcheck(),scope='Packing/scatter row ownership only, plus allFFN canonical baseline predicate. No quantizer/input format/dot/rescale/host routing/run changes.'))
 print(json.dumps(inventory,indent=2))

def freeze_smoke(a):
 assert a.plan_seal and a.plans_dir,'Root plan seal and directory required before smoke freeze';seal=pathlib.Path(a.plan_seal).resolve();plans=pathlib.Path(a.plans_dir).resolve();assert seal.exists();dest=OUT/'smoke';dest.mkdir(exist_ok=True)
 if (dest/'protocol.frozen.json').exists():raise RuntimeError('Never overwrite frozen selected-plan smoke')
 corpus=NFS/'runs/corpus-300-v1/corpus/prompts';jobs=[corpus/'train_programming_short_015.txt',corpus/'train_summarization_short_027.txt'];assert all(p.exists() for p in jobs)
 inventory=json.loads((OUT/'build_inventory.json').read_text());inputs=[SELF,OUT/'build_inventory.json',OUT/'numeric_helper_identity.json',seal,*jobs];models={}
 for name,cfg in CFG.items():
  d=dest/name;d.mkdir(exist_ok=True);bands=[];ledger=[];plan_hashes={}
  for layer in cfg['layers']:
   p=plans/f'{name}.L{layer}.cmd{a.command_ns}'/f'{a.family}.plan.json';plan=json.loads(p.read_text());assert plan['model']==name and plan['layer']==layer and plan['service']['command_ns']==a.command_ns and plan['selection']['family']==a.family
   bb,ll=lower_layer(plan,cfg);bands.extend(bb);ledger.extend(ll);plan_hashes[str(p)]=sha(p)
  if cfg['dense']:
   for projection,C,R in [('gate',cfg['H'],cfg['dense']),('up',cfg['H'],cfg['dense']),('down',cfg['dense'],cfg['H'])]:
    assert R%16==0
    for chip in range(16):bands.append((0,projection,0,C,R,chip,chip*(R//16),R//16))
  tsv=d/'physical_rows.tsv';tsv.write_text(''.join('\t'.join(map(str,row))+'\n' for row in bands));dump(d/'mapping_ledger.json',ledger);jobfile=d/'jobs.tsv';jobfile.write_text(''.join(f'{p}\t{p.stem}\n' for p in jobs));inputs += [cfg['source'],cfg['build'],cfg['protocol'],cfg['model'],pathlib.Path(inventory['models'][name]['source']),pathlib.Path(inventory['models'][name]['binary']),tsv,jobfile,d/'mapping_ledger.json'];models[name]=dict(model=str(cfg['model']),binary=inventory['models'][name]['binary'],rows_tsv=str(tsv),jobs_tsv=str(jobfile),plans_sha256=plan_hashes,raw_layers=cfg['layers'])
 inputs += [pathlib.Path(x) for x in json.loads(CFG['granite4tiny']['build'].read_text())['argv'] if x.endswith('.a')]
 spec=dict(frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ',time.gmtime()),plan_seal=str(seal),plan_seal_sha256=sha(seal),plans_directory=str(plans),command_ns=a.command_ns,family=a.family,steps=4,threads=8,models=models,inputs_sha256={str(p):sha(p) for p in inputs},prompts=[dict(path=str(p),id=p.stem,sha256=sha(p)) for p in jobs],mapping_contract='Native GU rows [0,I)gate+[I,2I)up; I is routedI or sharedI separately. Shared logical expert64 -> sourceexpert0. Exact whole-row chip ownership; CPU cache packs each projection/chip in reverse logical row order and scatters, not native-address flash programming. DeepSeek raw0dense fixed equal16 per projection.',scope='Existing twoTRAIN short prompts4decode, bothmodels unsharded allFFN canonical vs selectedcompiler packed fullrows, sourceGGUF host attention/router unchanged. No fresh/holdout input. All bitexact checks required. Shared uses selected bands.',output_policy='Preserve full perprompt reference FFN/logit snapshots for exact streaming comparison and row/coverage ledgers; all files in new experiment directory.')
 dump(dest/'protocol.frozen.json',spec);print(json.dumps({'status':'frozen','protocol_sha256':sha(dest/'protocol.frozen.json')},indent=2))

def verify(spec):
 for p,h in spec['inputs_sha256'].items():assert sha(p)==h,p
 for cfg in spec['models'].values():
  for p,h in cfg['plans_sha256'].items():assert sha(p)==h,p

def run_smoke():
 dest=OUT/'smoke';spec=json.loads((dest/'protocol.frozen.json').read_text());verify(spec)
 if (dest/'execution.json').exists() or any((dest/n/'run').exists() for n in spec['models']):raise RuntimeError('Never overwrite selected numeric run')
 execution={}
 for name,cfg in spec['models'].items():
  run=dest/name/'run';run.mkdir();argv=[cfg['binary'],cfg['model'],cfg['jobs_tsv'],str(spec['steps']),str(spec['threads']),cfg['rows_tsv'],str(run)];dump(run/'run_command.json',{'argv':argv,'protocol_sha256':sha(dest/'protocol.frozen.json')});start=time.monotonic()
  with (run/'stdout.log').open('w') as out,(run/'stderr.log').open('w') as err:p=subprocess.run(argv,stdout=out,stderr=err,timeout=480)
  execution[name]=dict(exit_code=p.returncode,wall_seconds=time.monotonic()-start);dump(run/'execution.json',execution[name]);assert p.returncode==0,execution
 dump(dest/'execution.json',execution);verify(spec);print(json.dumps(execution,indent=2))

def main():
 p=argparse.ArgumentParser();p.add_argument('action',choices=['build','mapping-selfcheck','freeze-smoke','run-smoke']);p.add_argument('--plans-dir');p.add_argument('--plan-seal');p.add_argument('--command-ns',type=int,default=250);p.add_argument('--family',default='whole_rows');a=p.parse_args()
 if a.action=='build':build()
 elif a.action=='mapping-selfcheck':print(json.dumps(selfcheck()))
 elif a.action=='freeze-smoke':freeze_smoke(a)
 else:run_smoke()
if __name__=='__main__':main()
