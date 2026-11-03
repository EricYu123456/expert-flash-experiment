#!/usr/bin/env python3
"""Small row-ID/wordline enumeration; no weights, routes, fitting or hardware."""
import collections,hashlib,itertools,json,math,pathlib,time
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2';OUT=BASE/'ordinary-gu-tile-sharing/contract-audit'
C=2048;ROWS=176;PAGE=16384;WAVE=65536;RPW=32;WL=192;RPB=6144
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def split_blocks(start,end):
 assert start%RPW==end%RPW==0
 result=[]
 while start<end:
  block=start//RPB;limit=min(end,(block+1)*RPB);waves=(limit-start)//RPW
  result.append(dict(block=block,start_wordline=(start%RPB)//RPW,waves=waves,pages=4*waves,first_row=start,end_row=limit,emitted_complete_rows=limit-start,input_bytes=2048,nominal_F32_output_bytes=(limit-start)*4))
  start=limit
 return result
def singleton(e):
 s=e*ROWS;t=s+ROWS;a=s//RPW*RPW;b=(t+RPW-1)//RPW*RPW
 return dict(expert=e,useful_first_row=s,useful_end_row=t,view_first_row=a,view_end_row=b,prefix_rows=s-a,suffix_rows=b-t,commands=split_blocks(a,b))
def union_views(selected):
 intervals=sorted((singleton(e)['view_first_row'],singleton(e)['view_end_row']) for e in selected);merged=[]
 for a,b in intervals:
  if merged and a<=merged[-1][1]:merged[-1][1]=max(merged[-1][1],b)
  else:merged.append([a,b])
 return merged,[c for a,b in merged for c in split_blocks(a,b)]
def audit(selected,E):
 selected=sorted(selected);merged,commands=union_views(selected)
 # Independent bitmap enumeration uses actual row-ID storage, not aliases.
 storage=[(e,r) for e in range(E) for r in range(ROWS)];assert len(storage)==len(set(storage))
 wlset=set()
 for e in selected:
  for row in range(e*ROWS,(e+1)*ROWS):wlset.add(row//RPW)
 emitted=[row for w in sorted(wlset) for row in range(w*RPW,(w+1)*RPW)];assert len(emitted)==len(set(emitted));assert max(emitted)<len(storage)
 commandrows=[row for c in commands for row in range(c['first_row'],c['end_row'])];assert commandrows==emitted
 expected={(e,r) for e in selected for r in range(ROWS)};kept=[storage[row] for row in emitted if storage[row][0] in selected];dropped=[storage[row] for row in emitted if storage[row][0] not in selected]
 assert set(kept)==expected and len(kept)==len(expected);assert len(emitted)==len(kept)+len(dropped)
 for cmd in commands:
  assert 0<=cmd['start_wordline']<WL and cmd['start_wordline']+cmd['waves']<=WL and cmd['pages']==cmd['waves']*4
  assert cmd['pages']*PAGE==cmd['emitted_complete_rows']*C
 single=[singleton(e) for e in selected];sw=sum(c['waves'] for v in single for c in v['commands']);sc=sum(len(v['commands']) for v in single);uw=sum(c['waves'] for c in commands)
 assert sw==6*len(selected) and uw==len(wlset) and uw<=sw
 return dict(selected=selected,union_intervals=merged,union_commands=commands,selected_useful_rows=len(expected),union_emitted_rows=len(emitted),dropped_nonselected_rows=len(dropped),partial_row_bytes=0,singleton_waves=sw,union_waves=uw,saved_waves=sw-uw,singleton_commands=sc,union_command_count=len(commands),physical_stored_row_copies=1)
def main():
 began=time.monotonic();OUT.mkdir(parents=True,exist_ok=False);assert PAGE%C==WAVE%C==0 and WAVE//C==32 and ROWS*C==360448 and ROWS*C//PAGE==22
 sources=[ROOT/'kernel/aif.h',ROOT/'kernel/aif.c',ROOT/'paper/aif.txt',ROOT/'docs/plane_wave_architecture_review.md',ROOT/'scripts/research_20261103_gu_coalescing.py',ROOT/'scripts/research_20261103_gu_coalescing_audit.py',BASE/'gu-coalescing/protocol.frozen.json',BASE/'gu-coalescing/prior_and_legality_audit.json',BASE/'gu-coalescing/independent_audit.json'];hashes={str(p):sha(p) for p in sources}
 allcases=[]
 for mask in range(1,256):allcases.append(audit([e for e in range(8) if mask>>e&1],8))
 for e in range(64):allcases.append(audit([e],64))
 for e in range(63):
  rec=audit([e,e+1],64);assert rec['union_waves']==(11 if e%2==0 else 12);allcases.append(rec)
 for e in range(59):
  rec=audit(list(range(e,e+6)),64);assert rec['union_waves']==(33 if e%2==0 else 34);allcases.append(rec)
 # Every short contiguous run follows the parity-aware exact formula.
 formula=0
 for n in range(1,9):
  for e in range(64-n+1):
   intervals,commands=union_views(range(e,e+n));assert len(intervals)==1;w=sum(c['waves'] for c in commands);assert w==(11*n+(e%2)+1)//2;formula+=1
 inventory=[dict(expert=e,physical_first_row=e*ROWS,physical_end_row=(e+1)*ROWS,physical_byte_start=e*ROWS*C,useful_weight_bytes=ROWS*C,stored_pages=22,first_page=e*22,singleton_view=singleton(e)) for e in range(64)]
 assert all(a['physical_end_row']==b['physical_first_row'] for a,b in zip(inventory,inventory[1:]));assert sum(v['useful_weight_bytes'] for v in inventory)==64*176*2048
 witnesses={name:audit(sel,64) for name,sel in dict(even_pair=[0,1],odd_pair=[1,2],separated=[0,2],block_cross_singleton=[34],block_cross_pair=[34,35],six_contiguous_even=list(range(6)),six_contiguous_odd=list(range(1,7)),six_cross_block=list(range(32,38))).items()}
 # Audit duplicated reads separately from stored copies and useful row use.
 sample=[0,1];counts=collections.Counter(row for e in sample for cmd in singleton(e)['commands'] for row in range(cmd['first_row'],cmd['end_row']));overlap=[r for r,n in counts.items() if n>1];assert len(overlap)==32 and overlap==list(range(160,192))
 ledger=dict(columns=C,page_bytes=PAGE,wave_bytes=WAVE,rows_per_page=8,rows_per_wave=32,rows_per_block=RPB,wordlines_per_block=192,per_expert_chip_GU_rows=176,per_expert_full_GU_rows=2816,per_expert_gate_rows=1408,per_expert_up_rows=1408,per_expert_chip_weight_bytes=360448,per_expert_chip_stored_pages=22,per_expert_chip_singleton_read_pages=24,per_expert_chip_singleton_read_waves=6,per_expert_chip_singleton_fringe_rows=16,singlecopy_expert_count=64,stored_rows_per_chip=11264,stored_bytes_per_chip=23068672,stored_pages_per_chip=1408,stored_wave_span_per_chip=352,reserved_LSB_blocks_per_chip=2,reserved_LSB_block_bytes_per_chip=25165824,max_geometric_command_rows=6144,max_geometric_command_pages=768,max_geometric_nominal_F32_output_bytes=24576,uniform176_GU_singleton_waves_per_expert=96,strong192_160_GU_waves_per_expert=88,selected_pair_singleton_duplicate_read_rows=32,selected_pair_stored_duplicate_rows=0,native_R_FIFO_pagecount_bitwidth_limits_verified=False)
 assert hashes=={str(p):sha(p) for p in sources};dump(OUT/'rowid_inventory.json',inventory);dump(OUT/'enumeration.json',allcases);dump(OUT/'witnesses.json',witnesses);dump(OUT/'geometry_ledger.json',ledger)
 report=dict(status='PASS',source_sha256=sha(__file__),input_sha256=hashes,enumerated_cases=len(allcases),contiguous_formula_checks=formula,all_selected_useful_rows_present_once=True,all_real_nonselected_fringe_rows_read_and_dropped=True,no_partial_rows_at_any_page_wave_block_boundary=True,onecopy_nonoverlapping_stored_rows=True,no_fake_physical_aliases=True,overlapping_readviews_are_repeat_reads=True,all_crossblock_commands_split=True,weights_or_routes_opened=False,model_inference_or_fitting=False,heldout_opened=False,geometry_ledger=ledger,limits=['Combinatorial row-ID/address proof, not native PE execution or weights numerical test.','Native scalar format/scaling, FIFO depth/backpressure, maxR/maxpagecount encodings and variable-range controller dispatch remain unverified.','No early expert-ready event inside a merged command; conservative whole-command drain then full-expert GU barrier.','Deep uniform176 has96 singleton GU waves/expert vs existing192/160 geometry88; sharing must be compared with strong geometry, not only its own96-wave singleton.'],seconds=time.monotonic()-began)
 dump(OUT/'report.json',report);(OUT/'audit_source.py').write_bytes(pathlib.Path(__file__).read_bytes());dump(OUT/'completion.json',dict(status='PASS',artifact_sha256={str(p):sha(p) for p in OUT.rglob('*') if p.is_file()}));print(json.dumps({k:report[k] for k in ['status','enumerated_cases','contiguous_formula_checks','weights_or_routes_opened','seconds']},indent=2))
if __name__=='__main__':main()
