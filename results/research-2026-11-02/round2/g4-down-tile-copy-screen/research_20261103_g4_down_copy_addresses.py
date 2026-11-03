#!/usr/bin/env python3
"""Virtual prefix-address proof preserving baseline GU/shared addresses."""
import collections,gzip,hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2';OUT=BASE/'g4-down-tile-copy-screen';SELF=pathlib.Path(__file__).resolve();PHYSICAL=BASE/'compact-coupled-solver/continuation/native/0107.physical.json.gz'
def sha(p):return hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest()
def dump(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def main():
 pp=OUT/'address_protocol.frozen.json';assert not pp.exists();files=[SELF,PHYSICAL,OUT/'work_arrays.npz',OUT/'copy_rowid_ledger.json'];hashes={str(p):sha(p) for p in files};dump(pp,dict(inputs_sha256=hashes,rule='Retain all GU/shared addresses and all unmodified Down allocations; relocate only original destination Downcore when present into contiguouscore+copy prefix, usingfreewaveholes thennewblock ifneeded. Freedoldcore isnotanotherstoredcopy. Everycopypayloadbytecounted evenifinexistingblockholes.'))
 old=json.loads(gzip.decompress(PHYSICAL.read_bytes()));a=np.load(OUT/'work_arrays.npz');layout=a['selected_layout'];rows=a['rows'];packets=json.loads((OUT/'copy_rowid_ledger.json').read_text());oldkeys={};occupied=[np.zeros(n*192,dtype=np.uint8) for n in old['chip_blocks']];alloc=[];oldalloc={};relocated=0
 for logical,address in zip(old['logical'],old['addresses']):
  e,p,c=logical['expert'],logical['phase'],logical['chip'];key=e,p,c;oldkeys[key]=(logical,address)
  if p==1 and e<64 and c==int(layout[e,1]):relocated+=1;continue
  at=address['block']*192+address['wordline'];n=address['reserved_wordlines'];assert not occupied[c][at:at+n].any();occupied[c][at:at+n]=1
  item=dict(id=len(alloc),expert=e,phase=p,chip=c,block=address['block'],wordline=address['wordline'],waves=n,pages=logical['pages'],columns=logical['columns'],row_ranges=[[logical['row_start'],logical['row_start']+logical['rows']]],role='baseline_unchanged');alloc.append(item);oldalloc[key]=item['id']
 def reserve(chip,n):
  bit=occupied[chip]
  for b in range(len(bit)//192):
   for wl in range(192-n+1):
    at=b*192+wl
    if not bit[at:at+n].any():bit[at:at+n]=1;return b,wl
  occupied[chip]=np.r_[bit,np.zeros(192,dtype=np.uint8)];occupied[chip][len(bit):len(bit)+n]=1;return len(bit)//192,0
 destination_alloc={}
 for e,(s,t) in enumerate(layout):
  s,t=int(s),int(t);l=packets[e];core=l['destination_original_rowids'];packet=l['source_logical_rowids'];rr=[]
  for ids in [core,packet]:
   if ids:assert ids==list(range(ids[0],ids[-1]+1));rr.append([ids[0],ids[-1]+1])
  nrows=len(core)+128;waves=nrows//128;block,wl=reserve(t,waves);item=dict(id=len(alloc),expert=e,phase=1,chip=t,block=block,wordline=wl,waves=waves,pages=waves*4,columns=512,row_ranges=rr,role='original_destination_core_plus_single_copied_source_suffix',original_core_rows=len(core),copied_rows=128);alloc.append(item);destination_alloc[e]=item['id']
 # Independent physical interval/row coverage audits over the final allocation.
 intervals=collections.defaultdict(list);coverage=collections.Counter();useful=pages=waves=0
 for x in alloc:
  n=sum(b-a for a,b in x['row_ranges']);assert x['wordline']+x['waves']<=192;assert x['pages']==(n*x['columns']+16383)//16384;assert x['waves']==(x['pages']+3)//4
  intervals[x['chip'],x['block']].append((x['wordline'],x['wordline']+x['waves']));useful+=n*x['columns'];pages+=x['pages'];waves+=x['waves']
  for lo,hi in x['row_ranges']:
   for r in range(lo,hi):coverage[x['expert'],x['phase'],r]+=1
 for rs in intervals.values():
  rs.sort();assert all(b<=c for (_,b),(c,_) in zip(rs,rs[1:]))
 for e in range(65):
  for p in [0,1]:
   R=rows[e,p].sum();copied=set(packets[e]['source_logical_rowids']) if e<64 and p==1 else set()
   assert all(coverage[e,p,r]==(2 if r in copied else 1) for r in range(R))
 views=[]
 for e in range(64):
  for choice in ['A','B']:
   ds={int(c):rr for c,rr in packets[e][choice+'_chip_logical_rows'].items()};v=[]
   for c,ids in ds.items():
    ai=destination_alloc[e] if c==int(layout[e,1]) else oldalloc[e,1,c];x=alloc[ai];physical=[r for lo,hi in x['row_ranges'] for r in range(lo,hi)];assert physical[:len(ids)]==ids;assert len(ids)%128==0
    v.append(dict(chip=c,allocation=ai,rows=len(ids),pages=len(ids)//32,waves=len(ids)//128,logical_rows=ids,original_logical_scale_rowids=ids))
   assert sorted(r for x in v for r in x['logical_rows'])==list(range(1536));views.append(dict(expert=e,choice=choice,jobs=v))
 assert useful-old['useful_bytes']==4194304 and pages*16384-old['programmed_page_bytes']==4194304 and waves*65536-old['reserved_wave_bytes']==4194304
 # GU and shared allocations have exactly their old physical geometry.
 for (e,p,c),(logical,address) in oldkeys.items():
  if p==0 or e==64:
   x=alloc[oldalloc[e,p,c]];assert (x['chip'],x['block'],x['wordline'],x['waves'],x['pages'])==(c,address['block'],address['wordline'],address['reserved_wordlines'],logical['pages'])
 assert hashes=={p:sha(p) for p in hashes};dump(OUT/'physical_copy_allocations.json',alloc);dump(OUT/'physical_down_A_B_views.json',views)
 report=dict(status='PASS',source_sha256=sha(SELF),protocol_sha256=sha(pp),all_original_GU_shared_addresses_preserved=True,only_original_destination_Down_cores_relocated=relocated,stored_row_coverage_exact=True,physical_ranges_disjoint=True,Down_A_B_wholewave_prefixes=True,Down_no_fringe_or_partialdot=True,original_scale_logical_rows_preserved=True,additional_useful_coefficient_bytes=useful-old['useful_bytes'],additional_programmed_page_bytes=pages*16384-old['programmed_page_bytes'],additional_reserved_wave_bytes=waves*65536-old['reserved_wave_bytes'],baseline_blocks=old['chip_blocks'],new_blocks=[len(bit)//192 for bit in occupied],baseline_allocated_LSB_bytes=old['allocated_lsb_block_bytes'],new_allocated_LSB_bytes=sum(len(bit)//192 for bit in occupied)*192*65536,unused_holes_do_not_reduce_copy_budget=True,limits='Virtual address/rowID proof only, not programmedweights or native numeric ABI. GU/shared addresses retained; destinationcores can move atstaticimageconstruction andoldslot isfree.');dump(OUT/'physical_copy_audit.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
