import collections,gzip,hashlib,itertools,json,math,pathlib
import numpy as np
ROOT=pathlib.Path('/home/ericyu/expert_flash_experiment')
D=ROOT/'results/research-2026-11-02/round2/existing-hardware'
S=ROOT/'results/research-2026-11-02/projection-study'
PAGE=16384; L=192; P=4; W=PAGE*P

def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
def read(p):
 with gzip.open(p,'rt') as f:return json.load(f)
packing=json.load(open(D/'packing.json'))
model=next(m for m in packing['models'] if m['model']=='mixtral')
paths=[D/'packing.json',D/'notes.json',ROOT/'scripts/research_20261102_existing_hardware.py',S/'mixtral.physical_plan.json.gz',S/'mixtral.frozen_plan.json']
paths += [D/('mixtral.'+n+'.physical_plan.json.gz') for n in ('global_expert_first','per_chip_ffd','row_safe_wave_fill')]
before={str(p):digest(p) for p in paths}
for p,h in packing['source_sha256'].items(): assert digest(pathlib.Path(p))==h
original=read(S/'mixtral.physical_plan.json.gz')
frozen=json.load(open(S/'mixtral.frozen_plan.json'))
dimensions={(p['layer'],p['name']):p['total_rows'] for p in frozen['phases']}

# Own audit: no imports/calls from author implementation or certificate.
def audit(value,check_native=True):
 records=value['streams'];assert len(records)==8192
 maxblocks=[0]*16;counts=[collections.Counter() for _ in range(16)]
 occupied=collections.defaultdict(list);rows=collections.defaultdict(list)
 nr={}
 if check_native:
  for si,sj,start,number in value['native_run_rows']:
   assert (si,sj) not in nr;nr[si,sj]=(start,number)
 for i,(old,r) in enumerate(zip(original['streams'],records)):
  assert r[:9]==old[:9]
  layer,expert,phase,chip,start,number,cols,pages,waves,commands,segments=r
  assert cols==({'gate_up':4096,'down':14336}[phase])
  assert number*cols==pages*PAGE
  assert pages==4*waves
  assert commands==len(segments)
  if phase=='down':assert commands==1
  rows[layer,expert,phase].append((start,start+number))
  page_cursor=0;wave_sum=0;output_row_cursor=start
  for j,(block,wl,nwaves,page_offset,npages,plane_counts) in enumerate(segments):
   assert page_offset==page_cursor
   assert npages==4*nwaves and plane_counts==[nwaves]*4
   assert 0<=wl<192 and wl+nwaves<=192
   assert page_offset*PAGE%cols==0 and npages*PAGE%cols==0
   row_start=start+page_offset*PAGE//cols;row_count=npages*PAGE//cols
   assert row_start==output_row_cursor
   if check_native:assert nr.pop((i,j))==(row_start,row_count)
   output_row_cursor+=row_count
   for plane in range(4):occupied[chip,plane].append((block*L+wl,block*L+wl+nwaves))
   page_cursor+=npages;wave_sum+=nwaves
   maxblocks[chip]=max(maxblocks[chip],block+1)
  assert (page_cursor,wave_sum,output_row_cursor)==(pages,waves,start+number)
  counts[chip].update(streams=1,pages=pages,waves=waves,commands=commands,rows=number)
 assert not nr
 for (layer,expert,phase),pieces in rows.items():
  end=0
  for start,stop in sorted(pieces):assert start==end;end=stop
  assert end==dimensions[layer,phase]
 assert len(rows)==32*8*2
 overlaps=0
 for key,ranges in occupied.items():
  end=-1
  for start,stop in sorted(ranges):
   assert start>=end,(key,start,end);end=stop
 assert all(c['waves']==43008 and c['pages']==172032 and c['streams']==512 for c in counts)
 assert len({c['commands'] for c in counts})==1
 return dict(blocks_per_chip=maxblocks,commands_per_chip=[c['commands'] for c in counts],
   total_native_commands=sum(c['commands'] for c in counts),stored_pages=sum(c['pages'] for c in counts),
   read_waves=sum(c['waves'] for c in counts),reserved_lsb_bytes=sum(maxblocks)*L*W,
   reserved_lsb_gib=sum(maxblocks)*L*W/2**30,whole_rows_exactly_once=True,
   native_run_rows_matches_every_segment=check_native,complete_columns_every_run=True,
   address_overlap=False,down_never_split=True)

checks={'original':audit(original,False)}
for name in ('global_expert_first','per_chip_ffd','row_safe_wave_fill'):
 value=read(D/('mixtral.'+name+'.physical_plan.json.gz'));checks[name]=audit(value)
 total=model['policies'][name]['audit']['total']
 for own,reported in [('total_native_commands','commands'),('stored_pages','pages'),('read_waves','waves'),('reserved_lsb_bytes','reserved_lsb_bytes')]:
  assert checks[name][own]==total[reported]

# Independently reconstruct the old no-extra-crossing allocator rule.
cursors=[0]*16;padding=[0]*16
for r in original['streams']:
 chip=r[3];waves=r[8];b,o=divmod(cursors[chip],192)
 if o+waves>192:
  padding[chip]+=192-o;cursors[chip]+=192-o;b,o=divmod(cursors[chip],192)
 assert r[10][0][0:3]==[b,o,waves]
 cursors[chip]+=waves
assert [(c+191)//192 for c in cursors]==[321]*16
assert len({p for p in padding})==1

# Independent, layer-local construction: 8GU*112 + 8D*56 = 7*192.
# Six GU whole; two GU split among holes. Never splits D or a dot product.
index={(r[0],r[1],r[2],r[3]):i for i,r in enumerate(original['streams'])}
new=[list(r) for r in original['streams']]
native=[]
for layer in range(32):
 for chip in range(16):
  # Each tuple is (relative block, offset, waves), in logical row order.
  down={0:[(0,0,56)],1:[(0,56,56)]}
  down.update({e:[(e-1,0,56)] for e in range(2,8)})
  gu={e:[(e+1,56,112)] for e in range(6)}
  gu[6]=[(0,112,80),(1,168,24),(2,168,8)]
  gu[7]=[(2,176,16),(3,168,24),(4,168,24),(5,168,24),(6,168,24)]
  for phase,table in [('gate_up',gu),('down',down)]:
   for expert,parts in table.items():
    i=index[layer,expert,phase,chip];old=original['streams'][i];segs=[];offset=0
    for j,(block,wl,waves) in enumerate(parts):
     pages=4*waves;segs.append([7*layer+block,wl,waves,offset,pages,[waves]*4])
     native.append([i,j,old[4]+offset*PAGE//old[6],pages*PAGE//old[6]])
     offset+=pages
    new[i]=old[:9]+[len(segs),segs]
value={**original,'streams':new,'native_run_rows':native}
checks['independent_layer_template']=audit(value)
assert checks['independent_layer_template']['blocks_per_chip']==[224]*16
assert checks['independent_layer_template']['commands_per_chip']==[704]*16

# A bipartite-component proof: GU112 demands and block residual capacities
# 192-56*d imply 14*k = 24*b-7*d. Every balanced connected component must
# contain b divisible by 7. With B=224 there are at most32 components, so
# GU run edges >=256+224-32=448; plus256 whole down commands =>704.
command_lower_bound=256+224-32+256
assert command_lower_bound==704

# Independently enumerate all top-2 pairs for author and own construction.
def overhead(records):
 extra=collections.defaultdict(lambda:collections.defaultdict(lambda:[0]*16))
 inventory=0
 for before,after in zip(original['streams'],records):
  layer,e,phase,chip=before[:4];delta=after[9]-before[9];inventory+=delta
  assert delta>=0
  if phase=='down':assert delta==0
  else:extra[layer][e][chip]=delta
 minimum=maximum=0;mean=0.0;cases=0
 for layer,perexpert in extra.items():
  costs=[]
  for selected in itertools.combinations(range(8),2):
   d=[sum(perexpert[e][c] for e in selected) for c in range(16)]
   ns=sum(d)*250+max((d[c]+d[c+8])*4096//2 for c in range(8))
   costs.append(ns);cases+=1
  minimum+=min(costs);maximum+=max(costs);mean+=sum(costs)/28
 return dict(layer_top2_cases=cases,extra_inventory_commands=inventory,
  extra_inventory_onfi_input_bytes=inventory*4096,
  minimum_extra_ns_per_token=minimum,maximum_extra_ns_per_token=maximum,
  synthetic_uniform_top2_mean_extra_ns_per_token=mean,
  assumptions='250ns/native command; full4096B input reload at2B/ns per shared channel; no extra PCIe; not calibrated hardware')
claimed=model['policies']['row_safe_wave_fill']['route_free_overhead_bounds']
authorover=overhead(read(D/'mixtral.row_safe_wave_fill.physical_plan.json.gz')['streams'])
assert authorover['minimum_extra_ns_per_token']==claimed['extra_control_plus_conservative_input_ns_per_token_minimum']
assert authorover['maximum_extra_ns_per_token']==claimed['extra_control_plus_conservative_input_ns_per_token_maximum']
assert authorover['synthetic_uniform_top2_mean_extra_ns_per_token']==claimed['uniform_top2_mean_extra_ns_per_token']
assert authorover['extra_inventory_commands']==3168 and authorover['extra_inventory_onfi_input_bytes']==12976128
ownover=overhead(new)

# Capacity still224 if the original gate/up matrices are kept separate.
# Cut a fused run wherever its logical stream offset crosses56waves (896rows).
separate_commands=[0]*16
for r in new:
 chip=r[3]
 if r[2]=='down':separate_commands[chip]+=1;continue
 for seg in r[10]:
  a=seg[3]//4;b=a+seg[2]
  separate_commands[chip]+=1+int(a<56<b)
assert separate_commands==[960]*16
assert {str(p):digest(p) for p in paths}==before
report=dict(checked_date='2026-11-02',status='independent_address_capacity_command_audit_passed_no_novelty_established',
 methodology='Own parser, range coverage, native-row identity, occupancy and896 top2 recomputations; did not import author audit/allocator functions or execute author script. No routes, weights or VM accessed; no author file modified.',
 source_sha256=before, auditor_script_sha256=digest(pathlib.Path(__file__)), checks=checks,
 original_allocator_reconstruction=dict(blocks_per_chip=321,boundary_padding_waves_per_chip=padding[0],
  rule='A short stream that would cross an extra block is advanced to the next block; this is our minimum-command/cold-run policy, not an AiF paper requirement.'),
 capacity_bounds=dict(waves_per_chip=43008,volume_lower_bound_blocks_per_chip=224,
  intact_GU112_count_per_chip=256,intact_down56_count_per_chip=256,
  intact_stream_lower_bound_blocks_per_chip=256,
  proof='Every intact112-wave GU needs a separate192-wave block; pairingGU112+down56 attains256. Allowing row-safe GU splits attains the43008/192=224 volume bound.',
  independent_224_template_native_command_lower_bound_per_chip=704,
  command_bound_conditions='Same four-plane-wave-aligned layout family and unchanged NAND waves;224 completely filled block stripes;256 complete down streams stay unsplit;GU segments are disjoint complete rows and may occupy arbitrary block holes. A connected GU/block component balances112*k=192*b-56*d, forcing b divisible by7. Thus at most32 components; GU edges>=256+224-32=448, plus256 down commands.',
  separate_gate_up_template=dict(blocks_per_chip=224,commands_per_chip=960,
   interpretation='Same physical addresses, additionally end native operations at the separate gate/up boundary; no fusion hardware is required for the capacity bound. Not claimed to minimize separate-projection commands.')),
 command_cost_audit=dict(author710=authorover,independent704=ownover,
  verdict='704 minimizes inventory commands under stated down-intact restriction, but concentrates splits on experts6/7 each layer and has worse worst-case routed cost. It does not latency-dominate author710; neither cost is hardware calibrated.'),
 primary_evidence=[
  dict(title='AiF, ISCA2025',url='https://doi.org/10.1145/3695053.3731073',local='paper/aif.txt',
   read_depth='Original supplied paper §4.1/§4.4/§5.2 and printedp538 footnotes8-9 checked.',
   evidence='aif_post evenly shards and sequentially places submatrices in IFP blocks; controller locates their addresses. Footnote9 explicitly dispatches multiple GEMV operations when a submatrix spans IFP blocks. The paper does not disclose this repo no-extra-crossing policy or a321block baseline.',
   limitation='Does not establish exact short-matrix cohabitation or controller run-list policy. Do not assert published implementation actually used224 blocks.'),
  dict(title='Cambricon-LLM, MICRO2024 preprint',url='https://arxiv.org/html/2409.15654v1',
   read_depth='Original §V-A rechecked.',evidence='Hardware-aware matrix tiling with page-sized atomic tiles and equal compute-core mapping is prior.',
   limitation='Different chiplet architecture; not proof of AiF free-block interface or this exact capacity count.'),
  dict(title='FlashAccel, arXiv2026 v2',url='https://arxiv.org/html/2607.10186v2',
   read_depth='Original §V-A rechecked.',evidence='Sequentially packs operator weight chunks across consecutive hyperpages and continues subsequent operators at the next available location.',
   limitation='Custom HBF geometry, not an AiF implementation; supports lack of novelty in generic cross-operator dense packing only.')],
 implementation_corrections_that_can_be_claimed=[
  'This repository minimum-command allocator over-reserves Mixtral IFP LSB stripes:321/chip versus256 with generic whole-stream order and224 with existing native row-GEMV segmentation.',
  'The serialized224plan has exact one-copy row coverage and no physical overlap; no output partial sum, precision change, new decoder, new PE operation or retained accumulator is needed.',
  'A controller software allocator/run map can issue existing block/offset/page-count GEMVs and concatenate complete rows. This is a design certificate, not implemented numerical hardware.',
  'Use a strong capacity-versus-native-command Pareto baseline:256blocks with unchanged commands;224 with paid row-safe segmentation. Include existing equalg16 ownership, not only the learned layout.'],
 claims_not_supported=[
  '30.218% capacity gain over AiF/SOTA.321blocks is our own restrictive policy, so this comparison is only a local implementation correction.',
  'A new MoE/coactivation-aware physical placement algorithm. The construction uses dimensions only and ordinary dense packing.',
  'A normal-read NAND latency gain. Pages and read waves are unchanged; split runs add control/input work.',
  'aif_post already exposes arbitrary physical addresses or the current metadata-only timing kernel implements the new physical mapping.',
  'The author710command plan is universally command-minimal or latency-optimal.',
  'Publication novelty from row-safe block splitting, first-fit-decreasing, a wave-volume capacity bound, or simple output-row concatenation.'],
 baseline_adjustments_required=[
  'Retire321blocks as a strong ordinary-read capacity baseline. Keep it only as explicitly labeled legacy/minimum-command allocator.',
  'For fixed-command comparisons use256blocks/chip, attainable by both expert-first order and FFD.',
  'For capacity comparisons allow224blocks/chip with existing native block split and all incurred metadata/control/input costs. Do not preclude block crossings solely to preserve CR runs in a noCR study.',
  'Recompute any capacity-budget/redudancy or reliability result whose budget was based on321 versus224/256. Which duty model is used matters: compacting holes can increase disturbance stress, as shown in the separate oracle.',
  'Preserve distinction between read-work lower bound and reserved capacity. Empty suffix slots were never read, so reclaiming them cannot speed up dense NAND service.',
  'Retain source-paper/controller uncertainty.224 is an existing-operation constructive baseline, not proof of the original authors undisclosed allocator.'],
 capacity_comparison=dict(original_to_224_reservation_saving_percent=100*(321-224)/321, generic_whole_stream256_to224_saving_percent=100*(256-224)/256, useful_weight_bytes_unchanged=True),
 model_label_caveat='packing.json labels6.4 bytes/ns as ecc_mac_ceiling; AiF publishes ECCLITE throughput/internal bandwidth but does not disclose MAC lane throughput. Ordinary28us read waves dominate this timing ceiling here; capacity/row proofs do not use a MAC assumption.',
 unverified_deployment_requirements=[
  'Controller run-list lookup, allocation and output concatenation implementation and cost.',
  'Zero additional PCIe bytes assumes the controller-held input can be reused within the same host operation; only PB residency is avoided by the conservative ONFI reload charge. This transport path is not measured.',
  'Device-specific legal programming order and GC behavior for the mapped LSB holes; actual model installation not performed.',
  'Concrete input/output transport overhead and PE result format; no rawI32 extension assumed.',
  'ECC parity, scale metadata, other tensors and runtime spare capacity are outside the42GiB routed-expert I8 reservation certificate.'])
(D/'independent_audit.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n')
print(json.dumps({'checks':{k:{x:v[x] for x in ['reserved_lsb_gib','total_native_commands','blocks_per_chip']} for k,v in checks.items()},'author_overhead':authorover,'independent_overhead':ownover},indent=2))
