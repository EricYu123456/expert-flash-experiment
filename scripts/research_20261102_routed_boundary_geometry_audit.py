#!/usr/bin/env python3
"""Independently reconstruct fixed A/B ownership directly from frozen inputs."""
import collections,hashlib,json,pathlib
import numpy as np
ROOT=pathlib.Path(__file__).resolve().parents[1];BASE=ROOT/'results/research-2026-11-02/round2';OUT=BASE/'routed-boundary-view-numeric'
def sha(p):
 h=hashlib.sha256()
 with open(p,'rb') as f:
  for b in iter(lambda:f.read(4<<20),b''):h.update(b)
 return h.hexdigest()
def main():
 m=json.loads((OUT/'manifest.json').read_text());s=json.loads((OUT/'summary.json').read_text());audit=json.loads((OUT/'independent_replay.json').read_text());assert audit['status']=='PASS'
 for p,h in m['source_provenance_sha256'].items():assert sha(p)==h,p
 planpath=next(pathlib.Path(p) for p in m['source_provenance_sha256'] if p.endswith('whole_rows.plan.json'));plan=json.loads(planpath.read_text())
 route=s['TRAIN_route']
 with np.load(route['archive']) as r:
  assert str(r['prompt_ids'][0])==route['prompt'] and r['split'].item()=='train';assert r['routes'][0,0,0].tolist()==s['experts']
 decisions={x['matrix']:x for x in m['decisions']};useful_rows=jobs=0
 for key,variants in m['views'].items():
  shared=key.startswith('shared.');expert=64 if shared else int(key.split('.')[1]);phase=0 if key.endswith('.gu') else 1
  original={b['chip']:list(range(b['row_start'],b['row_start']+b['rows'])) for b in plan['logical'] if b['expert']==expert and b['phase']==phase}
  for label,vs in variants.items():
   expected={c:rows.copy() for c,rows in original.items()}
   if key in decisions and label=='B':
    d=decisions[key];source,dest=d['source'],d['destination'];assert source^8==dest
    delta=len(original[source])-len(original[dest]);assert delta==d['delta_rows'] and delta in [42,43]
    packet=original[source][-delta:];assert packet==d['boundary_rowids'];expected[source]=original[source][:-delta];expected[dest]+=packet
   actual={v['chip']:v['rowids'] for v in vs};assert actual==expected,(key,label)
   useful_rows+=sum(map(len,actual.values()));jobs+=len(vs)
 # Independently reconstruct minimal useful physical copy inventory and aliases.
 aliases=collections.defaultdict(list)
 for a in m['allocations']:
  for slot,row in enumerate(a['rowids']):aliases[a['matrix'],row].append(dict(chip=a['chip'],allocation=a['id'],byte=a['byte_offset']+slot*a['columns']))
 duplicate_rows=[]
 for (key,row),locations in aliases.items():
  expected=2 if key in decisions and row in decisions[key]['boundary_rowids'] else 1;assert len(locations)==expected
  if expected==2:
   d=decisions[key];assert {v['chip'] for v in locations}=={d['source'],d['destination']};duplicate_rows.append(dict(matrix=key,row=row,aliases=locations))
 inv=m['inventory'];assert len(duplicate_rows)==inv['duplicated_boundary_rows'];sizes={}
 for label in ['A','B']:
  vs=[v for options in m['views'].values() for v in options[label]];sizes[label]=dict(native_jobs=len(vs),useful_rows=sum(v['rows'] for v in vs),requested_pages=sum((v['rows']*v['columns']+16383)//16384 for v in vs),requested_waves=sum((v['rows']*v['columns']+65535)//65536 for v in vs))
 assert sizes['A']==sizes['B'];reserved=sum(max(a['reserved_waves'] for a in [m['allocations'][v['allocation']]]) for v in [vv for options in m['views'].values() for vv in options['A']]);assert reserved==inv['max_view_reserved_waves']
 # Metadata byte sizes are reported separately; not claimed to be NAND payload.
 metadata={name:(OUT/name).stat().st_size for name in ['manifest.json','source_rows.json','logical_scales.f32','logical_scales.json','drain_drop_ledger.json']}
 report=dict(status='PASS',source_sha256=sha(__file__),all_original_source_hashes_unchanged=True,frozen_A_original_logical_rows_exact=True,B_only_moves_source_suffix=True,original_Down_shared_geometry_exact=True,TRAIN_route_exact=True,duplicate_logical_rows=len(duplicate_rows),useful_rows_across_two_views=useful_rows,native_jobs_across_two_views=jobs,view_costs=sizes,physical_inventory=inv,host_metadata_file_bytes=metadata,metadata_storage_and_transfer_cost_not_modelled=True,duplicate_aliases=duplicate_rows,limitations='Logical metadata retained on host; no native output or fringe/reset ABI validation, no controller timing, no end-to-end inference or quality claim.')
 p=OUT/'geometry_source_audit.json';assert not p.exists();p.write_text(json.dumps(report,indent=2)+'\n');print(json.dumps({k:v for k,v in report.items() if k!='duplicate_aliases'},indent=2))
if __name__=='__main__':main()
