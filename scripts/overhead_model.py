#!/usr/bin/env python3
"""AiF grouped expert layout including chip control and vector traffic.

Control latency is an exposed assumption, not a measured physical AiF cost.
Read phases, command dispatch, input distribution and output return are ordered
barriers. PCIe and ONFI transfer limits overlap within each distribution phase.
"""
import argparse
import csv
import json
from pathlib import Path
import struct
import aif_model as flash


def transfer_ns(chip_bytes, pcie_bytes=None):
    channels=[0]*8
    for chip,value in enumerate(chip_bytes):
        channels[chip % 8] += value
    host_bytes=sum(chip_bytes) if pcie_bytes is None else pcie_bytes
    return max(flash.ceildiv(host_bytes,8),max(flash.ceildiv(v,2) for v in channels))


def phase_traffic(experts, projections, group_chips, common_input, input_width=1, output_width=4):
    inputs,outputs,streams=[0]*16,[0]*16,[0]*16
    for expert in experts:
        first=(expert % (16//group_chips))*group_chips
        for rows,columns in projections:
            for shard in range(group_chips):
                count=rows//group_chips+(shard<rows%group_chips)
                if not count:
                    continue
                chip=first+shard
                if common_input:
                    inputs[chip]=max(inputs[chip],columns*input_width)
                else:
                    inputs[chip]+=columns*input_width
                outputs[chip]+=count*output_width
                rounds=flash.ceildiv(flash.ceildiv(count*columns,flash.PAGE),flash.PLANES)
                streams[chip]+=flash.ceildiv(rounds,192)
    return inputs,outputs,streams


def encode_v2(demands, inputs, outputs, streams, control_ns, host_ns,
              pcie_input_bytes=None,pcie_output_bytes=None):
    # V3 uses two formerly reserved header words for host-to-controller bytes.
    version=2 if pcie_input_bytes is None else 3
    header=struct.pack('<IIIIIIII',0x31464941,version,16,control_ns,host_ns,
                       pcie_input_bytes or 0,pcie_output_bytes or 0,0)
    return header+b''.join(struct.pack('<QQIIQQII',d.useful,d.pages,d.rounds,d.cold,
                         i,o,s,0) for d,i,o,s in zip(demands,inputs,outputs,streams))


def evaluate(events, dimensions, group_chips, *, stack_gate_up=False, control_ns=250,
             host_ns=2000, input_width=1, output_width=4, reuse_input=True,
             controller_fanout=True,
             mappings=None, selected_groups=None, binary_path=None):
    keys=('commands','model_ns','flash_ns','control_ns','host_ns','input_ns','output_ns',
          'useful_bytes','physical_pages','read_rounds','cold_reads','active_chip_commands',
          'chip_service_ns','vector_input_bytes','vector_output_bytes','chip_matrix_commands',
          'pcie_input_bytes','pcie_output_bytes')
    total={k:0 for k in keys}
    by_chip=[dict(useful=0,service=0) for _ in range(16)]
    tokens={}
    output=binary_path.open('wb') if binary_path else None
    try:
        for event in events:
            layer=event['layer']
            g=selected_groups[layer] if selected_groups is not None else group_chips
            experts=event['expert_ids']
            if mappings is not None:
                experts=[mappings[layer][e] for e in experts]
            dims=dimensions[layer]
            phases=[([dims['gate']],True),([dims['up']],True),([dims['down']],False)]
            if stack_gate_up:
                phases=[([(dims['gate'][0]+dims['up'][0],dims['up'][1])],True),([dims['down']],False)]
            token=event['prompt'],event['token_index']
            tk=tokens.setdefault(token,dict(flash=0,input=0,output=0,streams=0,commands=0))
            for projections,common in phases:
                demands=flash.phase_demand(experts,projections,g)
                inputs,outputs,streams=phase_traffic(experts,projections,g,common and reuse_input,
                                                    input_width,output_width)
                service=max(d.service_ns for d in demands)
                # Host supplies each logical operand once; controller fans it out.
                host_input=(projections[0][1]*input_width*(1 if common else len(experts))) if controller_fanout else sum(inputs)
                host_output=sum(outputs)
                tin,tout=transfer_ns(inputs,host_input),transfer_ns(outputs,host_output)
                nstreams=sum(streams)
                duration=service+tin+tout+control_ns*nstreams+host_ns
                tk['flash']+=service; tk['input']+=tin; tk['output']+=tout
                tk['streams']+=nstreams; tk['commands']+=1
                total['commands']+=1; total['model_ns']+=duration; total['flash_ns']+=service
                total['input_ns']+=tin; total['output_ns']+=tout
                total['control_ns']+=control_ns*nstreams; total['host_ns']+=host_ns
                total['vector_input_bytes']+=sum(inputs); total['vector_output_bytes']+=sum(outputs)
                total['pcie_input_bytes']+=host_input; total['pcie_output_bytes']+=host_output
                total['chip_matrix_commands']+=nstreams
                for chip,d in enumerate(demands):
                    total['useful_bytes']+=d.useful; total['physical_pages']+=d.pages
                    total['read_rounds']+=d.rounds; total['cold_reads']+=d.cold
                    total['active_chip_commands']+=d.useful>0
                    total['chip_service_ns']+=d.service_ns
                    by_chip[chip]['useful']+=d.useful; by_chip[chip]['service']+=d.service_ns
                if output:
                    output.write(encode_v2(demands,inputs,outputs,streams,control_ns,host_ns,
                                          host_input if controller_fanout else None,
                                          host_output if controller_fanout else None))
    finally:
        if output:
            output.close()
    latency=sorted(v['flash']+v['input']+v['output']+control_ns*v['streams']+host_ns*v['commands'] for v in tokens.values())
    total.update(decode_tokens=len(tokens),routing_records=len(events),group_chips=group_chips,
                 stack_gate_up=stack_gate_up,per_chip_control_ns=control_ns,per_host_command_ns=host_ns,
                 input_bytes_per_element=input_width,output_bytes_per_element=output_width,
                 reuse_gate_up_input=reuse_input,ffn_ms_per_token=total['model_ns']/len(tokens)/1e6,
                 controller_fanout=controller_fanout,
                 ffn_p50_ms=latency[flash.ceildiv(len(latency),2)-1]/1e6,
                 ffn_p95_ms=latency[flash.ceildiv(95*len(latency),100)-1]/1e6,
                 useful_bandwidth_fraction=total['useful_bytes']*1e9/(102_400_000_000*total['model_ns']),
                 active_chip_fraction=total['active_chip_commands']/(16*total['commands']),
                 chip_bytes=[c['useful'] for c in by_chip],
                 chip_service_ns_by_chip=[c['service'] for c in by_chip],
                 token_cost_coefficients=[dict(prompt=p,token_index=t,**v) for (p,t),v in sorted(tokens.items())],
                 model='metadata_only_int8_v3' if controller_fanout else 'metadata_only_int8_v2')
    return total


def sweep_cost(base,control_ns,host_ns):
    result={k:v for k,v in base.items() if not isinstance(v,(list,dict))}
    ns=base['flash_ns']+base['input_ns']+base['output_ns']+control_ns*base['chip_matrix_commands']+host_ns*base['commands']
    result.update(model_ns=ns,control_ns=control_ns*base['chip_matrix_commands'],
                  host_ns=host_ns*base['commands'],per_chip_control_ns=control_ns,
                  per_host_command_ns=host_ns,ffn_ms_per_token=ns/base['decode_tokens']/1e6,
                  useful_bandwidth_fraction=base['useful_bytes']*1e9/(102_400_000_000*ns))
    lat=sorted(v['flash']+v['input']+v['output']+control_ns*v['streams']+host_ns*v['commands']
               for v in base['token_cost_coefficients'])
    result.update(ffn_p50_ms=lat[flash.ceildiv(len(lat),2)-1]/1e6,
                  ffn_p95_ms=lat[flash.ceildiv(95*len(lat),100)-1]/1e6)
    return result


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--metadata',type=Path,required=True)
    p.add_argument('--trace',type=Path,action='append',required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args(); a.output_dir.mkdir(parents=True,exist_ok=True)
    metadata=json.loads(a.metadata.read_text())
    events=flash.load_events(a.trace); dims=flash.tensor_dimensions(metadata)
    base=[]
    for stacked in (False,True):
        for g in (1,2,4,8,16):
            tag=f'g{g}_'+('stacked' if stacked else 'separate')+'_overhead'
            r=evaluate(events,dims,g,stack_gate_up=stacked,binary_path=a.output_dir/(tag+'.bin'))
            r['label']=tag
            (a.output_dir/(tag+'.expected.json')).write_text(json.dumps(r,indent=2)+'\n')
            base.append(r)
    flash.write_csv(a.output_dir/'overhead_placement.csv',base)
    sensitivity=[]
    for r in base:
        for ctrl in (0,100,250,500,1000,2000,5000):
            for host in (0,2000,10000):
                sensitivity.append(sweep_cost(r,ctrl,host))
    flash.write_csv(a.output_dir/'control_sensitivity.csv',sensitivity)
    winners=[]
    for ctrl in (0,100,250,500,1000,2000,5000):
        for host in (0,2000,10000):
            candidates=[r for r in sensitivity if r['per_chip_control_ns']==ctrl and r['per_host_command_ns']==host]
            winner=min(candidates,key=lambda r:r['model_ns'])
            winners.append(winner)
    flash.write_csv(a.output_dir/'control_sensitivity_winners.csv',winners)
    manifest=dict(metadata=str(a.metadata.resolve()),traces=[str(p.resolve()) for p in a.trace],
                  model_sha256=metadata['sha256'],assumptions=dict(chip_to_channel='chip % 8',
                  channels=8,channel_bytes_per_ns=2,pcie_bytes_per_ns=8,
                  per_chip_control_ns=250,per_host_command_ns=2000,input_bytes_per_element=1,
                  output_bytes_per_element=4,gate_up_input_reuse=True,
                  pcie_input_policy='Each logical input supplied once; controller replicates to chips over ONFI',
                  default_costs='Illustrative exposed assumptions, not measured physical AiF latency',
                  phase_order='host, serialized controller chip commands, input barrier, flash, output barrier'),
                  limitations=['No measured hardware MAC-lane utilization.',
                    'Cost sweep, not physical calibration; VM ioctl cost is separate.',
                    'One matrix context per chip, no cross-expert plane packing.',
                    'Whole-output-row, block-contained short streams; no GC or general-data contention.',
                    'Routing, activation and weighted-combine computation excluded.',
                    'Shared experts and dense layers excluded: routed FFN projections only.'])
    (a.output_dir/'overhead_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(json.dumps([{k:r[k] for k in ('label','ffn_ms_per_token','useful_bandwidth_fraction')}
                      for r in base],indent=2))


if __name__=='__main__':
    main()
