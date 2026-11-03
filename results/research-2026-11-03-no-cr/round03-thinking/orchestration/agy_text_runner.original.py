import sys,json,subprocess,hashlib,time,datetime
from pathlib import Path
prefix=Path(sys.argv[1]);phase=sys.argv[2]
prompt=prefix.with_suffix('.prompt.txt').read_text()
start=datetime.datetime.now(datetime.timezone.utc); tick=time.monotonic()
record={'model':'gemini-3.8-flash-high','effort':'high','conversation':'5bd922e5-633c-445a-8437-350ece331ffa','started_at_utc':start.isoformat(),'purpose':phase,'useG1Credits':False,'prompt_sha256':hashlib.sha256(prompt.encode()).hexdigest(),'no_tools_requested':True,'experiment_authorized':False}
cmd=['/home/ericyu/.local/bin/agy','--conversation',record['conversation'],'--model',record['model'],'--effort','high','--mode','accept-edits','--settings',json.dumps({'useG1Credits':False,'artifactReviewPolicy':'always-proceed'}),'--disable-slash-commands','--output-format','stream-json','--print-timeout','0s','--print',prompt]
with prefix.with_suffix('.stream.jsonl').open('w') as out,prefix.with_suffix('.stderr.log').open('w') as err:
 p=subprocess.Popen(cmd,stdout=out,stderr=err,cwd='/home/ericyu/expert_flash_experiment');record['PID']=p.pid
 prefix.with_suffix('.launch.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record),flush=True); rc=p.wait()
record.update(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),OS_returncode=rc,wall_seconds=time.monotonic()-tick)
last={};durations={};denied=[]
for line in prefix.with_suffix('.stream.jsonl').read_text().splitlines():
 try:x=json.loads(line)
 except:continue
 if x.get('event')=='result':last=x.get('result',{})
 if x.get('event')=='step_update':
  y=x.get('step_update',{})
  if y.get('step_type')=='agent_response' and y.get('state')=='DONE' and isinstance(y.get('duration_seconds'),(float,int)):durations[y.get('step_index',-1)]=y['duration_seconds']
 if 'denied' in line.lower() or 'permission required' in line.lower():denied.append(line[:500])
response=last.get('response','');ok=rc==0 and last.get('status')=='SUCCESS' and bool(response.strip()) and not denied
record.update(result_status=last.get('status'),response_characters=len(response),denied_actions=denied,completed_response_duration_seconds=durations.get(max(durations)) if durations else None)
if ok:
 prefix.with_suffix('.md').write_text(response.rstrip()+'\n')
 record['note_sha256']=hashlib.sha256(prefix.with_suffix('.md').read_bytes()).hexdigest()
record['successful_substantive_seconds']=min(record['wall_seconds'],record.get('completed_response_duration_seconds') or 0) if ok else 0
record['candidate_formal_clock_seconds']=record['successful_substantive_seconds'] if phase.startswith('formal') else 0
prefix.with_suffix('.exit.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n');print(json.dumps(record),flush=True)
sys.exit(0 if ok else (rc or 5))
