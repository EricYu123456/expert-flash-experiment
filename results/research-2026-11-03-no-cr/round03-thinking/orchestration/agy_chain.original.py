import subprocess,sys,json,datetime
from pathlib import Path
b=Path('results/research-2026-11-03-no-cr/round03-thinking')
for name in sys.argv[1:]:
 p=b/name
 print(json.dumps({'chain_block':name,'status':'starting','at_utc':datetime.datetime.now(datetime.timezone.utc).isoformat()}),flush=True)
 r=subprocess.run([sys.executable,'/tmp/no_cr_round03_agy_text_runner.py',str(p),'formal_'+name])
 if r.returncode:
  print(json.dumps({'chain_block':name,'status':'stopped_failure','returncode':r.returncode}),flush=True);sys.exit(r.returncode)
print(json.dumps({'status':'chain_complete','experiment_authorized':False}),flush=True)
