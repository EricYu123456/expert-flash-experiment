"""Bind actual nested producer seal, without changing the frozen audit logic."""
from pathlib import Path
from datetime import datetime,timezone
import hashlib,json

HERE=Path(__file__).resolve().parent
SOURCE=HERE.parent/'g4-M32-coupled-cp'
SEAL=SOURCE/'producer-seal/completion.v2.json'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()

assert sha(SEAL)=='33b88a1eb0799891177060b2316dc68a45633d1d1ec3de2861158fbe5278e168'
x=json.loads(SEAL.read_text());files={}
for entry in x['files']:
    f=Path(entry['path']);assert sha(f)==entry['sha256'];files[str(f)]=entry['sha256']
files[str(SEAL.resolve())]=sha(SEAL)
receipt={'utc':datetime.now(timezone.utc).isoformat(),'status':'producer_complete_frozen_before_audit',
 'implementation_sha256':sha(HERE/'implementation.frozen.json'),'files':files,
 'binding_helper_source_sha256':sha(__file__),
 'note':'Actual producer seal is producer-seal/completion.v2.json, not the top-level completion.json anticipated by the unused audit bind() helper. Independent run() implementation/rules/source remain unchanged. All producer seal entries verified.'}
(HERE/'results_binding.frozen.json').write_text(json.dumps(receipt,indent=2)+'\n')
print(json.dumps({'files_verified':len(files),'binding_sha256':sha(HERE/'results_binding.frozen.json')}))
