#!/usr/bin/env python3
"""Audit case directories without changing any sealed reference source."""
import argparse,json,pathlib
import research_20261103_selected_plan_numeric_audit_v2 as independent
from research_20261103_selected_plan_numeric import OUT,sha
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['smoke','fresh'],required=True);phase=p.parse_args().phase;dest=OUT/('smoke-primary' if phase=='smoke' else 'second-heldout');summary=json.loads((dest/'summary.json').read_text());assert summary['status']=='complete';audits={}
for case in summary['cases']:
 independent.DEST=dest/case;independent.main();audits[case]=sha(dest/case/'independent/audit.json')
summary['independent_audits_passed']=True;summary['independent_audit_sha256']=audits;summary['independent_auditor_source_sha256']=sha(pathlib.Path(independent.__file__));summary['independent_controller_sha256']=sha(__file__);(dest/'summary.json').write_text(json.dumps(summary,indent=2)+'\n');print(json.dumps({'status':'pass','cases':audits},indent=2))
