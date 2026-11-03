#!/usr/bin/env python3
"""Keep frozen model source; normalize report-hash path argument only."""
from pathlib import Path
import hashlib
import json
import stage_certificate as stage

HERE=Path(__file__).resolve().parent
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

if __name__=='__main__':
    receipt=HERE/'reporting_erratum.frozen.json'
    if not receipt.exists():
        receipt.write_text(json.dumps(dict(status='frozen_before_reporting_rerun',
            source_sha256=sha(HERE/'stage_certificate.py'),controller_sha256=sha(__file__),
            protocol_sha256=sha(HERE/'protocol.frozen.json'),
            original_failure="sha(__file__) passed a string to a Path-only hash helper; arithmetic,48kernel jobs and8smalloracles had already passed before reportwriting.",
            correction='Normalize path argument with pathlib.Path in the hash helper only. Allsourceparameters,solverlogic,attainment andmodelsource unchanged. Re-run unchangedrunfunction.',
            no_model_source_mutation=True),indent=2)+'\n')
    stage.sha=sha
    stage.run()
