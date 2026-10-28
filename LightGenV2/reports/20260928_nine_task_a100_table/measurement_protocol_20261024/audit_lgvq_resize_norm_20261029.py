"""Saved-data audit of the newly included per-video resize/normalization."""
import json
from pathlib import Path
import sys
import audit_lgvq_compute_scope_20261029 as original
import lgvq_compute_scope_20261029 as scope
import lgvq_resize_norm_20261029 as preprocessing

def audit(leaf,destination):
    raw=json.loads((Path(leaf)/'raw_operator_records.json').read_text())
    for r in raw['patch']:
        if not r['resize_pixels_byte_equal'] or not r['normalized_pixels_byte_equal']:
            raise ValueError('Native resize/normalization bytes failed')
        if abs(sum(r[k] for k in ('resize_cpu_s','normalization_cpu_s','patch_cpu_s'))-r['thread_cpu_s'])>1e-9:
            raise ValueError('CPU preprocessing stage partition mismatch')
    scope.summarize_compute=preprocessing.summarize_compute
    return original.audit(leaf,destination)

if __name__=='__main__':print(json.dumps(audit(*sys.argv[1:]),ensure_ascii=True))
