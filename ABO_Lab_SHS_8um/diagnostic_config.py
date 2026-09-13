"""Isolated, explicitly non-production six-stage smoke configurations."""
import json,re
from pathlib import Path

ROOT=Path(__file__).resolve().parent

def job_config(spec):
    rel=spec.get('config','LAB.local.json')
    p=(ROOT/rel).resolve()
    if p==ROOT/'LAB.local.json':return rel
    if not p.is_relative_to((ROOT/'results/smoke_configs').resolve()) or p.suffix!='.json':
        raise ValueError('Diagnostic config must be under results/smoke_configs')
    c=json.loads(p.read_text(encoding='utf-8-sig'));validate(c)
    if spec.get('session',c['diagnostic_session'])!=c['diagnostic_session']:raise ValueError('Diagnostic session/config mismatch')
    return str(p)

def validate(c):
    if c.get('diagnostic_only') is not True:raise ValueError('Diagnostic-only label required')
    if not re.fullmatch('smoke_[A-Za-z0-9_-]{1,65}',c.get('diagnostic_session','')):raise ValueError('Diagnostic session must start smoke_')
    evidence=c.get('geometry_evidence',{})
    if evidence.get('method')!='measured_markers_with_asymmetric_check' or not evidence.get('report_sha256'):
        raise ValueError('Actual marker and asymmetric-check evidence required; never fabricated ROI')

def configure_generated(c):
    """Call after compat path registration, before importing the task module."""
    if not c.get('diagnostic_only'):return
    validate(c)
    import phase_encoding
    def isolated(root,config):
        validate(config)
        return Path(root)/'generated'/config['diagnostic_session']
    phase_encoding.generated_root=isolated
    # Some callers already imported patterns while registering helpers.
    import sys
    for name in ('patterns','dual_patterns','fresnel'):
        if name in sys.modules and hasattr(sys.modules[name],'generated_root'):
            sys.modules[name].generated_root=isolated

def select_queries(all_samples,indices,limit):
    """Fixed declared indices, not outcome-based cherry-picking; keep all titles."""
    if not isinstance(indices,list) or not 1<=len(indices)<=8 or len(indices)!=limit:
        raise ValueError('Diagnostic query count must match --limit (1..8)')
    images=[s for s in all_samples if s['kind']=='image'];titles=[s for s in all_samples if s['kind']=='title']
    if len(titles)!=100 or len(indices)!=len(set(indices)) or any(type(i) is not int or not 0<=i<len(images) for i in indices):
        raise ValueError('Invalid query indices or candidate count')
    return titles+[images[i] for i in indices]
