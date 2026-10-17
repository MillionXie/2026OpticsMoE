"""Explicit one-line storage repair of frozen convex trainer; no objective change."""
import argparse,ast,inspect,os,shutil,subprocess,sys
from pathlib import Path
import numpy as np
import torch
import caltech20_same_linear_convex as base
HERE,ROOT,s,engine,old=base.HERE,base.ROOT,base.s,base.engine,base.old
EXTRA=('caltech20_same_linear_contiguous_repair.py','check_caltech20_same_linear_contiguous_repair.py','CALTECH20_SAME_LINEAR_CONTIGUOUS_REPAIR_PROTOCOL_20261017.md')
CPU=HERE/'runs/smoke/caltech20_same_linear_contiguous_repair_cpu_20261017'
FAILED=base.folder('L2_rho03')
FAILURE_SHA='5251f1c1684bbc448349d8340d5b4d93c8ceeebc6fbbf11a3daa59c45eefb119'
FILES=base.FILES

def config():
    c=base.config();c.update(run=c['run']+'_contiguous_repair',storage_repair='only LBFGS Parameter tensor becomes contiguous; objective/values unchanged',failed_source_commit='08c4439308c84acdb9615eb9f620f8b3db6678ba',preserved_failure_sha256=FAILURE_SHA);return c

def folder(key):
    assert key in config()['cases'];return HERE/'runs/simulation'/config()['run']/key

def sources():
    z=base.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z

def identity():
    z=sources();git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();engine.verify_sources(z,git);return dict(sources=z,git_commit=git)

def claim(path,phase):s.save(path,dict(identity(),pid=os.getpid(),phase=phase,automatic_retry=False),exclusive=True)
def gone(pid):return base.gone(pid)
def admit(key):
    assert s.sha(FAILED/'failure.json')==FAILURE_SHA and not (FAILED/'fit/fit_result.json').exists() and not (FAILED/'once_test').exists();return base.admit(key)
def cpu_ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources() and p['preserved_failure_sha256']==FAILURE_SHA;engine.verify_sources(p['sources'],p['git_commit'])
objective=base.objective
training_data=base.training_data

# Compile only the frozen producer function in this adapter's explicit namespace.
# Exact AST match ensures there is precisely one semantic/storage edit.
tree=ast.parse(inspect.getsource(base.fit));edits=0
for node in ast.walk(tree):
    if isinstance(node,ast.Assign) and len(node.targets)==1 and isinstance(node.targets[0],ast.Name) and node.targets[0].id=='parameter':
        assert ast.unparse(node.value)=='torch.nn.Parameter(torch.tensor(anchor))'
        node.value=ast.parse('torch.nn.Parameter(torch.tensor(anchor).contiguous())',mode='eval').body;edits+=1
assert edits==1
exec(compile(ast.fix_missing_locations(tree),str(HERE/EXTRA[0])+':contiguous_fit','exec'),globals())
exec(compile(ast.parse(inspect.getsource(base.test)),str(HERE/EXTRA[0])+':same_test','exec'),globals())

def audit(key,phase):
    # Explicit dependency injection into the unchanged NumPy verifier, isolated
    # to this audit process. No frozen on-disk source or runtime is modified.
    import audit_caltech20_same_linear_convex as verifier
    verifier.m=sys.modules[__name__];verifier.audit(key,phase)

def campaign(key):
    assert os.environ.get('CUDA_VISIBLE_DEVICES')=='';cpu_ready();parent,_=admit(key);root=folder(key);assert not root.exists();root.parent.mkdir(parents=True,exist_ok=True);claim(root.with_name(root.name+'.campaign_claim.json'),'new_contiguous_engineering_repair_not_old_retry');root.mkdir();s.save(root/'metadata.json',dict(identity(),pid=os.getpid(),case=key,config=config(),parent=str(parent)),exclusive=True)
    with (root/'fit.log').open('x') as log:p=subprocess.Popen([sys.executable,str(Path(__file__).resolve()),'--phase','fit','--case',key],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES=''),stdout=log,stderr=subprocess.STDOUT);rc=p.wait()
    s.save(root/'producer_exit.json',dict(pid=p.pid,returncode=rc,actual_exit=True),exclusive=True);assert rc==0;claim(root/'audit_claim.json','unchanged_independent_NumPy_verifier_new_identity')
    with (root/'audit.log').open('x') as log:rc=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--phase','audit_fit','--case',key],cwd=ROOT,env=dict(os.environ,CUDA_VISIBLE_DEVICES='',OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'),stdout=log,stderr=subprocess.STDOUT).returncode
    assert rc==0 and s.read(root/'science/audit.json')['passed'];s.save(root/'audit_completed.json',dict(passed=True,actual_audit_returncode=rc,audit_sha256=s.sha(root/'science/audit.json')),exclusive=True);s.save(root/'status.json',dict(state='convex_head_science_complete_awaiting_review_noTEST',case=key))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign','test','audit_fit','audit_test'],required=True);p.add_argument('--case',required=True);a=p.parse_args()
    try:
        if a.phase.startswith('audit_'):audit(a.case,a.phase.split('_')[1])
        else:globals()[a.phase](a.case)
    except BaseException as e:
        out=folder(a.case);mark=out/(a.phase+'_failure.json') if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
