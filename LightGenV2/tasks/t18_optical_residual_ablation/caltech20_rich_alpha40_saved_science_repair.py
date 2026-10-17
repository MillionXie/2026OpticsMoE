"""New independent saved verification after frozen auditor's missing math import."""
import argparse,ast,inspect,json,math,os,subprocess,types
from pathlib import Path
import numpy as np
import torch
import caltech20_rich_alpha40_namespace_repair as legacy
import audit_caltech20_rich_alpha40_rebalance as kernel
s,HERE,ROOT=legacy.s,legacy.HERE,legacy.ROOT
OUT=HERE/'runs/smoke/caltech20_rich_alpha40_saved_science_import_checked_20261018'
FILE='caltech20_rich_alpha40_saved_science_repair.py'
def sources():
    z=legacy.sources();z[(HERE/FILE).relative_to(ROOT).as_posix()]=s.sha(HERE/FILE);return z
def identity():
    git=subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip();z=sources();legacy.engine.verify_sources(z,git);return dict(sources=z,git_commit=git)
def run(key):
    assert os.environ['CUDA_VISIBLE_DEVICES']=='';folder=legacy.OUT/key;failure=folder/'audit_failure.json';assert s.read(failure)['error']=='''NameError("name 'math' is not defined")'''
    assert not (folder/'science').exists() and not (folder/'audit_completed.json').exists()
    root=OUT/key;claim=root.with_name(root.name+'.claim.json');assert not root.exists();before=s.sha(failure)
    OUT.mkdir(exist_ok=True);s.save(claim,dict(identity(),key=key,pid=os.getpid(),original_failure_sha256=before,automatic_retry=False),exclusive=True);root.mkdir()
    tree=ast.parse(inspect.getsource(kernel.run));changed=0
    for node in ast.walk(tree):
        if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id=='dest' for t in node.targets):
            assert ast.unparse(node.value)=="out / 'science'";node.value=ast.parse("REVIEW_ROOT/'science'",mode='eval').body;changed+=1
    assert changed==1
    # Full original namespace retains math/NumPy/Torch/Path/saved helpers.
    namespace=dict(vars(kernel));namespace.update(m=legacy,REVIEW_ROOT=root)
    exec(compile(ast.fix_missing_locations(tree),__file__+':unchanged_saved_checks_new_destination','exec'),namespace)
    namespace['run'](key)
    assert s.sha(failure)==before and not (folder/'science').exists() and not (folder/'audit_completed.json').exists()
    assert s.read(root/'science/audit.json')['passed'];s.save(root/'review_identity.json',dict(identity(),training_sources=s.read(folder/'metadata.json')['sources'],training_git=s.read(folder/'metadata.json')['git_commit'],original_failure_sha256=before,model_forward_calls=0,TEST_images_read=0,thresholds_unchanged=True),exclusive=True)
    s.save(root/'completed.json',dict(passed=True,actual_audit_returncode=0,audit_sha256=s.sha(root/'science/audit.json'),verification_identity_sha256=s.sha(root/'review_identity.json')),exclusive=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--key',required=True);a=p.parse_args()
    try:run(a.key)
    except BaseException as e:
        root=OUT/a.key;mark=root/'failure.json' if root.exists() else OUT.with_name(OUT.name+'.'+a.key+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
