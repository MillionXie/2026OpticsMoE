"""Explicit imported-symbol namespace repair; both earlier failures preserved."""
import ast,inspect
import caltech20_rich_alpha40_rebalance as core
import caltech20_rich_alpha40_device_repair as prior
# Imported helpers such as remap are functions too; keep them, then explicitly
# recompile owned kernel definitions with this version's configuration namespace.
for n,v in vars(core).items():
    if not n.startswith('__'):globals()[n]=v
from caltech20_rich_alpha40_model import remap
Alpha40MoE=prior.Alpha40MoE
EXTRA=('caltech20_rich_alpha40_namespace_repair.py','check_caltech20_rich_alpha40_namespace_repair.py','audit_caltech20_rich_alpha40_namespace_repair.py','CALTECH20_RICH_ALPHA40_NAMESPACE_REPAIR_PROTOCOL_20261018.md')
FILES=(EXTRA[0],core.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
def config():
    c=core.config();c.update(run=c['run']+'_device_anchor_namespace_repair',engineering_repair='Migrate nonpersistent anchors and explicitly retain imported helper functions when compiling kernels; preserve both earlier failures.');return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_alpha40_namespace_repair_cpu_20261018'
def sources():
    z=prior.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def failed_preflight():
    evidence=prior.failed_preflight();failure=prior.CPU.with_name(prior.CPU.name+'.failure.json')
    assert s.read(failure)['error']=='''NameError("name 'remap' is not defined")'''
    assert not prior.CPU.exists() and not prior.OUT.exists()
    evidence['prior_CPU_import_failure_sha256']=s.sha(failure);return evidence
for node in ast.parse(inspect.getsource(core)).body:
    if isinstance(node,ast.FunctionDef) and node.name not in ('config','sources','ready'):
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),globals())
def ready():
    p=s.read(CPU/'checks.json');d=s.read(diagnosis.OUT/'diagnosis.json');assert p['passed'] and p['sources']==sources() and d['passed'] and d['sources']==diagnosis.sources()
    assert p['device_anchor_proof']['passed'] and p['original_failed_preflight']==failed_preflight() and d['TEST_images_read']==d['optimizer_steps']==0
    identity();old.controls();return p
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        folder=OUT/a.key if a.key else OUT;mark=folder/'failure.json' if folder.exists() else folder.with_name(folder.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
