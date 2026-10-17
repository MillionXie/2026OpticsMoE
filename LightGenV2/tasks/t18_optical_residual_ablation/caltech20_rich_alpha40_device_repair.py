"""Explicit training-anchor device repair; original failed source stays frozen."""
import ast,inspect
import caltech20_rich_alpha40_rebalance as core
for n,v in vars(core).items():
    if not n.startswith('__') and not inspect.isfunction(v):globals()[n]=v
from caltech20_rich_alpha40_model import Alpha40MoE as OriginalAlpha40MoE
EXTRA=('caltech20_rich_alpha40_device_repair.py','check_caltech20_rich_alpha40_device_repair.py','audit_caltech20_rich_alpha40_device_repair.py','CALTECH20_RICH_ALPHA40_DEVICE_REPAIR_PROTOCOL_20261018.md')
FILES=(EXTRA[0],core.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
class Alpha40MoE(OriginalAlpha40MoE):
    def _apply(self,fn,recurse=True):
        result=super()._apply(fn,recurse=recurse)
        # Training-only anchor tensors are not persistent buffers/parameters.
        # Module.cuda/to normally migrates neither Python dictionaries nor them.
        self.phase_anchors={n:fn(v) for n,v in self.phase_anchors.items()}
        return result
    def context(self,seed,count,cfg):
        assert all(v.device==dict(self.named_parameters())[n].device for n,v in self.phase_anchors.items())
        return super().context(seed,count,cfg)
def config():
    c=core.config();c.update(run=c['run']+'_device_anchor_repair',engineering_repair='Move every nonpersistent phase anchor with Module._apply; preserve original pre-update device failure and unchanged numeric thresholds.');return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_alpha40_device_repair_cpu_20261018'
def sources():
    z=core.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def failed_preflight():
    assert s.read(core.OUT/'status.json')['state']=='failed';ex=s.read(core.OUT/'campaign_exit.json');assert ex['returncode']==1 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists()
    meta=s.read(core.OUT/'metadata.json');assert meta['sources']==core.sources();engine.verify_sources(meta['sources'],meta['git_commit']);records={}
    for key in ('L2_rebalance','L4_rebalance','L6_rebalance'):
        folder=core.OUT/key;fail=s.read(folder/'failure.json');assert fail['error']=="RuntimeError('Expected all tensors to be on the same device, but found at least two devices, cuda:0 and cpu!')"
        e=s.read(core.OUT/(key+'.producer_exit.json'));assert e['returncode']==1 and e['actual_exit'] and not Path('/proc',str(e['pid']),'cmdline').exists()
        assert not any((folder/n).exists() for n in ('numeric_gate.json','metadata.json','history.json','last.pt','best.pt','fit_result.json','audit_claim.json'))
        records[key]=dict(failure_sha256=s.sha(folder/'failure.json'),producer_exit_sha256=s.sha(core.OUT/(key+'.producer_exit.json')),new_optimizer_steps=0,new_TEST_calls=0)
    assert not (core.OUT/'L4_fresh').exists();return dict(passed=True,original_failure_receipts=records)
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
