"""Per-depth TRAIN-evidence continuation; no deployed model changes."""
import ast,inspect
from functools import lru_cache
import caltech20_rich_pair50_balancedKD as core
import caltech20_rich_pair50_scaledTeacherKD as parent
import caltech20_rich_pair50_depthwise_resources as resources
for key,value in vars(core).items():
    if not key.startswith('__') and not inspect.isfunction(value):globals()[key]=value
FILES=('caltech20_rich_pair50_depthwise_continue.py', 'caltech20_rich_pair50_depthwise_continue_profile.json', 'check_caltech20_rich_pair50_depthwise_continue.py', 'audit_caltech20_rich_pair50_depthwise_continue.py', 'CALTECH20_RICH_PAIR50_DEPTHWISE_CONTINUE_PROTOCOL_20261017.md','caltech20_rich_pair50_depthwise_resources.py')
original_base=base
class BaseContext:
    GPU=resources.GPU
    slots=staticmethod(resources.slots)
    def __getattr__(self,name):return getattr(original_base,name)
base=BaseContext()
GPU=resources.GPU
RUN='caltech20_rich20_pair50_residual_depthwise90_s17_20261017'
OUT=HERE/'runs/simulation'/RUN
CPU=HERE/'runs/smoke/caltech20_rich_pair50_depthwise90_cpu_20261017'
BASE=parent.OUT
CASES={f'L{d}_rho03':(d,.3) for d in s.read(HERE/FILES[1])['active_depths']}
CURRENT_CASE=None
for definition in ast.parse(inspect.getsource(core)).body:
    if not isinstance(definition,ast.FunctionDef) or definition.name in ('config','sources','admit','_admit_saved_math'):continue
    for node in ast.walk(definition):
        if isinstance(node,ast.Constant):
            if type(node.value) is int:node.value={50:70,51:71,70:90,9850:13790,13790:17730}.get(node.value,node.value)
            elif isinstance(node.value,str):node.value=node.value.replace('audit_caltech20_rich_pair50_balancedKD.py','audit_caltech20_rich_pair50_depthwise_continue.py').replace('balancedKD20','depthwiseKD20')
    if definition.name=='teacher_targets':
        found=0
        for node in ast.walk(definition):
            if isinstance(node,ast.Assign) and any(isinstance(x,ast.Name) and x.id=='z' for x in node.targets):
                assert ast.unparse(node.value)=="logits / config()['temperature']";node.value=ast.parse("logits / config()['teacher_temperature']",mode='eval').body;found+=1
        assert found==1
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition],type_ignores=[])),__file__,'exec'),globals())
def config(key=None):
    p=dict(parent.config());p.update(s.read(HERE/FILES[1]));key=key or CURRENT_CASE
    if key is not None:assert key in CASES;p.update(p['per_depth'][str(CASES[key][0])])
    assert p['tune_epochs']==20 and p['temperature']==4. and p['teacher_temperature']==16. and p['KD_gradient_fraction']==.25 and p['KD_scale_cap']==.5 and p['new_inference_parameters']==0
    return p
def sources():
    z=parent.sources()
    for name in FILES:z[(HERE/name).relative_to(ROOT).as_posix()]=s.sha(HERE/name)
    return z
def admit():
    controls=parent.admit();state=s.read(BASE/'status.json');assert state['state']=='scaledTeacherKD20_science_and_reviewed_tests_complete' and not state['active'] and not state['pending'] and not state['failures']
    meta=s.read(BASE/'metadata.json');assert meta['sources']==parent.sources() and meta['config']==parent.config();engine.verify_sources(meta['sources'],meta['git_commit'])
    for receipt,folder in [('audit_completed.json','science'),('test_audit_completed.json','saved_test_audit')]:
        z=s.read(BASE/receipt);assert z['actual_audit_returncode']==0 and z['audit_sha256']==s.sha(BASE/folder/'audit.json') and s.read(BASE/folder/'audit.json')['passed']
    ex=s.read(BASE/'campaign_exit.json');started=s.read(BASE.with_name(BASE.name+'.started.json'));assert ex['returncode']==0 and ex['actual_exit']
    for pid in (meta['pid'],started['supervisor_pid']):
        p=Path('/proc',str(pid),'cmdline');assert not p.exists() or str(BASE).encode() not in p.read_bytes()
    for key in CASES:
        for path in (BASE/(key+'.fit_exit.json'),BASE/'once_tests'/(key+'.test_exit.json')):
            z=s.read(path);assert z['returncode']==0 and z['actual_exit'];p=Path('/proc',str(z['pid']),'cmdline');assert not p.exists() or str(BASE).encode() not in p.read_bytes()
        f=s.read(BASE/key/'fit_result.json');assert f['actual_joint_epochs']==70 and f['steps']==13790
        for field,name in [('last_sha256','last.pt'),('best_sha256','best.pt'),('history_sha256','history.json'),('method_trace_sha256','method_trace.npz')]:assert f[field]==s.sha(BASE/key/name)
        with np.load(BASE/key/'method_trace.npz',allow_pickle=False) as z:
            cosine=z['gradient_cosine'].mean(0);fraction=(z['group_scales']*z['KD_norm']/np.maximum(z['CE_norm'],1e-30)).mean(0);cap_active=(z['group_scales']>=.099999).mean(0)
        assert np.all(cosine>.1) and np.all(fraction<.20) and np.all(cap_active>.5),'Parent gradients do not support strengthening KD; do not start this strategy'
    return controls
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign','test'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args();CURRENT_CASE=a.key
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),pid=os.getpid(),automatic_retry=False),exclusive=True)
        raise
