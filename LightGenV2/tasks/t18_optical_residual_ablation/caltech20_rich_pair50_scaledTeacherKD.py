"""One TRAIN-evidence-based teacher-scale calibration; frozen paired50 graph."""
import ast,inspect
from functools import lru_cache
import caltech20_rich_pair50_balancedKD as core
# Rebind frozen reviewed training kernels into this module's own namespace.
# All reused source bytes are part of sources(); no old module/file is patched.
for key,value in vars(core).items():
    if not key.startswith('__') and not inspect.isfunction(value):globals()[key]=value
FILES=('caltech20_rich_pair50_scaledTeacherKD.py', 'caltech20_rich_pair50_scaledTeacherKD_profile.json', 'check_caltech20_rich_pair50_scaledTeacherKD.py', 'audit_caltech20_rich_pair50_scaledTeacherKD.py', 'CALTECH20_RICH_PAIR50_SCALEDTEACHERKD_PROTOCOL_20261017.md')
RUN='caltech20_rich20_pair50_residual_scaledTeacherKD20_s17_20261017'
OUT=HERE/'runs/simulation'/RUN
CPU=HERE/'runs/smoke/caltech20_rich_pair50_scaledTeacherKD20_cpu_20261017'
for definition in ast.parse(inspect.getsource(core)).body:
    if not isinstance(definition,ast.FunctionDef) or definition.name in ('config','sources'):continue
    if definition.name=='teacher_targets':
        replaced=0
        for node in ast.walk(definition):
            if isinstance(node,ast.Assign) and any(isinstance(a,ast.Name) and a.id=='z' for a in node.targets):
                assert ast.unparse(node.value)=="logits / config()['temperature']"
                node.value=ast.parse("logits / config()['teacher_temperature']",mode='eval').body;replaced+=1
        assert replaced==1
    for node in ast.walk(definition):
        if isinstance(node,ast.Constant) and isinstance(node.value,str):node.value=node.value.replace('audit_caltech20_rich_pair50_balancedKD.py','audit_caltech20_rich_pair50_scaledTeacherKD.py').replace('balancedKD20','scaledTeacherKD20')
    exec(compile(ast.fix_missing_locations(ast.Module(body=[definition],type_ignores=[])),__file__,'exec'),globals())
def config():
    p=dict(base.config());p.update(s.read(HERE/FILES[1]))
    assert p['tune_epochs']==20 and p['temperature']==4. and p['teacher_temperature']==16. and p['KD_gradient_fraction']==.25 and p['KD_scale_cap']==.1 and p['new_inference_parameters']==0
    return p
def sources():
    z=core.sources()
    for name in FILES:z[(HERE/name).relative_to(ROOT).as_posix()]=s.sha(HERE/name)
    return z
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign','test'],required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    try:globals()[a.phase](a)
    except BaseException as e:
        mark=a.out/'failure.json' if a.out.exists() else a.out.with_name(a.out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),pid=os.getpid(),automatic_retry=False),exclusive=True)
        raise
