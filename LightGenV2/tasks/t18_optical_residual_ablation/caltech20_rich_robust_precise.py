"""Six-scalar precision repair; original failed source/claims never changed."""
import ast,inspect
import caltech20_rich_robust as original
for n,v in vars(original).items():
    if not n.startswith('__') and not inspect.isfunction(v):globals()[n]=v
from caltech20_rich_robust_precise_model import PreciseMoE as RobustMoE,remap
FILES=('caltech20_rich_robust_precise.py','caltech20_rich_robust_precise_model.py','caltech20_rich_robust_precise_profile.json','check_caltech20_rich_robust_precise.py','audit_caltech20_rich_robust_precise.py','CALTECH20_RICH_ROBUST_PRECISE_PROTOCOL_20261018.md')

def config():
    c=original.config();c.update(s.read(HERE/FILES[2]));return c

OUT=SIM/config()['run']
CPU=HERE/'runs/smoke/caltech20_rich_robust_precise_cpu_20261018'

def sources():
    z=original.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z

def resource_overlap():
    slots=resources.slots();assert slots['counts'][resources.GPU[0]]==0 and slots['total']<=3
    for job in slots['jobs']:
        cmd=Path('/proc',str(job['pid']),'cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert str(HERE/original.FILES[0]) in cmd and any(('--key '+k) in cmd for k in ('L2_robust','L4_robust','L4_robustSAM'))
    old=original.OUT/'L6_robust';assert s.sha(old/'failure.json')==config()['failed_sha256'] and not (old/'last.pt').exists() and not (old/'numeric_gate.json').exists()
    ex=s.read(original.OUT/'L6_robust.producer_exit.json');assert ex['returncode']==1 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists()
    return dict(passed=True,slots=slots)

for node in ast.parse(inspect.getsource(original)).body:
    if isinstance(node,ast.FunctionDef) and node.name not in ('config','sources','campaign'):
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),globals())
text=inspect.getsource(original.campaign);assert "and resources.slots()['total']==0" in text
text=text.replace("and resources.slots()['total']==0","and resource_overlap()['passed']")
exec(compile(text,__file__+':same_campaign_kernel','exec'),globals())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');a=p.parse_args()
    try:fit(a.key) if a.phase=='fit' else campaign()
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
