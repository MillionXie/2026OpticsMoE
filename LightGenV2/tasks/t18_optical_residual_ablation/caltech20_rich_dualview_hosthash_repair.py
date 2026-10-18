"""Explicit host hashing repair; failed source346 outputs remain immutable."""
import ast,inspect,types
import caltech20_rich_targeted_dualview as original
for n,v in vars(original).items():
    if not n.startswith('__'):globals()[n]=v
EXTRA=('caltech20_rich_dualview_hosthash_repair.py','check_caltech20_rich_dualview_hosthash_repair.py','audit_caltech20_rich_dualview_hosthash_repair.py','CALTECH20_RICH_DUALVIEW_HOSTHASH_REPAIR_PROTOCOL_20261018.md')
FILES=(EXTRA[0],original.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
def config():
    c=original.config();c.update(run=c['run']+'_hosthash_checked',host_hash_repair='Explicit GPU tensor detach+CPU copy before checksum. Original346 failed before optimizer/TEST.');return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_dualview_hosthash_cpu_20261018'
def sources():
    z=original.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
class Alpha40MoE(original.Alpha40MoE):
    def __init__(self,depth,rho=.3):
        super().__init__(depth,rho);self.full_rms_gradient=ACTIVE_FULL_RMS
    def contract(self):
        c=super().contract();c.update(host_hash_engineering_repair=True);return c
for name in ('identity','claim','select','rates','seed_for','load_predictions','groups','incumbent','initialize','mixed','js_loss','ready','numeric_gate_kernel','fit','campaign'):
    f=getattr(original,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
text=inspect.getsource(original.gradients);site='s.digest(CLEAN_X)';assert text.count(site)==1;text=text.replace(site,'s.digest(CLEAN_X.detach().cpu())');exec(compile(text,__file__+':same_gradients_explicit_host_checksum','exec'),globals())
def gate(depth,fresh,data,case):
    a=torch.arange(8,device='cuda',dtype=torch.float32);assert s.digest(a.detach().cpu())==s.digest(torch.arange(8,dtype=torch.float32))
    f=types.FunctionType(original.gate.__code__,globals(),'original_gate',original.gate.__defaults__,original.gate.__closure__);p=f(depth,fresh,data,case);p['actual_GPU_host_hash_roundtrip_passed']=True;return p
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
