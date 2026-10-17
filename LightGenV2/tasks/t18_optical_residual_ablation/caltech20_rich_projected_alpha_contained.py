"""Representable physical-alpha containment, before any projected GPU run."""
import types
import numpy as np
import caltech20_rich_projected_alpha as prior
for n,v in vars(prior).items():
    if not n.startswith('__'):globals()[n]=v
ALPHA_MIN=float(np.nextafter(np.float32(.4001),np.float32(1.)))
class ContainedMoE(prior.Alpha40MoE):
    def project(self):
        with torch.no_grad():
            for p in self.fusion_logits:p.clamp_(ALPHA_MIN,.95)
    def contract(self):
        c=super().contract();c.update(id='singleRouter_EG_directAlpha04_fp32ContainedCurriculum_20261018',alpha_projection_parameter_min=ALPHA_MIN,alpha_physical_strict_min=.4001,alpha_projection_precision='project to nearest float32 strictly above nominal .4001; no relaxed bound');return c
Alpha40MoE=ContainedMoE
EXTRA=('caltech20_rich_projected_alpha_contained.py','check_caltech20_rich_projected_alpha_contained.py','audit_caltech20_rich_projected_alpha_contained.py','CALTECH20_RICH_PROJECTED_ALPHA_CONTAINMENT_PROTOCOL_20261018.md')
FILES=(EXTRA[0],prior.FILES[1],prior.FILES[2],EXTRA[1],EXTRA[2],EXTRA[3])
def config():
    c=prior.config();c.update(run=c['run']+'_fp32contained',alpha_projection_parameter_min=ALPHA_MIN,precision_containment='Nominal.4001 rounds below strict guard in float32. Project to the next representable float32 ABOVE .4001; keep original strict numeric range. Original328 has no GPU launch.');return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_projected_alpha_contained_cpu_20261018'
def sources():
    z=prior.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','groups','initialize','rates','seed_for','mixed','gradients','incumbent','load_predictions','gate','fit','campaign','alpha_schedule','ready'):
    f=getattr(prior,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        folder=OUT/a.key if a.key else OUT;mark=folder/'failure.json' if folder.exists() else folder.with_name(folder.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
