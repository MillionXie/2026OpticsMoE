"""Retained full-RMS/prototype models with TRAIN-only one-code SLM jitter."""
import math,types,importlib
import torch.nn.functional as F
import caltech20_rich_fresh_fullRMS as prior
for n,v in vars(prior).items():
    if not n.startswith('__') and n not in ('original','prior','closure'):globals()[n]=v
original=prior.original;closure=prior
FILES=('caltech20_rich_phasecode_consistency.py','caltech20_rich_phasecode_consistency_profile.json','check_caltech20_rich_phasecode_consistency.py','audit_caltech20_rich_phasecode_consistency.py','CALTECH20_RICH_PHASECODE_CONSISTENCY_PROTOCOL_20261018.md')
def config():
    c=prior.config();c.update(s.read(HERE/FILES[1]));return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_phasecode_consistency_CPU_20261018'
def sources():
    z=prior.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
def phase_noise(depth,seed,cfg):return np.clip(np.random.default_rng(np.random.SeedSequence([int(seed),20261018,421])).standard_normal((depth+1,1,16,16)).astype(np.float32),-2,2)*cfg['phase_jitter_sigma_codes']
class Alpha40MoE(prior.Alpha40MoE):
    def __init__(self,depth,rho=.3):super().__init__(depth,rho);self.phase_shift=None;self.phase_noise_SHA256=None;self.jittered_plane_calls=0;self.clean_plane_calls=0
    def context(self,seed,count,cfg):
        self.robust=None;self.offset=0;a=phase_noise(self.depth,seed,cfg);self.phase_noise_SHA256=s.digest(a);codes=F.interpolate(torch.tensor(a,device=self.raw_router.device),(518,518),mode='bilinear',align_corners=False)[:,0];self.phase_shift=torch.exp(2j*math.pi*codes/256);self.jittered_plane_calls=0;self.clean_plane_calls=0
    def plane(self,stage):
        result=super().plane(stage)
        if self.training:
            if self.linear.mask is not None:
                assert self.phase_shift is not None;self.jittered_plane_calls+=1;result=result*self.phase_shift[0 if stage=='router' else int(stage)+1]
            else:self.clean_plane_calls+=1
        return result
    def contract(self):
        c=super().contract();c.update(fresh_alpha45_full_RMS_and_TRAIN_prototypes=False,TRAIN_only_augmented_phasecode_jitter=True,clean_eval_export_exactly_unchanged=True,additional_inference_parameters=0);return c
def incumbent(depth):return prior.incumbent(depth)
def initialize(depth,fresh):
    assert not fresh;engine.seed(17);q,pt,_=incumbent(depth);model=Alpha40MoE(depth);model.load_state_dict(remap(pt['model'],q['alpha_low']));model.anchor();return model
for name in ('identity','claim','select','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','fit','mixed','campaign','feature_loss','target_data'):
    f=getattr(prior,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
f=prior.prototype_data.__wrapped__;prototype_data=lru_cache(None)(types.FunctionType(f.__code__,globals(),'prototype_data',f.__defaults__,f.__closure__))
f=prior.gradients;gradient_kernel=types.FunctionType(f.__code__,globals(),'same_dualview_prototype_gradients',f.__defaults__,f.__closure__)
def gradients(model,x,y,q,mass,mask,weights,cfg,micro):
    model.jittered_plane_calls=model.clean_plane_calls=0;result=gradient_kernel(model,x,y,q,mass,mask,weights,cfg,micro);expected=math.ceil(len(x)/micro)*(model.depth+1);assert model.jittered_plane_calls==model.clean_plane_calls==expected;model.view_info.update(phase_jitter_SHA256=model.phase_noise_SHA256,phase_jitter_sigma_codes=cfg['phase_jitter_sigma_codes'],jittered_plane_calls=model.jittered_plane_calls,clean_plane_calls=model.clean_plane_calls,phase_jitter_augmented_only=True);return result
def gate(depth,fresh,data,case):
    assert not fresh;model=initialize(depth,False).cuda().eval();q,pt,_=incumbent(depth);mod=importlib.import_module(q['module']);reference=getattr(mod,q['class_name'])(depth,.3).cuda().eval();reference.load_state_dict(pt['model']);idx=np.array([int((data[1]==k).nonzero()[0]) for k in range(20)]);pe=ce=0.
    with torch.no_grad():
        for off in range(0,20,8):x=base.encode(data[0][idx[off:off+8]].cuda(),None,100);a,b=model(x);p,c=reference(x);pe=max(pe,float((a-p).abs().max()));ce=max(ce,float((b-c).abs().max()))
    assert pe<=3e-6 and ce<=3e-6;del model,reference;torch.cuda.empty_cache();proof=numeric_gate_kernel(depth,False,data,case);proof.update(parent_clean_probability_maxabs=pe,parent_capture_maxabs=ce,phase_jitter_augmented_only=True,phase_jitter_sigma_codes=config()['phase_jitter_sigma_codes'],full_rms_gradient=True);return proof
def ready():
    q=s.read(CPU/'checks.json');assert q['passed'] and q['sources']==sources() and q['phase_rotation_and_RNG_proof'];identity();old.controls()
    for d in (4,6):prototype_data(d)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
