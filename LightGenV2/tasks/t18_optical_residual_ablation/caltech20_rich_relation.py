"""TRAIN-only teacher geometry guidance, no additional inference parameters."""
import ast,inspect,weakref
from functools import lru_cache
import caltech20_rich_robust_precise as core
for n,v in vars(core).items():
    if not n.startswith('__') and not inspect.isfunction(v):globals()[n]=v
from caltech20_rich_robust_precise_model import PreciseMoE as RobustMoE,remap
FILES=('caltech20_rich_relation.py','caltech20_rich_robust_precise_model.py','caltech20_rich_relation_profile.json','check_caltech20_rich_relation.py','audit_caltech20_rich_relation.py','CALTECH20_RICH_RELATION_PROTOCOL_20261018.md')
RELATION_TARGET=None

def config():
    c=core.config();c.update(s.read(HERE/FILES[2]));return c

OUT=SIM/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_relation_cpu_20261018'

def sources():
    z=core.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z

def resource_overlap():
    slots=resources.slots();assert slots['counts'][resources.GPU[1]]==slots['counts'][resources.GPU[2]]==0 and slots['total']<=2
    for j in slots['jobs']:
        cmd=Path('/proc',str(j['pid']),'cmdline').read_bytes().replace(b'\0',b' ').decode()
        assert (str(HERE/core.FILES[0]) in cmd and '--key L6_robust_precise' in cmd) or (str(HERE/core.original.FILES[0]) in cmd and '--key L4_robustSAM' in cmd)
    for key in ('L2_robust','L4_robust'):
        folder=core.original.OUT/key;ex=s.read(core.original.OUT/(key+'.producer_exit.json'));rec=s.read(folder/'audit_completed.json')
        assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists()
        assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(folder/'science/audit.json') and s.read(folder/'science/audit.json')['passed']
        assert s.read(folder/'fit_result.json')['selected_tune']==0,'Unexpected parent improvement: review before starting this strategy'
    return dict(passed=True,slots=slots)

for node in ast.parse(inspect.getsource(core.original)).body:
    if isinstance(node,ast.FunctionDef) and node.name not in ('config','sources','targets','gradients','fit','campaign'):
        exec(compile(ast.fix_missing_locations(ast.Module(body=[node],type_ignores=[])),__file__,'exec'),globals())

@lru_cache(None)
def teacher_geometry():
    kd.teacher_ready()
    with np.load(kd.teacher.OUT/'train_features.npz',allow_pickle=False) as z:
        assert z['ids'].tolist()==base.manifest()['split_ids']['train'];f=z['features'].astype(np.float32)
    f=f-f.mean(0,keepdims=True);f/=np.maximum(np.linalg.norm(f,axis=1,keepdims=True),1e-8)
    return f

def targets(idx,epoch,batch,cfg,device):
    global RELATION_TARGET
    q,mass=core.original.targets(idx,epoch,batch,cfg,device)
    rg=np.random.default_rng(np.random.SeedSequence([17,epoch,batch,20261013,4]));lam=float(rg.beta(cfg['mixup_alpha'],cfg['mixup_alpha']));lam=max(lam,1-lam);perm=rg.permutation(len(idx));f=teacher_geometry()[idx]
    RELATION_TARGET=torch.tensor(lam*f+(1-lam)*f[perm],device=device)
    return q,mass

def relation_loss(student,teacher):
    a=torch.nn.functional.normalize(student,dim=1,eps=1e-8);b=torch.nn.functional.normalize(teacher,dim=1,eps=1e-8)
    error=a@a.T-b@b.T;mask=~torch.eye(len(a),device=a.device,dtype=torch.bool)
    return error[mask].square().mean()

def gradients(model,x,y,q,mass,mask,weights,cfg,micro):
    assert RELATION_TARGET is not None and len(RELATION_TARGET)==len(x)
    if not hasattr(model,'relation_hook'):
        reference=weakref.ref(model)
        def keep_feature(module,args):
            obj=reference()
            if obj is not None:obj.relation_feature=args[0]
        model.relation_hook=model.linear.register_forward_pre_hook(keep_feature)
    model.train();model.zero_grad(set_to_none=True);features=[]
    # Cache full-macro features without optical graphs; no BatchNorm/running state.
    with torch.no_grad():
        for off in range(0,len(x),micro):
            model.offset=off;model.linear.mask=mask[off:off+micro];model(x[off:off+micro]);features.append(model.relation_feature.detach())
    leaf=torch.cat(features).detach().requires_grad_(True);model.linear.mask=mask
    prob=torch.softmax(model.linear(leaf),1)
    label=-(y*weights[None]*prob.clamp_min(1e-12).log()).sum(1).mean()+cfg['KD_weight']*kd.KD_loss(prob,q,mass,cfg)
    ce_grad=torch.autograd.grad(label,leaf)[0]
    rel=relation_loss(leaf,RELATION_TARGET);rg=torch.autograd.grad(rel,leaf)[0]
    cn=float(ce_grad.double().norm());rn=float(rg.double().norm());scale=min(cfg['relation_scale_cap'],cfg['relation_feature_gradient_fraction']*cn/(rn+1e-12));fg=rg.detach()*scale
    model.zero_grad(set_to_none=True);ps=[];cs=[];values=[];feature_error=0.
    for off in range(0,len(x),micro):
        model.offset=off;model.linear.mask=mask[off:off+micro];p,c=model(x[off:off+micro])
        feature_error=max(feature_error,float((model.relation_feature.detach()-leaf.detach()[off:off+micro]).abs().max()))
        value=(base.loss(model,p,c,y[off:off+micro],weights,cfg)+cfg['KD_weight']*kd.KD_loss(p,q[off:off+micro],mass[off:off+micro],cfg))*len(p)/len(x)
        (value+(model.relation_feature*fg[off:off+micro]).sum()).backward();values.append(float(value.detach()));ps.append(p.detach());cs.append(c.detach())
    assert feature_error<=3e-6,'Two-pass features differ: no stale/random surrogate permitted'
    model.linear.mask=None;model.relation_info=dict(loss=float(rel.detach()),scale=scale,classification_feature_grad_norm=cn,relation_feature_grad_norm=rn,feature_replay_maxabs=feature_error)
    return torch.cat(ps),torch.cat(cs),sum(values)+scale*float(rel.detach())

text=inspect.getsource(core.original.fit)
old="sam_passes=2 if case['sam'] else 1,adam_steps=steps,clip_norm=clip"
assert old in text;text=text.replace(old,"sam_passes=2 if case['sam'] else 1,adam_steps=steps,clip_norm=clip,relation_info=model.relation_info")
exec(compile(text,__file__+':same_fit_with_relation_trace','exec'),globals())
text=inspect.getsource(core.original.campaign);assert "and resources.slots()['total']==0" in text;text=text.replace("and resources.slots()['total']==0","and resource_overlap()['passed']")
exec(compile(text,__file__+':same_campaign','exec'),globals())
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');a=p.parse_args()
    try:fit(a.key) if a.phase=='fit' else campaign()
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
