"""TRAIN class-prototype consistency, unchanged hybrid inference graph."""
import types,importlib
import caltech20_rich_semantic_alignment as previous
for n,v in vars(previous).items():
    if not n.startswith('__') and n not in ('original','previous'):globals()[n]=v
original=previous.original
import caltech20_rich_centered_shrinkage_readout as readout
FILES=('caltech20_rich_prototype_consistency.py','caltech20_rich_prototype_consistency_profile.json','check_caltech20_rich_prototype_consistency.py','audit_caltech20_rich_prototype_consistency.py','CALTECH20_RICH_PROTOTYPE_CONSISTENCY_PROTOCOL_20261018.md')
def config():
    c=previous.config();c.update(s.read(HERE/FILES[1]));return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_prototype_consistency_cpu_20261018'
def sources():
    z=readout.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
class Alpha40MoE(previous.Alpha40MoE):
    def contract(self):
        c=super().contract();c.update(TRAIN_class_prototype_not_instance_target=True);return c
def remap(state,old_low):return {n:v.detach().clone() for n,v in state.items()} if old_low is None else original.remap(state,old_low)
@lru_cache(None)
def incumbent(depth):
    q=dict(readout.parents()[str(depth)]);pt=torch.load(q['path'],map_location='cpu',weights_only=False);mod,cls=q['model_class'].rsplit('.',1)
    q.update(module=mod,class_name=cls,alpha_low=None if 'dualview_hosthash' in mod else diagnosis.parents.parent(depth)[0]['alpha_low'],joint_budget_epochs=q['actual_joint_epochs'],selected_optical_epoch=q['selected_epoch'])
    pred={r:load_predictions(Path(q['features'][r])) for r in ('train','val','test')};return q,pt,pred
def initialize(depth,fresh):
    assert not fresh;engine.seed(17);q,pt,_=incumbent(depth);model=Alpha40MoE(depth);model.load_state_dict(remap(pt['model'],q['alpha_low']));model.anchor();return model
@lru_cache(None)
def prototype_data(depth):
    rec=s.read(alignment.OUT/'audit_completed.json');assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==config()['alignment_science_sha256']==s.sha(alignment.OUT/'science/audit.json')
    path=alignment.OUT/f'L{depth}_train_targets.npz';row=s.read(alignment.OUT/'result.json')['checks'][str(depth)];assert s.sha(path)==row['targets_sha256']
    with np.load(path,allow_pickle=False) as z:x,y,ids=[z[k].copy() for k in ('features','labels','ids')]
    assert ids.tolist()==base.manifest()['split_ids']['train'];centers=np.stack([x[y==k].mean(0) for k in range(20)]);centers=alignment.ln(centers).astype(np.float32);targets=centers[y]
    q,pt,pred=incumbent(depth);w=pt['model']['linear.weight'].numpy().astype(float);b=pt['model']['linear.bias'].numpy().astype(float);accuracy=float(((centers@w.T+b).argmax(1)==np.arange(20)).mean());assert accuracy>=.95
    features=pred['train']['features'].astype(float);p=pred['train']['probabilities'];soft=np.eye(20)[y]*(1-config()['label_smoothing'])+config()['label_smoothing']/20
    ce=(p-soft)@w;direction=features-targets;scale=min(.1,.2*float(np.linalg.norm(ce))/max(float(np.linalg.norm(direction)),1e-12));assert 0<scale<=.1
    return targets,scale,s.digest(targets),dict(source_target_SHA256=s.sha(path),TRAIN_class_prototype_parent_head_accuracy=accuracy,relative_clean_CE_feature_gradient=.2,centers_sha256=s.digest(centers),scale=scale)
def target_data(depth):return prototype_data(depth)[:3]
for name in ('identity','claim','select','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','fit','mixed','gradients','campaign','feature_loss'):
    f=getattr(previous,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
def gate(depth,fresh,data,case):
    model=initialize(depth,fresh).cuda().eval();q,pt,_=incumbent(depth);mod=importlib.import_module(q['module']);reference=getattr(mod,q['class_name'])(depth,.3).cuda().eval();reference.load_state_dict(pt['model'])
    with torch.no_grad():
        x=base.encode(data[0][:4].cuda(),None,100);p,c=model(x);a,b=reference(x);pe=float((p-a).abs().max());ce=float((c-b).abs().max());assert pe<=3e-6 and ce<=3e-6
    del model,reference;torch.cuda.empty_cache();proof=numeric_gate_kernel(depth,fresh,data,case);proof.update(parent_clean_probability_maxabs=pe,parent_clean_capture_maxabs=ce,full_rms_gradient=True,teacher_applied_to_clean_view_only=True,micro_comparison=[1,config()['microbatch_size']],TRAIN_class_prototype=prototype_data(depth)[3]);return proof
def ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources() and p['all_actual_kernel_namespaces_resolved'] and p['prototype_derivative_passed'];identity();old.controls()
    for d in (2,4,6):target_data(d)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
