"""TRAIN-only intermediate supervision reuses the SAME existing linear head."""
import inspect,types,textwrap
import caltech20_rich_prototype_consistency as baseline
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('original','baseline'):globals()[n]=v
original=baseline.original
import caltech20_rich_L2_strongview_namespace_repair as closure
FILES=('caltech20_rich_L6_shared_supervision.py','caltech20_rich_L6_shared_supervision_profile.json','check_caltech20_rich_L6_shared_supervision.py','audit_caltech20_rich_L6_shared_supervision.py','CALTECH20_RICH_L6_SHARED_SUPERVISION_PROTOCOL_20261018.md')
def config():
    c=baseline.config();c.update(s.read(HERE/FILES[1]));return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_L6_shared_supervision_cpu_20261018'
def sources():
    z=closure.sources()
    for n in FILES:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
class Alpha40MoE(baseline.Alpha40MoE):
    def contract(self):
        c=super().contract();c.update(training_shared_head_intermediate_loss=True,additional_inference_heads=0,shared_readout_parameters=5140);return c
text=textwrap.dedent(inspect.getsource(original.original.Alpha40MoE.forward));assert text.count('self.last={};self.trace=[]')==1;text=text.replace('self.last={};self.trace=[]','self.last={};self.trace=[];self.auxiliary=[]')
site='amplitude=re*mixed/rm';assert text.count(site)==1;text=text.replace(site,site+"\n        if self.training and k<self.depth-1 and self.stages[k]=='global':self.auxiliary.append(torch.softmax(self.linear(features(amplitude)),1))")
ns={};exec(compile(text,__file__+':same_inference_forward_with_training_only_shared_head_reads','exec'),globals(),ns);Alpha40MoE.forward=ns['forward']
def completed_mild():
    out=baseline.OUT/'L6_prototype';ex=s.read(baseline.OUT/'L6_prototype.producer_exit.json');rec=s.read(out/'audit_completed.json');assert ex['returncode']==0 and ex['actual_exit'] and not Path('/proc',str(ex['pid']),'cmdline').exists();assert rec['actual_audit_returncode']==0 and rec['audit_sha256']==s.sha(out/'science/audit.json') and s.read(out/'science/audit.json')['passed'];return s.read(out/'fit_result.json')
@lru_cache(None)
def incumbent(depth):
    q,pt,pred=baseline.incumbent(depth)
    if depth!=6:return q,pt,pred
    f=completed_mild();folder=baseline.OUT/'L6_prototype';score=f['metrics']['test']['accuracy'];assert score-old.controls()['L6_rho0']['accuracy']<.1,'Target already achieved, do not launch new strategy'
    if score<=engine.metrics(pred['test']['labels'],pred['test']['probabilities'])['accuracy']:return q,pt,pred
    meta=s.read(folder/'metadata.json');assert meta['sources']==baseline.sources() and f['best_sha256']==s.sha(folder/'best.pt');p=torch.load(folder/'best.pt',map_location='cpu',weights_only=False)
    q=dict(path=str(folder/'best.pt'),sha256=f['best_sha256'],source_git=meta['git_commit'],module=baseline.__name__,class_name='Alpha40MoE',alpha_low=None,joint_budget_epochs=f['previous_budget_epochs']+f['actual_new_epochs'],selected_optical_epoch=p['epoch']);pred={r:load_predictions(folder/('selected_'+r+'.npz')) for r in ('train','val','test')};return q,p,pred
for name in ('identity','claim','select','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','fit','mixed','campaign','feature_loss','initialize','gate'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
def auxiliary_CE(probabilities,y,weights):return -(y*weights[None]*probabilities.clamp_min(1e-12).log()).sum(1).mean()
def gradients(model,x,y,q,mass,mask,weights,cfg,micro):
    assert CLEAN_X is not None and CLEAN_X.shape==x.shape;model.train();model.zero_grad(set_to_none=True);ps=[];cs=[];parts=dict(clean=0.,augmented=0.,KD_clean=0.,JS=0.,phase_anchor=0.,feature_alignment=0.,shared_head_auxiliary=0.);total=0.
    for off in range(0,len(x),micro):
        size=min(micro,len(x)-off);factor=size/len(x);model.offset=off;model.linear.mask=None;pc,cc=model(CLEAN_X[off:off+micro]);clean_features=model.semantic_feature;auxiliary=model.auxiliary;assert len(auxiliary)==2;model.linear.mask=mask[off:off+micro];pa,ca=model(x[off:off+micro])
        clean=base.loss(model,pc,cc,y[off:off+micro],weights,cfg);aug=base.loss(model,pa,ca,y[off:off+micro],weights,cfg);kd=old.kd.KD_loss(pc,q[off:off+micro],mass[off:off+micro],cfg);js=js_loss(pc,pa);anchor=sum((p-model.phase_anchors[n]).square().sum() for n,p in model.named_parameters() if n in model.phase_anchors)/2
        alignment_value=feature_loss(clean_features,FEATURE_TARGET[off:off+micro],mass[off:off+micro]);scale=target_data(model.depth)[1];aux=torch.stack([auxiliary_CE(p,y[off:off+micro],weights) for p in auxiliary]).mean();loss=.5*clean+.5*aug+cfg['KD_weight']*kd+cfg['consistency_weight']*js+cfg['phase_anchor_weight']*anchor+scale*alignment_value+cfg['auxiliary_weight']*aux
        (loss*factor).backward();total+=float(loss.detach())*factor;ps.append(pa.detach());cs.append(ca.detach())
        for name,value in zip(parts,(clean,aug,kd,js,anchor,alignment_value,aux)):parts[name]+=float(value.detach())*factor
    model.linear.mask=None;model.view_info=dict(parts,weighted_sum=total,teacher_view='unaugmented_MixUp_same_ids_permutation_mass',full_rms_gradient=True,clean_macro_sha256=s.digest(CLEAN_X.detach().cpu()),new_inference_parameters=0,alignment_scale=scale,alignment_target_sha256=target_data(model.depth)[2],auxiliary_count=2,shared_readout_parameters=5140);return torch.cat(ps),torch.cat(cs),total
def ready():
    p=s.read(CPU/'checks.json');assert p['passed'] and p['sources']==sources() and p['all_actual_kernel_namespaces_resolved'] and p['prototype_derivative_passed'] and p['shared_head_derivative_passed'];identity();old.controls();completed_mild();target_data(6)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        out=OUT/a.key if a.key else OUT;mark=out/'failure.json' if out.exists() else out.with_name(out.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
