"""Isolate clean/augmented target allocation, retaining the deployed graph."""
import types,inspect
import caltech20_rich_rare_zoomout46 as baseline
import caltech20_rich_TEST_PT_selection46 as source_closure
for n,v in vars(baseline).items():
    if not n.startswith('__') and n not in ('baseline','source_closure','original','old'):globals()[n]=v
original=baseline.original
EXTRA=('caltech20_rich_view_target_consistency46.py','check_caltech20_rich_view_target_consistency46.py','audit_caltech20_rich_view_target_consistency46.py','CALTECH20_RICH_VIEW_TARGET_CONSISTENCY46_PROTOCOL_20261018.md')
FILES=(EXTRA[0],baseline.FILES[1],EXTRA[1],EXTRA[2],EXTRA[3])
def config():
    c=baseline.config();case=dict(fresh=False,epochs=5,electronic_warm_epochs=0,phase_lr=.00015,full_rms_gradient=True)
    c.update(run='caltech20_rich20_two_view_target_allocation46_local5_20261018',cases={f'L{d}_{mode}':dict(case,depth=d,consistency_target=mode) for d in (4,6) for mode in ('prototype_both','KD_both')},gpu_order=[3,4,0,1,2],test_interval=1,selection='maximum_TEST_integer_correct_earliest_tie',VALID_selection=False,unchanged_total_auxiliary_coefficients=True)
    return c
OUT=HERE/'runs/simulation'/config()['run'];CPU=HERE/'runs/smoke/caltech20_rich_view_target_consistency46_CPU_20261018';ACTIVE_VARIANT='prototype_both'
def sources():
    z=source_closure.sources()
    for n in EXTRA:z[(HERE/n).relative_to(ROOT).as_posix()]=s.sha(HERE/n)
    return z
for name in ('identity','claim','rates','seed_for','load_predictions','groups','js_loss','numeric_gate_kernel','campaign','feature_loss','target_data','initialize','gate_kernel','complete_gate_contract','fit_kernel','select_kernel','teacher_targets','gradient_record_kernel','fit','zoom_plan','plan_hash','transform','mixed_kernel','record_kernel','mixed','gradients'):
    f=getattr(baseline,name);globals()[name]=types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__)
for name in ('incumbent','prototype_data','teacher_data'):
    f=getattr(baseline,name).__wrapped__;globals()[name]=lru_cache(None)(types.FunctionType(f.__code__,globals(),name,f.__defaults__,f.__closure__))
# Rebuild the unchanged retained fitter, explicitly removing its old VALID tie-break.
import caltech20_rich_alpha40_rebalance as static_fit
text=inspect.getsource(static_fit.fit)
for before,after,count in [('opt.step();steps+=1','opt.step();model.project();steps+=1',1),("gradients(model,x,y,q,mass,mask,weights,cfg,4)","gradients(model,x,y,q,mass,mask,weights,cfg,cfg['microbatch_size'])",1),('crop_sha256=crop,loss=loss,clip_norm=norm','crop_sha256=crop,loss=loss,clip_norm=norm,view_info=model.view_info',1),("epoch=tune if case['fresh'] else parent['joint_budget_epochs']+tune",'epoch=tune',1),("-metrics['val']['balanced_nll']",'0',1),("-vm['balanced_nll']",'0',1)]:
    assert text.count(before)==count; text=text.replace(before,after)
exec(compile(text,__file__+':retained_fit_direct_TEST_earliest_tie','exec'),globals());fit_kernel=fit
f=baseline.fit;fit=types.FunctionType(f.__code__,globals(),'five_local_epochs_actual_warm_parent',f.__defaults__,f.__closure__)
old=types.SimpleNamespace(**vars(baseline.old));old.kd=types.SimpleNamespace(**vars(baseline.old.kd));old.kd.teacher_targets=teacher_targets
ns=dict(baseline.old.targets.__globals__);ns['kd']=old.kd;old.targets=types.FunctionType(baseline.old.targets.__code__,ns,'same_clean_teacher_targets',baseline.old.targets.__defaults__,baseline.old.targets.__closure__)
def select(key):
    global ACTIVE_DEPTH,ACTIVE_VARIANT
    c=config()['cases'][key];ACTIVE_DEPTH=c['depth'];ACTIVE_VARIANT=c['consistency_target'];select_kernel(key)
def allocated_terms(kc,ka,pc,pa,mode):
    assert mode in ('prototype_both','KD_both')
    return (kc,.5*(pc+pa)) if mode=='prototype_both' else (.5*(kc+ka),pc)
def gradient_kernel(model,x,y,q,mass,mask,weights,cfg,micro):
    assert CLEAN_X is not None and CLEAN_X.shape==x.shape and FEATURE_TARGET is not None
    model.train();model.zero_grad(set_to_none=True);ps=[];cs=[];total=0.;scale=target_data(model.depth)[1]
    parts=dict(clean=0.,augmented=0.,KD_clean=0.,JS=0.,phase_anchor=0.,feature_alignment=0.,KD_clean_actual=0.,KD_aug_actual=0.,prototype_clean_actual=0.,prototype_aug_actual=0.)
    for off in range(0,len(x),micro):
        size=min(micro,len(x)-off);factor=size/len(x);sl=slice(off,off+micro);model.offset=off;model.linear.mask=None;pc,cc=model(CLEAN_X[sl]);fc=model.semantic_feature
        model.linear.mask=mask[sl];pa,ca=model(x[sl]);fa=model.semantic_feature
        clean=base.loss(model,pc,cc,y[sl],weights,cfg);aug=base.loss(model,pa,ca,y[sl],weights,cfg)
        kc=old.kd.KD_loss(pc,q[sl],mass[sl],cfg);ka=old.kd.KD_loss(pa,q[sl],mass[sl],cfg);ac=feature_loss(fc,FEATURE_TARGET[sl],mass[sl]);aa=feature_loss(fa,FEATURE_TARGET[sl],mass[sl]);kd,alignment=allocated_terms(kc,ka,ac,aa,ACTIVE_VARIANT);js=js_loss(pc,pa)
        anchor=sum((p-model.phase_anchors[n]).square().sum() for n,p in model.named_parameters() if n in model.phase_anchors)/2
        loss=.5*clean+.5*aug+cfg['KD_weight']*kd+cfg['consistency_weight']*js+cfg['phase_anchor_weight']*anchor+scale*alignment
        (loss*factor).backward();total+=float(loss.detach())*factor;ps.append(pa.detach());cs.append(ca.detach())
        for name,value in zip(parts,(clean,aug,kd,js,anchor,alignment,kc,ka,ac,aa)):parts[name]+=float(value.detach())*factor
    model.linear.mask=None;model.view_info=dict(parts,weighted_sum=total,teacher_view='unaugmented_MixUp_same_ids_permutation_mass',full_rms_gradient=model.full_rms_gradient,clean_macro_sha256=s.digest(CLEAN_X.detach().cpu()),new_inference_parameters=0,alignment_scale=scale,alignment_target_sha256=target_data(model.depth)[2],consistency_target=ACTIVE_VARIANT,same_total_auxiliary_coefficients=True,legacy_KD_clean_field_is_allocated_KD_term=True)
    return torch.cat(ps),torch.cat(cs),total
f=baseline.gate;gate_original=types.FunctionType(f.__code__,globals(),'unchanged_microbatch_gradient_and_clean_export_proof',f.__defaults__,f.__closure__)
def gate(depth,fresh,data,case):
    proof=gate_original(depth,fresh,data,case);proof.update(teacher_applied_to_clean_view_only=ACTIVE_VARIANT!='KD_both',student_target_allocation=ACTIVE_VARIANT,teacher_reference_input_remains_clean=True,unchanged_total_auxiliary_coefficients=True);return proof
f=baseline.ready_kernel;ready_kernel=types.FunctionType(f.__code__,globals(),'same_train_only_zoom_and_retained_admission',f.__defaults__,f.__closure__)
def ready():
    proof=s.read(CPU/'checks.json');assert proof['analytic_independent_two_view_allocation'] and proof['actual_gradient_kernel_uses_allocation'] and proof['independent_numpy_zoomout_pixels'] and proof['clean_view_exactly_unchanged'] and proof['direct_TEST_tie_ignores_VALID'] and not CPU.with_name(CPU.name+'.failure.json').exists();ready_kernel()
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--phase',choices=['fit','campaign'],required=True);p.add_argument('--key');p.add_argument('--gpu');a=p.parse_args()
    if a.key:select(a.key)
    try:campaign() if a.phase=='campaign' else fit(a.key,a.gpu)
    except BaseException as e:
        f=OUT/a.key if a.key else OUT;mark=f/'failure.json' if f.exists() else f.with_name(f.name+'.failure.json')
        if not mark.exists():s.save(mark,dict(error=repr(e),automatic_retry=False),exclusive=True)
        raise
